"""
TrOCR Fine-Tuning Pipeline for Pediatric & Indian Cursive Handwriting.
======================================================================
Fine-tunes Microsoft TrOCR (Vision Transformer Encoder + RoBERTa Decoder)
on pediatric handwriting lines with Indian classroom vocabulary and names.

Optimized for NVIDIA RTX 4060 GPU (8GB VRAM):
  1. Selective ViT layer freezing (freezes bottom 8 layers, trains top 4 layers + decoder).
  2. PyTorch Automatic Mixed Precision (AMP FP16) with GradScaler.
  3. Gradient accumulation for effective batch sizes of 8-16 without VRAM overflow.
  4. On-the-fly streaming synthetic pediatric lines with pencil physics & NIST SD19 crops.
  5. Validation with Character Error Rate (CER) and Word Error Rate (WER).
  6. Checkpoint persistence in models/trocr/finetuned/.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    XLMRobertaTokenizer,
    get_cosine_schedule_with_warmup,
)

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.synthetic_line_generator import NISTLineSynthesizer, GRADE_SENTENCES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("finetune_trocr")


# ---------------------------------------------------------------------------
# Multi-Source Dataset Definitions (Real IAM + Synthetic Indian Pediatric)
# ---------------------------------------------------------------------------
class IAMLineStitcher:
    """Stitches real human handwriting word crops from IAM Words into multi-word text-lines."""

    def __init__(
        self,
        manifest_csv: str = "data/iam_words/train/manifest.csv",
        base_dir: str = "data/iam_words/train",
    ):
        import random
        self.rng = random.Random(42)
        self.base_dir = Path(base_dir)
        self.samples = []
        if Path(manifest_csv).exists():
            import pandas as pd
            df = pd.read_csv(manifest_csv).dropna(subset=["filepath", "text"])
            f_list = df["filepath"].astype(str).str.strip().tolist()
            t_list = df["text"].astype(str).str.strip().tolist()
            self.samples = [(f, t) for f, t in zip(f_list, t_list) if len(t) >= 1]
        logger.info(f"Loaded {len(self.samples):,} real human IAM word crops for multi-word line stitching.")

    def sample_line(
        self,
        min_words: int = 2,
        max_words: int = 5,
        target_h: int = 64,
    ) -> Tuple[np.ndarray, str]:
        if not self.samples:
            return np.full((target_h, 800), 255, dtype=np.uint8), "My name is Surya Teja"

        n = self.rng.randint(min_words, max_words)
        picked = self.rng.sample(self.samples, n)
        word_imgs = []
        valid_words = []
        for f, t in picked:
            im = cv2.imread(str(self.base_dir / f), cv2.IMREAD_GRAYSCALE)
            if im is not None and im.size > 0:
                h, w = im.shape
                new_w = max(4, int(w * (target_h / max(h, 1))))
                resized = cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA)
                word_imgs.append(resized)
                valid_words.append(t)

        if not word_imgs:
            return np.full((target_h, 800), 255, dtype=np.uint8), "My name is Surya Teja"

        total_w = sum(im.shape[1] for im in word_imgs) + (len(word_imgs) + 1) * 24
        canvas = np.full((target_h, max(total_w, 400)), 255, dtype=np.uint8)
        cur_x = self.rng.randint(10, 25)
        for w_im in word_imgs:
            ih, iw = w_im.shape
            clip_w = min(iw, canvas.shape[1] - cur_x)
            yo = max(0, (target_h - ih) // 2)
            ih_fit = min(ih, target_h - yo)
            if clip_w > 0 and ih_fit > 0:
                canvas[yo:yo+ih_fit, cur_x:cur_x+clip_w] = np.minimum(
                    canvas[yo:yo+ih_fit, cur_x:cur_x+clip_w],
                    w_im[:ih_fit, :clip_w],
                )
            cur_x += iw + self.rng.randint(16, 30)

        text = " ".join(valid_words)
        return canvas[:, :cur_x + 10], text


class HybridHandwritingLineDataset(Dataset):
    """
    Unified multi-dataset fusion combining:
      1. Real human handwriting word sequences from IAM (69,190 words).
      2. Synthetic pediatric lines with Indian student names and pencil physics (NIST SD19).
    """

    def __init__(
        self,
        processor: TrOCRProcessor,
        synthesizer: NISTLineSynthesizer,
        iam_stitcher: Optional[IAMLineStitcher] = None,
        num_samples: int = 1000,
        iam_ratio: float = 0.40,
        max_target_length: int = 64,
        seed: int = 42,
    ):
        import random
        self.processor = processor
        self.synthesizer = synthesizer
        self.iam_stitcher = iam_stitcher
        self.num_samples = num_samples
        self.iam_ratio = iam_ratio if (iam_stitcher and iam_stitcher.samples) else 0.0
        self.max_target_length = max_target_length
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.iam_stitcher and self.rng.random() < self.iam_ratio:
            img_np, text = self.iam_stitcher.sample_line()
        else:
            img_np, text = self.synthesizer.synthesize_line()

        if not text.strip():
            text = "My name is Surya Teja"
            img_np = np.full((64, 800), 255, dtype=np.uint8)

        # Convert to RGB PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB))

        # Process image
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize target text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so loss ignores padding
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels.tolist()
        ]
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels_tensor,
            "text": text,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    texts = [b["text"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "texts": texts,
    }


# ---------------------------------------------------------------------------
# Layer Freezing Utility
# ---------------------------------------------------------------------------
def freeze_encoder_layers(model: VisionEncoderDecoderModel, num_freeze_layers: int = 8):
    """
    Freezes the bottom N layers of the Vision Transformer encoder to preserve
    generic edge and curve representations, while leaving top layers and RoBERTa
    decoder fully trainable for pediatric domain adaptation.
    """
    encoder = model.encoder
    # Freeze patch embeddings
    if hasattr(encoder, "embeddings"):
        for p in encoder.embeddings.parameters():
            p.requires_grad = False
        logger.info("Froze ViT patch embeddings.")

    # Freeze encoder layers
    encoder_layers = []
    if hasattr(encoder, "layers"):
        encoder_layers = encoder.layers
    elif hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        encoder_layers = encoder.encoder.layer
    elif hasattr(encoder, "layer"):
        encoder_layers = encoder.layer

    frozen_count = 0
    for idx, layer in enumerate(encoder_layers):
        if idx < num_freeze_layers:
            for p in layer.parameters():
                p.requires_grad = False
            frozen_count += 1

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Freezing complete: Froze {frozen_count}/{len(encoder_layers)} ViT encoder layers. "
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params:.1%})"
    )


# ---------------------------------------------------------------------------
# Metric Evaluation
# ---------------------------------------------------------------------------
def compute_cer_and_wer(preds: List[str], targets: List[str]) -> Tuple[float, float]:
    """Computes Character Error Rate and Word Error Rate."""
    total_chars = 0
    char_errors = 0
    total_words = 0
    word_errors = 0

    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev[j + 1] + 1
                deletions = curr[j] + 1
                substitutions = prev[j] + (c1 != c2)
                curr.append(min(insertions, deletions, substitutions))
            prev = curr
        return prev[-1]

    for pred, target in zip(preds, targets):
        p_str = pred.strip().lower()
        t_str = target.strip().lower()

        char_errors += levenshtein(p_str, t_str)
        total_chars += max(len(t_str), 1)

        p_words = p_str.split()
        t_words = t_str.split()
        word_errors += levenshtein(p_words, t_words)
        total_words += max(len(t_words), 1)

    cer = char_errors / max(total_chars, 1)
    wer = word_errors / max(total_words, 1)
    return cer, wer


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def train_trocr(
    model_name_or_path: str = "models/trocr/base",
    output_dir: str = "models/trocr/finetuned",
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    lr: float = 5e-5,
    freeze_layers: int = 8,
    train_samples: int = 600,
    val_samples: int = 60,
    device: Optional[str] = None,
    dry_run: bool = False,
):
    """
    Executes mixed-precision fine-tuning of TrOCR for pediatric handwriting.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("STARTING TrOCR PEDIATRIC & INDIAN HANDWRITING FINE-TUNING")
    logger.info("=" * 70)
    logger.info(f"Target Device: {device}")
    if device == "cuda":
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 1. Resolve model loading target
    load_target = model_name_or_path
    if not Path(load_target).exists():
        fallback = "models/trocr/base"
        if Path(fallback).exists():
            load_target = fallback
        else:
            load_target = "microsoft/trocr-base-handwritten"

    logger.info(f"Loading base processor and weights from: {load_target}...")
    try:
        img_proc = AutoImageProcessor.from_pretrained(load_target)
        if "small" in str(load_target).lower():
            tokenizer = XLMRobertaTokenizer.from_pretrained(load_target)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(load_target)
    except Exception:
        img_proc = AutoImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")

    processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained(load_target)
    model.to(device)

    # Set decoder special tokens
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # 2. Selective layer freezing
    freeze_encoder_layers(model, num_freeze_layers=freeze_layers)

    # 3. Setup Dataset & Synthesizer (Multi-Source Hybrid Fusion)
    synthesizer = NISTLineSynthesizer()
    iam_stitcher = IAMLineStitcher()
    train_dataset = HybridHandwritingLineDataset(
        processor=processor,
        synthesizer=synthesizer,
        iam_stitcher=iam_stitcher,
        num_samples=train_samples,
        iam_ratio=0.45,
    )
    val_dataset = HybridHandwritingLineDataset(
        processor=processor,
        synthesizer=synthesizer,
        iam_stitcher=iam_stitcher,
        num_samples=val_samples,
        iam_ratio=0.45,
        seed=1337,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 4. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, int(total_steps * 0.1)),
        num_training_steps=max(total_steps, 20),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    logger.info(
        f"Training configuration: Epochs={epochs}, BatchSize={batch_size}, "
        f"GradAccum={grad_accum_steps}, EffectiveBatch={batch_size * grad_accum_steps}, "
        f"TotalSteps={total_steps}"
    )

    best_val_cer = 1.0

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        logger.info(f"\n--- Epoch {epoch}/{epochs} ---")
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum_steps
            step_count += 1

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % (grad_accum_steps * 10) == 0:
                avg_step_loss = running_loss / step_count
                lr_curr = scheduler.get_last_lr()[0]
                logger.info(f"  Step [{step+1}/{len(train_loader)}] Loss: {avg_step_loss:.4f} | LR: {lr_curr:.2e}")

            if dry_run and step >= 5:
                logger.info("Dry-run completed successfully! Verification passed.")
                return

        # Validation phase
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        logger.info("Evaluating on validation split...")
        with torch.no_grad():
            for v_batch in val_loader:
                v_pixels = v_batch["pixel_values"].to(device)
                v_labels = v_batch["labels"].to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    v_out = model(pixel_values=v_pixels, labels=v_labels)
                    val_loss += v_out.loss.item()

                # Generate transcriptions
                gen_ids = model.generate(v_pixels, max_new_tokens=32, num_beams=1)
                preds = processor.batch_decode(gen_ids, skip_special_tokens=True)
                val_preds.extend(preds)
                val_targets.extend(v_batch["texts"])

        avg_val_loss = val_loss / max(len(val_loader), 1)
        cer, wer = compute_cer_and_wer(val_preds, val_targets)
        logger.info(
            f"Validation Results: Loss={avg_val_loss:.4f} | CER={cer:.2%} | WER={wer:.2%}"
        )

        # Show sample predictions
        for s_idx in range(min(3, len(val_preds))):
            logger.info(f"  Target: '{val_targets[s_idx]}'")
            logger.info(f"  Pred  : '{val_preds[s_idx]}'")

        # Persist checkpoint if improved
        if cer < best_val_cer or epoch == epochs:
            best_val_cer = cer
            logger.info(f"--> Best model achieved! Saving checkpoint to {out_path}...")
            model.save_pretrained(out_path)
            processor.save_pretrained(out_path)
            logger.info(f"--> Saved fine-tuned TrOCR weights to {out_path} successfully!")

    logger.info("=" * 70)
    logger.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.info(f"Artifacts saved in: {out_path}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on pediatric handwriting.")
    parser.add_argument("--model", type=str, default="models/trocr/base", help="Model name or cache directory")
    parser.add_argument("--output_dir", type=str, default="models/trocr/finetuned", help="Save directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size per GPU step")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--freeze_layers", type=int, default=8, help="Number of bottom ViT layers to freeze")
    parser.add_argument("--train_samples", type=int, default=500, help="Synthetic lines per epoch")
    parser.add_argument("--val_samples", type=int, default=40, help="Validation lines")
    parser.add_argument("--dry_run", action="store_true", help="Quick test run for verification")

    args = parser.parse_args()

    train_trocr(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        freeze_layers=args.freeze_layers,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

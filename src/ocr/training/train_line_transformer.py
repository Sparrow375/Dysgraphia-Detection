"""
In-House Line-Level CNN-Transformer Training Script.
===================================================
Trains HandwritingTransformerOCR directly on full handwritten line strips (64 x 800)
using NIST-synthesized pediatric lines and IAM datasets.

Warm-starts from existing in-house word checkpoint (models/transformer_ocr/checkpoint_best.pth)
to transfer learned stroke feature representations into line-level sequence context.

GPU-accelerated with PyTorch AMP, CTC Loss, and OneCycleLR / CosineAnnealingLR.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.ocr.handwriting_transformer import HandwritingTransformerOCR, DEFAULT_ALPHABET
from src.data.synthetic_line_generator import NISTLineSynthesizer, PediatricSyntheticLineDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def collate_line_fn(batch):
    imgs, targets, target_lengths, raw_texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    flat_targets = torch.cat(targets, dim=0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return imgs, flat_targets, target_lengths, list(raw_texts)


def levenshtein_dist(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def calculate_metrics(preds: List[str], targets: List[str]) -> Tuple[float, float]:
    total_chars = 0
    char_errors = 0
    exact_words = 0
    total_words = 0

    for pred, target in zip(preds, targets):
        p_str = pred.strip()
        t_str = target.strip()

        ed = levenshtein_dist(p_str, t_str)
        char_errors += ed
        total_chars += max(len(t_str), 1)

        p_words = p_str.split()
        t_words = t_str.split()
        for pw, tw in zip(p_words, t_words):
            if pw == tw:
                exact_words += 1
        total_words += max(len(t_words), 1)

    cer = char_errors / max(total_chars, 1)
    word_acc = exact_words / max(total_words, 1)
    return cer, word_acc


def train_line_model(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 2e-4,
    samples_per_epoch: int = 4000,
    val_samples: int = 400,
    out_dir: str = "models/line_transformer",
    warmstart_ckpt: str = "models/transformer_ocr/checkpoint_best.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using training device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Line-Level Synthesizer
    logger.info("Initializing NIST Pediatric Line Synthesizer...")
    synthesizer = NISTLineSynthesizer(max_crops_per_class=120)

    train_ds = PediatricSyntheticLineDataset(synthesizer, n_samples=samples_per_epoch, target_h=64, target_w=800)
    val_ds = PediatricSyntheticLineDataset(synthesizer, n_samples=val_samples, target_h=64, target_w=800)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_line_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_line_fn
    )

    # 2. Build In-House Line Transformer Model (max_seq_len=256 for W=800)
    model = HandwritingTransformerOCR(
        alphabet=DEFAULT_ALPHABET,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=256,
    ).to(device)

    # 3. Warm-Start from Word Checkpoint
    start_epoch = 1
    best_cer = float("inf")
    best_word_acc = 0.0

    if warmstart_ckpt and Path(warmstart_ckpt).exists():
        try:
            logger.info(f"Warm-starting from in-house word checkpoint: {warmstart_ckpt}")
            ckpt = torch.load(warmstart_ckpt, map_location=device)
            state_dict = ckpt["model_state_dict"]
            # Exclude pos_embed if size differs
            if "pos_embed" in state_dict and state_dict["pos_embed"].shape != model.pos_embed.shape:
                logger.info(f"Adapting pos_embed from {state_dict['pos_embed'].shape} to {model.pos_embed.shape} via linear interpolation...")
                old_pe = state_dict["pos_embed"].permute(0, 2, 1)  # (1, dim, old_T)
                new_pe = nn.functional.interpolate(old_pe, size=model.max_seq_len, mode="linear", align_corners=False)
                state_dict["pos_embed"] = new_pe.permute(0, 2, 1)

            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully transferred CNN feature stem and Transformer encoder weights!")
        except Exception as e:
            logger.warning(f"Could not warm-start from {warmstart_ckpt}: {e}")

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    history = []
    logger.info(f"Starting In-House Line-Level CNN-Transformer Training ({epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Line Train]")
        for imgs, targets, target_lengths, _ in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                out = model(imgs)
                log_probs = out["log_probs"]  # (T, B, num_classes)
                T, B, _ = log_probs.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{train_loss / n_batches:.4f}")

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, targets, target_lengths, raw_texts in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    out = model(imgs)
                    log_probs = out["log_probs"]
                    T, B, _ = log_probs.shape
                    input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

                val_loss += loss.item()
                n_val_batches += 1

                preds = model.decode_greedy(out["logits"])
                all_preds.extend(preds)
                all_targets.extend(raw_texts)

        val_loss /= max(n_val_batches, 1)
        cer, word_acc = calculate_metrics(all_preds, all_targets)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val CER: {cer:.2%} | Val Word Acc: {word_acc:.2%}"
        )

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_cer": round(cer, 4),
            "val_word_acc": round(word_acc, 4),
        }
        history.append(epoch_stats)

        # Save checkpoint
        if cer < best_cer:
            best_cer = cer
            best_word_acc = word_acc
            ckpt_path = out_path / "checkpoint_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_cer": cer,
                    "val_word_acc": word_acc,
                    "alphabet": DEFAULT_ALPHABET,
                    "max_seq_len": 256,
                },
                ckpt_path,
            )
            logger.info(f"  --> Saved new best line-level checkpoint to {ckpt_path} (CER: {cer:.2%}, Word Acc: {word_acc:.2%})")

    # Save metrics JSON
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_cer": best_cer,
                "best_val_word_acc": best_word_acc,
                "history": history,
            },
            f,
            indent=2,
        )

    logger.info("In-House Line-Level CNN-Transformer training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--val_samples", type=int, default=300)
    parser.add_argument("--out_dir", type=str, default="models/line_transformer")
    parser.add_argument("--warmstart", type=str, default="models/transformer_ocr/checkpoint_best.pth")
    args = parser.parse_args()

    train_line_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_epoch=args.samples,
        val_samples=args.val_samples,
        out_dir=args.out_dir,
        warmstart_ckpt=args.warmstart,
    )

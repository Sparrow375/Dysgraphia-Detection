"""
Training Script for HandwritingTransformerOCR on IAM Words.
GPU-accelerated training using PyTorch CTC Loss, Mixed Precision (AMP), and OneCycleLR.

Usage:
    python src/ocr/training/train_transformer_ocr.py \
        --data_dir data/iam_words \
        --out_dir models/transformer_ocr \
        --epochs 15 \
        --batch_size 64 \
        --lr 3e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, os.path.abspath("."))

from src.ocr.handwriting_transformer import HandwritingTransformerOCR, DEFAULT_ALPHABET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IAM Words Dataset
# ---------------------------------------------------------------------------

class IAMWordsDataset(Dataset):
    """Loads handwritten word crops and labels from IAM manifest.csv."""

    def __init__(
        self,
        manifest_path: str | Path,
        alphabet: str = DEFAULT_ALPHABET,
        target_height: int = 64,
        max_width: int = 256,
        augment: bool = False,
    ):
        self.alphabet = alphabet
        self.target_height = target_height
        self.max_width = max_width
        self.augment = augment
        self.char2idx: Dict[str, int] = {c: i + 1 for i, c in enumerate(alphabet)}
        self.samples: List[Tuple[Path, str]] = []

        manifest_p = Path(manifest_path)
        if not manifest_p.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_p}")

        root_dir = manifest_p.parent
        with open(manifest_p, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filepath", "").strip()
                text = row.get("text", "").strip()
                if not fname or not text:
                    continue
                filtered = "".join(c for c in text if c in self.char2idx)
                if not filtered:
                    continue
                img_path = root_dir / fname
                if img_path.exists():
                    self.samples.append((img_path, filtered))

        logger.info(f"Loaded {len(self.samples):,} valid samples from {manifest_p}")

    def __len__(self) -> int:
        return len(self.samples)

    def encode(self, text: str) -> List[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.target_height, self.max_width), dtype=np.uint8)

        # Invert if dark ink on white background so ink is bright
        if np.mean(img) > 127:
            img = 255 - img

        # Resize height to 64 while maintaining aspect ratio
        h, w = img.shape
        scale = self.target_height / max(h, 1)
        new_w = min(int(w * scale), self.max_width)
        resized = cv2.resize(img, (max(new_w, 1), self.target_height), interpolation=cv2.INTER_LINEAR)

        # Right-pad canvas to max_width
        canvas = np.zeros((self.target_height, self.max_width), dtype=np.uint8)
        canvas[:, :new_w] = resized

        # Data augmentation
        if self.augment:
            if np.random.rand() < 0.25:
                angle = np.random.uniform(-4, 4)
                M = cv2.getRotationMatrix2D((self.max_width // 2, self.target_height // 2), angle, 1.0)
                canvas = cv2.warpAffine(canvas, M, (self.max_width, self.target_height), borderValue=0)

        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)  # (1, 64, max_width)
        return tensor, label


def collate_fn(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Batches images and converts string labels into flat CTC targets."""
    images, texts = zip(*batch)
    images_tensor = torch.stack(images, dim=0)

    char2idx = {c: i + 1 for i, c in enumerate(DEFAULT_ALPHABET)}
    target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    flat_targets = []
    for t in texts:
        flat_targets.extend([char2idx[c] for c in t if c in char2idx])
    targets_tensor = torch.tensor(flat_targets, dtype=torch.long)

    return images_tensor, targets_tensor, target_lengths, list(texts)


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def calculate_cer_and_acc(preds: List[str], targets: List[str]) -> Tuple[float, float]:
    """Computes Character Error Rate (Levenshtein) and exact Word Accuracy without external packages."""
    total_dist = 0
    total_chars = 0
    correct_words = 0

    for p, t in zip(preds, targets):
        # Pure DP Levenshtein distance
        n, m = len(p), len(t)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            new_dp = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if p[i - 1] == t[j - 1] else 1
                new_dp[j] = min(new_dp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost)
            dp = new_dp
        dist = dp[m]

        total_dist += dist
        total_chars += max(len(t), 1)
        if p == t:
            correct_words += 1

    cer = total_dist / max(total_chars, 1)
    word_acc = correct_words / max(len(targets), 1)
    return cer, word_acc


def train(
    data_dir: str = "data/iam_words",
    out_dir: str = "models/transformer_ocr",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    resume: bool = False,
    resume_path: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training HandwritingTransformerOCR on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_manifest = Path(data_dir) / "train" / "manifest.csv"
    val_manifest = Path(data_dir) / "val" / "manifest.csv"

    train_ds = IAMWordsDataset(train_manifest, augment=True)
    val_ds = IAMWordsDataset(val_manifest, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn
    )

    # Build model (5.44M parameters)
    model = HandwritingTransformerOCR(
        alphabet=DEFAULT_ALPHABET,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ).to(device)

    best_cer = float("inf")
    best_word_acc = 0.0
    history = []
    start_epoch = 1

    if resume or resume_path:
        ckpt_file = Path(resume_path) if resume_path else (out_path / "checkpoint_best.pth")
        if ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            best_cer = ckpt.get("val_cer", float("inf"))
            best_word_acc = ckpt.get("val_word_acc", 0.0)
            start_epoch = ckpt.get("epoch", 0) + 1
            logger.info(f"Resumed from {ckpt_file} at epoch {start_epoch - 1} (best CER: {best_cer:.2%}, best Word Acc: {best_word_acc:.2%})")
            metrics_file = out_path / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                        history = m_data.get("history", [])
                except Exception:
                    pass

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    end_epoch = start_epoch + epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{end_epoch:02d} [Train]")
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
        cer, word_acc = calculate_cer_and_acc(all_preds, all_targets)

        logger.info(
            f"Epoch {epoch:02d}/{end_epoch:02d} | Train Loss: {train_loss:.4f} | "
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

        # Save best model by CER
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
                },
                ckpt_path,
            )
            logger.info(f"  --> Saved new best checkpoint to {ckpt_path} (CER: {cer:.2%}, Word Acc: {word_acc:.2%})")

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

    logger.info("HandwritingTransformerOCR training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/iam_words")
    parser.add_argument("--out_dir", type=str, default="models/transformer_ocr")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint_best.pth")
    parser.add_argument("--resume_path", type=str, default=None, help="Explicit path to checkpoint to resume from")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        resume_path=args.resume_path,
    )

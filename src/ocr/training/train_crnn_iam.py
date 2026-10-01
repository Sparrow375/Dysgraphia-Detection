"""
CRNN Training Script — IAM Handwriting Dataset (GPU-Accelerated).

Trains the Convolutional Recurrent Neural Network on word-level handwriting
images from the IAM dataset using CTC loss, with CUDA, mixed precision (AMP),
OneCycleLR scheduling, and automatic checkpointing.

Usage:
    python src/ocr/training/train_crnn_iam.py \\
        --data_dir data/iam_words \\
        --out_dir models/crnn_iam \\
        --epochs 30 \\
        --batch_size 64 \\
        --lr 3e-4

Resume training from checkpoint:
    python src/ocr/training/train_crnn_iam.py \\
        --data_dir data/iam_words \\
        --out_dir models/crnn_iam \\
        --resume models/crnn_iam/checkpoint_best.pth
"""

from __future__ import annotations

import argparse
import csv
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------

DEFAULT_ALPHABET = (
    " abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?'\"-()/"
)


# ---------------------------------------------------------------------------
# Improved CRNN Architecture (deeper + wider)
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1,
                 pool: Optional[Tuple[int, int]] = None):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Simple residual block for the CNN backbone."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CRNN(nn.Module):
    """
    Deep CRNN: CNN backbone (with residual blocks) + BiLSTM + CTC head.

    Input:  (B, 1, 64, W)
    Output: (T, B, num_classes) log-softmax
    """

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 512,
        lstm_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        # CNN backbone — produces (B, 512, 1, T)
        self.cnn = nn.Sequential(
            ConvBNReLU(1, 64, pool=(2, 2)),            # → (B, 64, 32, W/2)
            ResBlock(64),
            ConvBNReLU(64, 128, pool=(2, 2)),          # → (B, 128, 16, W/4)
            ResBlock(128),
            ConvBNReLU(128, 256),                      # → (B, 256, 16, W/4)
            ResBlock(256),
            ConvBNReLU(256, 256, pool=(2, 1)),         # → (B, 256, 8, W/4)
            ConvBNReLU(256, 512),                      # → (B, 512, 8, W/4)
            ResBlock(512),
            ConvBNReLU(512, 512, pool=(2, 1)),         # → (B, 512, 4, W/4)
            ConvBNReLU(512, 512, k=2, p=0),            # → (B, 512, 3, W/4-1)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # → (B, 512, 1, T)

        # BiLSTM
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        f = self.cnn(x)                   # (B, 512, h', T)
        f = self.adaptive_pool(f)         # (B, 512, 1, T)
        f = f.squeeze(2)                  # (B, 512, T)
        f = f.permute(2, 0, 1)           # (T, B, 512)

        # BiLSTM
        out, _ = self.rnn(f)             # (T, B, 2*hidden)
        out = self.dropout(out)
        out = self.fc(out)               # (T, B, num_classes)
        return F.log_softmax(out, dim=2)


# ---------------------------------------------------------------------------
# Dataset loader (reads manifest CSV)
# ---------------------------------------------------------------------------

def normalize_word_image(img: np.ndarray, target_height: int = 64, max_width: int = 640) -> np.ndarray:
    """Resize image to fixed height while preserving aspect ratio, capping max width."""
    if img is None or img.size == 0:
        return np.zeros((target_height, 128), dtype=np.uint8)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_height, 128), dtype=np.uint8)
    scale = target_height / h
    new_w = max(1, min(int(w * scale), max_width))
    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    return resized


class IAMWordDataset(Dataset):
    """Word-level IAM dataset loaded from manifest CSV."""

    def __init__(
        self,
        manifest_path: str,
        alphabet: str = DEFAULT_ALPHABET,
        target_height: int = 64,
        augment: bool = False,
        max_label_len: int = 32,
    ):
        self.root = Path(manifest_path).parent
        self.alphabet = alphabet
        self.target_height = target_height
        self.augment = augment
        self.max_label_len = max_label_len
        self.char2idx: Dict[str, int] = {c: i + 1 for i, c in enumerate(alphabet)}

        self.samples: List[Tuple[str, str]] = []
        with open(manifest_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"].strip()
                if not text or len(text) > max_label_len:
                    continue
                # Filter to only chars in alphabet
                filtered = "".join(c for c in text if c in self.char2idx)
                if not filtered:
                    continue
                self.samples.append((row["filepath"], filtered))

        logger.info(f"  Loaded {len(self.samples):,} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def encode(self, text: str) -> List[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def __getitem__(self, idx: int):
        fname, label = self.samples[idx]
        img_path = self.root / fname

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.target_height, 128), dtype=np.uint8)

        # Ensure dark ink on light background
        if np.mean(img) > 127:
            img = 255 - img

        # Basic augmentation during training
        if self.augment:
            img = self._augment(img)

        img = normalize_word_image(img, self.target_height)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        target = torch.tensor(self.encode(label), dtype=torch.long)
        return tensor, target, label

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Lightweight augmentations: random noise, dilation, erosion."""
        # Random Gaussian noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 8, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Random morphological transform
        if np.random.rand() < 0.2:
            k = np.random.randint(1, 3)
            kernel = np.ones((k, k), np.uint8)
            if np.random.rand() < 0.5:
                img = cv2.dilate(img, kernel)
            else:
                img = cv2.erode(img, kernel)
        # Random brightness
        if np.random.rand() < 0.3:
            delta = np.random.randint(-30, 30)
            img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        return img


def collate_fn(batch):
    """Pad images and targets for CTC."""
    images, targets, texts = zip(*batch)

    max_w = max(im.shape[-1] for im in images)
    max_w = ((max_w + 15) // 16) * 16  # round up to multiple of 16

    B = len(images)
    H = images[0].shape[-2]
    padded = torch.zeros(B, 1, H, max_w)
    widths = torch.zeros(B, dtype=torch.long)

    for i, im in enumerate(images):
        w = im.shape[-1]
        padded[i, :, :, :w] = im
        widths[i] = w  # store original widths, not estimates

    flat_targets = torch.cat(targets)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    return padded, flat_targets, widths, target_lengths, texts


def compute_input_lengths(widths: torch.Tensor, model: "CRNN") -> torch.Tensor:
    """
    Compute actual CNN output time steps for each image width.
    The CNN applies two (2,1) max-pools and one extra conv that reduces W slightly.
    We trace through the width reductions:
      - MaxPool(2,2): W → W//2
      - MaxPool(2,2): W//2 → W//4
      - MaxPool(2,1): W//4 stays (stride-1 in W)
      - MaxPool(2,1): W//4 stays
      - ConvBNReLU(k=2,p=0): W//4 → W//4 - 1
      - AdaptiveAvgPool → stays
    So effective T = max(1, w//4 - 1)
    """
    return torch.clamp(widths // 4 - 1, min=1)


# ---------------------------------------------------------------------------
# CTC greedy decode (numpy, no torch needed)
# ---------------------------------------------------------------------------

def greedy_decode(log_probs_np: np.ndarray, alphabet: str) -> str:
    """Greedy CTC decode: (T, C) → string."""
    best = np.argmax(log_probs_np, axis=1)
    chars = []
    prev = 0
    for idx in best:
        if idx != 0 and idx != prev:
            if 1 <= idx <= len(alphabet):
                chars.append(alphabet[idx - 1])
        prev = idx
    return "".join(chars)


def compute_cer(pred: str, gt: str) -> float:
    """Character Error Rate."""
    if not gt:
        return 0.0 if not pred else 1.0
    # Simple edit distance via DP
    n, m = len(pred), len(gt)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            new_dp[j] = min(new_dp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost)
        dp = new_dp
    return dp[m] / m


# ---------------------------------------------------------------------------
# Training + Evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    device: str,
    grad_clip: float = 5.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, flat_targets, _widths, target_lengths, _ in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        flat_targets = flat_targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast('cuda'):
            log_probs = model(images)  # (T, B, C)
            T = log_probs.shape[0]     # actual CNN output length
            # Use actual T for ALL images in this batch (safe upper bound)
            input_lengths = torch.full((images.shape[0],), T, dtype=torch.long, device=device)
            loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        # OneCycleLR must be stepped after optimizer.step() (which happens inside scaler.step)
        if scheduler is not None and scaler.get_scale() > 0:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: CRNN,
    loader: DataLoader,
    device: str,
    alphabet: str,
    max_batches: int = 200,
) -> Dict[str, float]:
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    total_words = 0
    n = 0

    for batch_idx, (images, _, _, _, texts) in enumerate(tqdm(loader, desc="  Val", leave=False)):
        if batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        with autocast('cuda'):
            log_probs = model(images)  # (T, B, C)
        log_probs_np = log_probs.float().cpu().numpy()

        for b, gt in enumerate(texts):
            pred = greedy_decode(log_probs_np[:, b, :], alphabet)
            total_cer += compute_cer(pred, gt)
            wer = 0 if pred.strip() == gt.strip() else 1
            total_wer += wer
            n += 1

    total_words = n
    cer = total_cer / max(n, 1)
    wer = total_wer / max(n, 1)
    return {"CER": cer, "WER": wer, "n": n}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CRNN on IAM Handwriting Dataset.")
    parser.add_argument("--data_dir", default="data/iam_words",
                        help="Root with train/val/test subdirs")
    parser.add_argument("--out_dir", default="models/crnn_iam",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lstm_hidden", type=int, default=512)
    parser.add_argument("--lstm_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--target_height", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pth to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda' or 'cpu' (auto-detect if not set)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # ---- Datasets ----
    train_manifest = data_dir / "train" / "manifest.csv"
    val_manifest = data_dir / "val" / "manifest.csv"

    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {train_manifest}\n"
            "Run: python src/ocr/training/download_iam_dataset.py --out_dir data/iam_words"
        )

    train_ds = IAMWordDataset(str(train_manifest), augment=True, target_height=args.target_height)
    val_ds   = IAMWordDataset(str(val_manifest),   augment=False, target_height=args.target_height)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn,
        pin_memory=(device == "cuda"), persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
        pin_memory=(device == "cuda"), persistent_workers=(args.workers > 0),
    )

    # ---- Model ----
    num_classes = len(DEFAULT_ALPHABET) + 1  # +1 for CTC blank
    model = CRNN(
        num_classes=num_classes,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {n_params:.2f}M")

    # ---- Optimizer + Scheduler + Loss ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    scaler = GradScaler('cuda')

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_cer = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_cer = ckpt.get("best_cer", float("inf"))
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        elif start_epoch > 0:
            target_steps = start_epoch * len(train_loader)
            logger.info(f"Advancing scheduler across {target_steps} previous steps...")
            for _ in range(target_steps):
                scheduler.step()
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        logger.info(f"Resumed from epoch {start_epoch + 1} (best CER so far: {best_cer:.4f})")

    # ---- Training loop ----
    logger.info(f"\nStarting training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    logger.info(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train (scheduler is stepped per batch inside train_epoch)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device)

        # Evaluate every epoch
        metrics = evaluate(model, val_loader, device, DEFAULT_ALPHABET)
        cer = metrics["CER"]
        wer = metrics["WER"]

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Loss={train_loss:.4f} | CER={cer:.4f} | WER={wer:.4f} | "
            f"LR={lr_now:.2e} | {elapsed:.1f}s"
        )

        # Save checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_cer": best_cer,
            "CER": cer,
            "WER": wer,
            "alphabet": DEFAULT_ALPHABET,
            "lstm_hidden": args.lstm_hidden,
            "lstm_layers": args.lstm_layers,
        }
        torch.save(ckpt, out_dir / f"checkpoint_epoch{epoch+1:03d}.pth")

        if cer < best_cer:
            best_cer = cer
            torch.save(ckpt, out_dir / "checkpoint_best.pth")
            logger.info(f"  ★ New best CER: {best_cer:.4f} → saved checkpoint_best.pth")

        # Always save latest
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")

    logger.info(f"\nTraining complete. Best CER: {best_cer:.4f}")
    logger.info(f"Best model: {out_dir / 'checkpoint_best.pth'}")


if __name__ == "__main__":
    main()

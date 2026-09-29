"""
CRNN Model Training Script with CTC Loss.

Trains the Convolutional Recurrent Neural Network on word-level handwriting images
with optional dynamic dysgraphic augmentation.
"""

from __future__ import annotations

import argparse
import os
import time
import logging
from typing import List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.ocr.char_hypothesis import CRNNModel, DEFAULT_ALPHABET, ctc_greedy_decode
from src.ocr.training.dataset import HandwritingWordDataset, ctc_collate_fn, parse_iam_words_txt
from src.ocr.augmentation import DysgraphiaAugmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets, input_lengths, target_lengths, _) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # CRNN output: (B, T, num_classes) in log_softmax space
        log_probs = model(images)

        # PyTorch CTCLoss expects input shape: (T, B, C)
        log_probs = log_probs.permute(1, 0, 2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss encountered; skipping batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    alphabet: str,
) -> Tuple[float, float]:
    """
    Compute Character Error Rate (CER) and Word Error Rate (WER) on evaluation set.
    """
    model.eval()
    char_errors = 0
    total_chars = 0
    word_errors = 0
    total_words = 0

    try:
        import editdistance
        has_editdist = True
    except ImportError:
        has_editdist = False

    with torch.no_grad():
        for images, _, _, _, texts in dataloader:
            images = images.to(device)
            log_probs = model(images)  # (B, T, C)

            for b in range(images.size(0)):
                logits_np = log_probs[b].cpu().numpy()
                pred_text, _ = ctc_greedy_decode(logits_np, alphabet)
                gt_text = texts[b].strip()

                total_words += 1
                if pred_text.lower() != gt_text.lower():
                    word_errors += 1

                total_chars += len(gt_text)
                if has_editdist:
                    char_errors += editdistance.eval(pred_text.lower(), gt_text.lower())
                else:
                    char_errors += abs(len(pred_text) - len(gt_text))

    cer = char_errors / max(total_chars, 1)
    wer = word_errors / max(total_words, 1)
    return cer, wer


def main():
    parser = argparse.ArgumentParser(description="Train CRNN for handwriting OCR")
    parser.add_argument("--data-dir", type=str, default="data/iam/words", help="Path to word images directory")
    parser.add_argument("--words-txt", type=str, default="data/iam/words.txt", help="Path to IAM words.txt index")
    parser.add_argument("--output-dir", type=str, default="models/crnn", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--augment", action="store_true", help="Apply on-the-fly dysgraphic augmentation")
    parser.add_argument("--aug-severity", type=float, default=0.4, help="Severity of dysgraphic augmentation")
    args = parser.parse_args()

    if not HAS_TORCH:
        logger.error("PyTorch is required to run CRNN training.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    # Load dataset
    samples = parse_iam_words_txt(args.words_txt, args.data_dir)
    if not samples:
        logger.warning(f"No samples found at {args.words_txt}. Generating mock samples for pipeline verification.")
        samples = [("mock_img.png", "quick"), ("mock_img.png", "brown")]

    # Split train / val (90% / 10%)
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    augmenter = DysgraphiaAugmenter() if args.augment else None

    train_dataset = HandwritingWordDataset(
        train_samples,
        alphabet=DEFAULT_ALPHABET,
        augmenter=augmenter,
        augmentation_severity=args.aug_severity,
    )
    val_dataset = HandwritingWordDataset(
        val_samples,
        alphabet=DEFAULT_ALPHABET,
        augmenter=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )

    # Initialize CRNN model
    model = CRNNModel(
        img_height=64,
        in_channels=1,
        num_classes=len(DEFAULT_ALPHABET) + 1,
        hidden_size=256,
        num_lstm_layers=2,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_cer = float("inf")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        elapsed = time.time() - start_time

        logger.info(f"Epoch [{epoch}/{args.epochs}] - Loss: {loss:.4f} - Time: {elapsed:.1f}s")

        if epoch % 5 == 0 or epoch == args.epochs:
            cer, wer = evaluate(model, val_loader, device, DEFAULT_ALPHABET)
            logger.info(f"Epoch [{epoch}] Validation - CER: {cer:.2%} - WER: {wer:.2%}")

            if cer < best_cer:
                best_cer = cer
                best_path = os.path.join(args.output_dir, "crnn_best.pth")
                torch.save(model.state_dict(), best_path)
                logger.info(f"Saved new best model checkpoint to {best_path}")


if __name__ == "__main__":
    main()

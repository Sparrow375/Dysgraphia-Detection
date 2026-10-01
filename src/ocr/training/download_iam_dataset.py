"""
IAM Handwriting Dataset Downloader.

Downloads priyank-m/IAM_words_text_recognition from HuggingFace Hub
(69K train / 23K val / 23K test word images with text labels) and saves them
as PNG files + manifest CSV ready for the CRNN training pipeline.

Usage:
    python src/ocr/training/download_iam_dataset.py --out_dir data/iam_words
    python src/ocr/training/download_iam_dataset.py --out_dir data/iam_words --splits train val
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm


def download_split(
    split: str,
    out_dir: Path,
    dataset_id: str = "priyank-m/IAM_words_text_recognition",
) -> int:
    from datasets import load_dataset

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.csv"

    print(f"\n[IAM] Loading split='{split}' from '{dataset_id}' ...")
    ds = load_dataset(dataset_id, split=split, trust_remote_code=False)
    print(f"  Loaded {len(ds):,} samples.")

    n_written = 0
    n_skipped = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filepath", "text"])

        for i, sample in enumerate(tqdm(ds, desc=f"  Writing {split}")):
            text = (sample.get("text") or "").strip()
            if not text:
                n_skipped += 1
                continue

            pil_img = sample["image"]
            img_path = split_dir / f"{i:06d}.png"
            pil_img.save(str(img_path), format="PNG")
            writer.writerow([img_path.name, text])
            n_written += 1

    print(f"  Written: {n_written:,}  Skipped (empty label): {n_skipped}")
    print(f"  Manifest: {manifest_path}")
    return n_written


def main():
    parser = argparse.ArgumentParser(description="Download IAM words from HuggingFace.")
    parser.add_argument("--out_dir", default="data/iam_words")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--dataset_id", default="priyank-m/IAM_words_text_recognition")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in args.splits:
        manifest = out_dir / split / "manifest.csv"
        if manifest.exists():
            with open(manifest, encoding="utf-8") as f:
                existing = sum(1 for _ in f) - 1
            print(f"[IAM] Split '{split}' already exists ({existing:,} samples). Skipping.")
            total += existing
            continue
        total += download_split(split, out_dir, dataset_id=args.dataset_id)

    print(f"\n[IAM] Done. Total samples: {total:,}")
    print(f"Dataset root: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()

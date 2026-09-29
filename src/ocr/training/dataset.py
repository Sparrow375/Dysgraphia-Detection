"""
Handwriting Dataset Loaders with On-the-Fly Dysgraphic Augmentation.

Supports:
  1. Standard IAM Handwriting format (words.txt and word image directories)
  2. Custom segmented handwriting folders (image + label pairs)
  3. Dynamic on-the-fly dysgraphic augmentation via DysgraphiaAugmenter
  4. Variable-length padding collate function for CTC loss
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
import cv2

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:  # type: ignore
        pass

from src.ocr.char_hypothesis import DEFAULT_ALPHABET
from src.ocr.segmentation import normalize_word_image
from src.ocr.augmentation import DysgraphiaAugmenter


class HandwritingWordDataset(Dataset):
    """
    Word-level handwriting dataset for CRNN training with CTC loss.
    """

    def __init__(
        self,
        samples: List[Tuple[str, str]],  # (image_path, text_label)
        alphabet: str = DEFAULT_ALPHABET,
        target_height: int = 64,
        augmenter: Optional[DysgraphiaAugmenter] = None,
        augmentation_prob: float = 0.5,
        augmentation_severity: float = 0.4,
    ):
        self.samples = samples
        self.alphabet = alphabet
        self.target_height = target_height
        self.augmenter = augmenter
        self.augmentation_prob = augmentation_prob
        self.augmentation_severity = augmentation_severity

        # Build character to index mapping (+1 because 0 is CTC blank)
        self.char2idx: Dict[str, int] = {char: idx + 1 for idx, char in enumerate(alphabet)}
        self.blank_idx = 0

    def __len__(self) -> int:
        return len(self.samples)

    def encode_text(self, text: str) -> List[int]:
        """Convert string to list of character class indices (1-indexed)."""
        indices = []
        for char in text:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
        return indices

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback blank patch if image file is corrupted
            img = np.zeros((self.target_height, 128), dtype=np.uint8)

        # Invert if white background
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)

        # Apply dysgraphic augmentation on the fly with probability p
        if self.augmenter is not None and np.random.rand() < self.augmentation_prob:
            img, _ = self.augmenter.compose(
                img,
                severity=self.augmentation_severity,
                n_augmentations=np.random.randint(1, 4),
            )

        # Normalize word image to fixed height
        norm_img = normalize_word_image(img, target_height=self.target_height)

        # Convert to float tensor (1, H, W) normalized to [0, 1]
        tensor_img = norm_img.astype(np.float32) / 255.0

        target = self.encode_text(label)

        if HAS_TORCH:
            return (
                torch.from_numpy(tensor_img).unsqueeze(0),  # (1, H, W)
                torch.tensor(target, dtype=torch.long),
                label,
            )
        else:
            return tensor_img, target, label


def ctc_collate_fn(batch):
    """
    Custom collate function for batching variable-width word images and targets for CTC loss.

    Pads all images in the batch to the maximum width in the batch.
    """
    images, targets, texts = zip(*batch)

    # Determine max width in the batch
    max_w = max(img.shape[-1] for img in images)
    # Ensure max_w is divisible by 16 for CNN downsampling
    max_w = ((max_w + 15) // 16) * 16

    batch_size = len(images)
    h = images[0].shape[-2]

    # Create padded image batch tensor
    padded_images = torch.zeros((batch_size, 1, h, max_w), dtype=torch.float32)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, img in enumerate(images):
        w = img.shape[-1]
        padded_images[i, :, :, :w] = img
        # CNN has downsampling factor of 4 in width (2 maxpools of stride 2)
        input_lengths[i] = w // 4
        target_lengths[i] = len(targets[i])

    # Concatenate targets into 1D tensor for PyTorch CTC loss
    flat_targets = torch.cat(targets)

    return padded_images, flat_targets, input_lengths, target_lengths, texts


def parse_iam_words_txt(
    words_txt_path: str,
    iam_words_dir: str,
) -> List[Tuple[str, str]]:
    """
    Parse the IAM Handwriting Database `words.txt` index file.

    IAM words.txt format:
      word_id ok_status threshold x y w h grammatical_tag transcription
      e.g.: a01-000u-00-00 ok 154 408 768 27 51 NN A

    Directory structure:
      words/a01/a01-000u/a01-000u-00-00.png
    """
    samples: List[Tuple[str, str]] = []
    base_dir = Path(iam_words_dir)

    if not os.path.exists(words_txt_path):
        return samples

    with open(words_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            word_id = parts[0]
            status = parts[1]
            transcription = parts[8]

            # IAM label 'ok' means good segmentation
            if status != "ok":
                continue

            # Parse path parts: a01-000u-00-00 -> a01 / a01-000u / a01-000u-00-00.png
            id_parts = word_id.split("-")
            if len(id_parts) >= 2:
                dir1 = id_parts[0]
                dir2 = f"{id_parts[0]}-{id_parts[1]}"
                rel_path = base_dir / dir1 / dir2 / f"{word_id}.png"
                if rel_path.exists():
                    samples.append((str(rel_path), transcription))

    return samples

"""Unit tests for Stage 31: De-Noised 3-Channel Clean Ink Tensor Vision Transformer."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from baseline.train_denoised_vit import (
    Denoised3ChDataset,
    build_single_3ch_tensor,
    build_unfrozen_vision_model,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def test_build_single_3ch_tensor() -> None:
    """Verify 3-channel construction, shape, and strict zero background."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mask_path = tmp_path / "clean_ink.png"
        gray_path = tmp_path / "raw.png"

        # Create synthetic 100x300 image (banner crop)
        # Background: white paper (255)
        # Ink: dark stroke (50) in middle
        h, w = 100, 300
        raw = np.full((h, w), 240, dtype=np.uint8)
        raw[40:60, 50:250] = 40  # dark stroke

        # Mask: ink is black (0), background is white (255)
        mask = np.full((h, w), 255, dtype=np.uint8)
        mask[40:60, 50:250] = 0  # ink

        cv2.imwrite(str(gray_path), raw)
        cv2.imwrite(str(mask_path), mask)

        tensor_uint8 = build_single_3ch_tensor(mask_path, gray_path, target_size=224)

        assert tensor_uint8.shape == (224, 224, 3)
        assert tensor_uint8.dtype == np.uint8

        # Background corners must be strictly 0 across all 3 channels
        assert tensor_uint8[0, 0, 0] == 0
        assert tensor_uint8[0, 0, 1] == 0
        assert tensor_uint8[0, 0, 2] == 0
        assert tensor_uint8[220, 220, 0] == 0

        # Center ink stroke must have positive signal
        center_y, center_x = 112, 112
        assert tensor_uint8[center_y, center_x, 0] == 255  # Ch1: Binary ink
        assert tensor_uint8[center_y, center_x, 1] > 0    # Ch2: Pressure/Density
        assert tensor_uint8[center_y, center_x, 2] > 0    # Ch3: Distance transform


def test_denoised_dataset_shapes_and_items() -> None:
    """Verify PyTorch Dataset formatting, normalization, and tensor shapes."""
    dummy_tensors = {
        "uid_1": np.zeros((224, 224, 3), dtype=np.uint8),
        "uid_2": np.full((224, 224, 3), 128, dtype=np.uint8),
    }
    df = pd.DataFrame({
        "task_uid": ["uid_1", "uid_2"],
        "student_id": ["std_1", "std_2"],
        "label": [0.0, 1.0],
        "grade": [3, 4],
    })

    dataset = Denoised3ChDataset(df, dummy_tensors, is_train=False)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert item0["image"].shape == (3, 224, 224)
    assert item0["label"].item() == 0.0
    assert item0["student_id"] == "std_1"

    item1 = dataset[1]
    assert item1["image"].shape == (3, 224, 224)
    assert item1["label"].item() == 1.0
    assert item1["student_id"] == "std_2"


def test_build_unfrozen_swin_t_forward_backward() -> None:
    """Verify Swin-T initialization, parameter unfreezing, forward, and backward pass."""
    model, total_params = build_unfrozen_vision_model(
        model_name="swin_t", pretrained=False, dropout=0.1
    )
    assert total_params > 20_000_000

    # Test forward pass with dummy batch
    dummy_x = torch.randn(2, 3, 224, 224)
    logits = model(dummy_x).squeeze(-1)
    assert logits.shape == (2,)

    # Test backward pass
    loss = logits.sum()
    loss.backward()

    # Verify head gradients exist
    for p in model.head.parameters():
        assert p.grad is not None


def test_split_leakage_contrast() -> None:
    """Verify Run 2A (leaked) vs Run 2B (honest group k-fold) design invariants."""
    # Synthetic dataset with 20 students, each having 5 crops (100 crops total)
    records = []
    for s_idx in range(20):
        sid = f"student_{s_idx:02d}"
        label = 1 if s_idx < 4 else 0  # 4 positive, 16 negative
        for c_idx in range(5):
            records.append({
                "task_uid": f"{sid}_crop_{c_idx}",
                "student_id": sid,
                "label": label,
            })
    df = pd.DataFrame(records)

    # Run 2A split: random crop split
    train_2a, test_2a = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
    train_students_2a = set(train_2a["student_id"])
    test_students_2a = set(test_2a["student_id"])
    overlap_2a = train_students_2a.intersection(test_students_2a)
    assert len(overlap_2a) > 0, "Run 2A must exhibit student overlap (leakage)"

    # Run 2B split: StratifiedGroupKFold on student_id
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["student_id"]))
    train_2b = df.iloc[train_idx]
    test_2b = df.iloc[test_idx]

    train_students_2b = set(train_2b["student_id"])
    test_students_2b = set(test_2b["student_id"])
    overlap_2b = train_students_2b.intersection(test_students_2b)
    assert len(overlap_2b) == 0, "Run 2B must have ZERO student overlap"

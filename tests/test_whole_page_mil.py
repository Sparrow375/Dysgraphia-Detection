"""Unit tests for Stage 32: Whole-Page End-to-End MIL (DenseNet201 + Wide (128, 640) Crops)."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from baseline.train_whole_page_mil import (
    StudentBagDataset,
    WholePageMILModel,
    build_wide_3ch_tensor,
    collate_student_bags,
)


def test_build_wide_3ch_tensor() -> None:
    """Verify wide (128, 640, 3) construction, aspect preservation, and ruled line alignment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mask_path = tmp_path / "clean_ink.png"
        gray_path = tmp_path / "raw.png"
        json_path = tmp_path / "task.json"

        # Synthetic banner crop: 150 x 1500 (10:1 aspect ratio)
        h, w = 150, 1500
        raw = np.full((h, w), 245, dtype=np.uint8)
        raw[50:80, 200:1300] = 35  # dark handwriting stroke

        mask = np.full((h, w), 255, dtype=np.uint8)
        mask[50:80, 200:1300] = 0  # ink in mask

        # Ruled line metadata in JSON
        json_meta = {
            "crop_bbox": [100, 200, w, h],
            "physical_lines": [
                {
                    "baseline_slope": 0.0,
                    "baseline_intercept": 310.0, # in page: 310 - 200 = 110 in crop
                }
            ],
        }

        cv2.imwrite(str(gray_path), raw)
        cv2.imwrite(str(mask_path), mask)
        with open(json_path, "w") as f:
            json.dump(json_meta, f)

        t_wide = build_wide_3ch_tensor(mask_path, gray_path, json_path, target_h=128, target_w=640)

        assert t_wide.shape == (128, 640, 3)
        assert t_wide.dtype == np.uint8

        # Background canvas must be zero
        assert t_wide[0, 0, 0] == 0
        assert t_wide[0, 0, 1] == 0
        assert t_wide[0, 0, 2] == 0

        # Channel 1: Ink signal exists
        assert np.count_nonzero(t_wide[:, :, 0]) > 0

        # Channel 2: Pressure signal exists
        assert np.count_nonzero(t_wide[:, :, 1]) > 0

        # Channel 3: Ruled line signal exists
        assert np.count_nonzero(t_wide[:, :, 2]) > 0


def test_student_bag_dataset_and_collate() -> None:
    """Verify StudentBagDataset loading, data augmentations, and variable-length collate."""
    dummy_tensors = {
        f"crop_{i}": np.zeros((128, 640, 3), dtype=np.uint8)
        for i in range(10)
    }

    student_records = [
        {
            "student_id": "std_01",
            "school": "school_a",
            "label": 1.0,
            "grade": 4,
            "task_uids": ["crop_0", "crop_1", "crop_2", "crop_3"], # 4 crops
            "task_names": ["s1", "s2", "s3", "s4"],
        },
        {
            "student_id": "std_02",
            "school": "school_a",
            "label": 0.0,
            "grade": 5,
            "task_uids": ["crop_4", "crop_5", "crop_6", "crop_7", "crop_8"], # 5 crops
            "task_names": ["s1", "s2", "s3", "s4", "s5"],
        },
    ]

    dataset = StudentBagDataset(student_records, dummy_tensors, is_train=False)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert item0["images"].shape == (4, 3, 128, 640)
    assert item0["label"].item() == 1.0

    batch = [dataset[0], dataset[1]]
    collated = collate_student_bags(batch)

    # 4 + 5 = 9 crops total
    assert collated["all_images"].shape == (9, 3, 128, 640)
    assert collated["split_indices"] == [(0, 4), (4, 9)]
    assert collated["labels"].shape == (2,)


def test_whole_page_mil_gated_attention_forward_backward() -> None:
    """Verify Gated-Attention forward pass, attention weight normalization, and backward gradients."""
    model = WholePageMILModel(pooling_mode="gated_attention", pretrained=False, emb_dim=64, attn_dim=32)

    all_images = torch.randn(7, 3, 128, 640)  # 2 students: 3 crops and 4 crops
    split_indices = [(0, 3), (3, 7)]

    logits, attentions = model(all_images, split_indices)

    assert logits.shape == (2,)
    assert len(attentions) == 2

    # Check attention weights sum to 1.0 per student
    attn_s1 = attentions[0]
    attn_s2 = attentions[1]
    assert attn_s1.shape == (3,)
    assert attn_s2.shape == (4,)
    assert torch.isclose(attn_s1.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(attn_s2.sum(), torch.tensor(1.0), atol=1e-5)

    # Backward pass
    loss = logits.sum()
    loss.backward()

    # Gradients must exist in backbone and attention layers
    for p in model.attention_w.parameters():
        assert p.grad is not None


def test_whole_page_mil_mean_pooling() -> None:
    """Verify Uniform Mean Pooling mode for Run 3A."""
    model = WholePageMILModel(pooling_mode="mean_pooling", pretrained=False, emb_dim=64)

    all_images = torch.randn(6, 3, 128, 640)
    split_indices = [(0, 4), (4, 6)]

    logits, attentions = model(all_images, split_indices)

    assert logits.shape == (2,)
    assert len(attentions) == 2
    # Uniform weights should equal 1/K
    assert torch.allclose(attentions[0], torch.full((4,), 0.25))
    assert torch.allclose(attentions[1], torch.full((2,), 0.50))

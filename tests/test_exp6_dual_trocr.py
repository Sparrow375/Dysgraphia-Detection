"""Automated unit test suite for Experiment 6 Dual-Script TrOCR Vision + Multi-Modal MIL.

Verifies:
1. White-canvas physics (384x384x3, 255 background, dark ink strokes).
2. Duplicate bag augmentations (cutouts, micro-shear).
3. Dual-script sequence bottlenecks (Hindi 768 -> 32, English 384 -> 32).
4. Multi-modal forward pass (line fusion, gated attention, aux legibility loss).
5. SWA weight accumulation and parameter averaging.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from baseline.cache_dual_trocr_vision import (
    apply_micro_shear,
    apply_vertical_cutout,
    build_white_canvas_tensor,
)
from baseline.dual_trocr_multimodal_mil import (
    DualScriptMultiModalMILModel,
    DualScriptSequenceBottleneck,
    SWAModelTracker,
)


def test_white_canvas_physics(tmp_path):
    """Verify that white-canvas tensor has 384x384x3 shape, 255 background, and dark ink."""
    import cv2

    h, w = 100, 400
    # Create synthetic raw grayscale crop (bright paper ~220, dark ink line ~60)
    raw_gray = np.full((h, w), 220, dtype=np.uint8)
    raw_gray[40:60, 50:350] = 60  # Ink stroke

    # Create synthetic clean ink mask (0=ink, 255=background)
    mask_gray = np.full((h, w), 255, dtype=np.uint8)
    mask_gray[40:60, 50:350] = 0

    raw_path = tmp_path / "raw.png"
    mask_path = tmp_path / "mask.png"
    json_path = tmp_path / "dummy.json"

    cv2.imwrite(str(raw_path), raw_gray)
    cv2.imwrite(str(mask_path), mask_gray)

    tensor = build_white_canvas_tensor(mask_path, raw_path, json_path, target_size=384)

    assert tensor.shape == (384, 384, 3)
    assert tensor.dtype == np.uint8

    # Corner pixel should be pure white canvas (255, 255, 255)
    assert tensor[0, 0, 0] == 255
    assert tensor[0, 0, 1] == 255
    assert tensor[0, 0, 2] == 255

    # Center where ink stroke was pasted should be dark in Ch1 (inverted ink: 0)
    center_y, center_x = 192, 192
    assert tensor[center_y, center_x, 0] < 50
    assert tensor[center_y, center_x, 1] < 120


def test_duplicate_bag_augmentations():
    """Verify vertical cutout strips and micro-shear transformations."""
    canvas = np.zeros((384, 384, 3), dtype=np.uint8)

    # Cutout should introduce vertical 255 whiteout strips
    cutout = apply_vertical_cutout(canvas, num_strips=2, strip_w=20)
    assert cutout.shape == (384, 384, 3)
    assert np.count_nonzero(cutout == 255) >= 384 * 20 * 2 * 3

    # Micro-shear should maintain dimensions
    sheared = apply_micro_shear(canvas, max_shear_deg=1.5)
    assert sheared.shape == (384, 384, 3)


def test_dual_script_sequence_bottleneck():
    """Verify Hindi (768) and English (384) sequences project to identical R^32 vectors."""
    device = torch.device("cpu")
    bottleneck = DualScriptSequenceBottleneck(out_dim=32, dropout=0.0).to(device)

    # 1 Hindi sequence (24, 768), 1 English sequence (24, 384)
    tokens_hi = torch.randn(24, 768)
    tokens_en = torch.randn(24, 384)

    seq_list = [tokens_hi, tokens_en]
    scripts = ["devanagari", "latin"]

    out = bottleneck(seq_list, scripts)
    assert out.shape == (2, 32)
    assert not torch.isnan(out).any()


def test_dual_script_mil_forward_pass():
    """Verify end-to-end forward pass of DualScriptMultiModalMILModel."""
    device = torch.device("cpu")
    model = DualScriptMultiModalMILModel(math_dim=16, visual_dim=32, math_emb_dim=32).to(device)

    # Student 1: 2 sentences (1 Hindi, 1 English)
    # Student 2: 1 sentence (English)
    tokens_list = [
        torch.randn(24, 768),  # S1 L1 (Hindi)
        torch.randn(24, 384),  # S1 L2 (English)
        torch.randn(24, 384),  # S2 L1 (English)
    ]
    scripts_list = ["devanagari", "latin", "latin"]
    math_tensor = torch.randn(3, 16)
    splits = [(0, 2), (2, 3)]

    logits, attentions, aux_preds = model(tokens_list, scripts_list, math_tensor, splits)

    assert logits.shape == (2,)
    assert aux_preds.shape == (3, 2)
    assert len(attentions) == 2
    assert torch.isclose(attentions[0].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(attentions[1].sum(), torch.tensor(1.0), atol=1e-5)


def test_swa_model_tracker():
    """Verify SWA accumulates weights and correctly loads mean parameters."""
    toy = nn.Sequential(nn.Linear(4, 2, bias=False))
    tracker = SWAModelTracker(toy)

    # Model 1: weight = 1.0
    toy[0].weight.data.fill_(1.0)
    tracker.update(toy)

    # Model 2: weight = 3.0
    toy[0].weight.data.fill_(3.0)
    tracker.update(toy)

    # Model 3: weight = 5.0
    toy[0].weight.data.fill_(5.0)
    tracker.update(toy)

    # Average should be (1 + 3 + 5) / 3 = 3.0
    toy[0].weight.data.fill_(0.0)
    tracker.apply_to(toy)

    assert torch.allclose(toy[0].weight.data, torch.full((2, 4), 3.0))


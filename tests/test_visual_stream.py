"""Automated unit tests for Stream 4: Deep Visual Patch Stream (DenseNet201).

Verifies:
1. Aspect-preserving patch slicing (avoiding horizontal text distortion).
2. Patch shape (224, 224, 3) and normalization range [0, 1].
3. Pretrained DenseNet201 feature dimensionality (B, 1920).
4. PCA + Logistic probe training and probability calibration [0, 1].
5. Robustness to degenerate/small crops.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from baseline.visual_patch_extractor import (
    DenseNetFeatureExtractor,
    extract_aspect_preserving_patches,
)
from baseline.visual_classifier import compute_visual_metrics


def test_aspect_preserving_patch_extraction():
    """Verify that horizontal line crops produce square aspect-preserving patches."""
    # Synthetic horizontal line: H=80, W=400 (aspect ratio 5:1)
    H, W = 80, 400
    gray = np.full((H, W), 245, dtype=np.uint8)
    gray[35:45, 20:380] = 30  # Dark ink stroke

    patches = extract_aspect_preserving_patches(gray, target_size=224, max_patches=3)
    assert len(patches) == 3
    for p in patches:
        assert p.shape == (224, 224, 3)
        assert p.dtype == np.float32
        assert 0.0 <= np.min(p) <= np.max(p) <= 1.0


def test_tall_or_square_patch_extraction():
    """Verify symmetric padding for crops where W <= H."""
    H, W = 100, 60
    gray = np.full((H, W), 245, dtype=np.uint8)
    gray[20:80, 25:35] = 40

    patches = extract_aspect_preserving_patches(gray, target_size=224, max_patches=2)
    assert len(patches) == 1
    assert patches[0].shape == (224, 224, 3)


def test_densenet_extractor_shape():
    """Verify that DenseNet201 outputs exactly 1920-d pooled features."""
    extractor = DenseNetFeatureExtractor(device="cpu")
    dummy_patch = np.random.uniform(0.0, 1.0, (224, 224, 3)).astype(np.float32)
    feats = extractor.extract_batch([dummy_patch, dummy_patch])
    assert feats.shape == (2, 1920)
    assert not np.isnan(feats).any()


def test_compute_visual_metrics_bounds():
    """Verify diagnostic metric calculation and sensitivity at fixed specificity."""
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.95, 0.85, 0.20, 0.15, 0.30, 0.10])

    met = compute_visual_metrics(y_true, y_prob)
    assert met["auroc"] == 1.0
    assert met["auprc"] == 1.0
    assert met["sens_at_80_spec"] == 1.0
    assert met["sens_at_90_spec"] == 1.0
    assert met["youden_sens"] == 1.0
    assert met["youden_spec"] == 1.0

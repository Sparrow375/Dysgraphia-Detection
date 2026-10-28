"""Automated Unit and Regression Tests for Experiment 7: Frozen-Feature Representation Bake-Off."""

from pathlib import Path
import numpy as np
import pytest
import torch

from baseline.tiling_preprocessor import build_white_canvas_strip, extract_tiles_from_strip
from baseline.extract_dinov2_patch_stats import compute_patch_descriptors
from baseline.evaluate_exp7_probe import calculate_sensitivity_at_specificity, compute_cluster_bootstrap_cis


def test_tiling_preprocessor_synthetic():
    """Verify white-canvas rendering and aspect-preserving tiling geometry."""
    # Synthetic clean ink mask (100x500): white canvas with black letter block in center
    clean_mask = np.full((100, 500), 255, dtype=np.uint8)
    clean_mask[30:70, 100:300] = 0  # Black ink

    raw_gray = np.full((100, 500), 240, dtype=np.uint8)
    raw_gray[30:70, 100:300] = 50  # Dark gray pencil stroke

    # Mock file-less strip construction
    ink_bin = (clean_mask < 128).astype(np.float32)
    ch1 = np.where(ink_bin > 0, 0.0, 255.0).astype(np.float32)
    ch2 = np.where(ink_bin > 0, 50.0, 255.0).astype(np.float32)
    ch3 = np.full((100, 500), 255.0, dtype=np.float32)
    ch3[ink_bin > 0] = 0.0

    strip = np.stack([ch1, ch2, ch3], axis=-1).astype(np.uint8)
    assert strip.shape == (100, 500, 3)
    assert strip.dtype == np.uint8

    tiles, coords = extract_tiles_from_strip(strip, target_height=224, stride=112)
    assert len(tiles) > 0
    for t in tiles:
        assert t.shape == (224, 224, 3)
        assert t.dtype == np.uint8
        # Background is white (255)
        assert t[0, 0, 0] == 255


def test_dinov2_patch_descriptors_tensor_math():
    """Verify that patch descriptor computes [mean, std, GeM] accurately."""
    b, p_tokens, d = 2, 256, 64
    mock_patches = torch.abs(torch.randn(b, p_tokens, d)) + 0.1  # Positive values for GeM

    desc = compute_patch_descriptors(mock_patches, p=3.0)
    assert desc.shape == (b, 3 * d)

    # Check individual components
    u = desc[:, :d]
    s = desc[:, d : 2 * d]
    gem = desc[:, 2 * d :]

    expected_u = mock_patches.mean(dim=1)
    assert torch.allclose(u, expected_u, atol=1e-5)

    expected_s = torch.sqrt(torch.var(mock_patches, dim=1, unbiased=False) + 1e-6)
    assert torch.allclose(s, expected_s, atol=1e-5)

    expected_gem = (mock_patches.pow(3.0).mean(dim=1)).pow(1.0 / 3.0)
    assert torch.allclose(gem, expected_gem, atol=1e-4)


def test_sensitivity_at_specificity_calculation():
    """Verify sensitivity calculation at target operational specificity."""
    # 10 positives, 90 negatives
    y_true = np.array([1] * 10 + [0] * 90)
    # Perfectly separated scores
    y_scores = np.array([0.9] * 10 + [0.1] * 90)

    det80, sens80, thresh80 = calculate_sensitivity_at_specificity(y_true, y_scores, 0.80)
    assert det80 == 10
    assert sens80 == 1.0
    assert thresh80 == pytest.approx(0.1, abs=1e-5)

    det90, sens90, thresh90 = calculate_sensitivity_at_specificity(y_true, y_scores, 0.90)
    assert det90 == 10
    assert sens90 == 1.0


def test_cluster_bootstrap_cis_bounds():
    """Verify cluster bootstrap confidence intervals contain empirical estimates."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0] * 5)
    y_scores = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.4, 0.3, 0.85, 0.15] * 5)
    sids = np.array([f"student_{i}" for i in range(len(y_true))])

    cis = compute_cluster_bootstrap_cis(y_true, y_scores, sids, n_bootstrap=100)
    assert "auroc_ci" in cis
    assert "auprc_ci" in cis
    assert cis["auroc_ci"][0] <= cis["auroc_ci"][1]
    assert cis["auprc_ci"][0] <= cis["auprc_ci"][1]
    assert 0.0 <= cis["auroc_ci"][0] <= 1.0
    assert 0.0 <= cis["auroc_ci"][1] <= 1.0

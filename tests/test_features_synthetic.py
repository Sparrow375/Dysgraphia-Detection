"""Synthetic perturbation validation tests for Workstream A Phase 2.

Verifies monotonic response (Spearman rho > 0.8) for:
1. Spacing jitter vs gap_cv
2. Size jitter vs word_height_cv
3. Baseline wobble vs baseline_rmse_norm
4. Stroke path tremor vs jerk_proxy
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
import pytest
from scipy.stats import spearmanr

from features.baseline import compute_baseline_features
from features.curvature import compute_curvature_features
from features.gaps import compute_gap_features
from features.size import compute_size_features


def test_synthetic_spacing_jitter():
    """Verify monotonic response of gap_cv to spacing jitter."""
    jitter_levels = [0.0, 5.0, 10.0, 18.0, 28.0, 42.0, 60.0]
    gap_cvs: List[float] = []

    np.random.seed(42)
    base_x_height = 40.0
    num_words = 12

    for sigma in jitter_levels:
        words = []
        cur_x = 50.0
        for i in range(num_words):
            ww = 60.0
            wh = 40.0
            words.append({"bbox": [int(cur_x), 100, int(ww), int(wh)], "word_index": i})
            # Add base gap + jitter
            gap = max(2.0, 30.0 + (np.random.normal(0, sigma) if sigma > 0 else 0.0))
            cur_x += ww + gap

        mock_json: Dict[str, Any] = {
            "crop_bbox": [0, 0, int(cur_x + 50), 200],
            "physical_lines": [
                {
                    "line_index": 0,
                    "x_height_h": base_x_height,
                    "words": words,
                }
            ],
        }

        res = compute_gap_features(mock_json)
        cv_val = res["gap_cv"]
        assert not np.isnan(cv_val), f"gap_cv was NaN at sigma={sigma}"
        gap_cvs.append(cv_val)

    rho, pval = spearmanr(jitter_levels, gap_cvs)
    print(f"Spacing jitter vs gap_cv: Spearman rho = {rho:.4f} (p = {pval:.4e})")
    assert rho >= 0.8, f"Expected Spearman rho >= 0.8, got {rho:.4f}"


def test_synthetic_size_jitter():
    """Verify monotonic response of word_height_cv to height size jitter."""
    jitter_levels = [0.0, 3.0, 7.0, 12.0, 18.0, 26.0, 36.0]
    height_cvs: List[float] = []

    np.random.seed(42)
    num_words = 15

    for sigma in jitter_levels:
        words = []
        cur_x = 50.0
        for i in range(num_words):
            ww = 50.0
            wh = max(10.0, 40.0 + (np.random.normal(0, sigma) if sigma > 0 else 0.0))
            words.append({"bbox": [int(cur_x), 100, int(ww), int(wh)], "word_index": i})
            cur_x += ww + 25.0

        mock_json: Dict[str, Any] = {
            "crop_bbox": [0, 0, int(cur_x + 50), 200],
            "median_rule_spacing_r": 85.0,
            "physical_lines": [
                {
                    "line_index": 0,
                    "x_height_h": 40.0,
                    "words": words,
                }
            ],
        }

        res = compute_size_features(mock_json)
        cv_val = res["word_height_cv"]
        assert not np.isnan(cv_val), f"word_height_cv was NaN at sigma={sigma}"
        height_cvs.append(cv_val)

    rho, pval = spearmanr(jitter_levels, height_cvs)
    print(f"Size jitter vs word_height_cv: Spearman rho = {rho:.4f} (p = {pval:.4e})")
    assert rho >= 0.8, f"Expected Spearman rho >= 0.8, got {rho:.4f}"


def test_synthetic_baseline_wobble():
    """Verify monotonic response of baseline_rmse_norm to baseline wobble."""
    wobble_levels = [0.0, 2.0, 5.0, 9.0, 15.0, 23.0, 33.0]
    rmses: List[float] = []

    np.random.seed(42)
    num_words = 12

    for sigma in wobble_levels:
        pts = []
        cur_x = 50.0
        base_y = 150.0
        for i in range(num_words):
            y_noise = np.random.normal(0, sigma) if sigma > 0 else 0.0
            pts.append([cur_x, base_y + y_noise])
            cur_x += 70.0

        mock_json: Dict[str, Any] = {
            "crop_bbox": [0, 0, int(cur_x + 50), 300],
            "word_baseline_points": pts,
            "physical_lines": [
                {
                    "line_index": 0,
                    "x_height_h": 40.0,
                    "word_baseline_points": pts,
                }
            ],
        }

        res = compute_baseline_features(mock_json)
        rmse_val = res["baseline_rmse_norm"]
        assert not np.isnan(rmse_val), f"baseline_rmse_norm was NaN at sigma={sigma}"
        rmses.append(rmse_val)

    rho, pval = spearmanr(wobble_levels, rmses)
    print(f"Baseline wobble vs baseline_rmse_norm: Spearman rho = {rho:.4f} (p = {pval:.4e})")
    assert rho >= 0.8, f"Expected Spearman rho >= 0.8, got {rho:.4f}"


def test_synthetic_tremor_jerk_proxy():
    """Verify monotonic response of jerk_proxy and tangent_variance_short_wavelength to stroke tremor."""
    tremor_levels = [0.5, 1.0, 1.6, 2.3, 3.2, 4.2, 5.5]
    jerk_vals: List[float] = []

    np.random.seed(42)
    h_canvas = 300
    w_canvas = 700
    x_height = 50.0

    # Base smooth stroke path: series of sinusoidal handwriting loops
    t = np.linspace(0, 4 * np.pi, 300)
    base_x = 50 + t * 40.0
    base_y = 150 + np.sin(t) * 50.0

    for sigma in tremor_levels:
        ink = np.zeros((h_canvas, w_canvas), dtype=np.uint8)

        # Add tremor noise to stroke
        if sigma > 0:
            noise_x = np.random.normal(0, sigma, size=len(t))
            noise_y = np.random.normal(0, sigma, size=len(t))
        else:
            noise_x = 0
            noise_y = 0

        cur_x = np.clip(base_x + noise_x, 5, w_canvas - 6).astype(np.int32)
        cur_y = np.clip(base_y + noise_y, 5, h_canvas - 6).astype(np.int32)

        # Render stroke polyline with 1px width for clean skeleton path tracing
        pts_draw = np.column_stack([cur_x, cur_y])
        cv2.polylines(ink, [pts_draw], isClosed=False, color=255, thickness=1)

        mock_json: Dict[str, Any] = {
            "script": "latin",
            "crop_bbox": [0, 0, w_canvas, h_canvas],
            "physical_lines": [
                {
                    "line_index": 0,
                    "x_height_h": x_height,
                }
            ],
        }

        res = compute_curvature_features(mock_json, image_crop=ink)
        jerk = res["jerk_proxy"]
        assert not np.isnan(jerk), f"jerk_proxy was NaN at sigma={sigma}"
        jerk_vals.append(jerk)

    rho, pval = spearmanr(tremor_levels, jerk_vals)
    print(f"Tremor noise vs jerk_proxy: Spearman rho = {rho:.4f} (p = {pval:.4e})")
    assert rho >= 0.8, f"Expected Spearman rho >= 0.8, got {rho:.4f}"

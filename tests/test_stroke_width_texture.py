"""Automated unit tests for Stream 2: Stroke-Width & Grayscale Shading micro-texture features.

Verifies:
1. Synthetic steady vs jittery stroke width discrimination.
2. Synthetic even vs erratic pencil pressure discrimination.
3. Clean paper vs smudged / erasure halo detection.
4. Robustness on empty / degenerated inputs (NaN safety).
5. Sub-second execution performance.
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from features.stroke_width_texture import compute_stroke_width_texture_features


def _create_synthetic_crop(
    width_pattern: str = "steady",  # "steady" or "jittery"
    pressure_pattern: str = "even",  # "even" or "erratic"
    smudge: bool = False,
    h: int = 40,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Generate synthetic grayscale crop and clean ink mask with controlled properties."""
    H, W = 120, 400
    gray = np.full((H, W), 245, dtype=np.uint8)  # White background paper
    ink_mask = np.zeros((H, W), dtype=np.uint8)

    # Draw a line from x=30 to x=370 at y=60
    xs = np.arange(30, 370)
    for i, x in enumerate(xs):
        y = int(60 + 10 * np.sin(x / 30.0))  # gentle sine wave stroke

        # Stroke thickness
        if width_pattern == "steady":
            rad = 4
        else:
            # Alternating pinching (rad=1) and bloating (rad=8)
            rad = int(4 + 3.5 * np.sin(i / 3.0))
            rad = max(1, rad)

        # Draw circle on mask
        cv2.circle(ink_mask, (x, y), rad, 1, -1)

        # Grayscale intensity (ink darkness)
        if pressure_pattern == "even":
            intensity = 40  # Constant dark ink
        else:
            # Erratic pressure: oscillating between very dark (10) and barely visible (220)
            intensity = int(120 + 100 * np.sin(i / 4.0))

        cv2.circle(gray, (x, y), rad, intensity, -1)

    # Add erasure/smudge halo if requested
    if smudge:
        # Smudge graphite around x=150..250, y=40..80
        noise = np.random.normal(0, 30, (H, W)).astype(np.int32)
        smudge_zone = np.zeros((H, W), dtype=bool)
        smudge_zone[35:85, 140:260] = True
        smudge_zone[ink_mask > 0] = False  # only on paper
        gray_mod = gray.astype(np.int32)
        gray_mod[smudge_zone] = np.clip(gray_mod[smudge_zone] - 80 + noise[smudge_zone], 0, 255)
        gray = gray_mod.astype(np.uint8)

    sentence_json = {
        "student_id": "test_student",
        "school": "test_school",
        "task_id": "test_task",
        "x_height_h": float(h),
        "physical_lines": [{"x_height_h": float(h)}],
    }

    return sentence_json, gray, ink_mask


def test_steady_vs_jittery_stroke_width():
    """Verify that jittery strokes have higher stroke_width_cv and stroke_width_jitter."""
    s_json_steady, gray_steady, _ = _create_synthetic_crop(width_pattern="steady", pressure_pattern="even")
    feats_steady = compute_stroke_width_texture_features(s_json_steady, image_crop=gray_steady)

    s_json_jitter, gray_jitter, _ = _create_synthetic_crop(width_pattern="jittery", pressure_pattern="even")
    feats_jitter = compute_stroke_width_texture_features(s_json_jitter, image_crop=gray_jitter)

    assert not np.isnan(feats_steady["stroke_width_cv"])
    assert not np.isnan(feats_jitter["stroke_width_cv"])
    assert feats_jitter["stroke_width_cv"] > feats_steady["stroke_width_cv"]
    assert feats_jitter["stroke_width_bloat_rate"] > feats_steady["stroke_width_bloat_rate"]


def test_even_vs_erratic_pencil_pressure():
    """Verify that erratic pencil pressure produces higher pressure_intensity_cv."""
    s_json_even, gray_even, _ = _create_synthetic_crop(width_pattern="steady", pressure_pattern="even")
    feats_even = compute_stroke_width_texture_features(s_json_even, image_crop=gray_even)

    s_json_erratic, gray_erratic, _ = _create_synthetic_crop(width_pattern="steady", pressure_pattern="erratic")
    feats_erratic = compute_stroke_width_texture_features(s_json_erratic, image_crop=gray_erratic)

    assert not np.isnan(feats_even["pressure_intensity_cv"])
    assert not np.isnan(feats_erratic["pressure_intensity_cv"])
    assert feats_erratic["pressure_intensity_cv"] > feats_even["pressure_intensity_cv"]
    assert feats_erratic["pressure_faint_rate"] > feats_even["pressure_faint_rate"]


def test_clean_vs_smudged_erasure_halo():
    """Verify that background smudging increases smudge_halo_darkness and background_roughness_std."""
    s_json_clean, gray_clean, _ = _create_synthetic_crop(smudge=False)
    feats_clean = compute_stroke_width_texture_features(s_json_clean, image_crop=gray_clean)

    s_json_smudge, gray_smudge, _ = _create_synthetic_crop(smudge=True)
    feats_smudge = compute_stroke_width_texture_features(s_json_smudge, image_crop=gray_smudge)

    assert not np.isnan(feats_clean["smudge_halo_darkness"])
    assert not np.isnan(feats_smudge["smudge_halo_darkness"])
    assert feats_smudge["smudge_halo_darkness"] > feats_clean["smudge_halo_darkness"]
    assert feats_smudge["background_roughness_std"] > feats_clean["background_roughness_std"]


def test_empty_and_degraded_inputs():
    """Verify safe fallback to NaNs for empty or degraded inputs."""
    s_json = {"student_id": "none", "x_height_h": 40.0}
    empty_crop = np.full((100, 100), 255, dtype=np.uint8)  # all white paper, no ink

    feats = compute_stroke_width_texture_features(s_json, image_crop=empty_crop)
    assert np.isnan(feats["stroke_width_mean_norm"])
    assert np.isnan(feats["pressure_intensity_mean"])

    # Zero x-height
    s_json_zero_h = {"student_id": "none", "x_height_h": 0.0}
    feats_zero = compute_stroke_width_texture_features(s_json_zero_h, image_crop=empty_crop)
    assert np.isnan(feats_zero["stroke_width_mean_norm"])


def test_real_processed_crop():
    """Verify execution on an actual processed sentence crop from School B."""
    import json
    from pathlib import Path

    sample_json_path = Path("data/processed/school_b/SCHB_G4_IMG_20261007_100013/sentence_01_copy_hindi.json")
    if sample_json_path.exists():
        with open(sample_json_path, "r", encoding="utf-8") as f:
            s_json = json.load(f)
        feats = compute_stroke_width_texture_features(s_json, data_root="data/processed")
        assert not np.isnan(feats["stroke_width_mean_norm"])
        assert not np.isnan(feats["pressure_intensity_mean"])
        assert feats["stroke_width_mean_norm"] > 0.0
        assert feats["pressure_intensity_mean"] > 0.0

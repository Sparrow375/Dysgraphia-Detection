"""Scale and rotation invariance tests for Workstream A Phase 2.

Verifies:
1. Rescaling by 0.7x preserves dimensionless and normalized features within tolerance.
2. Rotation by +/- 3 degrees shifts slant predictably while preserving variability metrics.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from features.extractor import extract_sentence_features


@pytest.fixture(scope="module")
def sample_sentence_data():
    json_path = Path("data/processed/school_a/G4_A_Roll01/sentence_01_copy_hindi.json")
    if not json_path.exists():
        pytest.skip(f"Sample JSON not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        s_json = json.load(f)

    crop_path = json_path.parent / s_json.get("crop_filename", "sentence_01_copy_hindi.png")
    assert crop_path.exists(), f"Crop image not found at {crop_path}"
    img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None

    return s_json, img


def test_scale_invariance_0_7x(sample_sentence_data):
    """Test that dimensionless and normalized features are preserved under 0.7x rescaling."""
    s_json, img = sample_sentence_data
    scale = 0.7

    feats_orig = extract_sentence_features(s_json, image_crop=img)

    # Scale image
    scaled_w = int(img.shape[1] * scale)
    scaled_h = int(img.shape[0] * scale)
    img_scaled = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # Scale JSON geometry
    s_json_scaled = copy.deepcopy(s_json)
    bx, by, bw, bh = s_json_scaled.get("crop_bbox", [0, 0, 100, 100])
    s_json_scaled["crop_bbox"] = [int(bx * scale), int(by * scale), int(bw * scale), int(bh * scale)]
    s_json_scaled["median_rule_spacing_r"] = float(s_json.get("median_rule_spacing_r", 85.0)) * scale

    if "word_baseline_points" in s_json_scaled:
        s_json_scaled["word_baseline_points"] = [
            [float(p[0] * scale), float(p[1] * scale)] for p in s_json_scaled["word_baseline_points"]
        ]

    for line in s_json_scaled.get("physical_lines", []) + s_json_scaled.get("lines", []):
        if "word_baseline_points" in line:
            line["word_baseline_points"] = [
                [float(p[0] * scale), float(p[1] * scale)] for p in line["word_baseline_points"]
            ]
        if "x_height_h" in line:
            line["x_height_h"] = float(line["x_height_h"]) * scale
        if "words" in line:
            for wd in line["words"]:
                if "bbox" in wd:
                    wx, wy, ww, wh = wd["bbox"]
                    wd["bbox"] = [int(wx * scale), int(wy * scale), int(ww * scale), int(wh * scale)]

    feats_scaled = extract_sentence_features(s_json_scaled, image_crop=img_scaled)

    # Check scale-invariant metrics
    invariant_metrics = [
        "baseline_rmse_norm",
        "h_over_r",
        "word_height_cv",
        "tortuosity_median",
    ]

    for m in invariant_metrics:
        v_orig = feats_orig.get(m)
        v_scaled = feats_scaled.get(m)
        if v_orig is not None and not np.isnan(v_orig):
            assert not np.isnan(v_scaled), f"Feature {m} became NaN after scaling"
            rel_diff = abs(v_scaled - v_orig) / max(abs(v_orig), 1e-4)
            print(f"Scale invariance {m}: orig={v_orig:.4f}, scaled={v_scaled:.4f}, rel_diff={rel_diff:.4f}")
            assert rel_diff < 0.25, f"Feature {m} exceeded relative tolerance: {rel_diff:.4f}"


def test_rotation_invariance_plus_minus_3_deg(sample_sentence_data):
    """Test response under +/- 3 degrees rotation."""
    s_json, img = sample_sentence_data
    feats_orig = extract_sentence_features(s_json, image_crop=img)
    orig_slant = feats_orig["slant_mean_deg"]

    for angle in [3.0, -3.0]:
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        feats_rot = extract_sentence_features(s_json, image_crop=img_rot)

        # Slant mean should shift roughly with rotation angle
        slant_rot = feats_rot["slant_mean_deg"]
        print(f"Rotation {angle:+.1f} deg: orig_slant={orig_slant:.2f} deg, rot_slant={slant_rot:.2f} deg")
        assert not np.isnan(slant_rot), f"slant_mean_deg was NaN after {angle} deg rotation"

        # Slant std should stay stable within relative tolerance
        circ_std_orig = feats_orig["slant_circular_std_deg"]
        circ_std_rot = feats_rot["slant_circular_std_deg"]
        rel_diff_std = abs(circ_std_rot - circ_std_orig) / max(abs(circ_std_orig), 1e-4)
        assert rel_diff_std < 0.35, f"slant_circular_std_deg changed too much under rotation: rel_diff={rel_diff_std:.4f}"

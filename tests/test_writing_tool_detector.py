"""Unit tests for the Pen vs. Pencil Classifier and Tool Calibration (Stage 25).

Verifies:
1. Pure-math classification on synthetic pencil-like and pen-like patterns.
2. Stroke edge gradient and optical density calculation.
3. Tool-specific baseline calibration without NaNs or data leakage.
"""

import numpy as np
import pandas as pd
import pytest

from features.stroke_width_texture import classify_writing_tool, calibrate_stream2_by_tool


def test_synthetic_pencil_classification():
    """Verify soft, gray, wide stroke is classified as pencil."""
    H, W = 200, 300
    # Clean paper background ~240
    img = np.full((H, W), 240, dtype=np.uint8)
    # Graphite stroke: lighter gray (140-160), wider (diameter ~6px), soft edges
    img[95:105, 50:250] = 145
    # Soft blurry edges
    img[94, 50:250] = 185
    img[105, 50:250] = 185

    res = classify_writing_tool(img, bg_level=240.0)
    assert res["tool"] == "pencil"
    assert res["is_pencil"] == 1.0
    assert res["w_med"] >= 2.5
    assert res["sat_ratio"] < 0.05


def test_synthetic_pen_classification():
    """Verify sharp, saturated dark stroke is classified as pen."""
    H, W = 200, 300
    img = np.full((H, W), 245, dtype=np.uint8)
    # Ballpoint/Gel pen: saturated dark core (I ~ 20), crisp thin line (2-3px)
    img[98:101, 50:250] = 20

    res = classify_writing_tool(img, bg_level=245.0)
    assert res["tool"] == "pen"
    assert res["is_pencil"] == 0.0
    assert res["sat_ratio"] > 0.10
    assert res["edge_grad"] > 300.0


def test_calibrate_stream2_by_tool():
    """Verify Stream 2 tool-specific calibration normalizes features without NaNs."""
    rng = np.random.RandomState(42)
    N = 40
    is_pencil = np.array([1.0] * 20 + [0.0] * 20)
    labels = np.array([0, 0, 0, 0, 1] * 8)

    # Pencil distribution: higher pressure CV (0.40 vs 0.20)
    p_cv = np.where(is_pencil == 1.0, rng.normal(0.40, 0.05, N), rng.normal(0.20, 0.03, N))
    w_jit = np.where(is_pencil == 1.0, rng.normal(0.30, 0.04, N), rng.normal(0.50, 0.06, N))

    df = pd.DataFrame({
        "student_id": [f"S_{i}" for i in range(N)],
        "is_pencil": is_pencil,
        "label": labels,
        "pressure_intensity_cv": p_cv,
        "stroke_width_jitter": w_jit,
    })

    calib_df, params = calibrate_stream2_by_tool(df, ["pressure_intensity_cv", "stroke_width_jitter"])

    assert "pressure_intensity_cv_tool_norm" in calib_df.columns
    assert "stroke_width_jitter_tool_norm" in calib_df.columns
    assert not calib_df["pressure_intensity_cv_tool_norm"].isna().any()
    assert not calib_df["stroke_width_jitter_tool_norm"].isna().any()

    # Verify both tool groups have approximately zero mean on controls
    p_ctrl_norm = calib_df[(calib_df["is_pencil"] == 1.0) & (calib_df["label"] == 0)]["pressure_intensity_cv_tool_norm"]
    pen_ctrl_norm = calib_df[(calib_df["is_pencil"] == 0.0) & (calib_df["label"] == 0)]["pressure_intensity_cv_tool_norm"]
    assert abs(p_ctrl_norm.median()) < 0.2
    assert abs(pen_ctrl_norm.median()) < 0.2

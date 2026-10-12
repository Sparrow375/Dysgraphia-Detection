"""Unit tests for Stylus-Anchored Motion Predictor."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from models.motion_predictor import (
    PREDICTOR_PAPER_FEATURES,
    TARGET_KINEMATIC_METRICS,
    StylusMotionPredictor,
)


@pytest.fixture
def mock_matched_dataset():
    """Create a synthetic matched dataset with 50 students."""
    np.random.seed(42)
    n = 50
    data = {
        "student_id": [f"S_{i:03d}" for i in range(n)],
        "grade": np.random.choice([3, 4, 5], size=n),
        "label": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
    }

    # Add synthetic paper features
    for f in PREDICTOR_PAPER_FEATURES:
        data[f] = np.random.normal(loc=1.0, scale=0.3, size=n)

    # Add synthetic target kinematic features correlated with fragmentation
    data["stroke_count"] = 50 + 20 * data["endpoints_per_unit_width_mean"] + np.random.normal(0, 5, n)
    data["air_ratio"] = 0.5 + 0.1 * data["gap_fraction_above_2h_mean"] + np.random.normal(0, 0.05, n)
    data["mean_pause_sec"] = 1.0 + 0.3 * data["gap_cv_mean"] + np.random.normal(0, 0.1, n)
    data["cv_vel"] = 0.8 + 0.2 * data["jerk_proxy_mean"] + np.random.normal(0, 0.1, n)
    data["mean_acc"] = 10000.0 + np.random.normal(0, 1000, n)

    return pd.DataFrame(data)


def test_motion_predictor_fit_cv(mock_matched_dataset):
    """Verify fitting and 5-fold CV computation."""
    predictor = StylusMotionPredictor()
    cv_metrics = predictor.fit_cv(mock_matched_dataset, n_splits=3, random_state=42)

    assert "stroke_count" in cv_metrics
    assert "air_ratio" in cv_metrics
    assert "mean_pause_sec" in cv_metrics

    for tgt, met in cv_metrics.items():
        assert "pearson_r" in met
        assert not np.isnan(met["pearson_r"])
        assert met["n_samples"] == len(mock_matched_dataset)


def test_motion_predictor_predict(mock_matched_dataset):
    """Verify prediction produces expected columns and no NaNs."""
    predictor = StylusMotionPredictor()
    predictor.fit_cv(mock_matched_dataset, n_splits=3, random_state=42)

    preds_df = predictor.predict(mock_matched_dataset)
    for tgt in TARGET_KINEMATIC_METRICS:
        col = f"motion_pred_{tgt}"
        assert col in preds_df.columns
        assert not preds_df[col].isna().any()


def test_motion_predictor_save(mock_matched_dataset, tmp_path):
    """Verify model weights serialization to JSON."""
    predictor = StylusMotionPredictor()
    predictor.fit_cv(mock_matched_dataset, n_splits=3, random_state=42)

    save_path = tmp_path / "test_weights.json"
    predictor.save(save_path)

    assert save_path.exists()
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "models" in data
    assert "cv_metrics" in data
    assert "air_ratio" in data["models"]
    assert "coef" in data["models"]["air_ratio"]

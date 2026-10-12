"""Unit tests for Cross-Continental Slovak Dataset Validation."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline.slovak_validation import (
    SHARED_LATIN_FEATURES,
    SLOVAK_DIR,
    evaluate_zero_shot_and_cv,
)


def test_slovak_directory_exists():
    """Verify that the external Slovak dataset directory exists."""
    assert SLOVAK_DIR.exists(), f"Slovak dataset directory not found at: {SLOVAK_DIR}"
    subdirs = [d.name for d in SLOVAK_DIR.iterdir() if d.is_dir()]
    assert "Low Potential Dysgraphia" in subdirs
    assert "Potential Dysgraphia" in subdirs


def test_slovak_synthetic_evaluation():
    """Verify evaluation logic using synthetic data matching the Slovak format."""
    np.random.seed(42)
    n = 40
    data = {
        "filename": [f"img_{i}.jpg" for i in range(n)],
        "category": ["Low Potential Dysgraphia"] * 20 + ["Potential Dysgraphia"] * 20,
        "label": [0] * 20 + [1] * 20,
        "width": [500] * n,
        "height": [100] * n,
        "h_est": [40.0] * n,
    }

    # Add shared features with realistic difference
    for f in SHARED_LATIN_FEATURES:
        base_ctrl = np.random.normal(1.0, 0.2, 20)
        base_dysg = np.random.normal(1.5, 0.3, 20)
        data[f] = np.concatenate([base_ctrl, base_dysg])

    df_synth = pd.DataFrame(data)

    eval_res = evaluate_zero_shot_and_cv(df_synth)
    assert "zero_shot_transfer" in eval_res
    assert "within_slovak_cv" in eval_res
    assert "univariate_rankings" in eval_res

    for m_name, res in eval_res["within_slovak_cv"].items():
        assert 0.0 <= res["auroc"] <= 1.0
        assert 0.0 <= res["auprc"] <= 1.0

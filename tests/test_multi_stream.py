"""Unit tests for Multi-Stream unified benchmark."""

import numpy as np
import pytest

from models.multi_stream_experiments import (
    STREAM1_FEATURES,
    STREAM2_FEATURES,
    STREAM3_FEATURES,
    compute_metrics,
)


def test_stream1_features_list():
    """Verify Stream 1 features are task-agnostic geometric features."""
    assert len(STREAM1_FEATURES) == 14
    assert "components_per_unit_width_median" in STREAM1_FEATURES
    assert "components_per_unit_width_late_minus_early" in STREAM1_FEATURES
    # Verify no task-specific columns
    for f in STREAM1_FEATURES:
        assert "_dict_" not in f
        assert "_own_" not in f
        assert "words_written" not in f


def test_stream2_features_list():
    """Verify Stream 2 features are writing-tool-calibrated."""
    assert len(STREAM2_FEATURES) == 7
    assert "is_pencil" in STREAM2_FEATURES
    assert "pressure_intensity_cv_tool_norm" in STREAM2_FEATURES
    assert "stroke_width_jitter_tool_norm" in STREAM2_FEATURES
    assert "interaction_pencil_pressure" in STREAM2_FEATURES


def test_stream3_features_list():
    """Verify Stream 3 features are prompt-free TrOCR legibility metrics."""
    assert len(STREAM3_FEATURES) == 5
    assert "trocr_conf_mean" in STREAM3_FEATURES
    assert "trocr_conf_p10" in STREAM3_FEATURES
    assert "trocr_char_entropy_mean" in STREAM3_FEATURES
    assert "trocr_top2_margin_mean" in STREAM3_FEATURES
    assert "ink_to_char_density_ratio" in STREAM3_FEATURES


def test_compute_metrics_calculation():
    y_true = np.array([1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.3])

    met = compute_metrics(y_true, y_prob)
    assert met["auroc"] == 1.0
    assert met["auprc"] == 1.0
    assert 0.0 <= met["brier"] <= 1.0
    assert met["sens_at_80_spec"] == 1.0
    assert met["sens_at_90_spec"] == 1.0


def test_compute_metrics_single_class():
    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3])
    met = compute_metrics(y_true, y_prob)
    assert np.isnan(met["auroc"])
    assert np.isnan(met["auprc"])


def test_meta_learner_combination_bounded():
    p1 = np.array([0.8, 0.4, 0.1])
    p2 = np.array([0.9, 0.3, 0.2])
    p3 = np.array([0.7, 0.5, 0.1])
    # Convex combination test
    w1, w2, w3 = 0.40, 0.35, 0.25
    p_stack = w1 * p1 + w2 * p2 + w3 * p3
    assert np.all(p_stack >= 0.0)
    assert np.all(p_stack <= 1.0)

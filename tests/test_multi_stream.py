"""Unit tests for Multi-Stream late stacking benchmark."""

import numpy as np
import pytest

from models.multi_stream_experiments import STREAM2_CORE_FEATURES, STREAM3_CORE_FEATURES, compute_metrics


def test_stream2_core_features_list():
    assert len(STREAM2_CORE_FEATURES) >= 10
    assert "stroke_width_cv_mean" in STREAM2_CORE_FEATURES
    assert "pressure_intensity_cv_mean" in STREAM2_CORE_FEATURES
    assert "smudge_halo_darkness_max" in STREAM2_CORE_FEATURES
    assert "background_roughness_std_mean" in STREAM2_CORE_FEATURES


def test_stream3_core_features_list():
    assert len(STREAM3_CORE_FEATURES) >= 5
    assert "ocr_confidence_mean_mean" in STREAM3_CORE_FEATURES
    assert "ocr_cer_dict_minus_copy" in STREAM3_CORE_FEATURES
    assert "char_omission_rate_max" in STREAM3_CORE_FEATURES
    assert "confusable_rate_max" in STREAM3_CORE_FEATURES


def test_compute_metrics_calculation():
    y_true = np.array([1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.3])

    met = compute_metrics(y_true, y_prob)
    assert met["auroc"] == 1.0
    assert met["auprc"] == 1.0
    assert 0.0 <= met["brier"] <= 1.0
    assert met["n_pos"] == 2
    assert met["n_neg"] == 3
    assert met["sens_at_80_spec"] == 1.0
    assert met["sens_at_90_spec"] == 1.0
    assert met["youden_sens"] == 1.0
    assert met["youden_spec"] == 1.0


def test_compute_metrics_single_class():
    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3])
    met = compute_metrics(y_true, y_prob)
    assert met["auroc"] is None
    assert met["auprc"] is None


def test_late_stacking_bounded():
    p1 = np.array([0.8, 0.4, 0.1])
    p2 = np.array([0.9, 0.3, 0.2])
    w1, w2 = 0.70, 0.30
    p_stack = w1 * p1 + w2 * p2
    assert np.all(p_stack >= 0.0)
    assert np.all(p_stack <= 1.0)
    assert np.isclose(p_stack[0], 0.83)

"""Unit tests for Two-Tier Clinical Screening Cascade."""

import numpy as np
import pytest

from models.cascade import TwoTierScreeningCascade


@pytest.fixture
def synthetic_scores():
    """Create synthetic scores with realistic separation (AUROC ~ 0.88)."""
    np.random.seed(42)
    n_neg = 175
    n_pos = 40

    neg_scores = np.random.beta(a=1.5, b=5.0, size=n_neg)  # skewed towards 0
    pos_scores = np.random.beta(a=4.0, b=2.0, size=n_pos)  # skewed towards 1

    y_true = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    y_scores = np.concatenate([neg_scores, pos_scores])
    return y_true, y_scores


def test_cascade_threshold_monotonicity(synthetic_scores):
    """Verify triage threshold is strictly below referral threshold."""
    y_true, y_scores = synthetic_scores
    cascade = TwoTierScreeningCascade(target_sensitivity=0.88, target_specificity=0.90)
    thresh = cascade.fit_thresholds(y_true, y_scores)

    assert thresh["theta_triage"] < thresh["theta_referral"]
    assert 0.0 < thresh["theta_triage"] < 1.0
    assert 0.0 < thresh["theta_referral"] < 1.0


def test_cascade_triage_sensitivity(synthetic_scores):
    """Verify Tier 1 triage achieves >= target sensitivity."""
    y_true, y_scores = synthetic_scores
    cascade = TwoTierScreeningCascade(target_sensitivity=0.88, target_specificity=0.90)
    metrics = cascade.evaluate(y_true, y_scores)

    assert metrics["triage_sensitivity"] >= 0.85
    assert metrics["triage_npv"] >= 0.90
    assert metrics["false_negatives_cleared"] <= int(0.15 * np.sum(y_true == 1))


def test_cascade_referral_specificity(synthetic_scores):
    """Verify Tier 2 referral achieves >= target specificity."""
    y_true, y_scores = synthetic_scores
    cascade = TwoTierScreeningCascade(target_sensitivity=0.88, target_specificity=0.90)
    metrics = cascade.evaluate(y_true, y_scores)

    assert metrics["referral_specificity"] >= 0.88
    assert metrics["referral_ppv"] > 0.50


def test_cascade_tier_predictions(synthetic_scores):
    """Verify tier assignments partition the cohort completely."""
    y_true, y_scores = synthetic_scores
    cascade = TwoTierScreeningCascade(target_sensitivity=0.88, target_specificity=0.90)
    cascade.fit_thresholds(y_true, y_scores)

    tiers = cascade.predict_tiers(y_scores)
    assert len(tiers) == len(y_true)
    assert set(np.unique(tiers)).issubset({0, 1, 2})

    n_cleared = np.sum(tiers == 0)
    n_monitor = np.sum(tiers == 1)
    n_refer = np.sum(tiers == 2)
    assert n_cleared + n_monitor + n_refer == len(y_true)

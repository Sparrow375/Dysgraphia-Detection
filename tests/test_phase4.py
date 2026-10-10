"""Automated unit and integration tests for Phase 4 Modeling & Evaluation.
Validates zero student leakage across outer and inner CV folds, sample weights,
leakage-free imputation, Platt calibration, student pooling, and cluster bootstrap CIs.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from models.candidates import MajorityClassifier, get_candidate_registry
from models.cv import (
    compute_sample_weights,
    get_feature_columns,
    get_inner_cv_splits,
    get_outer_fold_split,
    impute_fold_features,
    load_dataset,
)
from models.evaluator import (
    aggregate_student_predictions,
    calculate_sensitivity_at_specificity,
    cluster_bootstrap_ci,
    compute_student_metrics,
    find_operational_threshold,
    fit_platt_calibrator,
)
from models.experiments import run_nested_cv_for_model


def test_cv_zero_student_leakage():
    """Verify that outer and inner cross-validation splits have strictly zero student leakage."""
    df = load_dataset("combined")

    # Check all 5 repeats x 5 folds
    for r in range(5):
        for k in range(5):
            train_df, test_df = get_outer_fold_split(df, repeat_idx=r, fold_idx=k)
            train_students = set(train_df["student_id"])
            test_students = set(test_df["student_id"])

            overlap = train_students.intersection(test_students)
            assert len(overlap) == 0, f"Outer CV leakage in repeat {r} fold {k}: {overlap}"

            # Check inner splits on this fold's training set
            inner_splits = get_inner_cv_splits(train_df, n_splits=3, random_state=42)
            assert len(inner_splits) == 3
            for in_tr_idx, in_val_idx in inner_splits:
                in_tr_s = set(train_df.iloc[in_tr_idx]["student_id"])
                in_val_s = set(train_df.iloc[in_val_idx]["student_id"])
                assert len(in_tr_s.intersection(in_val_s)) == 0, "Inner CV leakage detected!"


def test_sample_weights_properties():
    """Verify sample weights balance student contribution and normalize to mean 1.0."""
    df = load_dataset("combined")
    weights = compute_sample_weights(df, balance_classes=True)

    assert len(weights) == len(df)
    assert np.all(weights > 0.0), "All sample weights must be strictly positive"
    assert np.isclose(np.mean(weights), 1.0, atol=1e-5), f"Mean weight should be 1.0, got {np.mean(weights)}"

    # Verify student with 6 sentences has lower per-sentence weight than student with 4 sentences in same class
    student_counts = df["student_id"].value_counts()
    sid_4 = student_counts[student_counts == 4].index[0]
    sid_6 = student_counts[student_counts == 6].index[0]

    # Only compare if they have the same class label
    lbl_4 = df[df["student_id"] == sid_4]["label"].iloc[0]
    lbl_6 = df[df["student_id"] == sid_6]["label"].iloc[0]

    if lbl_4 == lbl_6:
        w_4 = weights[df["student_id"] == sid_4][0]
        w_6 = weights[df["student_id"] == sid_6][0]
        assert w_4 > w_6, "Students with fewer sentences should have higher individual sentence weight"


def test_impute_fold_features_leakage_free():
    """Verify test NaNs are imputed strictly using training medians without leakage."""
    train_data = {
        "feat_a": [10.0, 20.0, 30.0, np.nan],  # median should be 20.0
        "feat_b": [1.0, 2.0, np.nan, 4.0],     # median should be 2.0
    }
    test_data = {
        "feat_a": [np.nan, 100.0],
        "feat_b": [5.0, np.nan],
    }
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    feature_cols = ["feat_a", "feat_b"]
    X_train, X_test, medians = impute_fold_features(train_df, test_df, feature_cols)

    assert medians["feat_a"] == 20.0
    assert medians["feat_b"] == 2.0
    assert X_test[0, 0] == 20.0  # Imputed using train median, not test!
    assert X_test[1, 1] == 2.0
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_platt_calibration_and_student_pooling():
    """Verify Platt calibration produces probabilities in [0, 1] and aggregates by student."""
    X_train = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    w_train = np.ones(6)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    calibrated = fit_platt_calibrator(clf, X_train, y_train, w_train)
    X_test = np.array([[1.5], [8.5]])
    probs = calibrated.predict_proba(X_test)[:, 1]

    assert len(probs) == 2
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0
    assert probs[0] < probs[1]

    # Test student aggregation
    test_df = pd.DataFrame({
        "student_id": ["S1", "S1", "S2"],
        "label": [0, 0, 1],
    })
    sentence_probs = np.array([0.2, 0.4, 0.85])
    student_df = aggregate_student_predictions(test_df, sentence_probs)

    assert len(student_df) == 2
    assert np.isclose(student_df.loc[student_df["student_id"] == "S1", "pred_prob"].iloc[0], 0.3)
    assert np.isclose(student_df.loc[student_df["student_id"] == "S2", "pred_prob"].iloc[0], 0.85)


def test_student_metrics_and_cluster_bootstrap():
    """Verify metrics calculation and cluster bootstrap confidence interval bounds."""
    student_df = pd.DataFrame({
        "student_id": [f"S{i:02d}" for i in range(20)],
        "label": [0] * 15 + [1] * 5,
        "pred_prob": [0.1] * 12 + [0.7] * 3 + [0.8] * 4 + [0.2],
    })

    metrics = compute_student_metrics(student_df, threshold=0.5)
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["auprc"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0
    assert metrics["n_students"] == 20
    assert metrics["n_positives"] == 5

    # Test cluster bootstrap CIs
    ci_results = cluster_bootstrap_ci(student_df, threshold=0.5, n_bootstraps=100, random_state=42)
    for m in ["auroc", "auprc", "brier", "sensitivity", "specificity"]:
        assert m in ci_results
        assert ci_results[m]["ci_low"] <= ci_results[m]["mean"] <= ci_results[m]["ci_high"] + 1e-6


def test_smoke_nested_cv_single_repeat():
    """Fast smoke test of run_nested_cv_for_model on Hindi dataset (1 repeat, 2 models)."""
    df = load_dataset("hindi")

    # Smoke test Majority baseline
    maj_res = run_nested_cv_for_model(df, model_key="majority", n_repeats=1, n_outer_folds=2, n_inner_splits=2)
    assert maj_res["model_key"] == "majority"
    assert "point_metrics" in maj_res
    assert len(maj_res["student_predictions"]) > 0

    # Smoke test GradeOnly
    grade_res = run_nested_cv_for_model(df, model_key="grade_only", n_repeats=1, n_outer_folds=2, n_inner_splits=2)
    assert grade_res["model_key"] == "grade_only"
    assert grade_res["point_metrics"]["auroc"] >= 0.0

"""Unit tests for Task-Aware Dynamic Routing Architecture."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from models.task_router import (
    COPY_ONLY_FEATURES,
    FULL_PROTOCOL_FEATURES,
    NO_DICTATION_FEATURES,
    TaskAwareRouter,
    TaskRoute,
    TaskSubModel,
    detect_route_from_features,
    detect_route_from_tasks,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
STUDENT_DATASET_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level.csv"


def test_detect_route_from_tasks():
    """Verify route detection from task string lists."""
    full_tasks = ["sentence_01_copy_hindi", "sentence_02_copy_english", "sentence_03_dictated_hindi"]
    assert detect_route_from_tasks(full_tasks) == TaskRoute.FULL_PROTOCOL

    nodict_tasks = ["sentence_01_copy_hindi", "sentence_02_copy_english", "sentence_05_own_english"]
    assert detect_route_from_tasks(nodict_tasks) == TaskRoute.NO_DICTATION

    copy_only_tasks = ["sentence_01_copy_hindi", "sentence_02_copy_english"]
    assert detect_route_from_tasks(copy_only_tasks) == TaskRoute.COPY_ONLY


def test_detect_route_from_features():
    """Verify route detection from feature values."""
    # Case 1: Dictation present
    rec_full = {"components_per_unit_width_dict_mean": 4.5, "junctions_per_unit_width_own_mean": 2.1}
    assert detect_route_from_features(rec_full) == TaskRoute.FULL_PROTOCOL

    # Case 2: Dictation absent, Free-Write present
    rec_nodict = {"components_per_unit_width_dict_mean": np.nan, "junctions_per_unit_width_own_mean": 2.1}
    assert detect_route_from_features(rec_nodict) == TaskRoute.NO_DICTATION

    # Case 3: Both absent (Copy-only)
    rec_copy = {"components_per_unit_width_dict_mean": np.nan, "junctions_per_unit_width_own_mean": np.nan}
    assert detect_route_from_features(rec_copy) == TaskRoute.COPY_ONLY


def test_task_submodel_fit_predict():
    """Verify TaskSubModel fits and predicts calibrated probabilities without leakage."""
    rng = np.random.RandomState(42)
    feats = ["feat_a", "feat_b", "feat_c"]
    X_train = pd.DataFrame(rng.randn(50, 3), columns=feats)
    X_train.iloc[0:5, 0] = np.nan  # introduce missing values
    y_train = rng.randint(0, 2, size=50)

    sub_model = TaskSubModel("test_submodel", feats, model_type="rf", random_state=42)
    sub_model.fit(X_train, y_train)

    assert "feat_a" in sub_model.medians
    assert not np.isnan(sub_model.medians["feat_a"])

    X_test = pd.DataFrame(rng.randn(10, 3), columns=feats)
    X_test.iloc[2, 1] = np.nan
    probs = sub_model.predict_proba(X_test)

    assert len(probs) == 10
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert not np.any(np.isnan(probs))


@pytest.mark.skipif(not STUDENT_DATASET_PATH.exists(), reason="Student dataset not found")
def test_task_aware_router_real_data():
    """Verify TaskAwareRouter on real cohort dataset."""
    df = pd.read_csv(STUDENT_DATASET_PATH)
    df_a = df[df["school"] == "school_a"]
    df_b = df[df["school"] == "school_b"]

    router = TaskAwareRouter(random_state=42)
    router.fit(df_a)

    assert router.is_fitted
    assert router.sub_models[TaskRoute.FULL_PROTOCOL].clf is not None
    assert router.sub_models[TaskRoute.NO_DICTATION].clf is not None

    # Predict batch on School B
    preds = router.predict_proba(df_b)
    assert len(preds) == len(df_b)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    assert not np.any(np.isnan(preds))

    # Evaluate evaluation dictionary
    eval_metrics = router.evaluate(df_b)
    assert eval_metrics["n_students"] == 100
    assert eval_metrics["auroc"] > 0.70
    assert "route_breakdown" in eval_metrics
    assert "grade_breakdown" in eval_metrics
    assert TaskRoute.FULL_PROTOCOL.value in eval_metrics["route_breakdown"]
    assert TaskRoute.NO_DICTATION.value in eval_metrics["route_breakdown"]


@pytest.mark.skipif(not STUDENT_DATASET_PATH.exists(), reason="Student dataset not found")
def test_task_aware_router_single_student():
    """Verify predict_student metadata and tier classification."""
    df = pd.read_csv(STUDENT_DATASET_PATH)
    router = TaskAwareRouter(random_state=42)
    df_a = df[df["school"] == "school_a"]
    router.fit(df_a)

    sample_student = df.iloc[0]
    pred_res = router.predict_student(sample_student)

    assert "risk_score" in pred_res
    assert "predicted_label" in pred_res
    assert "route_assigned" in pred_res
    assert "tier" in pred_res
    assert 0.0 <= pred_res["risk_score"] <= 1.0
    assert pred_res["predicted_label"] in [0, 1]
    assert pred_res["route_assigned"] in [r.value for r in TaskRoute]

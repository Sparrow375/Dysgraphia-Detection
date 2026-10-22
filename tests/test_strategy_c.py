"""Unit and Regression Tests for Strategy C: Self-Supervised DINOv2 Vision Representation & Probe."""

import csv
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from models.strategy_c_dinov2 import (
    DINOv2CropEmbedding,
    aggregate_student_dinov2_features,
    run_dinov2_cross_validation,
    evaluate_hybrid_foundation_ensemble,
    INDEX_CSV,
)


def test_sentences_index_cohort_integrity():
    """Verify that sentences_index.csv contains all 215 students and locked folds."""
    assert INDEX_CSV.exists(), f"Missing index file: {INDEX_CSV}"
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 1100, f"Expected ~1,145 sentence crops, found {len(reader)}"
    students = set(r["student_id"] for r in reader)
    assert len(students) == 215, f"Expected exactly 215 unique students, found {len(students)}"

    # Verify locked 5x5 fold assignments are present and valid
    for r in reader:
        for rep in range(5):
            fold_col = f"repeat_{rep}_fold"
            assert fold_col in r, f"Missing fold column: {fold_col}"
            f_val = int(r[fold_col])
            assert 0 <= f_val <= 4, f"Invalid fold value: {f_val} in {fold_col}"


def test_aggregate_student_dinov2_features_synthetic():
    """Verify student aggregation, stress deltas, and metadata mapping with synthetic embeddings."""
    dim = 16  # Test with compact embedding dimension
    mock_crops = [
        DINOv2CropEmbedding(
            task_uid="S1_copy_hin",
            student_id="STU_001",
            school="school_a",
            grade=4,
            label=1,
            task_id="sentence_01_copy_hindi",
            script="devanagari",
            embedding=np.ones(dim, dtype=np.float32) * 1.0,
        ),
        DINOv2CropEmbedding(
            task_uid="S1_dict_hin",
            student_id="STU_001",
            school="school_a",
            grade=4,
            label=1,
            task_id="sentence_03_dictated_hindi",
            script="devanagari",
            embedding=np.ones(dim, dtype=np.float32) * 3.0,
        ),
        DINOv2CropEmbedding(
            task_uid="S2_copy_hin",
            student_id="STU_002",
            school="school_b",
            grade=5,
            label=0,
            task_id="sentence_01_copy_hindi",
            script="devanagari",
            embedding=np.ones(dim, dtype=np.float32) * 0.5,
        ),
    ]

    mock_index = [
        {
            "student_id": "STU_001",
            "school": "school_a",
            "grade": "4",
            "label": "1",
            "repeat_0_fold": "0",
            "repeat_1_fold": "1",
            "repeat_2_fold": "2",
            "repeat_3_fold": "3",
            "repeat_4_fold": "4",
        },
        {
            "student_id": "STU_002",
            "school": "school_b",
            "grade": "5",
            "label": "0",
            "repeat_0_fold": "1",
            "repeat_1_fold": "2",
            "repeat_2_fold": "3",
            "repeat_3_fold": "4",
            "repeat_4_fold": "0",
        },
    ]

    df_students = aggregate_student_dinov2_features(mock_crops, mock_index)

    assert len(df_students) == 2
    assert "student_id" in df_students.columns
    assert "stress_delta_norm" in df_students.columns
    assert "stress_cosine_dist" in df_students.columns

    stu1 = df_students[df_students["student_id"] == "STU_001"].iloc[0]
    assert stu1["label"] == 1
    assert stu1["grade"] == 4
    assert stu1["num_crops"] == 2
    # Mean of 1.0 and 3.0 is 2.0
    assert stu1["dinov2_mean_0"] == pytest.approx(2.0)
    # Norm of (3.0 - 1.0) * sqrt(16) = 2.0 * 4 = 8.0
    assert stu1["stress_delta_norm"] == pytest.approx(8.0)


def test_run_dinov2_cross_validation_synthetic():
    """Verify locked 5x5 cross-validation loop and metric outputs on synthetic cohort."""
    np.random.seed(42)
    n_students = 50
    dim = 8

    rows = []
    for i in range(n_students):
        label = 1 if i < 10 else 0
        row = {
            "student_id": f"STU_{i:03d}",
            "school": "school_a" if i % 2 == 0 else "school_b",
            "grade": 3 + (i % 6),
            "label": label,
            "stress_delta_norm": float(np.random.randn() + (2.0 if label == 1 else 0.0)),
            "stress_cosine_dist": float(np.random.rand() * 0.5),
            "max_deviation_from_mean": float(np.random.rand()),
        }
        # Add locked fold assignments
        for rep in range(5):
            row[f"repeat_{rep}_fold"] = i % 5

        # Feature embeddings with signal for label 1
        signal = 1.5 if label == 1 else 0.0
        for d in range(dim):
            row[f"dinov2_mean_{d}"] = float(np.random.randn() + signal)
            row[f"dinov2_std_{d}"] = float(np.abs(np.random.randn()))

        rows.append(row)

    df_students = pd.DataFrame(rows)

    res = run_dinov2_cross_validation(df_students, pca_components=4, C=0.1)

    assert "mean_auroc" in res
    assert "mean_auprc" in res
    assert "mean_brier" in res
    assert "sens_at_80_spec" in res
    assert "sens_at_90_spec" in res
    assert len(res["repeat_aurocs"]) == 5

    # With injected signal, AUROC should be > 0.60
    assert res["mean_auroc"] > 0.60
    assert 0.0 <= res["sens_at_80_spec"] <= 1.0
    assert 0.0 <= res["sens_at_90_spec"] <= 1.0


def test_evaluate_hybrid_foundation_ensemble():
    """Verify hybrid foundation ensemble combining DINOv2 and TabPFN."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_dinov2 = np.array([0.2, 0.3, 0.4, 0.1, 0.6, 0.7, 0.5, 0.8])
    p_tabpfn = np.array([0.1, 0.2, 0.3, 0.2, 0.8, 0.9, 0.7, 0.9])

    ensemble_res = evaluate_hybrid_foundation_ensemble(p_dinov2, p_tabpfn, y_true)

    assert "best_hybrid" in ensemble_res
    assert "all_combinations" in ensemble_res
    best = ensemble_res["best_hybrid"]
    assert "auroc" in best
    assert "alpha_dinov2" in best
    assert "alpha_tabpfn" in best
    assert best["auroc"] >= 0.90

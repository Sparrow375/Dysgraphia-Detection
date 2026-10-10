"""Automated Unit Tests for Workstream A Phase 3 (Dataset Assembly).

Verifies:
1. Dataset table existence, row counts (322 Hindi, 319 English, 641 combined), and column schemas.
2. One-hot task encoding and fold column mapping integrity.
3. Robust z-score clipping within [-5.0, 5.0].
4. Leakage-free fold imputation logic.
5. Zero student leakage across CV fold mappings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.dataset_assembly import impute_fold_features


@pytest.fixture(scope="module")
def dataset_tables():
    datasets_dir = Path("data/datasets")
    hindi_path = datasets_dir / "dataset_hindi.csv"
    english_path = datasets_dir / "dataset_english.csv"
    combined_path = datasets_dir / "dataset_combined.csv"

    assert hindi_path.exists(), f"Hindi dataset not found at {hindi_path}"
    assert english_path.exists(), f"English dataset not found at {english_path}"
    assert combined_path.exists(), f"Combined dataset not found at {combined_path}"

    df_h = pd.read_csv(hindi_path)
    df_e = pd.read_csv(english_path)
    df_c = pd.read_csv(combined_path)

    return df_h, df_e, df_c


def test_dataset_row_counts_and_scripts(dataset_tables):
    df_h, df_e, df_c = dataset_tables

    # School A subset checks
    assert len(df_h[df_h["school"] == "school_a"]) == 322, f"Expected 322 School A Hindi rows"
    assert len(df_e[df_e["school"] == "school_a"]) == 319, f"Expected 319 School A English rows"
    assert len(df_c[df_c["school"] == "school_a"]) == 641, f"Expected 641 School A Combined rows"

    # Multi-school totals
    assert len(df_h) == 570, f"Expected 570 total Hindi rows, got {len(df_h)}"
    assert len(df_e) == 575, f"Expected 575 total English rows, got {len(df_e)}"
    assert len(df_c) == 1145, f"Expected 1145 total Combined rows, got {len(df_c)}"

    # All rows in Hindi must be devanagari
    assert (df_h["script"] == "devanagari").all()
    # All rows in English must be latin
    assert (df_e["script"] == "latin").all()


def test_one_hot_task_encoding_integrity(dataset_tables):
    df_h, df_e, df_c = dataset_tables

    for df, name in [(df_h, "Hindi"), (df_e, "English"), (df_c, "Combined")]:
        assert "task_copy" in df.columns, f"Missing task_copy in {name}"
        assert "task_dictated" in df.columns, f"Missing task_dictated in {name}"
        assert "task_own" in df.columns, f"Missing task_own in {name}"

        # Every row must belong to exactly one task group
        task_sum = df["task_copy"] + df["task_dictated"] + df["task_own"]
        assert (task_sum == 1).all(), f"Row in {name} does not have exactly one task type"


def test_fold_columns_and_zero_leakage(dataset_tables):
    _, _, df_c = dataset_tables

    fold_cols = [c for c in df_c.columns if c.startswith("repeat_")]
    assert len(fold_cols) == 5, f"Expected 5 repeat fold columns, got {len(fold_cols)}"

    # Check zero student leakage across outer folds
    for rep_col in fold_cols:
        for f_id in range(5):
            test_students = set(df_c[df_c[rep_col] == f_id]["student_id"])
            train_students = set(df_c[df_c[rep_col] != f_id]["student_id"])
            overlap = test_students.intersection(train_students)
            assert len(overlap) == 0, f"Student leakage in {rep_col} fold {f_id}: {overlap}"


def test_robust_z_scores_clipping(dataset_tables):
    _, _, df_c = dataset_tables

    z_cols = [c for c in df_c.columns if c.startswith("z_")]
    assert len(z_cols) >= 20, f"Expected at least 20 z-feature columns, got {len(z_cols)}"

    for z_c in z_cols:
        valid_vals = df_c[z_c].dropna()
        if len(valid_vals) > 0:
            assert valid_vals.min() >= -5.0001, f"{z_c} min {valid_vals.min()} < -5.0"
            assert valid_vals.max() <= 5.0001, f"{z_c} max {valid_vals.max()} > 5.0"


def test_script_specific_feature_isolation(dataset_tables):
    df_h, df_e, _ = dataset_tables

    # Hindi table must not contain Latin-specific ascender_descender_ratio
    assert "raw_ascender_descender_ratio" not in df_h.columns
    assert "z_raw_ascender_descender_ratio" not in df_h.columns

    # English table must not contain Hindi-specific features
    assert "raw_matra_ratio" not in df_e.columns
    assert "raw_shirorekha_rms_deviation_norm" not in df_e.columns
    assert "z_raw_matra_ratio" not in df_e.columns


def test_leakage_free_imputation_helper():
    # Synthetic test to verify fold imputation helper
    train_data = pd.DataFrame({
        "student_id": ["S1", "S2", "S3", "S4"],
        "raw_feat1": [10.0, 20.0, np.nan, 40.0],  # Median = 20.0
        "z_feat2": [1.0, np.nan, 3.0, 5.0],      # Median = 3.0
    })
    test_data = pd.DataFrame({
        "student_id": ["S5", "S6"],
        "raw_feat1": [np.nan, 100.0],  # Should be imputed with 20.0 (train median), not 100.0!
        "z_feat2": [-1.0, np.nan],     # Should be imputed with 3.0 (train median)
    })

    train_imp, test_imp = impute_fold_features(
        train_df=train_data,
        test_df=test_data,
        feature_cols=["raw_feat1", "z_feat2"],
    )

    # Check train imputation
    assert train_imp["raw_feat1"].isna().sum() == 0
    assert train_imp.loc[2, "raw_feat1"] == 20.0
    assert train_imp.loc[1, "z_feat2"] == 3.0

    # Check test imputation uses strictly train medians
    assert test_imp["raw_feat1"].isna().sum() == 0
    assert test_imp.loc[0, "raw_feat1"] == 20.0  # Uses train median 20.0
    assert test_imp.loc[1, "z_feat2"] == 3.0    # Uses train median 3.0

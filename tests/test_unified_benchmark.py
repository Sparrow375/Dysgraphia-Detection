"""Unit tests for the Unified 215-Student Benchmark (Stage 25).

Verifies:
1. Exact cohort size: 215 students (115 School A + 100 School B; 40 positives, 175 controls).
2. Perfectly balanced 5-repeat x 5-fold partition:
   - Exactly 43 students per test fold (215 / 5 = 43).
   - Exactly 8 positive students per test fold (40 / 5 = 8).
   - Exactly 35 negative controls per test fold (175 / 5 = 35).
   - Exactly 23 School A students per test fold (115 / 5 = 23).
   - Exactly 20 School B students per test fold (100 / 5 = 20).
3. Zero student leakage across all outer test folds.
4. Zero student leakage in nested inner 3-fold validation splits.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_cohort_integrity():
    """Verify master manifest contains exactly 215 students with 40 positives and 175 controls."""
    manifest_path = WORKSPACE_ROOT / "data" / "manifest.csv"
    assert manifest_path.exists(), "data/manifest.csv missing!"
    df = pd.read_csv(manifest_path)

    assert len(df) == 215, f"Expected 215 rows in manifest, found {len(df)}"
    assert df["student_id"].nunique() == 215, "Non-unique student IDs found in manifest!"

    # School counts
    school_counts = df["school"].value_counts().to_dict()
    assert school_counts.get("school_a") == 115, f"Expected 115 School A students, got {school_counts.get('school_a')}"
    assert school_counts.get("school_b") == 100, f"Expected 100 School B students, got {school_counts.get('school_b')}"

    # Label counts
    pos = int((df["label"] == 1.0).sum())
    neg = int((df["label"] == 0.0).sum())
    assert pos == 40, f"Expected 40 positives, got {pos}"
    assert neg == 175, f"Expected 175 controls, got {neg}"


def test_nested_folds_balance_and_leakage():
    """Verify nested_folds.json enforces exact balance and zero leakage across all repeats and folds."""
    folds_json_path = WORKSPACE_ROOT / "data" / "folds" / "nested_folds.json"
    lookup_csv_path = WORKSPACE_ROOT / "data" / "folds" / "student_folds_lookup.csv"

    assert folds_json_path.exists(), "data/folds/nested_folds.json missing!"
    assert lookup_csv_path.exists(), "data/folds/student_folds_lookup.csv missing!"

    with open(folds_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["n_students"] == 215
    assert meta["n_positives"] == 40
    assert meta["n_negatives"] == 175
    assert meta["n_school_a"] == 115
    assert meta["n_school_b"] == 100

    lookup_df = pd.read_csv(lookup_csv_path)
    assert len(lookup_df) == 215

    for rep in data["repeats"]:
        rep_id = rep["repeat_id"]
        col_name = f"repeat_{rep_id}_fold"
        assert col_name in lookup_df.columns

        all_test_students = []
        for fold in rep["outer_folds"]:
            fold_id = fold["outer_fold_id"]
            test_students = fold["test_students"]
            train_students = fold["train_students"]

            # 1. Size checks
            assert len(test_students) == 43, f"Repeat {rep_id} Fold {fold_id}: expected 43 test, got {len(test_students)}"
            assert len(train_students) == 172, f"Repeat {rep_id} Fold {fold_id}: expected 172 train, got {len(train_students)}"
            assert fold["test_positives"] == 8, f"Repeat {rep_id} Fold {fold_id}: expected 8 pos, got {fold['test_positives']}"
            assert fold["test_school_a"] == 23, f"Repeat {rep_id} Fold {fold_id}: expected 23 School A, got {fold['test_school_a']}"
            assert fold["test_school_b"] == 20, f"Repeat {rep_id} Fold {fold_id}: expected 20 School B, got {fold['test_school_b']}"

            # 2. Outer zero-leakage check
            overlap = set(test_students).intersection(set(train_students))
            assert len(overlap) == 0, f"Leakage in Repeat {rep_id} Fold {fold_id}: {overlap}"
            all_test_students.extend(test_students)

            # 3. Inner zero-leakage checks
            for inner in fold["inner_folds"]:
                in_tr = inner["train_students"]
                in_val = inner["val_students"]
                in_overlap = set(in_tr).intersection(set(in_val))
                assert len(in_overlap) == 0, f"Inner leakage in Repeat {rep_id} Fold {fold_id} Inner {inner['inner_fold_id']}"
                assert set(in_tr).issubset(set(train_students))
                assert set(in_val).issubset(set(train_students))

        # Check all 215 students were covered once as test
        assert len(all_test_students) == 215
        assert len(set(all_test_students)) == 215

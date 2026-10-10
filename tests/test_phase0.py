"""Unit tests for Workstream A Phase 0 (Manifest and Nested CV Folds)."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest


def test_manifest_existence_and_schema():
    manifest_path = Path("data/manifest.csv")
    assert manifest_path.exists(), "data/manifest.csv does not exist"

    df = pd.read_csv(manifest_path)
    expected_cols = [
        "student_id",
        "school",
        "school_name",
        "grade",
        "section",
        "roll_number",
        "label",
        "label_source",
        "match_status",
        "image_filename",
        "relative_image_path",
        "is_held_out",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    assert len(df) == 215, f"Expected 215 total records, got {len(df)}"


def test_school_a_manifest_integrity():
    df = pd.read_csv("data/manifest.csv")
    df_a = df[df["school"] == "school_a"]

    # Verify total student count
    assert len(df_a) == 115, f"Expected 115 School A students, got {len(df_a)}"
    assert df_a["student_id"].nunique() == 115, "Student IDs must be unique"

    # Verify label counts
    n_pos = (df_a["label"] == 1).sum()
    n_neg = (df_a["label"] == 0).sum()
    assert n_pos == 24, f"Expected 24 positives, got {n_pos}"
    assert n_neg == 91, f"Expected 91 negatives, got {n_neg}"

    # Verify grade breakdown
    grade_counts = df_a["grade"].value_counts().to_dict()
    assert grade_counts[3] == 21, f"Expected 21 Grade 3 students, got {grade_counts.get(3)}"
    assert grade_counts[4] == 24, f"Expected 24 Grade 4 students, got {grade_counts.get(4)}"
    assert grade_counts[5] == 16, f"Expected 16 Grade 5 students, got {grade_counts.get(5)}"
    assert grade_counts[6] == 29, f"Expected 29 Grade 6 students, got {grade_counts.get(6)}"
    assert grade_counts[7] == 25, f"Expected 25 Grade 7 students, got {grade_counts.get(7)}"

    # Verify all image paths exist
    for _, row in df_a.iterrows():
        p = Path(row["relative_image_path"])
        assert p.exists(), f"Image path does not exist on disk: {p} (student: {row['student_id']})"


def test_school_b_manifest_integrity():
    df = pd.read_csv("data/manifest.csv")
    df_b = df[df["school"] == "school_b"]

    assert len(df_b) == 100, f"Expected 100 School B images, got {len(df_b)}"
    assert df_b["is_held_out"].all(), "All School B images must be flagged as held-out"
    assert (df_b["label"] == 1).sum() == 16, "Expected 16 School B dysgraphic positives"
    assert (df_b["label"] == 0).sum() == 84, "Expected 84 School B normal controls"

    for _, row in df_b.iterrows():
        p = Path(row["relative_image_path"])
        assert p.exists(), f"School B image does not exist: {p}"


def test_nested_cv_folds_no_student_leakage():
    folds_json = Path("data/folds/nested_folds.json")
    assert folds_json.exists(), "data/folds/nested_folds.json does not exist"

    with open(folds_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df_manifest = pd.read_csv("data/manifest.csv")
    all_students_a = set(df_manifest[df_manifest["school"] == "school_a"]["student_id"])

    repeats = data["repeats"]
    assert len(repeats) == 5, f"Expected 5 repeats, got {len(repeats)}"

    for r_idx, rep in enumerate(repeats):
        outer_folds = rep["outer_folds"]
        assert len(outer_folds) == 5, f"Expected 5 outer folds per repeat, got {len(outer_folds)}"

        tested_in_repeat = []

        for f_idx, fold in enumerate(outer_folds):
            train_set = set(fold["train_students"])
            test_set = set(fold["test_students"])

            # 1. Zero leakage between train and test in this fold
            overlap = train_set & test_set
            assert len(overlap) == 0, (
                f"Repeat {r_idx} Fold {f_idx}: Student leakage detected! Overlapping students: {overlap}"
            )

            # 2. Union equals full School A student set
            assert (train_set | test_set) == all_students_a, (
                f"Repeat {r_idx} Fold {f_idx}: Union does not match full School A student cohort"
            )

            # 3. Outer fold sizes
            assert len(test_set) == 23, f"Expected test set size 23, got {len(test_set)}"
            assert len(train_set) == 92, f"Expected train set size 92, got {len(train_set)}"

            # 4. Stratification: test must have positives
            assert fold["test_positives"] in [4, 5], f"Unexpected test positive count: {fold['test_positives']}"

            tested_in_repeat.extend(fold["test_students"])

            # 5. Inner folds test
            inner_folds = fold["inner_folds"]
            assert len(inner_folds) == 3, f"Expected 3 inner folds, got {len(inner_folds)}"
            for in_idx, in_fold in enumerate(inner_folds):
                in_train = set(in_fold["train_students"])
                in_val = set(in_fold["val_students"])

                # No leakage in inner fold
                assert len(in_train & in_val) == 0, (
                    f"Repeat {r_idx} Fold {f_idx} Inner {in_idx}: Leakage between inner train and val!"
                )
                assert (in_train | in_val) == train_set, (
                    f"Repeat {r_idx} Fold {f_idx} Inner {in_idx}: Inner union must equal outer train set"
                )

        # In each repeat, each student must be tested exactly once
        assert sorted(tested_in_repeat) == sorted(list(all_students_a)), (
            f"Repeat {r_idx}: Students tested do not form a complete partition of School A"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])

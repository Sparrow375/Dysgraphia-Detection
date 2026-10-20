"""Cross-validation split generation for Workstream A.

Generates nested CV folds on School A (Future Gen) only:
- Outer: Stratified Group 5-Fold x 5 Repeats
- Inner: Stratified Group 3-Fold (nested inside each outer training set)
Grouped strictly by student_id to prevent any student-level leakage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _generate_balanced_split(df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Generate a single 5-fold split across 215 students ensuring exact balance:
    - Exactly 8 positives per fold (40 / 5)
    - Exactly 35 controls per fold (175 / 5)
    - Exactly 23 School A per fold (115 / 5)
    - Exactly 20 School B per fold (100 / 5)
    - Total: exactly 43 students per fold (215 / 5)
    - Balanced grade distribution across folds.
    """
    rng = np.random.RandomState(seed)
    folds = np.zeros(len(df), dtype=int) - 1

    fold_pos4 = int(rng.choice(5))

    # 1. School A Positives (24 students) -> 4 folds get 5, fold_pos4 gets 4
    a_pos = df[(df["school"] == "school_a") & (df["label"] == 1.0)].copy()
    a_pos["jitter"] = rng.rand(len(a_pos))
    a_pos = a_pos.sort_values(["grade", "jitter"]).reset_index()
    target_a_pos = {i: 5 for i in range(5)}
    target_a_pos[fold_pos4] = 4
    curr_counts_ap = {i: 0 for i in range(5)}
    for idx, row in a_pos.iterrows():
        valid_folds = [f for f in range(5) if curr_counts_ap[f] < target_a_pos[f]]
        chosen = valid_folds[idx % len(valid_folds)]
        curr_counts_ap[chosen] += 1
        folds[row["index"]] = chosen

    # 2. School B Positives (16 students) -> 4 folds get 3, fold_pos4 gets 4 (total pos per fold == 8)
    b_pos = df[(df["school"] == "school_b") & (df["label"] == 1.0)].copy()
    b_pos["jitter"] = rng.rand(len(b_pos))
    b_pos = b_pos.sort_values(["grade", "jitter"]).reset_index()
    target_b_pos = {i: 3 for i in range(5)}
    target_b_pos[fold_pos4] = 4
    curr_counts_bp = {i: 0 for i in range(5)}
    for idx, row in b_pos.iterrows():
        valid_folds = [f for f in range(5) if curr_counts_bp[f] < target_b_pos[f]]
        chosen = valid_folds[idx % len(valid_folds)]
        curr_counts_bp[chosen] += 1
        folds[row["index"]] = chosen

    # 3. School A Negatives (91 students) -> 4 folds get 18, fold_pos4 gets 19 (total school_a == 23)
    a_neg = df[(df["school"] == "school_a") & (df["label"] == 0.0)].copy()
    a_neg["jitter"] = rng.rand(len(a_neg))
    a_neg = a_neg.sort_values(["grade", "jitter"]).reset_index()
    target_a_neg = {i: 18 for i in range(5)}
    target_a_neg[fold_pos4] = 19
    curr_counts_an = {i: 0 for i in range(5)}
    for idx, row in a_neg.iterrows():
        valid_folds = [f for f in range(5) if curr_counts_an[f] < target_a_neg[f]]
        chosen = valid_folds[idx % len(valid_folds)]
        curr_counts_an[chosen] += 1
        folds[row["index"]] = chosen

    # 4. School B Negatives (84 students) -> 4 folds get 17, fold_pos4 gets 16 (total school_b == 20, total neg == 35)
    b_neg = df[(df["school"] == "school_b") & (df["label"] == 0.0)].copy()
    b_neg["jitter"] = rng.rand(len(b_neg))
    b_neg = b_neg.sort_values(["grade", "jitter"]).reset_index()
    target_b_neg = {i: 17 for i in range(5)}
    target_b_neg[fold_pos4] = 16
    curr_counts_bn = {i: 0 for i in range(5)}
    for idx, row in b_neg.iterrows():
        valid_folds = [f for f in range(5) if curr_counts_bn[f] < target_b_neg[f]]
        chosen = valid_folds[idx % len(valid_folds)]
        curr_counts_bn[chosen] += 1
        folds[row["index"]] = chosen

    assert (folds >= 0).all(), "Unassigned folds detected!"
    return folds


def generate_nested_folds(
    manifest_path: str = "data/manifest.csv",
    output_dir: str = "data/folds",
    outer_n_splits: int = 5,
    outer_n_repeats: int = 5,
    inner_n_splits: int = 3,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Generate and persist unified nested cross-validation splits on all 215 students."""
    df_manifest = pd.read_csv(manifest_path)
    n_students = len(df_manifest)

    student_ids = df_manifest["student_id"].values
    labels = df_manifest["label"].astype(int).values
    schools = df_manifest["school"].values
    grades = df_manifest["grade"].values

    n_pos = int(labels.sum())
    n_neg = int(n_students - n_pos)

    print(f"Generating unified nested CV splits for {n_students} students ({n_pos} positives, {n_neg} controls across School A & B)...")

    folds_data: Dict[str, Any] = {
        "metadata": {
            "cohort": "unified_215",
            "n_students": n_students,
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "n_school_a": int((schools == "school_a").sum()),
            "n_school_b": int((schools == "school_b").sum()),
            "outer_n_splits": outer_n_splits,
            "outer_n_repeats": outer_n_repeats,
            "inner_n_splits": inner_n_splits,
            "base_seed": base_seed,
        },
        "repeats": [],
    }

    # Tracking table for easy lookup in CSV format
    lookup_df = pd.DataFrame({
        "student_id": student_ids,
        "school": schools,
        "grade": grades,
        "label": labels,
    })

    for repeat_idx in range(outer_n_repeats):
        repeat_seed = base_seed + (repeat_idx * 100)
        fold_assignments = _generate_balanced_split(df_manifest, seed=repeat_seed)

        repeat_entry: Dict[str, Any] = {
            "repeat_id": repeat_idx,
            "seed": repeat_seed,
            "outer_folds": [],
        }

        col_name = f"repeat_{repeat_idx}_fold"
        lookup_df[col_name] = fold_assignments

        for outer_fold_idx in range(outer_n_splits):
            test_mask = fold_assignments == outer_fold_idx
            train_mask = ~test_mask

            outer_train_students = list(student_ids[train_mask])
            outer_test_students = list(student_ids[test_mask])
            outer_train_labels = labels[train_mask]

            # Generate Inner 3-fold splits on outer_train
            inner_seed = repeat_seed + 10 + outer_fold_idx
            skf_inner = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=inner_seed)

            inner_folds: List[Dict[str, Any]] = []
            for inner_fold_idx, (in_train_rel_idx, in_val_rel_idx) in enumerate(
                skf_inner.split(outer_train_students, outer_train_labels)
            ):
                in_train_students = [outer_train_students[i] for i in in_train_rel_idx]
                in_val_students = [outer_train_students[i] for i in in_val_rel_idx]

                inner_folds.append(
                    {
                        "inner_fold_id": inner_fold_idx,
                        "train_students": in_train_students,
                        "val_students": in_val_students,
                        "n_train": len(in_train_students),
                        "n_val": len(in_val_students),
                        "train_positives": int(labels[train_mask][in_train_rel_idx].sum()),
                        "val_positives": int(labels[train_mask][in_val_rel_idx].sum()),
                    }
                )

            outer_fold_entry = {
                "outer_fold_id": outer_fold_idx,
                "test_students": outer_test_students,
                "train_students": outer_train_students,
                "n_test": len(outer_test_students),
                "n_train": len(outer_train_students),
                "test_positives": int(labels[test_mask].sum()),
                "train_positives": int(labels[train_mask].sum()),
                "test_school_a": int((schools[test_mask] == "school_a").sum()),
                "test_school_b": int((schools[test_mask] == "school_b").sum()),
                "inner_folds": inner_folds,
            }
            repeat_entry["outer_folds"].append(outer_fold_entry)

        folds_data["repeats"].append(repeat_entry)

    # Persist fold files
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "nested_folds.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(folds_data, f, indent=2)

    csv_path = out_dir / "student_folds_lookup.csv"
    lookup_df.to_csv(csv_path, index=False)

    print(f"Unified 215-student folds successfully written to:")
    print(f"  - Complete nested fold hierarchy: {json_path}")
    print(f"  - Student fold assignment lookup table: {csv_path}")

    # Summary of first repeat
    r0 = folds_data["repeats"][0]
    print("\nSummary of Repeat 0 Outer Folds:")
    for f in r0["outer_folds"]:
        print(
            f"  Fold {f['outer_fold_id']}: Train={f['n_train']} (Pos={f['train_positives']}), "
            f"Test={f['n_test']} (Pos={f['test_positives']}, SchA={f['test_school_a']}, SchB={f['test_school_b']})"
        )

    return folds_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CV folds for Workstream A")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cv_cfg = cfg.get("cross_validation", {})
    generate_nested_folds(
        manifest_path=cfg["paths"]["manifest_csv"],
        output_dir=cfg["paths"]["folds_dir"],
        outer_n_splits=cv_cfg.get("outer_n_splits", 5),
        outer_n_repeats=cv_cfg.get("outer_n_repeats", 5),
        inner_n_splits=cv_cfg.get("inner_n_splits", 3),
        base_seed=cfg.get("random_seed", 42),
    )

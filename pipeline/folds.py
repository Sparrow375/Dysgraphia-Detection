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


def generate_nested_folds(
    manifest_path: str = "data/manifest.csv",
    output_dir: str = "data/folds",
    outer_n_splits: int = 5,
    outer_n_repeats: int = 5,
    inner_n_splits: int = 3,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Generate and persist nested cross-validation splits on School A."""
    df_manifest = pd.read_csv(manifest_path)

    # Filter to School A only
    df_a = df_manifest[df_manifest["school"] == "school_a"].copy().reset_index(drop=True)
    n_students = len(df_a)

    student_ids = df_a["student_id"].values
    labels = df_a["label"].astype(int).values

    print(f"Generating nested CV splits for School A ({n_students} students, {labels.sum()} positives)...")

    folds_data: Dict[str, Any] = {
        "metadata": {
            "school": "school_a",
            "n_students": n_students,
            "n_positives": int(labels.sum()),
            "n_negatives": int(n_students - labels.sum()),
            "outer_n_splits": outer_n_splits,
            "outer_n_repeats": outer_n_repeats,
            "inner_n_splits": inner_n_splits,
            "base_seed": base_seed,
        },
        "repeats": [],
    }

    # Tracking table for easy lookup in CSV format
    lookup_df = pd.DataFrame({"student_id": student_ids, "label": labels, "grade": df_a["grade"].values})

    for repeat_idx in range(outer_n_repeats):
        repeat_seed = base_seed + (repeat_idx * 100)
        skf_outer = StratifiedKFold(n_splits=outer_n_splits, shuffle=True, random_state=repeat_seed)

        repeat_entry: Dict[str, Any] = {
            "repeat_id": repeat_idx,
            "seed": repeat_seed,
            "outer_folds": [],
        }

        col_name = f"repeat_{repeat_idx}_fold"
        lookup_df[col_name] = -1

        for outer_fold_idx, (train_idx, test_idx) in enumerate(skf_outer.split(student_ids, labels)):
            lookup_df.loc[test_idx, col_name] = outer_fold_idx

            outer_train_students = list(student_ids[train_idx])
            outer_test_students = list(student_ids[test_idx])
            outer_train_labels = labels[train_idx]

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
                        "train_positives": int(sum(df_a.set_index("student_id").loc[in_train_students, "label"])),
                        "val_positives": int(sum(df_a.set_index("student_id").loc[in_val_students, "label"])),
                    }
                )

            outer_fold_entry = {
                "outer_fold_id": outer_fold_idx,
                "test_students": outer_test_students,
                "train_students": outer_train_students,
                "n_test": len(outer_test_students),
                "n_train": len(outer_train_students),
                "test_positives": int(labels[test_idx].sum()),
                "train_positives": int(labels[train_idx].sum()),
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

    print(f"Folds successfully written to:")
    print(f"  - Complete nested fold hierarchy: {json_path}")
    print(f"  - Student fold assignment lookup table: {csv_path}")

    # Summary of first repeat
    r0 = folds_data["repeats"][0]
    print("\nSummary of Repeat 0 Outer Folds:")
    for f in r0["outer_folds"]:
        print(
            f"  Fold {f['outer_fold_id']}: Train={f['n_train']} (Pos={f['train_positives']}), "
            f"Test={f['n_test']} (Pos={f['test_positives']})"
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

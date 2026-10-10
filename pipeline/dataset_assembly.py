"""Phase 3: Dataset Assembly and Feature Preparation.

Assembles modeled sentence-level dataset tables from extracted features:
1. One-hot task encoding (copy, dictated, own-writing).
2. Merge student metadata, ground-truth labels, and nested CV fold assignments.
3. Robust z-score normalization per (grade x script x task) cell with minimum cell size fallback.
4. Export separate Hindi and English datasets plus unified combined dataset.
5. Provides leakage-free cross-validation fold median imputation utility.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Identify script-specific features
HINDI_ONLY_FEATURES = [
    "matra_ratio",
    "shirorekha_rms_deviation_norm",
    "shirorekha_breaks_per_word",
    "shirorekha_tilt_var",
]

ENGLISH_ONLY_FEATURES = [
    "ascender_descender_ratio",
]


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_task_group(task_name: str) -> str:
    """Classify task name into standard protocol task group."""
    t = str(task_name).lower()
    if "copy" in t:
        return "copy"
    elif "dictat" in t:
        return "dictated"
    elif "own" in t or "custom" in t or "free" in t:
        return "own"
    return "other"


def compute_robust_z_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_cell_size: int = 15,
    clip_range: Tuple[float, float] = (-5.0, 5.0),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute robust z-scores: z = (x - median) / (1.4826 * MAD) per cell.

    Primary cell: (grade, script, task_group).
    Fallback cell: (grade, script).
    """
    df_out = df.copy()
    ref_stats: Dict[str, Any] = {}

    # Create grouping columns if not present
    if "task_group" not in df_out.columns:
        df_out["task_group"] = df_out["task_name"].apply(get_task_group)

    for feat in feature_cols:
        z_col = f"z_{feat}"
        df_out[z_col] = np.nan
        ref_stats[feat] = {"primary_cells": {}, "fallback_cells": {}}

        # Fallback cell statistics: (grade, script)
        for (g, s), grp in df_out.groupby(["grade", "script"]):
            valid_vals = grp[feat].dropna()
            if len(valid_vals) >= 3:
                f_med = float(valid_vals.median())
                f_mad = float(np.median(np.abs(valid_vals - f_med)))
            else:
                f_med = float(df_out[feat].dropna().median()) if len(df_out[feat].dropna()) > 0 else 0.0
                f_mad = float(np.median(np.abs(df_out[feat].dropna() - f_med))) if len(df_out[feat].dropna()) > 0 else 1.0

            ref_stats[feat]["fallback_cells"][f"{g}_{s}"] = {
                "median": f_med,
                "mad": f_mad,
                "count": len(valid_vals),
            }

        # Primary cell statistics: (grade, script, task_group)
        for (g, s, tg), grp in df_out.groupby(["grade", "script", "task_group"]):
            valid_vals = grp[feat].dropna()
            cell_key = f"{g}_{s}_{tg}"

            if len(valid_vals) >= min_cell_size:
                c_med = float(valid_vals.median())
                c_mad = float(np.median(np.abs(valid_vals - c_med)))
                used_fallback = False
            else:
                # Fall back to (grade, script)
                fb = ref_stats[feat]["fallback_cells"].get(f"{g}_{s}", {"median": 0.0, "mad": 1.0})
                c_med = fb["median"]
                c_mad = fb["mad"]
                used_fallback = True

            scale = 1.4826 * max(c_mad, 1e-6)
            ref_stats[feat]["primary_cells"][cell_key] = {
                "median": c_med,
                "mad": c_mad,
                "scale": scale,
                "count": len(valid_vals),
                "used_fallback": used_fallback,
            }

            # Apply z-scoring to matching rows
            mask = (df_out["grade"] == g) & (df_out["script"] == s) & (df_out["task_group"] == tg)
            raw_vals = df_out.loc[mask, feat]
            z_vals = (raw_vals - c_med) / scale
            z_clipped = np.clip(z_vals, clip_range[0], clip_range[1])
            df_out.loc[mask, z_col] = z_clipped

    return df_out, ref_stats


def impute_fold_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Impute NaNs using training-fold medians with zero test leakage.

    Args:
        train_df: Training fold dataframe.
        test_df: Optional test fold dataframe.
        feature_cols: Feature columns to impute (defaults to all numeric feature cols).

    Returns:
        (imputed_train_df, imputed_test_df)
    """
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c.startswith("z_") or c.startswith("raw_")]

    train_imp = train_df.copy()
    test_imp = test_df.copy() if test_df is not None else None

    # Compute column medians strictly from train_df
    train_medians = train_df[feature_cols].median()

    for col in feature_cols:
        med_val = float(train_medians.get(col, 0.0))
        if np.isnan(med_val):
            med_val = 0.0

        train_imp[col] = train_imp[col].fillna(med_val)
        if test_imp is not None:
            test_imp[col] = test_imp[col].fillna(med_val)

    return train_imp, test_imp


def assemble_datasets(
    raw_features_csv: str = "data/features/features_raw.csv",
    student_folds_csv: str = "data/folds/student_folds_lookup.csv",
    manifest_csv: str = "data/manifest.csv",
    output_dir: str = "data/datasets",
    config_path: str = "configs/config.yaml",
) -> Dict[str, pd.DataFrame]:
    """Execute Phase 3 dataset assembly pipeline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    norm_cfg = cfg.get("normalization", {})
    min_cell_size = int(norm_cfg.get("min_cell_size", 15))
    clip_range = tuple(norm_cfg.get("clip_range", [-5.0, 5.0]))

    df_raw = pd.read_csv(raw_features_csv)
    print(f"Loaded raw features table: {df_raw.shape[0]} rows, {df_raw.shape[1]} cols")

    # Merge student folds lookup
    if os.path.exists(student_folds_csv):
        df_folds = pd.read_csv(student_folds_csv)
        # Select fold columns: repeat_0_fold .. repeat_4_fold
        fold_cols = [c for c in df_folds.columns if c.startswith("repeat_")]
        merge_subset = df_folds[["student_id"] + fold_cols].drop_duplicates(subset=["student_id"])
        df_merged = pd.merge(df_raw, merge_subset, on="student_id", how="left")
    else:
        df_merged = df_raw.copy()
        fold_cols = []

    # One-hot encode task types
    df_merged["task_group"] = df_merged["task_name"].apply(get_task_group)
    df_merged["task_copy"] = (df_merged["task_group"] == "copy").astype(int)
    df_merged["task_dictated"] = (df_merged["task_group"] == "dictated").astype(int)
    df_merged["task_own"] = (df_merged["task_group"] == "own").astype(int)

    # Identify quantitative feature columns
    non_feature_cols = [
        "student_id", "school", "grade", "task_id", "task_name", "script", "label",
        "json_path", "task_group", "task_copy", "task_dictated", "task_own"
    ] + fold_cols

    base_feature_cols = [c for c in df_raw.columns if c not in [
        "student_id", "school", "grade", "task_id", "task_name", "script", "label", "json_path"
    ]]

    # Drop baseline_slope_std: it is NaN for 81% of sentences (requires >= 2 lines per
    # sentence, but the vast majority span a single line). Including it adds near-zero
    # signal while requiring imputation across almost all rows, which can shift the
    # feature distribution toward the training-fold median and destabilize CV estimates.
    base_feature_cols = [c for c in base_feature_cols if c != "baseline_slope_std"]

    # Prefix raw feature columns with raw_ for clarity
    rename_raw = {c: f"raw_{c}" for c in base_feature_cols}
    df_merged = df_merged.rename(columns=rename_raw)
    prefixed_raw_cols = [f"raw_{c}" for c in base_feature_cols]

    # Compute robust z-scores per cell
    df_with_z, ref_stats = compute_robust_z_scores(
        df_merged,
        feature_cols=prefixed_raw_cols,
        min_cell_size=min_cell_size,
        clip_range=clip_range,
    )

    # Standard column ordering
    id_cols = [
        "student_id", "school", "grade", "task_id", "task_name", "script", "task_group",
        "task_copy", "task_dictated", "task_own", "label"
    ]
    meta_cols = id_cols + fold_cols + ["json_path"]
    z_feature_cols = [f"z_{c}" for c in prefixed_raw_cols]

    # Combine tables
    # 1. Hindi Dataset (Devanagari)
    hindi_df = df_with_z[df_with_z["script"] == "devanagari"].copy()
    # Remove Latin-only features
    hindi_cols_to_drop = [f"raw_{f}" for f in ENGLISH_ONLY_FEATURES] + [f"z_raw_{f}" for f in ENGLISH_ONLY_FEATURES]
    hindi_df = hindi_df.drop(columns=[c for c in hindi_cols_to_drop if c in hindi_df.columns])

    # 2. English Dataset (Latin)
    english_df = df_with_z[df_with_z["script"] == "latin"].copy()
    # Remove Hindi-only features
    english_cols_to_drop = [f"raw_{f}" for f in HINDI_ONLY_FEATURES] + [f"z_raw_{f}" for f in HINDI_ONLY_FEATURES]
    english_df = english_df.drop(columns=[c for c in english_cols_to_drop if c in english_df.columns])

    # 3. Combined Dataset (All sentences)
    combined_df = df_with_z.copy()

    # Save to disk
    hindi_csv = out_path / "dataset_hindi.csv"
    english_csv = out_path / "dataset_english.csv"
    combined_csv = out_path / "dataset_combined.csv"

    hindi_df.to_csv(hindi_csv, index=False)
    english_df.to_csv(english_csv, index=False)
    combined_df.to_csv(combined_csv, index=False)

    print(f"Hindi dataset saved to: {hindi_csv} ({len(hindi_df)} rows, {hindi_df.shape[1]} cols)")
    print(f"English dataset saved to: {english_csv} ({len(english_df)} rows, {english_df.shape[1]} cols)")
    print(f"Combined dataset saved to: {combined_csv} ({len(combined_df)} rows, {combined_df.shape[1]} cols)")

    # Save datasets metadata
    meta_path = out_path / "datasets_meta.json"
    meta_info = {
        "datasets": {
            "dataset_hindi": {
                "rows": len(hindi_df),
                "columns": hindi_df.shape[1],
                "path": str(hindi_csv.as_posix()),
                "script": "devanagari",
            },
            "dataset_english": {
                "rows": len(english_df),
                "columns": english_df.shape[1],
                "path": str(english_csv.as_posix()),
                "script": "latin",
            },
            "dataset_combined": {
                "rows": len(combined_df),
                "columns": combined_df.shape[1],
                "path": str(combined_csv.as_posix()),
                "script": "multilingual",
            },
        },
        "normalization_params": {
            "strategy": "pooled_robust_z_scores",
            "formula": "z = (x - median) / (1.4826 * MAD)",
            "min_cell_size": min_cell_size,
            "clip_range": list(clip_range),
        },
        "fold_columns": fold_cols,
        "one_hot_columns": ["task_copy", "task_dictated", "task_own"],
        "num_raw_features": len(prefixed_raw_cols),
        "num_z_features": len(z_feature_cols),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    return {
        "hindi": hindi_df,
        "english": english_df,
        "combined": combined_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble Phase 3 modeled datasets.")
    parser.add_argument("--features", default="data/features/features_raw.csv")
    parser.add_argument("--folds", default="data/folds/student_folds_lookup.csv")
    parser.add_argument("--output", default="data/datasets")
    args = parser.parse_args()

    assemble_datasets(
        raw_features_csv=args.features,
        student_folds_csv=args.folds,
        output_dir=args.output,
    )

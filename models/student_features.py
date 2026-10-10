"""Student-Level Cross-Task Feature Aggregation Pipeline.

Aggregates sentence-level features to student-level features across tasks,
capturing central tendencies, within-student motor instability, extreme
fragmentation episodes, task-difficulty gradients (dictation vs copy),
cross-script divergence (English vs Hindi), and fatigue effects.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
COMBINED_DATASET_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_combined.csv"
FOLDS_LOOKUP_PATH = WORKSPACE_ROOT / "data" / "folds" / "student_folds_lookup.csv"
OUTPUT_DATASET_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level.csv"
OUTPUT_META_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level_meta.json"

# Key features sensitive to cognitive/auditory stress and motor fatigue
KEY_DYSGRAPHIA_FEATURES = [
    "components_per_unit_width",
    "endpoints_per_unit_width",
    "junctions_per_unit_width",
    "baseline_rmse_norm",
    "gap_cv",
    "word_height_cv",
    "jerk_proxy",
    "words_written_ratio",
]


def extract_student_features(df_combined: pd.DataFrame, df_folds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentence-level features into rich student-level representations.

    Parameters
    ----------
    df_combined : pd.DataFrame
        Sentence-level dataset (e.g., dataset_combined.csv).
    df_folds : pd.DataFrame
        Student folds lookup (student_folds_lookup.csv).

    Returns
    -------
    pd.DataFrame
        115-row student-level dataset with engineered cross-task features.
    """
    raw_feature_cols = [c for c in df_combined.columns if c.startswith("raw_")]
    
    # Identify Hindi-only and English-only features
    hindi_cols = [
        "raw_matra_ratio",
        "raw_shirorekha_rms_deviation_norm",
        "raw_shirorekha_breaks_per_word",
        "raw_shirorekha_tilt_var",
    ]
    english_cols = ["raw_ascender_descender_ratio"]
    shared_cols = [c for c in raw_feature_cols if c not in hindi_cols and c not in english_cols]

    student_rows = []

    for student_id, sdf in df_combined.groupby("student_id"):
        s_grade = int(sdf["grade"].iloc[0])
        s_school = sdf["school"].iloc[0]
        s_label = int(sdf["label"].iloc[0])

        rec: Dict[str, object] = {
            "student_id": student_id,
            "school": s_school,
            "grade": s_grade,
            "label": s_label,
            "sentence_count": len(sdf),
        }

        # 1. Central Tendency & Instability across all tasks (Shared Features)
        for col in shared_cols:
            fname = col.replace("raw_", "")
            vals = sdf[col].dropna()
            if len(vals) > 0:
                rec[f"{fname}_mean"] = float(vals.mean())
                rec[f"{fname}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
                rec[f"{fname}_cv"] = float(vals.std() / (abs(vals.mean()) + 1e-6)) if len(vals) > 1 else 0.0
            else:
                rec[f"{fname}_mean"] = np.nan
                rec[f"{fname}_std"] = np.nan
                rec[f"{fname}_cv"] = np.nan

        # 2. Extreme / Worst-Case Values (Acutely reflects episodic motor breakdown)
        for col in [
            "raw_components_per_unit_width",
            "raw_endpoints_per_unit_width",
            "raw_junctions_per_unit_width",
            "raw_baseline_rmse_norm",
            "raw_gap_cv",
            "raw_word_height_cv",
            "raw_jerk_proxy",
        ]:
            fname = col.replace("raw_", "")
            vals = sdf[col].dropna()
            rec[f"{fname}_max"] = float(vals.max()) if len(vals) > 0 else np.nan

        vals_ratio = sdf["raw_words_written_ratio"].dropna()
        rec["words_written_ratio_min"] = float(vals_ratio.min()) if len(vals_ratio) > 0 else np.nan

        # 3. Script-Specific Features
        h_df = sdf[sdf["script"] == "devanagari"]
        for col in hindi_cols:
            fname = col.replace("raw_", "")
            vals = h_df[col].dropna()
            rec[f"{fname}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            rec[f"{fname}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
            if "matra" in col or "shirorekha_rms" in col:
                rec[f"{fname}_max"] = float(vals.max()) if len(vals) > 0 else np.nan

        e_df = sdf[sdf["script"] == "latin"]
        for col in english_cols:
            fname = col.replace("raw_", "")
            vals = e_df[col].dropna()
            rec[f"{fname}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            rec[f"{fname}_std"] = float(vals.std()) if len(vals) > 1 else 0.0

        # 4. Task Context Comparisons (Copy vs Dictated vs Free Writing)
        copy_df = sdf[sdf["task_name"].str.startswith("copy")]
        dict_df = sdf[sdf["task_name"].str.startswith("dictated")]
        own_df = sdf[sdf["task_name"].str.startswith("own")]

        for kf in KEY_DYSGRAPHIA_FEATURES:
            col = f"raw_{kf}"
            if col in sdf.columns:
                c_vals = copy_df[col].dropna()
                d_vals = dict_df[col].dropna()
                o_vals = own_df[col].dropna()

                c_m = float(c_vals.mean()) if len(c_vals) > 0 else np.nan
                d_m = float(d_vals.mean()) if len(d_vals) > 0 else np.nan
                o_m = float(o_vals.mean()) if len(o_vals) > 0 else np.nan

                rec[f"{kf}_copy_mean"] = c_m
                rec[f"{kf}_dict_mean"] = d_m
                rec[f"{kf}_own_mean"] = o_m

                # Task difficulty gradients: Dictation minus Copy (auditory-motor load penalty)
                if not np.isnan(d_m) and not np.isnan(c_m):
                    rec[f"{kf}_dict_minus_copy"] = d_m - c_m
                else:
                    rec[f"{kf}_dict_minus_copy"] = np.nan

                # Free writing minus Copy (compositional load penalty)
                if not np.isnan(o_m) and not np.isnan(c_m):
                    rec[f"{kf}_own_minus_copy"] = o_m - c_m
                else:
                    rec[f"{kf}_own_minus_copy"] = np.nan

        # 5. Cross-Script Divergence (English vs Hindi Motor Parity)
        for kf in KEY_DYSGRAPHIA_FEATURES:
            col = f"raw_{kf}"
            if col in sdf.columns:
                h_vals = h_df[col].dropna()
                e_vals = e_df[col].dropna()

                h_m = float(h_vals.mean()) if len(h_vals) > 0 else np.nan
                e_m = float(e_vals.mean()) if len(e_vals) > 0 else np.nan

                rec[f"{kf}_hindi_mean"] = h_m
                rec[f"{kf}_eng_mean"] = e_m

                if not np.isnan(e_m) and not np.isnan(h_m):
                    rec[f"{kf}_eng_minus_hindi"] = e_m - h_m
                else:
                    rec[f"{kf}_eng_minus_hindi"] = np.nan

        # 6. Fatigue / Temporal Degradation
        # Sort sentences by order and compare second half vs first half
        sorted_sdf = sdf.sort_values(by="task_id") if "task_id" in sdf.columns else sdf
        n_tasks = len(sorted_sdf)
        if n_tasks >= 4:
            first_half = sorted_sdf.iloc[:2]
            second_half = sorted_sdf.iloc[-2:]
            for kf in ["components_per_unit_width", "endpoints_per_unit_width", "baseline_rmse_norm"]:
                col = f"raw_{kf}"
                fh_v = first_half[col].dropna().mean()
                sh_v = second_half[col].dropna().mean()
                if not np.isnan(sh_v) and not np.isnan(fh_v):
                    rec[f"{kf}_fatigue_delta"] = float(sh_v - fh_v)
                else:
                    rec[f"{kf}_fatigue_delta"] = 0.0
        else:
            for kf in ["components_per_unit_width", "endpoints_per_unit_width", "baseline_rmse_norm"]:
                rec[f"{kf}_fatigue_delta"] = 0.0

        student_rows.append(rec)

    df_students = pd.DataFrame(student_rows)

    # Merge fold assignments from student_folds_lookup.csv
    fold_cols = [c for c in df_folds.columns if c.startswith("repeat_")]
    merge_cols = ["student_id"] + fold_cols
    df_students = df_students.merge(df_folds[merge_cols], on="student_id", how="left")

    return df_students


def compute_student_robust_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute robust z-scores grouped by grade for each engineered feature.

    Formula:
        z = (x - median(grade)) / (1.4826 * max(MAD(grade), 1e-6))
    clipped to [-5.0, 5.0].

    Parameters
    ----------
    df : pd.DataFrame
        Student-level dataframe containing raw features.

    Returns
    -------
    pd.DataFrame
        Dataframe with original features plus 'z_' prefixed robust z-scores.
    """
    meta_cols = ["student_id", "school", "grade", "label", "sentence_count"] + [
        c for c in df.columns if c.startswith("repeat_")
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    z_dict = {}
    for col in feature_cols:
        z_col = f"z_{col}"
        z_series = pd.Series(index=df.index, dtype=float)

        for grade_val, gdf in df.groupby("grade"):
            vals = gdf[col].dropna()
            if len(vals) < 3:
                cohort_vals = df[col].dropna()
                med = float(cohort_vals.median()) if len(cohort_vals) > 0 else 0.0
                mad = float(np.median(np.abs(cohort_vals - med))) if len(cohort_vals) > 0 else 1.0
            else:
                med = float(vals.median())
                mad = float(np.median(np.abs(vals - med)))

            denom = 1.4826 * max(mad, 1e-6)
            grade_z = (gdf[col] - med) / denom
            grade_z = grade_z.clip(lower=-5.0, upper=5.0)
            z_series.loc[gdf.index] = grade_z

        z_dict[z_col] = z_series

    df_z = pd.DataFrame(z_dict, index=df.index)
    df_out = pd.concat([df, df_z], axis=1)
    return df_out


def build_student_level_dataset(
    combined_path: Path = COMBINED_DATASET_PATH,
    folds_path: Path = FOLDS_LOOKUP_PATH,
    output_path: Path = OUTPUT_DATASET_PATH,
) -> pd.DataFrame:
    """Run full student-level feature extraction and persist to CSV."""
    df_combined = pd.read_csv(combined_path)
    df_folds = pd.read_csv(folds_path)

    df_student_raw = extract_student_features(df_combined, df_folds)
    df_student_full = compute_student_robust_z_scores(df_student_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_student_full.to_csv(output_path, index=False)

    meta_cols = ["student_id", "school", "grade", "label", "sentence_count"] + [
        c for c in df_student_full.columns if c.startswith("repeat_")
    ]
    raw_feats = [c for c in df_student_full.columns if not c.startswith("z_") and c not in meta_cols]
    z_feats = [c for c in df_student_full.columns if c.startswith("z_")]

    meta = {
        "n_students": len(df_student_full),
        "n_positive": int((df_student_full["label"] == 1).sum()),
        "n_negative": int((df_student_full["label"] == 0).sum()),
        "n_raw_features": len(raw_feats),
        "n_z_features": len(z_feats),
        "total_columns": len(df_student_full.columns),
        "raw_feature_names": raw_feats,
        "z_feature_names": z_feats,
    }

    import json
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully generated student-level dataset: {output_path}")
    print(f"Total students: {len(df_student_full)} ({meta['n_positive']} pos, {meta['n_negative']} neg)")
    print(f"Raw engineered features: {len(raw_feats)}, Z-score features: {len(z_feats)}")

    return df_student_full


if __name__ == "__main__":
    build_student_level_dataset()

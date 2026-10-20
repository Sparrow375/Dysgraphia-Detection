"""Student-Level Task-Agnostic Page Aggregation Pipeline (Stage 25 Overhaul).

Converts sentence-level and line-level handwriting features into pure,
task-agnostic intra-page distribution representations:
1. Unordered Line Treatment: No reliance on task labels (copy, dictated, own).
2. Intra-Page Distribution Metrics:
   - Page Median (_median): Baseline motor execution across lines.
   - Line-to-Line Volatility (_iqr / _std): Fluctuations across lines on the page.
   - Worst-Line Breakdown (_max / _p90): Acute motor collapse on messiest line.
   - Within-Page Fatigue Gradient (_late_minus_early): Bottom-half vs top-half lines.
3. Writing Tool Classifier Integration:
   - Pure-math pen vs pencil detector on page_normalized.png.
   - Tool-conditioned Stream 2 normalization (pencil vs pen baselines).
   - Interaction terms (is_pencil * pressure_intensity_cv, etc.).
4. Training Line-Drop Augmentation:
   - Subsamples 2 to 5 lines per student during training iterations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from features.stroke_width_texture import calibrate_stream2_by_tool, classify_writing_tool

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
COMBINED_DATASET_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_combined.csv"
RAW_FEATURES_PATH = WORKSPACE_ROOT / "data" / "features" / "features_raw.csv"
FOLDS_LOOKUP_PATH = WORKSPACE_ROOT / "data" / "folds" / "student_folds_lookup.csv"
OUTPUT_DATASET_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level.csv"
OUTPUT_META_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level_meta.json"

# Core Stream 1 Geometric Features
STREAM1_GEOMETRIC_FEATURES: List[str] = [
    "components_per_unit_width",
    "endpoints_per_unit_width",
    "junctions_per_unit_width",
    "baseline_rmse_norm",
    "baseline_slope_mean",
    "rule_offset_mean",
    "rule_offset_std",
    "slant_mean_deg",
    "slant_circular_std_deg",
    "jerk_proxy",
    "tangent_variance_short_wavelength",
    "kappa_sign_changes_per_h",
    "tortuosity_median",
    "gap_mean",
    "gap_cv",
    "gap_fraction_below_0_3h",
    "gap_fraction_above_2h",
    "word_height_cv",
    "h_over_r",
    "component_height_cv",
]

# Script-Specific Features
HINDI_GEOMETRIC_FEATURES: List[str] = [
    "matra_ratio",
    "shirorekha_rms_deviation_norm",
    "shirorekha_breaks_per_word",
    "shirorekha_tilt_var",
]

ENGLISH_GEOMETRIC_FEATURES: List[str] = [
    "ascender_descender_ratio",
]

# Core Stream 2 Stroke-Width & Micro-Texture Features
STREAM2_RAW_FEATURES: List[str] = [
    "stroke_width_mean_norm",
    "stroke_width_cv",
    "stroke_width_jitter",
    "stroke_width_pinch_rate",
    "stroke_width_bloat_rate",
    "pressure_intensity_mean",
    "pressure_intensity_cv",
    "pressure_gradient_mean",
    "pressure_faint_rate",
    "pressure_heavy_rate",
    "pressure_width_corr",
    "smudge_halo_darkness",
    "erasure_patch_ratio",
]


def sample_student_lines(
    sdf: pd.DataFrame,
    min_lines: int = 2,
    max_lines: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> pd.DataFrame:
    """Randomly sample a subset of lines for a student to augment training robustness.
    
    Ensures model performs reliably whether real-world scan has 2 lines or 10 lines.
    Preserves relative vertical line order.
    """
    if rng is None:
        rng = np.random.RandomState()
    n = len(sdf)
    if n <= min_lines:
        return sdf
    target_k = int(rng.randint(min_lines, min(max_lines, n) + 1))
    sampled_indices = sorted(rng.choice(n, size=target_k, replace=False))
    return sdf.iloc[sampled_indices].copy()


def compute_intra_page_metrics(vals: np.ndarray) -> Dict[str, float]:
    """Compute intra-page distribution metrics for a single feature series across lines."""
    v = vals[~np.isnan(vals)]
    if len(v) == 0:
        return {
            "median": np.nan,
            "iqr": np.nan,
            "std": np.nan,
            "max": np.nan,
            "p90": np.nan,
            "late_minus_early": 0.0,
        }

    med = float(np.median(v))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) >= 2 else 0.0
    sd = float(np.std(v)) if len(v) >= 2 else 0.0
    mx = float(np.max(v))
    p90 = float(np.percentile(v, 90))

    # Within-page fatigue gradient: bottom-half lines minus top-half lines
    k = len(v)
    if k == 2:
        late_minus_early = float(v[1] - v[0])
    elif k == 3:
        late_minus_early = float(v[2] - v[0])
    elif k >= 4:
        half = k // 2
        late_minus_early = float(np.median(v[half:]) - np.median(v[:half]))
    else:
        late_minus_early = 0.0

    return {
        "median": med,
        "iqr": iqr,
        "std": sd,
        "max": mx,
        "p90": p90,
        "late_minus_early": late_minus_early,
    }


def extract_student_features(
    df_combined: pd.DataFrame,
    df_folds: pd.DataFrame,
    data_root: str = "data/processed",
) -> pd.DataFrame:
    """Aggregate sentence-level features into task-agnostic student-level representations."""
    student_rows = []

    # Map student tool classification from page_normalized.png
    tool_cache: Dict[str, Dict[str, Any]] = {}
    processed_base = Path(data_root)

    for student_id, sdf in df_combined.groupby("student_id"):
        s_grade = int(sdf["grade"].iloc[0])
        s_school = sdf["school"].iloc[0]
        s_label = int(sdf["label"].iloc[0])

        # 1. Writing Tool Classification on page_normalized.png
        if student_id not in tool_cache:
            page_path = processed_base / s_school / student_id / "page_normalized.png"
            if page_path.exists():
                tool_res = classify_writing_tool(page_path)
            else:
                tool_res = {"tool": "pen", "is_pencil": 0.0, "w_med": 2.0}
            tool_cache[student_id] = tool_res
        else:
            tool_res = tool_cache[student_id]

        rec: Dict[str, Any] = {
            "student_id": student_id,
            "school": s_school,
            "grade": s_grade,
            "label": s_label,
            "line_count": len(sdf),
            "is_pencil": float(tool_res.get("is_pencil", 0.0)),
            "detected_tool": str(tool_res.get("tool", "pen")),
            "tool_w_med": float(tool_res.get("w_med", 2.0)),
        }

        # Ensure lines are ordered vertically (by task_id or page y)
        if "task_id" in sdf.columns:
            sorted_sdf = sdf.sort_values(by="task_id").copy()
        else:
            sorted_sdf = sdf.copy()

        # 2. STREAM 1: Intra-Page Geometric Distributions
        for feat in STREAM1_GEOMETRIC_FEATURES:
            col = f"raw_{feat}" if f"raw_{feat}" in sorted_sdf.columns else feat
            if col in sorted_sdf.columns:
                v = sorted_sdf[col].values.astype(float)
                dist = compute_intra_page_metrics(v)
                rec[f"{feat}_median"] = dist["median"]
                rec[f"{feat}_iqr"] = dist["iqr"]
                rec[f"{feat}_std"] = dist["std"]
                rec[f"{feat}_max"] = dist["max"]
                rec[f"{feat}_p90"] = dist["p90"]
                rec[f"{feat}_late_minus_early"] = dist["late_minus_early"]

        # 3. STREAM 1: Script-Specific Geometric Features
        h_df = sorted_sdf[sorted_sdf["script"] == "devanagari"]
        for feat in HINDI_GEOMETRIC_FEATURES:
            col = f"raw_{feat}" if f"raw_{feat}" in h_df.columns else feat
            if col in h_df.columns:
                v = h_df[col].values.astype(float)
                dist = compute_intra_page_metrics(v)
                rec[f"{feat}_median"] = dist["median"]
                rec[f"{feat}_iqr"] = dist["iqr"]
                rec[f"{feat}_p90"] = dist["p90"]
                rec[f"{feat}_max"] = dist["max"]

        e_df = sorted_sdf[sorted_sdf["script"] == "latin"]
        for feat in ENGLISH_GEOMETRIC_FEATURES:
            col = f"raw_{feat}" if f"raw_{feat}" in e_df.columns else feat
            if col in e_df.columns:
                v = e_df[col].values.astype(float)
                dist = compute_intra_page_metrics(v)
                rec[f"{feat}_median"] = dist["median"]
                rec[f"{feat}_iqr"] = dist["iqr"]

        # Cross-Script Divergence (English vs Hindi Motor Parity)
        for feat in ["components_per_unit_width", "endpoints_per_unit_width", "gap_cv", "word_height_cv"]:
            col = f"raw_{feat}" if f"raw_{feat}" in sorted_sdf.columns else feat
            if col in sorted_sdf.columns:
                h_vals = h_df[col].dropna()
                e_vals = e_df[col].dropna()
                h_m = float(h_vals.median()) if len(h_vals) > 0 else np.nan
                e_m = float(e_vals.median()) if len(e_vals) > 0 else np.nan
                rec[f"{feat}_eng_minus_hindi"] = (e_m - h_m) if not np.isnan(e_m) and not np.isnan(h_m) else 0.0

        # 4. STREAM 2: Stroke-Width & Grayscale Shading Micro-Texture
        for feat in STREAM2_RAW_FEATURES:
            col = f"raw_{feat}" if f"raw_{feat}" in sorted_sdf.columns else feat
            if col in sorted_sdf.columns:
                v = sorted_sdf[col].values.astype(float)
                dist = compute_intra_page_metrics(v)
                rec[f"{feat}_median"] = dist["median"]
                rec[f"{feat}_iqr"] = dist["iqr"]
                rec[f"{feat}_max"] = dist["max"]

        # Base Stream 2 aliases for calibration
        for feat in [
            "pressure_intensity_cv",
            "stroke_width_jitter",
            "stroke_width_bloat_rate",
            "pressure_faint_rate",
            "pressure_heavy_rate",
            "smudge_halo_darkness",
            "erasure_patch_ratio",
        ]:
            rec[feat] = rec.get(f"{feat}_median", np.nan)

        student_rows.append(rec)

    df_students = pd.DataFrame(student_rows)

    # 5. Tool-Conditioned Stream 2 Normalization & Interaction Terms
    df_students, _ = calibrate_stream2_by_tool(df_students)

    is_p = df_students["is_pencil"].values
    if "pressure_intensity_cv_tool_norm" in df_students.columns:
        df_students["interaction_pencil_pressure"] = is_p * df_students["pressure_intensity_cv_tool_norm"]
    if "stroke_width_jitter_tool_norm" in df_students.columns:
        df_students["interaction_pen_width_jitter"] = (1.0 - is_p) * df_students["stroke_width_jitter_tool_norm"]
    if "stroke_width_bloat_rate_tool_norm" in df_students.columns:
        df_students["interaction_pen_bloat"] = (1.0 - is_p) * df_students["stroke_width_bloat_rate_tool_norm"]

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
    """
    meta_cols = [
        "student_id",
        "school",
        "grade",
        "label",
        "line_count",
        "is_pencil",
        "detected_tool",
    ] + [c for c in df.columns if c.startswith("repeat_")]
    feature_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("z_")]

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
    raw_path: Path = RAW_FEATURES_PATH,
    folds_path: Path = FOLDS_LOOKUP_PATH,
    output_path: Path = OUTPUT_DATASET_PATH,
) -> pd.DataFrame:
    """Run full task-agnostic student-level feature extraction across all 215 students."""
    # Load sentence features from features_raw.csv (complete across School A and B)
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
    else:
        df_raw = pd.read_csv(COMBINED_DATASET_PATH)

    df_folds = pd.read_csv(folds_path)
    print(f"Building task-agnostic student dataset across {df_raw['student_id'].nunique()} students...")

    df_student_raw = extract_student_features(df_raw, df_folds)
    df_student_full = compute_student_robust_z_scores(df_student_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_student_full.to_csv(output_path, index=False)

    meta_cols = [
        "student_id",
        "school",
        "grade",
        "label",
        "line_count",
        "is_pencil",
        "detected_tool",
    ] + [c for c in df_student_full.columns if c.startswith("repeat_")]
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

    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully generated task-agnostic student dataset: {output_path}")
    print(f"Total students: {len(df_student_full)} ({meta['n_positive']} pos, {meta['n_negative']} neg)")
    print(f"Raw features: {len(raw_feats)}, Z-score features: {len(z_feats)}")

    return df_student_full


if __name__ == "__main__":
    build_student_level_dataset()

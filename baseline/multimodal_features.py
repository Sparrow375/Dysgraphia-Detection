"""Line-level multi-modal feature compilation and leak-free scaling for Experiments 4 & 5.

Features:
---------
- 22 standardized line-level physical math, pressure, stroke-width, tool, and TrOCR features.
- Leak-free robust z-score fitting strictly on training fold.
- Learning Under Privileged Information (LUPI): Continuous Wacom stylus motor severity targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"
FEATURES_RAW_CSV = DATA_DIR / "features" / "features_raw.csv"
TROCR_CACHE_JSON = DATA_DIR / "features" / "trocr_cache.json"
STUDENT_LEVEL_CSV = DATA_DIR / "datasets" / "dataset_student_level.csv"
STYLUS_CSV = DATA_DIR / "school_a" / "stylus_student_features.csv"

SELECTED_MATH_FEATURES_16 = [
    "baseline_rmse_norm",
    "rule_offset_std",
    "slant_circular_std_deg",
    "jerk_proxy",
    "kappa_sign_changes_per_h",
    "tortuosity_median",
    "gap_mean",
    "gap_cv",
    "gap_fraction_below_0_3h",
    "gap_fraction_above_2h",
    "word_height_cv",
    "component_height_cv",
    "components_per_unit_width",
    "stroke_width_cv",
    "pressure_intensity_cv",
    "trocr_mean_confidence",
]

SELECTED_MATH_FEATURES_22 = [
    "is_pencil",
    "baseline_rmse_norm",
    "rule_offset_std",
    "slant_circular_std_deg",
    "jerk_proxy",
    "kappa_sign_changes_per_h",
    "tortuosity_median",
    "gap_mean",
    "gap_cv",
    "gap_fraction_below_0_3h",
    "gap_fraction_above_2h",
    "word_height_cv",
    "component_height_cv",
    "components_per_unit_width",
    "stroke_width_cv",
    "stroke_width_jitter",
    "stroke_width_bloat_rate",
    "pressure_intensity_cv",
    "pressure_faint_rate",
    "pressure_heavy_rate",
    "trocr_mean_confidence",
    "trocr_low_conf_ratio",
]

SELECTED_MATH_FEATURES = SELECTED_MATH_FEATURES_22


def load_raw_multimodal_dataframe() -> pd.DataFrame:
    """Load sentences index, merge features_raw.csv, student tool indicators, and TrOCR legibility signals."""
    df_index = pd.read_csv(INDEX_CSV)
    df_raw = pd.read_csv(FEATURES_RAW_CSV)

    # Merge by school, student_id, task_id
    df_merged = pd.merge(
        df_index,
        df_raw,
        on=["school", "student_id", "task_id"],
        how="left",
        suffixes=("", "_raw"),
    )

    # Merge is_pencil from dataset_student_level.csv if available
    if STUDENT_LEVEL_CSV.exists():
        df_stu = pd.read_csv(STUDENT_LEVEL_CSV)
        if "is_pencil" in df_stu.columns:
            stu_tool = dict(zip(df_stu["student_id"], df_stu["is_pencil"]))
            df_merged["is_pencil"] = df_merged["student_id"].map(stu_tool).fillna(0.0)
        else:
            df_merged["is_pencil"] = 0.0
    else:
        df_merged["is_pencil"] = 0.0

    # Load TrOCR cache
    trocr_cache: Dict[str, Any] = {}
    if TROCR_CACHE_JSON.exists():
        with open(TROCR_CACHE_JSON) as f:
            trocr_cache = json.load(f)

    # Populate TrOCR signals
    trocr_mean_conf: List[float] = []
    trocr_low_conf: List[float] = []

    for _, row in df_merged.iterrows():
        sid = str(row["student_id"])
        tid = str(row["task_id"])
        key = f"{sid}_{tid}"

        if key in trocr_cache:
            entry = trocr_cache[key]
            m_conf = float(entry.get("mean_confidence", 0.70))
            l_conf = float(entry.get("low_conf_ratio", 0.20))
        else:
            # Fallback based on physical stroke fragmentation & jerk proxy
            comp_w_raw = row.get("components_per_unit_width", 1.0)
            comp_w = float(comp_w_raw) if pd.notna(comp_w_raw) else 1.0
            jerk_raw = row.get("jerk_proxy", 0.5)
            jerk = float(jerk_raw) if pd.notna(jerk_raw) else 0.5
            m_conf = float(np.clip(0.92 - (0.05 * comp_w + 0.08 * jerk), 0.15, 0.98))
            l_conf = float(np.clip(0.10 + (0.06 * comp_w + 0.05 * jerk), 0.02, 0.85))

        if pd.isna(m_conf):
            m_conf = 0.70
        if pd.isna(l_conf):
            l_conf = 0.20

        trocr_mean_conf.append(m_conf)
        trocr_low_conf.append(l_conf)

    df_merged["trocr_mean_confidence"] = trocr_mean_conf
    df_merged["trocr_low_conf_ratio"] = trocr_low_conf

    return df_merged


def load_wacom_stylus_lupi_targets() -> Dict[str, float]:
    """Load continuous Wacom stylus motor severity targets for School A students (LUPI).

    Returns:
        Dict mapping student_id -> continuous severity score q_i in [0.05, 0.95]
    """
    if not STYLUS_CSV.exists():
        return {}

    try:
        df_stylus = pd.read_csv(STYLUS_CSV)
    except Exception:
        return {}

    cols = ["air_ratio", "mean_pause_sec", "log_jerk", "cv_vel"]
    for c in cols:
        if c not in df_stylus.columns:
            return {}

    z_scores = []
    for c in cols:
        vals = df_stylus[c].dropna().values
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = max(1e-6, 1.4826 * mad)
        z = (df_stylus[c] - med) / scale
        z_scores.append(z)

    if not z_scores:
        return {}

    z_combined = np.mean(z_scores, axis=0)
    q_stylus = 1.0 / (1.0 + np.exp(-z_combined))
    q_stylus = np.clip(q_stylus, 0.05, 0.95)

    lupi_dict: Dict[str, float] = {}
    for sid, q in zip(df_stylus["student_id"], q_stylus):
        if pd.notna(q):
            lupi_dict[str(sid)] = float(q)

    return lupi_dict


def fit_and_scale_multimodal_features(
    df_all: pd.DataFrame,
    train_uids: List[str],
    test_uids: List[str],
    feature_cols: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Fit robust z-scores strictly on training crops and return feature dictionaries.

    Args:
        df_all: Merged dataset dataframe
        train_uids: Task UIDs for training students
        test_uids: Task UIDs for test students
        feature_cols: Optional list of features to extract (defaults to SELECTED_MATH_FEATURES_22)

    Returns:
        math_features_dict: Dict mapping task_uid -> np.ndarray (D,)
        aux_targets_dict: Dict mapping task_uid -> np.ndarray (2,) with [mean_conf, low_conf_ratio]
        meta_dict: Dict with training medians and MAD scales
    """
    if feature_cols is None:
        feature_cols = SELECTED_MATH_FEATURES_22

    df_train = df_all[df_all["task_uid"].isin(train_uids)].copy()

    # Compute training medians and MADs
    stats: Dict[str, Tuple[float, float]] = {}
    for col in feature_cols:
        vals = df_train[col].dropna().values
        med = float(np.median(vals)) if len(vals) > 0 else 0.0
        mad = float(np.median(np.abs(vals - med))) if len(vals) > 0 else 1.0
        scale = max(1e-6, 1.4826 * mad)
        stats[col] = (med, scale)

    math_dict: Dict[str, np.ndarray] = {}
    aux_dict: Dict[str, np.ndarray] = {}

    for _, row in df_all.iterrows():
        uid = str(row["task_uid"])
        vec = np.zeros(len(feature_cols), dtype=np.float32)

        for i, col in enumerate(feature_cols):
            val = row[col]
            med, scale = stats[col]
            if pd.isna(val):
                val = med
            z = np.clip((float(val) - med) / scale, -4.0, 4.0)
            vec[i] = np.float32(z)

        math_dict[uid] = vec

        # Auxiliary targets: raw [mean_conf, low_conf_ratio] in [0, 1]
        m_conf_raw = row.get("trocr_mean_confidence", 0.70)
        m_conf = float(m_conf_raw) if pd.notna(m_conf_raw) else 0.70
        l_conf_raw = row.get("trocr_low_conf_ratio", 0.20)
        l_conf = float(l_conf_raw) if pd.notna(l_conf_raw) else 0.20
        aux_dict[uid] = np.array([m_conf, l_conf], dtype=np.float32)

    return math_dict, aux_dict, stats

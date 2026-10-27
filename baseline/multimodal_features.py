"""Line-level multi-modal feature compilation and leak-free scaling for Experiment 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"
FEATURES_RAW_CSV = DATA_DIR / "features" / "features_raw.csv"
TROCR_CACHE_JSON = DATA_DIR / "features" / "trocr_cache.json"

SELECTED_MATH_FEATURES = [
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


def load_raw_multimodal_dataframe() -> pd.DataFrame:
    """Load sentences index, merge features_raw.csv and TrOCR legibility signals."""
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


def fit_and_scale_multimodal_features(
    df_all: pd.DataFrame,
    train_uids: List[str],
    test_uids: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Fit robust z-scores strictly on training crops and return feature dictionaries.

    Returns:
        math_features_dict: Dict mapping task_uid -> np.ndarray (16,)
        aux_targets_dict: Dict mapping task_uid -> np.ndarray (2,) with [mean_conf, low_conf_ratio]
        meta_dict: Dict with training medians and MADs
    """
    df_train = df_all[df_all["task_uid"].isin(train_uids)].copy()

    # Compute training medians and MADs
    stats: Dict[str, Tuple[float, float]] = {}
    for col in SELECTED_MATH_FEATURES:
        vals = df_train[col].dropna().values
        med = float(np.median(vals)) if len(vals) > 0 else 0.0
        mad = float(np.median(np.abs(vals - med))) if len(vals) > 0 else 1.0
        scale = max(1e-6, 1.4826 * mad)
        stats[col] = (med, scale)

    math_dict: Dict[str, np.ndarray] = {}
    aux_dict: Dict[str, np.ndarray] = {}

    for _, row in df_all.iterrows():
        uid = str(row["task_uid"])
        vec = np.zeros(len(SELECTED_MATH_FEATURES), dtype=np.float32)

        for i, col in enumerate(SELECTED_MATH_FEATURES):
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

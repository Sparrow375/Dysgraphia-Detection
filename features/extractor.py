"""Unified feature extraction orchestrator for Workstream A.

Extracts all 25+ Phase 2 features for a single sentence JSON, or batches
across all student sentence crops in the dataset to produce data/features/features_raw.csv.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from features.baseline import compute_baseline_features
from features.curvature import compute_curvature_features
from features.fragmentation import compute_fragmentation_features
from features.gaps import compute_gap_features
from features.hindi import compute_hindi_features
from features.rule_offset import compute_rule_offset_features
from features.size import compute_size_features
from features.slant import compute_slant_features

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # Baseline
    "baseline_rmse_norm": "RANSAC word-baseline residual RMSE normalized by x-height h",
    "baseline_slope_mean": "Mean baseline slope across lines within sentence",
    "baseline_slope_std": "Standard deviation of baseline slopes across lines (NaN if < 2 lines). EXCLUDED from model feature set (81% NaN — most sentences span a single line).",
    # Rule offset
    "rule_offset_mean": "Mean vertical offset of words from nearest ruling line, normalized by spacing r",
    "rule_offset_std": "Standard deviation of vertical rule offset (vertical adherence jitter)",
    # Slant
    "slant_mean_deg": "Doubled-angle circular mean slant angle of near-vertical strokes (deg)",
    "slant_circular_std_deg": "Doubled-angle circular standard deviation of stroke slant (deg)",
    # Curvature & Kinematics
    "jerk_proxy": "Mean |d(kappa)/ds| * h^2 dimensionless jerk proxy from Savitzky-Golay smoothed paths",
    "tangent_variance_short_wavelength": "Fraction of stroke tangent-angle variance at wavelengths < 0.5h (tremor)",
    "kappa_sign_changes_per_h": "Curvature sign flips with deadband per h of stroke path",
    "tortuosity_median": "Median stroke path length divided by chord length",
    # Gaps
    "gap_mean": "Mean inter-word gap normalized by x-height h",
    "gap_cv": "Coefficient of variation (std / mean) of inter-word gaps",
    "gap_fraction_below_0_3h": "Fraction of inter-word gaps < 0.3h (crowded/overlapping)",
    "gap_fraction_above_2h": "Fraction of inter-word gaps > 2.0h (hesitation/excessive spacing)",
    # Size & Proportions
    "word_height_cv": "Coefficient of variation of word bounding-box heights within sentence",
    "h_over_r": "Ratio of x-height h to ruled line spacing r (macro/micrographia)",
    "word_width_per_char": "Total word width divided by expected prompt character count (copy/dictation)",
    "component_height_cv": "Mean intra-word connected component height CV",
    "ascender_descender_ratio": "English: (ascender + descender height) relative to x-height h",
    "matra_ratio": "Hindi: (upper matra + lower matra height) relative to core band h",
    # Hindi-only
    "shirorekha_rms_deviation_norm": "Hindi: RMS deviation of shirorekha headline from straight line / h",
    "shirorekha_breaks_per_word": "Hindi: mean count of headline breaks/discontinuities per word",
    "shirorekha_tilt_var": "Hindi: variance of headline slopes across words",
    # Fragmentation & Completion
    "components_per_unit_width": "Connected ink components count divided by (sentence_width / h)",
    "junctions_per_unit_width": "Skeleton branch junctions count divided by (sentence_width / h)",
    "endpoints_per_unit_width": "Skeleton endpoints count divided by (sentence_width / h)",
    "words_written_ratio": "Detected word count divided by expected prompt word count",
    "lines_used": "Number of physical handwritten lines used by sentence",
}


def extract_sentence_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Extract all Phase 2 features for a single sentence JSON.

    Args:
        sentence_json: Loaded sentence JSON dictionary.
        image_crop: Optional grayscale or BGR crop image array.

    Returns:
        Flat dictionary of feature name -> float | NaN.
    """
    feats: Dict[str, float] = {}

    # 1. Baseline features
    feats.update(compute_baseline_features(sentence_json, image_crop))

    # 2. Rule offset features
    feats.update(compute_rule_offset_features(sentence_json, image_crop))

    # 3. Slant features
    feats.update(compute_slant_features(sentence_json, image_crop))

    # 4. Curvature & Dynamics
    feats.update(compute_curvature_features(sentence_json, image_crop))

    # 5. Inter-word gaps
    feats.update(compute_gap_features(sentence_json, image_crop))

    # 6. Size and proportion
    feats.update(compute_size_features(sentence_json, image_crop))

    # 7. Hindi specific
    feats.update(compute_hindi_features(sentence_json, image_crop))

    # 8. Fragmentation and completion
    feats.update(compute_fragmentation_features(sentence_json, image_crop))

    return feats


def _process_single_json(jp: str) -> Optional[Dict[str, Any]]:
    """Worker function to parse and extract features for a single sentence JSON."""
    try:
        with open(jp, "r", encoding="utf-8") as f:
            s_json = json.load(f)

        sid = str(s_json.get("student_id", Path(jp).parent.name))
        school = str(s_json.get("school", Path(jp).parent.parent.name))
        grade = int(s_json.get("grade", 0))
        task_id = str(s_json.get("task_id", Path(jp).stem))
        task_name = str(s_json.get("task_name", "task"))
        script = str(s_json.get("script", "devanagari"))

        crop_path = Path(jp).parent / s_json.get("crop_filename", f"{task_id}.png")
        img_crop = None
        if crop_path.exists():
            img_crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)

        feats = extract_sentence_features(s_json, image_crop=img_crop)

        row_dict = {
            "student_id": sid,
            "school": school,
            "grade": grade,
            "task_id": task_id,
            "task_name": task_name,
            "script": script,
            "json_path": str(Path(jp).as_posix()),
        }
        row_dict.update(feats)
        return row_dict
    except Exception as e:
        print(f"Error extracting features from {jp}: {e}")
        return None


def extract_dataset_features(
    processed_dir: str = "data/processed",
    manifest_csv: str = "data/manifest.csv",
    output_dir: str = "data/features",
    school_filter: Optional[str] = "school_a",
    workers: int = 6,
) -> pd.DataFrame:
    """Extract feature table across all processed sentence JSONs in the cohort.

    Saves:
        - data/features/features_raw.csv
        - data/features/features_meta.json
    """
    proc_base = Path(processed_dir)
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Load manifest for student ground-truth labels
    label_map: Dict[str, float] = {}
    if os.path.exists(manifest_csv):
        df_man = pd.read_csv(manifest_csv)
        for _, r in df_man.iterrows():
            sid = str(r["student_id"]).strip()
            lbl = float(r["label"]) if pd.notna(r["label"]) else float("nan")
            label_map[sid] = lbl

    # Collect sentence JSON files
    school_glob = "*" if (not school_filter or school_filter.lower() == "all") else school_filter
    search_path = proc_base / school_glob / "*" / "sentence_*.json"
    json_paths = sorted(glob.glob(str(search_path)))

    print(f"Found {len(json_paths)} sentence JSON schemas to extract features from.")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            raw_results = list(
                tqdm(
                    executor.map(_process_single_json, json_paths),
                    total=len(json_paths),
                    desc=f"Extracting features ({workers} workers)",
                )
            )
        rows = [r for r in raw_results if r is not None]
    else:
        rows = []
        for jp in tqdm(json_paths, desc="Extracting features"):
            r = _process_single_json(jp)
            if r is not None:
                rows.append(r)

    # Inject ground truth labels
    for r in rows:
        r["label"] = label_map.get(r["student_id"], float("nan"))

    df_feats = pd.DataFrame(rows)
    raw_csv_path = out_base / "features_raw.csv"
    df_feats.to_csv(raw_csv_path, index=False)
    print(f"Features raw table saved to: {raw_csv_path} ({df_feats.shape[0]} rows, {df_feats.shape[1]} cols)")

    # Save metadata JSON
    meta_path = out_base / "features_meta.json"
    feature_cols = [c for c in df_feats.columns if c not in [
        "student_id", "school", "grade", "task_id", "task_name", "script", "label", "json_path"
    ]]

    meta_info = {
        "num_rows": len(df_feats),
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
        "feature_descriptions": {f: FEATURE_DESCRIPTIONS.get(f, "Handwriting metric") for f in feature_cols},
        "school_filter": school_filter,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    return df_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Phase 2 feature table.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--output-dir", default="data/features")
    parser.add_argument("--school", default="school_a")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    extract_dataset_features(
        processed_dir=args.processed_dir,
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        school_filter=args.school,
        workers=args.workers,
    )

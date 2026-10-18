"""Multi-Stream Benchmark Engine: Stream 1 (Geometry) + Stream 3 (TrOCR Content).

Evaluates the clinical and empirical benefit of combining:
- Stream 1: Spatial, rule adherence, fragmentation, and curvature kinematics.
- Stream 3: Modern TrOCR handwriting legibility, omission ratio, and cognitive-motor collapse deltas.

Evaluates:
1. Stream 1 alone (Geometry Task-Aware Router).
2. Stream 3 alone (TrOCR Legibility & Linguistic Errors).
3. Early Feature Fusion (Stream 1 + Stream 3 concatenated).
4. Late Probability Stacking (Weighted ensemble of Stream 1 + Stream 3).

Assesses performance across:
- All School B (Grades 3-8, N=100)
- Matched Protocol (Grades 4-7, N=73)
- Zero-Dictation Subgroup (Grade 3, N=16)
- Partial-Dictation Subgroup (Grade 8, N=11)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from features.content import compute_content_features
from features.extractor import extract_sentence_features
from features.ocr_engine import get_trocr_cache
from models.student_features import extract_student_features
from models.task_router import FULL_PROTOCOL_FEATURES, NO_DICTATION_FEATURES, TaskAwareRouter, TaskRoute

REPORTS_DIR = WORKSPACE_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Key OCR features for student-level representation
STREAM3_CORE_FEATURES = [
    "ocr_confidence_mean_mean",
    "ocr_confidence_mean_min",
    "ocr_low_conf_ratio_mean",
    "ocr_omission_ratio_max",
    "ocr_cer_copy_mean",
    "ocr_cer_dict_minus_copy",
    "ocr_confidence_mean_dict_minus_copy",
]


def assemble_multistream_student_dataset(
    processed_dir: str = "data/processed",
    manifest_csv: str = "data/manifest.csv",
    use_cached_ocr_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble student-level dataset with both Stream 1 (Geometry) and Stream 3 (OCR)."""
    import glob

    p_base = Path(processed_dir)
    json_paths = sorted(glob.glob(str(p_base / "*" / "*" / "sentence_*.json")))
    trocr_cache = get_trocr_cache()

    manifest_df = pd.read_csv(manifest_csv)
    label_map = {str(r["student_id"]).strip(): float(r["label"]) for _, r in manifest_df.iterrows()}

    rows = []
    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                sj = json.load(f)
            sid = str(sj.get("student_id", Path(jp).parent.name))
            school = str(sj.get("school", Path(jp).parent.parent.name))
            grade = int(sj.get("grade", 0))
            task_id = str(sj.get("task_id", Path(jp).stem))
            task_name = str(sj.get("task_name", "task"))
            script = str(sj.get("script", "devanagari"))

            crop_p = Path(jp).parent / sj.get("crop_filename", f"{task_id}.png")
            img_crop = None
            if crop_p.exists():
                img_crop = cv2.imread(str(crop_p), cv2.IMREAD_GRAYSCALE)

            # Extract Stream 1 geometric features
            feats = extract_sentence_features(sj, image_crop=img_crop, include_ocr=False)

            # Extract Stream 3 OCR features
            cache_key = f"{sid}_{task_id}"
            if cache_key in trocr_cache:
                ocr_feats = compute_content_features(sj, img_crop, use_cache=True)
                feats.update(ocr_feats)
            elif not use_cached_ocr_only and img_crop is not None:
                ocr_feats = compute_content_features(sj, img_crop, use_cache=False)
                feats.update(ocr_feats)

            # Prepend raw_ prefix to align with student_features convention
            row_dict = {
                "student_id": sid,
                "school": school,
                "grade": grade,
                "task_id": task_id,
                "task_name": task_name,
                "script": script,
                "label": label_map.get(sid, np.nan),
            }
            for k, v in feats.items():
                row_dict[f"raw_{k}"] = v
            rows.append(row_dict)
        except Exception:
            continue

    df_sentences = pd.DataFrame(rows)

    # Dummy folds for student feature aggregation
    unique_students = df_sentences[["student_id"]].drop_duplicates()
    df_folds = unique_students.copy()
    df_folds["repeat_0_fold_0"] = 0

    df_student_all = extract_student_features(df_sentences, df_folds)

    df_train = df_student_all[df_student_all["school"] == "school_a"].copy().reset_index(drop=True)
    df_test = df_student_all[df_student_all["school"] == "school_b"].copy().reset_index(drop=True)

    return df_train, df_test


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute comprehensive diagnostic metrics."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "auprc": None, "brier": None, "n_pos": n_pos, "n_neg": n_neg}

    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    return {"auroc": auc, "auprc": auprc, "brier": brier, "n_pos": n_pos, "n_neg": n_neg}


def run_multistream_benchmark() -> pd.DataFrame:
    """Run full benchmark comparing Stream 1, Stream 3, and Fused Stacking on School B."""
    print("Assembling multi-stream dataset from processed sentence schemas & TrOCR cache...")
    df_train, df_test = assemble_multistream_student_dataset(use_cached_ocr_only=True)
    print(f"Loaded School A train cohort: {len(df_train)} students | School B test cohort: {len(df_test)} students.")

    # 1. STREAM 1 ALONE: Task-Aware Router
    print("\n--- Evaluating Stream 1 Alone (Geometric Task-Aware Router) ---")
    s1_router = TaskAwareRouter(random_state=42)
    s1_router.fit(df_train)
    s1_probs = s1_router.predict_proba(df_test)

    # 2. STREAM 3 ALONE: TrOCR Legibility & Linguistic Errors
    print("--- Evaluating Stream 3 Alone (TrOCR Legibility & Error Deltas) ---")
    available_s3 = [c for c in STREAM3_CORE_FEATURES if c in df_train.columns]
    print(f"Available Stream 3 features: {available_s3}")

    s3_train = df_train[available_s3].copy()
    s3_test = df_test[available_s3].copy()
    s3_meds = s3_train.median()

    X_train_s3 = s3_train.fillna(s3_meds).values
    y_train = df_train["label"].values.astype(int)
    X_test_s3 = s3_test.fillna(s3_meds).values
    y_test = df_test["label"].values.astype(int)

    clf_s3 = LogisticRegression(C=0.1, class_weight="balanced", random_state=42)
    clf_s3.fit(X_train_s3, y_train)
    s3_probs = clf_s3.predict_proba(X_test_s3)[:, 1]

    # 3. EARLY FUSION (Stream 1 + Stream 3 Concatenated)
    print("--- Evaluating Stream 1 + Stream 3 Early Fusion ---")
    fused_full_features = FULL_PROTOCOL_FEATURES + [f for f in available_s3 if f not in FULL_PROTOCOL_FEATURES]
    fused_train = df_train[fused_full_features].copy()
    fused_test = df_test[fused_full_features].copy()
    fused_meds = fused_train.median()

    X_train_fused = fused_train.fillna(fused_meds).values
    X_test_fused = fused_test.fillna(fused_meds).values

    clf_fused = HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.03, min_samples_leaf=4, class_weight="balanced", random_state=42
    )
    clf_fused.fit(X_train_fused, y_train)
    fused_probs = clf_fused.predict_proba(X_test_fused)[:, 1]

    # 4. LATE PROBABILITY STACKING (Weighted Blending)
    # 0.70 Stream 1 (Geometry Router) + 0.30 Stream 3 (TrOCR Legibility)
    print("--- Evaluating Stream 1 + Stream 3 Late Probability Stacking ---")
    stacked_probs = 0.70 * s1_probs + 0.30 * s3_probs

    # Evaluate across cohort slices
    subsets = {
        "All School B (Grades 3-8)": np.ones(len(df_test), dtype=bool),
        "Matched Grades 4-7": df_test["grade"].between(4, 7).values,
        "Grade 3 (Zero Dictation)": (df_test["grade"] == 3).values,
        "Grade 8 (Partial Dictation)": (df_test["grade"] == 8).values,
    }

    results = []
    models_dict = {
        "Stream 1 Alone (Geometry Router)": s1_probs,
        "Stream 3 Alone (TrOCR Legibility)": s3_probs,
        "Early Fusion (Stream 1 + Stream 3)": fused_probs,
        "Late Stacking (0.70 S1 + 0.30 S3)": stacked_probs,
    }

    for slice_name, mask in subsets.items():
        y_slice = y_test[mask]
        for model_name, probs in models_dict.items():
            prob_slice = probs[mask]
            met = compute_metrics(y_slice, prob_slice)
            results.append({
                "cohort_slice": slice_name,
                "model": model_name,
                "n_students": int(np.sum(mask)),
                "positives": met["n_pos"],
                "controls": met["n_neg"],
                "auroc": met["auroc"],
                "auprc": met["auprc"],
                "brier": met["brier"],
            })

    df_res = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "multi_stream_benchmark.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved Multi-Stream Benchmark results to {out_csv}!")

    out_json = REPORTS_DIR / "multi_stream_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return df_res


if __name__ == "__main__":
    df_results = run_multistream_benchmark()
    print("\n=== MULTI-STREAM BENCHMARK SUMMARY ===")
    for slice_name in df_results["cohort_slice"].unique():
        sdf = df_results[df_results["cohort_slice"] == slice_name]
        print(f"\n--- {slice_name} ---")
        for _, r in sdf.iterrows():
            auc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "N/A"
            prc_str = f"{r['auprc']:.4f}" if r["auprc"] is not None else "N/A"
            print(f"  {r['model']:<36} | AUROC: {auc_str} | AUPRC: {prc_str}")

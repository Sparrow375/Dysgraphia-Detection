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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from features.content import CANDIDATE_PROMPTS, clean_text, compute_content_features, levenshtein_distance
from features.extractor import extract_sentence_features
from features.letter_analytics import analyze_letter_errors
from features.ocr_engine import get_trocr_cache
from models.student_features import extract_student_features
from models.task_router import FULL_PROTOCOL_FEATURES, NO_DICTATION_FEATURES, TaskAwareRouter, TaskRoute

REPORTS_DIR = WORKSPACE_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Key Stream 2 features: stroke thickness, pressure variance, and smudge/erasure halos
STREAM2_CORE_FEATURES = [
    "stroke_width_mean_norm_mean",
    "stroke_width_cv_mean",
    "stroke_width_cv_max",
    "stroke_width_jitter_mean",
    "stroke_width_jitter_max",
    "stroke_width_pinch_rate_mean",
    "stroke_width_pinch_rate_max",
    "stroke_width_bloat_rate_mean",
    "stroke_width_bloat_rate_max",
    "pressure_intensity_mean_mean",
    "pressure_intensity_cv_mean",
    "pressure_intensity_cv_max",
    "pressure_gradient_mean_mean",
    "pressure_faint_rate_mean",
    "pressure_faint_rate_max",
    "pressure_heavy_rate_mean",
    "pressure_heavy_rate_max",
    "pressure_width_corr_mean",
    "smudge_halo_darkness_mean",
    "smudge_halo_darkness_max",
    "background_roughness_std_mean",
    "background_roughness_std_max",
    "erasure_patch_ratio_mean",
    "erasure_patch_ratio_max",
    "pressure_intensity_cv_dict_minus_copy",
    "stroke_width_jitter_dict_minus_copy",
    "stroke_width_bloat_rate_dict_minus_copy",
]

# Key OCR and letter-level graphomotor features for student-level representation
STREAM3_CORE_FEATURES = [
    "ocr_confidence_mean_mean",
    "ocr_low_conf_ratio_mean",
    "confusable_rate_mean",
    "confusable_rate_max",
    "midword_caps_rate_mean",
    "midword_caps_rate_max",
    "dropped_endings_rate_mean",
    "char_omission_rate_mean",
    "char_omission_rate_max",
    "motor_error_rate_mean",
    "motor_error_rate_max",
    "motor_vs_phonetic_ratio_mean",
    "motor_vs_phonetic_ratio_max",
    "letter_transpositions_max",
    "word_coda_errors_max",
    "conf_consistency_cv_mean",
    "conf_consistency_cv_max",
    "conf_trajectory_slope_min",
    "ocr_cer_copy_mean",
    "ocr_cer_dict_minus_copy",
    "char_omission_rate_dict_minus_copy",
    "confusable_rate_dict_minus_copy",
    "motor_error_rate_dict_minus_copy",
    "ocr_confidence_mean_dict_minus_copy",
    "forced_mean_logprob_mean",
    "forced_mean_logprob_min",
    "forced_low_prob_ratio_mean",
    "forced_mean_logprob_dict_minus_copy",
]


def assemble_multistream_student_dataset(
    processed_dir: str = "data/processed",
    manifest_csv: str = "data/manifest.csv",
    use_cached_ocr_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble student-level dataset with both Stream 1 (Geometry) and Stream 3 (OCR & Letter Analytics)."""
    raw_csv = WORKSPACE_ROOT / "data" / "features" / "features_raw.csv"
    trocr_cache = get_trocr_cache()

    if raw_csv.exists():
        df_raw = pd.read_csv(raw_csv)
        meta_cols = ["student_id", "school", "grade", "task_id", "task_name", "script", "json_path", "label"]
        feature_cols = [c for c in df_raw.columns if c not in meta_cols]
        rename_map = {c: f"raw_{c}" for c in feature_cols if not c.startswith("raw_")}
        df_sentences = df_raw.rename(columns=rename_map).copy()

        # Extract OCR and letter analytics for all sentences
        letter_rows = []
        for _, r in df_sentences.iterrows():
            sid = str(r["student_id"]).strip()
            tid = str(r["task_id"]).strip()
            task_name = str(r["task_name"]).strip().lower()
            k = f"{sid}_{tid}"

            if k in trocr_cache:
                c = trocr_cache[k]
                decoded = c.get("text", "")
                confs = c.get("token_confidences", [])
                cat = None
                if "copy" in task_name and "english" in task_name:
                    cat = "copy_english"
                elif "dict" in task_name and "english" in task_name:
                    cat = "dictated_english"
                elif "copy" in task_name and "hindi" in task_name:
                    cat = "copy_hindi"
                elif "dict" in task_name and "hindi" in task_name:
                    cat = "dictated_hindi"
                candidates = CANDIDATE_PROMPTS.get(cat, []) if cat else []
                best_cand = ""
                if candidates:
                    cl = clean_text(decoded)
                    best_cand = min(candidates, key=lambda p: levenshtein_distance(clean_text(p), cl))
                script = str(r.get("script", "latin")).lower()
                errs = analyze_letter_errors(best_cand, decoded, token_confidences=confs, script=script)
                row_dict = {f"raw_{k2}": v2 for k2, v2 in errs.items()}
                row_dict["raw_ocr_confidence_mean"] = c.get("mean_confidence", np.nan)
                row_dict["raw_ocr_low_conf_ratio"] = c.get("low_conf_ratio", np.nan)
                row_dict["raw_ocr_omission_ratio"] = c.get("omission_ratio", np.nan)
                row_dict["raw_ocr_cer"] = c.get("cer", np.nan)
                forced = c.get("forced_metrics", {})
                for fk in ["forced_mean_prob", "forced_min_prob", "forced_low_prob_ratio", "forced_mean_logprob", "forced_min_logprob"]:
                    row_dict[f"raw_{fk}"] = float(forced.get(fk, np.nan))
                letter_rows.append(row_dict)
            else:
                letter_rows.append({})

        df_letter = pd.DataFrame(letter_rows)
        for col in df_letter.columns:
            df_sentences[col] = df_letter[col]
    else:
        import glob
        p_base = Path(processed_dir)
        json_paths = sorted(glob.glob(str(p_base / "*" / "*" / "sentence_*.json")))
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

                feats = extract_sentence_features(sj, image_crop=img_crop, include_ocr=False)

                cache_key = f"{sid}_{task_id}"
                if cache_key in trocr_cache:
                    ocr_feats = compute_content_features(sj, img_crop, use_cache=True)
                    feats.update(ocr_feats)
                elif not use_cached_ocr_only and img_crop is not None:
                    ocr_feats = compute_content_features(sj, img_crop, use_cache=False)
                    feats.update(ocr_feats)

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
    """Compute comprehensive diagnostic metrics including sensitivity at fixed specificity."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return {
            "auroc": None, "auprc": None, "brier": None,
            "sens_at_80_spec": None, "sens_at_90_spec": None,
            "youden_sens": None, "youden_spec": None,
            "n_pos": n_pos, "n_neg": n_neg,
        }

    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr

    # Sens @ 80% Spec (spec >= 0.80)
    valid_80 = np.where(spec >= 0.80)[0]
    sens_80 = float(np.max(tpr[valid_80])) if len(valid_80) > 0 else 0.0

    # Sens @ 90% Spec (spec >= 0.90)
    valid_90 = np.where(spec >= 0.90)[0]
    sens_90 = float(np.max(tpr[valid_90])) if len(valid_90) > 0 else 0.0

    # Youden J = sens + spec - 1
    j_scores = tpr + spec - 1.0
    best_idx = int(np.argmax(j_scores))
    youden_sens = float(tpr[best_idx])
    youden_spec = float(spec[best_idx])

    return {
        "auroc": auc,
        "auprc": auprc,
        "brier": brier,
        "sens_at_80_spec": sens_80,
        "sens_at_90_spec": sens_90,
        "youden_sens": youden_sens,
        "youden_spec": youden_spec,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def run_multistream_benchmark() -> pd.DataFrame:
    """Run full benchmark comparing Stream 1, Stream 2, Stream 3, and 3-Stream Late Stacking on School B."""
    print("Assembling multi-stream dataset from processed sentence schemas & TrOCR cache...")
    df_train, df_test = assemble_multistream_student_dataset(use_cached_ocr_only=True)
    print(f"Loaded School A train cohort: {len(df_train)} students | School B test cohort: {len(df_test)} students.")

    y_train = df_train["label"].values.astype(int)
    y_test = df_test["label"].values.astype(int)

    # 1. STREAM 1 ALONE: Task-Aware Router (Geometry)
    print("\n--- Evaluating Stream 1 Alone (Geometric Task-Aware Router) ---")
    s1_router = TaskAwareRouter(random_state=42)
    s1_router.fit(df_train)
    s1_probs = s1_router.predict_proba(df_test)

    # 2. STREAM 2 ALONE: Stroke-Width & Grayscale Shading Micro-Texture
    print("--- Evaluating Stream 2 Alone (Stroke-Width & Pressure Variance) ---")
    available_s2 = [c for c in STREAM2_CORE_FEATURES if c in df_train.columns]
    print(f"Available Stream 2 features: {len(available_s2)} features")

    s2_train = df_train[available_s2].copy()
    s2_test = df_test[available_s2].copy()
    s2_meds = s2_train.median().fillna(0.0)

    X_train_s2 = s2_train.fillna(s2_meds).fillna(0.0).values
    X_test_s2 = s2_test.fillna(s2_meds).fillna(0.0).values

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    pipe_s2 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=42)),
    ])
    pipe_s2.fit(X_train_s2, y_train)
    s2_probs = pipe_s2.predict_proba(X_test_s2)[:, 1]

    # 3. STREAM 3 ALONE: TrOCR Legibility & Linguistic Errors
    print("--- Evaluating Stream 3 Alone (TrOCR Legibility & Error Deltas) ---")
    available_s3 = [c for c in STREAM3_CORE_FEATURES if c in df_train.columns]
    print(f"Available Stream 3 features: {len(available_s3)} features")

    s3_train = df_train[available_s3].copy()
    s3_test = df_test[available_s3].copy()
    s3_meds = s3_train.median().fillna(0.0)

    X_train_s3 = s3_train.fillna(s3_meds).fillna(0.0).values
    X_test_s3 = s3_test.fillna(s3_meds).fillna(0.0).values

    clf_s3 = LogisticRegression(C=0.1, class_weight="balanced", random_state=42)
    clf_s3.fit(X_train_s3, y_train)
    s3_probs = clf_s3.predict_proba(X_test_s3)[:, 1]

    # 4. 2-STREAM COMBINATIONS
    stacked_s1_s2 = 0.60 * s1_probs + 0.40 * s2_probs
    stacked_s1_s3 = 0.70 * s1_probs + 0.30 * s3_probs

    # 5. 3-STREAM LATE STACKING (Balanced: 0.50 S1 + 0.25 S2 + 0.25 S3)
    print("--- Evaluating 3-Stream Late Probability Stacking ---")
    stacked_3stream_balanced = 0.50 * s1_probs + 0.25 * s2_probs + 0.25 * s3_probs
    stacked_3stream_tuned = 0.45 * s1_probs + 0.35 * s2_probs + 0.20 * s3_probs

    # 6. EARLY FUSION (Concatenate all available S1 + S2 + S3 features)
    print("--- Evaluating 3-Stream Early Feature Fusion ---")
    all_fused_features = FULL_PROTOCOL_FEATURES + [
        f for f in available_s2 + available_s3 if f not in FULL_PROTOCOL_FEATURES
    ]
    fused_train = df_train[all_fused_features].copy()
    fused_test = df_test[all_fused_features].copy()
    fused_meds = fused_train.median().fillna(0.0)

    X_train_fused = fused_train.fillna(fused_meds).fillna(0.0).values
    X_test_fused = fused_test.fillna(fused_meds).fillna(0.0).values

    clf_fused = HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.03, min_samples_leaf=4, class_weight="balanced", random_state=42
    )
    clf_fused.fit(X_train_fused, y_train)
    fused_probs = clf_fused.predict_proba(X_test_fused)[:, 1]

    # 7. ROUTED MULTI-STREAM ENSEMBLE (Optimal Grade & Protocol Routing)
    # Grade 3 (Zero Dictation): S1 + S2 (motor spatial & pencil pressure)
    # Grades 4-7 (Matched Protocol): S1 (45%) + S2 (35%) + S3 (20%)
    # Grade 8 (Ballpoint Pens / Partial): S1 (30%) + S3 (70%)
    p_routed = np.zeros(len(df_test))
    for i, g in enumerate(df_test["grade"]):
        if g == 3:
            p_routed[i] = 0.60 * s1_probs[i] + 0.40 * s2_probs[i]
        elif 4 <= g <= 7:
            p_routed[i] = 0.45 * s1_probs[i] + 0.35 * s2_probs[i] + 0.20 * s3_probs[i]
        elif g == 8:
            p_routed[i] = 0.30 * s1_probs[i] + 0.70 * s3_probs[i]

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
        "Stream 2 Alone (Stroke-Width/Pressure)": s2_probs,
        "Stream 3 Alone (TrOCR Legibility)": s3_probs,
        "Late Stacking (0.60 S1 + 0.40 S2)": stacked_s1_s2,
        "Late Stacking (0.70 S1 + 0.30 S3)": stacked_s1_s3,
        "3-Stream Late Stacking (Balanced)": stacked_3stream_balanced,
        "3-Stream Late Stacking (Tuned 45/35/20)": stacked_3stream_tuned,
        "Routed Multi-Stream System (S1 + S2 + S3)": p_routed,
        "Early Fusion (S1 + S2 + S3)": fused_probs,
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
                "sens_at_80_spec": met["sens_at_80_spec"],
                "sens_at_90_spec": met["sens_at_90_spec"],
                "youden_sens": met["youden_sens"],
                "youden_spec": met["youden_spec"],
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
            s80_str = f"{r['sens_at_80_spec']*100:.1f}%" if r["sens_at_80_spec"] is not None else "N/A"
            s90_str = f"{r['sens_at_90_spec']*100:.1f}%" if r["sens_at_90_spec"] is not None else "N/A"
            ydn_str = f"Sens: {r['youden_sens']*100:.1f}%, Spec: {r['youden_spec']*100:.1f}%" if r["youden_sens"] is not None else "N/A"
            print(f"  {r['model']:<38} | AUC: {auc_str} | PRC: {prc_str} | Sens@80%: {s80_str} | Sens@90%: {s90_str} | Youden: ({ydn_str})")

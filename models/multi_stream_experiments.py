"""Unified Multi-Stream Benchmark Engine (Stage 25 Overhaul).

Evaluates the single unified screening architecture across all 215 students:
- Stream 1: Task-Agnostic Spatial Geometry & Intra-Page Volatility
- Stream 2: Writing-Tool-Calibrated Width, Pressure & Micro-Texture
- Stream 3: Domain-Adapted Prompt-Free TrOCR Legibility & Ink-to-Character Density
- Meta-Learner: Learned Logistic/Elastic-Net meta-model fit on inner CV folds.

Zero manual per-grade weight hacks. Zero task-routing dependencies.
Evaluated strictly with 5-repeat x 5-fold Stratified Group K-Fold nested CV.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from features.ocr_engine import OCRTaskResult, extract_prompt_free_trocr_features, get_trocr_cache
from models.cv import impute_fold_features
from features.stroke_width_texture import calibrate_stream2_by_tool

REPORTS_DIR = WORKSPACE_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. STREAM 1: Task-Agnostic Geometry & Intra-Page Volatility (14 features)
STREAM1_FEATURES: List[str] = [
    "components_per_unit_width_median",
    "components_per_unit_width_iqr",
    "components_per_unit_width_p90",
    "components_per_unit_width_late_minus_early",
    "endpoints_per_unit_width_median",
    "endpoints_per_unit_width_iqr",
    "junctions_per_unit_width_median",
    "baseline_rmse_norm_median",
    "baseline_rmse_norm_iqr",
    "gap_cv_median",
    "word_height_cv_median",
    "jerk_proxy_median",
    "tortuosity_median_median",
    "matra_ratio_iqr",
]

# 2. STREAM 2: Tool-Calibrated Width & Pressure (7 features)
STREAM2_FEATURES: List[str] = [
    "is_pencil",
    "pressure_intensity_cv_tool_norm",
    "stroke_width_jitter_tool_norm",
    "stroke_width_bloat_rate_tool_norm",
    "pressure_faint_rate_tool_norm",
    "pressure_heavy_rate_tool_norm",
    "interaction_pencil_pressure",
]

# 3. STREAM 3: Prompt-Free TrOCR Legibility (5 features)
STREAM3_FEATURES: List[str] = [
    "trocr_conf_mean",
    "trocr_conf_p10",
    "trocr_char_entropy_mean",
    "trocr_top2_margin_mean",
    "ink_to_char_density_ratio",
]


def load_multistream_cohort_dataset(
    student_dataset_path: str = "data/datasets/dataset_student_level.csv",
) -> pd.DataFrame:
    """Load unified 215-student dataset and inject prompt-free TrOCR features."""
    p = WORKSPACE_ROOT / student_dataset_path
    df = pd.read_csv(p)

    # Extract prompt-free TrOCR metrics from disk cache
    trocr_cache = get_trocr_cache()
    student_trocr: Dict[str, List[Dict[str, float]]] = {}

    for k, c in trocr_cache.items():
        sid = k.rsplit("_sentence_", 1)[0]
        t_res = OCRTaskResult(
            text=c.get("text", ""),
            tokens=c.get("tokens", []),
            token_confidences=c.get("token_confidences", []),
            mean_confidence=float(c.get("mean_confidence", 0.0)),
            min_confidence=float(c.get("min_confidence", 0.0)),
            std_confidence=float(c.get("std_confidence", 0.0)),
            low_conf_ratio=float(c.get("low_conf_ratio", 1.0)),
            char_count=int(c.get("char_count", 0)),
            word_count=int(c.get("word_count", 0)),
            line_count=int(c.get("line_count", 1)),
        )
        pf = extract_prompt_free_trocr_features(t_res)
        if sid not in student_trocr:
            student_trocr[sid] = []
        student_trocr[sid].append(pf)

    # Aggregate prompt-free TrOCR metrics per student
    trocr_rows = []
    for sid in df["student_id"]:
        entries = student_trocr.get(str(sid), [])
        if len(entries) > 0:
            c_means = [e["trocr_conf_mean"] for e in entries if not np.isnan(e["trocr_conf_mean"])]
            c_p10s = [e["trocr_conf_p10"] for e in entries if not np.isnan(e["trocr_conf_p10"])]
            entropies = [e["trocr_char_entropy_mean"] for e in entries if not np.isnan(e["trocr_char_entropy_mean"])]
            margins = [e["trocr_top2_margin_mean"] for e in entries if not np.isnan(e["trocr_top2_margin_mean"])]

            row = {
                "trocr_conf_mean": float(np.mean(c_means)) if c_means else np.nan,
                "trocr_conf_p10": float(np.min(c_p10s)) if c_p10s else np.nan,
                "trocr_char_entropy_mean": float(np.mean(entropies)) if entropies else np.nan,
                "trocr_top2_margin_mean": float(np.mean(margins)) if margins else np.nan,
            }
        else:
            row = {
                "trocr_conf_mean": np.nan,
                "trocr_conf_p10": np.nan,
                "trocr_char_entropy_mean": np.nan,
                "trocr_top2_margin_mean": np.nan,
            }

        # Physical ink components to character ratio
        comp_med = df.loc[df["student_id"] == sid, "components_per_unit_width_median"].values[0] if "components_per_unit_width_median" in df.columns else np.nan
        conf_m = row["trocr_conf_mean"]
        if not np.isnan(comp_med) and not np.isnan(conf_m):
            row["ink_to_char_density_ratio"] = float(comp_med / max(0.1, conf_m))
        else:
            row["ink_to_char_density_ratio"] = np.nan

        trocr_rows.append(row)

    df_trocr = pd.DataFrame(trocr_rows)
    df = pd.concat([df, df_trocr], axis=1)

    return df


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate AUROC, AUPRC, Brier, and sensitivity at specific operating points."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan, "sens_at_80_spec": np.nan, "sens_at_90_spec": np.nan}

    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr

    v80 = np.where(spec >= 0.80)[0]
    sens_80 = float(np.max(tpr[v80])) if len(v80) > 0 else 0.0

    v90 = np.where(spec >= 0.90)[0]
    sens_90 = float(np.max(tpr[v90])) if len(v90) > 0 else 0.0

    return {
        "auroc": auc,
        "auprc": auprc,
        "brier": brier,
        "sens_at_80_spec": sens_80,
        "sens_at_90_spec": sens_90,
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute empirical 95% percentile confidence intervals via student bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_auroc = []
    boot_auprc = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt_b = y_true[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        yp_b = y_prob[idx]
        boot_auroc.append(roc_auc_score(yt_b, yp_b))
        boot_auprc.append(average_precision_score(yt_b, yp_b))

    return {
        "auroc_ci": (float(np.percentile(boot_auroc, 2.5)), float(np.percentile(boot_auroc, 97.5))) if boot_auroc else (np.nan, np.nan),
        "auprc_ci": (float(np.percentile(boot_auprc, 2.5)), float(np.percentile(boot_auprc, 97.5))) if boot_auprc else (np.nan, np.nan),
    }


def run_unified_215_benchmark() -> pd.DataFrame:
    """Run full 5-repeat x 5-fold cross-validated benchmark across all 215 students."""
    print("Loading unified 215-student cohort with task-agnostic features & prompt-free TrOCR...")
    df = load_multistream_cohort_dataset()
    print(f"Cohort loaded: {len(df)} students ({int((df['label'] == 1).sum())} positives, {int((df['label'] == 0).sum())} controls).")

    folds_path = WORKSPACE_ROOT / "data" / "folds" / "nested_folds.json"
    with open(folds_path, "r", encoding="utf-8") as f:
        folds_data = json.load(f)

    # Feature subsets
    s1_cols = [c for c in STREAM1_FEATURES if c in df.columns]
    s2_cols = [c for c in STREAM2_FEATURES if c in df.columns]
    s3_cols = [c for c in STREAM3_FEATURES if c in df.columns]

    print(f"Features: Stream 1 (Geometry)={len(s1_cols)}, Stream 2 (Tool/Width)={len(s2_cols)}, Stream 3 (TrOCR)={len(s3_cols)}")

    # Containers for out-of-sample predictions across 5 repeats
    n_students = len(df)
    n_repeats = len(folds_data["repeats"])

    all_preds_s1 = np.zeros((n_repeats, n_students))
    all_preds_s2 = np.zeros((n_repeats, n_students))
    all_preds_s3 = np.zeros((n_repeats, n_students))
    all_preds_meta = np.zeros((n_repeats, n_students))
    all_y_true = np.zeros((n_repeats, n_students))

    sid_to_idx = {str(sid): i for i, sid in enumerate(df["student_id"])}

    for rep in folds_data["repeats"]:
        r_id = rep["repeat_id"]
        print(f"\n--- Running Repeat {r_id} (5-Fold Stratified Group Split) ---")

        for fold in rep["outer_folds"]:
            f_id = fold["outer_fold_id"]
            tr_sids = set(fold["train_students"])
            te_sids = set(fold["test_students"])

            tr_mask = df["student_id"].isin(tr_sids).values
            te_mask = df["student_id"].isin(te_sids).values

            df_tr = df[tr_mask].copy()
            df_te = df[te_mask].copy()

            y_tr = df_tr["label"].values.astype(int)
            y_te = df_te["label"].values.astype(int)

            # -------------------------------------------------------------
            # Inner 3-Fold CV on df_tr to generate out-of-fold meta training data
            # -------------------------------------------------------------
            inner_s1_oof = np.zeros(len(df_tr))
            inner_s2_oof = np.zeros(len(df_tr))
            inner_s3_oof = np.zeros(len(df_tr))

            for in_fold in fold["inner_folds"]:
                in_tr_sids = set(in_fold["train_students"])
                in_val_sids = set(in_fold["val_students"])

                in_tr_mask = df_tr["student_id"].isin(in_tr_sids).values
                in_val_mask = df_tr["student_id"].isin(in_val_sids).values

                in_df_tr = df_tr[in_tr_mask]
                in_df_val = df_tr[in_val_mask]
                in_y_tr = in_df_tr["label"].values.astype(int)

                # Stream 1 inner fit
                X_tr_s1, X_val_s1, _ = impute_fold_features(in_df_tr, in_df_val, s1_cols)
                p_s1 = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=42))])
                p_s1.fit(X_tr_s1, in_y_tr)
                inner_s1_oof[in_val_mask] = p_s1.predict_proba(X_val_s1)[:, 1]

                # Stream 2 inner fit
                X_tr_s2, X_val_s2, _ = impute_fold_features(in_df_tr, in_df_val, s2_cols)
                p_s2 = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=42))])
                p_s2.fit(X_tr_s2, in_y_tr)
                inner_s2_oof[in_val_mask] = p_s2.predict_proba(X_val_s2)[:, 1]

                # Stream 3 inner fit
                X_tr_s3, X_val_s3, _ = impute_fold_features(in_df_tr, in_df_val, s3_cols)
                p_s3 = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=42))])
                p_s3.fit(X_tr_s3, in_y_tr)
                inner_s3_oof[in_val_mask] = p_s3.predict_proba(X_val_s3)[:, 1]

            # Fit Meta-Learner purely on inner CV out-of-fold probability vectors
            X_meta_tr = np.column_stack([inner_s1_oof, inner_s2_oof, inner_s3_oof, df_tr["is_pencil"].values])
            meta_learner = LogisticRegression(C=0.5, class_weight="balanced", random_state=42)
            meta_learner.fit(X_meta_tr, y_tr)

            # -------------------------------------------------------------
            # Outer Model Fitting on entire df_tr and Out-of-Sample Test Inference
            # -------------------------------------------------------------
            # Outer Stream 1
            X_tr_s1_full, X_te_s1, _ = impute_fold_features(df_tr, df_te, s1_cols)
            pipe_s1_full = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=42))])
            pipe_s1_full.fit(X_tr_s1_full, y_tr)
            pred_s1 = pipe_s1_full.predict_proba(X_te_s1)[:, 1]

            # Outer Stream 2
            X_tr_s2_full, X_te_s2, _ = impute_fold_features(df_tr, df_te, s2_cols)
            pipe_s2_full = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=42))])
            pipe_s2_full.fit(X_tr_s2_full, y_tr)
            pred_s2 = pipe_s2_full.predict_proba(X_te_s2)[:, 1]

            # Outer Stream 3
            X_tr_s3_full, X_te_s3, _ = impute_fold_features(df_tr, df_te, s3_cols)
            pipe_s3_full = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=42))])
            pipe_s3_full.fit(X_tr_s3_full, y_tr)
            pred_s3 = pipe_s3_full.predict_proba(X_te_s3)[:, 1]

            # Outer Meta-Learner Prediction
            X_meta_te = np.column_stack([pred_s1, pred_s2, pred_s3, df_te["is_pencil"].values])
            pred_meta = meta_learner.predict_proba(X_meta_te)[:, 1]

            # Store predictions
            for sid, p1, p2, p3, pm, yt in zip(df_te["student_id"], pred_s1, pred_s2, pred_s3, pred_meta, y_te):
                idx = sid_to_idx[str(sid)]
                all_preds_s1[r_id, idx] = p1
                all_preds_s2[r_id, idx] = p2
                all_preds_s3[r_id, idx] = p3
                all_preds_meta[r_id, idx] = pm
                all_y_true[r_id, idx] = yt

    # Calculate Benchmark Metrics across all 5 repeats
    results = []

    models = {
        "Stream 1 Alone (Task-Agnostic Geometry)": all_preds_s1,
        "Stream 2 Alone (Tool-Calibrated Width/Pressure)": all_preds_s2,
        "Stream 3 Alone (Prompt-Free TrOCR Legibility)": all_preds_s3,
        "Unified Multi-Stream Meta-Learner (S1 + S2 + S3)": all_preds_meta,
    }

    for name, pred_matrix in models.items():
        rep_aurocs = []
        rep_auprcs = []
        rep_briers = []
        rep_s80 = []
        rep_s90 = []

        for r in range(n_repeats):
            m = compute_metrics(all_y_true[r], pred_matrix[r])
            rep_aurocs.append(m["auroc"])
            rep_auprcs.append(m["auprc"])
            rep_briers.append(m["brier"])
            rep_s80.append(m["sens_at_80_spec"])
            rep_s90.append(m["sens_at_90_spec"])

        # Flattened across all repeats for student bootstrap CI
        y_flat = all_y_true.flatten()
        p_flat = pred_matrix.flatten()
        ci = compute_bootstrap_ci(y_flat, p_flat)

        row = {
            "Model Architecture": name,
            "Mean AUROC": round(float(np.mean(rep_aurocs)), 4),
            "AUROC 95% CI": f"[{ci['auroc_ci'][0]:.3f}, {ci['auroc_ci'][1]:.3f}]",
            "Mean AUPRC": round(float(np.mean(rep_auprcs)), 4),
            "AUPRC 95% CI": f"[{ci['auprc_ci'][0]:.3f}, {ci['auprc_ci'][1]:.3f}]",
            "Mean Brier": round(float(np.mean(rep_briers)), 4),
            "Sens @ 80% Spec": f"{np.mean(rep_s80):.1%}",
            "Sens @ 90% Spec": f"{np.mean(rep_s90):.1%}",
        }
        results.append(row)

    res_df = pd.DataFrame(results)

    # Print honest scoreboard
    print("\n" + "=" * 80)
    print("HONEST UNIFIED 215-STUDENT SCOREBOARD (5-Repeat x 5-Fold Stratified Group CV)")
    print("Zero Grade Hacks | Zero Task Routing | Zero Leakage")
    print("=" * 80)
    print(res_df.to_string(index=False))

    # Save to disk
    out_csv = REPORTS_DIR / "unified_215_benchmark_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nBenchmark results saved to: {out_csv}")

    return res_df


if __name__ == "__main__":
    run_unified_215_benchmark()

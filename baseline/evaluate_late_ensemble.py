"""Late Out-of-Fold Ensemble Evaluation: Stage 4 Tabular + Dual-Script TrOCR Vision MIL.

Blends the 215 out-of-fold probability predictions from:
1. Engine A (Tabular): Task-agnostic physical motor, spacing, and multi-stream features.
2. Engine B (Vision MIL): Dual-script TrOCR handwriting Vision Transformers (both Exp 6 Multimodal and Pure Vision).

Evaluates across alpha in [0.0, 1.0], computing:
- Cohort AUROC & AUPRC
- Sensitivity @ 80% Specificity & Sensitivity @ 90% Specificity
- Analysis of rescued False Negatives (students missed by Tabular but detected by Vision).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
STUDENT_DATASET_PATH = DATA_DIR / "datasets" / "dataset_student_level.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_sensitivity_at_specificity(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_specificity: float = 0.80,
) -> Tuple[int, float, float]:
    """Calculate the number of positives detected, sensitivity, and operating threshold."""
    neg_scores = y_scores[y_true == 0]
    pos_scores = y_scores[y_true == 1]
    threshold = float(np.percentile(neg_scores, target_specificity * 100.0))
    detected = int(np.sum(pos_scores >= threshold))
    sensitivity = float(detected / max(1, len(pos_scores)))
    return detected, sensitivity, threshold


def compute_tabular_oof_predictions(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute out-of-fold predictions on repeat_0_fold for core tabular architectures."""
    from models.train_tabpfn_benchmark import FEATURE_SETS, resolve_features
    from models.multi_stream_experiments import STREAM1_FEATURES, STREAM2_FEATURES, STREAM3_FEATURES

    top30 = resolve_features(df, FEATURE_SETS["MultiPhenotype_Top30"])
    c15 = resolve_features(df, FEATURE_SETS["Clinical_Top15"])

    s1_cols = [c for c in STREAM1_FEATURES if c in df.columns]
    s2_cols = [c for c in STREAM2_FEATURES if c in df.columns]
    s3_cols = [c for c in STREAM3_FEATURES if c in df.columns]
    ms_all_cols = s1_cols + s2_cols + s3_cols

    y = df["label"].values.astype(int)
    n = len(df)

    oof_top30_lr = np.zeros(n)
    oof_top30_rf = np.zeros(n)
    oof_c15_gbm = np.zeros(n)
    oof_multistream = np.zeros(n)

    for f in range(5):
        tr = (df["repeat_0_fold"] != f).values
        te = (df["repeat_0_fold"] == f).values

        # 1. Top30 Regularized LR
        scaler30 = StandardScaler()
        X_tr_30 = scaler30.fit_transform(df.loc[tr, top30].fillna(df.loc[tr, top30].median()).values)
        X_te_30 = scaler30.transform(df.loc[te, top30].fillna(df.loc[tr, top30].median()).values)
        lr30 = LogisticRegression(C=0.08, class_weight="balanced", max_iter=1000, random_state=42)
        lr30.fit(X_tr_30, y[tr])
        oof_top30_lr[te] = lr30.predict_proba(X_te_30)[:, 1]

        # 2. Top30 Random Forest
        rf30 = RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=42)
        rf30.fit(df.loc[tr, top30].fillna(df.loc[tr, top30].median()).values, y[tr])
        oof_top30_rf[te] = rf30.predict_proba(df.loc[te, top30].fillna(df.loc[tr, top30].median()).values)[:, 1]

        # 3. Clinical Top15 HistGBM
        gbm15 = HistGradientBoostingClassifier(max_depth=2, min_samples_leaf=6, learning_rate=0.03, class_weight="balanced", random_state=42)
        gbm15.fit(df.loc[tr, c15].fillna(df.loc[tr, c15].median()).values, y[tr])
        oof_c15_gbm[te] = gbm15.predict_proba(df.loc[te, c15].fillna(df.loc[tr, c15].median()).values)[:, 1]

        # 4. Multi-Stream (S1+S2+S3) LR
        scaler_ms = StandardScaler()
        X_tr_ms = scaler_ms.fit_transform(df.loc[tr, ms_all_cols].fillna(df.loc[tr, ms_all_cols].median()).values)
        X_te_ms = scaler_ms.transform(df.loc[te, ms_all_cols].fillna(df.loc[tr, ms_all_cols].median()).values)
        lr_ms = LogisticRegression(C=0.1, class_weight="balanced", random_state=42)
        lr_ms.fit(X_tr_ms, y[tr])
        oof_multistream[te] = lr_ms.predict_proba(X_te_ms)[:, 1]

    # Stacking Tabular Ensemble (0.5 LR + 0.3 RF + 0.2 GBM)
    oof_tabular_stack = 0.50 * oof_top30_lr + 0.30 * oof_top30_rf + 0.20 * oof_c15_gbm

    return {
        "tabular_top30_lr": oof_top30_lr,
        "tabular_multistream": oof_multistream,
        "tabular_ensemble_stack": oof_tabular_stack,
    }


def evaluate_blend(
    y_true: np.ndarray,
    p_tab: np.ndarray,
    p_vis: np.ndarray,
    student_ids: List[str],
    tab_name: str,
    vis_name: str,
) -> Dict[str, Any]:
    """Sweep alpha from 0.0 to 1.0 and evaluate ensemble performance."""
    sweep_results = []
    alphas = np.linspace(0.0, 1.0, 21)

    # Standalone baseline operating thresholds
    _, _, th_tab80 = calculate_sensitivity_at_specificity(y_true, p_tab, 0.80)
    _, _, th_vis80 = calculate_sensitivity_at_specificity(y_true, p_vis, 0.80)

    for alpha in alphas:
        p_blend = (1.0 - alpha) * p_tab + alpha * p_vis
        auc = float(roc_auc_score(y_true, p_blend))
        auprc = float(average_precision_score(y_true, p_blend))
        brier = float(brier_score_loss(y_true, p_blend))

        det80, sens80, th_blend80 = calculate_sensitivity_at_specificity(y_true, p_blend, 0.80)
        det90, sens90, th_blend90 = calculate_sensitivity_at_specificity(y_true, p_blend, 0.90)

        # Check rescued positives at 80% spec
        rescued = []
        for i, sid in enumerate(student_ids):
            if y_true[i] == 1:
                tab_caught = p_tab[i] >= th_tab80
                blend_caught = p_blend[i] >= th_blend80
                if not tab_caught and blend_caught:
                    rescued.append({
                        "student_id": sid,
                        "p_tabular": float(p_tab[i]),
                        "p_vision": float(p_vis[i]),
                        "p_blend": float(p_blend[i]),
                    })

        sweep_results.append({
            "alpha": round(float(alpha), 2),
            "weight_tabular": round(float(1.0 - alpha), 2),
            "weight_vision": round(float(alpha), 2),
            "auroc": auc,
            "auprc": auprc,
            "brier": brier,
            "det_at_80_spec": det80,
            "sens_at_80_spec": sens80,
            "det_at_90_spec": det90,
            "sens_at_90_spec": sens90,
            "n_rescued": len(rescued),
            "rescued_students": rescued,
        })

    # Find optimal blend by AUROC and AUPRC
    best_auroc_row = max(sweep_results, key=lambda r: r["auroc"])
    best_auprc_row = max(sweep_results, key=lambda r: r["auprc"])

    return {
        "tabular_model": tab_name,
        "vision_model": vis_name,
        "best_auroc_blend": best_auroc_row,
        "best_auprc_blend": best_auprc_row,
        "sweep": sweep_results,
    }


def main():
    logger.info("=================================================================")
    logger.info("Executing Late-Stage Out-of-Fold Ensemble Evaluation")
    logger.info("=================================================================")

    df = pd.read_csv(STUDENT_DATASET_PATH)
    logger.info("Loaded student level dataset (%d students).", len(df))

    # Load Vision OOF Predictions
    exp6_csv = REPORTS_DIR / "exp6_dual_trocr_oof_predictions.csv"
    pure_vis_csv = REPORTS_DIR / "exp6_pure_vision_oof_predictions.csv"

    exp6_df = pd.read_csv(exp6_csv)
    merged = pd.merge(df, exp6_df[["student_id", "pred_prob"]].rename(columns={"pred_prob": "p_vis_multimodal"}), on="student_id")

    if pure_vis_csv.exists():
        pure_df = pd.read_csv(pure_vis_csv)
        merged = pd.merge(merged, pure_df[["student_id", "pred_prob"]].rename(columns={"pred_prob": "p_vis_pure"}), on="student_id")
        logger.info("Loaded Pure Vision TrOCR OOF predictions.")
    else:
        logger.info("Pure Vision CSV not found yet; will evaluate Multimodal TrOCR only.")

    y_true = merged["label"].values.astype(int)
    student_ids = list(merged["student_id"])

    # Compute Tabular OOF
    tab_preds = compute_tabular_oof_predictions(merged)

    all_analyses = {}

    vision_variants = [("Exp6_Multimodal_TrOCR", merged["p_vis_multimodal"].values)]
    if "p_vis_pure" in merged.columns:
        vision_variants.append(("Exp6_Pure_Vision_TrOCR", merged["p_vis_pure"].values))

    for vis_name, p_vis in vision_variants:
        for tab_name, p_tab in tab_preds.items():
            pair_key = f"{tab_name}_x_{vis_name}"
            logger.info("Evaluating Blend: %s", pair_key)
            res = evaluate_blend(y_true, p_tab, p_vis, student_ids, tab_name, vis_name)
            all_analyses[pair_key] = res

    # Export results JSON
    out_json = REPORTS_DIR / "late_ensemble_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    logger.info("Saved ensemble results JSON to %s", out_json)

    # Print summary scoreboard
    print("\n" + "="*90)
    print("LATE-STAGE MULTIMODAL ENSEMBLE SCOREBOARD (All 215 Students Out-of-Fold)")
    print("="*90)
    print(f"{'Ensemble Configuration':40s} | {'Best alpha':10s} | {'AUROC':7s} | {'AUPRC':7s} | {'Sens@80%':9s} | {'Sens@90%':9s} | {'Rescued'}")
    print("-" * 90)

    for k, a in all_analyses.items():
        best = a["best_auroc_blend"]
        print(f"{k:40s} | alpha={best['alpha']:.2f}     | {best['auroc']:.4f} | {best['auprc']:.4f} | {best['det_at_80_spec']}/40 ({best['sens_at_80_spec']*100:.1f}%) | {best['det_at_90_spec']}/40 ({best['sens_at_90_spec']*100:.1f}%) | +{best['n_rescued']} Positives")

    print("="*90)


if __name__ == "__main__":
    main()

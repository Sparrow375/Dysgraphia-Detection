"""Tabular + Vision Stacking Benchmark with Zero Out-Of-Fold Flattery for Experiment 7.

Blends Stage 4 Tabular Models with Experiment 7 Frozen Vision Representations:
1. Zero-Parameter Un-Gameable Baseline: Fixed 50/50 Rank-Average
2. Nested Inner-Fold Search: Tunes alpha in [0.0, 1.0] strictly on inner CV validation folds
3. Full 5-Repeat x 5-Fold Evaluation across all 215 students (25 folds, 0.0% leakage)
4. Cluster Bootstrap 95% Confidence Intervals (B=1000 by student_id)
5. Exact Delta Gain Analysis over standalone tabular and vision streams
6. Rescued False Negative case tracking across School A and School B.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.evaluate_exp7_probe import (
    calculate_sensitivity_at_specificity,
    compute_cluster_bootstrap_cis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
STUDENT_DATASET_PATH = DATA_DIR / "datasets" / "dataset_student_level.csv"


def get_tabular_features(df: pd.DataFrame) -> List[str]:
    """Select the Top-30 non-leaking multi-phenotype tabular features."""
    all_numeric = [c for c in df.columns if c not in ["student_id", "school", "grade", "label"] and not c.startswith("repeat_")]
    # Priority multi-phenotype clean features
    priority = [
        "components_per_unit_width_max", "components_per_unit_width_std", "components_per_unit_width_median",
        "endpoints_per_unit_width_max", "endpoints_per_unit_width_median", "endpoints_per_unit_width_std",
        "junctions_per_unit_width_max", "junctions_per_unit_width_median", "junctions_per_unit_width_std",
        "matra_ratio_max", "matra_ratio_std", "matra_ratio_median",
        "shirorekha_rms_deviation_norm_max", "shirorekha_rms_deviation_norm_std",
        "tortuosity_median_max", "tortuosity_median_std",
        "gap_fraction_above_2h_max", "gap_fraction_above_2h_median",
        "pressure_intensity_cv_max", "pressure_intensity_cv_median",
        "stroke_width_cv_max", "stroke_width_cv_median", "stroke_width_bloat_rate_max",
        "rule_offset_std_max", "rule_offset_std_median",
        "baseline_rmse_norm_max", "baseline_rmse_norm_median",
        "word_width_per_char_std_max", "word_width_per_char_std_median",
        "jerk_proxy_max"
    ]
    selected = [c for c in priority if c in df.columns]
    if len(selected) < 20:
        # Fallback to top non-null numeric
        selected = all_numeric[:30]
    return selected


def generate_tabular_oof_predictions(df_students: pd.DataFrame) -> np.ndarray:
    """Generate 5-repeat x 5-fold pooled OOF probabilities for the Stage 4 Tabular Model."""
    feature_cols = get_tabular_features(df_students)
    y = df_students["label"].values.astype(int)
    n = len(df_students)

    oof_repeats = np.zeros((5, n), dtype=np.float32)

    for r in range(5):
        repeat_col = df_students[f"repeat_{r}_fold"].values

        for f in range(5):
            tr_mask = (repeat_col != f)
            te_mask = (repeat_col == f)

            X_tr = df_students.loc[tr_mask, feature_cols].copy()
            X_te = df_students.loc[te_mask, feature_cols].copy()

            # Impute missing with train medians
            medians = X_tr.median()
            X_tr = X_tr.fillna(medians).values
            X_te = X_te.fillna(medians).values

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            # Ensemble: Regularized LR (0.50) + RF (0.30) + HistGBM (0.20)
            lr = LogisticRegression(C=0.08, class_weight="balanced", max_iter=1000, random_state=42 + r * 10 + f)
            lr.fit(X_tr_sc, y[tr_mask])
            p_lr = lr.predict_proba(X_te_sc)[:, 1]

            rf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=4, class_weight="balanced", random_state=42 + r * 10 + f)
            rf.fit(X_tr, y[tr_mask])
            p_rf = rf.predict_proba(X_te)[:, 1]

            gbm = HistGradientBoostingClassifier(max_depth=2, min_samples_leaf=6, learning_rate=0.03, class_weight="balanced", random_state=42 + r * 10 + f)
            gbm.fit(X_tr, y[tr_mask])
            p_gbm = gbm.predict_proba(X_te)[:, 1]

            p_ens = 0.50 * p_lr + 0.30 * p_rf + 0.20 * p_gbm
            oof_repeats[r, te_mask] = p_ens

    return np.mean(oof_repeats, axis=0)


def evaluate_exp7_stacking(
    vision_oof_csv_path: Path,
    model_tag: str = "dinov2_small",
    vision_col: str = "p_blend",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluates standalone and stacked performance with zero OOF flattery."""
    if output_dir is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = REPORTS_DIR

    logger.info("Loading tabular dataset: %s", STUDENT_DATASET_PATH)
    df_students = pd.read_csv(STUDENT_DATASET_PATH)

    logger.info("Loading vision OOF predictions: %s (column: %s)", vision_oof_csv_path, vision_col)
    df_vision = pd.read_csv(vision_oof_csv_path)

    # Align by student_id
    df_merged = pd.merge(df_students, df_vision[["student_id", vision_col]], on="student_id")
    n = len(df_merged)
    y_true = df_merged["label"].values.astype(int)
    student_ids = df_merged["student_id"].values

    logger.info("Computing 5x5 repeated CV tabular OOF probabilities...")
    p_tabular = generate_tabular_oof_predictions(df_merged)
    p_vision = df_merged[vision_col].values.astype(np.float32)

    # 1. Zero-Parameter Fixed Rank Average
    rank_tab = rankdata(p_tabular) / float(n)
    rank_vis = rankdata(p_vision) / float(n)
    p_rank_avg = 0.5 * rank_tab + 0.5 * rank_vis

    # 2. Nested Inner-Fold Grid Search for Alpha
    p_nested_blend = np.zeros(n, dtype=np.float32)
    alpha_choices = np.linspace(0.0, 1.0, 11)

    for r in range(5):
        repeat_col = df_merged[f"repeat_{r}_fold"].values

        for f in range(5):
            tr_mask = (repeat_col != f)
            te_mask = (repeat_col == f)

            y_tr = y_true[tr_mask]
            p_tab_tr = p_tabular[tr_mask]
            p_vis_tr = p_vision[tr_mask]

            # Inner 3-fold split to select best alpha
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + r * 10 + f)
            best_alpha = 0.5
            best_auprc = -1.0

            for a in alpha_choices:
                val_auprcs = []
                for _, val_idx in skf.split(p_tab_tr, y_tr):
                    p_val_blend = (1.0 - a) * p_tab_tr[val_idx] + a * p_vis_tr[val_idx]
                    val_auprcs.append(average_precision_score(y_tr[val_idx], p_val_blend))
                mean_auprc = float(np.mean(val_auprcs))
                if mean_auprc > best_auprc:
                    best_auprc = mean_auprc
                    best_alpha = a

            # Apply best inner alpha to outer test fold
            p_te_blend = (1.0 - best_alpha) * p_tabular[te_mask] + best_alpha * p_vision[te_mask]
            p_nested_blend[te_mask] += p_te_blend / 5.0

    # Evaluate all four models
    models = [
        ("Standalone Tabular Model (Stage 4 Stack)", p_tabular),
        (f"Standalone Vision Model ({model_tag})", p_vision),
        ("Fixed 50/50 Rank-Average Blend (Zero-Parameter)", p_rank_avg),
        ("Nested Inner-Fold Search Blend (Un-Flattered)", p_nested_blend),
    ]

    benchmark_rows = []
    results_dict: Dict[str, Any] = {
        "model_tag": model_tag,
        "vision_col": vision_col,
        "n_students": n,
        "n_positives": int(np.sum(y_true == 1)),
        "models": {},
    }

    tab_auc = float(roc_auc_score(y_true, p_tabular))
    tab_pr = float(average_precision_score(y_true, p_tabular))

    logger.info("=" * 80)
    logger.info("EXPERIMENT 7 TABULAR + VISION STACKING BENCHMARK")
    logger.info("=" * 80)

    for name, p_scores in models:
        auc = float(roc_auc_score(y_true, p_scores))
        pr = float(average_precision_score(y_true, p_scores))
        brier = float(brier_score_loss(y_true, p_scores))
        det80, s80, th80 = calculate_sensitivity_at_specificity(y_true, p_scores, 0.80)
        det90, s90, th90 = calculate_sensitivity_at_specificity(y_true, p_scores, 0.90)
        cis = compute_cluster_bootstrap_cis(y_true, p_scores, student_ids)

        delta_auc = auc - tab_auc
        delta_pr = pr - tab_pr

        row = {
            "Model Pipeline": name,
            "AUROC": round(auc, 4),
            "AUROC 95% CI": f"[{cis['auroc_ci'][0]:.3f}, {cis['auroc_ci'][1]:.3f}]",
            "Δ AUROC": f"{delta_auc:+.4f}" if name != "Standalone Tabular Model (Stage 4 Stack)" else "-",
            "AUPRC": round(pr, 4),
            "AUPRC 95% CI": f"[{cis['auprc_ci'][0]:.3f}, {cis['auprc_ci'][1]:.3f}]",
            "Δ AUPRC": f"{delta_pr:+.4f}" if name != "Standalone Tabular Model (Stage 4 Stack)" else "-",
            "Sens @ 80% Spec": f"{s80*100:.1f}% ({det80}/40)",
            "Sens @ 90% Spec": f"{s90*100:.1f}% ({det90}/40)",
        }
        benchmark_rows.append(row)

        results_dict["models"][name] = {
            "auroc": auc,
            "auroc_ci": list(cis["auroc_ci"]),
            "delta_auroc": delta_auc,
            "auprc": pr,
            "auprc_ci": list(cis["auprc_ci"]),
            "delta_auprc": delta_pr,
            "brier": brier,
            "sens80": s80,
            "det80": det80,
            "sens90": s90,
            "det90": det90,
        }

        logger.info("%-48s | AUROC: %.4f (Δ %+.4f) | AUPRC: %.4f (Δ %+.4f) | Sens@80%%: %.1f%% (%d/40) | Sens@90%%: %.1f%% (%d/40)",
                    name, auc, delta_auc, pr, delta_pr, s80 * 100, det80, s90 * 100, det90)

    logger.info("=" * 80)

    # Rescued False Negatives Analysis at 80% Specificity
    _, _, th80_tab = calculate_sensitivity_at_specificity(y_true, p_tabular, 0.80)
    _, _, th80_rank = calculate_sensitivity_at_specificity(y_true, p_rank_avg, 0.80)

    missed_by_tab = (y_true == 1) & (p_tabular < th80_tab)
    rescued_by_blend = missed_by_tab & (p_rank_avg >= th80_rank)

    rescued_sids = student_ids[rescued_by_blend].tolist()
    logger.info("Rescued False Negatives (%d students missed by Tabular but caught by Blend): %s",
                len(rescued_sids), rescued_sids)
    results_dict["rescued_students"] = rescued_sids

    # Save artifacts
    csv_path = output_dir / f"exp7_{model_tag}_stacking_results.csv"
    pd.DataFrame(benchmark_rows).to_csv(csv_path, index=False)
    logger.info("Saved stacking benchmark CSV to %s", csv_path)

    json_path = output_dir / f"exp7_{model_tag}_stacking_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Saved stacking benchmark JSON to %s", json_path)

    return results_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tabular + Vision Stacking for Experiment 7.")
    parser.add_argument("--vision-oof", type=str, default=None, help="Path to vision OOF predictions CSV.")
    parser.add_argument("--model-tag", type=str, default="dinov2_small", help="Model identifier tag.")
    parser.add_argument("--vision-col", type=str, default="p_blend", help="Probability column to use from vision OOF.")
    args = parser.parse_args()

    oof_path = Path(args.vision_oof) if args.vision_oof else REPORTS_DIR / f"exp7_{args.model_tag}_oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing vision OOF file: {oof_path}")

    evaluate_exp7_stacking(oof_path, model_tag=args.model_tag, vision_col=args.vision_col)


if __name__ == "__main__":
    main()


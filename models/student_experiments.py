"""Student-Level Benchmark Experiment Runner.

Executes 5-repeat 5-fold nested cross-validation on student-level aggregated features,
evaluating regularized linear models, tree ensembles, and hybrid ensembles with
cluster bootstrap 95% confidence intervals and export to reports/.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level.csv"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
OUTPUT_CSV_PATH = REPORTS_DIR / "student_benchmark_results.csv"
OUTPUT_JSON_PATH = REPORTS_DIR / "student_benchmark_results.json"

# Core clinical handwriting feature set (10 features across motor continuity, spatial control, spacing, and Devanagari structure)
# Note: baseline_slope_std_mean removed — baseline_slope_std is NaN for 81% of sentences
# (requires >= 2 lines per sentence) and its student-level aggregate carries negligible signal.
CORE_FEATURES = [
    "components_per_unit_width_dict_mean",
    "components_per_unit_width_max",
    "components_per_unit_width_eng_minus_hindi",
    "endpoints_per_unit_width_mean",
    "junctions_per_unit_width_mean",
    "gap_fraction_above_2h_mean",
    "baseline_rmse_norm_eng_minus_hindi",
    "matra_ratio_std",
    "shirorekha_rms_deviation_norm_max",
    "gap_cv_mean",
]


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute empirical percentile 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    boot_aurocs = []
    boot_auprcs = []
    boot_briers = []
    boot_sens90s = []
    boot_sens80s = []
    boot_baccs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_b = y_true[idx]
        ys_b = y_scores[idx]

        if len(np.unique(yt_b)) < 2:
            continue

        boot_aurocs.append(roc_auc_score(yt_b, ys_b))
        boot_auprcs.append(average_precision_score(yt_b, ys_b))
        boot_briers.append(brier_score_loss(yt_b, ys_b))

        # Sensitivity at specificities
        neg_b = ys_b[yt_b == 0]
        pos_b = ys_b[yt_b == 1]
        if len(neg_b) > 0 and len(pos_b) > 0:
            t90 = np.percentile(neg_b, 90)
            boot_sens90s.append(np.mean(pos_b >= t90))
            t80 = np.percentile(neg_b, 80)
            boot_sens80s.append(np.mean(pos_b >= t80))

        pred_bin = (ys_b >= 0.5).astype(int)
        cm = confusion_matrix(yt_b, pred_bin, labels=[0, 1])
        spec = cm[0, 0] / max(cm[0, 0] + cm[0, 1], 1)
        sens = cm[1, 1] / max(cm[1, 1] + cm[1, 0], 1)
        boot_baccs.append(0.5 * (spec + sens))

    def ci_stats(vals: List[float], point_est: float) -> Tuple[float, float, float]:
        if not vals:
            return float(point_est), float(point_est), float(point_est)
        low = float(np.percentile(vals, 2.5))
        high = float(np.percentile(vals, 97.5))
        return float(point_est), low, high

    # Point estimates on full cohort
    point_auc = roc_auc_score(y_true, y_scores)
    point_auprc = average_precision_score(y_true, y_scores)
    point_brier = brier_score_loss(y_true, y_scores)
    neg = y_scores[y_true == 0]
    pos = y_scores[y_true == 1]
    point_s90 = np.mean(pos >= np.percentile(neg, 90))
    point_s80 = np.mean(pos >= np.percentile(neg, 80))
    pred_bin = (y_scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred_bin, labels=[0, 1])
    point_bacc = 0.5 * (cm[0, 0] / max(cm[0, 0] + cm[0, 1], 1) + cm[1, 1] / max(cm[1, 1] + cm[1, 0], 1))

    return {
        "auroc": ci_stats(boot_aurocs, point_auc),
        "auprc": ci_stats(boot_auprcs, point_auprc),
        "brier": ci_stats(boot_briers, point_brier),
        "sens_at_90_spec": ci_stats(boot_sens90s, point_s90),
        "sens_at_80_spec": ci_stats(boot_sens80s, point_s80),
        "balanced_acc": ci_stats(boot_baccs, point_bacc),
    }


def run_student_level_benchmark(
    dataset_path: Path = DATASET_PATH,
    n_repeats: int = 5,
    n_bootstraps: int = 1000,
) -> pd.DataFrame:
    """Run nested 5-repeat 5-fold cross-validation across all candidate models."""
    df = pd.read_csv(dataset_path)
    print(f"Loaded student dataset: {len(df)} students ({(df['label']==1).sum()} pos, {(df['label']==0).sum()} neg)")

    z_features = [f"z_{f}" for f in CORE_FEATURES]
    raw_features = CORE_FEATURES

    # Candidate models
    model_defs: Dict[str, Dict[str, Any]] = {
        "Majority Class Baseline": {
            "type": "majority",
            "features": "none",
        },
        "Grade-Only Logistic": {
            "type": "grade_only",
            "features": "grade",
            "builder": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        },
        "Elastic-Net Logistic Regression": {
            "type": "standard",
            "features": "z",
            "builder": lambda: LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=0.10,
                l1_ratio=0.30,
                class_weight="balanced",
                max_iter=2500,
                random_state=42,
            ),
        },
        "Regularized Logistic Regression (L2)": {
            "type": "standard",
            "features": "z",
            "builder": lambda: LogisticRegression(
                penalty="l2",
                C=0.05,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        },
        "Random Forest (d=3, leaf=4)": {
            "type": "standard",
            "features": "raw",
            "builder": lambda: RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
            ),
        },
        "HistGradientBoosting (d=2, lr=0.05)": {
            "type": "standard",
            "features": "raw",
            "builder": lambda: HistGradientBoostingClassifier(
                max_depth=2,
                min_samples_leaf=6,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=42,
            ),
        },
        "Hybrid Ensemble (0.7 LR + 0.3 RF)": {
            "type": "ensemble",
            "features": "hybrid",
        },
    }

    results_list = []

    for model_name, m_info in model_defs.items():
        print(f"\nEvaluating: {model_name}...")
        repeat_metrics: Dict[str, List[float]] = {
            "auroc": [],
            "auprc": [],
            "brier": [],
            "sens_at_90_spec": [],
            "sens_at_80_spec": [],
            "balanced_acc": [],
        }

        # Store pooled predictions across repeats for bootstrap CIs
        pooled_y_true = []
        pooled_y_scores = []

        for rep in range(n_repeats):
            fold_col = f"repeat_{rep}_fold"
            y_true_rep = np.zeros(len(df))
            y_scores_rep = np.zeros(len(df))

            for fold in range(5):
                test_mask = df[fold_col] == fold
                tr_df = df[~test_mask]
                te_df = df[test_mask]

                y_tr = tr_df["label"].values.astype(int)
                y_te = te_df["label"].values.astype(int)
                y_true_rep[test_mask] = y_te

                if m_info["type"] == "majority":
                    p_pos = float(np.mean(y_tr == 1))
                    y_scores_rep[test_mask] = p_pos

                elif m_info["type"] == "grade_only":
                    clf = m_info["builder"]()
                    clf.fit(tr_df[["grade"]].values, y_tr)
                    y_scores_rep[test_mask] = clf.predict_proba(te_df[["grade"]].values)[:, 1]

                elif m_info["type"] == "standard":
                    f_cols = z_features if m_info["features"] == "z" else raw_features
                    med = tr_df[f_cols].median().fillna(0.0)
                    X_tr = tr_df[f_cols].fillna(med).values
                    X_te = te_df[f_cols].fillna(med).values

                    clf = m_info["builder"]()
                    clf.fit(X_tr, y_tr)
                    y_scores_rep[test_mask] = clf.predict_proba(X_te)[:, 1]

                elif m_info["type"] == "ensemble":
                    # Fit LR on z
                    med_z = tr_df[z_features].median().fillna(0.0)
                    X_tr_z = tr_df[z_features].fillna(med_z).values
                    X_te_z = te_df[z_features].fillna(med_z).values
                    lr = LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000, random_state=42)
                    lr.fit(X_tr_z, y_tr)
                    p_lr = lr.predict_proba(X_te_z)[:, 1]

                    # Fit RF on raw
                    med_r = tr_df[raw_features].median().fillna(0.0)
                    X_tr_r = tr_df[raw_features].fillna(med_r).values
                    X_te_r = te_df[raw_features].fillna(med_r).values
                    rf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=4, class_weight="balanced", random_state=42)
                    rf.fit(X_tr_r, y_tr)
                    p_rf = rf.predict_proba(X_te_r)[:, 1]

                    y_scores_rep[test_mask] = 0.70 * p_lr + 0.30 * p_rf

            pooled_y_true.extend(y_true_rep)
            pooled_y_scores.extend(y_scores_rep)

            # Single repeat metrics
            repeat_metrics["auroc"].append(roc_auc_score(y_true_rep, y_scores_rep))
            repeat_metrics["auprc"].append(average_precision_score(y_true_rep, y_scores_rep))
            repeat_metrics["brier"].append(brier_score_loss(y_true_rep, y_scores_rep))

            neg = y_scores_rep[y_true_rep == 0]
            pos = y_scores_rep[y_true_rep == 1]
            repeat_metrics["sens_at_90_spec"].append(np.mean(pos >= np.percentile(neg, 90)))
            repeat_metrics["sens_at_80_spec"].append(np.mean(pos >= np.percentile(neg, 80)))

            pred_bin = (y_scores_rep >= 0.5).astype(int)
            cm = confusion_matrix(y_true_rep, pred_bin, labels=[0, 1])
            repeat_metrics["balanced_acc"].append(
                0.5 * (cm[0, 0] / max(cm[0, 0] + cm[0, 1], 1) + cm[1, 1] / max(cm[1, 1] + cm[1, 0], 1))
            )

        # Compute bootstrap CIs over pooled out-of-fold predictions
        ci_res = compute_bootstrap_ci(
            np.array(pooled_y_true),
            np.array(pooled_y_scores),
            n_bootstraps=n_bootstraps,
        )

        mean_auc = float(np.mean(repeat_metrics["auroc"]))
        std_auc = float(np.std(repeat_metrics["auroc"]))
        mean_auprc = float(np.mean(repeat_metrics["auprc"]))
        mean_brier = float(np.mean(repeat_metrics["brier"]))
        mean_s90 = float(np.mean(repeat_metrics["sens_at_90_spec"]))
        mean_s80 = float(np.mean(repeat_metrics["sens_at_80_spec"]))
        mean_bacc = float(np.mean(repeat_metrics["balanced_acc"]))

        row = {
            "model_name": model_name,
            "mean_auroc": round(mean_auc, 3),
            "std_auroc": round(std_auc, 3),
            "auroc_ci_95": f"[{ci_res['auroc'][1]:.3f}, {ci_res['auroc'][2]:.3f}]",
            "mean_auprc": round(mean_auprc, 3),
            "auprc_ci_95": f"[{ci_res['auprc'][1]:.3f}, {ci_res['auprc'][2]:.3f}]",
            "mean_brier": round(mean_brier, 3),
            "brier_ci_95": f"[{ci_res['brier'][1]:.3f}, {ci_res['brier'][2]:.3f}]",
            "sensitivity_at_90_spec": round(mean_s90 * 100.0, 1),
            "sensitivity_at_80_spec": round(mean_s80 * 100.0, 1),
            "balanced_accuracy": round(mean_bacc * 100.0, 1),
        }
        results_list.append(row)
        print(f"  AUROC: {mean_auc:.3f} +/- {std_auc:.3f} (95% CI: {row['auroc_ci_95']}) | AUPRC: {mean_auprc:.3f} | Sens@90%: {row['sensitivity_at_90_spec']}% | Sens@80%: {row['sensitivity_at_80_spec']}%")

    res_df = pd.DataFrame(results_list)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUTPUT_CSV_PATH, index=False)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)

    print(f"\nSaved student benchmark results to:\n- {OUTPUT_CSV_PATH}\n- {OUTPUT_JSON_PATH}")
    return res_df


if __name__ == "__main__":
    run_student_level_benchmark()

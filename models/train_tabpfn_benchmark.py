"""Strategy D: TabPFN (Prior-Data Fitted Network) Benchmark Runner.

Evaluates TabPFN foundation transformer for tabular data on the complete
215-student cohort (School A + School B, 40 positives, 175 controls) across
locked 5-repeat x 5-fold cross-validation.

Features evaluated:
1. Curated 5 Core Features (motor fragmentation, Devanagari headline, matra)
2. Clinical Top-15 (spacing fatigue, stroke tremor, pressure faint rate, baseline drift)
3. Expanded Top-30 Multi-Phenotype Set

Usage:
    python -m models.train_tabpfn_benchmark [--token YOUR_TABPFN_TOKEN] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATASET_PATH = DATA_DIR / "datasets" / "dataset_student_level.csv"
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Curated Feature Sets
FEATURE_SETS: Dict[str, List[str]] = {
    "Curated_5": [
        "components_per_unit_width_max",
        "components_per_unit_width_std",
        "components_per_unit_width_eng_minus_hindi",
        "matra_ratio_max",
        "shirorekha_rms_deviation_norm_max",
    ],
    "Clinical_Top15": [
        "gap_fraction_above_2h_late_minus_early",
        "kappa_sign_changes_per_h_iqr",
        "pressure_faint_rate_tool_norm",
        "baseline_rmse_norm_std",
        "interaction_pen_bloat",
        "component_height_cv_iqr",
        "components_per_unit_width_iqr",
        "erasure_patch_ratio",
        "slant_mean_deg_std",
        "gap_fraction_above_2h_std",
        "smudge_halo_darkness_max",
        "pressure_intensity_cv_median",
        "baseline_slope_mean_late_minus_early",
        "tangent_variance_short_wavelength_iqr",
        "gap_cv_p90",
    ],
    "MultiPhenotype_Top30": [
        "components_per_unit_width_max",
        "components_per_unit_width_std",
        "components_per_unit_width_iqr",
        "components_per_unit_width_eng_minus_hindi",
        "component_height_cv_iqr",
        "gap_fraction_above_2h_late_minus_early",
        "gap_fraction_above_2h_std",
        "gap_cv_p90",
        "gap_mean_median",
        "kappa_sign_changes_per_h_iqr",
        "tangent_variance_short_wavelength_iqr",
        "tortuosity_median_std",
        "pressure_faint_rate_tool_norm",
        "pressure_intensity_cv_median",
        "interaction_pen_bloat",
        "baseline_rmse_norm_std",
        "baseline_slope_mean_late_minus_early",
        "slant_mean_deg_std",
        "slant_mean_deg_late_minus_early",
        "matra_ratio_max",
        "matra_ratio_iqr",
        "shirorekha_rms_deviation_norm_max",
        "shirorekha_breaks_per_word_max",
        "erasure_patch_ratio",
        "erasure_patch_ratio_max",
        "smudge_halo_darkness_max",
        "endpoints_per_unit_width_std",
        "junctions_per_unit_width_std",
        "stroke_width_bloat_rate_median",
        "line_count",
    ],
}


def resolve_features(df: pd.DataFrame, feature_list: List[str]) -> List[str]:
    """Resolve feature names, falling back to z-score prefixed columns if needed."""
    resolved = []
    for f in feature_list:
        if f in df.columns:
            resolved.append(f)
        elif f"z_{f}" in df.columns:
            resolved.append(f"z_{f}")
    return resolved


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute standard clinical screening metrics."""
    auc = float(roc_auc_score(y_true, y_score))
    pr = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    sens_80 = float(tpr[np.where(fpr <= 0.20)[0][-1]]) if np.any(fpr <= 0.20) else 0.0
    sens_90 = float(tpr[np.where(fpr <= 0.10)[0][-1]]) if np.any(fpr <= 0.10) else 0.0

    return {
        "auroc": auc,
        "auprc": pr,
        "brier": brier,
        "sens_80": sens_80,
        "sens_90": sens_90,
    }


def run_tabpfn_cv(
    df: pd.DataFrame,
    features: List[str],
    device: str = "cpu",
    n_repeats: int = 5,
    n_folds: int = 5,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Run locked 5x5 cross-validation using TabPFN."""
    from tabpfn import TabPFNClassifier

    X = df[features].fillna(df[features].median()).values
    y = df["label"].values.astype(int)

    all_repeat_oof = []
    repeat_aurocs = []
    repeat_auprcs = []

    for r in range(n_repeats):
        oof_probs = np.zeros(len(df))
        for f in range(n_folds):
            train_mask = (df[f"repeat_{r}_fold"] != f).values
            val_mask = (df[f"repeat_{r}_fold"] == f).values

            clf = TabPFNClassifier(device=device)
            clf.fit(X[train_mask], y[train_mask])
            oof_probs[val_mask] = clf.predict_proba(X[val_mask])[:, 1]

        rep_auc = roc_auc_score(y, oof_probs)
        rep_pr = average_precision_score(y, oof_probs)
        repeat_aurocs.append(rep_auc)
        repeat_auprcs.append(rep_pr)
        all_repeat_oof.append(oof_probs)

    mean_oof = np.mean(all_repeat_oof, axis=0)
    overall_metrics = evaluate_predictions(y, mean_oof)
    overall_metrics["mean_repeat_auroc"] = float(np.mean(repeat_aurocs))
    overall_metrics["std_repeat_auroc"] = float(np.std(repeat_aurocs))
    overall_metrics["mean_repeat_auprc"] = float(np.mean(repeat_auprcs))
    overall_metrics["std_repeat_auprc"] = float(np.std(repeat_auprcs))

    return overall_metrics, mean_oof


def run_baseline_cv(
    df: pd.DataFrame,
    features: List[str],
    model_type: str = "elastic_net",
    n_repeats: int = 5,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Run baseline regularized linear or tree model for direct comparison."""
    X = df[features].fillna(df[features].median()).values
    y = df["label"].values.astype(int)

    all_repeat_oof = []

    for r in range(n_repeats):
        oof_probs = np.zeros(len(df))
        for f in range(n_folds):
            train_mask = (df[f"repeat_{r}_fold"] != f).values
            val_mask = (df[f"repeat_{r}_fold"] == f).values

            if model_type == "elastic_net":
                clf = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=0.10,
                    l1_ratio=0.30,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42 + r * 10 + f,
                )
            elif model_type == "random_forest":
                clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=3,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42 + r * 10 + f,
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            clf.fit(X[train_mask], y[train_mask])
            oof_probs[val_mask] = clf.predict_proba(X[val_mask])[:, 1]

        all_repeat_oof.append(oof_probs)

    mean_oof = np.mean(all_repeat_oof, axis=0)
    return evaluate_predictions(y, mean_oof)


def main():
    parser = argparse.ArgumentParser(description="Run TabPFN Benchmark on Multi-School Cohort.")
    parser.add_argument("--token", type=str, default="", help="TabPFN PriorLabs API token")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--repeats", type=int, default=5, help="Number of CV repeats (default 5)")
    args = parser.parse_args()

    if args.token:
        os.environ["TABPFN_TOKEN"] = args.token
    elif "TABPFN_TOKEN" not in os.environ:
        token_env = os.environ.get("TABPFN_TOKEN", "")
        if not token_env:
            print("NOTE: TABPFN_TOKEN environment variable is not currently set.")
            print("If TabPFN prompts for license acceptance, enter your free API key from https://ux.priorlabs.ai/account")

    if not DATASET_PATH.exists():
        print(f"Dataset missing: {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded student-level dataset: {len(df)} students")
    print(f"Class distribution: {df['label'].value_counts().to_dict()} (0=Control, 1=Dysgraphia)")
    print(f"School distribution: {df['school'].value_counts().to_dict()}\n")

    summary_rows = []

    for fset_name, raw_features in FEATURE_SETS.items():
        active_cols = resolve_features(df, raw_features)
        print(f"\n{'='*70}")
        print(f"FEATURE SET: {fset_name} ({len(active_cols)} active features)")
        print(f"{'='*70}")

        # Baseline: Elastic-Net
        en_metrics = run_baseline_cv(df, active_cols, model_type="elastic_net", n_repeats=args.repeats)
        print(f"  • Elastic-Net LR     : AUROC = {en_metrics['auroc']:.4f} | AUPRC = {en_metrics['auprc']:.4f} | Sens@80% = {en_metrics['sens_80']*100:.1f}% | Sens@90% = {en_metrics['sens_90']*100:.1f}%")
        summary_rows.append({"feature_set": fset_name, "model": "ElasticNet_LR", **en_metrics})

        # Baseline: Random Forest
        rf_metrics = run_baseline_cv(df, active_cols, model_type="random_forest", n_repeats=args.repeats)
        print(f"  • Random Forest (d=3): AUROC = {rf_metrics['auroc']:.4f} | AUPRC = {rf_metrics['auprc']:.4f} | Sens@80% = {rf_metrics['sens_80']*100:.1f}% | Sens@90% = {rf_metrics['sens_90']*100:.1f}%")
        summary_rows.append({"feature_set": fset_name, "model": "RandomForest", **rf_metrics})

        # TabPFN Foundation Model
        try:
            print(f"  • Running TabPFN (Prior-Data Fitted Network, device={args.device})...")
            tab_metrics, oof_probs = run_tabpfn_cv(df, active_cols, device=args.device, n_repeats=args.repeats)
            print(f"  • TabPFN Foundation  : AUROC = {tab_metrics['auroc']:.4f} | AUPRC = {tab_metrics['auprc']:.4f} | Sens@80% = {tab_metrics['sens_80']*100:.1f}% | Sens@90% = {tab_metrics['sens_90']*100:.1f}%")
            summary_rows.append({"feature_set": fset_name, "model": "TabPFN_Foundation", **tab_metrics})

            # Save predictions
            oof_export = pd.DataFrame({
                "student_id": df["student_id"],
                "school": df["school"],
                "grade": df["grade"],
                "label": df["label"],
                f"prob_tabpfn_{fset_name.lower()}": oof_probs,
            })
            oof_path = REPORTS_DIR / f"tabpfn_oof_predictions_{fset_name.lower()}.csv"
            oof_export.to_csv(oof_path, index=False)

        except Exception as e:
            print(f"  [!] TabPFN execution note: {e}")
            if "TabPFNLicenseError" in str(e) or "ux.priorlabs.ai" in str(e):
                print("      -> Action: Sign in for free at https://ux.priorlabs.ai/account and pass --token <KEY>")

    df_summary = pd.DataFrame(summary_rows)
    report_csv = REPORTS_DIR / "strategy_d_tabpfn_benchmark_summary.csv"
    df_summary.to_csv(report_csv, index=False)
    print(f"\n[DONE] Strategy D Benchmark Summary saved to: {report_csv}")


if __name__ == "__main__":
    main()

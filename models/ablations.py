"""Feature Family Ablation and Permutation Importance Analysis.

Evaluates drop-one-family ablations across the 4 core dysgraphia feature families:
1. Motor Fragmentation (components, endpoints, junctions)
2. Spatial Baseline Alignment (slope stability, baseline drift)
3. Word Spacing & Sizing (gap CV, large gap fraction)
4. Devanagari Script Constraints (matra variation, shirorekha RMS deviation)

Also computes permutation feature importance on test folds across 5 CV repeats.
Exports:
- reports/ablation_study_results.csv
- reports/permutation_importance_results.csv
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level.csv"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
OUTPUT_ABLATION_CSV = REPORTS_DIR / "ablation_study_results.csv"
OUTPUT_IMPORTANCE_CSV = REPORTS_DIR / "permutation_importance_results.csv"

# Core clinical feature families
FEATURE_FAMILIES: Dict[str, List[str]] = {
    "Motor Fragmentation": [
        "components_per_unit_width_dict_mean",
        "components_per_unit_width_max",
        "components_per_unit_width_eng_minus_hindi",
        "endpoints_per_unit_width_mean",
        "junctions_per_unit_width_mean",
    ],
    "Spatial Baseline Alignment": [
        "baseline_rmse_norm_eng_minus_hindi",
        "baseline_slope_std_mean",
    ],
    "Word Spacing & Sizing": [
        "gap_fraction_above_2h_mean",
        "gap_cv_mean",
    ],
    "Devanagari Script Constraints": [
        "matra_ratio_std",
        "shirorekha_rms_deviation_norm_max",
    ],
}

ALL_FEATURES = [f for fam in FEATURE_FAMILIES.values() for f in fam]


def run_ablation_study(
    df: pd.DataFrame,
    c_val: float = 0.05,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Run drop-one-family ablation across 5 cross-validation repeats."""
    print("Running Drop-One-Family Feature Ablation Study...")

    experiments = {"Full Model (All Families)": ALL_FEATURES}
    for fam_name, fam_feats in FEATURE_FAMILIES.items():
        ablated_feats = [f for f in ALL_FEATURES if f not in fam_feats]
        experiments[f"Drop: {fam_name}"] = ablated_feats

    results = []
    baseline_auc = 0.0

    for exp_name, feat_list in experiments.items():
        z_feats = [f"z_{f}" for f in feat_list]
        rep_aucs = []
        rep_auprcs = []

        for rep in range(n_repeats):
            fold_col = f"repeat_{rep}_fold"
            y_true = np.zeros(len(df))
            preds = np.zeros(len(df))

            for fold in range(5):
                test_mask = df[fold_col] == fold
                tr_df = df[~test_mask]
                te_df = df[test_mask]

                med = tr_df[z_feats].median().fillna(0.0)
                tr_X = tr_df[z_feats].fillna(med).values
                te_X = te_df[z_feats].fillna(med).values

                clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, random_state=42)
                clf.fit(tr_X, tr_df["label"].values)
                preds[test_mask] = clf.predict_proba(te_X)[:, 1]
                y_true[test_mask] = te_df["label"].values

            rep_aucs.append(roc_auc_score(y_true, preds))
            rep_auprcs.append(average_precision_score(y_true, preds))

        mean_auc = float(np.mean(rep_aucs))
        std_auc = float(np.std(rep_aucs))
        mean_auprc = float(np.mean(rep_auprcs))

        if exp_name == "Full Model (All Families)":
            baseline_auc = mean_auc
            delta_auc = 0.0
        else:
            delta_auc = mean_auc - baseline_auc

        row = {
            "experiment": exp_name,
            "feature_count": len(feat_list),
            "mean_auroc": round(mean_auc, 3),
            "std_auroc": round(std_auc, 3),
            "delta_auroc": round(delta_auc, 3),
            "mean_auprc": round(mean_auprc, 3),
        }
        results.append(row)
        print(f"  {exp_name:<35} | Feats: {len(feat_list):2d} | AUC: {mean_auc:.3f} (Delta: {delta_auc:+.3f}) | AUPRC: {mean_auprc:.3f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_ABLATION_CSV, index=False)
    print(f"Saved ablation study results to: {OUTPUT_ABLATION_CSV}")
    return res_df


def run_permutation_importance(
    df: pd.DataFrame,
    c_val: float = 0.05,
    n_repeats: int = 5,
    n_permutations: int = 20,
) -> pd.DataFrame:
    """Compute out-of-fold permutation feature importance across CV repeats."""
    print("\nComputing Permutation Feature Importances on Out-of-Fold Test Sets...")
    z_all = [f"z_{f}" for f in ALL_FEATURES]
    rng = np.random.RandomState(42)

    importances: Dict[str, List[float]] = {f: [] for f in ALL_FEATURES}

    for rep in range(n_repeats):
        fold_col = f"repeat_{rep}_fold"

        for fold in range(5):
            test_mask = df[fold_col] == fold
            tr_df = df[~test_mask]
            te_df = df[test_mask]

            med = tr_df[z_all].median().fillna(0.0)
            tr_X = tr_df[z_all].fillna(med).values
            te_X_orig = te_df[z_all].fillna(med).values
            y_te = te_df["label"].values.astype(int)

            clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(tr_X, tr_df["label"].values)

            # Baseline AUC on this fold
            base_auc = roc_auc_score(y_te, clf.predict_proba(te_X_orig)[:, 1])

            # Permute each feature column
            for feat_idx, feat_name in enumerate(ALL_FEATURES):
                perm_drops = []
                for _ in range(n_permutations):
                    te_X_perm = te_X_orig.copy()
                    te_X_perm[:, feat_idx] = rng.permutation(te_X_perm[:, feat_idx])
                    perm_auc = roc_auc_score(y_te, clf.predict_proba(te_X_perm)[:, 1])
                    perm_drops.append(base_auc - perm_auc)
                importances[feat_name].append(float(np.mean(perm_drops)))

    imp_rows = []
    for f in ALL_FEATURES:
        mean_drop = float(np.mean(importances[f]))
        std_drop = float(np.std(importances[f]))
        imp_rows.append({
            "feature": f,
            "mean_auc_drop": round(mean_drop, 4),
            "std_auc_drop": round(std_drop, 4),
        })

    imp_df = pd.DataFrame(imp_rows).sort_values(by="mean_auc_drop", ascending=False)
    imp_df.to_csv(OUTPUT_IMPORTANCE_CSV, index=False)
    print(f"Top 5 most important features by permutation test drop:\n{imp_df.head(5).to_string(index=False)}")
    print(f"Saved permutation importance results to: {OUTPUT_IMPORTANCE_CSV}")
    return imp_df


if __name__ == "__main__":
    df_data = pd.read_csv(DATASET_PATH)
    run_ablation_study(df_data)
    run_permutation_importance(df_data)

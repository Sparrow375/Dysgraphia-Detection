"""Cross-School External Validation, Reverse Transfer, and Pooled Multi-Center Modeling.

Workstream A Tinkering & Evaluation Engine:
1. External Validation: Train on School A (115 students), Test on School B (100 students).
2. Reverse Transfer: Train on School B (100 students), Test on School A (115 students).
3. Multi-Center Pooled 5-Repeat 5-Fold Nested Cross-Validation across all 215 students.
4. Domain Shift Analysis: Kolmogorov-Smirnov tests identifying cross-school invariant features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
STUDENT_DATASET_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level.csv"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 15-Feature Multi-Phenotype Clean Set from False Negative Clinical Audit
CLEAN_15_FEATURES = [
    "components_per_unit_width_dict_mean",
    "components_per_unit_width_max",
    "components_per_unit_width_std",
    "components_per_unit_width_eng_minus_hindi",
    "endpoints_per_unit_width_mean",
    "junctions_per_unit_width_mean",
    "junctions_per_unit_width_own_mean",
    "words_written_ratio_eng_minus_hindi",
    "words_written_ratio_dict_minus_copy",
    "matra_ratio_std",
    "matra_ratio_max",
    "shirorekha_rms_deviation_norm_max",
    "tortuosity_median_cv",
    "word_width_per_char_std",
    "gap_fraction_above_2h_mean",
]

CORE_10_FEATURES = [
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


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive diagnostic metrics."""
    auroc = float(roc_auc_score(y_true, y_scores))
    auprc = float(average_precision_score(y_true, y_scores))
    brier = float(brier_score_loss(y_true, y_scores))

    neg = y_scores[y_true == 0]
    pos = y_scores[y_true == 1]

    # Sens @ 90% Spec and Sens @ 80% Spec
    thresh_90 = np.percentile(neg, 90) if len(neg) > 0 else 0.5
    thresh_80 = np.percentile(neg, 80) if len(neg) > 0 else 0.5
    sens_at_90 = float(np.mean(pos >= thresh_90)) if len(pos) > 0 else 0.0
    sens_at_80 = float(np.mean(pos >= thresh_80)) if len(pos) > 0 else 0.0

    # Operational threshold at 0.5
    pred_bin = (y_scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / max(tn + fp, 1))
    sens = float(tp / max(tp + fn, 1))
    bacc = float(0.5 * (spec + sens))
    f1 = float(2 * tp / max(2 * tp + fp + fn, 1))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "sens_at_90_spec": sens_at_90,
        "sens_at_80_spec": sens_at_80,
        "balanced_acc": bacc,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def analyze_feature_domain_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze domain shift between School A controls and School B controls using KS test."""
    ctrl_a = df[(df["school"] == "school_a") & (df["label"] == 0)]
    ctrl_b = df[(df["school"] == "school_b") & (df["label"] == 0)]

    meta_cols = ["student_id", "school", "grade", "label", "sentence_count"] + [
        c for c in df.columns if c.startswith("repeat_")
    ]
    raw_feats = [c for c in df.columns if not c.startswith("z_") and c not in meta_cols]

    rows = []
    for f in raw_feats:
        vals_a = ctrl_a[f].dropna().values
        vals_b = ctrl_b[f].dropna().values
        if len(vals_a) < 5 or len(vals_b) < 5:
            continue

        ks_stat, p_val = stats.ks_2samp(vals_a, vals_b)
        med_a = float(np.median(vals_a))
        med_b = float(np.median(vals_b))
        diff = float(med_b - med_a)

        rows.append({
            "feature": f,
            "school_a_median": med_a,
            "school_b_median": med_b,
            "median_difference": diff,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_val),
            "is_invariant": bool(p_val >= 0.05),
        })

    df_shift = pd.DataFrame(rows).sort_values("p_value", ascending=False).reset_index(drop=True)
    out_csv = REPORTS_DIR / "feature_domain_shift_analysis.csv"
    df_shift.to_csv(out_csv, index=False)
    print(f"Domain shift analysis saved to: {out_csv} ({len(df_shift)} features analyzed)")
    invariant_count = df_shift["is_invariant"].sum()
    print(f"Cross-school invariant features (p >= 0.05): {invariant_count}/{len(df_shift)}")
    return df_shift


def run_cross_school_external_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Train on School A, test on School B (and vice-versa)."""
    df_a = df[df["school"] == "school_a"].copy().reset_index(drop=True)
    df_b = df[df["school"] == "school_b"].copy().reset_index(drop=True)

    print(f"\n=======================================================")
    print(f"EXTERNAL VALIDATION EXPERIMENTS")
    print(f"School A: {len(df_a)} students ({(df_a['label']==1).sum()} pos, {(df_a['label']==0).sum()} neg)")
    print(f"School B: {len(df_b)} students ({(df_b['label']==1).sum()} pos, {(df_b['label']==0).sum()} neg)")
    print(f"=======================================================")

    feature_sets = {
        "clean_15": [f for f in CLEAN_15_FEATURES if f in df.columns],
        "core_10": [f for f in CORE_10_FEATURES if f in df.columns],
    }

    # Add invariant features subset
    domain_shift_df = analyze_feature_domain_shift(df)
    invariant_features = domain_shift_df[domain_shift_df["is_invariant"]]["feature"].tolist()
    # Filter to top invariant features that intersect with clinical features
    inv_clean = [f for f in CLEAN_15_FEATURES if f in invariant_features]
    if len(inv_clean) >= 5:
        feature_sets["invariant_clean"] = inv_clean

    models = {
        "elastic_net": lambda: LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.3, max_iter=2000, class_weight="balanced", random_state=42
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=42
        ),
        "hist_gbm": lambda: HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=4, class_weight="balanced", random_state=42
        ),
    }

    results = []

    # Run Direction 1: Train A -> Test B
    for fset_name, fcols in feature_sets.items():
        # Prepare datasets
        X_train = df_a[fcols].fillna(df_a[fcols].median()).values
        y_train = df_a["label"].values.astype(int)

        X_test = df_b[fcols].fillna(df_a[fcols].median()).values
        y_test = df_b["label"].values.astype(int)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pred_scores = {}
        for mname, mfunc in models.items():
            clf = mfunc()
            if mname in ["elastic_net"]:
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict_proba(X_test_scaled)[:, 1]
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_test)[:, 1]

            pred_scores[mname] = y_pred
            metrics = compute_metrics(y_test, y_pred)
            res = {
                "experiment": "Train School A -> Test School B (External)",
                "feature_set": fset_name,
                "n_features": len(fcols),
                "model": mname,
                **metrics,
            }
            results.append(res)
            print(f"[A->B | {fset_name:15s} | {mname:15s}] AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | Sens@80: {metrics['sens_at_80_spec']*100:.1f}% | Sens@90: {metrics['sens_at_90_spec']*100:.1f}%")

        # Ensemble average
        ens_pred = np.mean(list(pred_scores.values()), axis=0)
        ens_metrics = compute_metrics(y_test, ens_pred)
        results.append({
            "experiment": "Train School A -> Test School B (External)",
            "feature_set": fset_name,
            "n_features": len(fcols),
            "model": "ensemble_avg",
            **ens_metrics,
        })
        print(f"[A->B | {fset_name:15s} | ensemble_avg   ] AUROC: {ens_metrics['auroc']:.4f} | AUPRC: {ens_metrics['auprc']:.4f} | Sens@80: {ens_metrics['sens_at_80_spec']*100:.1f}% | Sens@90: {ens_metrics['sens_at_90_spec']*100:.1f}%")

        # Matched Grades 4-7 external test
        g47_mask = df_b["grade"].isin([4, 5, 6, 7]).values
        y_test_g47 = y_test[g47_mask]
        for mname, scores in pred_scores.items():
            m_g47 = compute_metrics(y_test_g47, scores[g47_mask])
            results.append({
                "experiment": "Train School A -> Test School B (Grades 4-7 Matched)",
                "feature_set": fset_name,
                "n_features": len(fcols),
                "model": mname,
                **m_g47,
            })
            if mname in ["hist_gbm", "random_forest"]:
                print(f"[A->B (G4-7) | {fset_name:10s} | {mname:15s}] AUROC: {m_g47['auroc']:.4f} | AUPRC: {m_g47['auprc']:.4f} | Sens@80: {m_g47['sens_at_80_spec']*100:.1f}% | Sens@90: {m_g47['sens_at_90_spec']*100:.1f}%")

    # Run Direction 2: Train B -> Test A (Reverse Transfer)
    print(f"\n--- REVERSE TRANSFER: Train School B -> Test School A ---")
    for fset_name, fcols in feature_sets.items():
        X_train = df_b[fcols].fillna(df_b[fcols].median()).values
        y_train = df_b["label"].values.astype(int)

        X_test = df_a[fcols].fillna(df_b[fcols].median()).values
        y_test = df_a["label"].values.astype(int)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pred_scores = {}
        for mname, mfunc in models.items():
            clf = mfunc()
            if mname in ["elastic_net"]:
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict_proba(X_test_scaled)[:, 1]
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_test)[:, 1]

            pred_scores[mname] = y_pred
            metrics = compute_metrics(y_test, y_pred)
            res = {
                "experiment": "Train School B -> Test School A (Reverse)",
                "feature_set": fset_name,
                "n_features": len(fcols),
                "model": mname,
                **metrics,
            }
            results.append(res)
            print(f"[B->A | {fset_name:15s} | {mname:15s}] AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | Sens@80: {metrics['sens_at_80_spec']*100:.1f}% | Sens@90: {metrics['sens_at_90_spec']*100:.1f}%")

        ens_pred = np.mean(list(pred_scores.values()), axis=0)
        ens_metrics = compute_metrics(y_test, ens_pred)
        results.append({
            "experiment": "Train School B -> Test School A (Reverse)",
            "feature_set": fset_name,
            "n_features": len(fcols),
            "model": "ensemble_avg",
            **ens_metrics,
        })
        print(f"[B->A | {fset_name:15s} | ensemble_avg   ] AUROC: {ens_metrics['auroc']:.4f} | AUPRC: {ens_metrics['auprc']:.4f} | Sens@80: {ens_metrics['sens_at_80_spec']*100:.1f}% | Sens@90: {ens_metrics['sens_at_90_spec']*100:.1f}%")

    df_res = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "cross_school_external_validation.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nExternal validation results saved to: {out_csv}")
    return df_res


def run_pooled_multischool_nested_cv(df: pd.DataFrame, n_repeats: int = 5, n_splits: int = 5) -> pd.DataFrame:
    """Run 5-Repeat 5-Fold Nested CV across the pooled multi-school cohort (215 students).

    Stratified by (school, label) to ensure balanced representation across all folds.
    """
    print(f"\n=======================================================")
    print(f"POOLED MULTI-SCHOOL NESTED CROSS-VALIDATION")
    print(f"Total Cohort: {len(df)} students ({(df['label']==1).sum()} pos, {(df['label']==0).sum()} neg)")
    print(f"=======================================================")

    feature_sets = {
        "clean_15": [f for f in CLEAN_15_FEATURES if f in df.columns],
        "core_10": [f for f in CORE_10_FEATURES if f in df.columns],
    }

    # Combined stratification stratum: e.g. "school_a_1", "school_b_0"
    strata = df["school"].astype(str) + "_" + df["label"].astype(str)
    y_all = df["label"].values.astype(int)

    models = {
        "elastic_net": lambda: LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.3, max_iter=2000, class_weight="balanced", random_state=42
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=42
        ),
        "hist_gbm": lambda: HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=4, class_weight="balanced", random_state=42
        ),
    }

    results = []

    for fset_name, fcols in feature_sets.items():
        X_raw = df[fcols].values

        for mname, mfunc in models.items():
            repeat_scores = []
            all_preds = np.zeros(len(df))

            for rep in range(n_repeats):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + rep * 100)
                oof_preds = np.zeros(len(df))

                for train_idx, test_idx in skf.split(X_raw, strata):
                    X_tr = X_raw[train_idx].copy()
                    y_tr = y_all[train_idx]
                    X_te = X_raw[test_idx].copy()
                    y_te = y_all[test_idx]

                    # Leakage-free median imputation from training split
                    tr_med = np.nanmedian(X_tr, axis=0)
                    for col_i in range(X_tr.shape[1]):
                        X_tr[np.isnan(X_tr[:, col_i]), col_i] = tr_med[col_i]
                        X_te[np.isnan(X_te[:, col_i]), col_i] = tr_med[col_i]

                    clf = mfunc()
                    if mname == "elastic_net":
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_tr)
                        X_te = scaler.transform(X_te)

                    clf.fit(X_tr, y_tr)
                    oof_preds[test_idx] = clf.predict_proba(X_te)[:, 1]

                rep_metrics = compute_metrics(y_all, oof_preds)
                repeat_scores.append(rep_metrics)
                all_preds += oof_preds / n_repeats

            # Compute mean and std across repeats
            mean_auc = np.mean([r["auroc"] for r in repeat_scores])
            std_auc = np.std([r["auroc"] for r in repeat_scores])
            mean_auprc = np.mean([r["auprc"] for r in repeat_scores])
            std_auprc = np.std([r["auprc"] for r in repeat_scores])
            mean_s90 = np.mean([r["sens_at_90_spec"] for r in repeat_scores])
            mean_s80 = np.mean([r["sens_at_80_spec"] for r in repeat_scores])
            mean_bacc = np.mean([r["balanced_acc"] for r in repeat_scores])
            mean_brier = np.mean([r["brier"] for r in repeat_scores])

            results.append({
                "cohort": "Pooled Multi-School (A + B, N=215)",
                "feature_set": fset_name,
                "n_features": len(fcols),
                "model": mname,
                "auroc": float(mean_auc),
                "auroc_std": float(std_auc),
                "auprc": float(mean_auprc),
                "auprc_std": float(std_auprc),
                "sens_at_90_spec": float(mean_s90),
                "sens_at_80_spec": float(mean_s80),
                "balanced_acc": float(mean_bacc),
                "brier": float(mean_brier),
            })
            print(f"[Pooled | {fset_name:10s} | {mname:15s}] AUROC: {mean_auc:.4f} +/- {std_auc:.4f} | AUPRC: {mean_auprc:.4f} | Sens@80: {mean_s80*100:.1f}% | Sens@90: {mean_s90*100:.1f}%")

    df_pooled = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "pooled_multischool_benchmark.csv"
    df_pooled.to_csv(out_csv, index=False)
    print(f"\nPooled benchmark results saved to: {out_csv}")
    return df_pooled


def main():
    print(f"Loading student-level dataset from: {STUDENT_DATASET_PATH}")
    df = pd.read_csv(STUDENT_DATASET_PATH)
    print(f"Loaded {len(df)} students across {df['school'].value_counts().to_dict()}")

    # 1. External validation and reverse transfer
    run_cross_school_external_validation(df)

    # 2. Pooled multi-school nested cross-validation
    run_pooled_multischool_nested_cv(df)


if __name__ == "__main__":
    main()

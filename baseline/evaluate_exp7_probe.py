"""Locked 5x5 Repeated Cross-Validation Harness & Script-Routed Linear Probe for Experiment 7.

Evaluates frozen feature representations across:
1. Outer 5-Repeat x 5-Fold Stratified Group CV (25 folds, 0.0% student leakage)
2. Inner 3-Fold CV for leakage-free hyperparameter tuning:
   - PCA components K in [16, 24, 32]
   - Regularized LogisticRegression C in [0.01, 0.05, 0.1, 0.5, 1.0] with class_weight='balanced'
3. Two-Tier Script Reporting:
   - Tier 1: Independent Hindi-only model (P_Hindi) and English-only model (P_English)
   - Tier 2: Fused models:
     (a) Decision-level probability average P_blend = 0.5 * P_Hindi + 0.5 * P_English
     (b) Joint concatenated feature model P_joint
4. Cluster Bootstrap 95% Confidence Intervals (B=1000 iterations by student_id):
   - AUROC, AUPRC, Sens @ 80% Spec, Sens @ 90% Spec, Brier Loss
5. Exports structured benchmark artifacts (CSV, JSON, OOF predictions CSV).
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = REPO_ROOT / "reports"


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


def compute_cluster_bootstrap_cis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    student_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute empirical 95% percentile confidence intervals via student cluster bootstrap."""
    rng = np.random.RandomState(seed)
    unique_students = np.unique(student_ids)
    n_students = len(unique_students)

    student_to_indices = {sid: np.where(student_ids == sid)[0] for sid in unique_students}

    boot_aurocs: List[float] = []
    boot_auprcs: List[float] = []
    boot_sens80: List[float] = []
    boot_sens90: List[float] = []
    boot_briers: List[float] = []

    for _ in range(n_bootstrap):
        sampled_students = rng.choice(unique_students, size=n_students, replace=True)
        idx_list = []
        for s in sampled_students:
            idx_list.extend(student_to_indices[s])
        sample_idx = np.array(idx_list)

        b_y = y_true[sample_idx]
        b_p = y_scores[sample_idx]

        if len(np.unique(b_y)) < 2:
            continue

        boot_aurocs.append(roc_auc_score(b_y, b_p))
        boot_auprcs.append(average_precision_score(b_y, b_p))
        boot_briers.append(brier_score_loss(b_y, b_p))

        _, s80, _ = calculate_sensitivity_at_specificity(b_y, b_p, 0.80)
        _, s90, _ = calculate_sensitivity_at_specificity(b_y, b_p, 0.90)
        boot_sens80.append(s80)
        boot_sens90.append(s90)

    def ci(arr: List[float]) -> Tuple[float, float]:
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "auroc_ci": ci(boot_aurocs),
        "auprc_ci": ci(boot_auprcs),
        "sens80_ci": ci(boot_sens80),
        "sens90_ci": ci(boot_sens90),
        "brier_ci": ci(boot_briers),
    }


def tune_and_fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k_candidates: List[int] = [16, 24, 32],
    c_candidates: List[float] = [0.01, 0.05, 0.1, 0.5, 1.0],
    seed: int = 42,
) -> np.ndarray:
    """Tune PCA dimensions and L2 regularizer C via inner 3-fold CV and infer test probabilities."""
    best_score = -1.0
    best_k = k_candidates[0]
    best_c = c_candidates[0]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    # Search grid
    for k in k_candidates:
        if k > X_train.shape[0] or k > X_train.shape[1]:
            continue
        for c in c_candidates:
            val_aurocs = []
            for tr_idx, val_idx in skf.split(X_train, y_train):
                # Standardize
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[tr_idx])
                X_v = scaler.transform(X_train[val_idx])

                # PCA
                pca = PCA(n_components=k, random_state=seed)
                X_tr_pca = pca.fit_transform(X_tr)
                X_v_pca = pca.transform(X_v)

                # Logistic Regression
                lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=seed)
                lr.fit(X_tr_pca, y_train[tr_idx])
                p_val = lr.predict_proba(X_v_pca)[:, 1]

                if len(np.unique(y_train[val_idx])) == 2:
                    val_aurocs.append(roc_auc_score(y_train[val_idx], p_val))

            mean_score = float(np.mean(val_aurocs)) if val_aurocs else 0.0
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
                best_c = c

    # Fit best configuration on full training fold
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)

    pca_final = PCA(n_components=best_k, random_state=seed)
    X_train_pca = pca_final.fit_transform(X_train_scaled)
    X_test_pca = pca_final.transform(X_test_scaled)

    lr_final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=seed)
    lr_final.fit(X_train_pca, y_train)

    test_probs = lr_final.predict_proba(X_test_pca)[:, 1]
    return test_probs


def evaluate_exp7_probe(
    features_npz_path: Path,
    model_tag: str = "dinov2_small",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete 5x5 repeated CV evaluation on cached features."""
    if output_dir is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = REPORTS_DIR

    logger.info("Loading features from %s...", features_npz_path)
    data = np.load(features_npz_path, allow_pickle=True)

    student_ids = data["student_ids"]
    y_true = data["student_labels"].astype(int)
    grades = data["student_grades"].astype(int)
    schools = data["student_schools"]

    hindi_feats = data["hindi_student_features"]
    english_feats = data["english_student_features"]
    joint_feats = data["joint_student_features"]

    n_students = len(student_ids)
    logger.info("Loaded %d students (%d positives, %d controls)",
                n_students, int(np.sum(y_true == 1)), int(np.sum(y_true == 0)))

    # Track out-of-fold probability predictions across 5 repeats
    oof_hindi_repeats = np.zeros((5, n_students), dtype=np.float32)
    oof_english_repeats = np.zeros((5, n_students), dtype=np.float32)
    oof_blend_repeats = np.zeros((5, n_students), dtype=np.float32)
    oof_joint_repeats = np.zeros((5, n_students), dtype=np.float32)

    fold_metrics_summary: List[Dict[str, Any]] = []
    start_time = time.time()

    for r in range(5):
        repeat_fold_col = data[f"repeat_{r}_fold"]

        for f in range(5):
            test_mask = (repeat_fold_col == f)
            train_mask = (repeat_fold_col != f)

            y_tr = y_true[train_mask]
            y_te = y_true[test_mask]

            seed = 42 + r * 10 + f

            # 1. Hindi Model Probe
            X_tr_h = hindi_feats[train_mask]
            X_te_h = hindi_feats[test_mask]
            p_hindi_te = tune_and_fit_linear_probe(X_tr_h, y_tr, X_te_h, seed=seed)
            oof_hindi_repeats[r, test_mask] = p_hindi_te

            # 2. English Model Probe
            X_tr_e = english_feats[train_mask]
            X_te_e = english_feats[test_mask]
            p_eng_te = tune_and_fit_linear_probe(X_tr_e, y_tr, X_te_e, seed=seed)
            oof_english_repeats[r, test_mask] = p_eng_te

            # 3. Decision Blend
            p_blend_te = 0.5 * (p_hindi_te + p_eng_te)
            oof_blend_repeats[r, test_mask] = p_blend_te

            # 4. Joint Concatenated Feature Model
            X_tr_j = joint_feats[train_mask]
            X_te_j = joint_feats[test_mask]
            p_joint_te = tune_and_fit_linear_probe(X_tr_j, y_tr, X_te_j, seed=seed)
            oof_joint_repeats[r, test_mask] = p_joint_te

            # Record fold metrics
            fold_metrics_summary.append({
                "repeat": r,
                "fold": f,
                "n_test": int(np.sum(test_mask)),
                "n_pos": int(np.sum(y_te == 1)),
                "hindi_auroc": float(roc_auc_score(y_te, p_hindi_te)),
                "english_auroc": float(roc_auc_score(y_te, p_eng_te)),
                "blend_auroc": float(roc_auc_score(y_te, p_blend_te)),
                "joint_auroc": float(roc_auc_score(y_te, p_joint_te)),
                "blend_auprc": float(average_precision_score(y_te, p_blend_te)),
                "joint_auprc": float(average_precision_score(y_te, p_joint_te)),
            })

        logger.info("Completed Repeat %d/5 in %.1fs", r + 1, time.time() - start_time)

    # Average probabilities across the 5 repeats for pooled OOF evaluation
    p_hindi_mean = np.mean(oof_hindi_repeats, axis=0)
    p_english_mean = np.mean(oof_english_repeats, axis=0)
    p_blend_mean = np.mean(oof_blend_repeats, axis=0)
    p_joint_mean = np.mean(oof_joint_repeats, axis=0)

    # Compile Comprehensive Metrics
    architectures = [
        ("Hindi Model Alone", p_hindi_mean),
        ("English Model Alone", p_english_mean),
        ("Decision Blend (0.5 Hindi + 0.5 English)", p_blend_mean),
        ("Joint Concatenated Multilingual Model", p_joint_mean),
    ]

    benchmark_rows = []
    full_results_dict: Dict[str, Any] = {
        "model_tag": model_tag,
        "n_students": n_students,
        "n_positives": int(np.sum(y_true == 1)),
        "architectures": {},
        "fold_breakdowns": fold_metrics_summary,
    }

    logger.info("=" * 80)
    logger.info("EXPERIMENT 7 FROZEN PROBE BENCHMARK RESULTS (ALL 215 STUDENTS)")
    logger.info("=" * 80)

    for name, p_scores in architectures:
        auc = float(roc_auc_score(y_true, p_scores))
        pr = float(average_precision_score(y_true, p_scores))
        brier = float(brier_score_loss(y_true, p_scores))

        det80, s80, th80 = calculate_sensitivity_at_specificity(y_true, p_scores, 0.80)
        det90, s90, th90 = calculate_sensitivity_at_specificity(y_true, p_scores, 0.90)

        # Bootstrap CIs
        cis = compute_cluster_bootstrap_cis(y_true, p_scores, student_ids)

        row = {
            "Model Architecture": name,
            "AUROC": round(auc, 4),
            "AUROC 95% CI": f"[{cis['auroc_ci'][0]:.3f}, {cis['auroc_ci'][1]:.3f}]",
            "AUPRC": round(pr, 4),
            "AUPRC 95% CI": f"[{cis['auprc_ci'][0]:.3f}, {cis['auprc_ci'][1]:.3f}]",
            "Brier Loss": round(brier, 4),
            "Sens @ 80% Spec": f"{s80*100:.1f}% ({det80}/40)",
            "Sens @ 90% Spec": f"{s90*100:.1f}% ({det90}/40)",
            "Thresh @ 80%": round(th80, 4),
            "Thresh @ 90%": round(th90, 4),
        }
        benchmark_rows.append(row)

        full_results_dict["architectures"][name] = {
            "auroc": auc,
            "auroc_ci": list(cis["auroc_ci"]),
            "auprc": pr,
            "auprc_ci": list(cis["auprc_ci"]),
            "brier": brier,
            "brier_ci": list(cis["brier_ci"]),
            "sens80": s80,
            "sens80_ci": list(cis["sens80_ci"]),
            "det80": det80,
            "sens90": s90,
            "sens90_ci": list(cis["sens90_ci"]),
            "det90": det90,
        }

        logger.info("%-42s | AUROC: %.4f %s | AUPRC: %.4f %s | Sens@80%%: %.1f%% | Sens@90%%: %.1f%%",
                    name, auc, row["AUROC 95% CI"], pr, row["AUPRC 95% CI"], s80 * 100, s90 * 100)

    logger.info("=" * 80)

    # Save benchmark table
    df_results = pd.DataFrame(benchmark_rows)
    csv_path = output_dir / f"exp7_{model_tag}_probe_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info("Saved benchmark CSV to %s", csv_path)

    json_path = output_dir / f"exp7_{model_tag}_probe_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results_dict, f, indent=2)
    logger.info("Saved benchmark JSON to %s", json_path)

    # Save out-of-fold predictions
    df_oof = pd.DataFrame({
        "student_id": student_ids,
        "school": schools,
        "grade": grades,
        "label": y_true,
        "p_hindi": p_hindi_mean,
        "p_english": p_english_mean,
        "p_blend": p_blend_mean,
        "p_joint": p_joint_mean,
    })
    oof_csv = output_dir / f"exp7_{model_tag}_oof_predictions.csv"
    df_oof.to_csv(oof_csv, index=False)
    logger.info("Saved OOF predictions to %s", oof_csv)

    return full_results_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate Experiment 7 Frozen Probe under 5x5 Repeated CV.")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to pre-cached features .npz.")
    parser.add_argument("--model-tag", type=str, default="dinov2_small",
                        help="Model identifier tag (e.g. dinov2_small or dinov2_base).")
    args = parser.parse_args()

    feat_path = Path(args.features) if args.features else FEATURES_DIR / f"exp7_{args.model_tag}_tiles.npz"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing feature cache: {feat_path}")

    evaluate_exp7_probe(feat_path, model_tag=args.model_tag)


if __name__ == "__main__":
    main()

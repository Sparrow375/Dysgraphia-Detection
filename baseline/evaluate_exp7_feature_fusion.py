"""
Experiment 7: Feature-Level Intermediate Fusion
Combines DINOv2 frozen patch-token PCA components with macro-page tabular z-scores
inside a single unified regularized Logistic Regression (L2) head.

Motivation:
- Local 224x224 tile embeddings capture stroke micro-texture, edge jitter, and graphite dynamics.
- Macro tabular features capture whole-page geometry (baseline slope drift, inter-word gap variability,
  component height variability across lines, page-level erasures).
- Rather than averaging two miscalibrated probability scores at the end (which pulls down AUROC),
  intermediate concatenation into a single regularized head allows the linear model to jointly weight
  local stroke features and global layout features with L2 shrinkage.
"""

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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exp7_feature_fusion")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
STUDENT_DATASET_CSV = DATA_DIR / "datasets" / "dataset_student_level.csv"

# Pre-defined macro feature sets
MACRO_SETS = {
    "macro8": [
        "baseline_slope_mean_late_minus_early",
        "baseline_rmse_norm_std",
        "gap_fraction_above_2h_std",
        "gap_cv_p90",
        "gap_mean_median",
        "components_per_unit_width_std",
        "component_height_cv_iqr",
        "rule_offset_std_median",
    ],
    "macro10": [
        "baseline_slope_mean_late_minus_early",
        "baseline_rmse_norm_std",
        "gap_fraction_above_2h_std",
        "gap_cv_p90",
        "gap_mean_median",
        "components_per_unit_width_std",
        "components_per_unit_width_max",
        "component_height_cv_iqr",
        "rule_offset_std_median",
        "erasure_patch_ratio",
    ],
    "macro12": [
        "baseline_slope_mean_late_minus_early",
        "baseline_rmse_norm_std",
        "gap_fraction_above_2h_std",
        "gap_cv_p90",
        "gap_mean_median",
        "components_per_unit_width_std",
        "components_per_unit_width_max",
        "component_height_cv_iqr",
        "rule_offset_std_median",
        "erasure_patch_ratio",
        "slant_mean_deg_late_minus_early",
        "stroke_width_bloat_rate_median",
    ],
}


def calculate_sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.80
) -> Tuple[float, float, int]:
    """Calculate sensitivity at a target specificity threshold."""
    neg_scores = y_score[y_true == 0]
    pos_scores = y_score[y_true == 1]
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return 0.5, 0.0, 0

    threshold = float(np.percentile(neg_scores, 100.0 * target_spec))
    detected = int(np.sum(pos_scores >= threshold))
    sens = float(detected / len(pos_scores))
    return threshold, sens, detected


def cluster_bootstrap_cis(
    student_ids: np.ndarray,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals clustered by student."""
    rng = np.random.RandomState(seed)
    unique_students = np.unique(student_ids)
    n_students = len(unique_students)

    student_indices = {s: np.where(student_ids == s)[0] for s in unique_students}

    boot_aurocs, boot_auprcs, boot_sens80, boot_sens90, boot_briers = [], [], [], [], []

    for _ in range(n_bootstrap):
        sampled_students = rng.choice(unique_students, size=n_students, replace=True)
        sampled_indices = np.concatenate([student_indices[s] for s in sampled_students])

        b_y = y_true[sampled_indices]
        b_p = y_probs[sampled_indices]

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


def tune_and_fit_fused_head(
    X_train_vis: np.ndarray,
    T_train_tab: Optional[np.ndarray],
    y_train: np.ndarray,
    X_test_vis: np.ndarray,
    T_test_tab: Optional[np.ndarray],
    k_candidates: List[int] = [16, 24, 32],
    c_candidates: List[float] = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    seed: int = 42,
) -> np.ndarray:
    """
    Tune PCA dimensions and L2 regularizer C via inner 3-fold CV.
    Combines visual PCA components with standardized macro-page tabular features.
    """
    best_score = -1.0
    best_k = k_candidates[0]
    best_c = c_candidates[0]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    for k in k_candidates:
        if k > X_train_vis.shape[0] or k > X_train_vis.shape[1]:
            continue
        for c in c_candidates:
            val_aurocs = []
            for tr_idx, val_idx in skf.split(X_train_vis, y_train):
                # 1. Scale and PCA-reduce visual features on train fold
                scaler_vis = StandardScaler()
                X_tr_sc = scaler_vis.fit_transform(X_train_vis[tr_idx])
                X_val_sc = scaler_vis.transform(X_train_vis[val_idx])

                pca = PCA(n_components=k, random_state=seed)
                Z_tr = pca.fit_transform(X_tr_sc)
                Z_val = pca.transform(X_val_sc)

                # 2. Scale tabular features on train fold if present
                if T_train_tab is not None and T_train_tab.shape[1] > 0:
                    scaler_tab = StandardScaler()
                    T_tr_sc = scaler_tab.fit_transform(T_train_tab[tr_idx])
                    T_val_sc = scaler_tab.transform(T_train_tab[val_idx])

                    # Concatenate PCA components and standardized tabular features
                    W_tr = np.hstack([Z_tr, T_tr_sc])
                    W_val = np.hstack([Z_val, T_val_sc])
                else:
                    W_tr = Z_tr
                    W_val = Z_val

                # 3. Fit balanced L2 Logistic Regression
                lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=seed)
                lr.fit(W_tr, y_train[tr_idx])
                p_val = lr.predict_proba(W_val)[:, 1]

                if len(np.unique(y_train[val_idx])) == 2:
                    val_aurocs.append(roc_auc_score(y_train[val_idx], p_val))

            mean_score = float(np.mean(val_aurocs)) if val_aurocs else 0.0
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
                best_c = c

    # Fit best configuration on full training fold
    scaler_vis_final = StandardScaler()
    X_train_sc = scaler_vis_final.fit_transform(X_train_vis)
    X_test_sc = scaler_vis_final.transform(X_test_vis)

    pca_final = PCA(n_components=best_k, random_state=seed)
    Z_train = pca_final.fit_transform(X_train_sc)
    Z_test = pca_final.transform(X_test_sc)

    if T_train_tab is not None and T_train_tab.shape[1] > 0:
        scaler_tab_final = StandardScaler()
        T_train_sc = scaler_tab_final.fit_transform(T_train_tab)
        T_test_sc = scaler_tab_final.transform(T_test_tab)

        W_train = np.hstack([Z_train, T_train_sc])
        W_test = np.hstack([Z_test, T_test_sc])
    else:
        W_train = Z_train
        W_test = Z_test

    lr_final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=seed)
    lr_final.fit(W_train, y_train)

    test_probs = lr_final.predict_proba(W_test)[:, 1]
    return test_probs


def load_macro_tabular_features(student_ids: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Load and align macro tabular features for given student IDs."""
    if not STUDENT_DATASET_CSV.exists():
        raise FileNotFoundError(f"Student tabular dataset not found: {STUDENT_DATASET_CSV}")

    df = pd.read_csv(STUDENT_DATASET_CSV)
    df = df.set_index("student_id")

    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Features not found in dataset_student_level.csv: {missing_cols}")

    # Reindex to match student_ids order exactly
    df_aligned = df.reindex(student_ids)

    # Impute median if any missing
    for col in feature_names:
        if df_aligned[col].isna().any():
            med = df_aligned[col].median()
            df_aligned[col] = df_aligned[col].fillna(med)

    return df_aligned[feature_names].values.astype(np.float32)


def run_feature_fusion_evaluation(
    features_npz_path: Path,
    feature_set_name: str = "macro10",
    model_tag: str = "dinov2_small",
) -> Dict[str, Any]:
    """Execute complete 5x5 repeated CV evaluation for Intermediate Feature Fusion."""
    logger.info("Loading vision features from %s...", features_npz_path)
    data = np.load(features_npz_path, allow_pickle=True)

    student_ids = data["student_ids"]
    y_true = data["student_labels"].astype(int)
    grades = data["student_grades"].astype(int)
    schools = data["student_schools"]

    hindi_feats = data["hindi_student_features"]
    english_feats = data["english_student_features"]
    joint_feats = data["joint_student_features"]

    macro_cols = MACRO_SETS.get(feature_set_name, MACRO_SETS["macro10"])
    logger.info("Using feature set '%s' with %d macro tabular features:", feature_set_name, len(macro_cols))
    for col in macro_cols:
        logger.info("  - %s", col)

    tabular_matrix = load_macro_tabular_features(student_ids, macro_cols)
    logger.info("Tabular matrix shape: %s", tabular_matrix.shape)

    n_students = len(student_ids)
    oof_vision_only = np.zeros((5, n_students), dtype=np.float32)
    oof_fused_hindi = np.zeros((5, n_students), dtype=np.float32)
    oof_fused_english = np.zeros((5, n_students), dtype=np.float32)
    oof_fused_blend = np.zeros((5, n_students), dtype=np.float32)
    oof_fused_joint = np.zeros((5, n_students), dtype=np.float32)

    start_time = time.time()

    for r in range(5):
        repeat_fold_col = data[f"repeat_{r}_fold"]

        for f in range(5):
            test_mask = (repeat_fold_col == f)
            train_mask = (repeat_fold_col != f)

            y_tr = y_true[train_mask]
            y_te = y_true[test_mask]

            seed = 42 + r * 10 + f

            # A. Standalone Vision Joint (Baseline)
            p_vis_joint = tune_and_fit_fused_head(
                joint_feats[train_mask], None, y_tr, joint_feats[test_mask], None, seed=seed
            )
            oof_vision_only[r, test_mask] = p_vis_joint

            # B. Fused Hindi (Hindi Visual PCA + Macro Tabular)
            p_fused_h = tune_and_fit_fused_head(
                hindi_feats[train_mask],
                tabular_matrix[train_mask],
                y_tr,
                hindi_feats[test_mask],
                tabular_matrix[test_mask],
                seed=seed,
            )
            oof_fused_hindi[r, test_mask] = p_fused_h

            # C. Fused English (English Visual PCA + Macro Tabular)
            p_fused_e = tune_and_fit_fused_head(
                english_feats[train_mask],
                tabular_matrix[train_mask],
                y_tr,
                english_feats[test_mask],
                tabular_matrix[test_mask],
                seed=seed,
            )
            oof_fused_english[r, test_mask] = p_fused_e

            # D. Fused Decision Blend: 0.5 * Hindi_Fused + 0.5 * Eng_Fused
            oof_fused_blend[r, test_mask] = 0.5 * (p_fused_h + p_fused_e)

            # E. Fused Joint (Joint Multilingual Visual PCA + Macro Tabular in single head)
            p_fused_j = tune_and_fit_fused_head(
                joint_feats[train_mask],
                tabular_matrix[train_mask],
                y_tr,
                joint_feats[test_mask],
                tabular_matrix[test_mask],
                seed=seed,
            )
            oof_fused_joint[r, test_mask] = p_fused_j

    elapsed = time.time() - start_time
    logger.info("5x5 CV Feature Fusion completed in %.1fs", elapsed)

    # Average across 5 repeats
    mean_p_vis = np.mean(oof_vision_only, axis=0)
    mean_p_h_fused = np.mean(oof_fused_hindi, axis=0)
    mean_p_e_fused = np.mean(oof_fused_english, axis=0)
    mean_p_blend_fused = np.mean(oof_fused_blend, axis=0)
    mean_p_joint_fused = np.mean(oof_fused_joint, axis=0)

    def compute_all_metrics(y_p: np.ndarray, name: str) -> Dict[str, Any]:
        auc = float(roc_auc_score(y_true, y_p))
        ap = float(average_precision_score(y_true, y_p))
        brier = float(brier_score_loss(y_true, y_p))
        th80, s80, det80 = calculate_sensitivity_at_specificity(y_true, y_p, 0.80)
        th90, s90, det90 = calculate_sensitivity_at_specificity(y_true, y_p, 0.90)
        cis = cluster_bootstrap_cis(student_ids, y_true, y_p)
        return {
            "name": name,
            "auroc": auc,
            "auprc": ap,
            "brier": brier,
            "sens_at_80_spec": s80,
            "det_at_80_spec": det80,
            "sens_at_90_spec": s90,
            "det_at_90_spec": det90,
            "auroc_ci": cis["auroc_ci"],
            "auprc_ci": cis["auprc_ci"],
            "sens80_ci": cis["sens80_ci"],
            "sens90_ci": cis["sens90_ci"],
            "brier_ci": cis["brier_ci"],
        }

    results = {
        "model_tag": model_tag,
        "feature_set": feature_set_name,
        "macro_features": macro_cols,
        "vision_only_joint": compute_all_metrics(mean_p_vis, "Vision Only Joint (Baseline)"),
        "fused_hindi": compute_all_metrics(mean_p_h_fused, "Fused Hindi Head (Visual PCA + Tabular)"),
        "fused_english": compute_all_metrics(mean_p_e_fused, "Fused English Head (Visual PCA + Tabular)"),
        "fused_decision_blend": compute_all_metrics(mean_p_blend_fused, "Fused Decision Blend (0.5 H + 0.5 E)"),
        "fused_joint_head": compute_all_metrics(mean_p_joint_fused, "Unified Fused Joint Head (Visual PCA + Tabular)"),
    }

    # Save summary JSON
    out_json = REPORTS_DIR / f"exp7_{model_tag}_{feature_set_name}_fusion_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Save OOF predictions CSV
    df_oof = pd.DataFrame({
        "student_id": student_ids,
        "school": schools,
        "grade": grades,
        "label": y_true,
        "p_vision_only": mean_p_vis,
        "p_fused_hindi": mean_p_h_fused,
        "p_fused_english": mean_p_e_fused,
        "p_fused_blend": mean_p_blend_fused,
        "p_fused_joint": mean_p_joint_fused,
    })
    out_csv = REPORTS_DIR / f"exp7_{model_tag}_{feature_set_name}_fusion_oof.csv"
    df_oof.to_csv(out_csv, index=False)

    # Print summary scoreboard
    logger.info("\n" + "=" * 90)
    logger.info("EXP 7 FEATURE-LEVEL INTERMEDIATE FUSION BENCHMARK (%s, %s)", model_tag, feature_set_name)
    logger.info("=" * 90)
    logger.info(f"{'Model Configuration':<42} | {'AUROC':<7} | {'AUPRC':<7} | {'Sens@80%':<8} | {'Sens@90%':<8}")
    logger.info("-" * 90)
    for k in ["vision_only_joint", "fused_hindi", "fused_english", "fused_decision_blend", "fused_joint_head"]:
        m = results[k]
        logger.info(
            f"{m['name']:<42} | {m['auroc']:.4f}  | {m['auprc']:.4f}  | "
            f"{m['sens_at_80_spec']*100:.1f}% ({m['det_at_80_spec']}/40) | "
            f"{m['sens_at_90_spec']*100:.1f}% ({m['det_at_90_spec']}/40)"
        )
    logger.info("=" * 90)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Exp 7 Feature-Level Intermediate Fusion")
    parser.add_argument(
        "--features",
        type=Path,
        default=DATA_DIR / "features" / "exp7_dinov2_small_tiles.npz",
        help="Path to precomputed vision features NPZ",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="dinov2_small",
        help="Model identifier tag",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="macro10",
        choices=["macro8", "macro10", "macro12"],
        help="Predefined macro tabular feature set",
    )
    args = parser.parse_args()

    run_feature_fusion_evaluation(
        features_npz_path=args.features,
        feature_set_name=args.feature_set,
        model_tag=args.model_tag,
    )


if __name__ == "__main__":
    main()


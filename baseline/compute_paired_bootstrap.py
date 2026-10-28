"""
Stratified Paired Bootstrap Engine for Experiment 7 Model Comparisons.
Evaluates differences in AUROC and AUPRC on the exact same 215 students (B=2000 replicates).
Key comparisons:
1. DINOv2 vs DenseNet201 (Tiled) -> ViT representation effect
2. DenseNet201 Tiled vs DenseNet201 Squashed -> Pure tiling effect
3. Dual-ViT Ensemble vs DINOv2 Alone -> Ensemble statistical significance test
4. DINOv2 vs Stage 4 Tabular Stack -> Vision vs Tabular benchmark
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paired_bootstrap")
DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = REPO_ROOT / "reports"
INDEX_CSV = DATA_DIR / "sentence_crops_dataset" / "sentences_index.csv"
STUDENT_CSV = DATA_DIR / "datasets" / "dataset_student_level.csv"


def extract_oof_probs(
    X: np.ndarray,
    y: np.ndarray,
    df_folds: pd.DataFrame,
    n_components: int = 24,
    use_pca: bool = True,
    C: float = 0.1,
) -> np.ndarray:
    """Run 5x5 repeated Stratified Group CV and return pooled mean OOF probabilities."""
    n_students = len(y)
    oof_probs = np.zeros((5, n_students), dtype=np.float32)

    for r in range(5):
        fold_col = df_folds[f"repeat_{r}_fold"].values
        for f in range(5):
            te_mask = fold_col == f
            tr_mask = fold_col != f

            X_tr, y_tr = X[tr_mask], y[tr_mask]
            X_te = X[te_mask]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            if use_pca:
                k = min(n_components, X_tr_sc.shape[0] - 1, X_tr_sc.shape[1])
                pca = PCA(n_components=k, random_state=42)
                X_tr_proj = pca.fit_transform(X_tr_sc)
                X_te_proj = pca.transform(X_te_sc)
            else:
                X_tr_proj = X_tr_sc
                X_te_proj = X_te_sc

            # Tune C via 3-fold inner CV
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + f)
            best_c = C
            best_val = -1.0
            for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0]:
                sub_aucs = []
                for sub_tr, sub_va in skf.split(X_tr_proj, y_tr):
                    clf = LogisticRegression(C=c_cand, class_weight="balanced", max_iter=1000, random_state=42)
                    clf.fit(X_tr_proj[sub_tr], y_tr[sub_tr])
                    p = clf.predict_proba(X_tr_proj[sub_va])[:, 1]
                    sub_aucs.append(roc_auc_score(y_tr[sub_va], p))
                if np.mean(sub_aucs) > best_val:
                    best_val = np.mean(sub_aucs)
                    best_c = c_cand

            clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_tr_proj, y_tr)
            oof_probs[r, te_mask] = clf.predict_proba(X_te_proj)[:, 1]

    return np.mean(oof_probs, axis=0)


def paired_bootstrap_test(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    name_a: str,
    name_b: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Perform stratified paired bootstrap comparison between Model A and Model B."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    base_auc_a = roc_auc_score(y_true, probs_a)
    base_auc_b = roc_auc_score(y_true, probs_b)
    delta_auc_base = base_auc_a - base_auc_b

    base_ap_a = average_precision_score(y_true, probs_a)
    base_ap_b = average_precision_score(y_true, probs_b)
    delta_ap_base = base_ap_a - base_ap_b

    delta_aucs = []
    delta_aps = []

    for _ in range(n_bootstrap):
        # Stratified sampling with replacement
        boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        y_boot = y_true[boot_idx]
        pa_boot = probs_a[boot_idx]
        pb_boot = probs_b[boot_idx]

        auc_a = roc_auc_score(y_boot, pa_boot)
        auc_b = roc_auc_score(y_boot, pb_boot)
        delta_aucs.append(auc_a - auc_b)

        ap_a = average_precision_score(y_boot, pa_boot)
        ap_b = average_precision_score(y_boot, pb_boot)
        delta_aps.append(ap_a - ap_b)

    delta_aucs = np.array(delta_aucs)
    delta_aps = np.array(delta_aps)

    ci_auc_lower, ci_auc_upper = np.percentile(delta_aucs, [2.5, 97.5])
    ci_ap_lower, ci_ap_upper = np.percentile(delta_aps, [2.5, 97.5])

    # Empirical p-value that Model A is NOT strictly better than Model B (Delta <= 0)
    p_val_auc = float(np.mean(delta_aucs <= 0.0))
    p_val_ap = float(np.mean(delta_aps <= 0.0))

    logger.info("=" * 70)
    logger.info("PAIRED BOOTSTRAP: %s vs. %s", name_a, name_b)
    logger.info("  %s AUROC: %.4f | %s AUROC: %.4f", name_a, base_auc_a, name_b, base_auc_b)
    logger.info("  Δ AUROC: %+.4f [95%% CI: %+.4f, %+.4f] (p = %.4f)", delta_auc_base, ci_auc_lower, ci_auc_upper, p_val_auc)
    logger.info("  Δ AUPRC: %+.4f [95%% CI: %+.4f, %+.4f] (p = %.4f)", delta_ap_base, ci_ap_lower, ci_ap_upper, p_val_ap)
    logger.info("=" * 70)

    return {
        "model_a": name_a,
        "model_b": name_b,
        "auc_a": float(base_auc_a),
        "auc_b": float(base_auc_b),
        "delta_auc": float(delta_auc_base),
        "ci_auc_lower": float(ci_auc_lower),
        "ci_auc_upper": float(ci_auc_upper),
        "p_val_auc": p_val_auc,
        "ap_a": float(base_ap_a),
        "ap_b": float(base_ap_b),
        "delta_ap": float(delta_ap_base),
        "ci_ap_lower": float(ci_ap_lower),
        "ci_ap_upper": float(ci_ap_upper),
        "p_val_ap": p_val_ap,
    }


def main():
    logger.info("Loading feature caches across all models...")
    df_index = pd.read_csv(INDEX_CSV)
    
    # Base alignment from DINOv2
    dino_data = np.load(FEATURES_DIR / "exp7_dinov2_small_tiles.npz")
    student_ids = dino_data["student_ids"]
    y = dino_data["student_labels"]
    df_folds = df_index.drop_duplicates(subset=["student_id"]).set_index("student_id").reindex(student_ids)

    # 1. DINOv2 Joint Multilingual
    logger.info("Computing DINOv2 Joint Multilingual OOF...")
    X_dino_joint = dino_data["joint_student_features"]
    p_dino_joint = extract_oof_probs(X_dino_joint, y, df_folds, n_components=24, use_pca=True)

    # 2. DINOv2 Decision Blend (0.50 Hindi + 0.50 English)
    logger.info("Computing DINOv2 Decision Blend OOF...")
    p_dino_hi = extract_oof_probs(dino_data["hindi_student_features"], y, df_folds, n_components=24, use_pca=True)
    p_dino_en = extract_oof_probs(dino_data["english_student_features"], y, df_folds, n_components=24, use_pca=True)
    p_dino_blend = 0.50 * p_dino_hi + 0.50 * p_dino_en

    # 3. DenseNet201 Tiled
    logger.info("Computing DenseNet201 Tiled OOF...")
    dense_tiled_data = np.load(FEATURES_DIR / "exp7_densenet201_tiles.npz")
    dense_map = {sid: i for i, sid in enumerate(dense_tiled_data["student_ids"])}
    dense_idx = [dense_map[sid] for sid in student_ids]
    p_dense_hi = extract_oof_probs(dense_tiled_data["hindi_student_features"][dense_idx], y, df_folds, n_components=24, use_pca=True)
    p_dense_en = extract_oof_probs(dense_tiled_data["english_student_features"][dense_idx], y, df_folds, n_components=24, use_pca=True)
    p_dense_tiled_blend = 0.50 * p_dense_hi + 0.50 * p_dense_en

    # 4. DenseNet201 Squashed
    logger.info("Extracting / Loading DenseNet201 Squashed OOF...")
    from baseline.ablate_squashed_densenet import extract_squashed_densenet
    X_dense_sq, y_sq, sq_sids = extract_squashed_densenet()
    sq_map = {sid: i for i, sid in enumerate(sq_sids)}
    sq_idx = [sq_map[sid] for sid in student_ids]
    p_dense_squashed = extract_oof_probs(X_dense_sq[sq_idx], y, df_folds, n_components=24, use_pca=True)

    # 5. CLIP-ViT-B/32
    logger.info("Computing CLIP Decision Blend OOF...")
    clip_data = np.load(FEATURES_DIR / "exp7_clip_tiles.npz")
    clip_map = {sid: i for i, sid in enumerate(clip_data["student_ids"])}
    clip_idx = [clip_map[sid] for sid in student_ids]
    p_clip_hi = extract_oof_probs(clip_data["hindi_student_features"][clip_idx], y, df_folds, n_components=24, use_pca=True)
    p_clip_en = extract_oof_probs(clip_data["english_student_features"][clip_idx], y, df_folds, n_components=24, use_pca=True)
    p_clip_blend = 0.50 * p_clip_hi + 0.50 * p_clip_en

    # 6. Dual-ViT Fixed 50/50 Rank Average
    logger.info("Computing Dual-ViT Fixed 50/50 Rank Average OOF...")
    p_dual_vit_rank = 0.50 * (rankdata(p_dino_joint) / len(y)) + 0.50 * (rankdata(p_clip_blend) / len(y))

    # 7. Stage 4 Tabular Stack (Top-30 LR)
    logger.info("Computing Stage 4 Tabular Stack OOF...")
    df_tab = pd.read_csv(STUDENT_CSV).set_index("student_id").reindex(student_ids)
    num_cols = df_tab.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if not c.startswith("repeat_") and c not in ["label", "fold"]]
    X_tab = df_tab[num_cols].fillna(df_tab[num_cols].median()).values.astype(np.float32)
    p_tabular = extract_oof_probs(X_tab, y, df_folds, n_components=30, use_pca=True)

    # Save all raw predictions
    all_oof = {
        "student_ids": student_ids,
        "y_true": y,
        "dinov2_joint": p_dino_joint,
        "dinov2_blend": p_dino_blend,
        "densenet_tiled": p_dense_tiled_blend,
        "densenet_squashed": p_dense_squashed,
        "clip_blend": p_clip_blend,
        "dual_vit_50_50_rank": p_dual_vit_rank,
        "tabular_top30": p_tabular,
    }
    np.savez_compressed(FEATURES_DIR / "exp7_all_models_oof_probs.npz", **all_oof)
    logger.info("Saved all model out-of-fold probability vectors to %s", FEATURES_DIR / "exp7_all_models_oof_probs.npz")

    # Run Paired Bootstrap Comparisons
    results = {}

    # Comparison 1: DINOv2 vs DenseNet201 (Tiled) -> ViT representation effect
    results["dinov2_vs_densenet_tiled"] = paired_bootstrap_test(
        y, p_dino_joint, p_dense_tiled_blend, "DINOv2-Small (Joint)", "DenseNet201 (Tiled)"
    )

    # Comparison 2: DenseNet201 Tiled vs DenseNet201 Squashed -> Pure tiling effect
    results["densenet_tiled_vs_squashed"] = paired_bootstrap_test(
        y, p_dense_tiled_blend, p_dense_squashed, "DenseNet201 (Tiled)", "DenseNet201 (Squashed)"
    )

    # Comparison 3: Dual-ViT 50/50 Rank Ensemble vs DINOv2 Alone
    results["dual_vit_vs_dinov2_joint"] = paired_bootstrap_test(
        y, p_dual_vit_rank, p_dino_joint, "Dual-ViT 50/50 Rank Ensemble", "DINOv2-Small (Joint)"
    )

    # Comparison 4: DINOv2 vs Stage 4 Tabular Stack
    results["dinov2_joint_vs_tabular"] = paired_bootstrap_test(
        y, p_dino_joint, p_tabular, "DINOv2-Small (Joint)", "Stage 4 Tabular Stack"
    )

    out_file = REPORTS_DIR / "exp7_paired_bootstrap_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved paired bootstrap results to %s", out_file)


if __name__ == "__main__":
    main()

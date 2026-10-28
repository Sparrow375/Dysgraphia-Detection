"""
Experiment 7 Squeeze: Breakdown Tile Pooling & Logit-Standardized Dual-ViT Consensus.
Tests:
1. Tile-level pooling enhancements:
   - 90th percentile (p90) to capture localized handwriting breakdown.
   - Generalized Mean (GeM, p=3) across tiles.
   - Tile-to-Tile consecutive delta (|e_{t+1} - e_t|) measuring motor fatigue & jitter.
2. Logit-Standardized Dual-ViT Blending to eliminate tail threshold shifts at 90% specificity.
3. 5x5 Repeated Cross-Validation on all 215 students with Paired Bootstrap comparisons.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("squeeze_tile_pooling")

DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = REPO_ROOT / "reports"

DINO_NPZ = FEATURES_DIR / "exp7_dinov2_small_tiles.npz"
CLIP_NPZ = FEATURES_DIR / "exp7_clip_tiles.npz"


def calculate_sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.90
) -> Tuple[float, int, int]:
    """Calculate sensitivity at a fixed operational specificity cutoff."""
    neg_scores = y_score[y_true == 0]
    pos_scores = y_score[y_true == 1]
    n_pos = len(pos_scores)

    # Threshold: (1 - target_spec) percentile of negatives
    threshold = float(np.percentile(neg_scores, 100.0 * target_spec))
    tp = int(np.sum(pos_scores >= threshold))
    sens = float(tp / n_pos) if n_pos > 0 else 0.0
    return sens, tp, n_pos


def build_student_matrix(
    tile_data: Dict[str, Any],
    include_p90: bool = True,
    include_gem: bool = False,
    include_delta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, pd.DataFrame]:
    """Aggregate tile descriptors into student vectors using customizable pooling operators."""
    all_desc = tile_data["tile_descriptors"]  # (N_tiles, D)
    tile_stu = np.array(tile_data["tile_student_ids"])
    tile_scr = np.array(tile_data["tile_scripts"])
    tile_tasks = np.array(tile_data["tile_task_uids"])
    student_ids = list(tile_data["student_ids"])
    y = np.array(tile_data["student_labels"], dtype=np.int64)

    desc_dim = all_desc.shape[1]

    df_folds = pd.DataFrame({"student_id": student_ids})
    for r in range(5):
        df_folds[f"repeat_{r}_fold"] = tile_data[f"repeat_{r}_fold"]

    hindi_vecs: List[np.ndarray] = []
    english_vecs: List[np.ndarray] = []
    joint_vecs: List[np.ndarray] = []

    for sid in student_ids:
        stu_mask = tile_stu == sid
        stu_descs = all_desc[stu_mask]
        stu_scripts = tile_scr[stu_mask]
        stu_tasks = tile_tasks[stu_mask]

        hin_mask = stu_scripts == "devanagari"
        eng_mask = stu_scripts == "latin"

        def pool_tiles(descs: np.ndarray, tasks: np.ndarray) -> np.ndarray:
            if len(descs) == 0:
                parts_count = 2 + (1 if include_p90 else 0) + (1 if include_gem else 0) + (2 if include_delta else 0)
                return np.zeros(parts_count * desc_dim, dtype=np.float32)

            parts = [np.mean(descs, axis=0), np.std(descs, axis=0)]

            if include_p90:
                parts.append(np.percentile(descs, 90.0, axis=0))

            if include_gem:
                gem_val = np.power(
                    np.mean(np.power(np.maximum(np.abs(descs), 1e-6), 3.0), axis=0), 1.0 / 3.0
                )
                parts.append(gem_val)

            if include_delta:
                # Group by sentence task to compute consecutive intra-task tile delta
                unique_tasks = pd.unique(tasks)
                deltas = []
                for t in unique_tasks:
                    t_descs = descs[tasks == t]
                    if len(t_descs) > 1:
                        deltas.append(np.abs(np.diff(t_descs, axis=0)))
                if deltas:
                    all_deltas = np.concatenate(deltas, axis=0)
                    parts.append(np.mean(all_deltas, axis=0))
                    parts.append(np.std(all_deltas, axis=0))
                else:
                    parts.append(np.zeros(desc_dim, dtype=np.float32))
                    parts.append(np.zeros(desc_dim, dtype=np.float32))

            return np.concatenate(parts, axis=-1).astype(np.float32)

        h_vec = pool_tiles(stu_descs[hin_mask], stu_tasks[hin_mask])
        e_vec = pool_tiles(stu_descs[eng_mask], stu_tasks[eng_mask])
        j_vec = np.concatenate([h_vec, e_vec], axis=-1)

        hindi_vecs.append(h_vec)
        english_vecs.append(e_vec)
        joint_vecs.append(j_vec)

    X_hi = np.array(hindi_vecs, dtype=np.float32)
    X_en = np.array(english_vecs, dtype=np.float32)
    X_joint = np.array(joint_vecs, dtype=np.float32)

    return X_hi, X_en, X_joint, student_ids, y, df_folds


def run_5x5_cv_probe(
    X: np.ndarray,
    y: np.ndarray,
    df_folds: pd.DataFrame,
    n_components: int = 24,
    C: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Execute 5-repeat 5-fold CV linear probe, returning pooled OOF probabilities and raw logits."""
    n_students = len(y)
    oof_probs_repeats = np.zeros((5, n_students), dtype=np.float32)
    oof_logits_repeats = np.zeros((5, n_students), dtype=np.float32)

    for r in range(5):
        fold_col = df_folds[f"repeat_{r}_fold"].values
        for f in range(5):
            te_mask = fold_col == f
            tr_mask = fold_col != f

            X_tr, y_tr = X[tr_mask], y[tr_mask]
            X_te = X[te_mask]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            pca = PCA(n_components=n_components, random_state=42)
            X_tr_p = pca.fit_transform(X_tr_s)
            X_te_p = pca.transform(X_te_s)

            clf = LogisticRegression(
                C=C, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42
            )
            clf.fit(X_tr_p, y_tr)

            oof_probs_repeats[r, te_mask] = clf.predict_proba(X_te_p)[:, 1]
            oof_logits_repeats[r, te_mask] = clf.decision_function(X_te_p)

    oof_probs = oof_probs_repeats.mean(axis=0)
    oof_logits = oof_logits_repeats.mean(axis=0)

    auc = float(roc_auc_score(y, oof_probs))
    ap = float(average_precision_score(y, oof_probs))
    s80, tp80, _ = calculate_sensitivity_at_specificity(y, oof_probs, 0.80)
    s90, tp90, _ = calculate_sensitivity_at_specificity(y, oof_probs, 0.90)

    metrics = {
        "auroc": round(auc, 4),
        "auprc": round(ap, 4),
        "sens_at_80": round(s80, 4),
        "tp_at_80": tp80,
        "sens_at_90": round(s90, 4),
        "tp_at_90": tp90,
    }
    return oof_probs, oof_logits, metrics


def paired_bootstrap(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute paired bootstrap difference intervals on identical student predictions."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    base_auc_diff = float(roc_auc_score(y_true, probs_a) - roc_auc_score(y_true, probs_b))
    base_ap_diff = float(average_precision_score(y_true, probs_a) - average_precision_score(y_true, probs_b))

    s90_a, _, _ = calculate_sensitivity_at_specificity(y_true, probs_a, 0.90)
    s90_b, _, _ = calculate_sensitivity_at_specificity(y_true, probs_b, 0.90)
    base_s90_diff = float(s90_a - s90_b)

    delta_aucs, delta_aps, delta_s90s = [], [], []
    for _ in range(n_boot):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        y_b = y_true[b_idx]
        pa_b = probs_a[b_idx]
        pb_b = probs_b[b_idx]

        delta_aucs.append(roc_auc_score(y_b, pa_b) - roc_auc_score(y_b, pb_b))
        delta_aps.append(average_precision_score(y_b, pa_b) - average_precision_score(y_b, pb_b))

        sa, _, _ = calculate_sensitivity_at_specificity(y_b, pa_b, 0.90)
        sb, _, _ = calculate_sensitivity_at_specificity(y_b, pb_b, 0.90)
        delta_s90s.append(sa - sb)

    delta_aucs = np.array(delta_aucs)
    delta_aps = np.array(delta_aps)
    delta_s90s = np.array(delta_s90s)

    ci_auc = [float(np.percentile(delta_aucs, 2.5)), float(np.percentile(delta_aucs, 97.5))]
    ci_ap = [float(np.percentile(delta_aps, 2.5)), float(np.percentile(delta_aps, 97.5))]
    ci_s90 = [float(np.percentile(delta_s90s, 2.5)), float(np.percentile(delta_s90s, 97.5))]

    p_val_auc = float(np.mean(delta_aucs <= 0.0))
    p_val_s90 = float(np.mean(delta_s90s <= 0.0))

    return {
        "delta_auroc": round(base_auc_diff, 4),
        "ci_auroc_95": [round(c, 4) for c in ci_auc],
        "p_val_auroc": round(p_val_auc, 4),
        "delta_auprc": round(base_ap_diff, 4),
        "ci_auprc_95": [round(c, 4) for c in ci_ap],
        "delta_sens90": round(base_s90_diff, 4),
        "ci_sens90_95": [round(c, 4) for c in ci_s90],
        "p_val_sens90": round(p_val_s90, 4),
    }


def main():
    logger.info("Loading cached tile representations from disk...")
    dino_raw = np.load(DINO_NPZ)
    clip_raw = np.load(CLIP_NPZ)

    logger.info("DINO tiles: %s | CLIP tiles: %s", dino_raw["tile_descriptors"].shape, clip_raw["tile_descriptors"].shape)

    # Test Matrix of Pooling Configurations
    configs = [
        {"name": "Cfg0_Baseline_MeanStd", "p90": False, "gem": False, "delta": False},
        {"name": "Cfg1_MeanStd_p90", "p90": True, "gem": False, "delta": False},
        {"name": "Cfg2_MeanStd_Delta", "p90": False, "gem": False, "delta": True},
        {"name": "Cfg3_MeanStd_p90_Delta", "p90": True, "gem": False, "delta": True},
        {"name": "Cfg4_MeanStd_p90_GeM_Delta", "p90": True, "gem": True, "delta": True},
    ]

    all_results = {}

    for cfg in configs:
        cname = cfg["name"]
        logger.info("=" * 60)
        logger.info("Evaluating Configuration: %s (p90=%s, gem=%s, delta=%s)", cname, cfg["p90"], cfg["gem"], cfg["delta"])

        # 1. Aggregate DINOv2
        X_dino_hi, X_dino_en, X_dino_joint, student_ids, y, df_folds = build_student_matrix(
            dino_raw, include_p90=cfg["p90"], include_gem=cfg["gem"], include_delta=cfg["delta"]
        )
        # 2. Aggregate CLIP
        X_clip_hi, X_clip_en, X_clip_joint, _, _, _ = build_student_matrix(
            clip_raw, include_p90=cfg["p90"], include_gem=cfg["gem"], include_delta=cfg["delta"]
        )

        logger.info("  DINO Joint Matrix: %s | CLIP Joint Matrix: %s", X_dino_joint.shape, X_clip_joint.shape)

        # 3. Fit 5x5 Probes
        p_dino, logit_dino, m_dino = run_5x5_cv_probe(X_dino_joint, y, df_folds, n_components=24)
        p_clip, logit_clip, m_clip = run_5x5_cv_probe(X_clip_joint, y, df_folds, n_components=24)

        # 4. Ensembles:
        # A. Raw Soft Prob Blend (0.60 DINO + 0.40 CLIP)
        p_dual_soft = 0.60 * p_dino + 0.40 * p_clip
        m_dual_soft = {
            "auroc": round(roc_auc_score(y, p_dual_soft), 4),
            "auprc": round(average_precision_score(y, p_dual_soft), 4),
            "sens_at_80": round(calculate_sensitivity_at_specificity(y, p_dual_soft, 0.80)[0], 4),
            "tp_at_80": calculate_sensitivity_at_specificity(y, p_dual_soft, 0.80)[1],
            "sens_at_90": round(calculate_sensitivity_at_specificity(y, p_dual_soft, 0.90)[0], 4),
            "tp_at_90": calculate_sensitivity_at_specificity(y, p_dual_soft, 0.90)[1],
        }

        # B. Logit-Standardized Blend
        # Standardize logits across cohort to have equal scale and zero offset
        z_dino = (logit_dino - np.mean(logit_dino)) / (np.std(logit_dino) + 1e-6)
        z_clip = (logit_clip - np.mean(logit_clip)) / (np.std(logit_clip) + 1e-6)
        z_dual_blend = 0.50 * z_dino + 0.50 * z_clip
        m_dual_logit = {
            "auroc": round(roc_auc_score(y, z_dual_blend), 4),
            "auprc": round(average_precision_score(y, z_dual_blend), 4),
            "sens_at_80": round(calculate_sensitivity_at_specificity(y, z_dual_blend, 0.80)[0], 4),
            "tp_at_80": calculate_sensitivity_at_specificity(y, z_dual_blend, 0.80)[1],
            "sens_at_90": round(calculate_sensitivity_at_specificity(y, z_dual_blend, 0.90)[0], 4),
            "tp_at_90": calculate_sensitivity_at_specificity(y, z_dual_blend, 0.90)[1],
        }

        # C. Fixed 50/50 Rank Average
        r_dino = rankdata(p_dino) / len(y)
        r_clip = rankdata(p_clip) / len(y)
        r_dual_rank = 0.50 * r_dino + 0.50 * r_clip
        m_dual_rank = {
            "auroc": round(roc_auc_score(y, r_dual_rank), 4),
            "auprc": round(average_precision_score(y, r_dual_rank), 4),
            "sens_at_80": round(calculate_sensitivity_at_specificity(y, r_dual_rank, 0.80)[0], 4),
            "tp_at_80": calculate_sensitivity_at_specificity(y, r_dual_rank, 0.80)[1],
            "sens_at_90": round(calculate_sensitivity_at_specificity(y, r_dual_rank, 0.90)[0], 4),
            "tp_at_90": calculate_sensitivity_at_specificity(y, r_dual_rank, 0.90)[1],
        }

        logger.info("  DINOv2 Standalone: AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%%, Sens@90=%.1f%% (%d/40)",
                    m_dino["auroc"], m_dino["auprc"], m_dino["sens_at_80"]*100, m_dino["sens_at_90"]*100, m_dino["tp_at_90"])
        logger.info("  CLIP Standalone:   AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%%, Sens@90=%.1f%% (%d/40)",
                    m_clip["auroc"], m_clip["auprc"], m_clip["sens_at_80"]*100, m_clip["sens_at_90"]*100, m_clip["tp_at_90"])
        logger.info("  Dual Soft Blend:   AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%%, Sens@90=%.1f%% (%d/40)",
                    m_dual_soft["auroc"], m_dual_soft["auprc"], m_dual_soft["sens_at_80"]*100, m_dual_soft["sens_at_90"]*100, m_dual_soft["tp_at_90"])
        logger.info("  Dual Logit Blend:  AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%%, Sens@90=%.1f%% (%d/40)",
                    m_dual_logit["auroc"], m_dual_logit["auprc"], m_dual_logit["sens_at_80"]*100, m_dual_logit["sens_at_90"]*100, m_dual_logit["tp_at_90"])
        logger.info("  Dual Rank Blend:   AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%%, Sens@90=%.1f%% (%d/40)",
                    m_dual_rank["auroc"], m_dual_rank["auprc"], m_dual_rank["sens_at_80"]*100, m_dual_rank["sens_at_90"]*100, m_dual_rank["tp_at_90"])

        all_results[cname] = {
            "dino": m_dino,
            "clip": m_clip,
            "dual_soft": m_dual_soft,
            "dual_logit": m_dual_logit,
            "dual_rank": m_dual_rank,
            "raw_predictions": {
                "p_dino": [round(float(x), 4) for x in p_dino],
                "p_clip": [round(float(x), 4) for x in p_clip],
                "p_dual_logit": [round(float(x), 4) for x in z_dual_blend],
                "p_dual_rank": [round(float(x), 4) for x in r_dual_rank],
            }
        }

    # Paired Bootstrap Comparison: Best Enriched Config vs Baseline Cfg0
    base_rank = np.array(all_results["Cfg0_Baseline_MeanStd"]["raw_predictions"]["p_dual_rank"])
    best_cfg_name = "Cfg3_MeanStd_p90_Delta"
    if best_cfg_name in all_results:
        best_logit = np.array(all_results[best_cfg_name]["raw_predictions"]["p_dual_logit"])
        best_rank = np.array(all_results[best_cfg_name]["raw_predictions"]["p_dual_rank"])

        logger.info("=" * 60)
        logger.info("Running Paired Bootstrap: %s (Logit) vs Cfg0 Baseline (Rank)...", best_cfg_name)
        pb_logit = paired_bootstrap(y, best_logit, base_rank, n_boot=2000)
        logger.info("  Δ AUROC: %+.4f [95%% CI: %+.4f, %+.4f] (p = %.4f)", pb_logit["delta_auroc"], pb_logit["ci_auroc_95"][0], pb_logit["ci_auroc_95"][1], pb_logit["p_val_auroc"])
        logger.info("  Δ Sens@90: %+.4f [95%% CI: %+.4f, %+.4f] (p = %.4f)", pb_logit["delta_sens90"], pb_logit["ci_sens90_95"][0], pb_logit["ci_sens90_95"][1], pb_logit["p_val_sens90"])
        all_results["paired_bootstrap_best_vs_baseline"] = pb_logit

    # Save to JSON
    out_file = REPORTS_DIR / "exp7_squeeze_tile_pooling_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved complete results to %s", out_file)


if __name__ == "__main__":
    main()


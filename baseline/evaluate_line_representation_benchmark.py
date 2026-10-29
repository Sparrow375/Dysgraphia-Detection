"""
Evaluation script to benchmark Line-Decomposed Vision Features (DINOv2 + CLIP)
on the exact 215-student cohort with 5x5 repeated cross-validation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("line_benchmark")

DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = REPO_ROOT / "reports"

OOF_NPZ = FEATURES_DIR / "exp7_all_models_oof_probs.npz"
LINE_DINO_NPZ = FEATURES_DIR / "exp7_line_dinov2_small_tiles.npz"
LINE_CLIP_NPZ = FEATURES_DIR / "exp7_line_clip_tiles.npz"


def tune_and_fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k_candidates: List[int] = [16, 24, 32],
    c_candidates: List[float] = [0.01, 0.05, 0.1, 0.5, 1.0],
    seed: int = 42,
) -> np.ndarray:
    best_score = -1.0
    best_k = k_candidates[0]
    best_c = c_candidates[0]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    for k in k_candidates:
        if k > X_train.shape[0] or k > X_train.shape[1]:
            continue
        for c in c_candidates:
            val_aurocs = []
            for tr_idx, val_idx in skf.split(X_train, y_train):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[tr_idx])
                X_v = scaler.transform(X_train[val_idx])

                pca = PCA(n_components=k, random_state=seed)
                X_tr_pca = pca.fit_transform(X_tr)
                X_v_pca = pca.transform(X_v)

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

    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)

    pca_final = PCA(n_components=best_k, random_state=seed)
    X_train_pca = pca_final.fit_transform(X_train_scaled)
    X_test_pca = pca_final.transform(X_test_scaled)

    lr_final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=seed)
    lr_final.fit(X_train_pca, y_train)

    return lr_final.predict_proba(X_test_pca)[:, 1]


def run_5x5_cv(
    X_hin: np.ndarray,
    X_eng: np.ndarray,
    X_joint: np.ndarray,
    y: np.ndarray,
    fold_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    n = len(y)
    oof_h = np.zeros(n, dtype=np.float32)
    oof_e = np.zeros(n, dtype=np.float32)
    oof_j = np.zeros(n, dtype=np.float32)
    oof_b = np.zeros(n, dtype=np.float32)

    for r in range(5):
        fold_col = fold_df[f"repeat_{r}_fold"].values
        for f in range(5):
            te_mask = (fold_col == f)
            tr_mask = (fold_col != f)

            y_tr = y[tr_mask]
            y_te = y[te_mask]
            seed = 42 + r * 10 + f

            p_h = tune_and_fit_probe(X_hin[tr_mask], y_tr, X_hin[te_mask], seed=seed)
            p_e = tune_and_fit_probe(X_eng[tr_mask], y_tr, X_eng[te_mask], seed=seed)
            p_j = tune_and_fit_probe(X_joint[tr_mask], y_tr, X_joint[te_mask], seed=seed)

            oof_h[te_mask] += p_h / 5.0
            oof_e[te_mask] += p_e / 5.0
            oof_j[te_mask] += p_j / 5.0
            oof_b[te_mask] += (0.5 * (p_h + p_e)) / 5.0

    return {
        "hindi": oof_h,
        "english": oof_e,
        "joint": oof_j,
        "blend": oof_b,
    }


def calc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    auc = float(roc_auc_score(y_true, y_score))
    pr = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))

    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    n_pos = len(pos)

    t80 = float(np.percentile(neg, 80.0))
    tp80 = int(np.sum(pos >= t80))

    t90 = float(np.percentile(neg, 90.0))
    tp90 = int(np.sum(pos >= t90))

    return {
        "auroc": auc,
        "auprc": pr,
        "brier": brier,
        "sens80": float(tp80 / n_pos),
        "tp80": tp80,
        "sens90": float(tp90 / n_pos),
        "tp90": tp90,
    }


def main():
    logger.info("Loading baseline and line features...")
    oof_data = np.load(OOF_NPZ)
    sids_benchmark = list(oof_data["student_ids"])
    y_true = oof_data["y_true"]
    n_cohort = len(sids_benchmark)
    logger.info("Cohort size: %d students (%d pos, %d neg)", n_cohort, int(np.sum(y_true == 1)), int(np.sum(y_true == 0)))

    # Fold dataframe for the 215 students
    dino_raw = np.load(LINE_DINO_NPZ)
    line_sids_map = {sid: i for i, sid in enumerate(dino_raw["student_ids"])}

    align_idx = [line_sids_map[s] for s in sids_benchmark]

    fold_df = pd.DataFrame({"student_id": sids_benchmark})
    for r in range(5):
        fold_df[f"repeat_{r}_fold"] = dino_raw[f"repeat_{r}_fold"][align_idx]

    # Align DINOv2
    X_dino_h = dino_raw["hindi_student_features"][align_idx]
    X_dino_e = dino_raw["english_student_features"][align_idx]
    X_dino_j = dino_raw["joint_student_features"][align_idx]

    # Align CLIP
    clip_raw = np.load(LINE_CLIP_NPZ)
    clip_sids_map = {sid: i for i, sid in enumerate(clip_raw["student_ids"])}
    clip_align_idx = [clip_sids_map[s] for s in sids_benchmark]

    X_clip_h = clip_raw["hindi_student_features"][clip_align_idx]
    X_clip_e = clip_raw["english_student_features"][clip_align_idx]
    X_clip_j = clip_raw["joint_student_features"][clip_align_idx]

    logger.info("--- Evaluating LINE DINOv2 (5x5 CV) ---")
    dino_oofs = run_5x5_cv(X_dino_h, X_dino_e, X_dino_j, y_true, fold_df)
    m_dino_j = calc_metrics(y_true, dino_oofs["joint"])
    m_dino_b = calc_metrics(y_true, dino_oofs["blend"])
    logger.info("Line DINOv2 Joint: AUROC=%.4f, AUPRC=%.4f, Sens@90=%.1f%% (%d/37)",
                m_dino_j["auroc"], m_dino_j["auprc"], m_dino_j["sens90"]*100, m_dino_j["tp90"])
    logger.info("Line DINOv2 Blend: AUROC=%.4f, AUPRC=%.4f, Sens@90=%.1f%% (%d/37)",
                m_dino_b["auroc"], m_dino_b["auprc"], m_dino_b["sens90"]*100, m_dino_b["tp90"])

    logger.info("--- Evaluating LINE CLIP ViT-B/32 (5x5 CV) ---")
    clip_oofs = run_5x5_cv(X_clip_h, X_clip_e, X_clip_j, y_true, fold_df)
    m_clip_j = calc_metrics(y_true, clip_oofs["joint"])
    m_clip_b = calc_metrics(y_true, clip_oofs["blend"])
    logger.info("Line CLIP Joint  : AUROC=%.4f, AUPRC=%.4f, Sens@90=%.1f%% (%d/37)",
                m_clip_j["auroc"], m_clip_j["auprc"], m_clip_j["sens90"]*100, m_clip_j["tp90"])
    logger.info("Line CLIP Blend  : AUROC=%.4f, AUPRC=%.4f, Sens@90=%.1f%% (%d/37)",
                m_clip_b["auroc"], m_clip_b["auprc"], m_clip_b["sens90"]*100, m_clip_b["tp90"])

    # Line Dual-ViT 50-50 Rank Consensus
    r_dino = rankdata(dino_oofs["joint"]) / n_cohort
    r_clip = rankdata(clip_oofs["joint"]) / n_cohort
    p_line_dual = 0.50 * r_dino + 0.50 * r_clip
    m_line_dual = calc_metrics(y_true, p_line_dual)
    logger.info("Line Dual-ViT 50/50 Rank: AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%% (%d/37), Sens@90=%.1f%% (%d/37)",
                m_line_dual["auroc"], m_line_dual["auprc"],
                m_line_dual["sens80"]*100, m_line_dual["tp80"],
                m_line_dual["sens90"]*100, m_line_dual["tp90"])

    # Sentence Crop Dual-ViT benchmark
    p_orig_dual = oof_data["dual_vit_50_50_rank"]
    m_orig_dual = calc_metrics(y_true, p_orig_dual)
    logger.info("Sentence Crop Dual-ViT  : AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%% (%d/37), Sens@90=%.1f%% (%d/37)",
                m_orig_dual["auroc"], m_orig_dual["auprc"],
                m_orig_dual["sens80"]*100, m_orig_dual["tp80"],
                m_orig_dual["sens90"]*100, m_orig_dual["tp90"])

    # Compare key students
    check_students = [
        ("G7_A_Roll02", "School A G7 control (multi-line crop)"),
        ("SCHB_G6_IMG_20261007_102238", "School B G6 control (multi-line crop)"),
        ("G3_A_Roll05", "School A G3 fancy control"),
        ("G3_A_Roll12", "School A G3 no word space positive"),
        ("SCHB_G7_IMG_20261007_105512", "School B G7 bad hindi positive"),
        ("SCHB_G4_IMG_20261007_100350", "School B G4 reclassified positive"),
    ]

    print("\n" + "=" * 90)
    print(f"{'Student ID':<28} | {'Label':<5} | {'Sent Dual-ViT':<14} | {'Line Dual-ViT':<14} | {'Difference':<10}")
    print("=" * 90)
    for sid, note in check_students:
        if sid in sids_benchmark:
            idx = sids_benchmark.index(sid)
            lbl = y_true[idx]
            s_orig = p_orig_dual[idx]
            s_line = p_line_dual[idx]
            diff = s_line - s_orig
            print(f"{sid:<28} | {lbl:<5} | {s_orig:<14.4f} | {s_line:<14.4f} | {diff:+10.4f}  ({note})")
    print("=" * 90)

    # Save line Dual-ViT out-of-fold predictions
    np.savez_compressed(
        FEATURES_DIR / "exp7_line_dual_vit_oof.npz",
        student_ids=sids_benchmark,
        y_true=y_true,
        line_dinov2_joint=dino_oofs["joint"],
        line_clip_joint=clip_oofs["joint"],
        line_dual_vit_rank=p_line_dual,
    )
    logger.info("Saved line Dual-ViT OOF to %s", FEATURES_DIR / "exp7_line_dual_vit_oof.npz")


if __name__ == "__main__":
    main()

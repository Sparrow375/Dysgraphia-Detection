"""
Experiment 7: Hierarchical Line + Sentence Vision & Bilingual OCR Evaluation.
Evaluates:
1. Hierarchical Dual-ViT Vision (80% Sentence + 20% Line)
2. Standalone Line-Decomposed English TrOCR
3. Hierarchical Tri-Modal Champion (84% Hierarchical Vision + 8% Line English OCR + 8% Tabular)
4. Hierarchical Asymmetric Phenotypic Rescue (Hierarchical Vision + beta * max(0, OCR - tau) + 0.05 Tabular)
5. Paired Stratified Bootstrap (B=2,000) against Previous Sentence Champion.
"""

from __future__ import annotations

import argparse
import json
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
logger = logging.getLogger("hierarchical_champion")

DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OOF_NPZ = FEATURES_DIR / "exp7_all_models_oof_probs.npz"
LINE_VISION_NPZ = FEATURES_DIR / "exp7_line_dual_vit_oof.npz"
LINE_OCR_NPZ = FEATURES_DIR / "exp7_line_bilingual_ocr_student_features.npz"
SCORES_JSON = FEATURES_DIR / "exp7_student_model_scores.json"

OUTPUT_JSON = REPORTS_DIR / "exp7_hierarchical_line_champion_results.json"
OUTPUT_MD = REPORTS_DIR / "exp7_hierarchical_line_champion_report.md"


def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    auc = float(roc_auc_score(y_true, y_score))
    pr = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))

    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    n_pos = len(pos)

    t80 = float(np.percentile(neg, 80.0))
    tp80 = int(np.sum(pos >= t80))
    sens80 = float(tp80 / n_pos) if n_pos > 0 else 0.0

    t90 = float(np.percentile(neg, 90.0))
    tp90 = int(np.sum(pos >= t90))
    sens90 = float(tp90 / n_pos) if n_pos > 0 else 0.0

    return {
        "auroc": auc,
        "auprc": pr,
        "brier": brier,
        "sens_at_80": sens80,
        "det_at_80": tp80,
        "sens_at_90": sens90,
        "det_at_90": tp90,
    }


def run_5x5_cv_probe(
    X: np.ndarray,
    y: np.ndarray,
    fold_df: pd.DataFrame,
    n_components: int = 12,
    C: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    n = len(y)
    oof_preds = np.zeros(n, dtype=np.float32)

    for r in range(5):
        repeat_col = fold_df[f"repeat_{r}_fold"].values
        for f in range(5):
            te_mask = (repeat_col == f)
            tr_mask = (repeat_col != f)

            X_tr, y_tr = X[tr_mask], y[tr_mask]
            X_te = X[te_mask]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            pca = PCA(n_components=min(n_components, X_tr.shape[1], X_tr.shape[0]), random_state=seed + r * 10 + f)
            X_tr_pca = pca.fit_transform(X_tr_s)
            X_te_pca = pca.transform(X_te_s)

            lr = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed + r * 10 + f)
            lr.fit(X_tr_pca, y_tr)
            p_te = lr.predict_proba(X_te_pca)[:, 1]

            oof_preds[te_mask] += p_te / 5.0

    return oof_preds


def paired_bootstrap_delta(
    y_true: np.ndarray,
    p_candidate: np.ndarray,
    p_baseline: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    auc_diffs = []
    pr_diffs = []

    for _ in range(n_bootstraps):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        y_b = y_true[b_idx]
        p_c_b = p_candidate[b_idx]
        p_b_b = p_baseline[b_idx]

        auc_c = roc_auc_score(y_b, p_c_b)
        auc_base = roc_auc_score(y_b, p_b_b)
        auc_diffs.append(auc_c - auc_base)

        pr_c = average_precision_score(y_b, p_c_b)
        pr_base = average_precision_score(y_b, p_b_b)
        pr_diffs.append(pr_c - pr_base)

    auc_diffs = np.array(auc_diffs)
    pr_diffs = np.array(pr_diffs)

    p_value = 2.0 * min(np.mean(auc_diffs <= 0.0), np.mean(auc_diffs >= 0.0))
    p_value = float(max(p_value, 1.0 / n_bootstraps))

    return {
        "delta_auroc_mean": float(np.mean(auc_diffs)),
        "delta_auroc_ci_low": float(np.percentile(auc_diffs, 2.5)),
        "delta_auroc_ci_high": float(np.percentile(auc_diffs, 97.5)),
        "delta_auprc_mean": float(np.mean(pr_diffs)),
        "p_value": p_value,
    }


def evaluate():
    logger.info("Loading OOF caches and feature representations...")
    oof_data = np.load(OOF_NPZ)
    line_vis = np.load(LINE_VISION_NPZ)
    line_ocr = np.load(LINE_OCR_NPZ)

    sids = list(oof_data["student_ids"])
    y = oof_data["y_true"]
    n_students = len(sids)

    fold_df = pd.DataFrame({"student_id": sids})
    for r in range(5):
        fold_df[f"repeat_{r}_fold"] = line_ocr[f"repeat_{r}_fold"]

    # 1. Vision Streams
    p_sent_dual = oof_data["dual_vit_50_50_rank"]
    p_line_dual = line_vis["line_dual_vit_rank"]

    r_sent = rankdata(p_sent_dual) / n_students
    r_line = rankdata(p_line_dual) / n_students

    # Hierarchical Vision (80% Sentence Macro-Context + 20% Line Micro-Stroke)
    r_vis_hier = 0.80 * r_sent + 0.20 * r_line
    m_vis_hier = calculate_metrics(y, r_vis_hier)
    logger.info("Hierarchical Dual-ViT Alone: AUROC=%.4f, AUPRC=%.4f, Sens@90=%.1f%% (%d/37)",
                m_vis_hier["auroc"], m_vis_hier["auprc"], m_vis_hier["sens_at_90"]*100, m_vis_hier["det_at_90"])

    # 2. Line OCR Streams
    logger.info("Fitting Line English OCR Probe...")
    p_line_en_ocr = run_5x5_cv_probe(line_ocr["X_english_ocr"], y, fold_df, n_components=12, C=0.1)
    m_line_en_ocr = calculate_metrics(y, p_line_en_ocr)
    logger.info("Line English OCR Alone: AUROC=%.4f, AUPRC=%.4f, Sens@90=%.1f%% (%d/37)",
                m_line_en_ocr["auroc"], m_line_en_ocr["auprc"], m_line_en_ocr["sens_at_90"]*100, m_line_en_ocr["det_at_90"])

    r_ocr = rankdata(p_line_en_ocr) / n_students
    r_tab = rankdata(oof_data["tabular_top30"]) / n_students

    # 3. Model Comparisons
    # Previous Champion (Sentence Dual-ViT 84% + Sent OCR 8% + Tabular 8%)
    p_prev_champ = 0.84 * r_sent + 0.08 * r_ocr + 0.08 * r_tab
    m_prev_champ = calculate_metrics(y, p_prev_champ)

    # Hierarchical Tri-Modal Champion (Hierarchical Vision 84% + Line OCR 8% + Tabular 8%)
    p_hier_champ = 0.84 * r_vis_hier + 0.08 * r_ocr + 0.08 * r_tab
    m_hier_champ = calculate_metrics(y, p_hier_champ)
    logger.info("Hierarchical Tri-Modal Champion: AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%% (%d/37), Sens@90=%.1f%% (%d/37)",
                m_hier_champ["auroc"], m_hier_champ["auprc"],
                m_hier_champ["sens_at_80"]*100, m_hier_champ["det_at_80"],
                m_hier_champ["sens_at_90"]*100, m_hier_champ["det_at_90"])

    # Hierarchical Asymmetric Phenotypic Rescue (tau=0.70, beta=0.15)
    raw_asym = r_vis_hier + 0.15 * np.maximum(0.0, r_ocr - 0.70) + 0.05 * r_tab
    p_asym_rescue = np.clip(raw_asym / (1.0 + 0.15 * 0.30 + 0.05), 0.0, 1.0)
    m_asym_rescue = calculate_metrics(y, p_asym_rescue)
    logger.info("Hierarchical Asymmetric Rescue : AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%% (%d/37), Sens@90=%.1f%% (%d/37)",
                m_asym_rescue["auroc"], m_asym_rescue["auprc"],
                m_asym_rescue["sens_at_80"]*100, m_asym_rescue["det_at_80"],
                m_asym_rescue["sens_at_90"]*100, m_asym_rescue["det_at_90"])

    # 4. Bootstrap Delta
    boot_hier_vs_prev = paired_bootstrap_delta(y, p_hier_champ, p_prev_champ)
    logger.info("Bootstrap Delta (Hierarchical vs Previous Champion): Delta AUROC=+%.4f [%.4f, %.4f], p=%.4f",
                boot_hier_vs_prev["delta_auroc_mean"],
                boot_hier_vs_prev["delta_auroc_ci_low"],
                boot_hier_vs_prev["delta_auroc_ci_high"],
                boot_hier_vs_prev["p_value"])

    # 5. Key Students Audit
    th90_prev = np.percentile(p_prev_champ[y==0], 90.0)
    th90_hier = np.percentile(p_hier_champ[y==0], 90.0)
    th90_asym = np.percentile(p_asym_rescue[y==0], 90.0)

    check_students = [
        ("G7_A_Roll02", "Control - Multi-line prompt"),
        ("SCHB_G6_IMG_20261007_102238", "Control - Multi-line prompt"),
        ("G3_A_Roll05", "Control - Fancy handwriting"),
        ("G3_A_Roll12", "Positive - 0 word space, baseline drift"),
        ("SCHB_G7_IMG_20261007_105512", "Positive - Butchered Hindi"),
        ("SCHB_G4_IMG_20261007_100350", "Positive - Newly reclassified"),
        ("G7_A_Roll15", "Positive - Pure motor dysgraphia"),
        ("SCHB_G6_IMG_20261007_102039", "Positive - Variable English letters"),
    ]

    student_records = []
    for sid, note in check_students:
        if sid in sids:
            i = sids.index(sid)
            lbl = int(y[i])
            s_pr = float(p_prev_champ[i])
            s_hi = float(p_hier_champ[i])
            s_as = float(p_asym_rescue[i])
            student_records.append({
                "student_id": sid,
                "label": lbl,
                "note": note,
                "prev_score": s_pr,
                "hier_score": s_hi,
                "asym_score": s_as,
                "hier_status": "CAUGHT" if (lbl == 1 and s_hi >= th90_hier) else ("FALSE ALARM" if (lbl == 0 and s_hi >= th90_hier) else "CLEAN/MISSED"),
            })

    # Save results json
    results_payload = {
        "hierarchical_vision": m_vis_hier,
        "line_english_ocr": m_line_en_ocr,
        "previous_champion": m_prev_champ,
        "hierarchical_tri_modal": m_hier_champ,
        "hierarchical_asymmetric_rescue": m_asym_rescue,
        "bootstrap_hier_vs_prev": boot_hier_vs_prev,
        "cutoffs_at_90_spec": {
            "prev": float(th90_prev),
            "hier": float(th90_hier),
            "asym": float(th90_asym),
        },
        "key_students": student_records,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_payload, f, indent=2)
    logger.info("Saved benchmark JSON to %s", OUTPUT_JSON)

    # Update exp7_all_models_oof_probs.npz with new hierarchical columns
    oof_dict = {k: oof_data[k] for k in oof_data.files}
    oof_dict["hierarchical_champion_prob"] = p_hier_champ
    oof_dict["hierarchical_asym_rescue_prob"] = p_asym_rescue
    oof_dict["line_english_ocr_prob"] = p_line_en_ocr
    oof_dict["hierarchical_vision_prob"] = r_vis_hier
    np.savez_compressed(OOF_NPZ, **oof_dict)
    logger.info("Updated %s with hierarchical champion probabilities", OOF_NPZ)

    # Update QA web app score cache
    from pipeline.export_student_model_scores import main as export_scores
    export_scores()
    logger.info("Exported updated scores to QA web app")


if __name__ == "__main__":
    evaluate()

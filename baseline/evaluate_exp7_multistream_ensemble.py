"""
Experiment 7: Comprehensive Multi-Stream Ensemble Benchmark
Fuses diverse representations across all 215 students under 5x5 Repeated CV:
1. DINOv2-Small (Reigning Benchmark: 0.8427 AUROC)
2. OpenAI CLIP-ViT-B/32 (Multimodal Semantic ViT: 0.8151 AUROC)
3. Candidate 4 VLM Rubric Features (Qwen2-VL-2B: 0.6166 AUROC)
4. Macro-Page Tabular Features (Top-10 Global Layout z-scores: 0.6870 AUROC)

Evaluates:
- Standalone streams
- Dual-ViT Vision Consensus (DINOv2 + CLIP)
- Intermediate Feature Fusion (DINOv2 + Tabular + VLM)
- Decision-Level Blends (Nested & Rank Average)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp7_multistream")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = REPO_ROOT / "reports"
INDEX_CSV = DATA_DIR / "sentence_crops_dataset" / "sentences_index.csv"
STUDENT_CSV = DATA_DIR / "datasets" / "dataset_student_level.csv"

TOP10_MACRO = [
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
]


def load_all_features() -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    """Load DINOv2, CLIP, VLM rubric, and tabular features aligned by student_id."""
    # 1. DINOv2
    dino_data = np.load(FEATURES_DIR / "exp7_dinov2_small_tiles.npz")
    student_ids = dino_data["student_ids"]
    labels = dino_data["student_labels"]
    X_dino = dino_data["joint_student_features"]

    # 2. CLIP
    clip_data = np.load(FEATURES_DIR / "exp7_clip_tiles.npz")
    clip_sids = clip_data["student_ids"]
    clip_map = {sid: i for i, sid in enumerate(clip_sids)}
    clip_indices = [clip_map[sid] for sid in student_ids]
    X_clip = clip_data["joint_student_features"][clip_indices]

    # 3. VLM Rubric
    with open(FEATURES_DIR / "vlm_rubric_cache.json", "r", encoding="utf-8") as f:
        vlm_cache = json.load(f)
    df_index = pd.read_csv(INDEX_CSV)
    vlm_feats_list = []
    rubric_criteria = ["legibility", "letter_formation", "spacing_regularity", "size_consistency", "baseline_adherence", "stroke_fluidity", "pressure_uniformity"]
    for sid in student_ids:
        s_rows = df_index[df_index["student_id"] == sid]
        s_scores = {c: [] for c in rubric_criteria}
        for _, r in s_rows.iterrows():
            tuid = str(r["task_uid"])
            if tuid in vlm_cache:
                for c in rubric_criteria:
                    s_scores[c].append(vlm_cache[tuid][c])
        f_vec = []
        for c in rubric_criteria:
            vals = s_scores[c] if s_scores[c] else [3.0]
            f_vec.extend([float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0, float(np.min(vals))])
        vlm_feats_list.append(f_vec)
    X_vlm = np.array(vlm_feats_list, dtype=np.float32)

    # 4. Tabular Macro
    df_tab = pd.read_csv(STUDENT_CSV).set_index("student_id").reindex(student_ids)
    avail_cols = [c for c in TOP10_MACRO if c in df_tab.columns]
    X_tab = df_tab[avail_cols].fillna(df_tab[avail_cols].median()).values.astype(np.float32)

    df_folds = df_index.drop_duplicates(subset=["student_id"]).set_index("student_id").reindex(student_ids)

    feature_dict = {
        "dinov2": X_dino,
        "clip": X_clip,
        "vlm": X_vlm,
        "tab_macro": X_tab,
    }
    return feature_dict, labels, student_ids, df_folds


def run_5x5_probe(
    X: np.ndarray,
    y: np.ndarray,
    df_folds: pd.DataFrame,
    n_components: int = 24,
    use_pca: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run exact 5x5 repeated Stratified Group CV."""
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
            best_c = 0.1
            best_val = -1.0
            for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
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

    mean_oof = np.mean(oof_probs, axis=0)
    auc = float(roc_auc_score(y, mean_oof))
    ap = float(average_precision_score(y, mean_oof))
    brier = float(brier_score_loss(y, mean_oof))

    neg = mean_oof[y == 0]
    pos = mean_oof[y == 1]
    th80 = float(np.percentile(neg, 80.0))
    det80 = int(np.sum(pos >= th80))
    th90 = float(np.percentile(neg, 90.0))
    det90 = int(np.sum(pos >= th90))

    metrics = {
        "auroc": auc,
        "auprc": ap,
        "brier": brier,
        "sens_at_80": det80 / len(pos),
        "det_at_80": det80,
        "sens_at_90": det90 / len(pos),
        "det_at_90": det90,
    }
    return mean_oof, metrics


def main() -> None:
    logger.info("Loading all extracted representations (DINOv2, CLIP, VLM, Tabular)...")
    feats, y, sids, df_folds = load_all_features()
    logger.info("Aligned %d students: %d positives, %d controls", len(y), np.sum(y == 1), np.sum(y == 0))

    results = {}

    # 1. DINOv2 Standalone
    logger.info("Evaluating DINOv2-Small Standalone...")
    p_dino, m_dino = run_5x5_probe(feats["dinov2"], y, df_folds, n_components=24, use_pca=True)
    results["dinov2_standalone"] = m_dino
    logger.info("DINOv2: AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%% (%d/40), Sens@90=%.1f%% (%d/40)",
                m_dino["auroc"], m_dino["auprc"], m_dino["sens_at_80"]*100, m_dino["det_at_80"], m_dino["sens_at_90"]*100, m_dino["det_at_90"])

    # 2. CLIP Standalone
    logger.info("Evaluating CLIP-ViT-B/32 Standalone...")
    p_clip, m_clip = run_5x5_probe(feats["clip"], y, df_folds, n_components=24, use_pca=True)
    results["clip_standalone"] = m_clip
    logger.info("CLIP: AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%% (%d/40), Sens@90=%.1f%% (%d/40)",
                m_clip["auroc"], m_clip["auprc"], m_clip["sens_at_80"]*100, m_clip["det_at_80"], m_clip["sens_at_90"]*100, m_clip["det_at_90"])

    # 3. Dual-ViT Consensus (DINOv2 + CLIP Decision Blend)
    logger.info("Evaluating Dual-ViT Consensus (DINOv2 + CLIP)...")
    # Soft average
    p_dual_soft = 0.60 * p_dino + 0.40 * p_clip
    # Rank average
    p_dual_rank = 0.50 * (rankdata(p_dino) / len(y)) + 0.50 * (rankdata(p_clip) / len(y))
    auc_dual_soft = float(roc_auc_score(y, p_dual_soft))
    ap_dual_soft = float(average_precision_score(y, p_dual_soft))
    auc_dual_rank = float(roc_auc_score(y, p_dual_rank))
    ap_dual_rank = float(average_precision_score(y, p_dual_rank))

    neg_d = p_dual_soft[y == 0]
    pos_d = p_dual_soft[y == 1]
    det80_d = int(np.sum(pos_d >= np.percentile(neg_d, 80.0)))
    det90_d = int(np.sum(pos_d >= np.percentile(neg_d, 90.0)))

    results["dual_vit_soft"] = {
        "auroc": auc_dual_soft, "auprc": ap_dual_soft,
        "sens_at_80": det80_d / 40, "det_at_80": det80_d,
        "sens_at_90": det90_d / 40, "det_at_90": det90_d,
    }
    logger.info("Dual-ViT Soft (60 DINO + 40 CLIP): AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%% (%d/40), Sens@90=%.1f%% (%d/40)",
                auc_dual_soft, ap_dual_soft, det80_d / 40 * 100, det80_d, det90_d / 40 * 100, det90_d)

    # 4. Concatenated Multi-Stream Intermediate Feature Fusion
    # Concatenate: 24 DINOv2 PCA + 16 CLIP PCA + VLM Rubric (21) + Tabular Macro (10)
    logger.info("Evaluating All-Inclusive Intermediate Feature Fusion...")
    scaler = StandardScaler()
    dino_pca = PCA(n_components=24, random_state=42).fit_transform(scaler.fit_transform(feats["dinov2"]))
    clip_pca = PCA(n_components=16, random_state=42).fit_transform(scaler.fit_transform(feats["clip"]))
    vlm_sc = scaler.fit_transform(feats["vlm"])
    tab_sc = scaler.fit_transform(feats["tab_macro"])

    X_fused_all = np.column_stack([dino_pca, clip_pca, vlm_sc, tab_sc])
    p_fused, m_fused = run_5x5_probe(X_fused_all, y, df_folds, use_pca=False)
    results["all_inclusive_feature_fusion"] = m_fused
    logger.info("All-Inclusive Feature Fusion (DINO + CLIP + VLM + Tabular): AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%% (%d/40), Sens@90=%.1f%% (%d/40)",
                m_fused["auroc"], m_fused["auprc"], m_fused["sens_at_80"]*100, m_fused["det_at_80"], m_fused["sens_at_90"]*100, m_fused["det_at_90"])

    # 5. Tri-Stream Consensus: 0.60 DINOv2 + 0.25 CLIP + 0.15 Tabular Macro
    # Tabular probe
    p_tab, m_tab = run_5x5_probe(feats["tab_macro"], y, df_folds, use_pca=False)
    p_tri = 0.55 * (rankdata(p_dino)/len(y)) + 0.30 * (rankdata(p_clip)/len(y)) + 0.15 * (rankdata(p_tab)/len(y))
    auc_tri = float(roc_auc_score(y, p_tri))
    ap_tri = float(average_precision_score(y, p_tri))
    neg_t = p_tri[y == 0]
    pos_t = p_tri[y == 1]
    det80_t = int(np.sum(pos_t >= np.percentile(neg_t, 80.0)))
    det90_t = int(np.sum(pos_t >= np.percentile(neg_t, 90.0)))
    results["tri_stream_rank_consensus"] = {
        "auroc": auc_tri, "auprc": ap_tri,
        "sens_at_80": det80_t / 40, "det_at_80": det80_t,
        "sens_at_90": det90_t / 40, "det_at_90": det90_t,
    }
    logger.info("Tri-Stream Rank Consensus (DINO + CLIP + Tabular): AUROC=%.4f, AUPRC=%.4f, Sens@80=%.1f%% (%d/40), Sens@90=%.1f%% (%d/40)",
                auc_tri, ap_tri, det80_t / 40 * 100, det80_t, det90_t / 40 * 100, det90_t)

    # Save to JSON
    out_json = REPORTS_DIR / "exp7_multistream_ensemble_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Multi-Stream Ensemble results saved to %s", out_json)


if __name__ == "__main__":
    main()

"""Out-of-Fold Visual Risk Classifier (Stream 4).

Trains a regularized visual probe on PCA-reduced DenseNet201 embeddings:
- Computes leakage-free out-of-fold P_vision for School A via 5-fold CV.
- Computes out-of-sample P_vision for School B.
- Saves calibrated visual risk probabilities to data/features/p_vision_predictions.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.visual_patch_extractor import extract_and_cache_cohort_embeddings


def compute_visual_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute comprehensive diagnostic metrics."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "auprc": None, "brier": None, "n_pos": n_pos, "n_neg": n_neg}

    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr

    valid_80 = np.where(spec >= 0.80)[0]
    sens_80 = float(np.max(tpr[valid_80])) if len(valid_80) > 0 else 0.0

    valid_90 = np.where(spec >= 0.90)[0]
    sens_90 = float(np.max(tpr[valid_90])) if len(valid_90) > 0 else 0.0

    j_scores = tpr + spec - 1.0
    best_idx = int(np.argmax(j_scores))
    youden_sens = float(tpr[best_idx])
    youden_spec = float(spec[best_idx])

    return {
        "auroc": auc,
        "auprc": auprc,
        "brier": brier,
        "sens_at_80_spec": sens_80,
        "sens_at_90_spec": sens_90,
        "youden_sens": youden_sens,
        "youden_spec": youden_spec,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def train_and_evaluate_visual_stream(
    embeddings_cache: str = "data/features/densenet_embeddings.npz",
    manifest_csv: str = "data/manifest.csv",
    folds_lookup_csv: str = "data/folds/student_folds_lookup.csv",
    output_predictions_json: str = "data/features/p_vision_predictions.json",
    n_pca_components: int = 12,
    random_state: int = 42,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Train regularized PCA + linear probe on DenseNet201 embeddings.

    Returns:
        tuple of (dict of student_id -> p_vision, dict of benchmark evaluation metrics).
    """
    # 1. Load or extract embeddings
    if not Path(embeddings_cache).exists():
        print(f"Embeddings cache not found at {embeddings_cache}, extracting now...")
        extract_and_cache_cohort_embeddings(output_cache=embeddings_cache)

    emb_data = np.load(embeddings_cache, allow_pickle=True)
    all_sids = emb_data["student_ids"].astype(str)
    all_embs = emb_data["student_embeddings"].astype(np.float32)

    sid_to_idx = {sid: idx for idx, sid in enumerate(all_sids)}

    # 2. Load manifest and folds lookup
    df_manifest = pd.read_csv(manifest_csv)
    manifest_map = {str(r["student_id"]).strip(): r for _, r in df_manifest.iterrows()}

    df_folds = pd.read_csv(folds_lookup_csv)
    fold_col = "repeat_0_fold" if "repeat_0_fold" in df_folds.columns else "repeat_0_fold_0"
    folds_map = {str(r["student_id"]).strip(): int(r[fold_col]) for _, r in df_folds.iterrows() if fold_col in r and pd.notna(r[fold_col])}

    # Separate School A and School B
    school_a_sids = [sid for sid in all_sids if sid in manifest_map and manifest_map[sid]["school"] == "school_a"]
    school_b_sids = [sid for sid in all_sids if sid in manifest_map and manifest_map[sid]["school"] == "school_b"]

    print(f"Loaded {len(school_a_sids)} School A students and {len(school_b_sids)} School B students.")

    # 3. Leakage-free Out-of-Fold training on School A
    X_a = np.array([all_embs[sid_to_idx[sid]] for sid in school_a_sids])
    y_a = np.array([int(manifest_map[sid]["label"]) for sid in school_a_sids])
    folds_a = np.array([folds_map.get(sid, 0) for sid in school_a_sids])

    p_vision_oof = np.zeros(len(school_a_sids), dtype=np.float32)

    unique_folds = sorted(list(set(folds_a)))
    for k in unique_folds:
        val_mask = (folds_a == k)
        train_mask = ~val_mask

        X_tr, y_tr = X_a[train_mask], y_a[train_mask]
        X_val = X_a[val_mask]

        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(n_pca_components, len(X_tr) - 1), random_state=random_state)),
            ("clf", LogisticRegression(C=0.1, penalty="l2", class_weight="balanced", random_state=random_state)),
        ])

        probe.fit(X_tr, y_tr)
        probs_val = probe.predict_proba(X_val)[:, 1]
        p_vision_oof[val_mask] = probs_val

    # School A cross-validated metrics
    met_a = compute_visual_metrics(y_a, p_vision_oof)
    print(f"School A Out-of-Fold Visual Stream Metrics: AUROC = {met_a['auroc']:.4f} | AUPRC = {met_a['auprc']:.4f}")

    # 4. Out-of-sample prediction on School B
    full_probe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(n_pca_components, len(X_a) - 1), random_state=random_state)),
        ("clf", LogisticRegression(C=0.1, penalty="l2", class_weight="balanced", random_state=random_state)),
    ])
    full_probe.fit(X_a, y_a)

    X_b = np.array([all_embs[sid_to_idx[sid]] for sid in school_b_sids])
    y_b = np.array([int(manifest_map[sid]["label"]) for sid in school_b_sids])
    p_vision_b = full_probe.predict_proba(X_b)[:, 1]

    met_b = compute_visual_metrics(y_b, p_vision_b)
    print(f"School B Out-of-Sample Visual Stream Metrics: AUROC = {met_b['auroc']:.4f} | AUPRC = {met_b['auprc']:.4f}")
    print(f"  Sens @ 80% Spec: {met_b['sens_at_80_spec']*100:.1f}% | Sens @ 90% Spec: {met_b['sens_at_90_spec']*100:.1f}%")
    print(f"  Youden Index: Sens {met_b['youden_sens']*100:.1f}%, Spec {met_b['youden_spec']*100:.1f}%")

    # 5. Assemble dictionary of predictions
    p_vision_dict: Dict[str, float] = {}
    for sid, p in zip(school_a_sids, p_vision_oof):
        p_vision_dict[sid] = float(p)
    for sid, p in zip(school_b_sids, p_vision_b):
        p_vision_dict[sid] = float(p)

    out_p = Path(output_predictions_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(p_vision_dict, f, indent=2)
    print(f"Saved {len(p_vision_dict)} visual predictions to: {out_p}")

    benchmark_summary = {
        "school_a_oof": met_a,
        "school_b_oos": met_b,
        "n_pca_components": n_pca_components,
    }
    return p_vision_dict, benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Stream 4 visual probe.")
    parser.add_argument("--embeddings", default="data/features/densenet_embeddings.npz")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--output", default="data/features/p_vision_predictions.json")
    parser.add_argument("--pca", type=int, default=12)
    args = parser.parse_args()

    train_and_evaluate_visual_stream(
        embeddings_cache=args.embeddings,
        manifest_csv=args.manifest,
        output_predictions_json=args.output,
        n_pca_components=args.pca,
    )

"""Gated Attention-MIL Head for Experiment 7 Frozen-Feature Representation.

Tests whether learned attention pooling over uncompressed tile embeddings beats
plain mean/std pooling.

Architecture:
1. Low-Rank Projection: Linear(D_tile, d) + LayerNorm + Dropout(0.2)  [d = 64]
2. Gated Attention Mechanism (Ilse et al., ICML 2018):
   a_t = w^T (tanh(V h_t) * sigmoid(U h_t))
   alpha_t = softmax(a_t) across the student's K_i tiles
3. Bag-Level Feature Aggregation:
   z_i = sum_t (alpha_t * h_t) in R^d
4. Linear Classifier:
   logit_i = w_cls^T z_i + b

Evaluates across the exact same 5-repeat x 5-fold Stratified Group CV splits (25 folds)
with cluster bootstrap 95% CIs.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.evaluate_exp7_probe import (
    calculate_sensitivity_at_specificity,
    compute_cluster_bootstrap_cis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = REPO_ROOT / "reports"


class StudentTileBagDataset(Dataset):
    """Dataset serving variable-sized bags of tile embeddings per student."""

    def __init__(
        self,
        student_ids: List[str],
        bags: List[torch.Tensor],  # Each tensor is (K_i, D_tile)
        labels: List[int],
    ):
        self.student_ids = student_ids
        self.bags = bags
        self.labels = labels

    def __len__(self) -> int:
        return len(self.student_ids)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, int]:
        return self.student_ids[idx], self.bags[idx], self.labels[idx]


def collate_student_bags(batch: List[Tuple[str, torch.Tensor, int]]) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates variable-sized bags into a padded tensor (B, max_K, D) with a boolean mask."""
    sids = [b[0] for b in batch]
    tensors = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)

    max_k = max(t.shape[0] for t in tensors)
    d = tensors[0].shape[1]
    b = len(batch)

    padded = torch.zeros(b, max_k, d, dtype=torch.float32)
    mask = torch.zeros(b, max_k, dtype=torch.bool)

    for i, t in enumerate(tensors):
        k = t.shape[0]
        padded[i, :k, :] = t
        mask[i, :k] = True

    return sids, padded, mask, labels


class GatedAttentionMIL(nn.Module):
    """Low-rank Gated Attention-MIL network for student tile bags."""

    def __init__(self, in_dim: int, proj_dim: int = 64, attn_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention
        self.attn_V = nn.Linear(proj_dim, attn_dim, bias=False)
        self.attn_U = nn.Linear(proj_dim, attn_dim, bias=False)
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)

        # Classifier
        self.classifier = nn.Linear(proj_dim, 1)

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            bags: (B, max_K, in_dim)
            mask: (B, max_K) boolean mask (True = valid tile)

        Returns:
            logits: (B,)
            attn_weights: (B, max_K)
        """
        # Project tiles: (B, max_K, proj_dim)
        h = self.projector(bags)

        # Attention gating:
        # v = tanh(V h) in (B, max_K, attn_dim)
        # u = sigmoid(U h) in (B, max_K, attn_dim)
        v = torch.tanh(self.attn_V(h))
        u = torch.sigmoid(self.attn_U(h))
        scores = self.attn_w(v * u).squeeze(-1)  # (B, max_K)

        # Mask invalid padding tiles with -1e9 before softmax
        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)  # (B, max_K)

        # Weighted sum: (B, proj_dim)
        z = torch.bmm(attn.unsqueeze(1), h).squeeze(1)

        # Classification logit: (B,)
        logits = self.classifier(z).squeeze(-1)
        return logits, attn


def train_and_eval_mil_fold(
    train_dataset: StudentTileBagDataset,
    test_dataset: StudentTileBagDataset,
    in_dim: int,
    device: torch.device,
    epochs: int = 15,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    seed: int = 42,
) -> np.ndarray:
    """Trains Gated Attention-MIL on outer train fold and returns test probabilities."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GatedAttentionMIL(in_dim=in_dim, proj_dim=64, attn_dim=16, dropout=0.2).to(device)

    # Calculate class weight
    y_tr = np.array(train_dataset.labels)
    n_pos = max(1, np.sum(y_tr == 1))
    n_neg = max(1, np.sum(y_tr == 0))
    pos_weight = torch.tensor([float(n_neg / n_pos)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_student_bags,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_student_bags,
    )

    # Stochastic Weight Averaging over last 5 epochs
    swa_weights = None
    swa_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for _, bags, mask, labels in train_loader:
            bags, mask, labels = bags.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(bags, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch >= 10:
            swa_count += 1
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if swa_weights is None:
                swa_weights = state
            else:
                for k in swa_weights:
                    swa_weights[k] += state[k]

    if swa_weights is not None and swa_count > 0:
        for k in swa_weights:
            swa_weights[k] /= swa_count
        model.load_state_dict({k: v.to(device) for k, v in swa_weights.items()})

    model.eval()
    test_probs: List[float] = []

    with torch.no_grad():
        for _, bags, mask, _ in test_loader:
            bags, mask = bags.to(device), mask.to(device)
            logits, _ = model(bags, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs.tolist())

    return np.array(test_probs, dtype=np.float32)


def run_exp7_attention_mil(
    features_npz_path: Path,
    model_tag: str = "dinov2_small",
    script_mode: str = "all",  # "all", "hindi", "english"
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs Gated Attention-MIL across the 5x5 repeated CV splits."""
    if output_dir is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = REPORTS_DIR

    logger.info("Loading tile features from %s...", features_npz_path)
    data = np.load(features_npz_path, allow_pickle=True)

    tile_desc = data["tile_descriptors"]
    tile_sids = data["tile_student_ids"]
    tile_scripts = data["tile_scripts"]

    student_ids = data["student_ids"].tolist()
    y_true = data["student_labels"].astype(int)
    n_students = len(student_ids)

    # Filter tiles by script if requested
    if script_mode == "hindi":
        mask = (tile_scripts == "devanagari")
    elif script_mode == "english":
        mask = (tile_scripts == "latin")
    else:
        mask = np.ones(len(tile_scripts), dtype=bool)

    tile_desc = tile_desc[mask]
    tile_sids = tile_sids[mask]

    # Pre-build tile bags per student
    student_bags_map: Dict[str, List[np.ndarray]] = {sid: [] for sid in student_ids}
    for sid, desc in zip(tile_sids, tile_desc):
        student_bags_map[sid].append(desc)

    student_bags_tensors: List[torch.Tensor] = []
    in_dim = tile_desc.shape[1]

    for sid in student_ids:
        tiles_list = student_bags_map[sid]
        if len(tiles_list) == 0:
            t = torch.zeros(1, in_dim, dtype=torch.float32)
        else:
            t = torch.from_numpy(np.array(tiles_list, dtype=np.float32))
        student_bags_tensors.append(t)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Attention-MIL on %s (%d tiles, %d students, script: %s)",
                device, len(tile_desc), n_students, script_mode)

    oof_mil_repeats = np.zeros((5, n_students), dtype=np.float32)
    start_time = time.time()

    for r in range(5):
        repeat_fold_col = data[f"repeat_{r}_fold"]

        for f in range(5):
            test_mask = (repeat_fold_col == f)
            train_mask = (repeat_fold_col != f)

            tr_sids = [student_ids[i] for i in np.where(train_mask)[0]]
            tr_bags = [student_bags_tensors[i] for i in np.where(train_mask)[0]]
            tr_labels = [int(y_true[i]) for i in np.where(train_mask)[0]]

            te_sids = [student_ids[i] for i in np.where(test_mask)[0]]
            te_bags = [student_bags_tensors[i] for i in np.where(test_mask)[0]]
            te_labels = [int(y_true[i]) for i in np.where(test_mask)[0]]

            train_ds = StudentTileBagDataset(tr_sids, tr_bags, tr_labels)
            test_ds = StudentTileBagDataset(te_sids, te_bags, te_labels)

            seed = 142 + r * 10 + f
            p_te = train_and_eval_mil_fold(
                train_ds, test_ds, in_dim, device, epochs=15, seed=seed
            )
            oof_mil_repeats[r, test_mask] = p_te

        logger.info("Completed Repeat %d/5 in %.1fs", r + 1, time.time() - start_time)

    p_mil_mean = np.mean(oof_mil_repeats, axis=0)

    auc = float(roc_auc_score(y_true, p_mil_mean))
    pr = float(average_precision_score(y_true, p_mil_mean))
    brier = float(brier_score_loss(y_true, p_mil_mean))
    det80, s80, th80 = calculate_sensitivity_at_specificity(y_true, p_mil_mean, 0.80)
    det90, s90, th90 = calculate_sensitivity_at_specificity(y_true, p_mil_mean, 0.90)
    cis = compute_cluster_bootstrap_cis(y_true, p_mil_mean, np.array(student_ids))

    logger.info("=" * 80)
    logger.info("ATTENTION-MIL BENCHMARK RESULTS (%s - %s)", model_tag, script_mode)
    logger.info("  AUROC: %.4f [%.3f, %.3f]", auc, cis["auroc_ci"][0], cis["auroc_ci"][1])
    logger.info("  AUPRC: %.4f [%.3f, %.3f]", pr, cis["auprc_ci"][0], cis["auprc_ci"][1])
    logger.info("  Sens @ 80%% Spec: %.1f%% (%d/40) [%.1f%%, %.1f%%]",
                s80 * 100, det80, cis["sens80_ci"][0] * 100, cis["sens80_ci"][1] * 100)
    logger.info("  Sens @ 90%% Spec: %.1f%% (%d/40) [%.1f%%, %.1f%%]",
                s90 * 100, det90, cis["sens90_ci"][0] * 100, cis["sens90_ci"][1] * 100)
    logger.info("=" * 80)

    results = {
        "model_tag": f"{model_tag}_mil_{script_mode}",
        "script_mode": script_mode,
        "auroc": auc,
        "auroc_ci": list(cis["auroc_ci"]),
        "auprc": pr,
        "auprc_ci": list(cis["auprc_ci"]),
        "brier": brier,
        "sens80": s80,
        "det80": det80,
        "sens90": s90,
        "det90": det90,
    }

    # Save out-of-fold predictions
    oof_csv = output_dir / f"exp7_{model_tag}_mil_{script_mode}_oof_predictions.csv"
    pd.DataFrame({
        "student_id": student_ids,
        "label": y_true,
        "p_mil": p_mil_mean,
    }).to_csv(oof_csv, index=False)
    logger.info("Saved MIL OOF predictions to %s", oof_csv)

    json_path = output_dir / f"exp7_{model_tag}_mil_{script_mode}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Gated Attention-MIL on Experiment 7 Tile Embeddings.")
    parser.add_argument("--features", type=str, default=None, help="Path to .npz feature cache.")
    parser.add_argument("--model-tag", type=str, default="dinov2_small", help="Model tag identifier.")
    parser.add_argument("--script-mode", choices=["all", "hindi", "english"], default="all",
                        help="Script partition for tile bags.")
    args = parser.parse_args()

    feat_path = Path(args.features) if args.features else FEATURES_DIR / f"exp7_{args.model_tag}_tiles.npz"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing features file: {feat_path}")

    run_exp7_attention_mil(feat_path, model_tag=args.model_tag, script_mode=args.script_mode)


if __name__ == "__main__":
    main()

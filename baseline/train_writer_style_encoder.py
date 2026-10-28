"""
Experiment 7, Candidate 2: Self-Supervised Contrastive Writer-Style Encoder
Trains a metric-learning projection network on native stroke-scale tile descriptors
to maximize similarity between tiles written by the same student (across tasks and scripts)
while pushing apart tiles written by different students.

Crucially:
- Zero dysgraphia labels are used during pretraining.
- Pretraining is executed strictly inside the training fold (172 students) of each fold,
  guaranteeing zero leakage to held-out test students (43 students).
- A downstream linear probe is evaluated on the frozen learned writer-style embeddings.
"""

import argparse
import json
import logging
import random
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("writer_style_encoder")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Khosla et al., NeurIPS 2020).
    Here 'labels' are student IDs (writer identities).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [N, D] (L2 normalized)
        labels: [N] (integer student IDs)
        """
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Dot-product similarity matrix scaled by temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability, subtract max per row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        # Avoid division by zero for anchors without other positives in batch
        valid_anchors = mask_pos_pairs > 0
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / mask_pos_pairs[valid_anchors]
        loss = -mean_log_prob_pos.mean()
        return loss


class WriterStyleEncoder(nn.Module):
    """
    Projector network mapping tile descriptors into an L2-normalized writer-style space.
    """

    def __init__(self, in_dim: int = 1152, hidden_dim: int = 512, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class StudentBalancedBatchSampler:
    """
    Samples N_students per batch, and K_tiles per student,
    ensuring every batch has multiple positive pairs per writer.
    """

    def __init__(
        self,
        student_to_indices: Dict[int, List[int]],
        n_students_per_batch: int = 32,
        k_tiles_per_student: int = 4,
        n_batches_per_epoch: int = 100,
        seed: int = 42,
    ):
        self.student_to_indices = {s: idxs for s, idxs in student_to_indices.items() if len(idxs) >= 2}
        self.student_list = list(self.student_to_indices.keys())
        self.n_students = n_students_per_batch
        self.k_tiles = k_tiles_per_student
        self.n_batches = n_batches_per_epoch
        self.rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            sampled_students = self.rng.sample(
                self.student_list, min(self.n_students, len(self.student_list))
            )
            for s in sampled_students:
                idxs = self.student_to_indices[s]
                if len(idxs) >= self.k_tiles:
                    sampled_idxs = self.rng.sample(idxs, self.k_tiles)
                else:
                    sampled_idxs = self.rng.choices(idxs, k=self.k_tiles)
                batch.extend(sampled_idxs)
            yield batch

    def __len__(self):
        return self.n_batches


def train_encoder_on_fold(
    train_tiles: np.ndarray,
    train_student_int_ids: np.ndarray,
    in_dim: int = 1152,
    hidden_dim: int = 512,
    out_dim: int = 128,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> WriterStyleEncoder:
    """Train WriterStyleEncoder on training fold using SupCon loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build student -> indices mapping
    student_to_indices: Dict[int, List[int]] = {}
    for idx, s_id in enumerate(train_student_int_ids):
        student_to_indices.setdefault(int(s_id), []).append(idx)

    model = WriterStyleEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = SupConLoss(temperature=0.07)

    X_tensor = torch.from_numpy(train_tiles).float().to(device)
    y_tensor = torch.from_numpy(train_student_int_ids).long().to(device)

    sampler = StudentBalancedBatchSampler(
        student_to_indices,
        n_students_per_batch=32,
        k_tiles_per_student=4,
        n_batches_per_epoch=80,
        seed=seed,
    )

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        n_steps = 0
        for batch_idxs in sampler:
            batch_idxs_t = torch.tensor(batch_idxs, dtype=torch.long, device=device)
            feats = X_tensor[batch_idxs_t]
            labels = y_tensor[batch_idxs_t]

            optimizer.zero_grad()
            z = model(feats)
            loss = criterion(z, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        scheduler.step()

    model.eval()
    return model


def extract_student_style_vectors(
    model: WriterStyleEncoder,
    all_tiles: np.ndarray,
    all_student_ids: np.ndarray,
    all_scripts: np.ndarray,
    unique_student_ids: np.ndarray,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, np.ndarray]:
    """Extract mean and std student-level style embeddings across Hindi, English, and Joint."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(all_tiles).float().to(device)
        # Compute in chunks to avoid any VRAM spike
        chunk_size = 4096
        z_list = []
        for i in range(0, len(X_tensor), chunk_size):
            chunk = X_tensor[i : i + chunk_size]
            z_chunk = model(chunk).cpu().numpy()
            z_list.append(z_chunk)
        all_z = np.vstack(z_list)

    out_dim = all_z.shape[1]
    n_students = len(unique_student_ids)

    hindi_student_style = np.zeros((n_students, 2 * out_dim), dtype=np.float32)
    english_student_style = np.zeros((n_students, 2 * out_dim), dtype=np.float32)
    joint_student_style = np.zeros((n_students, 4 * out_dim), dtype=np.float32)

    for i, s_id in enumerate(unique_student_ids):
        s_mask = (all_student_ids == s_id)
        s_z = all_z[s_mask]
        s_scripts = all_scripts[s_mask]

        h_mask = (s_scripts == "devanagari")
        e_mask = (s_scripts == "latin")

        h_z = s_z[h_mask] if np.any(h_mask) else s_z
        e_z = s_z[e_mask] if np.any(e_mask) else s_z

        h_mean = np.mean(h_z, axis=0) if len(h_z) > 0 else np.zeros(out_dim)
        h_std = np.std(h_z, axis=0) if len(h_z) > 0 else np.zeros(out_dim)
        hindi_student_style[i] = np.concatenate([h_mean, h_std])

        e_mean = np.mean(e_z, axis=0) if len(e_z) > 0 else np.zeros(out_dim)
        e_std = np.std(e_z, axis=0) if len(e_z) > 0 else np.zeros(out_dim)
        english_student_style[i] = np.concatenate([e_mean, e_std])

        joint_student_style[i] = np.concatenate([hindi_student_style[i], english_student_style[i]])

    return {
        "hindi": hindi_student_style,
        "english": english_student_style,
        "joint": joint_student_style,
    }


def evaluate_writer_style_encoder(
    features_npz_path: Path,
    epochs_per_fold: int = 25,
    repeat_idx: int = 0,
) -> Dict[str, Any]:
    """
    Benchmark Candidate 2 (Self-Supervised Writer-Style Encoder)
    across all 5 folds of the evaluation repeat.
    """
    logger.info("Loading precomputed tiles from %s...", features_npz_path)
    data = np.load(features_npz_path, allow_pickle=True)

    all_tiles = data["tile_descriptors"]
    all_tile_students = data["tile_student_ids"]
    all_tile_scripts = data["tile_scripts"]

    unique_students = data["student_ids"]
    y_true = data["student_labels"].astype(int)
    schools = data["student_schools"]
    grades = data["student_grades"]

    fold_col = data[f"repeat_{repeat_idx}_fold"]
    n_students = len(unique_students)

    # Map student string IDs to integers
    student_id_map = {s: i for i, s in enumerate(unique_students)}
    all_tile_student_ints = np.array([student_id_map[s] for s in all_tile_students], dtype=np.int64)

    logger.info(
        "Candidate 2 Dataset: %d tiles, %d unique students (%d positives)",
        len(all_tiles), n_students, int(np.sum(y_true == 1))
    )

    oof_probs_hindi = np.zeros(n_students, dtype=np.float32)
    oof_probs_english = np.zeros(n_students, dtype=np.float32)
    oof_probs_blend = np.zeros(n_students, dtype=np.float32)
    oof_probs_joint = np.zeros(n_students, dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    start_time = time.time()

    for f in range(5):
        test_mask = (fold_col == f)
        train_mask = (fold_col != f)

        test_student_set = set(unique_students[test_mask])
        train_tile_mask = np.isin(all_tile_students, unique_students[train_mask])

        logger.info(
            "Fold %d: Training Writer-Style Encoder on %d tiles (172 students)...",
            f, int(np.sum(train_tile_mask))
        )

        fold_train_tiles = all_tiles[train_tile_mask]
        fold_train_student_ints = all_tile_student_ints[train_tile_mask]

        # Train encoder strictly on training fold tiles
        model = train_encoder_on_fold(
            train_tiles=fold_train_tiles,
            train_student_int_ids=fold_train_student_ints,
            in_dim=all_tiles.shape[1],
            hidden_dim=512,
            out_dim=128,
            epochs=epochs_per_fold,
            device=device,
            seed=42 + f * 10,
        )

        # Extract student style vectors for all 215 students using this fold's trained encoder
        styles = extract_student_style_vectors(
            model=model,
            all_tiles=all_tiles,
            all_student_ids=all_tile_students,
            all_scripts=all_tile_scripts,
            unique_student_ids=unique_students,
            device=device,
        )

        y_tr = y_true[train_mask]
        y_te = y_true[test_mask]

        # Fit downstream regularized LR probe for each stream
        for name, arr, oof_arr in [
            ("hindi", styles["hindi"], oof_probs_hindi),
            ("english", styles["english"], oof_probs_english),
            ("joint", styles["joint"], oof_probs_joint),
        ]:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(arr[train_mask])
            X_te = scaler.transform(arr[test_mask])

            # Inner tuning for regularizer C
            best_c = 0.1
            best_score = -1.0
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + f)
            for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
                val_aucs = []
                for sub_tr, sub_val in skf.split(X_tr, y_tr):
                    clf = LogisticRegression(C=c_cand, class_weight="balanced", max_iter=1000, random_state=42)
                    clf.fit(X_tr[sub_tr], y_tr[sub_tr])
                    p_val = clf.predict_proba(X_tr[sub_val])[:, 1]
                    val_aucs.append(roc_auc_score(y_tr[sub_val], p_val))
                if np.mean(val_aucs) > best_score:
                    best_score = np.mean(val_aucs)
                    best_c = c_cand

            final_clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=42)
            final_clf.fit(X_tr, y_tr)
            oof_arr[test_mask] = final_clf.predict_proba(X_te)[:, 1]

        oof_probs_blend[test_mask] = 0.5 * (oof_probs_hindi[test_mask] + oof_probs_english[test_mask])

    elapsed = time.time() - start_time
    logger.info("Completed 5-fold evaluation in %.1fs", elapsed)

    def calc_metrics(y_p: np.ndarray, name: str) -> Dict[str, Any]:
        auc = float(roc_auc_score(y_true, y_p))
        ap = float(average_precision_score(y_true, y_p))
        brier = float(brier_score_loss(y_true, y_p))

        neg_scores = y_p[y_true == 0]
        pos_scores = y_p[y_true == 1]
        th80 = float(np.percentile(neg_scores, 80.0))
        det80 = int(np.sum(pos_scores >= th80))
        th90 = float(np.percentile(neg_scores, 90.0))
        det90 = int(np.sum(pos_scores >= th90))

        return {
            "name": name,
            "auroc": auc,
            "auprc": ap,
            "brier": brier,
            "sens_at_80_spec": det80 / len(pos_scores),
            "det_at_80_spec": det80,
            "sens_at_90_spec": det90 / len(pos_scores),
            "det_at_90_spec": det90,
        }

    results = {
        "candidate": "Candidate 2: Self-Supervised Contrastive Writer-Style Encoder",
        "hindi": calc_metrics(oof_probs_hindi, "Writer-Style Hindi"),
        "english": calc_metrics(oof_probs_english, "Writer-Style English"),
        "blend": calc_metrics(oof_probs_blend, "Writer-Style Decision Blend"),
        "joint": calc_metrics(oof_probs_joint, "Writer-Style Joint"),
    }

    logger.info("\n" + "=" * 80)
    logger.info("CANDIDATE 2 SCOREBOARD (All 215 Students, 5-Fold OOF)")
    logger.info("=" * 80)
    logger.info(f"{'Stream':<28} | {'AUROC':<7} | {'AUPRC':<7} | {'Sens@80%':<8} | {'Sens@90%':<8}")
    logger.info("-" * 80)
    for k in ["hindi", "english", "blend", "joint"]:
        m = results[k]
        logger.info(
            f"{m['name']:<28} | {m['auroc']:.4f}  | {m['auprc']:.4f}  | "
            f"{m['sens_at_80_spec']*100:.1f}% ({m['det_at_80_spec']}/40) | "
            f"{m['sens_at_90_spec']*100:.1f}% ({m['det_at_90_spec']}/40)"
        )
    logger.info("=" * 80)

    out_json = REPORTS_DIR / "exp7_candidate2_writer_style_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Candidate 2: Writer-Style Encoder")
    parser.add_argument(
        "--features",
        type=Path,
        default=DATA_DIR / "features" / "exp7_dinov2_small_tiles.npz",
        help="Path to precomputed features NPZ",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Epochs per fold",
    )
    args = parser.parse_args()

    evaluate_writer_style_encoder(
        features_npz_path=args.features,
        epochs_per_fold=args.epochs,
    )


if __name__ == "__main__":
    main()


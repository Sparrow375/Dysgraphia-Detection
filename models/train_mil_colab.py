"""Strategy B: Handwriting ViT Encoder + Gated Attention MIL (Colab & CLI Runner).

Dual-ViT Feature Extraction (TrOCR English + Devanagari) + Gated Attention Pooling.
Evaluates across the locked 5-repeat x 5-fold Stratified Group K-Fold benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Gated Attention MIL Architecture (~581 Trainable Parameters)
# ==============================================================================

class GatedAttention(nn.Module):
    """Ilse et al. (ICML 2018) Gated Attention Pooling Layer."""

    def __init__(self, in_features: int = 64, attention_dim: int = 4):
        super().__init__()
        self.in_features = in_features
        self.attention_dim = attention_dim
        self.attention_v = nn.Linear(in_features, attention_dim, bias=False)
        self.attention_u = nn.Linear(in_features, attention_dim, bias=False)
        self.attention_w = nn.Linear(attention_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.attention_v.weight)
        nn.init.xavier_uniform_(self.attention_u.weight)
        nn.init.xavier_uniform_(self.attention_w.weight)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v = torch.tanh(self.attention_v(h))
        u = torch.sigmoid(self.attention_u(h))
        scores = self.attention_w(v * u)

        if mask is not None:
            scores = scores.squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=-1)
            weights = weights.unsqueeze(-1)
        else:
            weights = torch.softmax(scores, dim=1)

        z = torch.sum(weights * h, dim=1)
        return z, weights.squeeze(-1)


class GatedAttentionMIL(nn.Module):
    """Multiple Instance Learning Network with Gated Attention Pooling."""

    def __init__(
        self,
        encoder_dim: int = 64,
        projected_dim: int = 64,
        attention_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.projected_dim = projected_dim

        if encoder_dim != projected_dim:
            self.projection = nn.Sequential(
                nn.Linear(encoder_dim, projected_dim),
                nn.LayerNorm(projected_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.projection = nn.Identity()

        self.attention = GatedAttention(
            in_features=projected_dim,
            attention_dim=attention_dim,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(projected_dim, 1),
        )
        self._init_classifier()

    def _init_classifier(self) -> None:
        if isinstance(self.classifier[-1], nn.Linear):
            nn.init.xavier_uniform_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proj = self.projection(h)
        z, attn_weights = self.attention(proj, mask=mask)
        logits = self.classifier(z).squeeze(-1)
        return logits, attn_weights

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# 2. Dataset & Collation
# ==============================================================================

class StudentBagDataset(Dataset):
    """Dataset storing precomputed word embeddings grouped by student."""

    def __init__(
        self,
        student_bags: List[torch.Tensor],
        labels: List[int],
        student_ids: List[str],
        word_records: Optional[List[List[Dict[str, Any]]]] = None,
    ):
        self.student_bags = student_bags
        self.labels = labels
        self.student_ids = student_ids
        self.word_records = word_records

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "features": self.student_bags[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "student_id": self.student_ids[idx],
            "num_words": self.student_bags[idx].shape[0],
        }
        if self.word_records is not None:
            item["word_records"] = self.word_records[idx]
        return item


def collate_student_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    max_words = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]

    padded_features = torch.zeros((batch_size, max_words, feature_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.float32)
    student_ids: List[str] = []

    for i, item in enumerate(batch):
        bag = item["features"]
        k = bag.shape[0]
        padded_features[i, :k, :] = bag
        mask[i, :k] = True
        labels[i] = item["label"]
        student_ids.append(item["student_id"])

    return {
        "features": padded_features,
        "mask": mask,
        "labels": labels,
        "student_ids": student_ids,
    }


# ==============================================================================
# 3. Dual-ViT Feature Extraction (Colab GPU)
# ==============================================================================

def extract_vit_embeddings_for_words(
    dataset_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
    project_dim: int = 64,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    """Extract frozen TrOCR ViT embeddings for all words in dataset.
    
    Uses microsoft/trocr-small-handwritten for Latin script and
    paudelanil/trocr-devanagari-2 for Devanagari script.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    index_csv = dataset_dir / "words_index.csv"
    with open(index_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    logger.info(f"Loaded {len(records)} word crop records from {index_csv}")

    # Separate records by script
    latin_records = [r for r in records if r["script"] == "latin"]
    devanagari_records = [r for r in records if r["script"] == "devanagari"]

    logger.info(f"Latin words: {len(latin_records)} | Devanagari words: {len(devanagari_records)}")

    word_embeddings: Dict[str, torch.Tensor] = {}

    # 1. Process Latin Crops (English)
    logger.info("Loading English TrOCR: microsoft/trocr-small-handwritten...")
    eng_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    eng_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten").to(device)
    eng_encoder = eng_model.encoder
    eng_encoder.eval()
    for p in eng_encoder.parameters():
        p.requires_grad = False

    logger.info(f"Extracting ViT representations for {len(latin_records)} English words...")
    for i in tqdm(range(0, len(latin_records), batch_size), desc="English ViT"):
        batch_recs = latin_records[i : i + batch_size]
        pil_images = []
        valid_recs = []
        for r in batch_recs:
            img_p = dataset_dir / r["rel_path"]
            if img_p.exists():
                try:
                    img = Image.open(img_p).convert("RGB")
                    pil_images.append(img)
                    valid_recs.append(r)
                except Exception:
                    pass

        if not pil_images:
            continue

        inputs = eng_processor(pil_images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            hidden = eng_encoder(inputs).last_hidden_state  # (B, S, D_enc)
            # Mean pool across spatial patches
            embeds = hidden.mean(dim=1).cpu()  # (B, D_enc)

        for rec, emb in zip(valid_recs, embeds):
            word_embeddings[rec["word_id"]] = emb

    del eng_model, eng_encoder, eng_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Process Devanagari Crops (Hindi)
    logger.info("Loading Devanagari TrOCR: paudelanil/trocr-devanagari-2...")
    hin_processor = TrOCRProcessor.from_pretrained("paudelanil/trocr-devanagari-2")
    hin_model = VisionEncoderDecoderModel.from_pretrained("paudelanil/trocr-devanagari-2").to(device)
    hin_encoder = hin_model.encoder
    hin_encoder.eval()
    for p in hin_encoder.parameters():
        p.requires_grad = False

    logger.info(f"Extracting ViT representations for {len(devanagari_records)} Hindi words...")
    for i in tqdm(range(0, len(devanagari_records), batch_size), desc="Hindi ViT"):
        batch_recs = devanagari_records[i : i + batch_size]
        pil_images = []
        valid_recs = []
        for r in batch_recs:
            img_p = dataset_dir / r["rel_path"]
            if img_p.exists():
                try:
                    img = Image.open(img_p).convert("RGB")
                    pil_images.append(img)
                    valid_recs.append(r)
                except Exception:
                    pass

        if not pil_images:
            continue

        inputs = hin_processor(pil_images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            hidden = hin_encoder(inputs).last_hidden_state
            embeds = hidden.mean(dim=1).cpu()

        for rec, emb in zip(valid_recs, embeds):
            word_embeddings[rec["word_id"]] = emb

    del hin_model, hin_encoder, hin_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Project to Shared Space (if dimensions differ or to reduce to project_dim)
    # Check dimensions
    dims = {k: v.shape[0] for k, v in list(word_embeddings.items())[:10]}
    logger.info(f"Raw ViT embedding dimensions sample: {dims}")

    # Standardize via PCA projection to project_dim
    from sklearn.decomposition import PCA
    all_keys = list(word_embeddings.keys())
    # Group by dimension
    dim_groups: Dict[int, List[str]] = {}
    for k in all_keys:
        d = word_embeddings[k].shape[0]
        dim_groups.setdefault(d, []).append(k)

    projected_embeddings: Dict[str, torch.Tensor] = {}
    for dim, keys in dim_groups.items():
        logger.info(f"Projecting {len(keys)} embeddings of dimension {dim} to {project_dim}...")
        mat = torch.stack([word_embeddings[k] for k in keys]).numpy()
        pca = PCA(n_components=min(project_dim, dim, mat.shape[0]))
        proj_mat = pca.fit_transform(mat)
        # Pad with zeros if n_components < project_dim
        if proj_mat.shape[1] < project_dim:
            pad = np.zeros((proj_mat.shape[0], project_dim - proj_mat.shape[1]))
            proj_mat = np.hstack([proj_mat, pad])
        for idx, k in enumerate(keys):
            projected_embeddings[k] = torch.tensor(proj_mat[idx], dtype=torch.float32)

    return projected_embeddings, records


# ==============================================================================
# 4. 5x5 Cross-Validation Training & Evaluation Loop
# ==============================================================================

def train_and_evaluate_5x5(
    student_bags_by_id: Dict[str, torch.Tensor],
    student_meta_by_id: Dict[str, Dict[str, Any]],
    projected_dim: int = 64,
    attention_dim: int = 4,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train and evaluate Gated-Attention MIL across locked 5-repeat x 5-fold CV splits."""
    all_sids = sorted(list(student_bags_by_id.keys()))
    pos_count = sum(1 for sid in all_sids if student_meta_by_id[sid]["label"] == 1)
    neg_count = len(all_sids) - pos_count
    pos_weight = float(neg_count / max(1, pos_count))

    logger.info(f"Starting 5-repeat x 5-fold CV on {len(all_sids)} students (Pos={pos_count}, Neg={neg_count}, weight={pos_weight:.2f})")

    repeat_aurocs: List[float] = []
    repeat_auprcs: List[float] = []
    oof_predictions: Dict[str, List[float]] = {sid: [] for sid in all_sids}
    oof_attention: Dict[str, List[float]] = {}

    for r in range(5):
        fold_probs: Dict[str, float] = {}

        for f in range(5):
            # Split train and val
            train_sids = [sid for sid in all_sids if student_meta_by_id[sid][f"repeat_{r}_fold"] != f]
            val_sids = [sid for sid in all_sids if student_meta_by_id[sid][f"repeat_{r}_fold"] == f]

            train_bags = [student_bags_by_id[sid] for sid in train_sids]
            train_labels = [student_meta_by_id[sid]["label"] for sid in train_sids]

            val_bags = [student_bags_by_id[sid] for sid in val_sids]
            val_labels = [student_meta_by_id[sid]["label"] for sid in val_sids]

            train_ds = StudentBagDataset(train_bags, train_labels, train_sids)
            val_ds = StudentBagDataset(val_bags, val_labels, val_sids)

            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_student_bags)
            val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_student_bags)

            model = GatedAttentionMIL(
                encoder_dim=projected_dim,
                projected_dim=projected_dim,
                attention_dim=attention_dim,
                dropout=0.15,
            ).to(device)

            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

            best_val_loss = float("inf")
            best_probs: Dict[str, float] = {}
            best_attns: Dict[str, np.ndarray] = {}

            for epoch in range(epochs):
                model.train()
                for batch in train_loader:
                    feats = batch["features"].to(device)
                    mask = batch["mask"].to(device)
                    lbls = batch["labels"].to(device)

                    optimizer.zero_grad()
                    logits, _ = model(feats, mask=mask)
                    loss = criterion(logits, lbls)
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                # Evaluate on validation fold
                model.eval()
                val_loss = 0.0
                curr_probs: Dict[str, float] = {}
                curr_attns: Dict[str, np.ndarray] = {}

                with torch.no_grad():
                    for batch in val_loader:
                        feats = batch["features"].to(device)
                        mask = batch["mask"].to(device)
                        lbls = batch["labels"].to(device)

                        logits, attns = model(feats, mask=mask)
                        loss = criterion(logits, lbls)
                        val_loss += loss.item() * len(lbls)

                        probs = torch.sigmoid(logits).cpu().numpy()
                        attns_np = attns.cpu().numpy()

                        for sid_b, prob_b, attn_b, m_b in zip(batch["student_ids"], probs, attns_np, mask.cpu().numpy()):
                            curr_probs[sid_b] = float(prob_b)
                            curr_attns[sid_b] = attn_b[m_b]

                val_loss /= max(1, len(val_sids))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_probs = curr_probs
                    best_attns = curr_attns

            fold_probs.update(best_probs)
            if r == 0:
                oof_attention.update(best_attns)

        # Record repeat metrics
        y_true = [student_meta_by_id[sid]["label"] for sid in all_sids]
        y_pred = [fold_probs[sid] for sid in all_sids]
        rep_auroc = float(roc_auc_score(y_true, y_pred))
        rep_auprc = float(average_precision_score(y_true, y_pred))
        repeat_aurocs.append(rep_auroc)
        repeat_auprcs.append(rep_auprc)

        logger.info(f"Repeat {r}: AUROC = {rep_auroc:.4f} | AUPRC = {rep_auprc:.4f}")
        for sid in all_sids:
            oof_predictions[sid].append(fold_probs[sid])

    # Aggregate ensemble predictions
    mean_oof = {sid: float(np.mean(probs)) for sid, probs in oof_predictions.items()}
    y_true_all = [student_meta_by_id[sid]["label"] for sid in all_sids]
    y_pred_all = [mean_oof[sid] for sid in all_sids]

    final_auroc = float(roc_auc_score(y_true_all, y_pred_all))
    final_auprc = float(average_precision_score(y_true_all, y_pred_all))
    final_brier = float(brier_score_loss(y_true_all, y_pred_all))

    # Sensitivity @ 80% & 90% Specificity
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true_all, y_pred_all)
    sens_80 = float(tpr[np.where(fpr <= 0.20)[0][-1]]) if np.any(fpr <= 0.20) else 0.0
    sens_90 = float(tpr[np.where(fpr <= 0.10)[0][-1]]) if np.any(fpr <= 0.10) else 0.0

    logger.info("=================================================================")
    logger.info("STRATEGY B (Gated-Attention MIL) 5x5 CV SCOREBOARD:")
    logger.info(f"  • AUROC: {final_auroc:.4f} [5-repeat mean: {np.mean(repeat_aurocs):.4f} ± {np.std(repeat_aurocs):.4f}]")
    logger.info(f"  • AUPRC: {final_auprc:.4f} [5-repeat mean: {np.mean(repeat_auprcs):.4f} ± {np.std(repeat_auprcs):.4f}]")
    logger.info(f"  • Brier Loss: {final_brier:.4f}")
    logger.info(f"  • Sens @ 80% Spec: {sens_80:.1%}")
    logger.info(f"  • Sens @ 90% Spec: {sens_90:.1%}")
    logger.info("=================================================================")

    return {
        "auroc": final_auroc,
        "auprc": final_auprc,
        "brier": final_brier,
        "sens_80_spec": sens_80,
        "sens_90_spec": sens_90,
        "repeat_aurocs": repeat_aurocs,
        "repeat_auprcs": repeat_auprcs,
        "mean_oof_predictions": mean_oof,
        "all_repeat_predictions": oof_predictions,
        "sample_attentions": oof_attention,
    }


def main():
    parser = argparse.ArgumentParser(description="Strategy B: Handwriting ViT + Gated Attention MIL")
    parser.add_argument("--data_dir", type=str, default="data/word_crops_dataset", help="Path to unzipped word crops dataset")
    parser.add_argument("--project_dim", type=int, default=64, help="Latent feature dimension")
    parser.add_argument("--attention_dim", type=int, default=4, help="Attention projection dimension")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv", type=str, default="p_mil_vit_predictions.csv", help="Output path for out-of-fold predictions")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}. Run pipeline/package_word_crops_dataset.py first.")

    cache_path = data_dir / "cached_word_embeddings.pt"
    if cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}...")
        cached_data = torch.load(cache_path, weights_only=False)
        proj_embeds = cached_data["embeddings"]
        records = cached_data["records"]
    else:
        logger.info("Extracting ViT embeddings from scratch...")
        proj_embeds, records = extract_vit_embeddings_for_words(
            dataset_dir=data_dir,
            device=args.device,
            project_dim=args.project_dim,
        )
        torch.save({"embeddings": proj_embeds, "records": records}, cache_path)
        logger.info(f"Saved cached embeddings to {cache_path}")

    # Group into student bags
    student_bags_by_id: Dict[str, torch.Tensor] = {}
    student_meta_by_id: Dict[str, Dict[str, Any]] = {}
    student_word_map: Dict[str, List[torch.Tensor]] = {}

    for r in records:
        sid = r["student_id"]
        wid = r["word_id"]
        if wid in proj_embeds:
            student_word_map.setdefault(sid, []).append(proj_embeds[wid])
            if sid not in student_meta_by_id:
                student_meta_by_id[sid] = {
                    "school": r["school"],
                    "grade": int(r["grade"]),
                    "label": int(r["label"]),
                    "repeat_0_fold": int(r["repeat_0_fold"]),
                    "repeat_1_fold": int(r["repeat_1_fold"]),
                    "repeat_2_fold": int(r["repeat_2_fold"]),
                    "repeat_3_fold": int(r["repeat_3_fold"]),
                    "repeat_4_fold": int(r["repeat_4_fold"]),
                }

    for sid, tensors in student_word_map.items():
        student_bags_by_id[sid] = torch.stack(tensors)

    # Run 5x5 Cross-Validation
    results = train_and_evaluate_5x5(
        student_bags_by_id=student_bags_by_id,
        student_meta_by_id=student_meta_by_id,
        projected_dim=args.project_dim,
        attention_dim=args.attention_dim,
        epochs=args.epochs,
        device=args.device,
    )

    # Export out-of-fold predictions
    out_rows = []
    for sid, p_mean in results["mean_oof_predictions"].items():
        meta = student_meta_by_id[sid]
        reps = results["all_repeat_predictions"][sid]
        out_rows.append({
            "student_id": sid,
            "school": meta["school"],
            "grade": meta["grade"],
            "label": meta["label"],
            "pred_prob_mil": p_mean,
            "repeat_0_prob": reps[0] if len(reps) > 0 else p_mean,
            "repeat_1_prob": reps[1] if len(reps) > 1 else p_mean,
            "repeat_2_prob": reps[2] if len(reps) > 2 else p_mean,
            "repeat_3_prob": reps[3] if len(reps) > 3 else p_mean,
            "repeat_4_prob": reps[4] if len(reps) > 4 else p_mean,
        })

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    logger.info(f"Exported out-of-fold predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

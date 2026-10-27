"""Stage 33: Experiment 4 — Joint Multi-Modal Gated-Attention MIL (Vision + Line Math/OCR + Auxiliary Loss).

Scientific Objectives:
----------------------
1. Partial Backbone Freezing (Stop Epoch-2 Memorization):
   - Freeze conv0 through denseblock2 (~3M params) to preserve ImageNet edge/loop primitives.
   - Unfreeze denseblock3, transition3, and denseblock4 (~15M params) for handwriting motor ataxia specialization.
2. Horizontal Sequence Pooling (Option 3 - Spatial Left-to-Right Preservation):
   - Replace 2D Global Average Pooling with vertical pool -> 1D sequence conv over 20 horizontal bins.
   - Preserves letter sequence, horizontal spacing rhythm, and within-sentence motor fatigue gradients.
3. Line-Level Multi-Modal Fusion (Option 1):
   - Combine compressed visual vector v_k (R^32) with line-level math & TrOCR vector m_k (R^16).
   - Trainable Gated Attention mechanism observes both visual appearance and physical math measurements.
4. Multi-Task Auxiliary Legibility Loss (Option 2):
   - DenseNet201 visual embedding v_k must also predict line-level TrOCR legibility targets across all 1,145 lines:
       L_total = L_Student_BCE + lambda_aux * MSE(TrOCR_targets, AuxHead(v_k))
   - Provides 1,145 continuous supervision targets, preventing binary student-bag memorization.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import DenseNet201_Weights
import torchvision.transforms as transforms

from baseline.multimodal_features import (
    fit_and_scale_multimodal_features,
    load_raw_multimodal_dataframe,
)
from baseline.train_whole_page_mil import (
    FlushingStreamHandler,
    precompute_or_load_wide_3ch_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[FlushingStreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"
REPORTS_DIR = REPO_ROOT / "reports"


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class MultiModalStudentBagDataset(Dataset):
    """Dataset delivering student bags with wide visual images, line-level math features, and aux targets."""

    def __init__(
        self,
        student_records: List[Dict[str, Any]],
        tensors_dict: Dict[str, np.ndarray],
        math_dict: Dict[str, np.ndarray],
        aux_dict: Dict[str, np.ndarray],
        is_train: bool = True,
    ) -> None:
        self.student_records = student_records
        self.tensors_dict = tensors_dict
        self.math_dict = math_dict
        self.aux_dict = aux_dict
        self.is_train = is_train

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.is_train:
            self.aug_pil = transforms.Compose([
                transforms.RandomRotation(degrees=(-4, 4), fill=0),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.92, 1.08), fill=0),
            ])
            self.random_erasing = transforms.RandomErasing(
                p=0.30,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0.0,
            )
        else:
            self.aug_pil = None
            self.random_erasing = None

    def __len__(self) -> int:
        return len(self.student_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.student_records[idx]
        task_uids = rec["task_uids"]

        bag_images: List[torch.Tensor] = []
        bag_math: List[torch.Tensor] = []
        bag_aux: List[torch.Tensor] = []

        for uid in task_uids:
            # 1. Image
            uint8_arr = self.tensors_dict[uid]
            pil_img = Image.fromarray(uint8_arr)

            if self.is_train and self.aug_pil is not None:
                pil_img = self.aug_pil(pil_img)

            t_img = transforms.functional.to_tensor(pil_img)
            t_img = self.norm(t_img)

            if self.is_train and self.random_erasing is not None:
                t_img = self.random_erasing(t_img)

            bag_images.append(t_img)

            # 2. Math & OCR features
            math_vec = torch.tensor(self.math_dict[uid], dtype=torch.float32)
            bag_math.append(math_vec)

            # 3. Aux targets [mean_conf, low_conf_ratio]
            aux_vec = torch.tensor(self.aux_dict[uid], dtype=torch.float32)
            bag_aux.append(aux_vec)

        return {
            "student_id": rec["student_id"],
            "school": rec["school"],
            "label": torch.tensor(rec["label"], dtype=torch.float32),
            "grade": rec["grade"],
            "task_names": rec["task_names"],
            "num_crops": len(task_uids),
            "images": torch.stack(bag_images, dim=0),  # (K, 3, 128, 640)
            "math": torch.stack(bag_math, dim=0),      # (K, 16)
            "aux": torch.stack(bag_aux, dim=0),        # (K, 2)
        }


def collate_multimodal_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-sized student bags into batched visual, tabular, and auxiliary tensors."""
    student_ids = [b["student_id"] for b in batch]
    schools = [b["school"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    grades = [b["grade"] for b in batch]
    task_names_list = [b["task_names"] for b in batch]
    num_crops_list = [b["num_crops"] for b in batch]

    all_images = torch.cat([b["images"] for b in batch], dim=0)
    all_math = torch.cat([b["math"] for b in batch], dim=0)
    all_aux = torch.cat([b["aux"] for b in batch], dim=0)

    split_indices: List[Tuple[int, int]] = []
    curr = 0
    for k in num_crops_list:
        split_indices.append((curr, curr + k))
        curr += k

    return {
        "student_ids": student_ids,
        "schools": schools,
        "labels": labels,
        "grades": grades,
        "task_names_list": task_names_list,
        "num_crops_list": num_crops_list,
        "all_images": all_images,
        "all_math": all_math,
        "all_aux": all_aux,
        "split_indices": split_indices,
    }


class HorizontalSequencePooler(nn.Module):
    """Vertical pool -> 1D sequence convolution over 20 horizontal spatial steps."""

    def __init__(self, in_channels: int = 1920, seq_channels: int = 64, out_dim: int = 32, dropout: float = 0.4) -> None:
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels, seq_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(seq_channels),
            nn.ReLU(),
            nn.Conv1d(seq_channels, seq_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(seq_channels),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(seq_channels * 2, out_dim),
            nn.ReLU(),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Args: feat_map of shape (N, 1920, 4, 20). Returns: (N, out_dim)."""
        # Vertical pooling across height bins
        seq = feat_map.mean(dim=2)  # (N, 1920, 20)
        h_seq = self.conv1d(seq)    # (N, seq_channels, 20)

        # Dual pooling: both max-over-time (salient spikes) and mean-over-time (global stroke consistency)
        h_max = h_seq.max(dim=2)[0]   # (N, seq_channels)
        h_mean = h_seq.mean(dim=2)    # (N, seq_channels)
        h_pooled = torch.cat([h_max, h_mean], dim=1)  # (N, seq_channels * 2)

        return self.proj(h_pooled)  # (N, out_dim)


class MultiModalWholePageMILModel(nn.Module):
    """Joint Multi-Modal MIL model with partial backbone freeze, sequence pooling, and auxiliary loss."""

    def __init__(
        self,
        math_dim: int = 16,
        visual_dim: int = 32,
        math_emb_dim: int = 32,
        attn_dim: int = 32,
        pretrained: bool = True,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        # 1. Backbone: DenseNet201
        weights = DenseNet201_Weights.DEFAULT if pretrained else None
        densenet = models.densenet201(weights=weights)
        self.features = densenet.features

        # 2. Freeze conv0 through denseblock2 (keep denseblock3 & denseblock4 trainable)
        freeze_until = "denseblock2"
        freeze = True
        for name, child in self.features.named_children():
            for param in child.parameters():
                param.requires_grad = not freeze
            if name == freeze_until:
                freeze = False

        # 3. Horizontal Sequence Pooler
        self.sequence_pooler = HorizontalSequencePooler(
            in_channels=1920,
            seq_channels=64,
            out_dim=visual_dim,
            dropout=dropout,
        )

        # 4. Multi-Task Auxiliary Legibility Head
        self.aux_head = nn.Sequential(
            nn.Linear(visual_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Predicts [trocr_mean_confidence, trocr_low_conf_ratio]
        )

        # 5. Math Feature Projection
        self.math_proj = nn.Sequential(
            nn.Linear(math_dim, math_emb_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        # 6. Combined Line Vector & Gated Attention
        fused_dim = visual_dim + math_emb_dim  # 32 + 32 = 64
        self.attention_V = nn.Linear(fused_dim, attn_dim)
        self.attention_U = nn.Linear(fused_dim, attn_dim)
        self.attention_w = nn.Linear(attn_dim, 1)

        # 7. Student Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, 1),
        )

    def forward(
        self,
        all_images: torch.Tensor,
        all_math: torch.Tensor,
        split_indices: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Forward pass over concatenated crops and math features.

        Returns:
            student_logits: (B,)
            all_attentions: List of attention weight tensors per student
            aux_preds: (sum(K), 2)
        """
        # Visual feature extraction
        feat_map = self.features(all_images)           # (sum(K), 1920, 4, 20)
        v_emb = self.sequence_pooler(feat_map)         # (sum(K), 32)
        aux_preds = self.aux_head(v_emb)               # (sum(K), 2)

        # Math feature projection
        m_emb = self.math_proj(all_math)               # (sum(K), 32)

        # Line-level fusion
        x_line = torch.cat([v_emb, m_emb], dim=-1)     # (sum(K), 64)

        student_logits_list: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        # Gated attention pooling per student
        for start_idx, end_idx in split_indices:
            student_lines = x_line[start_idx:end_idx]  # (K_i, 64)
            v = torch.tanh(self.attention_V(student_lines))
            u = torch.sigmoid(self.attention_U(student_lines))
            gated = v * u
            attn_logits = self.attention_w(gated).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=0)

            z_student = torch.sum(student_lines * attn_weights.unsqueeze(-1), dim=0, keepdim=True)
            logit = self.classifier(z_student).squeeze(-1)

            all_attentions.append(attn_weights)
            student_logits_list.append(logit)

        student_logits = torch.cat(student_logits_list, dim=0)
        return student_logits, all_attentions, aux_preds


def evaluate_multimodal_dataset(
    model: MultiModalWholePageMILModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    aux_criterion: nn.Module,
    lambda_aux: float = 0.3,
) -> Dict[str, Any]:
    """Evaluate Multi-Modal MIL model computing metrics and attention weights."""
    model.eval()
    total_loss = 0.0
    all_probs: List[float] = []
    all_labels: List[float] = []
    all_students: List[str] = []
    all_attentions_dict: Dict[str, List[float]] = {}
    task_names_per_student: Dict[str, List[str]] = {}

    with torch.no_grad():
        for batch in loader:
            all_images = batch["all_images"].to(device, non_blocking=True)
            all_math = batch["all_math"].to(device, non_blocking=True)
            all_aux = batch["all_aux"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            split_indices = batch["split_indices"]

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, attentions, aux_preds = model(all_images, all_math, split_indices)
                loss_main = criterion(logits, labels)
                loss_aux = aux_criterion(aux_preds, all_aux)
                loss = loss_main + (lambda_aux * loss_aux if lambda_aux > 0 else 0.0)

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits.float()).cpu().numpy().tolist()
            labels_np = labels.cpu().numpy().tolist()

            all_probs.extend(probs)
            all_labels.extend(labels_np)
            all_students.extend(batch["student_ids"])

            for sid, tasks, attn in zip(batch["student_ids"], batch["task_names_list"], attentions):
                all_attentions_dict[sid] = attn.cpu().numpy().tolist()
                task_names_per_student[sid] = tasks

    avg_loss = total_loss / max(1, len(all_labels))
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float("nan")
    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = float("nan")

    return {
        "loss": avg_loss,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "y_true": y_true,
        "y_prob": y_prob,
        "student_ids": all_students,
        "attentions": all_attentions_dict,
        "tasks": task_names_per_student,
    }


def train_single_multimodal_run(
    run_name: str,
    lambda_aux: float,
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    tensors_dict: Dict[str, np.ndarray],
    math_dict: Dict[str, np.ndarray],
    aux_dict: Dict[str, np.ndarray],
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train Joint Multi-Modal Gated-Attention MIL model."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labels = [r["label"] for r in train_records]
    test_labels = [r["label"] for r in test_records]
    n_pos = max(1.0, float(sum(train_labels)))
    n_neg = max(1.0, float(len(train_labels) - n_pos))
    pos_weight_val = n_neg / n_pos
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)

    logger.info(f"=== Starting {run_name} on {device} (lambda_aux={lambda_aux}) ===")
    logger.info(
        f"Train students: {len(train_records)} ({sum(train_labels):.0f} pos, {len(train_records) - sum(train_labels):.0f} neg) | "
        f"Test students: {len(test_records)} ({sum(test_labels):.0f} pos, {len(test_records) - sum(test_labels):.0f} neg)"
    )

    train_dataset = MultiModalStudentBagDataset(train_records, tensors_dict, math_dict, aux_dict, is_train=True)
    test_dataset = MultiModalStudentBagDataset(test_records, tensors_dict, math_dict, aux_dict, is_train=False)
    train_eval_dataset = MultiModalStudentBagDataset(train_records, tensors_dict, math_dict, aux_dict, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_bags, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_bags, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_bags, pin_memory=True)

    model = MultiModalWholePageMILModel(pretrained=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    aux_criterion = nn.MSELoss()

    # Different learning rates: lower for denseblock3-4 (5e-5), higher for heads (1e-4)
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = (
        list(model.sequence_pooler.parameters())
        + list(model.aux_head.parameters())
        + list(model.math_proj.parameters())
        + list(model.attention_V.parameters())
        + list(model.attention_U.parameters())
        + list(model.attention_w.parameters())
        + list(model.classifier.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.5},
        {"params": head_params, "lr": lr},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    history: List[Dict[str, Any]] = []
    best_test_auroc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            all_images = batch["all_images"].to(device, non_blocking=True)
            all_math = batch["all_math"].to(device, non_blocking=True)
            all_aux = batch["all_aux"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            split_indices = batch["split_indices"]

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _, aux_preds = model(all_images, all_math, split_indices)
                loss_main = criterion(logits, labels)
                loss_aux = aux_criterion(aux_preds, all_aux)
                loss = loss_main + (lambda_aux * loss_aux if lambda_aux > 0 else 0.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(labels)
            n_samples += len(labels)

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_samples)

        train_res = evaluate_multimodal_dataset(model, train_eval_loader, device, criterion, aux_criterion, lambda_aux)
        test_res = evaluate_multimodal_dataset(model, test_loader, device, criterion, aux_criterion, lambda_aux)

        if test_res["auroc"] > best_test_auroc:
            best_test_auroc = test_res["auroc"]
            best_epoch = epoch

        row_hist = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_auroc": train_res["auroc"],
            "train_auprc": train_res["auprc"],
            "test_loss": test_res["loss"],
            "test_auroc": test_res["auroc"],
            "test_auprc": test_res["auprc"],
            "test_brier": test_res["brier"],
        }
        history.append(row_hist)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Train AUROC: {train_res['auroc']:.4f} | "
            f"Test Loss: {test_res['loss']:.4f} | Test AUROC: {test_res['auroc']:.4f} | "
            f"Test AUPRC: {test_res['auprc']:.4f} (Best: {best_test_auroc:.4f} @ Ep {best_epoch})"
        )

    final_eval = evaluate_multimodal_dataset(model, test_loader, device, criterion, aux_criterion, lambda_aux)
    final_eval["best_test_auroc"] = best_test_auroc
    final_eval["best_epoch"] = best_epoch

    return history, final_eval


def run_experiment_4(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    seed: int = 42,
    which_run: str = "both",
) -> Dict[str, Any]:
    """Execute Experiment 4: Run 4A and Run 4B under honest student_id split."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load data
    df_merged = load_raw_multimodal_dataframe()
    logger.info(f"Loaded {len(df_merged)} multi-modal rows across {df_merged['student_id'].nunique()} students.")

    # 2. Load wide tensors
    tensors_dict = precompute_or_load_wide_3ch_dataset(df_merged, target_h=128, target_w=640)

    # 3. Assemble student records
    student_records: List[Dict[str, Any]] = []
    for sid, group in df_merged.groupby("student_id"):
        sorted_group = group.sort_values("task_id")
        rec = {
            "student_id": sid,
            "school": group["school"].iloc[0],
            "label": float(group["label"].max()),
            "grade": int(group["grade"].iloc[0]) if "grade" in group else 0,
            "task_uids": sorted_group["task_uid"].tolist(),
            "task_names": sorted_group["task_id"].tolist(),
        }
        student_records.append(rec)

    df_students = pd.DataFrame([
        {"student_id": r["student_id"], "label": r["label"], "school": r["school"]}
        for r in student_records
    ])

    # 4. StratifiedGroupKFold on student_id (honest 80/20 split)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df_students, df_students["label"], groups=df_students["student_id"]))

    train_students = set(df_students.iloc[train_idx]["student_id"])
    test_students = set(df_students.iloc[test_idx]["student_id"])
    assert len(train_students.intersection(test_students)) == 0, "Zero student leakage guaranteed"

    train_records = [r for r in student_records if r["student_id"] in train_students]
    test_records = [r for r in student_records if r["student_id"] in test_students]

    train_uids = [uid for r in train_records for uid in r["task_uids"]]
    test_uids = [uid for r in test_records for uid in r["task_uids"]]

    # 5. Scale features strictly on training fold
    math_dict, aux_dict, _ = fit_and_scale_multimodal_features(df_merged, train_uids, test_uids)

    results: Dict[str, Any] = {
        "experiment": "Experiment 4: Joint Multi-Modal Gated-Attention MIL (Vision + Line Math/OCR)",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "seed": seed,
    }

    # RUN 4A: Multi-Modal MIL without auxiliary loss
    if which_run in ["4a", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING RUN 4A: Multi-Modal MIL (Vision + Math/OCR) [lambda_aux = 0.0]")
        logger.info("=" * 70)
        hist_4a, final_4a = train_single_multimodal_run(
            run_name="Run 4A (Multi-Modal MIL Baseline)",
            lambda_aux=0.0,
            train_records=train_records,
            test_records=test_records,
            tensors_dict=tensors_dict,
            math_dict=math_dict,
            aux_dict=aux_dict,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )
        df_hist_4a = pd.DataFrame(hist_4a)
        csv_path_4a = REPORTS_DIR / "multimodal_mil_run4a_history.csv"
        df_hist_4a.to_csv(csv_path_4a, index=False)
        logger.info(f"Saved Run 4A history to {csv_path_4a}")

        results["run4a"] = {
            "history": hist_4a,
            "final_train_auroc": hist_4a[-1]["train_auroc"],
            "final_test_auroc": hist_4a[-1]["test_auroc"],
            "final_test_auprc": hist_4a[-1]["test_auprc"],
            "final_test_brier": hist_4a[-1]["test_brier"],
            "best_test_auroc": final_4a["best_test_auroc"],
            "best_epoch": final_4a["best_epoch"],
        }

    # RUN 4B: Multi-Modal MIL with Multi-Task Auxiliary Legibility Loss
    if which_run in ["4b", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING RUN 4B: Full Multi-Task Multi-Modal MIL [lambda_aux = 0.3]")
        logger.info("=" * 70)
        hist_4b, final_4b = train_single_multimodal_run(
            run_name="Run 4B (Multi-Task Multi-Modal MIL)",
            lambda_aux=0.3,
            train_records=train_records,
            test_records=test_records,
            tensors_dict=tensors_dict,
            math_dict=math_dict,
            aux_dict=aux_dict,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )
        df_hist_4b = pd.DataFrame(hist_4b)
        csv_path_4b = REPORTS_DIR / "multimodal_mil_run4b_history.csv"
        df_hist_4b.to_csv(csv_path_4b, index=False)
        logger.info(f"Saved Run 4B history to {csv_path_4b}")

        # Compute task attention breakdown (Dysgraphic vs Normal)
        task_attns_pos: Dict[str, List[float]] = {}
        task_attns_neg: Dict[str, List[float]] = {}

        for sid, task_list in final_4b["tasks"].items():
            attns = final_4b["attentions"][sid]
            y = int(final_4b["y_true"][final_4b["student_ids"].index(sid)])
            target_dict = task_attns_pos if y == 1 else task_attns_neg

            for t_name, a_val in zip(task_list, attns):
                clean_name = t_name.replace("sentence_0", "s").replace("sentence_", "s")
                if clean_name not in target_dict:
                    target_dict[clean_name] = []
                target_dict[clean_name].append(a_val)

        mean_attns_pos = {k: float(np.mean(v)) for k, v in task_attns_pos.items()}
        mean_attns_neg = {k: float(np.mean(v)) for k, v in task_attns_neg.items()}

        results["run4b"] = {
            "history": hist_4b,
            "final_train_auroc": hist_4b[-1]["train_auroc"],
            "final_test_auroc": hist_4b[-1]["test_auroc"],
            "final_test_auprc": hist_4b[-1]["test_auprc"],
            "final_test_brier": hist_4b[-1]["test_brier"],
            "best_test_auroc": final_4b["best_test_auroc"],
            "best_epoch": final_4b["best_epoch"],
            "task_attentions_dysgraphic": mean_attns_pos,
            "task_attentions_control": mean_attns_neg,
        }

    # Save summary JSON
    json_path = REPORTS_DIR / "multimodal_mil_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved comparison summary to {json_path}")

    # Generate comparative plot
    if "run4a" in results and "run4b" in results:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            epochs_arr = np.array([h["epoch"] for h in hist_4a])

            ax1.plot(epochs_arr, [h["train_auroc"] for h in hist_4a], label="Run 4A Train (No Aux Loss)", color="#1f77b4", lw=2, ls="--")
            ax1.plot(epochs_arr, [h["test_auroc"] for h in hist_4a], label="Run 4A Test (No Aux Loss)", color="#1f77b4", lw=2.5)
            ax1.plot(epochs_arr, [h["train_auroc"] for h in hist_4b], label="Run 4B Train (+ Aux Loss)", color="#d62728", lw=2, ls="--")
            ax1.plot(epochs_arr, [h["test_auroc"] for h in hist_4b], label="Run 4B Test (+ Aux Loss)", color="#d62728", lw=2.5)
            ax1.axhline(0.7857, color="purple", ls=":", alpha=0.7, label="0.7857 Exp 3B Vision MIL")
            ax1.axhline(0.7893, color="gray", ls=":", alpha=0.7, label="0.7893 Exp 1 Raw DenseNet")
            ax1.set_title("Multi-Modal AUROC Trajectories (Zero Student Leakage)", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Epoch", fontsize=10)
            ax1.set_ylabel("Student AUROC", fontsize=10)
            ax1.set_ylim([0.50, 1.02])
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="lower right", fontsize=8.5)

            # Attention Distribution Plot
            pos_attns = results["run4b"]["task_attentions_dysgraphic"]
            neg_attns = results["run4b"]["task_attentions_control"]
            all_task_keys = sorted(list(set(list(pos_attns.keys()) + list(neg_attns.keys()))))

            x_indices = np.arange(len(all_task_keys))
            width = 0.35

            val_pos = [pos_attns.get(k, 0.0) for k in all_task_keys]
            val_neg = [neg_attns.get(k, 0.0) for k in all_task_keys]

            ax2.bar(x_indices - width / 2, val_pos, width, label="Dysgraphic Students", color="#d62728", alpha=0.85)
            ax2.bar(x_indices + width / 2, val_neg, width, label="Control Students", color="#2ca02c", alpha=0.85)
            ax2.axhline(1.0 / max(1, len(all_task_keys)), color="black", ls="--", alpha=0.6, label="Uniform Weight")
            ax2.set_xticks(x_indices)
            ax2.set_xticklabels([k.replace("1_copy_hindi", "Copy HI").replace("2_copy_english", "Copy EN").replace("3_dictated_hindi", "Dict HI").replace("4_dictated_english", "Dict EN").replace("5_own_hindi", "Own HI").replace("6_own_english", "Own EN") for k in all_task_keys], rotation=25, ha="right", fontsize=9)
            ax2.set_title("Multi-Modal Gated-Attention Task Weights", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Mean Attention Weight", fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right", fontsize=8.5)

            plt.suptitle("Experiment 4: Joint Multi-Modal Gated-Attention MIL (Vision + Math/OCR)", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            plot_path = REPORTS_DIR / "multimodal_mil_curves.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            logger.info(f"Generated comparative plot at {plot_path}")
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: Joint Multi-Modal Gated-Attention MIL")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (students per batch)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run", type=str, default="both", choices=["4a", "4b", "both"], help="Which run to execute")
    args = parser.parse_args()

    run_experiment_4(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        which_run=args.run,
    )


if __name__ == "__main__":
    main()

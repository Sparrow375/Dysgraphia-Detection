"""Stage 32: Experiment 3 — Whole-Page End-to-End MIL (DenseNet201 + Wide (128, 640) Crops + Gated-Attention).

Scientific Rationale:
---------------------
When taking the median/average probability across a student's sentences, performance plateaus
because a dysgraphic child might produce neat lines during slow, deliberate copying, but collapse
into severe motor dysfluencies under auditory pacing (dictation) or high cognitive load (free writing).
Averaging across the page dilutes the clinical breakdown signal.

In Experiment 3, we implement Whole-Page End-to-End Multiple Instance Learning (MIL):
1. Backbone: Fully unfrozen DenseNet201 (retaining sliding 3x3 filters that provenly achieved 0.7893 in Exp 1).
2. Input Representation: Wide (128, 640, 3) rectangular de-noised tensors preserving natural aspect ratios:
   - Channel 1 (Binary Shape): Pure clean ink strokes from {task_id}_clean_ink.png (background = 0.0).
   - Channel 2 (Pen Pressure): Normalized optical density masked strictly to ink strokes (background = 0.0).
   - Channel 3 (Ruled Line Baseline): Idealized 2-pixel trajectory of the ruled line (background = 0.0).
3. Whole-Page Gated Attention Pooling:
   Instead of heuristic averaging, a trainable Gated Attention mechanism (Ilse et al., 2018) learns
   to automatically upweight the worst, most dysfluent lines on the page:
     Student Score = MLP( sum_k (alpha_k * Backbone(Crop_k)) )

Two Comparative Runs:
----------------------
Run 3A: Whole-Page Uniform Mean Pooling (Averaging lines across page).
Run 3B: Whole-Page Gated-Attention MIL (End-to-End attention picking the worst lines).
Both evaluated under the exact same honest student_id split (0.0% student leakage).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
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


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit for real-time unbuffered progress logging."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


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
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = REPO_ROOT / "reports"
CACHE_NPZ_PATH = FEATURES_DIR / "denoised_wide_3ch_tensors_128x640.npz"


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_wide_3ch_tensor(
    clean_mask_path: Path,
    raw_gray_path: Path,
    json_path: Path,
    target_h: int = 128,
    target_w: int = 640,
) -> np.ndarray:
    """Construct custom wide 3-channel tensor (Binary Shape, Pressure, Ruled Line).

    Returns:
        np.ndarray of shape (target_h, target_w, 3) with dtype uint8 (values 0..255).
    """
    # 1. Load clean ink mask
    mask_gray = cv2.imread(str(clean_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"Missing clean ink mask: {clean_mask_path}")
    ink_bin = (mask_gray < 128).astype(np.float32)

    # 2. Load raw grayscale crop
    raw_gray = cv2.imread(str(raw_gray_path), cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        raise FileNotFoundError(f"Missing raw gray crop: {raw_gray_path}")

    if ink_bin.shape != raw_gray.shape:
        ink_bin = cv2.resize(
            ink_bin, (raw_gray.shape[1], raw_gray.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # Standardize polarity (paper bright, ink dark)
    if np.mean(raw_gray) < 128:
        raw_gray = 255 - raw_gray

    h, w = raw_gray.shape

    # Channel 1: Binary Shape (0 or 1.0)
    ch1 = ink_bin

    # Channel 2: Pencil/Pen Pressure (Optical Density masked to ink)
    non_ink = raw_gray[ink_bin == 0]
    bg_level = float(np.percentile(non_ink, 90)) if len(non_ink) > 50 else 255.0
    bg_level = max(128.0, bg_level)
    optical_density = np.clip((bg_level - raw_gray.astype(np.float32)) / bg_level, 0.0, 1.0)
    ch2 = optical_density * ink_bin

    # Channel 3: Ruled Line Baseline Reference
    rule_map = np.zeros((h, w), dtype=np.float32)
    has_rule = False

    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
            bbox = meta.get("crop_bbox", [0, 0, w, h])
            bx, by = bbox[0], bbox[1]
            lines = meta.get("physical_lines", [])
            for pl in lines:
                slope = pl.get("baseline_slope")
                intercept = pl.get("baseline_intercept")
                if slope is not None and intercept is not None:
                    xs = np.arange(w)
                    ys = np.round(slope * (xs + bx) + intercept - by).astype(int)
                    valid = (ys >= 0) & (ys < h)
                    for x_val, y_val in zip(xs[valid], ys[valid]):
                        y_lo = max(0, y_val - 1)
                        y_hi = min(h, y_val + 2)
                        rule_map[y_lo:y_hi, x_val] = 1.0
                    has_rule = True
        except Exception:
            pass

    # Fallback to horizontal morphological filtering if JSON trajectory is missing
    if not has_rule or np.count_nonzero(rule_map) < 50:
        no_ink = raw_gray.copy()
        no_ink[ink_bin > 0] = 255
        inverted = 255 - no_ink
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1))
        opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        _, bin_rule = cv2.threshold(opened, 25, 1.0, cv2.THRESH_BINARY)
        rule_map = bin_rule.astype(np.float32)

    ch3 = rule_map

    # Stack into (H, W, 3)
    t3 = np.stack([ch1, ch2, ch3], axis=-1)

    # Proportional aspect-preserving resize into (target_h, target_w)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(t3, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pitch black canvas of zeros
    canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized

    # Convert to uint8 (0..255) for compact RAM caching
    return (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)


def precompute_or_load_wide_3ch_dataset(
    df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
    cache_path: Path = CACHE_NPZ_PATH,
    target_h: int = 128,
    target_w: int = 640,
) -> Dict[str, np.ndarray]:
    """Precompute all 1,145 wide 3-channel tensors and cache in RAM / disk."""
    if cache_path.exists():
        logger.info(f"Loading cached wide 3-channel tensors from {cache_path}...")
        npz = np.load(cache_path)
        tensor_dict = {key: npz[key] for key in npz.files}
        if len(tensor_dict) == len(df):
            logger.info(f"Loaded all {len(tensor_dict)} wide tensors from cache successfully.")
            return tensor_dict

    logger.info(f"Precomputing wide (128, 640, 3) de-noised tensors for {len(df)} crops...")
    tensor_dict: Dict[str, np.ndarray] = {}

    for idx, row in df.iterrows():
        school = str(row["school"])
        sid = str(row["student_id"])
        task_id = str(row["task_id"])
        task_uid = str(row["task_uid"])

        p_mask = processed_dir / school / sid / f"{task_id}_clean_ink.png"
        p_gray = processed_dir / school / sid / f"{task_id}.png"
        p_json = processed_dir / school / sid / f"{task_id}.json"

        tensor_uint8 = build_wide_3ch_tensor(
            clean_mask_path=p_mask,
            raw_gray_path=p_gray,
            json_path=p_json,
            target_h=target_h,
            target_w=target_w,
        )
        tensor_dict[task_uid] = tensor_uint8

        if (idx + 1) % 200 == 0 or (idx + 1) == len(df):
            logger.info(f"Precomputed {idx + 1}/{len(df)} wide tensors...")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **tensor_dict)
    logger.info(f"Saved {len(tensor_dict)} precomputed wide tensors to {cache_path}.")
    return tensor_dict


class StudentBagDataset(Dataset):
    """Dataset where each sample is an entire student bag containing K sentence crops."""

    def __init__(
        self,
        student_records: List[Dict[str, Any]],
        tensors_dict: Dict[str, np.ndarray],
        is_train: bool = True,
    ) -> None:
        self.student_records = student_records
        self.tensors_dict = tensors_dict
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
                p=0.35,
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
        task_names = rec["task_names"]

        bag_tensors: List[torch.Tensor] = []
        for uid in task_uids:
            uint8_arr = self.tensors_dict[uid]  # Shape (128, 640, 3)
            pil_img = Image.fromarray(uint8_arr)

            if self.is_train and self.aug_pil is not None:
                pil_img = self.aug_pil(pil_img)

            t_img = transforms.functional.to_tensor(pil_img)
            t_img = self.norm(t_img)

            if self.is_train and self.random_erasing is not None:
                t_img = self.random_erasing(t_img)

            bag_tensors.append(t_img)

        # Stack into (K, 3, 128, 640)
        bag_tensor = torch.stack(bag_tensors, dim=0)

        return {
            "student_id": rec["student_id"],
            "school": rec["school"],
            "label": torch.tensor(rec["label"], dtype=torch.float32),
            "grade": rec["grade"],
            "images": bag_tensor,
            "task_names": task_names,
            "num_crops": len(task_uids),
        }


def collate_student_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-sized student bags into a single batched structure for GPU processing."""
    student_ids = [b["student_id"] for b in batch]
    schools = [b["school"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    grades = [b["grade"] for b in batch]
    task_names_list = [b["task_names"] for b in batch]
    num_crops_list = [b["num_crops"] for b in batch]

    # Concatenate all crops across bags: shape (sum(K_i), 3, 128, 640)
    all_images = torch.cat([b["images"] for b in batch], dim=0)

    # Slice indices mapping each student to their crops
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
        "split_indices": split_indices,
    }


class WholePageMILModel(nn.Module):
    """Whole-Page End-to-End MIL model with DenseNet201 backbone and Gated-Attention pooling."""

    def __init__(
        self,
        pooling_mode: str = "gated_attention",
        pretrained: bool = True,
        emb_dim: int = 256,
        attn_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.pooling_mode = pooling_mode.lower()

        # Backbone: DenseNet201 features
        weights = DenseNet201_Weights.DEFAULT if pretrained else None
        densenet = models.densenet201(weights=weights)
        self.features = densenet.features
        in_features = 1920

        # Unfreeze all backbone weights for end-to-end motor learning
        for p in self.features.parameters():
            p.requires_grad = True

        # Projection layer
        self.proj = nn.Sequential(
            nn.Linear(in_features, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Gated-Attention mechanism (Ilse et al., 2018)
        if self.pooling_mode == "gated_attention":
            self.attention_V = nn.Linear(emb_dim, attn_dim)
            self.attention_U = nn.Linear(emb_dim, attn_dim)
            self.attention_w = nn.Linear(attn_dim, 1)

        # Student-level classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(
        self,
        all_images: torch.Tensor,
        split_indices: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass over concatenated batch of crops, pooling per student bag.

        Args:
            all_images: Tensor of shape (sum(K_i), 3, 128, 640)
            split_indices: List of (start_idx, end_idx) per student

        Returns:
            student_logits: Tensor of shape (B,)
            all_attentions: List of attention weight tensors per student
        """
        # 1. Feature extraction through DenseNet201 stem & dense blocks
        feat = self.features(all_images)  # Shape (sum(K), 1920, 4, 20)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # Shape (sum(K), 1920)
        emb = self.proj(pooled)  # Shape (sum(K), emb_dim)

        student_logits_list: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        # 2. Per-student aggregation (Attention MIL vs Uniform Mean)
        for start_idx, end_idx in split_indices:
            student_crops_emb = emb[start_idx:end_idx]  # Shape (K_i, emb_dim)
            k_i = student_crops_emb.shape[0]

            if self.pooling_mode == "gated_attention":
                v = torch.tanh(self.attention_V(student_crops_emb))  # (K_i, attn_dim)
                u = torch.sigmoid(self.attention_U(student_crops_emb))  # (K_i, attn_dim)
                gated = v * u
                attn_logits = self.attention_w(gated).squeeze(-1)  # (K_i,)
                attn_weights = F.softmax(attn_logits, dim=0)  # (K_i,)
                # Weighted sum across lines
                z = torch.sum(student_crops_emb * attn_weights.unsqueeze(-1), dim=0, keepdim=True)
                all_attentions.append(attn_weights)
            else:
                # Uniform mean pooling across lines
                z = torch.mean(student_crops_emb, dim=0, keepdim=True)
                uniform_weights = torch.full((k_i,), 1.0 / k_i, device=emb.device)
                all_attentions.append(uniform_weights)

            logit = self.classifier(z).squeeze(-1)  # Shape (1,)
            student_logits_list.append(logit)

        student_logits = torch.cat(student_logits_list, dim=0)  # Shape (B,)
        return student_logits, all_attentions


def evaluate_mil_dataset(
    model: WholePageMILModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, Any]:
    """Evaluate Whole-Page MIL model computing student-level metrics and attention breakdowns."""
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
            labels = batch["labels"].to(device, non_blocking=True)
            split_indices = batch["split_indices"]

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, attentions = model(all_images, split_indices)
                loss = criterion(logits, labels)

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


def train_single_mil_run(
    run_name: str,
    pooling_mode: str,
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    tensors_dict: Dict[str, np.ndarray],
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train Whole-Page MIL model across epochs."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labels = [r["label"] for r in train_records]
    test_labels = [r["label"] for r in test_records]
    n_pos = max(1.0, float(sum(train_labels)))
    n_neg = max(1.0, float(len(train_labels) - n_pos))
    pos_weight_val = n_neg / n_pos
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)

    logger.info(f"=== Starting {run_name} ({pooling_mode.upper()}) on {device} ===")
    logger.info(
        f"Train students: {len(train_records)} ({sum(train_labels):.0f} pos, {len(train_labels) - sum(train_labels):.0f} neg) | "
        f"Test students: {len(test_records)} ({sum(test_labels):.0f} pos, {len(test_records) - sum(test_labels):.0f} neg)"
    )
    logger.info(f"Class imbalance pos_weight = {pos_weight_val:.2f}")

    train_dataset = StudentBagDataset(train_records, tensors_dict, is_train=True)
    test_dataset = StudentBagDataset(test_records, tensors_dict, is_train=False)
    train_eval_dataset = StudentBagDataset(train_records, tensors_dict, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_student_bags,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_student_bags,
        pin_memory=(device.type == "cuda"),
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_student_bags,
        pin_memory=(device.type == "cuda"),
    )

    model = WholePageMILModel(pooling_mode=pooling_mode, pretrained=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            labels = batch["labels"].to(device, non_blocking=True)
            split_indices = batch["split_indices"]

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _ = model(all_images, split_indices)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(labels)
            n_samples += len(labels)

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_samples)

        # Evaluation
        train_res = evaluate_mil_dataset(model, train_eval_loader, device, criterion)
        test_res = evaluate_mil_dataset(model, test_loader, device, criterion)

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

    final_eval = evaluate_mil_dataset(model, test_loader, device, criterion)
    final_eval["best_test_auroc"] = best_test_auroc
    final_eval["best_epoch"] = best_epoch

    return history, final_eval


def run_experiment_3(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    seed: int = 42,
    which_run: str = "both",
) -> Dict[str, Any]:
    """Execute Experiment 3: Run 3A (Mean Pooling) and Run 3B (Gated-Attention MIL)."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load dataset index
    df = pd.read_csv(INDEX_CSV)
    logger.info(f"Loaded {len(df)} sentence crops across {df['student_id'].nunique()} students.")

    # 2. Precompute / load wide 3-channel tensors (128, 640, 3)
    tensors_dict = precompute_or_load_wide_3ch_dataset(df, target_h=128, target_w=640)

    # 3. Assemble student bags
    student_records: List[Dict[str, Any]] = []
    for sid, group in df.groupby("student_id"):
        # Sort crops by task protocol order
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

    # 4. StratifiedGroupKFold on student_id (honest 80/20 split, zero leakage)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df_students, df_students["label"], groups=df_students["student_id"]))

    train_students = set(df_students.iloc[train_idx]["student_id"])
    test_students = set(df_students.iloc[test_idx]["student_id"])
    overlap = train_students.intersection(test_students)
    assert len(overlap) == 0, "Zero student leakage guaranteed"

    train_records = [r for r in student_records if r["student_id"] in train_students]
    test_records = [r for r in student_records if r["student_id"] in test_students]

    results: Dict[str, Any] = {
        "experiment": "Experiment 3: Whole-Page End-to-End MIL (DenseNet201 + Wide (128, 640) Crops)",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "seed": seed,
    }

    # RUN 3A: Whole-Page Mean Pooling
    if which_run in ["3a", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING RUN 3A: Whole-Page Uniform Mean Pooling (Averaging Lines)")
        logger.info("=" * 70)
        hist_3a, final_3a = train_single_mil_run(
            run_name="Run 3A (Whole-Page Mean Pooling)",
            pooling_mode="mean_pooling",
            train_records=train_records,
            test_records=test_records,
            tensors_dict=tensors_dict,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )
        df_hist_3a = pd.DataFrame(hist_3a)
        csv_path_3a = REPORTS_DIR / "whole_page_mil_run3a_history.csv"
        df_hist_3a.to_csv(csv_path_3a, index=False)
        logger.info(f"Saved Run 3A history to {csv_path_3a}")

        results["run3a"] = {
            "history": hist_3a,
            "final_train_auroc": hist_3a[-1]["train_auroc"],
            "final_test_auroc": hist_3a[-1]["test_auroc"],
            "final_test_auprc": hist_3a[-1]["test_auprc"],
            "final_test_brier": hist_3a[-1]["test_brier"],
            "best_test_auroc": final_3a["best_test_auroc"],
            "best_epoch": final_3a["best_epoch"],
        }

    # RUN 3B: Whole-Page Gated-Attention MIL
    if which_run in ["3b", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING RUN 3B: Whole-Page Gated-Attention MIL (Attention Picking Worst Lines)")
        logger.info("=" * 70)
        hist_3b, final_3b = train_single_mil_run(
            run_name="Run 3B (Whole-Page Gated-Attention MIL)",
            pooling_mode="gated_attention",
            train_records=train_records,
            test_records=test_records,
            tensors_dict=tensors_dict,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )
        df_hist_3b = pd.DataFrame(hist_3b)
        csv_path_3b = REPORTS_DIR / "whole_page_mil_run3b_history.csv"
        df_hist_3b.to_csv(csv_path_3b, index=False)
        logger.info(f"Saved Run 3B history to {csv_path_3b}")

        # Compute task attention breakdown (Dysgraphic vs Normal)
        task_attns_pos: Dict[str, List[float]] = {}
        task_attns_neg: Dict[str, List[float]] = {}

        for sid, task_list in final_3b["tasks"].items():
            attns = final_3b["attentions"][sid]
            y = int(final_3b["y_true"][final_3b["student_ids"].index(sid)])
            target_dict = task_attns_pos if y == 1 else task_attns_neg

            for t_name, a_val in zip(task_list, attns):
                clean_name = t_name.replace("sentence_0", "s").replace("sentence_", "s")
                if clean_name not in target_dict:
                    target_dict[clean_name] = []
                target_dict[clean_name].append(a_val)

        mean_attns_pos = {k: float(np.mean(v)) for k, v in task_attns_pos.items()}
        mean_attns_neg = {k: float(np.mean(v)) for k, v in task_attns_neg.items()}

        results["run3b"] = {
            "history": hist_3b,
            "final_train_auroc": hist_3b[-1]["train_auroc"],
            "final_test_auroc": hist_3b[-1]["test_auroc"],
            "final_test_auprc": hist_3b[-1]["test_auprc"],
            "final_test_brier": hist_3b[-1]["test_brier"],
            "best_test_auroc": final_3b["best_test_auroc"],
            "best_epoch": final_3b["best_epoch"],
            "task_attentions_dysgraphic": mean_attns_pos,
            "task_attentions_control": mean_attns_neg,
        }

    # Save summary JSON
    json_path = REPORTS_DIR / "whole_page_mil_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved comparison summary to {json_path}")

    # Generate comparative plot if both runs were executed
    if "run3a" in results and "run3b" in results:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            epochs_arr = np.array([h["epoch"] for h in hist_3a])

            # Trajectory Plot
            ax1.plot(epochs_arr, [h["train_auroc"] for h in hist_3a], label="Run 3A Train (Mean)", color="#1f77b4", lw=2, ls="--")
            ax1.plot(epochs_arr, [h["test_auroc"] for h in hist_3a], label="Run 3A Test (Mean)", color="#1f77b4", lw=2.5)
            ax1.plot(epochs_arr, [h["train_auroc"] for h in hist_3b], label="Run 3B Train (Gated-Attn)", color="#d62728", lw=2, ls="--")
            ax1.plot(epochs_arr, [h["test_auroc"] for h in hist_3b], label="Run 3B Test (Gated-Attn)", color="#d62728", lw=2.5)
            ax1.axhline(0.7893, color="purple", ls=":", alpha=0.7, label="0.7893 Exp 1 Raw DenseNet Baseline")
            ax1.axhline(0.6929, color="gray", ls=":", alpha=0.7, label="0.6929 Exp 2 De-Noised Swin-T")
            ax1.set_title("Whole-Page AUROC Trajectories (Zero Student Leakage)", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Epoch", fontsize=10)
            ax1.set_ylabel("Student AUROC", fontsize=10)
            ax1.set_ylim([0.45, 1.02])
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="lower right", fontsize=8.5)

            # Attention Distribution Plot
            pos_attns = results["run3b"]["task_attentions_dysgraphic"]
            neg_attns = results["run3b"]["task_attentions_control"]
            all_task_keys = sorted(list(set(list(pos_attns.keys()) + list(neg_attns.keys()))))

            x_indices = np.arange(len(all_task_keys))
            width = 0.35

            val_pos = [pos_attns.get(k, 0.0) for k in all_task_keys]
            val_neg = [neg_attns.get(k, 0.0) for k in all_task_keys]

            ax2.bar(x_indices - width / 2, val_pos, width, label="Dysgraphic Students", color="#d62728", alpha=0.85)
            ax2.bar(x_indices + width / 2, val_neg, width, label="Control Students", color="#2ca02c", alpha=0.85)
            ax2.axhline(1.0 / max(1, len(all_task_keys)), color="black", ls="--", alpha=0.6, label="Uniform Weight Baseline")
            ax2.set_xticks(x_indices)
            ax2.set_xticklabels([k.replace("1_copy_hindi", "Copy HI").replace("2_copy_english", "Copy EN").replace("3_dictated_hindi", "Dict HI").replace("4_dictated_english", "Dict EN").replace("5_own_hindi", "Own HI").replace("6_own_english", "Own EN") for k in all_task_keys], rotation=25, ha="right", fontsize=9)
            ax2.set_title("Gated-Attention Task Weight Distribution", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Mean Attention Weight", fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right", fontsize=8.5)

            plt.suptitle("Experiment 3: Whole-Page End-to-End MIL (DenseNet201 + Wide (128, 640) Crops)", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            plot_path = REPORTS_DIR / "whole_page_mil_curves.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            logger.info(f"Generated comparative plot at {plot_path}")
        except Exception as e:
            logger.warning(f"Could not generate matplotlib plot: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: Whole-Page End-to-End MIL")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (students per batch)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run", type=str, default="both", choices=["3a", "3b", "both"], help="Which run to execute")
    args = parser.parse_args()

    run_experiment_3(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        which_run=args.run,
    )


if __name__ == "__main__":
    main()

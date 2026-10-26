"""Stage 30: "Baseline Paper" Full Unfreeze Experiment (DenseNet201 / ConvNeXt-Base).

Scientific Objective:
---------------------
To settle the fundamental hypothesis:
"Is the reported 95%+ AUROC in published baseline handwriting papers an artifact
of crop-level data leakage (where crops from the same child's notebook appear in
both train and test sets, allowing the network to memorize individual handwriting style
and paper lighting), or does an unfrozen deep CNN learn invariant visual stroke patterns
that generalize honestly to unseen students?"

Two Benchmark Runs:
-------------------
Run 1A (The "Baseline Paper" Crop Split):
  - Standard random 80/20 train/test split pooling all crops together, strictly ignoring `student_id`.
  - Same child's sentences appear in both train and test.
  - Prediction: Train & Validation AUROC rapidly shoot past 0.95+ within 15 epochs.

Run 1B (The Honest student_id Split):
  - StratifiedGroupKFold split strictly by `student_id` (zero student overlap between train and test).
  - Evaluates both raw crop-level AUROC and student-pooled predicted probabilities (mean pooling).
  - Diagnostic Test: Does Test AUROC jump to 0.78-0.85 (showing visual signal our math formulas missed),
    or does it crash towards chance/0.55 (proving CNNs memorize paper background/illumination)?

Features & Optimizations:
-------------------------
- Architecture: Fully unfrozen DenseNet201 (~20M parameters) or ConvNeXt-Base (~88M parameters).
- Aspect-preserving proportional resizing with white canvas padding (224x224).
- Aggressive data augmentation: random ±5° tilt, 0.85-1.15x zoom, random erasing (paper cutout simulation).
- PyTorch mixed precision (AMP) for high throughput on NVIDIA RTX GPUs.
- Epoch-by-epoch loss, Train AUROC, Test Crop AUROC, and Test Student AUROC trajectory tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import (
    DenseNet201_Weights,
    ConvNeXt_Base_Weights,
    DenseNet121_Weights,
)
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
REPORTS_DIR = REPO_ROOT / "reports"


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def pad_to_square(img: Image.Image, target_size: int = 224) -> Image.Image:
    """Aspect-preserving proportional resizing onto a neutral white square canvas."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (target_size, target_size), (255, 255, 255))
    scale = target_size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


# In-memory image cache mapping rel_path -> pre-padded PIL Image
GLOBAL_IMAGE_CACHE: Dict[str, Image.Image] = {}


def preload_images(
    df: pd.DataFrame,
    dataset_dir: Path,
    target_size: int = 224,
) -> Dict[str, Image.Image]:
    """Preload and pad all crops in memory once to eliminate disk IO bottleneck."""
    cached: Dict[str, Image.Image] = {}
    rel_paths = df["rel_path"].unique()
    logger.info(f"Preloading and padding {len(rel_paths)} crops into RAM at {target_size}x{target_size}...")
    for rel_path in rel_paths:
        if rel_path in GLOBAL_IMAGE_CACHE:
            cached[rel_path] = GLOBAL_IMAGE_CACHE[rel_path]
            continue
        img_path = dataset_dir / rel_path
        if not img_path.exists():
            continue
        try:
            pil_img = Image.open(img_path).convert("RGB")
            padded = pad_to_square(pil_img, target_size=target_size)
            GLOBAL_IMAGE_CACHE[rel_path] = padded
            cached[rel_path] = padded
        except Exception as e:
            logger.warning(f"Error reading {img_path}: {e}")
    logger.info(f"Preloading complete: {len(GLOBAL_IMAGE_CACHE)} crops cached in RAM.")
    return GLOBAL_IMAGE_CACHE


class SentenceCropDataset(Dataset):
    """PyTorch Dataset for sentence crops with aspect padding and aggressive augmentation."""

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Path,
        image_cache: Optional[Dict[str, Image.Image]] = None,
        target_size: int = 224,
        is_train: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.image_cache = image_cache if image_cache is not None else GLOBAL_IMAGE_CACHE
        self.target_size = target_size
        self.is_train = is_train

        # Base normalization for ImageNet backbones
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Augmentation pipeline: random ±5° tilt, 0.85-1.15x zoom, random erasing
        if self.is_train:
            self.aug_pil = transforms.Compose([
                transforms.RandomRotation(degrees=(-5, 5), fill=255),
                transforms.RandomAffine(degrees=0, scale=(0.85, 1.15), fill=255),
            ])
            self.aug_tensor = transforms.Compose([
                transforms.ToTensor(),
                self.norm,
                transforms.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.20),
                    ratio=(0.3, 3.3),
                    value=1.0,  # White erase simulating paper cutouts
                ),
            ])
        else:
            self.aug_pil = None
            self.aug_tensor = transforms.Compose([
                transforms.ToTensor(),
                self.norm,
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = row["rel_path"]

        if rel_path in self.image_cache:
            canvas = self.image_cache[rel_path]
        else:
            img_path = self.root_dir / rel_path
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to read image {img_path}: {e}")
                pil_img = Image.new("RGB", (self.target_size, self.target_size), (255, 255, 255))
            canvas = pad_to_square(pil_img, target_size=self.target_size)
            self.image_cache[rel_path] = canvas

        if self.is_train and self.aug_pil is not None:
            canvas = self.aug_pil(canvas)

        tensor_img = self.aug_tensor(canvas)
        label = float(row["label"])

        return {
            "image": tensor_img,
            "label": torch.tensor(label, dtype=torch.float32),
            "student_id": str(row["student_id"]),
            "task_uid": str(row["task_uid"]),
            "grade": int(row.get("grade", 0)),
        }


def build_unfrozen_model(
    model_name: str = "densenet201",
    pretrained: bool = True,
    dropout: float = 0.2,
) -> Tuple[nn.Module, int]:
    """Build a fully unfrozen CNN backbone with a single linear classification head.
    
    Returns:
        (model, num_total_parameters)
    """
    model_name = model_name.lower()
    if model_name == "densenet201":
        weights = DenseNet201_Weights.DEFAULT if pretrained else None
        model = models.densenet201(weights=weights)
        in_features = model.classifier.in_features  # 1920
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
    elif model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features  # 1024
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
    elif model_name == "convnext_base":
        weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = models.convnext_base(weights=weights)
        in_features = model.classifier[2].in_features  # 1024
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Ensure ALL parameters are unfrozen (requires_grad = True)
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        param.requires_grad = True
        total_params += param.numel()
        trainable_params += param.numel()

    logger.info(
        f"Built fully unfrozen {model_name}: {total_params:,} parameters "
        f"({trainable_params / 1e6:.2f}M trainable)."
    )
    return model, total_params


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, Any]:
    """Evaluate model on a dataset, computing crop-level and student-pooled AUROC."""
    model.eval()
    total_loss = 0.0
    all_probs: List[float] = []
    all_labels: List[float] = []
    all_students: List[str] = []
    all_uids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(images).squeeze(-1)
                loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits.float()).cpu().numpy().tolist()
            labels_np = labels.cpu().numpy().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels_np)
            all_students.extend(batch["student_id"])
            all_uids.extend(batch["task_uid"])

    avg_loss = total_loss / max(1, len(all_labels))
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    # Crop-level metrics
    try:
        crop_auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        crop_auroc = float("nan")
    try:
        crop_auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        crop_auprc = float("nan")

    # Student-level pooled metrics (mean of crop probabilities per student)
    df_eval = pd.DataFrame({
        "student_id": all_students,
        "y_true": y_true,
        "y_prob": y_prob,
    })
    df_student = df_eval.groupby("student_id").agg({
        "y_true": "max",  # ground truth label is invariant per student
        "y_prob": "mean",  # mean predicted probability
    }).reset_index()

    y_student_true = df_student["y_true"].values
    y_student_prob = df_student["y_prob"].values

    try:
        student_auroc = float(roc_auc_score(y_student_true, y_student_prob))
    except Exception:
        student_auroc = float("nan")
    try:
        student_auprc = float(average_precision_score(y_student_true, y_student_prob))
    except Exception:
        student_auprc = float("nan")

    return {
        "loss": avg_loss,
        "crop_auroc": crop_auroc,
        "crop_auprc": crop_auprc,
        "student_auroc": student_auroc,
        "student_auprc": student_auprc,
        "df_predictions": df_eval,
    }


def train_single_run(
    run_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_dir: Path,
    model_name: str = "densenet201",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train unfrozen CNN for a single split and log epoch-by-epoch curves."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"=== Starting {run_name} on {device} ({model_name}) ===")
    logger.info(
        f"Train crops: {len(train_df)} ({train_df['label'].sum():.0f} pos, "
        f"{len(train_df) - train_df['label'].sum():.0f} neg, {train_df['student_id'].nunique()} students) | "
        f"Test crops: {len(test_df)} ({test_df['label'].sum():.0f} pos, "
        f"{len(test_df) - test_df['label'].sum():.0f} neg, {test_df['student_id'].nunique()} students)"
    )

    # Calculate class imbalance weighting: neg / pos
    n_pos = max(1.0, float(train_df["label"].sum()))
    n_neg = max(1.0, float(len(train_df) - n_pos))
    pos_weight_val = n_neg / n_pos
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    logger.info(f"Class imbalance pos_weight = {pos_weight_val:.2f}")

    train_dataset = SentenceCropDataset(train_df, dataset_dir, is_train=True)
    test_dataset = SentenceCropDataset(test_df, dataset_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows safe
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    # Loader for measuring training set AUROC without data augmentation
    train_eval_dataset = SentenceCropDataset(train_df, dataset_dir, is_train=False)
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model, total_params = build_unfrozen_model(model_name=model_name, pretrained=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    history: List[Dict[str, Any]] = []
    best_test_auroc = -1.0
    final_eval: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(images).squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * len(labels)

        scheduler.step()
        train_epoch_loss = running_train_loss / len(train_df)

        # Full evaluations
        train_eval = evaluate_dataset(model, train_eval_loader, device, criterion)
        test_eval = evaluate_dataset(model, test_loader, device, criterion)

        ep_stats = {
            "epoch": epoch,
            "train_loss": train_epoch_loss,
            "train_crop_auroc": train_eval["crop_auroc"],
            "train_student_auroc": train_eval["student_auroc"],
            "test_loss": test_eval["loss"],
            "test_crop_auroc": test_eval["crop_auroc"],
            "test_crop_auprc": test_eval["crop_auprc"],
            "test_student_auroc": test_eval["student_auroc"],
            "test_student_auprc": test_eval["student_auprc"],
        }
        history.append(ep_stats)

        if test_eval["student_auroc"] > best_test_auroc or test_eval["crop_auroc"] > best_test_auroc:
            best_test_auroc = max(
                test_eval["student_auroc"] if not np.isnan(test_eval["student_auroc"]) else 0.0,
                test_eval["crop_auroc"] if not np.isnan(test_eval["crop_auroc"]) else 0.0,
            )

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Train AUROC: {train_eval['crop_auroc']:.4f} | "
            f"Test Loss: {test_eval['loss']:.4f} | "
            f"Test Crop AUROC: {test_eval['crop_auroc']:.4f} | "
            f"Test Student AUROC: {test_eval['student_auroc']:.4f}"
        )

        final_eval = test_eval

    return history, final_eval


def run_experiment_1(
    dataset_dir: Path = SENTENCE_DATASET_DIR,
    model_name: str = "densenet201",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = 42,
    which_run: str = "both",
) -> Dict[str, Any]:
    """Execute Run 1A, Run 1B, or both, recording comparisons and plotting curves."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df_index = pd.read_csv(INDEX_CSV)
    logger.info(f"Loaded {len(df_index)} sentence crops across {df_index['student_id'].nunique()} students.")

    # Warm up in-memory image cache once
    preload_images(df_index, dataset_dir, target_size=224)

    results: Dict[str, Any] = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "seed": seed,
    }

    # ----------------------------------------------------
    # RUN 1A: The "Baseline Paper" Crop Split (Random 80/20, ignoring student_id)
    # ----------------------------------------------------
    if which_run in ("1a", "both"):
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING RUN 1A: The 'Baseline Paper' Crop Split (Random 80/20)")
        logger.info("Crops pooled together ignoring student_id (Student Leakage Intended)")
        logger.info("=" * 70)

        train_df_1a, test_df_1a = train_test_split(
            df_index,
            test_size=0.20,
            stratify=df_index["label"],
            random_state=seed,
        )

        # Count student overlap
        train_students = set(train_df_1a["student_id"])
        test_students = set(test_df_1a["student_id"])
        overlap = train_students.intersection(test_students)
        logger.info(
            f"Run 1A Student Overlap: {len(overlap)} / {len(test_students)} test students "
            f"({len(overlap) / len(test_students) * 100:.1f}%) have crops in the training set!"
        )

        history_1a, final_1a = train_single_run(
            run_name="Run 1A (The 'Baseline Paper' Crop Split)",
            train_df=train_df_1a,
            test_df=test_df_1a,
            dataset_dir=dataset_dir,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )

        df_hist_1a = pd.DataFrame(history_1a)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path_1a = REPORTS_DIR / "unfrozen_cnn_run1a_history.csv"
        df_hist_1a.to_csv(csv_path_1a, index=False)
        logger.info(f"Saved Run 1A history to {csv_path_1a}")

        results["run1a"] = {
            "history": history_1a,
            "student_overlap_pct": len(overlap) / len(test_students) * 100,
            "final_train_crop_auroc": df_hist_1a["train_crop_auroc"].iloc[-1],
            "final_test_crop_auroc": df_hist_1a["test_crop_auroc"].iloc[-1],
            "max_test_crop_auroc": df_hist_1a["test_crop_auroc"].max(),
        }

    # ----------------------------------------------------
    # RUN 1B: The Honest student_id Split (StratifiedGroupKFold 80/20)
    # ----------------------------------------------------
    if which_run in ("1b", "both"):
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING RUN 1B: The Honest student_id Split (StratifiedGroupKFold)")
        logger.info("Zero student overlap between train and test. Student probability pooling.")
        logger.info("=" * 70)

        # Use StratifiedGroupKFold(n_splits=5) so fold 0 is 80% train students, 20% test students
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(sgkf.split(df_index, y=df_index["label"], groups=df_index["student_id"]))
        train_idx, test_idx = splits[0]

        train_df_1b = df_index.iloc[train_idx].copy()
        test_df_1b = df_index.iloc[test_idx].copy()

        train_students_1b = set(train_df_1b["student_id"])
        test_students_1b = set(test_df_1b["student_id"])
        overlap_1b = train_students_1b.intersection(test_students_1b)
        logger.info(
            f"Run 1B Student Overlap: {len(overlap_1b)} students (Guaranteed 0.0% leakage)."
        )

        history_1b, final_1b = train_single_run(
            run_name="Run 1B (The Honest student_id Split)",
            train_df=train_df_1b,
            test_df=test_df_1b,
            dataset_dir=dataset_dir,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )

        df_hist_1b = pd.DataFrame(history_1b)
        csv_path_1b = REPORTS_DIR / "unfrozen_cnn_run1b_history.csv"
        df_hist_1b.to_csv(csv_path_1b, index=False)
        logger.info(f"Saved Run 1B history to {csv_path_1b}")

        results["run1b"] = {
            "history": history_1b,
            "student_overlap_pct": 0.0,
            "final_train_crop_auroc": df_hist_1b["train_crop_auroc"].iloc[-1],
            "final_test_crop_auroc": df_hist_1b["test_crop_auroc"].iloc[-1],
            "final_test_student_auroc": df_hist_1b["test_student_auroc"].iloc[-1],
            "max_test_crop_auroc": df_hist_1b["test_crop_auroc"].max(),
            "max_test_student_auroc": df_hist_1b["test_student_auroc"].max(),
        }

    # Save summary JSON
    summary_path = REPORTS_DIR / "unfrozen_cnn_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved comparison summary to {summary_path}")

    # Generate comparative plot if both runs completed
    if which_run == "both":
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            epochs_arr = df_hist_1a["epoch"].values

            # Plot 1A
            ax1.plot(epochs_arr, df_hist_1a["train_crop_auroc"], label="Train AUROC (Crops)", color="#1f77b4", lw=2.5)
            ax1.plot(epochs_arr, df_hist_1a["test_crop_auroc"], label="Test AUROC (Crops - Leaked)", color="#ff7f0e", lw=2.5, ls="--")
            ax1.set_title("Run 1A: 'Baseline Paper' Crop Split\n(Random 80/20, 85%+ Student Leakage)", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Epoch", fontsize=10)
            ax1.set_ylabel("AUROC", fontsize=10)
            ax1.set_ylim([0.45, 1.02])
            ax1.axhline(0.95, color="green", ls=":", alpha=0.7, label="0.95 Paper Benchmark")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="lower right")

            # Plot 1B
            ax2.plot(epochs_arr, df_hist_1b["train_crop_auroc"], label="Train AUROC (Crops)", color="#1f77b4", lw=2.5)
            ax2.plot(epochs_arr, df_hist_1b["test_crop_auroc"], label="Test AUROC (Crops - Honest)", color="#2ca02c", lw=2, ls="--")
            ax2.plot(epochs_arr, df_hist_1b["test_student_auroc"], label="Test AUROC (Student Pooled)", color="#d62728", lw=2.5)
            ax2.set_title("Run 1B: Honest student_id Split\n(StratifiedGroupKFold, Zero Student Leakage)", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Epoch", fontsize=10)
            ax2.axhline(0.70, color="gray", ls=":", alpha=0.7, label="0.70 Tabular Baseline")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="lower right")

            plt.suptitle(f"The 'Baseline Paper' Scam Verdict: DenseNet201 Full Unfreeze ({model_name})", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            plot_path = REPORTS_DIR / "unfrozen_cnn_curves.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            logger.info(f"Generated comparison curve plot at {plot_path}")
        except Exception as e:
            logger.warning(f"Could not generate matplotlib plot: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Paper Full Unfreeze Experiment")
    parser.add_argument("--model", type=str, default="densenet201", choices=["densenet201", "densenet121", "convnext_base"])
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training/eval")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run", type=str, default="both", choices=["1a", "1b", "both"], help="Which run to execute")
    args = parser.parse_args()

    run_experiment_1(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        which_run=args.run,
    )


if __name__ == "__main__":
    main()

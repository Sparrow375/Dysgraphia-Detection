"""Stage 31: Experiment 2 — The "De-Noised Big Gun" (3-Channel Clean Ink Tensor -> Vision Transformer).

Scientific Objective:
---------------------
When feeding raw grayscale images into giant vision models (DenseNet, Swin-T, ConvNeXt),
the network often expends its parameter capacity memorizing yellow background lighting,
paper grain, desk shadows, and notebook ruled lines.

In this experiment, we feed a custom 3-channel de-noised tensor where the background
is pitch black (0.0) across all 3 channels, forcing the Vision Transformer's 28M parameters
to focus purely on handwriting mechanics:
  - Channel 1 (Binary Shape): Clean ink mask ({task_id}_clean_ink.png) with ruled lines zeroed out.
  - Channel 2 (Pencil/Pen Pressure): Optical density masked strictly to ink pixels (D(x, y) in [0, 1]).
  - Channel 3 (Stroke Thickness): Euclidean distance transform map normalized to [0, 1].

Two Benchmark Splits:
---------------------
Run 2A (The De-Noised Crop Split):
  - Standard random 80/20 train/test split pooling all crops together (ignoring student_id).
  - Tests whether eliminating background paper texture reduces artificial crop leakage inflation.

Run 2B (The Honest student_id Split):
  - StratifiedGroupKFold split strictly by student_id (0.0% student leakage).
  - Evaluates both raw crop-level AUROC and student-level pooled probabilities.
  - Diagnostic Test: Does eliminating paper background noise allow the Vision Transformer
    to generalize better (pushing beyond the 0.7893 AUROC achieved on raw pixels in Exp 1)?
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import (
    ConvNeXt_Small_Weights,
    DenseNet201_Weights,
    Swin_T_Weights,
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
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = REPO_ROOT / "reports"
CACHE_NPZ_PATH = FEATURES_DIR / "denoised_3ch_tensors_224.npz"


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_single_3ch_tensor(
    clean_mask_path: Path,
    raw_gray_path: Path,
    target_size: int = 224,
) -> np.ndarray:
    """Construct custom 3-channel tensor (Binary Shape, Pressure, Thickness) padded onto pitch black canvas.
    
    Returns:
        np.ndarray of shape (target_size, target_size, 3) with dtype uint8 (values 0..255).
    """
    # 1. Load clean mask
    mask_gray = cv2.imread(str(clean_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"Could not load mask from {clean_mask_path}")
    ink_bin = (mask_gray < 128).astype(np.float32)

    # 2. Load raw grayscale crop
    raw_gray = cv2.imread(str(raw_gray_path), cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        raise FileNotFoundError(f"Could not load gray image from {raw_gray_path}")

    # Match dimensions if needed
    if ink_bin.shape != raw_gray.shape:
        ink_bin = cv2.resize(
            ink_bin, (raw_gray.shape[1], raw_gray.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # Standardize polarity: paper should be bright, ink dark
    if np.mean(raw_gray) < 128:
        raw_gray = 255 - raw_gray

    # Channel 1: Binary Shape (0 or 1.0)
    ch1 = ink_bin

    # Channel 2: Pencil/Pen Pressure (Optical Density masked to ink)
    non_ink = raw_gray[ink_bin == 0]
    bg_level = float(np.percentile(non_ink, 90)) if len(non_ink) > 50 else 255.0
    bg_level = max(128.0, bg_level)
    optical_density = np.clip((bg_level - raw_gray.astype(np.float32)) / bg_level, 0.0, 1.0)
    ch2 = optical_density * ink_bin

    # Channel 3: Stroke Thickness (Euclidean Distance Transform)
    dist_map = cv2.distanceTransform((ink_bin * 255).astype(np.uint8), cv2.DIST_L2, 5)
    ink_dists = dist_map[ink_bin > 0]
    max_d = float(np.percentile(ink_dists, 99)) if len(ink_dists) > 0 else 1.0
    max_d = max(1.0, max_d)
    ch3 = np.clip(dist_map / max_d, 0.0, 1.0) * ink_bin

    # Stack channels: shape (H, W, 3)
    t3 = np.stack([ch1, ch2, ch3], axis=-1)

    # Aspect-preserving proportional resizing onto black canvas
    h, w, c = t3.shape
    scale = target_size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(t3, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, c), dtype=np.float32)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized

    # Convert to uint8 (0..255) for compact RAM caching
    return (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)


def precompute_or_load_3ch_dataset(
    df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
    cache_path: Path = CACHE_NPZ_PATH,
    target_size: int = 224,
) -> Dict[str, np.ndarray]:
    """Precompute all 1,145 3-channel tensors and cache in RAM / disk."""
    if cache_path.exists():
        logger.info(f"Loading cached 3-channel tensors from {cache_path}...")
        npz = np.load(cache_path)
        tensor_dict = {key: npz[key] for key in npz.files}
        if len(tensor_dict) == len(df):
            logger.info(f"Loaded all {len(tensor_dict)} tensors from cache successfully.")
            return tensor_dict

    logger.info(f"Precomputing 3-channel de-noised tensors for {len(df)} crops...")
    tensor_dict: Dict[str, np.ndarray] = {}

    for idx, row in df.iterrows():
        school = str(row["school"])
        sid = str(row["student_id"])
        task_id = str(row["task_id"])
        task_uid = str(row["task_uid"])

        p_mask = processed_dir / school / sid / f"{task_id}_clean_ink.png"
        p_gray = processed_dir / school / sid / f"{task_id}.png"

        tensor_uint8 = build_single_3ch_tensor(p_mask, p_gray, target_size=target_size)
        tensor_dict[task_uid] = tensor_uint8

        if (idx + 1) % 200 == 0 or (idx + 1) == len(df):
            logger.info(f"Precomputed {idx + 1}/{len(df)} tensors...")

    # Save to disk cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **tensor_dict)
    logger.info(f"Saved {len(tensor_dict)} precomputed tensors to {cache_path}.")
    return tensor_dict


class Denoised3ChDataset(Dataset):
    """PyTorch Dataset for custom 3-channel de-noised handwriting tensors."""

    def __init__(
        self,
        df: pd.DataFrame,
        tensors_dict: Dict[str, np.ndarray],
        is_train: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tensors_dict = tensors_dict
        self.is_train = is_train

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.is_train:
            self.aug_pil = transforms.Compose([
                transforms.RandomRotation(degrees=(-5, 5), fill=0),
                transforms.RandomAffine(degrees=0, scale=(0.85, 1.15), fill=0),
            ])
            self.random_erasing = transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.20),
                ratio=(0.3, 3.3),
                value=0.0,  # Erase to black background (simulating pen lifts/breaks)
            )
        else:
            self.aug_pil = None
            self.random_erasing = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        task_uid = str(row["task_uid"])
        uint8_tensor = self.tensors_dict[task_uid]  # Shape (224, 224, 3)

        # Convert to PIL for spatial transforms
        pil_img = Image.fromarray(uint8_tensor)

        if self.is_train and self.aug_pil is not None:
            pil_img = self.aug_pil(pil_img)

        # To tensor in [0.0, 1.0] and normalize
        tensor_img = transforms.functional.to_tensor(pil_img)
        tensor_img = self.norm(tensor_img)

        if self.is_train and self.random_erasing is not None:
            tensor_img = self.random_erasing(tensor_img)

        label = float(row["label"])

        return {
            "image": tensor_img,
            "label": torch.tensor(label, dtype=torch.float32),
            "student_id": str(row["student_id"]),
            "task_uid": task_uid,
            "grade": int(row.get("grade", 0)),
        }


def build_unfrozen_vision_model(
    model_name: str = "swin_t",
    pretrained: bool = True,
    dropout: float = 0.2,
) -> Tuple[nn.Module, int]:
    """Build a fully unfrozen vision model (Swin-T, ConvNeXt-Small, or DenseNet201).
    
    Returns:
        (model, total_parameters)
    """
    model_name = model_name.lower()
    if model_name == "swin_t":
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features  # 768
        model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
    elif model_name == "convnext_small":
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features  # 768
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
    elif model_name == "densenet201":
        weights = DenseNet201_Weights.DEFAULT if pretrained else None
        model = models.densenet201(weights=weights)
        in_features = model.classifier.in_features  # 1920
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    total_params = 0
    for param in model.parameters():
        param.requires_grad = True
        total_params += param.numel()

    logger.info(
        f"Built fully unfrozen {model_name}: {total_params:,} parameters "
        f"({total_params / 1e6:.2f}M trainable)."
    )
    return model, total_params


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, Any]:
    """Evaluate model computing both crop-level and student-pooled AUROC."""
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

    try:
        crop_auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        crop_auroc = float("nan")
    try:
        crop_auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        crop_auprc = float("nan")

    df_eval = pd.DataFrame({
        "student_id": all_students,
        "y_true": y_true,
        "y_prob": y_prob,
    })
    df_student = df_eval.groupby("student_id").agg({
        "y_true": "max",
        "y_prob": "mean",
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
    tensors_dict: Dict[str, np.ndarray],
    model_name: str = "swin_t",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train unfrozen Vision Transformer on 3-channel de-noised tensors."""
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

    n_pos = max(1.0, float(train_df["label"].sum()))
    n_neg = max(1.0, float(len(train_df) - n_pos))
    pos_weight_val = n_neg / n_pos
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    logger.info(f"Class imbalance pos_weight = {pos_weight_val:.2f}")

    train_dataset = Denoised3ChDataset(train_df, tensors_dict, is_train=True)
    test_dataset = Denoised3ChDataset(test_df, tensors_dict, is_train=False)
    train_eval_dataset = Denoised3ChDataset(train_df, tensors_dict, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model, total_params = build_unfrozen_vision_model(model_name=model_name, pretrained=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    history: List[Dict[str, Any]] = []
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


def run_experiment_2(
    model_name: str = "swin_t",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = 42,
    which_run: str = "both",
) -> Dict[str, Any]:
    """Execute Experiment 2: Run 2A and/or Run 2B with comparative curve generation."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df_index = pd.read_csv(INDEX_CSV)
    logger.info(f"Loaded {len(df_index)} sentence crops across {df_index['student_id'].nunique()} students.")

    # 1. Precompute or load 3-channel de-noised tensors
    tensors_dict = precompute_or_load_3ch_dataset(df_index, PROCESSED_DIR, CACHE_NPZ_PATH, target_size=224)

    results: Dict[str, Any] = {
        "experiment": "Experiment 2: De-Noised 3-Channel Vision Transformer",
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "seed": seed,
    }

    # ----------------------------------------------------
    # RUN 2A: The De-Noised Crop Split (Random 80/20)
    # ----------------------------------------------------
    if which_run in ("2a", "both"):
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING RUN 2A: The De-Noised Crop Split (Random 80/20)")
        logger.info("3-Channel Tensors (Shape + Pressure + Thickness), Black Background")
        logger.info("=" * 70)

        train_df_2a, test_df_2a = train_test_split(
            df_index,
            test_size=0.20,
            stratify=df_index["label"],
            random_state=seed,
        )

        train_students = set(train_df_2a["student_id"])
        test_students = set(test_df_2a["student_id"])
        overlap = train_students.intersection(test_students)
        logger.info(
            f"Run 2A Student Overlap: {len(overlap)} / {len(test_students)} test students "
            f"({len(overlap) / len(test_students) * 100:.1f}%)"
        )

        history_2a, final_2a = train_single_run(
            run_name="Run 2A (De-Noised Crop Split)",
            train_df=train_df_2a,
            test_df=test_df_2a,
            tensors_dict=tensors_dict,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )

        df_hist_2a = pd.DataFrame(history_2a)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path_2a = REPORTS_DIR / "denoised_vit_run2a_history.csv"
        df_hist_2a.to_csv(csv_path_2a, index=False)
        logger.info(f"Saved Run 2A history to {csv_path_2a}")

        results["run2a"] = {
            "history": history_2a,
            "student_overlap_pct": len(overlap) / len(test_students) * 100,
            "final_train_crop_auroc": df_hist_2a["train_crop_auroc"].iloc[-1],
            "final_test_crop_auroc": df_hist_2a["test_crop_auroc"].iloc[-1],
            "final_test_student_auroc": df_hist_2a["test_student_auroc"].iloc[-1],
            "max_test_crop_auroc": df_hist_2a["test_crop_auroc"].max(),
            "max_test_student_auroc": df_hist_2a["test_student_auroc"].max(),
        }

    # ----------------------------------------------------
    # RUN 2B: The Honest student_id Split (StratifiedGroupKFold 80/20)
    # ----------------------------------------------------
    if which_run in ("2b", "both"):
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING RUN 2B: The Honest student_id Split (StratifiedGroupKFold)")
        logger.info("3-Channel Tensors, Zero Student Overlap, Student Probability Pooling")
        logger.info("=" * 70)

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(sgkf.split(df_index, y=df_index["label"], groups=df_index["student_id"]))
        train_idx, test_idx = splits[0]

        train_df_2b = df_index.iloc[train_idx].copy()
        test_df_2b = df_index.iloc[test_idx].copy()

        train_students_2b = set(train_df_2b["student_id"])
        test_students_2b = set(test_df_2b["student_id"])
        overlap_2b = train_students_2b.intersection(test_students_2b)
        logger.info(f"Run 2B Student Overlap: {len(overlap_2b)} students (0.0% leakage).")

        history_2b, final_2b = train_single_run(
            run_name="Run 2B (De-Noised Honest student_id Split)",
            train_df=train_df_2b,
            test_df=test_df_2b,
            tensors_dict=tensors_dict,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )

        df_hist_2b = pd.DataFrame(history_2b)
        csv_path_2b = REPORTS_DIR / "denoised_vit_run2b_history.csv"
        df_hist_2b.to_csv(csv_path_2b, index=False)
        logger.info(f"Saved Run 2B history to {csv_path_2b}")

        results["run2b"] = {
            "history": history_2b,
            "student_overlap_pct": 0.0,
            "final_train_crop_auroc": df_hist_2b["train_crop_auroc"].iloc[-1],
            "final_test_crop_auroc": df_hist_2b["test_crop_auroc"].iloc[-1],
            "final_test_student_auroc": df_hist_2b["test_student_auroc"].iloc[-1],
            "max_test_crop_auroc": df_hist_2b["test_crop_auroc"].max(),
            "max_test_student_auroc": df_hist_2b["test_student_auroc"].max(),
        }

    # Save summary JSON
    summary_path = REPORTS_DIR / "denoised_vit_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved comparison summary to {summary_path}")

    # Plot comparative trajectories
    if which_run == "both":
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            epochs_arr = df_hist_2a["epoch"].values

            # Plot 2A
            ax1.plot(epochs_arr, df_hist_2a["train_crop_auroc"], label="Train AUROC (Crops)", color="#1f77b4", lw=2.5)
            ax1.plot(epochs_arr, df_hist_2a["test_crop_auroc"], label="Test AUROC (Crops - Leaked)", color="#ff7f0e", lw=2.5, ls="--")
            ax1.set_title(f"Run 2A: De-Noised Crop Split ({model_name.upper()})\n(Random 80/20, Student Leakage)", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Epoch", fontsize=10)
            ax1.set_ylabel("AUROC", fontsize=10)
            ax1.set_ylim([0.45, 1.02])
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="lower right")

            # Plot 2B
            ax2.plot(epochs_arr, df_hist_2b["train_crop_auroc"], label="Train AUROC (Crops)", color="#1f77b4", lw=2.5)
            ax2.plot(epochs_arr, df_hist_2b["test_crop_auroc"], label="Test AUROC (Crops - Honest)", color="#2ca02c", lw=2, ls="--")
            ax2.plot(epochs_arr, df_hist_2b["test_student_auroc"], label="Test AUROC (Student Pooled)", color="#d62728", lw=2.5)
            ax2.set_title(f"Run 2B: De-Noised Honest student_id Split ({model_name.upper()})\n(StratifiedGroupKFold, Zero Leakage)", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Epoch", fontsize=10)
            ax2.axhline(0.7893, color="purple", ls=":", alpha=0.7, label="0.7893 Exp 1 DenseNet Baseline")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="lower right")

            plt.suptitle(f"Experiment 2: De-Noised 3-Channel Tensor to {model_name.upper()} (~28M Params)", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            plot_path = REPORTS_DIR / "denoised_vit_curves.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            logger.info(f"Generated comparison curve plot at {plot_path}")
        except Exception as e:
            logger.warning(f"Could not generate matplotlib plot: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: De-Noised 3-Channel Vision Transformer")
    parser.add_argument("--model", type=str, default="swin_t", choices=["swin_t", "convnext_small", "densenet201"])
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training/eval")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run", type=str, default="both", choices=["2a", "2b", "both"], help="Which run to execute")
    args = parser.parse_args()

    run_experiment_2(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        which_run=args.run,
    )


if __name__ == "__main__":
    main()

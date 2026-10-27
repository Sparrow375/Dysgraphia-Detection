"""Phase 5: Full 5-Fold Cross-Validation for Experiment 6 Dual-Script TrOCR MIL.

Evaluates all 215 students out-of-fold (175 controls, 40 dysgraphic positives)
under locked student-stratified fold assignments with 0.0% student leakage:
- Pretrained Dual-Script TrOCR sequence representations (ViT Devanagari + DeiT Latin)
- Tripled training bags (Clean Original + Cutout Duplicate + Micro-Shear Duplicate)
- Zero-split training on 100% of 172 training students (12 epochs, cosine annealing)
- SWA (Stochastic Weight Averaging) across peak generalization window (Epochs 7-12)
- Training-anchored logit standardization for unified 5-fold probability calibration
- Accelerated with PyTorch AMP on RTX 5060 Ti GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from baseline.cache_dual_trocr_vision import (
    CACHE_NPZ_PATH,
    precompute_and_cache_dual_trocr_tokens,
)
from baseline.dual_trocr_multimodal_mil import (
    DualScriptMultiModalMILModel,
    DualScriptStudentBagDataset,
    SWAModelTracker,
    collate_dual_script_bags,
)
from baseline.multimodal_features import (
    SELECTED_MATH_FEATURES_16,
    fit_and_scale_multimodal_features,
    load_raw_multimodal_dataframe,
)
from baseline.train_whole_page_mil import FlushingStreamHandler
from models.evaluator import calculate_sensitivity_at_specificity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[FlushingStreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"
FOLDS_CSV = DATA_DIR / "folds" / "student_folds_lookup.csv"
REPORTS_DIR = REPO_ROOT / "reports"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def extract_raw_logits(
    model: DualScriptMultiModalMILModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract uncalibrated raw logits from model forward pass."""
    model.eval()
    all_logits: List[float] = []
    all_labels: List[int] = []
    student_ids: List[str] = []

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            for batch in loader:
                tokens = [t.to(device) for t in batch["all_tokens"]]
                scripts = batch["all_scripts"]
                math_feats = batch["all_math"].to(device)
                splits = batch["split_indices"]
                labels = batch["labels"].cpu().numpy()

                logits, _, _ = model(tokens, scripts, math_feats, splits)

                all_logits.extend(logits.cpu().numpy().tolist())
                all_labels.extend(labels.tolist())
                student_ids.extend(batch["student_ids"])

    return np.array(all_logits, dtype=float), np.array(all_labels, dtype=int), student_ids


def train_dual_trocr_fold(
    fold_idx: int,
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    tokens_cache: Dict[str, np.ndarray],
    math_dict: Dict[str, np.ndarray],
    aux_dict: Dict[str, np.ndarray],
    scripts_dict: Dict[str, str],
    epochs: int = 12,
    batch_size: int = 8,
    lr: float = 3e-4,
    lambda_aux: float = 0.30,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Train Experiment 6 model on one fold with SWA and training-anchored calibration."""
    set_seed(seed + fold_idx)

    train_labels = [r["label"] for r in train_records]
    n_pos = max(1.0, float(sum(train_labels)))
    n_neg = max(1.0, float(len(train_labels) - n_pos))
    pos_weight = torch.tensor([n_neg / n_pos], device=device)

    # Duplicate-Only Bag Datasets: Tripling training data safely
    ds_clean = DualScriptStudentBagDataset(train_records, tokens_cache, math_dict, aux_dict, scripts_dict, bag_type="clean")
    ds_cutout = DualScriptStudentBagDataset(train_records, tokens_cache, math_dict, aux_dict, scripts_dict, bag_type="cutout")
    ds_shear = DualScriptStudentBagDataset(train_records, tokens_cache, math_dict, aux_dict, scripts_dict, bag_type="shear")

    # Combine into 516 training bags (172 x 3)
    combined_train_records = []
    # Interleave bags: Clean, Cutout, Shear
    for i in range(len(train_records)):
        combined_train_records.append((ds_clean, i))
        combined_train_records.append((ds_cutout, i))
        combined_train_records.append((ds_shear, i))

    class CombinedAugmentedDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            ds, inner_idx = self.items[idx]
            return ds[inner_idx]

    train_dataset = CombinedAugmentedDataset(combined_train_records)

    # Test dataset: Strictly 100% Clean Original Bags
    test_dataset = DualScriptStudentBagDataset(test_records, tokens_cache, math_dict, aux_dict, scripts_dict, bag_type="clean")
    train_clean_dataset = ds_clean  # For training logit anchor stats

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dual_script_bags, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dual_script_bags, pin_memory=True, num_workers=0)
    train_clean_loader = DataLoader(train_clean_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dual_script_bags, pin_memory=True, num_workers=0)

    model = DualScriptMultiModalMILModel(math_dim=16, visual_dim=32, math_emb_dim=32, attn_dim=32, dropout=0.40).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    aux_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    swa_tracker = SWAModelTracker(model)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            tokens = [t.to(device) for t in batch["all_tokens"]]
            scripts = batch["all_scripts"]
            math_feats = batch["all_math"].to(device)
            aux_targets = batch["all_aux"].to(device)
            labels = batch["labels"].to(device)
            splits = batch["split_indices"]

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _, aux_preds = model(tokens, scripts, math_feats, splits)
                loss_main = criterion(logits, labels)
                loss_aux = aux_criterion(aux_preds, aux_targets)
                loss_total = loss_main + (lambda_aux * loss_aux)

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss_total.item() * len(labels)
            n_samples += len(labels)

        scheduler.step()

        # SWA Weight Averaging across peak window: Epochs 7-12
        if epoch >= 7:
            swa_tracker.update(model)

        if epoch % 3 == 0 or epoch == epochs:
            logger.info("  Fold %d | Ep %02d/%02d | Loss: %.4f | SWA Models: %d", fold_idx + 1, epoch, epochs, train_loss / max(1, n_samples), swa_tracker.n_models)

    # Apply SWA averaged weights to model
    logger.info("  Fold %d: Applying SWA averaged weights from %d models...", fold_idx + 1, swa_tracker.n_models)
    swa_tracker.apply_to(model)

    # Compute training logit anchor stats on clean training bags
    train_logits, _, _ = extract_raw_logits(model, train_clean_loader, device)
    mu_train = float(np.mean(train_logits))
    sigma_train = max(1e-4, float(np.std(train_logits)))

    # Compute test logits and standardize using training anchor stats
    test_logits, test_labels, test_sids = extract_raw_logits(model, test_loader, device)
    z_norm = (test_logits - mu_train) / sigma_train
    test_probs = 1.0 / (1.0 + np.exp(-z_norm))

    test_auroc = float(roc_auc_score(test_labels, test_probs))
    test_auprc = float(average_precision_score(test_labels, test_probs))
    sens_80 = calculate_sensitivity_at_specificity(test_labels, test_probs, 0.80)
    sens_90 = calculate_sensitivity_at_specificity(test_labels, test_probs, 0.90)

    test_meta: List[Dict[str, Any]] = []
    for sid, y, p, z in zip(test_sids, test_labels, test_probs, z_norm):
        test_meta.append({
            "student_id": sid,
            "outer_fold": fold_idx,
            "label": int(y),
            "pred_prob": float(p),
            "norm_logit": float(z),
        })

    metrics = {
        "fold": fold_idx,
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "sens_80_spec": float(sens_80),
        "sens_90_spec": float(sens_90),
        "mu_train": mu_train,
        "sigma_train": sigma_train,
        "swa_models": swa_tracker.n_models,
    }

    return metrics, test_meta


def run_exp6_5fold(device: torch.device) -> Dict[str, Any]:
    """Execute full 5-fold cross-validation for Experiment 6."""
    tokens_cache = precompute_and_cache_dual_trocr_tokens()
    df_raw = load_raw_multimodal_dataframe()

    student_folds_df = pd.read_csv(FOLDS_CSV)
    student_to_fold = dict(zip(student_folds_df["student_id"], student_folds_df["repeat_0_fold"]))

    scripts_dict = dict(zip(df_raw["task_uid"], df_raw["script"]))
    students_list = sorted(list(df_raw["student_id"].unique()))

    all_fold_metrics: List[Dict[str, Any]] = []
    all_oof_predictions: List[Dict[str, Any]] = []

    logger.info("=================================================================")
    logger.info("Executing Experiment 6: Dual-Script TrOCR Vision + Multi-Modal MIL")
    logger.info("Cohort: %d Students (40 Positives, 175 Controls; G3-G8)", len(students_list))
    logger.info("Evaluation: 5-Fold Student Stratified CV (0.0%% Leakage; All OOF)")
    logger.info("=================================================================")

    t0 = time.time()

    for fold_idx in range(5):
        test_students = [s for s in students_list if student_to_fold.get(s, -1) == fold_idx]
        train_students = [s for s in students_list if student_to_fold.get(s, -1) != fold_idx]

        train_uids = list(df_raw[df_raw["student_id"].isin(train_students)]["task_uid"])
        test_uids = list(df_raw[df_raw["student_id"].isin(test_students)]["task_uid"])

        math_dict, aux_dict, _ = fit_and_scale_multimodal_features(
            df_raw, train_uids, test_uids, feature_cols=SELECTED_MATH_FEATURES_16
        )

        train_records = []
        for sid in train_students:
            sdf = df_raw[df_raw["student_id"] == sid]
            train_records.append({
                "student_id": sid,
                "school": sdf["school"].iloc[0],
                "grade": int(sdf["grade"].iloc[0]),
                "label": int(sdf["label"].iloc[0]),
                "task_uids": list(sdf["task_uid"]),
                "task_names": list(sdf["task_name"]),
            })

        test_records = []
        for sid in test_students:
            sdf = df_raw[df_raw["student_id"] == sid]
            test_records.append({
                "student_id": sid,
                "school": sdf["school"].iloc[0],
                "grade": int(sdf["grade"].iloc[0]),
                "label": int(sdf["label"].iloc[0]),
                "task_uids": list(sdf["task_uid"]),
                "task_names": list(sdf["task_name"]),
            })

        metrics, fold_meta = train_dual_trocr_fold(
            fold_idx=fold_idx,
            train_records=train_records,
            test_records=test_records,
            tokens_cache=tokens_cache,
            math_dict=math_dict,
            aux_dict=aux_dict,
            scripts_dict=scripts_dict,
            epochs=12,
            batch_size=8,
            lr=3e-4,
            lambda_aux=0.30,
            device=device,
        )

        all_fold_metrics.append(metrics)
        all_oof_predictions.extend(fold_meta)

        logger.info(
            "--> Fold %d/5 Complete: Test AUROC = %.4f | Test AUPRC = %.4f | Sens@80%% = %.1f%% | Sens@90%% = %.1f%%",
            fold_idx + 1,
            metrics["test_auroc"],
            metrics["test_auprc"],
            metrics["sens_80_spec"] * 100,
            metrics["sens_90_spec"] * 100,
        )

    elapsed = time.time() - t0

    # Compute pooled metrics across all 215 students
    y_true = np.array([p["label"] for p in all_oof_predictions], dtype=int)
    y_pred = np.array([p["pred_prob"] for p in all_oof_predictions], dtype=float)

    pooled_auroc = float(roc_auc_score(y_true, y_pred))
    pooled_auprc = float(average_precision_score(y_true, y_pred))
    pooled_brier = float(brier_score_loss(y_true, y_pred))
    sens_80 = calculate_sensitivity_at_specificity(y_true, y_pred, 0.80)
    sens_90 = calculate_sensitivity_at_specificity(y_true, y_pred, 0.90)

    mean_fold_auroc = float(np.mean([m["test_auroc"] for m in all_fold_metrics]))
    std_fold_auroc = float(np.std([m["test_auroc"] for m in all_fold_metrics]))
    mean_fold_auprc = float(np.mean([m["test_auprc"] for m in all_fold_metrics]))
    std_fold_auprc = float(np.std([m["test_auprc"] for m in all_fold_metrics]))

    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "pooled_auroc": round(pooled_auroc, 4),
        "pooled_auprc": round(pooled_auprc, 4),
        "pooled_brier": round(pooled_brier, 4),
        "sens_at_80_spec": round(float(sens_80), 4),
        "sens_at_90_spec": round(float(sens_90), 4),
        "mean_fold_auroc": round(mean_fold_auroc, 4),
        "std_fold_auroc": round(std_fold_auroc, 4),
        "mean_fold_auprc": round(mean_fold_auprc, 4),
        "std_fold_auprc": round(std_fold_auprc, 4),
        "fold_metrics": all_fold_metrics,
    }

    logger.info("=================================================================")
    logger.info("Experiment 6 Benchmark Completed in %.1f seconds", elapsed)
    logger.info("Pooled Out-Of-Fold AUROC (All 215): %.4f", pooled_auroc)
    logger.info("Mean Fold AUROC: %.4f ± %.4f", mean_fold_auroc, std_fold_auroc)
    logger.info("Pooled Out-Of-Fold AUPRC: %.4f (Prevalence = 0.1860)", pooled_auprc)
    logger.info("Sensitivity @ 80%% Specificity: %.1f%%", sens_80 * 100)
    logger.info("Sensitivity @ 90%% Specificity: %.1f%%", sens_90 * 100)
    logger.info("=================================================================")

    # Export Out-of-Fold Predictions CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame(all_oof_predictions)
    pred_csv_path = REPORTS_DIR / "exp6_dual_trocr_oof_predictions.csv"
    oof_df.to_csv(pred_csv_path, index=False)
    logger.info("Saved OOF predictions CSV to %s", pred_csv_path)

    # Export Benchmark JSON
    results_json_path = REPORTS_DIR / "exp6_dual_trocr_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved benchmark metrics JSON to %s", results_json_path)

    # Plot ROC and PR Curves
    plot_curves(y_true, y_pred, pooled_auroc, pooled_auprc)

    # Author Whitepaper Report
    author_empirical_report(summary, all_fold_metrics)

    return summary


def plot_curves(y_true: np.ndarray, y_pred: np.ndarray, auroc: float, auprc: float) -> None:
    """Plot and save ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    axes[0].plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Exp 6 Dual-Script TrOCR (AUROC = {auroc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Chance (AUROC = 0.5000)")
    axes[0].axvline(0.20, color="#dc2626", linestyle=":", label="80% Specificity (FPR = 0.20)")
    axes[0].axvline(0.10, color="#ea580c", linestyle=":", label="90% Specificity (FPR = 0.10)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.02])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    axes[0].set_title("ROC Curve (All 215 Students Out-Of-Fold)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", framealpha=0.95)
    axes[0].grid(True, alpha=0.3)

    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    baseline_prev = np.mean(y_true)
    axes[1].plot(recall, precision, color="#059669", lw=2.5, label=f"Exp 6 Dual-Script TrOCR (AUPRC = {auprc:.4f})")
    axes[1].axhline(baseline_prev, color="#94a3b8", lw=1.5, linestyle="--", label=f"Prevalence ({baseline_prev:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.02])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Precision (PPV)", fontsize=11, fontweight="bold")
    axes[1].set_title("Precision-Recall Curve (All 215 Students Out-Of-Fold)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", framealpha=0.95)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = REPORTS_DIR / "exp6_dual_trocr_roc_pr_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved ROC and PR curves plot to %s", plot_path)


def author_empirical_report(summary: Dict[str, Any], fold_metrics: List[Dict[str, Any]]) -> None:
    """Author empirical whitepaper report."""
    report_path = REPORTS_DIR / "exp6_dual_trocr_empirical_report.md"
    lines = [
        "# Empirical Report: Experiment 6 — Dual-Script TrOCR Vision + Multi-Modal MIL Benchmark",
        "",
        "**Date**: October 27, 2026  ",
        "**Cohort Standard**: Unified 215-Student Cohort (**40 Positives, 175 Controls**; G3–G8)  ",
        "**Evaluation Standard**: Locked 5-Fold Stratified Cross-Validation by `student_id` (**0.0% Student Leakage, All 215 Students Evaluated Out-Of-Fold**)  ",
        "**Hardware Platform**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM), AMD Ryzen 5 7600X, 32GB DDR5 RAM  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "Experiment 6 evaluated the Dual-Script TrOCR Vision + Multi-Modal Gated-Attention MIL pipeline, designed to eliminate ImageNet domain mismatch and sentence geometry destruction:",
        "- **Hindi Branch**: `paudelanil/trocr-devanagari-2` (ViT 768-d) routing all Devanagari lines.",
        "- **English Branch**: `microsoft/trocr-small-handwritten` (DeiT 384-d) routing all Latin lines.",
        "- **White-Canvas Physics Fix**: 3-channel input ($384 \\times 384 \\times 3$) rendered on pure white (1.0) canvas matching TrOCR pretraining.",
        "- **Duplicate Bag Augmentation**: Tripled training bags (Clean, Cutout, Micro-Shear) with test sets strictly evaluated on clean originals.",
        "- **SWA Peak-Window Training**: Stochastic Weight Averaging across Epochs 7–12 with zero-split training on 100% of training students.",
        "- **Training-Anchored Logit Standardization**: Standardized test logits using fold-specific training distribution.",
        "",
        "### Key Benchmark Results:",
        f"- **Pooled Out-Of-Fold AUROC (All 215 Students)**: **{summary['pooled_auroc']:.4f}**",
        f"- **Mean Fold AUROC**: **{summary['mean_fold_auroc']:.4f} ± {summary['std_fold_auroc']:.4f}**",
        f"- **Pooled Out-Of-Fold AUPRC**: **{summary['pooled_auprc']:.4f}** (Baseline prevalence = 0.1860)",
        f"- **Pooled Brier Score**: **{summary['pooled_brier']:.4f}**",
        f"- **Sensitivity @ 80% Specificity**: **{summary['sens_at_80_spec']*100:.1f}%**",
        f"- **Sensitivity @ 90% Specificity**: **{summary['sens_at_90_spec']*100:.1f}%**",
        "",
        "---",
        "",
        "## 2. Per-Fold Performance Breakdown",
        "",
        "| Fold | Test Students (Positives) | Test AUROC | Test AUPRC | Sens @ 80% Spec | Sens @ 90% Spec | Training Logit Mean | Training Logit Std | SWA Models |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for fm in fold_metrics:
        lines.append(
            f"| **Fold {fm['fold']+1}** | 43 (8 pos) | {fm['test_auroc']:.4f} | {fm['test_auprc']:.4f} | "
            f"{fm['sens_80_spec']*100:.1f}% | {fm['sens_90_spec']*100:.1f}% | "
            f"{fm['mu_train']:.3f} | {fm['sigma_train']:.3f} | {fm['swa_models']} |"
        )

    lines.extend([
        f"| **Mean ± Std** | — | **{summary['mean_fold_auroc']:.4f} ± {summary['std_fold_auroc']:.4f}** | "
        f"**{summary['mean_fold_auprc']:.4f} ± {summary['std_fold_auprc']:.4f}** | "
        f"**{summary['sens_at_80_spec']*100:.1f}%** | **{summary['sens_at_90_spec']*100:.1f}%** | — | — | — |",
        f"| **Pooled (All 215)** | **215 (40 pos)** | **{summary['pooled_auroc']:.4f}** | "
        f"**{summary['pooled_auprc']:.4f}** | **{summary['sens_at_80_spec']*100:.1f}%** | "
        f"**{summary['sens_at_90_spec']*100:.1f}%** | — | — | — |",
        "",
        "---",
        "",
        "## 3. Comparison Across All Experimental Paradigms",
        "",
        "| Paradigm / Stage | Architecture | Cross-Validation Mode | Cohort AUROC | Cohort AUPRC | Sens @ 80% Spec | Sens @ 90% Spec | Clinical Verdict |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        "| **Stage 1** | Classical Tabular Baseline | 5x5 Nested CV (All 215) | 0.7330 | 0.3850 | 50.0% | 35.0% | Baseline |",
        "| **Stage 4** | **Clinical Top-15 Tabular Ensemble** | **5x5 Nested CV (All 215)** | **0.7970** | **0.4420** | **62.5%** | **42.5%** | **Gold Standard (Best Cohort Generalizer)** |",
        "| **Stage 27** | TabPFN Tabular Foundation Model | 5x5 Nested CV (All 215) | 0.7840 | 0.4180 | 57.5% | 40.0% | Strong zero-shot tabular prior |",
        "| **Stage 30** | DenseNet201 Unfrozen CNN on Raw Crops | Single Honest Split | 0.7893 | 0.4410 | 50.0% | 37.5% | Vulnerable to paper shortcuts |",
        "| **Stage 32** | Wide 3-Ch Visual Gated-Attention MIL | Single Honest Split | 0.7857 | 0.6784 | 55.0% | 37.5% | Proved line-level gated attention |",
        "| **Stage 33** | Exp 4B Multi-Modal MIL (15M Params) | **Full 5-Fold CV (All 215)** | **0.6877 (Peak: 0.7725)** | **0.3727** | **45.0%** | **35.0%** | Learned visual signal; late overfit |",
        "| **Stage 34** | Exp 5 De-Cheated MIL (6 Regularizers) | **Full 5-Fold CV (All 215)** | **0.5611** | **0.2259** | **22.5%** | **12.5%** | Over-regularization underfitting trap |",
        f"| **Stage 36** | **Exp 6 Dual-Script TrOCR MIL (SWA)** | **Full 5-Fold CV (All 215)** | **{summary['pooled_auroc']:.4f}** | **{summary['pooled_auprc']:.4f}** | **{summary['sens_at_80_spec']*100:.1f}%** | **{summary['sens_at_90_spec']*100:.1f}%** | **Dual-Script Pretrained ViT Benchmark** |",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved empirical whitepaper to %s", report_path)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_exp6_5fold(device)


if __name__ == "__main__":
    main()


"""Full 5-Fold Student-Stratified Cross-Validation for Experiment 4 Multi-Modal MIL.

Evaluates all 215 students out-of-fold (175 controls, 40 dysgraphic positives)
under locked student-stratified fold assignments with 0.0% student leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
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
from torch.utils.data import DataLoader

from baseline.multimodal_features import (
    fit_and_scale_multimodal_features,
    load_raw_multimodal_dataframe,
)
from baseline.train_multimodal_mil import (
    MultiModalStudentBagDataset,
    MultiModalWholePageMILModel,
    collate_multimodal_bags,
    evaluate_multimodal_dataset,
)
from baseline.train_whole_page_mil import (
    FlushingStreamHandler,
    precompute_or_load_wide_3ch_dataset,
)
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


def train_fold(
    fold_idx: int,
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    tensors_dict: Dict[str, np.ndarray],
    math_dict: Dict[str, np.ndarray],
    aux_dict: Dict[str, np.ndarray],
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    lambda_aux: float = 0.3,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    seed: int = 42,
) -> Tuple[MultiModalWholePageMILModel, Dict[str, Any], List[Dict[str, Any]]]:
    """Train Run 4B model on one fold with auxiliary loss."""
    set_seed(seed + fold_idx)

    train_labels = [r["label"] for r in train_records]
    n_pos = max(1.0, float(sum(train_labels)))
    n_neg = max(1.0, float(len(train_labels) - n_pos))
    pos_weight = torch.tensor([n_neg / n_pos], device=device)

    train_dataset = MultiModalStudentBagDataset(train_records, tensors_dict, math_dict, aux_dict, is_train=True)
    test_dataset = MultiModalStudentBagDataset(test_records, tensors_dict, math_dict, aux_dict, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_bags, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_bags, pin_memory=True)

    model = MultiModalWholePageMILModel(pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    aux_criterion = nn.MSELoss()

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

    best_test_auroc = 0.0
    best_test_eval: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

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
                loss = loss_main + (lambda_aux * loss_aux)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(labels)
            n_samples += len(labels)

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_samples)

        # Evaluate on test fold
        test_eval = evaluate_multimodal_dataset(model, test_loader, device, criterion, aux_criterion, lambda_aux=lambda_aux)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "test_loss": test_eval["loss"],
            "test_auroc": test_eval["auroc"],
            "test_auprc": test_eval["auprc"],
            "test_brier": test_eval["brier"],
        })

        if test_eval["auroc"] > best_test_auroc:
            best_test_auroc = test_eval["auroc"]
            best_test_eval = test_eval

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"Fold {fold_idx + 1} | Ep {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Test AUROC: {test_eval['auroc']:.4f} | "
                f"Test AUPRC: {test_eval['auprc']:.4f} | "
                f"(Best: {best_test_auroc:.4f})"
            )

    # Use the evaluation from the final trained model (or best)
    # For fair out-of-fold concatenation without post-hoc early stopping leak, evaluate model at epoch 20
    final_test_eval = evaluate_multimodal_dataset(model, test_loader, device, criterion, aux_criterion, lambda_aux=lambda_aux)
    final_test_eval["best_auroc"] = best_test_auroc

    return model, final_test_eval, history


def run_5fold_cv(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    lambda_aux: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute full 5-fold cross-validation across all 215 students."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Starting 5-Fold Cross-Validation on {device} (Cohort: 215 students) ===")

    # 1. Load data and locked fold mapping
    df_raw_all = load_raw_multimodal_dataframe()
    df_folds = pd.read_csv(FOLDS_CSV)

    # Merge fold assignments: locked repeat_0_fold (0, 1, 2, 3, 4)
    fold_map = dict(zip(df_folds["student_id"], df_folds["repeat_0_fold"]))
    df_raw_all["fold"] = df_raw_all["student_id"].map(fold_map)

    # Preload wide (128, 640, 3) tensors
    tensors_dict = precompute_or_load_wide_3ch_dataset(df_raw_all, target_h=128, target_w=640)

    # Assemble student records
    student_records: List[Dict[str, Any]] = []
    for sid, group in df_raw_all.groupby("student_id"):
        sorted_group = group.sort_values("task_id")
        rec = {
            "student_id": sid,
            "school": group["school"].iloc[0],
            "label": float(group["label"].max()),
            "grade": int(group["grade"].iloc[0]) if "grade" in group else 0,
            "fold": int(group["fold"].iloc[0]),
            "task_uids": sorted_group["task_uid"].tolist(),
            "task_names": sorted_group["task_id"].tolist(),
        }
        student_records.append(rec)

    total_students = len(student_records)
    total_pos = sum(r["label"] for r in student_records)
    total_neg = total_students - total_pos
    logger.info(f"Cohort verified: {total_students} students ({total_pos:.0f} Positives, {total_neg:.0f} Controls)")

    oof_student_ids: List[str] = []
    oof_schools: List[str] = []
    oof_grades: List[int] = []
    oof_labels: List[float] = []
    oof_probs: List[float] = []
    oof_folds: List[int] = []

    per_fold_metrics: List[Dict[str, Any]] = []

    for fold_idx in range(5):
        train_records = [r for r in student_records if r["fold"] != fold_idx]
        test_records = [r for r in student_records if r["fold"] == fold_idx]

        train_pos = sum(r["label"] for r in train_records)
        test_pos = sum(r["label"] for r in test_records)

        logger.info("\n" + "=" * 65)
        logger.info(
            f"FOLD {fold_idx + 1}/5 | Train: {len(train_records)} ({train_pos:.0f} pos, {len(train_records) - train_pos:.0f} neg) | "
            f"Test: {len(test_records)} ({test_pos:.0f} pos, {len(test_records) - test_pos:.0f} neg)"
        )
        logger.info("=" * 65)

        train_uids = [uid for r in train_records for uid in r["task_uids"]]
        test_uids = [uid for r in test_records for uid in r["task_uids"]]

        # Leak-free feature fitting strictly on training students of this fold
        math_dict, aux_dict, _ = fit_and_scale_multimodal_features(df_raw_all, train_uids, test_uids)

        model, test_eval, hist = train_fold(
            fold_idx=fold_idx,
            train_records=train_records,
            test_records=test_records,
            tensors_dict=tensors_dict,
            math_dict=math_dict,
            aux_dict=aux_dict,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_aux=lambda_aux,
            device=device,
            seed=seed,
        )

        fold_sids = test_eval["student_ids"]
        fold_probs = test_eval["y_prob"]
        fold_labels = test_eval["y_true"]

        # Track OOF predictions
        for sid, p, y in zip(fold_sids, fold_probs, fold_labels):
            rec = next(r for r in test_records if r["student_id"] == sid)
            oof_student_ids.append(sid)
            oof_schools.append(rec["school"])
            oof_grades.append(rec["grade"])
            oof_labels.append(y)
            oof_probs.append(p)
            oof_folds.append(fold_idx)

        per_fold_metrics.append({
            "fold": fold_idx + 1,
            "test_auroc": test_eval["auroc"],
            "best_auroc": test_eval["best_auroc"],
            "test_auprc": test_eval["auprc"],
            "test_brier": test_eval["brier"],
            "sens_at_80_spec": calculate_sensitivity_at_specificity(fold_labels, fold_probs, 0.80),
            "sens_at_90_spec": calculate_sensitivity_at_specificity(fold_labels, fold_probs, 0.90),
        })

    # Assemble comprehensive Out-Of-Fold dataframe across all 215 students
    df_oof = pd.DataFrame({
        "student_id": oof_student_ids,
        "school": oof_schools,
        "grade": oof_grades,
        "fold": oof_folds,
        "label": oof_labels,
        "pred_prob": oof_probs,
    })

    oof_csv_path = REPORTS_DIR / "multimodal_mil_5fold_oof_predictions.csv"
    df_oof.to_csv(oof_csv_path, index=False)
    logger.info(f"\nSaved all 215 Out-Of-Fold predictions to {oof_csv_path}")

    y_true_all = np.array(df_oof["label"])
    y_prob_all = np.array(df_oof["pred_prob"])

    pooled_auroc = float(roc_auc_score(y_true_all, y_prob_all))
    pooled_auprc = float(average_precision_score(y_true_all, y_prob_all))
    pooled_brier = float(brier_score_loss(y_true_all, y_prob_all))
    pooled_sens_80 = calculate_sensitivity_at_specificity(y_true_all, y_prob_all, 0.80)
    pooled_sens_90 = calculate_sensitivity_at_specificity(y_true_all, y_prob_all, 0.90)

    mean_fold_auroc = float(np.mean([m["test_auroc"] for m in per_fold_metrics]))
    std_fold_auroc = float(np.std([m["test_auroc"] for m in per_fold_metrics]))
    mean_fold_auprc = float(np.mean([m["test_auprc"] for m in per_fold_metrics]))
    std_fold_auprc = float(np.std([m["test_auprc"] for m in per_fold_metrics]))

    logger.info("\n" + "=" * 70)
    logger.info("FINAL POOLED OUT-OF-FOLD BENCHMARK RESULTS (ALL 215 STUDENTS)")
    logger.info("=" * 70)
    logger.info(f"Cohort Size: {len(df_oof)} students (Positives: {int(y_true_all.sum())}, Controls: {int(len(y_true_all) - y_true_all.sum())})")
    logger.info(f"Pooled OOF AUROC:             {pooled_auroc:.4f}")
    logger.info(f"Pooled OOF AUPRC:             {pooled_auprc:.4f}  (Baseline Prevalence = {y_true_all.mean():.4f})")
    logger.info(f"Pooled OOF Brier Score:       {pooled_brier:.4f}")
    logger.info(f"Sensitivity @ 80% Specificity:{pooled_sens_80:.4f}  ({pooled_sens_80 * 100:.1f}%)")
    logger.info(f"Sensitivity @ 90% Specificity:{pooled_sens_90:.4f}  ({pooled_sens_90 * 100:.1f}%)")
    logger.info(f"Mean Fold AUROC:              {mean_fold_auroc:.4f} ± {std_fold_auroc:.4f}")
    logger.info(f"Mean Fold AUPRC:              {mean_fold_auprc:.4f} ± {std_fold_auprc:.4f}")
    logger.info("=" * 70)

    results_summary = {
        "experiment": "Experiment 4: Full 5-Fold Cross-Validation Multi-Modal MIL",
        "cohort_size": len(df_oof),
        "num_positives": int(y_true_all.sum()),
        "num_controls": int(len(y_true_all) - y_true_all.sum()),
        "pooled_oof_auroc": pooled_auroc,
        "pooled_oof_auprc": pooled_auprc,
        "pooled_oof_brier": pooled_brier,
        "pooled_sens_at_80_spec": pooled_sens_80,
        "pooled_sens_at_90_spec": pooled_sens_90,
        "mean_fold_auroc": mean_fold_auroc,
        "std_fold_auroc": std_fold_auroc,
        "mean_fold_auprc": mean_fold_auprc,
        "std_fold_auprc": std_fold_auprc,
        "per_fold_metrics": per_fold_metrics,
    }

    json_path = REPORTS_DIR / "multimodal_mil_5fold_results.json"
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved full 5-fold metrics summary to {json_path}")

    # Generate Publication-Grade ROC and PR curves
    try:
        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
        prec, rec, _ = precision_recall_curve(y_true_all, y_prob_all)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # ROC Curve
        ax1.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"Pooled OOF ROC (AUROC = {pooled_auroc:.4f})")
        ax1.plot([0, 1], [0, 1], color="gray", ls="--", alpha=0.7)
        ax1.scatter([0.10], [pooled_sens_90], color="#d62728", s=60, zorder=5, label=f"Sens@90%Spec: {pooled_sens_90:.1%}")
        ax1.scatter([0.20], [pooled_sens_80], color="#ff7f0e", s=60, zorder=5, label=f"Sens@80%Spec: {pooled_sens_80:.1%}")
        ax1.set_title("Pooled Out-Of-Fold ROC (N = 215 Students)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
        ax1.set_xlim([-0.02, 1.02])
        ax1.set_ylim([-0.02, 1.02])
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right", fontsize=9)

        # Precision-Recall Curve
        ax2.plot(rec, prec, color="#d62728", lw=2.5, label=f"Pooled OOF PR (AUPRC = {pooled_auprc:.4f})")
        ax2.axhline(y_true_all.mean(), color="gray", ls="--", alpha=0.7, label=f"Prevalence: {y_true_all.mean():.1%}")
        ax2.set_title("Pooled Out-Of-Fold Precision-Recall", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Recall (Sensitivity)", fontsize=10)
        ax2.set_ylabel("Precision (PPV)", fontsize=10)
        ax2.set_xlim([-0.02, 1.02])
        ax2.set_ylim([-0.02, 1.02])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=9)

        plt.suptitle("Experiment 4: Multi-Modal Gated-Attention MIL — 5-Fold Cross-Validation (N=215)", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        plot_path = REPORTS_DIR / "multimodal_mil_5fold_roc_pr_curve.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved 5-fold ROC/PR curve visualization to {plot_path}")
    except Exception as e:
        logger.warning(f"Could not save ROC/PR figure: {e}")

    return results_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: Full 5-Fold Cross-Validation Multi-Modal MIL")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (students per batch)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda-aux", type=float, default=0.3, help="Auxiliary legibility loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_5fold_cv(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_aux=args.lambda_aux,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

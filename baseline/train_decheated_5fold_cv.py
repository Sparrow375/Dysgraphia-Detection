"""Full 5-Fold Cross-Validation for Experiment 5: De-Cheated Multi-Modal Gated-Attention MIL.

Evaluates all 215 students out-of-fold under:
1. 98.8% frozen DenseNet201 + bottleneck dilated sequence convolution (~180k trainable params).
2. Anti-silhouette augmentations: random sub-windowing (128x480), strip cutout, bag line-dropout.
3. Modality dropout (forced independence: 30% drop vision, 20% drop math).
4. Expanded 22 physical math/tool/OCR features.
5. Soft stylus LUPI targets + label smoothing [0.05, 0.95].
6. Leak-free inner-validation early stopping + EMA (decay=0.99) + Platt calibration per fold.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline.decheated_multimodal_mil import (
    DeCheatedMultiModalMILModel,
    DeCheatedStudentBagDataset,
    EMAWeightTracker,
    collate_decheated_bags,
)
from baseline.multimodal_features import (
    SELECTED_MATH_FEATURES_22,
    fit_and_scale_multimodal_features,
    load_raw_multimodal_dataframe,
    load_wacom_stylus_lupi_targets,
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


def evaluate_decheated_dataset(
    model: DeCheatedMultiModalMILModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    aux_criterion: nn.Module,
    lambda_aux: float = 0.3,
) -> Dict[str, Any]:
    """Evaluate De-Cheated MIL model computing metrics."""
    model.eval()
    total_loss = 0.0
    all_logits: List[float] = []
    all_labels: List[float] = []
    all_students: List[str] = []

    with torch.no_grad():
        for batch in loader:
            all_images = batch["all_images"].to(device, non_blocking=True)
            all_math = batch["all_math"].to(device, non_blocking=True)
            all_aux = batch["all_aux"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            split_indices = batch["split_indices"]

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _, aux_preds = model(all_images, all_math, split_indices, force_modality="both")
                loss_main = criterion(logits, labels)
                loss_aux = aux_criterion(aux_preds, all_aux)
                loss = loss_main + (lambda_aux * loss_aux)

            total_loss += loss.item() * len(labels)
            logits_np = logits.cpu().numpy().tolist()
            labels_np = labels.cpu().numpy().tolist()

            all_logits.extend(logits_np)
            all_labels.extend(labels_np)
            all_students.extend(batch["student_ids"])

    avg_loss = total_loss / max(1, len(all_labels))
    y_true = np.array(all_labels)
    logits_arr = np.array(all_logits)
    probs_arr = 1.0 / (1.0 + np.exp(-np.clip(logits_arr, -15.0, 15.0)))

    try:
        auroc = float(roc_auc_score(y_true, probs_arr))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, probs_arr))
    except Exception:
        auprc = float("nan")
    try:
        brier = float(brier_score_loss(y_true, probs_arr))
    except Exception:
        brier = float("nan")

    return {
        "loss": avg_loss,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "y_true": y_true,
        "y_prob": probs_arr,
        "logits": logits_arr,
        "student_ids": all_students,
    }


def train_single_decheated_fold(
    fold_idx: int,
    outer_train_records: List[Dict[str, Any]],
    outer_test_records: List[Dict[str, Any]],
    tensors_dict: Dict[str, np.ndarray],
    df_raw_all: pd.DataFrame,
    lupi_dict: Dict[str, float],
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    lambda_aux: float = 0.3,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train on one outer fold using leak-free inner validation for early stopping & calibration."""
    set_seed(seed + fold_idx)

    # 1. Stratified inner-validation split inside outer_train_records (~15% inner-val)
    df_inner = pd.DataFrame([
        {"student_id": r["student_id"], "label": r["label"]}
        for r in outer_train_records
    ])
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + fold_idx)
    inner_tr_idx, inner_val_idx = next(sgkf.split(df_inner, df_inner["label"], groups=df_inner["student_id"]))

    inner_train_sids = set(df_inner.iloc[inner_tr_idx]["student_id"])
    inner_val_sids = set(df_inner.iloc[inner_val_idx]["student_id"])

    inner_train_records = [r for r in outer_train_records if r["student_id"] in inner_train_sids]
    inner_val_records = [r for r in outer_train_records if r["student_id"] in inner_val_sids]

    logger.info(
        f"Fold {fold_idx + 1} | Inner Train: {len(inner_train_records)} (Pos: {sum(r['label'] for r in inner_train_records):.0f}) | "
        f"Inner Val: {len(inner_val_records)} (Pos: {sum(r['label'] for r in inner_val_records):.0f}) | "
        f"Test: {len(outer_test_records)} (Pos: {sum(r['label'] for r in outer_test_records):.0f})"
    )

    # 2. Scale features strictly on inner training students
    inner_train_uids = [uid for r in inner_train_records for uid in r["task_uids"]]
    inner_val_uids = [uid for r in inner_val_records for uid in r["task_uids"]]
    outer_test_uids = [uid for r in outer_test_records for uid in r["task_uids"]]

    math_dict, aux_dict, _ = fit_and_scale_multimodal_features(
        df_raw_all,
        inner_train_uids,
        inner_val_uids + outer_test_uids,
        feature_cols=SELECTED_MATH_FEATURES_22,
    )

    # 3. Create datasets and dataloaders
    train_dataset = DeCheatedStudentBagDataset(
        inner_train_records, tensors_dict, math_dict, aux_dict, lupi_dict=lupi_dict, is_train=True, sub_w=480
    )
    inner_val_dataset = DeCheatedStudentBagDataset(
        inner_val_records, tensors_dict, math_dict, aux_dict, lupi_dict={}, is_train=False, sub_w=480
    )
    outer_test_dataset = DeCheatedStudentBagDataset(
        outer_test_records, tensors_dict, math_dict, aux_dict, lupi_dict={}, is_train=False, sub_w=480
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_decheated_bags, pin_memory=True)
    val_loader = DataLoader(inner_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_decheated_bags, pin_memory=True)
    test_loader = DataLoader(outer_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_decheated_bags, pin_memory=True)

    # 4. Instantiate model and optimizer
    model = DeCheatedMultiModalMILModel(pretrained=True).to(device)

    # Trainable parameter groups
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    pooler_head_params = (
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
        {"params": pooler_head_params, "lr": lr},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Loss functions: BCE on soft labels + MSE on TrOCR targets
    criterion_soft = nn.BCEWithLogitsLoss()
    criterion_eval = nn.BCEWithLogitsLoss()
    aux_criterion = nn.MSELoss()

    ema = EMAWeightTracker(model, decay=0.99)

    best_val_auroc = 0.0
    best_epoch = 1
    best_weights_state: Optional[Dict[str, torch.Tensor]] = None

    # 5. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            all_images = batch["all_images"].to(device, non_blocking=True)
            all_math = batch["all_math"].to(device, non_blocking=True)
            all_aux = batch["all_aux"].to(device, non_blocking=True)
            soft_labels = batch["soft_labels"].to(device, non_blocking=True)
            split_indices = batch["split_indices"]

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _, aux_preds = model(all_images, all_math, split_indices, force_modality=None)
                loss_main = criterion_soft(logits, soft_labels)
                loss_aux = aux_criterion(aux_preds, all_aux)
                loss = loss_main + (lambda_aux * loss_aux)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update EMA weights
            ema.update(model)

            train_loss += loss.item() * len(soft_labels)
            n_samples += len(soft_labels)

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_samples)

        # Evaluate on inner validation set with EMA weights
        ema.apply_shadow(model)
        val_eval = evaluate_decheated_dataset(model, val_loader, device, criterion_eval, aux_criterion, lambda_aux=lambda_aux)
        ema.restore_backup(model)

        if val_eval["auroc"] > best_val_auroc:
            best_val_auroc = val_eval["auroc"]
            best_epoch = epoch
            best_weights_state = {k: v.cpu().clone() for k, v in ema.shadow.items()}

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"Fold {fold_idx + 1} | Ep {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
                f"Inner Val AUROC: {val_eval['auroc']:.4f} | (Best: {best_val_auroc:.4f} @ Ep {best_epoch})"
            )

    # 6. Load best EMA checkpoint for calibration & test inference
    if best_weights_state is not None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in best_weights_state:
                param.data.copy_(best_weights_state[name].to(device))

    # 7. Fit Platt Scaler on Inner Validation Logits
    val_eval_best = evaluate_decheated_dataset(model, val_loader, device, criterion_eval, aux_criterion, lambda_aux=lambda_aux)
    val_logits_2d = val_eval_best["logits"].reshape(-1, 1)
    val_y = val_eval_best["y_true"]

    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(val_logits_2d, val_y)

    # 8. Predict on held-out outer test fold
    test_eval_raw = evaluate_decheated_dataset(model, test_loader, device, criterion_eval, aux_criterion, lambda_aux=lambda_aux)
    test_logits_2d = test_eval_raw["logits"].reshape(-1, 1)
    test_cal_probs = calibrator.predict_proba(test_logits_2d)[:, 1]

    test_y = test_eval_raw["y_true"]
    cal_auroc = float(roc_auc_score(test_y, test_cal_probs))
    cal_auprc = float(average_precision_score(test_y, test_cal_probs))
    cal_brier = float(brier_score_loss(test_y, test_cal_probs))
    sens_80 = calculate_sensitivity_at_specificity(test_y, test_cal_probs, 0.80)
    sens_90 = calculate_sensitivity_at_specificity(test_y, test_cal_probs, 0.90)

    logger.info(
        f"Fold {fold_idx + 1} Final Test | Cal AUROC: {cal_auroc:.4f} | AUPRC: {cal_auprc:.4f} | "
        f"Sens@80%Spec: {sens_80 * 100:.1f}% | Sens@90%Spec: {sens_90 * 100:.1f}% (Best Ep: {best_epoch})"
    )

    fold_results = {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "best_inner_val_auroc": best_val_auroc,
        "test_auroc": cal_auroc,
        "test_auprc": cal_auprc,
        "test_brier": cal_brier,
        "sens_at_80_spec": sens_80,
        "sens_at_90_spec": sens_90,
        "student_ids": test_eval_raw["student_ids"],
        "y_true": test_y.tolist(),
        "cal_probs": test_cal_probs.tolist(),
        "raw_logits": test_eval_raw["logits"].tolist(),
    }

    return fold_results, val_eval_best


def run_decheated_5fold_cv(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    lambda_aux: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute full 5-fold cross-validation for Experiment 5."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Starting Experiment 5: De-Cheated Multi-Modal MIL on {device} ===")

    # 1. Load data
    df_raw_all = load_raw_multimodal_dataframe()
    df_folds = pd.read_csv(FOLDS_CSV)
    lupi_dict = load_wacom_stylus_lupi_targets()
    logger.info(f"Loaded Wacom stylus continuous targets for {len(lupi_dict)} School A students.")

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
    logger.info(f"Verified cohort: {total_students} students ({total_pos:.0f} Positives, {total_students - total_pos:.0f} Controls)")

    all_oof_records: List[Dict[str, Any]] = []
    per_fold_metrics: List[Dict[str, Any]] = []

    for fold_idx in range(5):
        outer_train_records = [r for r in student_records if r["fold"] != fold_idx]
        outer_test_records = [r for r in student_records if r["fold"] == fold_idx]

        logger.info("\n" + "=" * 70)
        logger.info(f"STARTING OUTER FOLD {fold_idx + 1}/5")
        logger.info("=" * 70)

        fold_res, _ = train_single_decheated_fold(
            fold_idx=fold_idx,
            outer_train_records=outer_train_records,
            outer_test_records=outer_test_records,
            tensors_dict=tensors_dict,
            df_raw_all=df_raw_all,
            lupi_dict=lupi_dict,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_aux=lambda_aux,
            device=device,
            seed=seed,
        )

        per_fold_metrics.append({
            "fold": fold_idx + 1,
            "best_epoch": fold_res["best_epoch"],
            "best_inner_val_auroc": fold_res["best_inner_val_auroc"],
            "test_auroc": fold_res["test_auroc"],
            "test_auprc": fold_res["test_auprc"],
            "test_brier": fold_res["test_brier"],
            "sens_at_80_spec": fold_res["sens_at_80_spec"],
            "sens_at_90_spec": fold_res["sens_at_90_spec"],
        })

        for sid, p, y in zip(fold_res["student_ids"], fold_res["cal_probs"], fold_res["y_true"]):
            rec = next(r for r in outer_test_records if r["student_id"] == sid)
            all_oof_records.append({
                "student_id": sid,
                "school": rec["school"],
                "grade": rec["grade"],
                "fold": fold_idx + 1,
                "label": y,
                "pred_prob": p,
            })

    # Assemble comprehensive Out-Of-Fold dataframe across all 215 students
    df_oof = pd.DataFrame(all_oof_records)
    oof_csv_path = REPORTS_DIR / "decheated_mil_5fold_oof_predictions.csv"
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

    logger.info("\n" + "=" * 75)
    logger.info("FINAL POOLED OUT-OF-FOLD BENCHMARK RESULTS (ALL 215 STUDENTS)")
    logger.info("=" * 75)
    logger.info(f"Cohort Size: {len(df_oof)} students (Positives: {int(y_true_all.sum())}, Controls: {int(len(y_true_all) - y_true_all.sum())})")
    logger.info(f"Pooled OOF AUROC:             {pooled_auroc:.4f}")
    logger.info(f"Pooled OOF AUPRC:             {pooled_auprc:.4f}  (Baseline Prevalence = {y_true_all.mean():.4f})")
    logger.info(f"Pooled OOF Brier Score:       {pooled_brier:.4f}")
    logger.info(f"Sensitivity @ 80% Specificity:{pooled_sens_80:.4f}  ({pooled_sens_80 * 100:.1f}%)")
    logger.info(f"Sensitivity @ 90% Specificity:{pooled_sens_90:.4f}  ({pooled_sens_90 * 100:.1f}%)")
    logger.info(f"Mean Fold AUROC:              {mean_fold_auroc:.4f} ± {std_fold_auroc:.4f}")
    logger.info(f"Mean Fold AUPRC:              {mean_fold_auprc:.4f} ± {std_fold_auprc:.4f}")
    logger.info("=" * 75)

    results_summary = {
        "experiment": "Experiment 5: De-Cheated Multi-Modal Gated-Attention MIL (5-Fold CV)",
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

    json_path = REPORTS_DIR / "decheated_mil_5fold_results.json"
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved full 5-fold metrics summary to {json_path}")

    # Publication-Grade ROC and PR curves
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

        plt.suptitle("Experiment 5: De-Cheated Multi-Modal MIL — 5-Fold Cross-Validation (N=215)", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        plot_path = REPORTS_DIR / "decheated_mil_5fold_roc_pr_curve.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved 5-fold ROC/PR curve visualization to {plot_path}")
    except Exception as e:
        logger.warning(f"Could not save ROC/PR figure: {e}")

    return results_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 5: De-Cheated Multi-Modal Gated-Attention MIL (5-Fold CV)")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (students per batch)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda-aux", type=float, default=0.3, help="Auxiliary legibility loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_decheated_5fold_cv(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_aux=args.lambda_aux,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


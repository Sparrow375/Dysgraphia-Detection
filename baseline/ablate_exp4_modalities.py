"""Phase 1: 4-Way Modality Ablation Audit on Experiment 4 Multi-Modal MIL.

Evaluates the exact standalone contributions across the full unified 215-student cohort:
- Run A (Vision Only): Wide 3-channel MIL without Math or OCR (m_k = 0, lambda_aux = 0).
- Run B (Math + OCR Only): Line-level Gated-Attention MIL with visual branch zeroed out (v_k = 0).
- Run C (Vision + Math, No OCR): Drops TrOCR features and auxiliary loss.
- Run D (Vision + OCR, No Math): Drops geometric math features, retains TrOCR confidence + aux loss.

Evaluated across all 5 locked outer folds (0.0% student leakage; all 215 students OOF).
Accelerated via VRAM tensor resident caching & PyTorch AMP on RTX 5060 Ti GPU.
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

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from baseline.multimodal_features import (
    SELECTED_MATH_FEATURES_16,
    fit_and_scale_multimodal_features,
    load_raw_multimodal_dataframe,
)
from baseline.train_multimodal_mil import (
    MultiModalWholePageMILModel,
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


def precompute_vram_tensors(tensors_dict: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert all 1,145 wide uint8 arrays into pre-normalized float16 tensors resident in VRAM."""
    logger.info("Pre-converting %d wide images into VRAM resident float16 tensors...", len(tensors_dict))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    vram_dict: Dict[str, torch.Tensor] = {}
    for uid, arr in tensors_dict.items():
        t = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255.0
        t = (t - mean) / std
        vram_dict[uid] = t.half()  # (3, 128, 640) in FP16

    logger.info("VRAM resident caching complete (%0.2f MB allocated).", torch.cuda.memory_allocated(device) / (1024**2))
    return vram_dict


class FastVRAMBagDataset(Dataset):
    """Zero-overhead dataset slicing pre-computed VRAM tensors directly on GPU."""

    def __init__(
        self,
        student_records: List[Dict[str, Any]],
        vram_tensors: Dict[str, torch.Tensor],
        math_dict: Dict[str, np.ndarray],
        aux_dict: Dict[str, np.ndarray],
        device: torch.device,
        is_train: bool = True,
    ) -> None:
        self.student_records = student_records
        self.vram_tensors = vram_tensors
        self.math_dict = math_dict
        self.aux_dict = aux_dict
        self.device = device
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.student_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.student_records[idx]
        task_uids = rec["task_uids"]

        bag_images = [self.vram_tensors[uid] for uid in task_uids]
        bag_math = [torch.tensor(self.math_dict[uid], dtype=torch.float32, device=self.device) for uid in task_uids]
        bag_aux = [torch.tensor(self.aux_dict[uid], dtype=torch.float32, device=self.device) for uid in task_uids]

        return {
            "student_id": rec["student_id"],
            "school": rec["school"],
            "label": torch.tensor(rec["label"], dtype=torch.float32, device=self.device),
            "grade": rec["grade"],
            "num_crops": len(task_uids),
            "images": torch.stack(bag_images, dim=0),  # (K, 3, 128, 640) FP16 on GPU
            "math": torch.stack(bag_math, dim=0),      # (K, 16) FP32 on GPU
            "aux": torch.stack(bag_aux, dim=0),        # (K, 2) FP32 on GPU
        }


def collate_fast_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    student_ids = [b["student_id"] for b in batch]
    schools = [b["school"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    grades = [b["grade"] for b in batch]
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
        "num_crops_list": num_crops_list,
        "all_images": all_images,
        "all_math": all_math,
        "all_aux": all_aux,
        "split_indices": split_indices,
    }


class AblatedMultiModalMILModel(MultiModalWholePageMILModel):
    """Subclass of MultiModalWholePageMILModel supporting surgical ablation masking."""

    def __init__(
        self,
        math_dim: int = 16,
        visual_dim: int = 32,
        math_emb_dim: int = 32,
        attn_dim: int = 32,
        pretrained: bool = True,
        dropout: float = 0.35,
        mode: str = "full",
    ) -> None:
        super().__init__(
            math_dim=math_dim,
            visual_dim=visual_dim,
            math_emb_dim=math_emb_dim,
            attn_dim=attn_dim,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.mode = mode

    def forward(
        self,
        all_images: torch.Tensor,
        all_math: torch.Tensor,
        split_indices: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        if self.mode == "run_b":
            num_crops = all_math.size(0)
            v_emb = torch.zeros((num_crops, 32), device=all_math.device, dtype=torch.float32)
            aux_preds = torch.zeros((num_crops, 2), device=all_math.device, dtype=torch.float32)
        else:
            feat_map = self.features(all_images)
            v_emb = self.sequence_pooler(feat_map)
            aux_preds = self.aux_head(v_emb)

        masked_math = all_math.clone()
        if self.mode == "run_a":
            masked_math.zero_()
            m_emb = torch.zeros_like(v_emb)
        elif self.mode == "run_c":
            masked_math[:, 15] = 0.0
            m_emb = self.math_proj(masked_math)
        elif self.mode == "run_d":
            masked_math[:, :15] = 0.0
            m_emb = self.math_proj(masked_math)
        else:
            m_emb = self.math_proj(masked_math)

        x_line = torch.cat([v_emb, m_emb], dim=-1)

        student_logits_list: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        for start_idx, end_idx in split_indices:
            student_lines = x_line[start_idx:end_idx]
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


def evaluate_fast_dataset(
    model: AblatedMultiModalMILModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    model.eval()
    all_preds: List[float] = []
    all_labels: List[int] = []
    student_meta: List[Dict[str, Any]] = []

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            for batch in loader:
                images = batch["all_images"].float()
                math_feats = batch["all_math"]
                splits = batch["split_indices"]
                labels = batch["labels"].cpu().numpy()

                logits, _, _ = model(images, math_feats, splits)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_preds.extend(probs.tolist())
                all_labels.extend(labels.tolist())

                for i, sid in enumerate(batch["student_ids"]):
                    student_meta.append({
                        "student_id": sid,
                        "school": batch["schools"][i],
                        "grade": batch["grades"][i],
                        "label": int(labels[i]),
                        "pred_prob": float(probs[i]),
                        "num_lines": int(batch["num_crops_list"][i]),
                    })

    return np.array(all_labels, dtype=int), np.array(all_preds, dtype=float), student_meta


def train_ablation_fold(
    fold_idx: int,
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    vram_tensors: Dict[str, torch.Tensor],
    math_dict: Dict[str, np.ndarray],
    aux_dict: Dict[str, np.ndarray],
    mode: str = "run_a",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: torch.device = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    set_seed(seed + fold_idx)

    train_labels = [r["label"] for r in train_records]
    n_pos = max(1.0, float(sum(train_labels)))
    n_neg = max(1.0, float(len(train_labels) - n_pos))
    pos_weight = torch.tensor([n_neg / n_pos], device=device)

    train_dataset = FastVRAMBagDataset(train_records, vram_tensors, math_dict, aux_dict, device=device, is_train=True)
    test_dataset = FastVRAMBagDataset(test_records, vram_tensors, math_dict, aux_dict, device=device, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fast_bags)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fast_bags)

    model = AblatedMultiModalMILModel(mode=mode, pretrained=(mode != "run_b")).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    aux_criterion = nn.MSELoss()

    lambda_aux = 0.3 if mode in ("full", "run_d") else 0.0

    if mode == "run_b":
        trainable_params = (
            list(model.math_proj.parameters())
            + list(model.attention_V.parameters())
            + list(model.attention_U.parameters())
            + list(model.attention_w.parameters())
            + list(model.classifier.parameters())
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=lr * 3.0, weight_decay=1e-4)
    else:
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
        ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_test_auroc = 0.0
    best_epoch = 0
    final_meta: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            images = batch["all_images"].float()
            math_feats = batch["all_math"]
            aux_targets = batch["all_aux"]
            splits = batch["split_indices"]
            labels = batch["labels"]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, _, aux_preds = model(images, math_feats, splits)
                loss_student = criterion(logits, labels)
                loss_aux = aux_criterion(aux_preds, aux_targets) if lambda_aux > 0 else torch.tensor(0.0, device=device)
                loss_total = loss_student + lambda_aux * loss_aux

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Fast test evaluation
        y_test, p_test, meta = evaluate_fast_dataset(model, test_loader, device)
        try:
            ep_test_auroc = float(roc_auc_score(y_test, p_test))
        except Exception:
            ep_test_auroc = 0.5

        if ep_test_auroc > best_test_auroc:
            best_test_auroc = ep_test_auroc
            best_epoch = epoch

        if epoch == epochs:
            final_meta = meta

    # Final epoch evaluation
    y_test, p_test, _ = evaluate_fast_dataset(model, test_loader, device)
    final_test_auroc = float(roc_auc_score(y_test, p_test))
    final_test_auprc = float(average_precision_score(y_test, p_test))
    sens_80 = calculate_sensitivity_at_specificity(y_test, p_test, 0.80)
    sens_90 = calculate_sensitivity_at_specificity(y_test, p_test, 0.90)

    fold_metrics = {
        "fold": fold_idx,
        "mode": mode,
        "best_epoch": best_epoch,
        "peak_test_auroc": best_test_auroc,
        "final_test_auroc": final_test_auroc,
        "final_test_auprc": final_test_auprc,
        "sens_80_spec": float(sens_80),
        "sens_90_spec": float(sens_90),
    }

    return fold_metrics, final_meta


def run_ablation_5fold(mode: str, vram_tensors: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    df_raw = load_raw_multimodal_dataframe()
    student_folds_df = pd.read_csv(FOLDS_CSV)
    student_to_fold = dict(zip(student_folds_df["student_id"], student_folds_df["repeat_0_fold"]))

    students_list = sorted(list(df_raw["student_id"].unique()))

    all_fold_metrics: List[Dict[str, Any]] = []
    all_oof_predictions: List[Dict[str, Any]] = []

    mode_names = {
        "run_a": "Run A (Vision Only — No Math/OCR)",
        "run_b": "Run B (Math + OCR Only — No Vision)",
        "run_c": "Run C (Vision + Math — No OCR/Aux)",
        "run_d": "Run D (Vision + OCR — No Math)",
    }
    logger.info("=================================================================")
    logger.info("Executing 5-Fold Cross-Validation: %s", mode_names.get(mode, mode))
    logger.info("=================================================================")
    sys.stdout.flush()

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

        metrics, fold_meta = train_ablation_fold(
            fold_idx=fold_idx,
            train_records=train_records,
            test_records=test_records,
            vram_tensors=vram_tensors,
            math_dict=math_dict,
            aux_dict=aux_dict,
            mode=mode,
            epochs=20,
            batch_size=8,
            device=device,
        )

        for m in fold_meta:
            m["outer_fold"] = fold_idx
            m["mode"] = mode
            all_oof_predictions.append(m)

        all_fold_metrics.append(metrics)
        logger.info(
            "--> Fold %d/5 Complete: Peak AUROC = %.4f (Ep %d) | Final AUROC = %.4f | Final AUPRC = %.4f",
            fold_idx + 1,
            metrics["peak_test_auroc"],
            metrics["best_epoch"],
            metrics["final_test_auroc"],
            metrics["final_test_auprc"],
        )
        sys.stdout.flush()

    elapsed = time.time() - t0

    y_true = np.array([p["label"] for p in all_oof_predictions], dtype=int)
    y_pred = np.array([p["pred_prob"] for p in all_oof_predictions], dtype=float)

    pooled_auroc = float(roc_auc_score(y_true, y_pred))
    pooled_auprc = float(average_precision_score(y_true, y_pred))
    pooled_brier = float(brier_score_loss(y_true, y_pred))
    sens_80 = calculate_sensitivity_at_specificity(y_true, y_pred, 0.80)
    sens_90 = calculate_sensitivity_at_specificity(y_true, y_pred, 0.90)

    mean_fold_auroc = float(np.mean([m["final_test_auroc"] for m in all_fold_metrics]))
    std_fold_auroc = float(np.std([m["final_test_auroc"] for m in all_fold_metrics]))
    mean_peak_auroc = float(np.mean([m["peak_test_auroc"] for m in all_fold_metrics]))
    std_peak_auroc = float(np.std([m["peak_test_auroc"] for m in all_fold_metrics]))

    summary = {
        "mode": mode,
        "name": mode_names.get(mode, mode),
        "elapsed_seconds": round(elapsed, 1),
        "pooled_auroc": round(pooled_auroc, 4),
        "pooled_auprc": round(pooled_auprc, 4),
        "pooled_brier": round(pooled_brier, 4),
        "sens_at_80_spec": round(float(sens_80), 4),
        "sens_at_90_spec": round(float(sens_90), 4),
        "mean_fold_auroc": round(mean_fold_auroc, 4),
        "std_fold_auroc": round(std_fold_auroc, 4),
        "mean_peak_auroc": round(mean_peak_auroc, 4),
        "std_peak_auroc": round(std_peak_auroc, 4),
        "fold_metrics": all_fold_metrics,
        "oof_predictions": all_oof_predictions,
    }

    logger.info(
        "===== %s Complete (%0.1fs): Pooled AUROC = %.4f | Mean Peak = %.4f ± %.4f | Pooled AUPRC = %.4f =====",
        mode_names.get(mode, mode),
        elapsed,
        pooled_auroc,
        mean_peak_auroc,
        std_peak_auroc,
        pooled_auprc,
    )
    sys.stdout.flush()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fast 4-Way Modality Ablation Matrix for Experiment 4.")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "run_a", "run_b", "run_c", "run_d"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running on device: %s", device)

    df_raw = load_raw_multimodal_dataframe()
    tensors_dict = precompute_or_load_wide_3ch_dataset(df_raw, target_h=128, target_w=640)
    vram_tensors = precompute_vram_tensors(tensors_dict, device)

    modes_to_run = ["run_a", "run_b", "run_c", "run_d"] if args.mode == "all" else [args.mode]

    all_results: Dict[str, Any] = {}

    for m in modes_to_run:
        res = run_ablation_5fold(m, vram_tensors, device)
        all_results[m] = res

    # Save JSON summary
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_DIR / "exp4_modality_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved ablation results JSON: %s", out_json)

    # Generate Markdown Whitepaper Report
    out_md = REPORTS_DIR / "exp4_modality_ablation_report.md"
    lines = [
        "# Phase 1: 4-Way Modality Ablation Audit on Experiment 4 Multi-Modal MIL",
        "",
        "**Date**: October 27, 2026  ",
        "**Cohort Standard**: Unified 215-Student Cohort (**40 Positives, 175 Controls**; G3–G8)  ",
        "**Evaluation Standard**: Locked 5-Fold Stratified Cross-Validation by `student_id` (**0.0% Student Leakage, All 215 Students Evaluated Out-Of-Fold**)  ",
        "**Hardware Platform**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM), AMD Ryzen 5 7600X  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Modality Contribution Matrix",
        "",
        "To resolve the exact standalone contributions of Vision, Physical Math, and TrOCR Readability in Experiment 4 before launching Experiment 6, we executed the 4-way modality ablation matrix across all 5 folds:",
        "",
        "| Run / Modality Configuration | Vision Branch | Physical Math (15 Feats) | TrOCR Confidence (Feat 16) | Aux TrOCR Loss | Pooled AUROC | Mean Peak AUROC | Pooled AUPRC | Sens @ 80% Spec | Sens @ 90% Spec |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m in ["run_a", "run_b", "run_c", "run_d"]:
        if m in all_results:
            r = all_results[m]
            vis = "Yes" if m != "run_b" else "No"
            math_feat = "Yes" if m in ("run_b", "run_c") else "No"
            trocr_feat = "Yes" if m in ("run_b", "run_d") else "No"
            aux_l = "Yes" if m == "run_d" else "No"
            lines.append(
                f"| **{r['name']}** | {vis} | {math_feat} | {trocr_feat} | {aux_l} | "
                f"**{r['pooled_auroc']:.4f}** | **{r['mean_peak_auroc']:.4f}** | {r['pooled_auprc']:.4f} | "
                f"{r['sens_at_80_spec']*100:.1f}% | {r['sens_at_90_spec']*100:.1f}% |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Per-Fold Detailed Breakdown",
        "",
    ])

    for m in ["run_a", "run_b", "run_c", "run_d"]:
        if m in all_results:
            r = all_results[m]
            lines.append(f"### {r['name']}")
            lines.append("")
            lines.append("| Fold | Peak AUROC | Best Epoch | Final AUROC | Final AUPRC | Sens @ 80% Spec | Sens @ 90% Spec |")
            lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for fm in r["fold_metrics"]:
                lines.append(
                    f"| **Fold {fm['fold']+1}** | {fm['peak_test_auroc']:.4f} | Ep {fm['best_epoch']} | "
                    f"{fm['final_test_auroc']:.4f} | {fm['final_test_auprc']:.4f} | "
                    f"{fm['sens_80_spec']*100:.1f}% | {fm['sens_90_spec']*100:.1f}% |"
                )
            lines.append(
                f"| **Mean ± Std** | **{r['mean_peak_auroc']:.4f} ± {r['std_peak_auroc']:.4f}** | — | "
                f"**{r['mean_fold_auroc']:.4f} ± {r['std_fold_auroc']:.4f}** | **{r['pooled_auprc']:.4f}** | "
                f"**{r['sens_at_80_spec']*100:.1f}%** | **{r['sens_at_90_spec']*100:.1f}%** |"
            )
            lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Key Scientific Takeaways & Foundation for Experiment 6",
        "",
        "1. **Vision vs. Tabular Independence**: Comparing Run A (Vision Only) vs. Run B (Math+OCR Only) establishes the visual representation's inherent ceiling.",
        "2. **The TrOCR Readability Delta**: Comparing Run C (Vision + Math, No OCR) vs. Run 4B demonstrates whether TrOCR confidence provided an incremental boost or regularizing anchor.",
        "3. **Dual-Script TrOCR Motivation**: Since ImageNet DenseNet201 filters were never trained on handwriting strokes, substituting native Hindi/English TrOCR Vision Encoders (Phase 2) replaces dog/car primitives with true stroke geometry.",
    ])

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved ablation report Markdown: %s", out_md)


if __name__ == "__main__":
    main()

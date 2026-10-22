"""Strategy A: LoRA-Fine-Tuned TrOCR Perplexity & Predictive Entropy Engine.

Clinical Concept:
-----------------
Off-the-shelf TrOCR models were trained on adult handwriting corpora (e.g. IAM).
Indian 3rd-8th grade school children's handwriting on ruled paper appears strange to them.
To overcome this without overfitting on our 215-student cohort (40 dysgraphic, 175 control),
we fine-tune TrOCR with Low-Rank Adaptation (LoRA) strictly on the healthy control students
to predict the actual prompt text they were assigned to write (~35,000 characters of strong supervision).
The vision encoder and base transformer decoder remain 99.6% frozen.

At screening time, a student's handwriting sentences are evaluated against the target prompt:
- Normal/Control handwriting produces high target token confidence, low cross-entropy (NLL),
  and tight predictive Shannon entropy.
- Dysgraphic handwriting (tremor, malformed letter loops, stroke reversals, erratic sizing)
  violates the learned control motor manifold, causing teacher-forced Perplexity (NLL) and
  token uncertainty (Shannon entropy) to spike dramatically.

Cohort & Clinical Standard:
---------------------------
- Unified model evaluated across the complete 215-student cohort (School A + School B).
- Evaluated strictly using the locked 5-repeat x 5-fold Stratified Group Cross-Validation splits.
- Zero per-grade weight tuning or single-grade slices.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"


@dataclass
class SentenceScoringResult:
    """Biomarkers derived from teacher-forced TrOCR token distribution."""
    task_uid: str
    student_id: str
    school: str
    grade: int
    label: int
    task_id: str
    script: str
    forced_nll_mean: float
    forced_nll_max: float
    perplexity: float
    char_entropy_mean: float
    char_entropy_max: float
    token_conf_mean: float
    token_conf_min: float
    low_conf_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_english_lora_trocr(
    model_name: str = "microsoft/trocr-small-handwritten",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """Initialize English TrOCR with LoRA adapter on decoder attention projections."""
    if get_peft_model is None:
        raise ImportError("peft library is required. Install via `pip install peft`.")

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = (
        processor.tokenizer.cls_token_id
        if processor.tokenizer.cls_token_id is not None
        else processor.tokenizer.bos_token_id
    )

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.decoder = get_peft_model(model.decoder, lora_config)
    model.to(device)
    return model, processor


def get_hindi_lora_trocr(
    model_name: str = "paudelanil/trocr-devanagari-2",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    device: str = "cpu",
) -> Tuple[Any, Any, Any]:
    """Initialize Hindi Devanagari TrOCR with LoRA adapter on RoBERTa decoder."""
    if get_peft_model is None:
        raise ImportError("peft library is required. Install via `pip install peft`.")

    image_processor = ViTImageProcessor(
        do_normalize=True,
        do_rescale=True,
        do_resize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        size={"height": 224, "width": 224},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    if start_id is None:
        start_id = getattr(model.config.decoder, "bos_token_id", 0)

    model.config.decoder_start_token_id = start_id
    model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.decoder = get_peft_model(model.decoder, lora_config)
    model.to(device)
    return model, tokenizer, image_processor


def score_sentence_crop(
    model: Any,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute teacher-forced NLL, perplexity, and predictive entropy on a sentence crop."""
    pixel_values = pixel_values.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, labels=labels)
        logits = outputs.logits

    # Shift logits so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    # Mask valid tokens
    valid_mask = (shift_labels != pad_token_id) & (shift_labels != -100)
    if valid_mask.sum() == 0:
        return {
            "forced_nll_mean": 15.0,
            "forced_nll_max": 20.0,
            "perplexity": 10000.0,
            "char_entropy_mean": 5.0,
            "char_entropy_max": 8.0,
            "token_conf_mean": 0.0,
            "token_conf_min": 0.0,
            "low_conf_ratio": 1.0,
        }

    valid_losses = token_losses[valid_mask]
    nll_mean = float(valid_losses.mean().item())
    nll_max = float(valid_losses.max().item())
    perplexity = float(torch.exp(valid_losses.mean()).item())

    # Posterior probabilities and Shannon entropy
    probs = F.softmax(shift_logits, dim=-1)
    # Focus on top 50 logits to avoid vocabulary noise
    topk_probs, _ = torch.topk(probs, k=min(50, probs.size(-1)), dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    entropy = -(topk_probs * torch.log2(torch.clamp(topk_probs, min=1e-9))).sum(dim=-1)

    valid_entropy = entropy[valid_mask]
    entropy_mean = float(valid_entropy.mean().item())
    entropy_max = float(valid_entropy.max().item())

    # Target token probabilities
    target_confs = torch.gather(probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    valid_confs = target_confs[valid_mask]
    conf_mean = float(valid_confs.mean().item())
    conf_min = float(valid_confs.min().item())
    low_conf_ratio = float((valid_confs < 0.40).float().mean().item())

    return {
        "forced_nll_mean": nll_mean,
        "forced_nll_max": nll_max,
        "perplexity": min(perplexity, 50000.0),
        "char_entropy_mean": entropy_mean,
        "char_entropy_max": entropy_max,
        "token_conf_mean": conf_mean,
        "token_conf_min": conf_min,
        "low_conf_ratio": low_conf_ratio,
    }


def aggregate_student_biomarkers(sentence_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate sentence-level perplexity and entropy into student-level feature vectors."""
    students: Dict[str, List[Dict[str, Any]]] = {}
    for r in sentence_results:
        sid = str(r["student_id"])
        if sid not in students:
            students[sid] = []
        students[sid].append(r)

    student_features: Dict[str, Dict[str, Any]] = {}
    for sid, items in students.items():
        label = int(items[0]["label"])
        school = items[0]["school"]
        grade = int(items[0]["grade"])

        nll_means = [float(x["forced_nll_mean"]) for x in items]
        nll_maxes = [float(x["forced_nll_max"]) for x in items]
        perplexities = [float(x["perplexity"]) for x in items]
        entropies = [float(x["char_entropy_mean"]) for x in items]
        entropy_maxes = [float(x["char_entropy_max"]) for x in items]
        confs = [float(x["token_conf_mean"]) for x in items]
        conf_mins = [float(x["token_conf_min"]) for x in items]
        low_confs = [float(x["low_conf_ratio"]) for x in items]

        # Script breakdowns
        eng_items = [x for x in items if x.get("script") == "latin"]
        hin_items = [x for x in items if x.get("script") == "devanagari"]

        eng_nll = float(np.mean([x["forced_nll_mean"] for x in eng_items])) if eng_items else float(np.mean(nll_means))
        hin_nll = float(np.mean([x["forced_nll_mean"] for x in hin_items])) if hin_items else float(np.mean(nll_means))

        student_features[sid] = {
            "student_id": sid,
            "school": school,
            "grade": grade,
            "label": label,
            "n_sentences": len(items),
            # Primary Biomarkers
            "trocr_nll_mean": float(np.mean(nll_means)),
            "trocr_nll_max": float(np.max(nll_maxes)),
            "trocr_perplexity_mean": float(np.mean(perplexities)),
            "trocr_entropy_mean": float(np.mean(entropies)),
            "trocr_entropy_max": float(np.max(entropy_maxes)),
            "trocr_conf_mean": float(np.mean(confs)),
            "trocr_conf_min": float(np.min(conf_mins)),
            "trocr_low_conf_ratio": float(np.mean(low_confs)),
            # Cognitive & Script Stress Deltas
            "trocr_script_nll_divergence": abs(eng_nll - hin_nll),
        }

    return student_features


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute AUROC, AUPRC, Brier Loss, and Sensitivities at 80% and 90% Specificity."""
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Calculate Sens @ 80% and 90% Spec
    thresholds = np.linspace(0.0, 1.0, 1001)
    sens_at_80_spec = 0.0
    sens_at_90_spec = 0.0

    neg_mask = y_true == 0
    pos_mask = y_true == 1
    total_pos = pos_mask.sum()
    total_neg = neg_mask.sum()

    if total_pos > 0 and total_neg > 0:
        for th in thresholds:
            preds = (y_prob >= th).astype(int)
            spec = float((preds[neg_mask] == 0).sum() / total_neg)
            sens = float((preds[pos_mask] == 1).sum() / total_pos)
            if spec >= 0.80 and sens > sens_at_80_spec:
                sens_at_80_spec = sens
            if spec >= 0.90 and sens > sens_at_90_spec:
                sens_at_90_spec = sens

    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "sens_at_80_spec": sens_at_80_spec,
        "sens_at_90_spec": sens_at_90_spec,
    }

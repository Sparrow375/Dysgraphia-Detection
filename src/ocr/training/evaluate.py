"""
Evaluation & Benchmarking Suite for Context-Aware OCR.

Computes:
  1. Character Error Rate (CER) via edit distance
  2. Word Error Rate (WER)
  3. Context Rescue Rate (% of words corrected by contextual LM)
  4. Expected Calibration Error (ECE) for confidence assessment
  5. Ablation comparisons: (Visual-only vs +Stroke vs +LM vs Full Pipeline)
"""

from __future__ import annotations

import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import editdistance
    HAS_EDITDIST = True
except ImportError:
    HAS_EDITDIST = False

from src.ocr.utils import TranscriptionResult, WordHypothesis
from src.ocr.pipeline import ContextAwareOCRPipeline


def compute_cer(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute Character Error Rate (CER) across pairs."""
    total_dist = 0
    total_chars = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_c = pred.strip()
        gt_c = gt.strip()
        total_chars += len(gt_c)

        if HAS_EDITDIST:
            total_dist += editdistance.eval(pred_c, gt_c)
        else:
            # Fallback simple edit distance
            total_dist += _simple_levenshtein(pred_c, gt_c)

    return total_dist / max(total_chars, 1)


def compute_wer(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute Word Error Rate (WER) across pairs."""
    total_dist = 0
    total_words = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_words = pred.strip().split()
        gt_words = gt.strip().split()
        total_words += len(gt_words)

        if HAS_EDITDIST:
            total_dist += editdistance.eval(pred_words, gt_words)
        else:
            total_dist += _simple_levenshtein(pred_words, gt_words)

    return total_dist / max(total_words, 1)


def compute_calibration_ece(
    confidences: List[float],
    accuracies: List[bool],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for confidence calibration.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = [
            (c, a) for c, a in zip(confidences, accuracies)
            if bin_lower <= c < bin_upper or (i == n_bins - 1 and c == bin_upper)
        ]

        if in_bin:
            bin_conf = np.mean([item[0] for item in in_bin])
            bin_acc = np.mean([1.0 if item[1] else 0.0 for item in in_bin])
            bin_weight = len(in_bin) / total_samples
            ece += bin_weight * abs(bin_conf - bin_acc)

    return float(ece)


def _simple_levenshtein(seq1, seq2) -> int:
    """Basic Levenshtein dynamic programming fallback."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return int(matrix[size_x - 1, size_y - 1])


def run_benchmark(
    pipeline: ContextAwareOCRPipeline,
    test_cases: List[Tuple[str, str]],  # (image_path, ground_truth_text)
) -> Dict[str, float]:
    """
    Run full benchmark across test images.
    """
    predictions: List[str] = []
    ground_truths: List[str] = []
    confidences: List[float] = []
    word_accuracies: List[bool] = []
    total_words = 0
    rescued_words = 0

    for img_path, gt in test_cases:
        try:
            res: TranscriptionResult = pipeline.transcribe(img_path)
            predictions.append(res.full_text)
            ground_truths.append(gt)

            for line in res.lines:
                for w in line.words:
                    total_words += 1
                    confidences.append(w.confidence)
                    # Check if rescued by LM
                    if w.metadata.get("context_rescued", False):
                        rescued_words += 1

        except Exception as e:
            predictions.append("")
            ground_truths.append(gt)

    cer = compute_cer(predictions, ground_truths)
    wer = compute_wer(predictions, ground_truths)
    rescue_rate = rescued_words / max(total_words, 1)

    return {
        "CER": round(cer, 4),
        "WER": round(wer, 4),
        "ContextRescueRate": round(rescue_rate, 4),
        "TotalEvaluatedWords": total_words,
    }

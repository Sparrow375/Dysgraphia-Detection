"""Unit and Regression Tests for Strategy A: LoRA-Adapted TrOCR Perplexity & Entropy."""

import pytest
import numpy as np
import torch
import csv
from pathlib import Path

from models.trocr_lora_perplexity import (
    score_sentence_crop,
    aggregate_student_biomarkers,
    compute_clinical_metrics,
    INDEX_CSV,
)


def test_sentences_index_cohort_integrity():
    """Verify that sentences_index.csv contains all 215 students and locked folds."""
    assert INDEX_CSV.exists(), f"Missing index file: {INDEX_CSV}"
    with open(INDEX_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 1100, f"Expected ~1,145 sentence crops, found {len(reader)}"
    students = set(r["student_id"] for r in reader)
    assert len(students) == 215, f"Expected exactly 215 unique students, found {len(students)}"

    # Verify locked 5x5 fold assignments are present and valid
    for r in reader:
        for rep in range(5):
            fold_col = f"repeat_{rep}_fold"
            assert fold_col in r, f"Missing fold column: {fold_col}"
            f_val = int(r[fold_col])
            assert 0 <= f_val <= 4, f"Invalid fold value: {f_val} in {fold_col}"


def test_score_sentence_crop_synthetic():
    """Verify teacher-forced scoring function on synthetic tensors."""
    vocab_size = 50
    seq_len = 10
    batch_size = 1

    # Mock model
    class MockOutput:
        def __init__(self, logits):
            self.logits = logits

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, pixel_values, labels):
            # Produce deterministic logits
            logits = torch.zeros((batch_size, seq_len, vocab_size))
            # Put high probability on target labels
            for t in range(seq_len - 1):
                target_tok = labels[0, t + 1]
                logits[0, t, target_tok] = 10.0  # high confidence
            return MockOutput(logits)

    mock_model = MockModel()
    dummy_pixels = torch.zeros((1, 3, 64, 64))
    dummy_labels = torch.tensor([[0, 5, 12, 19, 24, 30, 42, 1, 1, 1]])
    pad_id = 1

    res = score_sentence_crop(
        mock_model,
        dummy_pixels,
        dummy_labels,
        pad_token_id=pad_id,
        device="cpu",
    )

    assert "forced_nll_mean" in res
    assert "perplexity" in res
    assert "char_entropy_mean" in res
    assert "token_conf_mean" in res
    assert "token_conf_min" in res
    assert "low_conf_ratio" in res

    # High confidence on target labels should yield low NLL and high confidence
    assert res["forced_nll_mean"] < 1.0
    assert res["token_conf_mean"] > 0.90
    assert res["low_conf_ratio"] == 0.0


def test_aggregate_student_biomarkers():
    """Verify student-level biomarker aggregation across tasks."""
    mock_sentences = [
        {
            "task_uid": "S1_copy_eng",
            "student_id": "STU_001",
            "school": "school_a",
            "grade": 4,
            "label": 1,
            "task_id": "copy_english",
            "script": "latin",
            "forced_nll_mean": 4.5,
            "forced_nll_max": 6.2,
            "perplexity": 90.0,
            "char_entropy_mean": 3.2,
            "char_entropy_max": 4.5,
            "token_conf_mean": 0.25,
            "token_conf_min": 0.05,
            "low_conf_ratio": 0.60,
        },
        {
            "task_uid": "S1_copy_hin",
            "student_id": "STU_001",
            "school": "school_a",
            "grade": 4,
            "label": 1,
            "task_id": "copy_hindi",
            "script": "devanagari",
            "forced_nll_mean": 5.1,
            "forced_nll_max": 7.0,
            "perplexity": 160.0,
            "char_entropy_mean": 3.8,
            "char_entropy_max": 5.0,
            "token_conf_mean": 0.18,
            "token_conf_min": 0.02,
            "low_conf_ratio": 0.75,
        },
    ]

    stu_map = aggregate_student_biomarkers(mock_sentences)
    assert "STU_001" in stu_map
    s = stu_map["STU_001"]
    assert s["student_id"] == "STU_001"
    assert s["label"] == 1
    assert s["n_sentences"] == 2
    assert s["trocr_nll_mean"] == pytest.approx((4.5 + 5.1) / 2.0)
    assert s["trocr_nll_max"] == 7.0
    assert s["trocr_low_conf_ratio"] == pytest.approx((0.60 + 0.75) / 2.0)
    assert s["trocr_script_nll_divergence"] == pytest.approx(abs(4.5 - 5.1))


def test_clinical_metrics_computation():
    """Verify AUROC, AUPRC, and Sens@Spec calculations."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.7, 0.85, 0.65, 0.9, 0.8])

    metrics = compute_clinical_metrics(y_true, y_prob)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["sens_at_80_spec"] == 1.0
    assert metrics["sens_at_90_spec"] == 1.0
    assert metrics["brier"] < 0.10

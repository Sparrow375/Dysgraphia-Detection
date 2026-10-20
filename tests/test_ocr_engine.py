"""Unit tests for Dual-Engine TrOCR handwritten text recognition pipeline."""

import cv2
import numpy as np
import pytest

from features.ocr_engine import (
    DevanagariTrOCREngine,
    DualEngineOCRPipeline,
    EnglishTrOCREngine,
    OCRLineResult,
    OCRTaskResult,
    get_devanagari_engine,
    get_english_engine,
    segment_lines_projection,
)


def test_ocr_line_and_task_data_structures():
    line = OCRLineResult(text="hello world", confidence=0.88, token_confs=[0.9, 0.86], bbox=[0, 0, 100, 30])
    assert line.text == "hello world"
    assert line.confidence == 0.88
    assert len(line.token_confs) == 2

    task_res = OCRTaskResult(
        text="hello world",
        tokens=["hello", "world"],
        token_confidences=[0.9, 0.86],
        mean_confidence=0.88,
        min_confidence=0.86,
        std_confidence=0.02,
        low_conf_ratio=0.0,
        char_count=10,
        word_count=2,
        line_count=1,
        lines=[{"text": "hello world", "confidence": 0.88, "bbox": [0, 0, 100, 30]}],
    )
    d = task_res.to_dict()
    assert d["mean_confidence"] == 0.88
    assert d["word_count"] == 2
    assert len(d["lines"]) == 1


def test_segment_lines_projection():
    # Synthetic image with 2 horizontal text bars
    canvas = np.full((160, 400), 255, dtype=np.uint8)
    # Line 1: y in [20, 60]
    canvas[20:60, 50:350] = 0
    # Line 2: y in [90, 130]
    canvas[90:130, 50:350] = 0

    lines = segment_lines_projection(canvas, min_line_height=20)
    assert len(lines) == 2
    assert lines[0][0] <= 25 and lines[0][1] >= 55
    assert lines[1][0] <= 95 and lines[1][1] >= 125


def test_english_trocr_inference():
    engine = get_english_engine()
    assert isinstance(engine, EnglishTrOCREngine)

    # Render clean handwritten-style line
    img = np.full((100, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "cricket in the park", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    res = engine.transcribe_line(img)
    assert isinstance(res, OCRLineResult)
    assert len(res.text) > 0
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.token_confs) > 0


def test_dual_pipeline_caching():
    pipeline = DualEngineOCRPipeline()
    dummy_key = "test_student_task_mock"

    sentence_json = {
        "student_id": "test_student",
        "task_id": "task_mock",
        "task_name": "copy_english",
        "script": "latin",
        "crop_bbox": [0, 0, 400, 100],
    }
    dummy_crop = np.full((100, 400, 3), 255, dtype=np.uint8)

    # First call with mock cached entry to test cache hit
    pipeline.cache[dummy_key] = {
        "text": "mocked cached sentence",
        "tokens": ["mocked", "cached", "sentence"],
        "token_confidences": [0.95, 0.92, 0.90],
        "mean_confidence": 0.923,
        "min_confidence": 0.90,
        "std_confidence": 0.02,
        "low_conf_ratio": 0.0,
        "char_count": 20,
        "word_count": 3,
        "line_count": 1,
        "lines": [],
    }

    res = pipeline.transcribe_task(dummy_crop, sentence_json, use_cache=True)
    assert res.text == "mocked cached sentence"
    assert res.mean_confidence == 0.923
    assert res.word_count == 3

    # Clean up test entry
    del pipeline.cache[dummy_key]


def test_extract_prompt_free_trocr_features():
    from features.ocr_engine import extract_prompt_free_trocr_features

    task_res = OCRTaskResult(
        text="the quick brown fox",
        tokens=["the", "quick", "brown", "fox"],
        token_confidences=[0.95, 0.90, 0.85, 0.40, 0.35],
        mean_confidence=0.69,
        min_confidence=0.35,
        std_confidence=0.28,
        low_conf_ratio=0.4,
        char_count=16,
        word_count=4,
        line_count=1,
    )
    feats = extract_prompt_free_trocr_features(task_res, components_per_unit_width=4.5, image_width=800.0)

    assert 0.0 <= feats["trocr_conf_mean"] <= 1.0
    assert 0.0 <= feats["trocr_conf_p10"] <= 1.0
    assert feats["trocr_char_entropy_mean"] > 0.0
    assert 0.0 <= feats["trocr_top2_margin_mean"] <= 1.0
    assert feats["ink_to_char_density_ratio"] > 0.0


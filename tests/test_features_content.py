"""Unit tests for content, OCR legibility, and linguistic error features."""

import numpy as np
import pytest

from features.content import (
    clean_text,
    compute_content_features,
    levenshtein_distance,
    word_levenshtein_distance,
)


def test_levenshtein_distance():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("mother", "mother") == 0
    assert levenshtein_distance("mother", "moth") == 2


def test_word_levenshtein_distance():
    tokens1 = ["on", "sundays", "we", "play"]
    tokens2 = ["on", "sundays", "we", "play"]
    assert word_levenshtein_distance(tokens1, tokens2) == 0

    tokens3 = ["on", "sundays", "play"]
    assert word_levenshtein_distance(tokens1, tokens3) == 1

    tokens4 = ["in", "mondays", "we", "eat"]
    assert word_levenshtein_distance(tokens1, tokens4) == 3


def test_clean_text():
    raw = "On Sundays, we play cricket!  123"
    cleaned = clean_text(raw)
    assert cleaned == "on sundays we play cricket 123"

    hindi_raw = "हमारे स्कूल मे एक सूंदर बगीचा है।"
    hindi_cleaned = clean_text(hindi_raw)
    assert "हमारे" in hindi_cleaned
    assert "बगीचा" in hindi_cleaned


def test_compute_content_features_empty():
    feats = compute_content_features({"task_name": "copy_english"}, None)
    assert np.isnan(feats["ocr_confidence_mean"])
    assert feats["ocr_words_recognized"] == 0.0
    assert np.isnan(feats["ocr_cer"])


def test_compute_content_features_synthetic_image():
    # Create white canvas with black text using OpenCV
    import cv2

    img = np.full((120, 800, 3), 255, dtype=np.uint8)
    cv2.putText(
        img,
        "on sundays we play cricket",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    sentence_json = {
        "task_name": "copy_english",
        "script": "latin",
        "crop_bbox": [0, 0, 800, 120],
    }

    feats = compute_content_features(sentence_json, img)
    # Check that keys exist
    expected_keys = [
        "ocr_confidence_mean",
        "ocr_confidence_std",
        "ocr_low_conf_ratio",
        "ocr_words_recognized",
        "ocr_word_completion_ratio",
        "ocr_char_count",
        "ocr_char_completion_ratio",
        "ocr_omission_ratio",
        "ocr_cer",
        "ocr_wer",
    ]
    for k in expected_keys:
        assert k in feats

    # OCR should recognize at least some words with confidence > 0
    assert feats["ocr_words_recognized"] > 0
    assert feats["ocr_confidence_mean"] > 0.0
    assert not np.isnan(feats["ocr_cer"])

    # Verify letter-level analytics features exist and are populated
    letter_keys = [
        "char_omission_count",
        "char_omission_rate",
        "word_omission_count",
        "dropped_endings_count",
        "confusable_swaps_count",
        "midword_caps_count",
        "early_line_error_rate",
        "late_line_error_rate",
        "fatigue_error_ratio",
        "conf_trajectory_slope",
        "conf_consistency_cv",
    ]
    for lk in letter_keys:
        assert lk in feats
        assert not np.isnan(feats[lk])

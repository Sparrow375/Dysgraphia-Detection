"""Unit tests for Workstream A Phase 1 (Preprocessing, Rules, Segmentation, Skeleton)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.preprocess import preprocess_page
from pipeline.rules import process_ruled_lines
from pipeline.segment import extract_lines_and_segment, group_lines_into_sentence_blocks
from pipeline.skeleton import process_sentence_skeleton


@pytest.fixture(scope="module")
def sample_image():
    img_path = Path("data/school_a/raw/grade 4/future_gen_4/DocScanner 06-Oct-2026 10-11 AM_1.jpeg")
    assert img_path.exists(), f"Sample test image not found at {img_path}"
    img = cv2.imread(str(img_path))
    assert img is not None
    return img


def test_preprocess_page_shape_and_layers(sample_image):
    res = preprocess_page(sample_image, target_width=2000)
    assert res["grayscale_norm"].shape[1] == 2000, "Warped width must be exactly 2000px"
    assert res["grayscale_norm"].shape[0] > 2000, "Portrait height must exceed width"
    assert res["binary_ink"].shape == res["grayscale_norm"].shape

    # Check ink density is within normal range (3% - 15%)
    ink_ratio = (res["binary_ink"].sum() / res["binary_ink"].size) * 100
    assert 2.0 <= ink_ratio <= 18.0, f"Unexpected ink pixel percentage: {ink_ratio:.2f}%"


def test_rules_detection_and_deskew(sample_image):
    prep = preprocess_page(sample_image, target_width=2000)
    rule_res = process_ruled_lines(prep["binary_ink"], prep["grayscale_norm"])

    assert len(rule_res["ruled_lines"]) >= 15, "Expected at least 15 ruled lines detected"
    r = rule_res["median_spacing_r"]
    assert 50.0 <= r <= 150.0, f"Ruled line spacing r={r} out of expected range [50, 150]"

    # Check rule removal retained the core handwriting ink
    ink_clean_sum = rule_res["ink_clean"].sum()
    ink_rules_sum = rule_res["ink_with_rules"].sum()
    assert ink_clean_sum > 0.35 * ink_rules_sum, "Rule removal erased too much handwriting ink"
    assert rule_res["rule_mask"].sum() > 10000, "Rule mask must capture printed rule lines"


def test_line_and_sentence_segmentation(sample_image):
    prep = preprocess_page(sample_image, target_width=2000)
    rule_res = process_ruled_lines(prep["binary_ink"], prep["grayscale_norm"])

    lines = extract_lines_and_segment(
        rule_res["ink_clean"],
        rule_res["ruled_lines"],
        rule_res["median_spacing_r"],
    )
    assert len(lines) >= 6, f"Expected at least 6 physical handwritten lines, got {len(lines)}"

    sentences = group_lines_into_sentence_blocks(lines, prep["grayscale_norm"].shape)
    assert len(sentences) >= 1, "Expected at least 1 sentence block"

    for s in sentences:
        assert s["word_count"] >= 0
        bx, by, bw, bh = s["crop_bbox"]
        assert bw > 50 and bh > 20, "Sentence crop bbox must have valid dimensions"


def test_skeleton_and_graph(sample_image):
    prep = preprocess_page(sample_image, target_width=2000)
    rule_res = process_ruled_lines(prep["binary_ink"], prep["grayscale_norm"])

    # Test on a small crop
    crop = rule_res["ink_clean"][400:600, 200:1000]
    skel_res = process_sentence_skeleton(crop, x_height_h=60.0)

    assert "skeleton_mask" in skel_res
    graph = skel_res["graph"]
    assert graph["num_nodes"] > 0
    assert graph["junction_count"] >= 0
    assert graph["endpoint_count"] >= 0
    assert graph["total_path_length"] > 0


def test_processed_output_directory_integrity():
    student_dir = Path("data/processed/school_a/G3_A_Roll01")
    if not student_dir.exists():
        pytest.skip("G3_A_Roll01 processed directory not generated yet")

    assert (student_dir / "page_normalized.png").exists()
    assert (student_dir / "overlay_debug.png").exists()

    json_files = list(student_dir.glob("sentence_*.json"))
    assert len(json_files) >= 1, "Expected at least 1 sentence JSON file"

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "student_id" in data
        assert "task_id" in data
        assert "physical_lines" in data
        assert "skeleton_graph" in data
        assert "qa_flags" in data

        png_path = student_dir / data["crop_filename"]
        assert png_path.exists(), f"Referenced sentence crop image missing: {png_path}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

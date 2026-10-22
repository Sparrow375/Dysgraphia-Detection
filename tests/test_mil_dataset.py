"""Unit tests for Word Crops Dataset packaging and integrity."""

import csv
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
WORD_CROPS_DIR = DATA_DIR / "word_crops_dataset"
ZIP_PATH = DATA_DIR / "word_crops_dataset.zip"


def test_packaged_dataset_files_exist():
    """Verify packaged zip archive and metadata files exist on disk."""
    assert ZIP_PATH.exists(), f"Word crops ZIP archive missing: {ZIP_PATH}"
    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    assert zip_size_mb > 20.0, f"Zip archive too small: {zip_size_mb:.2f} MB"

    index_csv = WORD_CROPS_DIR / "words_index.csv"
    assert index_csv.exists(), f"Index CSV missing: {index_csv}"

    meta_json = WORD_CROPS_DIR / "dataset_meta.json"
    assert meta_json.exists(), f"Metadata JSON missing: {meta_json}"


def test_dataset_cohort_coverage():
    """Verify all 215 students and 16,469 word crops are correctly indexed."""
    meta_json = WORD_CROPS_DIR / "dataset_meta.json"
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_students"] == 215
    assert meta["students_with_crops"] == 215
    assert meta["total_word_crops"] == 16469
    assert meta["positive_students"] == 40
    assert meta["control_students"] == 175
    assert meta["latin_words"] > 7000
    assert meta["devanagari_words"] > 8000


def test_words_index_integrity():
    """Verify column format, school balance, and locked fold assignments."""
    index_csv = WORD_CROPS_DIR / "words_index.csv"
    students_seen = set()
    schools_seen = set()
    scripts_seen = set()

    with open(index_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row_count = 0
        for row in reader:
            row_count += 1
            students_seen.add(row["student_id"])
            schools_seen.add(row["school"])
            scripts_seen.add(row["script"])

            # Check fold assignments are valid integers 0-4
            for r in range(5):
                fold_val = int(row[f"repeat_{r}_fold"])
                assert fold_val in {0, 1, 2, 3, 4}, f"Invalid fold: {fold_val}"

            # Check crop dimensions are valid
            assert int(row["crop_width"]) >= 4
            assert int(row["crop_height"]) >= 4

    assert row_count == 16469
    assert len(students_seen) == 215
    assert schools_seen == {"school_a", "school_b"}
    assert scripts_seen == {"latin", "devanagari"}

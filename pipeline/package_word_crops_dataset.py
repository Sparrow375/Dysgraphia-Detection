"""Word Crops Packaging Pipeline for Strategy B: Gated-Attention MIL.

Crops all ~16,469 segmented handwriting words across the complete 215-student
cohort (School A + School B), attaches locked 5-repeat x 5-fold cross-validation
assignments, and creates a lightweight Google Colab-ready ZIP package:
`data/word_crops_dataset.zip` (~20-25 MB).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
FOLDS_PATH = DATA_DIR / "folds" / "student_folds_lookup.csv"

OUTPUT_DIR = DATA_DIR / "word_crops_dataset"
IMAGES_DIR = OUTPUT_DIR / "images"
ZIP_PATH = DATA_DIR / "word_crops_dataset.zip"


def load_student_metadata() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Load student manifests and locked 5x5 fold assignments."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not FOLDS_PATH.exists():
        raise FileNotFoundError(f"Fold lookup not found: {FOLDS_PATH}")

    manifest_map: Dict[str, Dict[str, Any]] = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row["student_id"]).strip()
            manifest_map[sid] = {
                "school": str(row.get("school", "")).strip(),
                "grade": int(float(row.get("grade", 0))),
                "label": int(float(row.get("label", 0))),
            }

    folds_map: Dict[str, Dict[str, int]] = {}
    with open(FOLDS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row["student_id"]).strip()
            folds_map[sid] = {
                "repeat_0_fold": int(row["repeat_0_fold"]),
                "repeat_1_fold": int(row["repeat_1_fold"]),
                "repeat_2_fold": int(row["repeat_2_fold"]),
                "repeat_3_fold": int(row["repeat_3_fold"]),
                "repeat_4_fold": int(row["repeat_4_fold"]),
            }

    return manifest_map, folds_map


def crop_word_image(
    page_img: np.ndarray,
    bbox: List[int],
    padding: int = 4,
) -> Optional[np.ndarray]:
    """Extract word crop with padding, preserving exact aspect ratio."""
    bx, by, bw, bh = bbox
    h_page, w_page = page_img.shape[:2]

    x1 = max(0, bx - padding)
    y1 = max(0, by - padding)
    x2 = min(w_page, bx + bw + padding)
    y2 = min(h_page, by + bh + padding)

    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 4 or (y2 - y1) < 4:
        return None

    crop = page_img[y1:y2, x1:x2]
    return crop


def package_dataset(
    output_dir: Path = OUTPUT_DIR,
    zip_path: Path = ZIP_PATH,
    padding: int = 4,
    clean_previous: bool = True,
) -> Dict[str, Any]:
    """Extract all word crops and generate Colab-ready dataset archive."""
    manifest_map, folds_map = load_student_metadata()

    if clean_previous and output_dir.exists():
        logger.info(f"Cleaning previous output directory: {output_dir}")
        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    words_records: List[Dict[str, Any]] = []
    student_stats: Dict[str, int] = {}
    latin_words_count = 0
    devanagari_words_count = 0
    pos_words_count = 0
    neg_words_count = 0

    logger.info(f"Processing word crops across {len(manifest_map)} students...")

    for sid, meta in manifest_map.items():
        school = meta["school"]
        label = meta["label"]
        grade = meta["grade"]
        folds = folds_map.get(sid, {f"repeat_{r}_fold": -1 for r in range(5)})

        student_dir = PROCESSED_DIR / school / sid
        if not student_dir.exists():
            logger.warning(f"Student directory missing: {student_dir}")
            continue

        page_path = student_dir / "page_normalized.png"
        page_img: Optional[np.ndarray] = None
        if page_path.exists():
            page_img = cv2.imread(str(page_path))

        sentence_files = sorted(list(student_dir.glob("sentence_*.json")))
        student_word_count = 0

        for s_json_path in sentence_files:
            try:
                with open(s_json_path, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading {s_json_path}: {e}")
                continue

            task_id = str(s_data.get("task_id", s_json_path.stem))
            task_name = str(s_data.get("task_name", "")).lower()
            script_meta = str(s_data.get("script", "latin")).lower()
            is_hindi = "hindi" in task_name or script_meta == "devanagari"
            script = "devanagari" if is_hindi else "latin"

            physical_lines = s_data.get("physical_lines", [])
            crop_bbox = s_data.get("crop_bbox")

            # Fallback to task crop image if page_normalized is absent
            task_img: Optional[np.ndarray] = None
            if page_img is None:
                task_img_path = student_dir / f"{task_id}.png"
                if task_img_path.exists():
                    task_img = cv2.imread(str(task_img_path))

            for line_idx, pline in enumerate(physical_lines):
                words = pline.get("words", [])
                for w_idx, word in enumerate(words):
                    bbox = word.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue

                    crop: Optional[np.ndarray] = None
                    if page_img is not None:
                        crop = crop_word_image(page_img, bbox, padding=padding)
                    elif task_img is not None and crop_bbox and len(crop_bbox) >= 4:
                        # Convert to crop-relative coordinates
                        bx, by, bw, bh = bbox
                        lx = max(0, bx - crop_bbox[0])
                        ly = max(0, by - crop_bbox[1])
                        crop = crop_word_image(task_img, [lx, ly, bw, bh], padding=padding)

                    if crop is None or crop.size == 0:
                        continue

                    word_id = f"{sid}_{task_id}_l{line_idx}_w{w_idx}"
                    crop_filename = f"{word_id}.png"
                    crop_path = images_dir / crop_filename

                    # Save compressed word crop image
                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 7])

                    rel_image_path = f"images/{crop_filename}"
                    record = {
                        "word_id": word_id,
                        "student_id": sid,
                        "school": school,
                        "grade": grade,
                        "label": label,
                        "task_id": task_id,
                        "script": script,
                        "line_index": line_idx,
                        "word_index": w_idx,
                        "bbox_x": bbox[0],
                        "bbox_y": bbox[1],
                        "bbox_w": bbox[2],
                        "bbox_h": bbox[3],
                        "crop_width": crop.shape[1],
                        "crop_height": crop.shape[0],
                        "rel_path": rel_image_path,
                        "repeat_0_fold": folds["repeat_0_fold"],
                        "repeat_1_fold": folds["repeat_1_fold"],
                        "repeat_2_fold": folds["repeat_2_fold"],
                        "repeat_3_fold": folds["repeat_3_fold"],
                        "repeat_4_fold": folds["repeat_4_fold"],
                    }
                    words_records.append(record)
                    student_word_count += 1

                    if script == "latin":
                        latin_words_count += 1
                    else:
                        devanagari_words_count += 1

                    if label == 1:
                        pos_words_count += 1
                    else:
                        neg_words_count += 1

        student_stats[sid] = student_word_count

    # Write words index CSV
    index_csv_path = output_dir / "words_index.csv"
    logger.info(f"Writing words index to {index_csv_path} ({len(words_records)} words)...")
    if words_records:
        fieldnames = list(words_records[0].keys())
        with open(index_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(words_records)

    # Write dataset summary JSON
    w_counts = list(student_stats.values())
    meta_summary = {
        "total_students": len(student_stats),
        "students_with_crops": sum(1 for c in w_counts if c > 0),
        "total_word_crops": len(words_records),
        "latin_words": latin_words_count,
        "devanagari_words": devanagari_words_count,
        "positive_students": sum(1 for sid, m in manifest_map.items() if m["label"] == 1),
        "control_students": sum(1 for sid, m in manifest_map.items() if m["label"] == 0),
        "positive_word_crops": pos_words_count,
        "control_word_crops": neg_words_count,
        "words_per_student_mean": float(np.mean(w_counts)) if w_counts else 0.0,
        "words_per_student_median": float(np.median(w_counts)) if w_counts else 0.0,
        "words_per_student_min": int(np.min(w_counts)) if w_counts else 0,
        "words_per_student_max": int(np.max(w_counts)) if w_counts else 0,
        "locked_folds": "5-repeat x 5-fold Stratified Group K-Fold (nested_folds.json)",
    }

    meta_json_path = output_dir / "dataset_meta.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_summary, f, indent=2)

    logger.info("Summary statistics:")
    logger.info(f"  • Students: {meta_summary['total_students']} (Pos: {meta_summary['positive_students']}, Neg: {meta_summary['control_students']})")
    logger.info(f"  • Total Crops: {meta_summary['total_word_crops']} (English: {latin_words_count}, Hindi: {devanagari_words_count})")
    logger.info(f"  • Words per student: min={meta_summary['words_per_student_min']}, max={meta_summary['words_per_student_max']}, mean={meta_summary['words_per_student_mean']:.1f}")

    # Create ZIP archive
    logger.info(f"Creating ZIP archive: {zip_path}...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = Path(root) / file
                arcname = full_path.relative_to(output_dir)
                zf.write(full_path, arcname=arcname)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(f"Successfully packaged dataset: {zip_path} ({zip_size_mb:.2f} MB)")
    meta_summary["zip_size_mb"] = float(zip_size_mb)
    meta_summary["zip_path"] = str(zip_path)

    return meta_summary


if __name__ == "__main__":
    package_dataset()

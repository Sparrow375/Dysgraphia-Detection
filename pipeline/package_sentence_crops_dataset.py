"""Sentence Crops Packaging Pipeline for Strategy A: LoRA TrOCR on Controls.

Collects all 1,145 full sentence crops across the complete 215-student cohort
(School A + School B), pairs them with task ground truth prompts and locked
5-repeat x 5-fold CV splits, and creates a lightweight Google Colab-ready ZIP package:
`data/sentence_crops_dataset.zip` (~30-35 MB).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
FOLDS_PATH = DATA_DIR / "folds" / "student_folds_lookup.csv"

OUTPUT_DIR = DATA_DIR / "sentence_crops_dataset"
IMAGES_DIR = OUTPUT_DIR / "images"
ZIP_PATH = DATA_DIR / "sentence_crops_dataset.zip"

# Standard target prompt texts for each task
PROMPT_MAP: Dict[str, str] = {
    "copy_hindi": "हमारे स्कूल में एक सुंदर बगीचा है",
    "copy_english": "on sundays we play cricket in the park with four friends",
    "dictated_hindi": "मेरी माँ रसोई में खाना बनाती है",
    "dictated_english": "my brother plays football with his friends",
    "own_english": "my name is",
    "own_hindi": "मेरा नाम",
}


def load_metadata() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Load student manifests and locked 5x5 fold assignments."""
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
                f"repeat_{r}_fold": int(row[f"repeat_{r}_fold"]) for r in range(5)
            }

    return manifest_map, folds_map


def package_sentence_dataset(
    output_dir: Path = OUTPUT_DIR,
    zip_path: Path = ZIP_PATH,
    jpeg_quality: int = 90,
    clean_previous: bool = True,
) -> Dict[str, Any]:
    """Package all full sentence crops into a compressed Colab-ready dataset."""
    manifest_map, folds_map = load_metadata()

    if clean_previous and output_dir.exists():
        logger.info(f"Cleaning previous output directory: {output_dir}")
        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    control_crops_count = 0
    dysgraphic_crops_count = 0
    latin_count = 0
    devanagari_count = 0

    logger.info(f"Traversing sentence crops across {len(manifest_map)} students...")

    for sid, meta in manifest_map.items():
        school = meta["school"]
        label = meta["label"]
        grade = meta["grade"]
        folds = folds_map.get(sid, {f"repeat_{r}_fold": -1 for r in range(5)})

        student_dir = PROCESSED_DIR / school / sid
        if not student_dir.exists():
            continue

        sentence_jsons = sorted(list(student_dir.glob("sentence_*.json")))

        for s_json_path in sentence_jsons:
            try:
                with open(s_json_path, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading {s_json_path}: {e}")
                continue

            tid = str(s_data.get("task_id", s_json_path.stem))
            raw_task_name = str(s_data.get("task_name", "")).lower()

            # Clean and normalize task name
            if "copy_hindi" in tid or "copy_hindi" in raw_task_name:
                task_name = "copy_hindi"
                script = "devanagari"
            elif "copy_english" in tid or "copy_english" in raw_task_name:
                task_name = "copy_english"
                script = "latin"
            elif "dictated_hindi" in tid or "dictated_hindi" in raw_task_name:
                task_name = "dictated_hindi"
                script = "devanagari"
            elif "dictated_english" in tid or "dictated_english" in raw_task_name:
                task_name = "dictated_english"
                script = "latin"
            elif "own_english" in tid or "own_english" in raw_task_name:
                task_name = "own_english"
                script = "latin"
            elif "own_hindi" in tid or "own_hindi" in raw_task_name:
                task_name = "own_hindi"
                script = "devanagari"
            else:
                task_name = raw_task_name or tid
                script = "devanagari" if "hindi" in task_name else "latin"

            img_path = student_dir / f"{tid}.png"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                continue

            h, w = img.shape[:2]
            target_prompt = PROMPT_MAP.get(task_name, "")

            out_filename = f"{sid}_{tid}.jpg"
            out_img_path = images_dir / out_filename

            # Save clean compressed JPEG
            cv2.imwrite(str(out_img_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            rel_path = f"images/{out_filename}"
            task_uid = f"{sid}_{tid}"

            record = {
                "task_uid": task_uid,
                "student_id": sid,
                "school": school,
                "grade": grade,
                "label": label,
                "task_id": tid,
                "task_name": task_name,
                "script": script,
                "target_prompt": target_prompt,
                "rel_path": rel_path,
                "img_width": w,
                "img_height": h,
                "repeat_0_fold": folds["repeat_0_fold"],
                "repeat_1_fold": folds["repeat_1_fold"],
                "repeat_2_fold": folds["repeat_2_fold"],
                "repeat_3_fold": folds["repeat_3_fold"],
                "repeat_4_fold": folds["repeat_4_fold"],
            }
            records.append(record)

            if label == 1:
                dysgraphic_crops_count += 1
            else:
                control_crops_count += 1

            if script == "latin":
                latin_count += 1
            else:
                devanagari_count += 1

    # Save index CSV
    index_csv_path = output_dir / "sentences_index.csv"
    if records:
        fieldnames = list(records[0].keys())
        with open(index_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    logger.info(f"Indexed {len(records)} sentence crops -> {index_csv_path}")

    # Create ZIP archive
    logger.info(f"Creating ZIP archive at {zip_path}...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                abs_file = Path(root) / file
                rel_path = abs_file.relative_to(output_dir)
                zf.write(abs_file, rel_path)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(f"ZIP package created: {zip_path} ({zip_size_mb:.2f} MB)")

    summary = {
        "total_crops": len(records),
        "control_crops": control_crops_count,
        "dysgraphic_crops": dysgraphic_crops_count,
        "latin_crops": latin_count,
        "devanagari_crops": devanagari_count,
        "zip_path": str(zip_path),
        "zip_size_mb": float(zip_size_mb),
    }

    # Save summary metadata
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = package_sentence_dataset()
    print("\n" + "=" * 60)
    print("STRATEGY A SENTENCE CROPS DATASET SUMMARY:")
    print(f"  • Total Crops: {summary['total_crops']} (Controls: {summary['control_crops']}, Dysgraphic: {summary['dysgraphic_crops']})")
    print(f"  • Scripts: {summary['latin_crops']} English (Latin), {summary['devanagari_crops']} Hindi (Devanagari)")
    print(f"  • Package Size: {summary['zip_size_mb']:.2f} MB at {summary['zip_path']}")
    print("=" * 60)

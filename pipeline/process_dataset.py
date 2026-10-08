"""End-to-end dataset preprocessing & segmentation orchestrator for Workstream A.

Processes raw handwriting images and generates the standard output hierarchy:
data/processed/<school>/<student_id>/
  - page_normalized.png
  - overlay_debug.png
  - sentence_01_copy_hindi.png
  - sentence_01_copy_hindi.json
  - ...
  - sentence_06_own_english.png
  - sentence_06_own_english.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from pipeline.preprocess import preprocess_page
from pipeline.rules import process_ruled_lines
from pipeline.segment import extract_lines_and_segment, group_lines_into_sentence_blocks
from pipeline.skeleton import process_sentence_skeleton


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_debug_overlay(
    grayscale_deskewed: np.ndarray,
    ruled_lines: List[Dict[str, Any]],
    sentences: List[Dict[str, Any]],
) -> np.ndarray:
    """Create color visualization overlay for visual QA gate inspection."""
    h, w = grayscale_deskewed.shape[:2]
    overlay = cv2.cvtColor(grayscale_deskewed, cv2.COLOR_GRAY2BGR)

    # 1. Draw ruled lines in cyan
    for rl in ruled_lines:
        y_c = int(rl["y_center"])
        if 0 <= y_c < h:
            cv2.line(overlay, (0, y_c), (w - 1, y_c), (255, 255, 0), 1)

    # 2. Draw sentences and words
    colors = {
        "devanagari": (0, 200, 0),    # Green
        "latin": (0, 165, 255),        # Orange
    }

    for s_idx, s in enumerate(sentences):
        bx, by, bw, bh = s["crop_bbox"]
        # Magenta box for sentence crop
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)
        label_text = f"{s['task_name']} ({s['script']}, {s['word_count']} words)"
        cv2.putText(
            overlay,
            label_text,
            (bx + 10, max(20, by - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw individual words
        for line in s["lines"]:
            c_color = colors.get(line["script"], (0, 255, 255))
            for wd in line["words"]:
                wx, wy, ww, wh = wd["bbox"]
                cv2.rectangle(overlay, (wx, wy), (wx + ww, wy + wh), c_color, 1)

    return overlay


def process_single_student(
    row: pd.Series,
    output_base_dir: str = "data/processed",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Process a single student sheet and write outputs to disk."""
    student_id = str(row["student_id"])
    school = str(row["school"])
    img_rel_path = str(row["relative_image_path"])

    student_out_dir = Path(output_base_dir) / school / student_id
    student_out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(img_rel_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {img_rel_path}")

    # Step 1: Page Normalization & Sauvola Binarization
    prep = preprocess_page(
        image_bgr=image_bgr,
        target_width=2000,
        sauvola_window=31,
        sauvola_k=0.2,
    )

    # Step 2: Ruled-Line Detection, Deskewing, and Rule Removal
    rule_res = process_ruled_lines(
        binary_ink=prep["binary_ink"],
        grayscale_norm=prep["grayscale_norm"],
        min_spacing=35,
    )

    # Save normalized deskewed grayscale page
    cv2.imwrite(str(student_out_dir / "page_normalized.png"), rule_res["grayscale_deskewed"])

    # Step 3: Line Segmentation and Script Classification
    lines = extract_lines_and_segment(
        ink_clean=rule_res["ink_clean"],
        ruled_lines=rule_res["ruled_lines"],
        median_spacing_r=rule_res["median_spacing_r"],
    )

    # Step 4: Sentence Block Assembly (6-prompt sequence)
    sentences = group_lines_into_sentence_blocks(
        lines=lines,
        page_shape=rule_res["grayscale_deskewed"].shape,
    )

    # Step 5: Process each sentence crop & skeleton graph
    sentence_metadata_list = []
    for s in sentences:
        task_id = s["task_id"]
        task_name = s["task_name"]
        script = s["script"]
        bx, by, bw, bh = s["crop_bbox"]

        # Sentence crops
        # Inverted clean ink crop: 0 = ink, 255 = paper (standard image representation)
        clean_ink_crop = rule_res["ink_clean"][by : by + bh, bx : bx + bw]
        crop_display = np.where(clean_ink_crop == 1, 0, 255).astype(np.uint8)

        crop_png_name = f"{task_id}.png"
        crop_png_path = student_out_dir / crop_png_name
        cv2.imwrite(str(crop_png_path), crop_display)

        # Skeletonization & Graph Extraction
        avg_h = float(np.mean([l["x_height_h"] for l in s["lines"]])) if s["lines"] else 40.0
        skel_res = process_sentence_skeleton(clean_ink_crop, x_height_h=avg_h, spur_ratio=0.15)

        # Construct Sentence JSON schema
        sentence_schema = {
            "student_id": student_id,
            "school": school,
            "grade": int(row["grade"]),
            "task_id": task_id,
            "task_name": task_name,
            "script": script,
            "crop_filename": crop_png_name,
            "crop_bbox": [bx, by, bw, bh],
            "median_rule_spacing_r": rule_res["median_spacing_r"],
            "median_slope": rule_res["median_slope"],
            "physical_lines": [
                {
                    "line_index": l_idx,
                    "bbox": l["bbox"],
                    "baseline_y": l["baseline_y"],
                    "x_height_h": l["x_height_h"],
                    "script": l["script"],
                    "words": l["words"],
                }
                for l_idx, l in enumerate(s["lines"])
            ],
            "skeleton_graph": {
                "num_nodes": skel_res["graph"]["num_nodes"],
                "num_edges": skel_res["graph"]["num_edges"],
                "junction_count": skel_res["graph"]["junction_count"],
                "endpoint_count": skel_res["graph"]["endpoint_count"],
                "total_path_length": skel_res["graph"]["total_path_length"],
                "mean_branch_length": skel_res["graph"]["mean_branch_length"],
            },
            "qa_flags": {
                "word_count_detected": s["word_count"],
                "line_count": len(s["lines"]),
                "low_contrast": bool(prep["binary_ink"].sum() < 5000),
            },
        }

        json_path = student_out_dir / f"{task_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sentence_schema, f, indent=2)

        sentence_metadata_list.append(sentence_schema)

    # Step 6: Create and save visual QA overlay render
    overlay = draw_debug_overlay(rule_res["grayscale_deskewed"], rule_res["ruled_lines"], sentences)
    cv2.imwrite(str(student_out_dir / "overlay_debug.png"), overlay)

    return {
        "student_id": student_id,
        "school": school,
        "sentences_count": len(sentences),
        "lines_count": len(lines),
        "output_dir": str(student_out_dir),
    }


def process_dataset(
    manifest_path: str = "data/manifest.csv",
    output_dir: str = "data/processed",
    school_filter: Optional[str] = "school_a",
    limit: Optional[int] = None,
    config_path: str = "configs/config.yaml",
) -> pd.DataFrame:
    """Batch process students from manifest.csv."""
    cfg = load_config(config_path) if os.path.exists(config_path) else None
    df = pd.read_csv(manifest_path)

    if school_filter:
        df = df[df["school"] == school_filter].copy()

    if limit is not None:
        df = df.head(limit)

    print(f"Starting Phase 1 processing for {len(df)} sheets (School filter: {school_filter})...")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sheets"):
        try:
            res = process_single_student(row, output_base_dir=output_dir, config=cfg)
            res["status"] = "SUCCESS"
            results.append(res)
        except Exception as e:
            results.append(
                {
                    "student_id": row["student_id"],
                    "school": row["school"],
                    "status": "FAILED",
                    "error": str(e),
                }
            )

    res_df = pd.DataFrame(results)
    success_count = (res_df["status"] == "SUCCESS").sum()
    print(f"\nProcessing Complete: {success_count}/{len(df)} sheets processed successfully.")
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset for Workstream A Phase 1")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--school", default="school_a")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    process_dataset(
        manifest_path=args.manifest,
        output_dir=args.output,
        school_filter=args.school,
        limit=args.limit,
    )

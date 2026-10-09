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


def _json_serialize_default(obj: Any) -> Any:
    """Helper to convert numpy scalars and arrays to native Python types for JSON."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


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

    # 2. Draw sentences, words, and fitted baselines
    colors = {
        "devanagari": (0, 200, 0),     # Green
        "latin": (0, 165, 255),         # Orange
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

        # Draw individual words & baseline points
        for line in s["lines"]:
            c_color = colors.get(line["script"], (0, 255, 255))
            for wd in line["words"]:
                wx, wy, ww, wh = wd["bbox"]
                cv2.rectangle(overlay, (wx, wy), (wx + ww, wy + wh), c_color, 1)

                # Baseline sample point at bottom of word (red dot)
                cx = int(wd.get("cx", wx + ww / 2.0))
                bot = int(wd.get("bottom", wy + wh))
                cv2.circle(overlay, (cx, bot), 3, (0, 0, 255), -1)

            # Draw line fitted baseline in yellow
            m = line.get("baseline_slope", 0.0)
            c = line.get("baseline_intercept", line.get("baseline_y", by + bh))
            x1, x2 = line["bbox"][0], line["bbox"][0] + line["bbox"][2]
            y1 = int(round(m * x1 + c))
            y2 = int(round(m * x2 + c))
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

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

    # Step 1: Page Normalization & Otsu Binarization (no destructive illumination flattening)
    prep = preprocess_page(
        image_bgr=image_bgr,
        target_width=2000,
    )

    # Step 2: Ruled-Line Detection, Deskewing, and Clean Rule Removal
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

        # Sentence crops: saved from clean original deskewed grayscale (natural appearance)
        grayscale_crop = rule_res["grayscale_deskewed"][by : by + bh, bx : bx + bw]
        crop_png_name = f"{task_id}.png"
        crop_png_path = student_out_dir / crop_png_name
        cv2.imwrite(str(crop_png_path), grayscale_crop)

        # Clean ink binary crop for skeleton graph extraction
        clean_ink_crop = rule_res["ink_clean"][by : by + bh, bx : bx + bw]

        # Skeletonization & Graph Extraction
        avg_h = float(np.mean([l["x_height_h"] for l in s["lines"]])) if s["lines"] else 40.0
        skel_res = process_sentence_skeleton(clean_ink_crop, x_height_h=avg_h, spur_ratio=0.15)

        # Construct Sentence JSON schema with per-word baseline mappings
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
            "sentence_baseline_slope": s.get("sentence_baseline_slope", 0.0),
            "sentence_baseline_intercept": s.get("sentence_baseline_intercept", float(by + bh)),
            "sentence_baseline_rmse": s.get("sentence_baseline_rmse", 0.0),
            "word_baseline_points": s.get("word_baseline_points", []),
            "word_baseline_residuals": s.get("word_baseline_residuals", []),
            "physical_lines": [
                {
                    "line_index": l_idx,
                    "bbox": l["bbox"],
                    "baseline_y": l["baseline_y"],
                    "baseline_slope": l.get("baseline_slope", 0.0),
                    "baseline_intercept": l.get("baseline_intercept", l["baseline_y"]),
                    "baseline_rmse": l.get("baseline_rmse", 0.0),
                    "x_height_h": l["x_height_h"],
                    "script": l["script"],
                    "words": l["words"],
                    "word_baseline_points": l.get("word_baseline_points", []),
                    "word_baseline_residuals": l.get("word_baseline_residuals", []),
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
            json.dump(sentence_schema, f, indent=2, default=_json_serialize_default)

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
    manifest = pd.read_csv(manifest_path)

    if school_filter:
        df = manifest[manifest["school"] == school_filter].copy()
    else:
        df = manifest.copy()

    if limit is not None:
        df = df.head(limit)

    config = load_config(config_path) if os.path.exists(config_path) else None

    results = []
    print(f"Starting Phase 1 processing for {len(df)} sheets...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sheets"):
        try:
            res = process_single_student(row, output_base_dir=output_dir, config=config)
            results.append(res)
        except Exception as e:
            print(f"Error processing {row['student_id']}: {e}")
            results.append({
                "student_id": row["student_id"],
                "school": row["school"],
                "error": str(e),
            })

    results_df = pd.DataFrame(results)
    print(f"Completed processing {len(results)} sheets.")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workstream A Phase 1 Dataset Processor")
    parser.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--output-dir", default="data/processed", help="Path to output base directory")
    parser.add_argument("--school", default="school_a", help="Filter by school (default: school_a)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sheets to process")
    args = parser.parse_args()

    process_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        school_filter=args.school,
        limit=args.limit,
    )

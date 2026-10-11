"""Pipeline & Feature Inspection Engine for Workstream A QA.

Generates deep multi-layer visual overlays and quantitative feature diagnostics:
1. Raw grayscale crop (with paper rules)
2. Clean ink mask (rules subtracted, strokes repaired)
3. Ruled line mask (detected rule lines highlighted in red)
4. Skeleton graph overlay (green skeleton stroke, red endpoints, cyan junctions)
5. Baseline & word alignment overlay (word bounding boxes, bottom dots, fitted baseline)
6. Side-by-side feature comparison (Clean Ink vs Raw Crop) identifying rule interference
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.rules import process_ruled_lines
from pipeline.skeleton import process_sentence_skeleton
from features.extractor import extract_sentence_features, FEATURE_DESCRIPTIONS
from features.utils import get_x_height


def generate_task_overlays(
    gray_page: np.ndarray,
    rule_res: Dict[str, Any],
    s_data: Dict[str, Any],
    student_dir: Path,
    task_id: str,
) -> Dict[str, str]:
    """Generate high-resolution visual inspection layers for a single task sentence.

    Returns dict of relative file paths for:
      - raw_url: grayscale crop
      - clean_ink_url: clean ink binary mask
      - rule_mask_url: rule pixels highlighted in red
      - skeleton_url: skeleton graph with endpoints and junctions
      - baseline_url: words and fitted baseline
    """
    bx, by, bw, bh = s_data["crop_bbox"]
    h_page, w_page = gray_page.shape[:2]

    # Clamp coordinates
    bx = max(0, min(bx, w_page - 10))
    by = max(0, min(by, h_page - 10))
    bw = max(10, min(bw, w_page - bx))
    bh = max(10, min(bh, h_page - by))

    crop_gray = gray_page[by : by + bh, bx : bx + bw]
    crop_clean_ink = rule_res["ink_clean"][by : by + bh, bx : bx + bw]
    crop_rule_mask = rule_res["rule_mask"][by : by + bh, bx : bx + bw]

    # 1. Raw grayscale crop (already exists as task_id.png, ensure valid)
    raw_path = student_dir / f"{task_id}.png"
    if not raw_path.exists():
        cv2.imwrite(str(raw_path), crop_gray)

    # 2. Clean ink layer (black ink on white paper, no rules)
    clean_ink_path = student_dir / f"{task_id}_clean_ink.png"
    vis_clean_ink = np.where(crop_clean_ink > 0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(clean_ink_path), vis_clean_ink)

    # 3. Ruled line mask overlay (Red highlight on paper background)
    rule_mask_path = student_dir / f"{task_id}_rule_mask.png"
    vis_rule = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    # Highlight rule pixels in bright coral red
    vis_rule[crop_rule_mask > 0] = [30, 30, 240]
    cv2.imwrite(str(rule_mask_path), vis_rule)

    # 4. Skeleton & Graph Topology Overlay
    skel_path = student_dir / f"{task_id}_skeleton.png"
    x_h = get_x_height(s_data)
    skel_res = process_sentence_skeleton(crop_clean_ink, x_height_h=x_h, spur_ratio=0.15)
    skel_mask = skel_res["skeleton_mask"]
    graph = skel_res["graph"]

    # Subtle contrast background
    vis_skel = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    vis_skel = cv2.addWeighted(vis_skel, 0.35, np.full_like(vis_skel, 255), 0.65, 0)

    # Dilate skeleton slightly (2px) for sharp visibility on high-res scans
    skel_thick = cv2.dilate(skel_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    vis_skel[skel_thick > 0] = [0, 200, 0]  # Bright emerald green stroke path

    # Render endpoints (degree 1) in red and junctions (degree >= 3) in sky blue
    # Extract endpoint and junction coordinates from skel_mask
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_mask, -1, kernel)
    
    # Endpoints: skeleton pixel with exactly 1 neighbor
    ep_y, ep_x = np.where((skel_mask == 1) & (neighbor_count == 1))
    for y, x in zip(ep_y, ep_x):
        cv2.circle(vis_skel, (int(x), int(y)), 4, (0, 0, 240), -1, cv2.LINE_AA)
        cv2.circle(vis_skel, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # Junctions: skeleton pixel with >= 3 neighbors
    jc_y, jc_x = np.where((skel_mask == 1) & (neighbor_count >= 3))
    for y, x in zip(jc_y, jc_x):
        cv2.circle(vis_skel, (int(x), int(y)), 4, (240, 180, 0), -1, cv2.LINE_AA)
        cv2.circle(vis_skel, (int(x), int(y)), 5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(skel_path), vis_skel)

    # 5. Baseline & Word Residuals Overlay
    baseline_path = student_dir / f"{task_id}_baseline.png"
    vis_base = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    lines = s_data.get("physical_lines") or s_data.get("lines") or []
    script = s_data.get("script", "devanagari")
    word_color = (0, 180, 0) if script == "devanagari" else (0, 140, 255)

    all_bottom_points = []
    for line in lines:
        for wd in line.get("words", []):
            wx, wy, ww, wh = wd["bbox"]
            # Map page coordinates to crop coordinates
            lx = wx - bx
            ly = wy - by
            cv2.rectangle(vis_base, (lx, ly), (lx + ww, ly + wh), word_color, 1)

            # Sample point at bottom center
            cx = int(wd.get("cx", wx + ww / 2.0) - bx)
            bot = int(wd.get("bottom", wy + wh) - by)
            if 0 <= cx < bw and 0 <= bot < bh:
                cv2.circle(vis_base, (cx, bot), 4, (0, 0, 255), -1, cv2.LINE_AA)
                all_bottom_points.append((cx, bot))

        # Fitted baseline regression line
        slope = float(line.get("baseline_slope", 0.0))
        intercept = float(line.get("baseline_intercept", line.get("baseline_y", by + bh)))
        # Map line intercept to crop coordinates: y_crop = slope * (x_page) + intercept - by
        x1_page = line["bbox"][0] if "bbox" in line else bx
        x2_page = x1_page + (line["bbox"][2] if "bbox" in line else bw)
        y1_crop = int(round(slope * x1_page + intercept - by))
        y2_crop = int(round(slope * x2_page + intercept - by))
        x1_crop = int(max(0, min(x1_page - bx, bw - 1)))
        x2_crop = int(max(0, min(x2_page - bx, bw - 1)))

        cv2.line(vis_base, (x1_crop, y1_crop), (x2_crop, y2_crop), (0, 230, 255), 2, cv2.LINE_AA)

    # Nearest ruling line reference (cyan dotted/solid line)
    rule_spacing = float(rule_res.get("median_spacing_r", 80.0))
    for rl in rule_res.get("ruled_lines", []):
        y_c = int(rl["y_center"] - by)
        if 0 <= y_c < bh:
            cv2.line(vis_base, (0, y_c), (bw - 1, y_c), (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(baseline_path), vis_base)

    # Return relative URL endpoints
    school = s_data.get("school", "school_a")
    sid = s_data.get("student_id", student_dir.name)
    timestamp = int(time.time())

    return {
        "raw_url": f"/files/processed/{school}/{sid}/{task_id}.png?t={timestamp}",
        "clean_ink_url": f"/files/processed/{school}/{sid}/{task_id}_clean_ink.png?t={timestamp}",
        "rule_mask_url": f"/files/processed/{school}/{sid}/{task_id}_rule_mask.png?t={timestamp}",
        "skeleton_url": f"/files/processed/{school}/{sid}/{task_id}_skeleton.png?t={timestamp}",
        "baseline_url": f"/files/processed/{school}/{sid}/{task_id}_baseline.png?t={timestamp}",
        "endpoints_count": int(skel_res["graph"]["endpoint_count"]),
        "junctions_count": int(skel_res["graph"]["junction_count"]),
        "rule_pixels": int(crop_rule_mask.sum()),
        "clean_ink_pixels": int(crop_clean_ink.sum()),
        "crop_width": int(bw),
        "crop_height": int(bh),
    }


def compute_task_inspection_features(
    s_data: Dict[str, Any],
    crop_gray: np.ndarray,
    crop_clean_ink: np.ndarray,
    crop_rule_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    """Compute features with clean ink mask vs raw crop and generate diagnostics.

    Identifies whether ruled lines severely compromised feature integrity.
    """
    # 1. Clean feature extraction (rules removed)
    clean_gray_sim = np.where(crop_clean_ink > 0, 0, 255).astype(np.uint8)
    feats_clean = extract_sentence_features(s_data, image_crop=clean_gray_sim)

    # 2. Raw feature extraction (with ruled lines)
    feats_raw = extract_sentence_features(s_data, image_crop=crop_gray)

    # 3. Rule contamination metric
    rule_pix = float(crop_rule_mask.sum())
    clean_pix = float(crop_clean_ink.sum())
    total_pix = max(1.0, rule_pix + clean_pix)
    rule_pct = float(rule_pix / total_pix * 100.0)

    feature_cards = []
    for f_name, desc in FEATURE_DESCRIPTIONS.items():
        v_clean = feats_clean.get(f_name, float("nan"))
        v_raw = feats_raw.get(f_name, float("nan"))

        is_nan = np.isnan(v_clean)
        status = "normal"
        status_msg = "Healthy feature extraction"

        if is_nan:
            status = "nan"
            status_msg = "Not applicable or insufficient words/lines"
        elif not np.isnan(v_raw):
            # Check delta caused by ruled lines
            abs_diff = abs(v_clean - v_raw)
            rel_diff = abs_diff / max(1e-4, abs(v_clean))

            if f_name == "components_per_unit_width" and abs_diff > 0.4:
                status = "rule_leakage"
                status_msg = f"Ruled line bridged letters! Diff: {v_clean - v_raw:+.2f} ({rel_diff*100:.0f}%)"
            elif f_name == "ascender_descender_ratio" and abs_diff > 20.0:
                status = "rule_leakage"
                status_msg = f"Ruled line injected false peripheral ink! Raw={v_raw:.1f} vs Clean={v_clean:.1f}"
            elif f_name == "jerk_proxy" and abs_diff > 3.0:
                status = "rule_leakage"
                status_msg = f"Ruled line smoothed tremor! Clean jerk={v_clean:.1f} vs Raw={v_raw:.1f}"
            elif rel_diff > 0.35 and abs_diff > 0.1:
                status = "warning"
                status_msg = f"Ruled lines shifted feature by {rel_diff*100:.0f}%"

        # Check clinical risk threshold
        if status == "normal" and not is_nan:
            if f_name == "components_per_unit_width" and v_clean > 2.2:
                status = "risk"
                status_msg = "High stroke fragmentation (dyspraxic marker)"
            elif f_name == "jerk_proxy" and v_clean > 25.0:
                status = "risk"
                status_msg = "Elevated stroke jerkiness / tremor"
            elif f_name == "baseline_rmse_norm" and v_clean > 0.25:
                status = "risk"
                status_msg = "Significant baseline misalignment"
            elif f_name == "words_written_ratio" and v_clean < 0.5:
                status = "risk"
                status_msg = "Severe task completion collapse"

        feature_cards.append({
            "name": f_name,
            "label": f_name.replace("_", " ").title(),
            "clean_value": None if is_nan else round(float(v_clean), 4),
            "raw_value": None if np.isnan(v_raw) else round(float(v_raw), 4),
            "description": desc,
            "status": status,
            "status_msg": status_msg,
        })

    return feature_cards


def inspect_student_pipeline(student_dir: Path) -> Dict[str, Any]:
    """Inspect full preprocessing and feature extraction pipeline for a student."""
    page_path = student_dir / "page_normalized.png"
    if not page_path.exists():
        return {"error": "page_normalized.png missing"}

    gray_page = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if gray_page is None:
        return {"error": "Failed to read page_normalized.png"}

    h, w = gray_page.shape[:2]

    # Preprocessing check
    _, binary_ink = cv2.threshold(gray_page, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rule_res = process_ruled_lines(binary_ink, gray_page, min_spacing=40)

    # Collect sentence JSONs
    json_files = sorted(student_dir.glob("sentence_*.json"))
    tasks_inspection = []
    total_rule_pixels_page = int(rule_res["rule_mask"].sum())
    total_ink_pixels_page = int(binary_ink.sum())

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                s_data = json.load(f)

            task_id = s_data.get("task_id", jf.stem)
            bx, by, bw, bh = s_data["crop_bbox"]
            crop_gray = gray_page[by : by + bh, bx : bx + bw]
            crop_clean_ink = rule_res["ink_clean"][by : by + bh, bx : bx + bw]
            crop_rule_mask = rule_res["rule_mask"][by : by + bh, bx : bx + bw]

            # Generate visual overlay bundle
            overlays = generate_task_overlays(
                gray_page=gray_page,
                rule_res=rule_res,
                s_data=s_data,
                student_dir=student_dir,
                task_id=task_id,
            )

            # Compute feature comparison & warnings
            features = compute_task_inspection_features(
                s_data=s_data,
                crop_gray=crop_gray,
                crop_clean_ink=crop_clean_ink,
                crop_rule_mask=crop_rule_mask,
            )

            rule_leakage_detected = any(f["status"] == "rule_leakage" for f in features)

            tasks_inspection.append({
                "task_id": task_id,
                "task_name": s_data.get("task_name", task_id),
                "script": s_data.get("script", "devanagari"),
                "crop_bbox": [bx, by, bw, bh],
                "word_count": s_data.get("qa_flags", {}).get("word_count_detected", 0),
                "overlays": overlays,
                "rule_leakage_detected": rule_leakage_detected,
                "features": features,
            })
        except Exception as e:
            continue

    return {
        "student_id": student_dir.name,
        "page_width": w,
        "page_height": h,
        "median_rule_spacing_r": float(rule_res.get("median_spacing_r", 80.0)),
        "median_slope": float(rule_res.get("median_slope", 0.0)),
        "detected_rule_lines_count": len(rule_res.get("ruled_lines", [])),
        "total_page_rule_pixels": total_rule_pixels_page,
        "tasks": tasks_inspection,
    }

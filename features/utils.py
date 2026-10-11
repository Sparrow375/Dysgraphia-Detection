"""Shared utility functions for feature extraction modules.

Centralises helpers that are otherwise copy-pasted across baseline.py,
rule_offset.py, slant.py, curvature.py, gaps.py, size.py, fragmentation.py,
and hindi.py to prevent silent divergence between modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def get_x_height(sentence_json: Dict[str, Any], default: float = 40.0) -> float:
    """Retrieve or robustly estimate x-height h (core-band height in pixels).

    Priority order:
      1. Median of ``x_height_h`` values stored per physical line — set by the
         preprocessing pipeline from the actual ink distribution.
      2. Crop-bbox fallback: ``(bbox_height / lines_used) * 0.45``.
         Dividing by ``lines_used`` converts a multi-line crop height into a
         per-line estimate before applying the core-band fraction.  The fraction
         0.45 is calibrated to the School A scans (~45 % of inter-line spacing
         is occupied by the core band for both Devanagari and Latin tasks).
         The old constant ``bbox_height * 0.4`` ignored ``lines_used``, causing
         grade-3 crops (which are often taller) to produce an inflated h estimate
         that corrupted every h-normalised feature for that grade.

    Returns
    -------
    float
        Estimated x-height in pixels. Always >= 15.0.
    """
    lines = sentence_json.get("physical_lines") or sentence_json.get("lines") or []
    h_vals = [
        float(l["x_height_h"])
        for l in lines
        if l.get("x_height_h") and float(l["x_height_h"]) > 5.0
    ]
    if h_vals:
        return float(np.median(h_vals))

    # Fallback: derive from crop geometry
    bbox = sentence_json.get("crop_bbox", [0, 0, 100, 100])
    bh = float(bbox[3]) if len(bbox) >= 4 else 100.0
    lines_used = max(1, len(lines))
    per_line_h = bh / lines_used
    return max(15.0, per_line_h * 0.45)


def load_or_binarize_ink(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
    data_root: str = "data/processed",
) -> Optional[np.ndarray]:
    """Return a clean binary ink mask (uint8, 1=ink) with ruled lines removed.

    Priority order:
      1. If clean ink file ({task_id}_clean_ink.png) exists on disk, load it directly.
      2. If image_crop is passed or only grayscale is available, perform Otsu thresholding
         followed by long-rule morphological filtering and crossing stroke restoration
         to ensure printed notebook guidelines do not contaminate handwriting features.

    Parameters
    ----------
    sentence_json:
        Parsed sentence JSON dict.
    image_crop:
        Optional pre-loaded grayscale or BGR image array.
    data_root:
        Root directory under which school/student_id/crop_filename is resolved.

    Returns
    -------
    np.ndarray or None
        Binary mask with dtype uint8 (values 0 or 1), or None if unavailable.
    """
    crop_fn = sentence_json.get("crop_filename")
    sid = sentence_json.get("student_id")
    school = sentence_json.get("school", "school_a")

    # Priority 1: Check for clean ink image on disk
    if crop_fn and sid:
        clean_fn = Path(crop_fn).stem + "_clean_ink.png"
        clean_path = Path(data_root) / school / sid / clean_fn
        if clean_path.exists():
            clean_gray = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
            if clean_gray is not None:
                return (clean_gray < 128).astype(np.uint8)

    # Priority 2: Use provided image_crop or load crop from disk
    gray = None
    if image_crop is not None:
        gray = (
            cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            if len(image_crop.shape) == 3
            else image_crop
        )
    elif crop_fn and sid:
        path = Path(data_root) / school / sid / crop_fn
        if path.exists():
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        return None

    # Binarize with Otsu
    _, bin_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (bin_ink > 0).astype(np.uint8)

    # Thickness-aware ruled-line removal with stroke & Shirorekha preservation
    h, w = ink.shape[:2]
    script = str(sentence_json.get("script", "devanagari")).lower()
    if w >= 80 and h >= 20:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(60, w // 3), 1))
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        row_proj = opened.sum(axis=1)

        # Detect candidate rule line rows (local peaks with projection > 0.15*w)
        line_y_centers = []
        for r in range(1, h - 1):
            if row_proj[r] > 0.15 * w and row_proj[r] >= row_proj[r - 1] and row_proj[r] >= row_proj[r + 1]:
                if not any(abs(r - prev) < 20 for prev in line_y_centers):
                    line_y_centers.append(r)

        if line_y_centers:
            ink_clean = ink.copy()
            for y_line in line_y_centers:
                for x in range(w):
                    if ink[y_line, x] == 0:
                        continue
                    yt = y_line
                    while yt >= 0 and ink[yt, x] == 1:
                        yt -= 1
                    yb = y_line
                    while yb < h and ink[yb, x] == 1:
                        yb += 1

                    run_total = yb - yt - 1
                    dist_below = yb - 1 - y_line

                    # Shirorekha protection in Devanagari:
                    if script == "devanagari" and dist_below >= 8:
                        continue

                    # Letter crossing & body protection:
                    if run_total >= 5:
                        continue

                    # Pure printed rule line in gap/margin
                    if run_total <= 4:
                        ink_clean[yt + 1 : yb, x] = 0

            # Remove residual pepper noise / tiny specks (<= 4 pixels)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_clean, connectivity=8)
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] <= 4:
                    ink_clean[labels == lbl] = 0

            return ink_clean

    return ink

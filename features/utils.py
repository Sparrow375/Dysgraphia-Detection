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
    """Return a binary ink mask (uint8, 1=ink) for the sentence crop.

    If ``image_crop`` is supplied it is binarised directly via Otsu thresholding.
    Otherwise the crop image is loaded from disk using the path recorded in the
    sentence JSON schema.

    Parameters
    ----------
    sentence_json:
        Parsed sentence JSON dict.
    image_crop:
        Optional pre-loaded grayscale or BGR image array.
    data_root:
        Root directory under which ``school/student_id/crop_filename`` is resolved.

    Returns
    -------
    np.ndarray or None
        Binary mask with dtype uint8 (values 0 or 1), or None if the image
        cannot be loaded.
    """
    if image_crop is not None:
        gray = (
            cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            if len(image_crop.shape) == 3
            else image_crop
        )
        _, bin_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return (bin_ink > 0).astype(np.uint8)

    crop_fn = sentence_json.get("crop_filename")
    sid = sentence_json.get("student_id")
    school = sentence_json.get("school", "school_a")

    if crop_fn and sid:
        path = Path(data_root) / school / sid / crop_fn
        if path.exists():
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                _, bin_ink = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                return (bin_ink > 0).astype(np.uint8)

    return None

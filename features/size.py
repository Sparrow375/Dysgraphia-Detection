"""Size, proportion, and font metric features for Workstream A.

Definitions:
- word_height_cv: CV of word heights within sentence.
- h_over_r: ratio of x-height h to ruled-line spacing r.
- word_width_per_char: sum of word widths / expected prompt characters (copy/dictation; NaN for own writing).
- component_height_cv: mean intra-word connected component height CV.
- ascender_descender_ratio: (English) ascender + descender height relative to h.
- matra_ratio: (Hindi) matra height (above shirorekha + below baseline) relative to core band h.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Expected character counts from standardized study protocol
EXPECTED_CHAR_COUNTS: Dict[str, int] = {
    "copy_hindi": 34,
    "copy_english": 74,
    "dictated_hindi": 28,
    "dictated_english": 38,
}


def _get_x_height(sentence_json: Dict[str, Any], default: float = 40.0) -> float:
    """Retrieve or estimate x-height h."""
    lines = sentence_json.get("physical_lines") or sentence_json.get("lines") or []
    h_vals = [float(l["x_height_h"]) for l in lines if l.get("x_height_h") and float(l["x_height_h"]) > 5.0]
    if h_vals:
        return float(np.median(h_vals))
    bbox = sentence_json.get("crop_bbox", [0, 0, 100, 100])
    bh = bbox[3] if len(bbox) >= 4 else 100
    return max(15.0, float(bh * 0.4))


def _load_or_binarize_ink(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Retrieve or compute binary ink mask for sentence crop."""
    if image_crop is not None:
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
        _, bin_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return (bin_ink > 0).astype(np.uint8)

    crop_fn = sentence_json.get("crop_filename")
    sid = sentence_json.get("student_id")
    school = sentence_json.get("school", "school_a")

    if crop_fn and sid:
        path = Path("data/processed") / school / sid / crop_fn
        if path.exists():
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                _, bin_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                return (bin_ink > 0).astype(np.uint8)

    return None


def compute_size_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute size, proportion, and character metric features.

    Returns:
        dict with keys:
            - word_height_cv: float | NaN
            - h_over_r: float | NaN
            - word_width_per_char: float | NaN
            - component_height_cv: float | NaN
            - ascender_descender_ratio: float | NaN
            - matra_ratio: float | NaN
    """
    h = _get_x_height(sentence_json)
    r = float(sentence_json.get("median_rule_spacing_r", 0.0))
    script = str(sentence_json.get("script", "devanagari")).lower()
    task_name = str(sentence_json.get("task_name", "")).lower()

    lines = sentence_json.get("physical_lines") or sentence_json.get("lines") or []
    all_words: List[Dict[str, Any]] = []
    for l in lines:
        all_words.extend(l.get("words", []))

    # 1. h_over_r
    if r > 10.0 and h > 0:
        h_over_r = float(h / r)
    else:
        h_over_r = float("nan")

    # If fewer than 3 words, set word-level metrics to NaN
    if len(all_words) < 3 or h <= 0:
        return {
            "word_height_cv": float("nan"),
            "h_over_r": h_over_r,
            "word_width_per_char": float("nan"),
            "component_height_cv": float("nan"),
            "ascender_descender_ratio": float("nan"),
            "matra_ratio": float("nan"),
        }

    # 2. word_height_cv
    word_heights = [float(w["bbox"][3]) for w in all_words if "bbox" in w and len(w["bbox"]) == 4]
    if len(word_heights) >= 3:
        arr_h = np.array(word_heights, dtype=np.float64)
        mean_h = float(np.mean(arr_h))
        std_h = float(np.std(arr_h, ddof=1))
        word_height_cv = float(std_h / mean_h) if mean_h > 0 else float("nan")
    else:
        word_height_cv = float("nan")

    # 3. word_width_per_char (for copy/dictation)
    matched_task = None
    for k in EXPECTED_CHAR_COUNTS:
        if k in task_name:
            matched_task = k
            break

    if matched_task and matched_task in EXPECTED_CHAR_COUNTS:
        exp_chars = EXPECTED_CHAR_COUNTS[matched_task]
        total_word_width = sum(float(w["bbox"][2]) for w in all_words if "bbox" in w and len(w["bbox"]) == 4)
        word_width_per_char = float(total_word_width / exp_chars) if exp_chars > 0 else float("nan")
    else:
        word_width_per_char = float("nan")

    # 4. component_height_cv, ascender_descender_ratio, matra_ratio
    ink_mask = _load_or_binarize_ink(sentence_json, image_crop)

    component_cvs: List[float] = []
    asc_desc_ratios: List[float] = []
    matra_ratios: List[float] = []

    crop_bbox = sentence_json.get("crop_bbox", [0, 0, 0, 0])
    bx, by = crop_bbox[0], crop_bbox[1]

    if ink_mask is not None and ink_mask.sum() > 20:
        img_h, img_w = ink_mask.shape[:2]

        for wd in all_words:
            if "bbox" not in wd or len(wd["bbox"]) != 4:
                continue
            wx, wy, ww, wh = wd["bbox"]
            # Map page coordinates to crop coordinates
            lx = max(0, min(wx - bx, img_w - 5))
            ly = max(0, min(wy - by, img_h - 5))
            lw = max(5, min(ww, img_w - lx))
            lh = max(5, min(wh, img_h - ly))

            word_patch = ink_mask[ly : ly + lh, lx : lx + lw]
            if word_patch.sum() < 5:
                continue

            # Connected components within word
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(word_patch, connectivity=8)
            # Filter background (label 0)
            comp_h = [float(stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 4]

            if len(comp_h) >= 2:
                c_mean = float(np.mean(comp_h))
                c_std = float(np.std(comp_h, ddof=1))
                if c_mean > 0:
                    component_cvs.append(c_std / c_mean)

            # Script-specific height ratios
            if script == "latin":
                # For Latin: ascender / descender ratio
                # Core band is middle 50% of word; ascender extends above, descender below
                # If word has significant height compared to h
                if lh > h:
                    extra_h = lh - h
                    asc_desc_ratios.append(float(extra_h / h))
                else:
                    asc_desc_ratios.append(0.0)

            elif script == "devanagari":
                # For Devanagari: matra ratio
                # Portions extending above shirorekha or below baseline
                if lh > h:
                    extra_matra = lh - h
                    matra_ratios.append(float(extra_matra / h))
                else:
                    matra_ratios.append(0.0)

    comp_height_cv = float(np.mean(component_cvs)) if component_cvs else float("nan")

    if script == "latin" and asc_desc_ratios:
        asc_desc_ratio = float(np.mean(asc_desc_ratios))
        matra_ratio = float("nan")
    elif script == "devanagari" and matra_ratios:
        asc_desc_ratio = float("nan")
        matra_ratio = float(np.mean(matra_ratios))
    else:
        asc_desc_ratio = float("nan")
        matra_ratio = float("nan")

    return {
        "word_height_cv": word_height_cv,
        "h_over_r": h_over_r,
        "word_width_per_char": word_width_per_char,
        "component_height_cv": comp_height_cv,
        "ascender_descender_ratio": asc_desc_ratio,
        "matra_ratio": matra_ratio,
    }

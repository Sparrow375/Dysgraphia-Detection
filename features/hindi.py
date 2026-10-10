"""Hindi / Devanagari-specific feature extraction.

Definitions:
- shirorekha_rms_deviation_norm: Root-mean-square deviation of headline from straight line / h.
- shirorekha_breaks_per_word: Discontinuities / gaps in the shirorekha per word.
- shirorekha_tilt_var: Variance of headline tilt slopes across words.
- Returns NaN if script is not 'devanagari' or fewer than 3 words.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from features.utils import get_x_height as _get_x_height_shared, load_or_binarize_ink as _load_or_binarize_ink_shared


def _get_x_height(sentence_json: Dict[str, Any], default: float = 40.0) -> float:
    """Retrieve or estimate x-height h."""
    return _get_x_height_shared(sentence_json, default=default)


def _load_or_binarize_ink(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Retrieve or compute binary ink mask for sentence crop."""
    return _load_or_binarize_ink_shared(sentence_json, image_crop)


def analyze_word_shirorekha(
    word_patch: np.ndarray,
    h: float,
) -> Optional[Tuple[float, int, float]]:
    """Analyze shirorekha in a single Devanagari word patch.

    Returns:
        (rms_deviation, break_count, tilt_slope) or None if undetectable.
    """
    ph, pw = word_patch.shape[:2]
    if pw < 10 or ph < 10 or word_patch.sum() < 15:
        return None

    # Horizontal projection in upper 15%-50% of word to find headline row
    y_start = max(1, int(0.12 * ph))
    y_end = max(y_start + 2, int(0.55 * ph))
    proj = np.sum(word_patch[y_start:y_end, :], axis=1)

    if len(proj) == 0 or np.max(proj) < 5:
        return None

    peak_local_y = int(np.argmax(proj))
    y_shiro = y_start + peak_local_y

    # Inspect band around headline: [y_shiro - 3, y_shiro + 3]
    b_top = max(0, y_shiro - 3)
    b_bot = min(ph, y_shiro + 4)
    band = word_patch[b_top:b_bot, :]  # (band_h, pw)

    # Column-wise detection
    col_has_ink = np.any(band > 0, axis=0)  # bool array of length pw
    ink_cols = np.where(col_has_ink)[0]

    if len(ink_cols) < 8:
        return None

    # Track headline y-coordinates for ink columns
    y_pts: List[float] = []
    x_pts: List[float] = []

    for c in ink_cols:
        rows_with_ink = np.where(band[:, c] > 0)[0] + b_top
        # Take center of ink in this column
        y_pts.append(float(np.mean(rows_with_ink)))
        x_pts.append(float(c))

    x_arr = np.array(x_pts, dtype=np.float64)
    y_arr = np.array(y_pts, dtype=np.float64)

    # Linear fit y = m*x + c
    if len(x_arr) >= 4:
        p = np.polyfit(x_arr, y_arr, 1)
        slope = float(p[0])
        fitted_y = np.polyval(p, x_arr)
        rms = float(np.sqrt(np.mean((y_arr - fitted_y) ** 2)))
    else:
        slope = 0.0
        rms = float(np.std(y_arr))

    # Count breaks (consecutive gaps of no-ink between first and last ink column)
    c_start, c_end = ink_cols[0], ink_cols[-1]
    active_span = col_has_ink[c_start : c_end + 1]

    # Find transitions from True to False (start of break)
    breaks = 0
    in_break = False
    for v in active_span:
        if not v and not in_break:
            breaks += 1
            in_break = True
        elif v:
            in_break = False

    return rms, breaks, slope


def compute_hindi_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Devanagari shirorekha deviation, breaks, and tilt variability.

    Returns:
        dict with keys:
            - shirorekha_rms_deviation_norm: float | NaN
            - shirorekha_breaks_per_word: float | NaN
            - shirorekha_tilt_var: float | NaN
    """
    nan_dict = {
        "shirorekha_rms_deviation_norm": float("nan"),
        "shirorekha_breaks_per_word": float("nan"),
        "shirorekha_tilt_var": float("nan"),
    }

    script = str(sentence_json.get("script", "devanagari")).lower()
    if script != "devanagari":
        return nan_dict

    h = _get_x_height(sentence_json)
    ink_mask = _load_or_binarize_ink(sentence_json, image_crop)

    lines = sentence_json.get("physical_lines") or sentence_json.get("lines") or []
    all_words: List[Dict[str, Any]] = []
    for l in lines:
        all_words.extend(l.get("words", []))

    if len(all_words) < 3 or ink_mask is None or ink_mask.sum() < 20 or h <= 5.0:
        return nan_dict

    crop_bbox = sentence_json.get("crop_bbox", [0, 0, 0, 0])
    bx, by = crop_bbox[0], crop_bbox[1]
    img_h, img_w = ink_mask.shape[:2]

    rms_list: List[float] = []
    breaks_list: List[int] = []
    slopes_list: List[float] = []

    for wd in all_words:
        if "bbox" not in wd or len(wd["bbox"]) != 4:
            continue
        wx, wy, ww, wh = wd["bbox"]
        lx = max(0, min(wx - bx, img_w - 5))
        ly = max(0, min(wy - by, img_h - 5))
        lw = max(5, min(ww, img_w - lx))
        lh = max(5, min(wh, img_h - ly))

        patch = ink_mask[ly : ly + lh, lx : lx + lw]
        res = analyze_word_shirorekha(patch, h)
        if res is not None:
            rms_val, brk_val, slp_val = res
            rms_list.append(rms_val)
            breaks_list.append(brk_val)
            slopes_list.append(slp_val)

    if len(rms_list) < 2:
        return nan_dict

    # 1. Shirorekha RMS deviation normalized by h
    mean_rms = float(np.mean(rms_list))
    rms_norm = float(mean_rms / h) if h > 0 else float("nan")

    # 2. Breaks per word
    breaks_per_word = float(np.mean(breaks_list))

    # 3. Tilt variability (variance of slopes)
    tilt_var = float(np.var(slopes_list, ddof=1)) if len(slopes_list) >= 2 else 0.0

    return {
        "shirorekha_rms_deviation_norm": rms_norm,
        "shirorekha_breaks_per_word": breaks_per_word,
        "shirorekha_tilt_var": tilt_var,
    }

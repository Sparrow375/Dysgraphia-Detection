"""Ruled-line detection, deskewing, and rule removal module for Workstream A.

Implements:
1. Horizontal morphological opening and projection peak detection to locate ruled lines.
2. RANSAC linear fitting (y = ax + b) for each line to capture tilt and intercept.
3. Deskewing of grayscale and binary ink images by the median rule slope.
4. Rule removal and crossing stroke restoration using vertical morphological closing.
5. Vertical notebook margin line removal.
6. Extraction of median spacing r and rule parameters for alignment features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor


def detect_ruled_lines(
    binary_ink: np.ndarray,
    min_spacing: int = 40,
    horizontal_kernel_width: int = 51,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Detect ruled lines using horizontal opening and RANSAC linear fitting.

    Returns:
        ruled_lines: list of dicts with 'slope', 'intercept', 'y_center'
        median_slope: float median slope across lines
        median_spacing_r: float median line spacing in pixels
    """
    h, w = binary_ink.shape[:2]

    # Horizontal opening to isolate long printed rule segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))
    opened = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, kernel)
    proj = opened.sum(axis=1)

    max_p = proj.max()
    if max_p == 0:
        return [], 0.0, 80.0

    peaks, _ = find_peaks(proj, height=0.15 * max_p, distance=min_spacing)
    if len(peaks) == 0:
        return [], 0.0, 80.0

    ruled_lines: List[Dict[str, Any]] = []
    slopes: List[float] = []

    # Fit each line with RANSAC
    delta_y = 4  # Search window around peak row
    for p in peaks:
        y_min = max(0, p - delta_y)
        y_max = min(h, p + delta_y + 1)
        sub_strip = opened[y_min:y_max, :]

        y_coords, x_coords = np.where(sub_strip > 0)
        if len(x_coords) < 100:
            ruled_lines.append({"slope": 0.0, "intercept": float(p), "y_center": float(p)})
            slopes.append(0.0)
            continue

        global_y = y_coords + y_min
        X = x_coords.reshape(-1, 1)
        Y = global_y

        try:
            ransac = RANSACRegressor(residual_threshold=2.0, max_trials=100, random_state=42)
            ransac.fit(X, Y)
            slope = float(ransac.estimator_.coef_[0])
            intercept = float(ransac.estimator_.intercept_)
            if abs(slope) > 0.15:
                slope = 0.0
                intercept = float(p)
        except Exception:
            slope = 0.0
            intercept = float(p)

        ruled_lines.append({"slope": slope, "intercept": intercept, "y_center": float(p)})
        slopes.append(slope)

    median_slope = float(np.median(slopes)) if slopes else 0.0

    if len(peaks) > 1:
        spacings = np.diff([line["y_center"] for line in ruled_lines])
        median_spacing_r = float(np.median(spacings))
    else:
        median_spacing_r = 80.0

    return ruled_lines, median_slope, median_spacing_r


def deskew_image(image: np.ndarray, slope: float) -> np.ndarray:
    """Deskew image by rotating by -arctan(slope)."""
    if abs(slope) < 1e-4:
        return image

    angle_deg = np.degrees(np.arctan(slope))
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    deskewed = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR if image.ndim == 3 else cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0 if image.ndim == 2 and image.dtype == np.uint8 and image.max() <= 1 else 255,
    )
    return deskewed


def remove_rules_and_restore_strokes(
    binary_ink: np.ndarray,
    ruled_lines: List[Dict[str, Any]],
    median_spacing_r: float,
    rule_thickness: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove ruled-line and vertical margin pixels while strictly preserving handwriting strokes and Shirorekha.

    Uses vertical run-length thickness awareness:
    1. A printed ruled line has a thin vertical profile (typically 1-3px).
    2. Handwritten strokes, letters, and Devanagari Shirorekha have larger thickness
       or connect to character bodies extending above/below the line.
    3. At each column x along the detected line trajectory y_line(x), continuous vertical
       ink runs are measured:
       - Runs <= 4px with no downward character body are marked for erasure (bare printed rule).
       - Runs crossing genuine letters (>= 5px) or with downward character bodies are
         PRESERVED intact, preventing horizontal stroke slicing.
    4. Residual speckles/pepper noise (<= 4px area) along rule bands are eliminated.

    Returns:
        ink_clean: binary ink mask with rules removed and handwriting preserved intact
        rule_mask: binary mask of detected rule pixels that were removed
    """
    binary_ink = (binary_ink > 0).astype(np.uint8)
    h, w = binary_ink.shape[:2]
    rule_mask = np.zeros((h, w), dtype=np.uint8)
    x_all = np.arange(w)

    for line in ruled_lines:
        slope = line["slope"]
        intercept = line["intercept"]
        y_line = np.clip(np.round(slope * x_all + intercept).astype(int), 0, h - 1)

        active_x = np.where(binary_ink[y_line, x_all] == 1)[0]
        for x in active_x:
            yl = y_line[x]
            # Measure upward run
            yt = yl
            while yt >= 0 and binary_ink[yt, x] == 1:
                yt -= 1
            # Measure downward run
            yb = yl
            while yb < h and binary_ink[yb, x] == 1:
                yb += 1

            run_total = yb - yt - 1
            dist_below = yb - 1 - yl

            # 1. Shirorekha protection: if characters hang downward from the line >= 8px,
            # this horizontal ink is a headline connecting characters. DO NOT ERASE.
            if dist_below >= 8:
                continue

            # 2. Letter crossing & stroke body protection: if the stroke extends
            # vertically >= 5px through the line, preserve it intact.
            if run_total >= 5:
                continue

            # 3. Pure printed rule in gaps/margins (thin line <= 4px)
            if run_total <= 4:
                rule_mask[yt + 1 : yb, x] = 1

    # Subtract confirmed rule pixels
    ink_no_rules = np.where(rule_mask == 1, 0, binary_ink).astype(np.uint8)

    # Detect and remove long vertical margin lines (>= 120px tall)
    kernel_vert_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120))
    v_lines = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, kernel_vert_line)
    v_lines_dilated = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    ink_clean = np.where(v_lines_dilated == 1, 0, ink_no_rules).astype(np.uint8)

    # Remove residual speckles/pepper noise (<= 4 pixels)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_clean, connectivity=8)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] <= 4:
            ink_clean[labels == lbl] = 0

    return ink_clean, rule_mask


def process_ruled_lines(
    binary_ink: np.ndarray,
    grayscale_norm: np.ndarray,
    min_spacing: int = 40,
    rule_thickness: int = 11,
) -> Dict[str, Any]:
    """Complete rule detection, deskewing, and removal pipeline.

    Returns dict with:
        - ruled_lines: List of fitted line parameters
        - median_spacing_r: float
        - median_slope: float
        - ink_with_rules: deskewed binary ink mask with rules intact
        - ink_clean: deskewed binary ink mask with rules removed and strokes restored
        - grayscale_deskewed: deskewed grayscale image
        - rule_mask: binary mask of ruled lines
    """
    lines, median_slope, r = detect_ruled_lines(binary_ink, min_spacing=min_spacing)

    # Deskew all layers by median slope
    ink_deskewed = deskew_image(binary_ink, median_slope)
    gray_deskewed = deskew_image(grayscale_norm, median_slope)

    # Re-detect lines on deskewed image so slopes become ~0
    lines_deskewed, _, r = detect_ruled_lines(ink_deskewed, min_spacing=min_spacing)

    # Remove rules and restore crossed strokes with full coverage
    ink_clean, rule_mask = remove_rules_and_restore_strokes(
        ink_deskewed, lines_deskewed, median_spacing_r=r, rule_thickness=rule_thickness
    )

    return {
        "ruled_lines": lines_deskewed,
        "median_spacing_r": r,
        "median_slope": median_slope,
        "ink_with_rules": ink_deskewed,
        "ink_clean": ink_clean,
        "grayscale_deskewed": gray_deskewed,
        "rule_mask": rule_mask,
    }


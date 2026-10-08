"""Ruled-line detection, deskewing, and rule removal module for Workstream A.

Implements:
1. Horizontal morphological opening and projection peak detection to locate ruled lines.
2. RANSAC linear fitting (y = ax + b) for each line to capture tilt and intercept.
3. Deskewing of grayscale and binary ink images by the median rule slope.
4. Rule removal and crossing stroke restoration using vertical morphological closing.
5. Extraction of median spacing r and rule parameters for alignment features.
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
    horizontal_kernel_width: int = 61,
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

    peaks, _ = find_peaks(proj, height=0.20 * max_p, distance=min_spacing)
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
            # Fallback to horizontal line at y=p
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
            # Reject extreme slopes that cannot be ruled notebook lines (|slope| > 0.15)
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
    rule_thickness: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove ruled-line pixels and restore intersecting descender/ascender strokes.

    Returns:
        ink_clean: binary ink mask with rules removed and crossed strokes repaired
        rule_mask: binary mask of detected rule pixels
    """
    h, w = binary_ink.shape[:2]
    rule_mask = np.zeros((h, w), dtype=np.uint8)
    x_all = np.arange(w)

    half_thick = max(1, rule_thickness // 2)

    for line in ruled_lines:
        slope = line["slope"]
        intercept = line["intercept"]
        y_line = np.round(slope * x_all + intercept).astype(int)

        for dy in range(-half_thick, half_thick + 1):
            y_cur = np.clip(y_line + dy, 0, h - 1)
            rule_mask[y_cur, x_all] = 1

    # Only remove rule pixels where horizontal opening confirmed long rule line ink
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1))
    confirmed_rules = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, kernel_horiz) & rule_mask

    # Step 1: Subtract rule pixels
    ink_no_rules = np.where(confirmed_rules == 1, 0, binary_ink).astype(np.uint8)

    # Step 2: Stroke restoration
    # Localized vertical closing (5x1) limited strictly to the confirmed rule band
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vert_closed = cv2.morphologyEx(ink_no_rules, cv2.MORPH_CLOSE, kernel_vert)

    # Restore pixels only where strokes crossed the rule band
    restored_strokes = (vert_closed == 1) & (confirmed_rules == 1)
    ink_clean = np.where(restored_strokes, 1, ink_no_rules).astype(np.uint8)

    return ink_clean, confirmed_rules


def process_ruled_lines(
    binary_ink: np.ndarray,
    grayscale_norm: np.ndarray,
    min_spacing: int = 40,
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

    # Re-detect lines on deskewed image so slopes become 0
    lines_deskewed, _, r = detect_ruled_lines(ink_deskewed, min_spacing=min_spacing)

    # Remove rules and restore crossed strokes
    ink_clean, rule_mask = remove_rules_and_restore_strokes(
        ink_deskewed, lines_deskewed, median_spacing_r=r, rule_thickness=3
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

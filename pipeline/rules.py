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


def estimate_skew_angle(binary_ink: np.ndarray) -> float:
    """Estimate global page skew angle in degrees using Hough line segments on ruled lines."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 1))
    opened = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, kernel)
    lines = cv2.HoughLinesP(opened, 1, np.pi / 180, threshold=120, minLineLength=80, maxLineGap=25)
    if lines is None:
        return 0.0

    angles = []
    for l in lines:
        x1, y1, x2, y2 = l.flatten()
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 60:
            deg = float(np.degrees(np.arctan2(dy, dx)))
            if abs(deg) < 10.0:
                angles.append(deg)

    return float(np.median(angles)) if angles else 0.0


def deskew_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Deskew image by rotating by -angle_deg around image center."""
    if abs(angle_deg) < 1e-3:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    interp = cv2.INTER_CUBIC if image.ndim == 3 or image.dtype != np.uint8 or image.max() > 1 else cv2.INTER_NEAREST
    deskewed = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0 if image.ndim == 2 and image.max() <= 1 else 255,
    )
    return deskewed


def detect_ruled_lines(
    binary_ink: np.ndarray,
    min_spacing: int = 35,
    horizontal_kernel_width: int = 51,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Detect horizontal ruled lines on deskewed ink using horizontal opening and linear fitting.

    Returns:
        ruled_lines: list of dicts with 'line_index', 'slope', 'intercept', 'y_center'
        median_slope: float median line slope across detected rules
        median_spacing_r: float median line spacing in pixels
    """
    h, w = binary_ink.shape[:2]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))
    opened = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, kernel)
    proj = opened.sum(axis=1)

    max_p = proj.max()
    if max_p == 0:
        return [], 0.0, 80.0

    peaks, _ = find_peaks(proj, height=0.10 * max_p, distance=min_spacing)
    if len(peaks) == 0:
        return [], 0.0, 80.0

    ruled_lines: List[Dict[str, Any]] = []
    for idx, p in enumerate(peaks):
        y_min = max(0, int(p) - 15)
        y_max = min(h, int(p) + 16)
        pts = np.where(opened[y_min:y_max, :] == 1)
        if len(pts[0]) > 80:
            ys = pts[0] + y_min
            xs = pts[1]
            try:
                poly = np.polyfit(xs, ys, 1)
                slope = float(poly[0])
                intercept = float(poly[1])
                if abs(slope) > 0.05:
                    slope = 0.0
                    intercept = float(p)
            except Exception:
                slope = 0.0
                intercept = float(p)
        else:
            slope = 0.0
            intercept = float(p)

        y_center = float(slope * (w / 2.0) + intercept)
        ruled_lines.append({
            "line_index": idx,
            "slope": slope,
            "intercept": intercept,
            "y_center": y_center,
        })

    if len(ruled_lines) > 1:
        spacings = np.diff([line["y_center"] for line in ruled_lines])
        median_spacing_r = float(np.median(spacings))
    else:
        median_spacing_r = 80.0

    median_slope = float(np.median([line["slope"] for line in ruled_lines])) if ruled_lines else 0.0
    return ruled_lines, median_slope, median_spacing_r


def remove_rules_and_restore_strokes(
    binary_ink: np.ndarray,
    ruled_lines: List[Dict[str, Any]],
    median_spacing_r: float,
    rule_thickness: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Isolate clean ink layer for stroke analysis, masking rules while restoring crossing strokes.

    Returns:
        ink_clean: binary ink mask with rules removed and crossed strokes repaired
        rule_mask: binary mask of detected rule pixels
    """
    h, w = binary_ink.shape[:2]
    rule_mask = np.zeros((h, w), dtype=np.uint8)
    half_thick = max(1, rule_thickness // 2)
    x_all = np.arange(w)

    for line in ruled_lines:
        slope = line.get("slope", 0.0)
        intercept = line.get("intercept", line.get("y_center", 0.0))
        y_line = np.round(slope * x_all + intercept).astype(int)

        for dy in range(-half_thick, half_thick + 1):
            y_cur = np.clip(y_line + dy, 0, h - 1)
            rule_mask[y_cur, x_all] = 1

    # Also capture actual curved/bowed printed rule pixels using horizontal opening
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1))
    morph_rules = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, kernel_h)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(morph_rules)
    for i in range(1, n_lbl):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        # Only keep genuine ruled line segments (width >= 140px, far longer than any handwriting word)
        if cw >= 140:
            comp_mask = (lbls == i).astype(np.uint8)
            comp_dil = cv2.dilate(comp_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
            rule_mask |= comp_dil

    # Subtract horizontal rule pixels from ink
    ink_no_rules = np.where(rule_mask == 1, 0, binary_ink).astype(np.uint8)

    # Detect and remove long vertical margin lines (>= 120px tall)
    kernel_vert_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120))
    v_lines = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, kernel_vert_line)
    v_lines_dilated = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    ink_no_lines = np.where(v_lines_dilated == 1, 0, ink_no_rules).astype(np.uint8)

    # Stroke restoration where strokes cross rules:
    # Use vertical & slightly dilated morphological closing across the subtracted rule band
    # (3, 11) kernel bridges vertical and slanted strokes (up to 30 deg slant)
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    vert_closed = cv2.morphologyEx(ink_no_lines, cv2.MORPH_CLOSE, kernel_vert)
    restored_strokes = (vert_closed == 1) & (rule_mask == 1) & (binary_ink == 1)

    # Only restore if it connects to genuine stroke pixels above/below
    kernel_vert_stroke = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    is_vert = cv2.morphologyEx(ink_no_lines, cv2.MORPH_OPEN, kernel_vert_stroke)
    is_vert_dilated = cv2.dilate(is_vert, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11)))
    restored_strokes = restored_strokes & (is_vert_dilated == 1)

    ink_clean = np.where(restored_strokes == 1, 1, ink_no_lines).astype(np.uint8)

    return ink_clean, rule_mask


def process_ruled_lines(
    binary_ink: np.ndarray,
    grayscale_norm: np.ndarray,
    min_spacing: int = 35,
) -> Dict[str, Any]:
    """Complete rule detection, deskewing, and multi-layer extraction.

    Preserves ruled lines as a reference layer for alignment and baseline residual features,
    while generating a stroke-restored ink-only layer for kinematic and spatial feature extraction.

    Returns dict with:
        - ruled_lines: List of detected rule parameters
        - median_spacing_r: float
        - median_slope: float
        - skew_angle_deg: float
        - ink_with_rules: deskewed binary ink mask with rules intact (reference layer)
        - ink_clean: deskewed binary ink mask with rules masked and crossed strokes restored
        - grayscale_deskewed: deskewed grayscale image with rules intact (reference layer)
        - rule_mask: binary mask of ruled lines
    """
    skew_angle = estimate_skew_angle(binary_ink)

    # Deskew all layers by the median skew angle
    ink_deskewed = deskew_image(binary_ink, skew_angle)
    gray_deskewed = deskew_image(grayscale_norm, skew_angle)

    # Detect all ruled lines on deskewed ink
    lines_deskewed, _, r = detect_ruled_lines(ink_deskewed, min_spacing=min_spacing)

    # Create separate ink-only layer for stroke/skeleton analysis with crossing stroke restoration
    ink_clean, rule_mask = remove_rules_and_restore_strokes(
        ink_deskewed, lines_deskewed, median_spacing_r=r, rule_thickness=7
    )

    return {
        "ruled_lines": lines_deskewed,
        "median_spacing_r": r,
        "median_slope": float(np.tan(np.radians(skew_angle))),
        "skew_angle_deg": skew_angle,
        "ink_with_rules": ink_deskewed,
        "ink_clean": ink_clean,
        "grayscale_deskewed": gray_deskewed,
        "rule_mask": rule_mask,
    }

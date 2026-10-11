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
    grayscale_image: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove ruled lines while strictly preserving handwriting strokes and Shirorekha.

    Uses Vectorized Grayscale Line Modeling & Optical Density Subtraction:
    1. Cubic polynomial trajectory tracking y_smooth(x) fitted strictly across confirmed empty
       columns (where vertical stroke opening 1x10 has zero support). This eliminates trajectory
       kinking onto handwriting strokes (e.g. over Shirorekhas or letters like 'मे').
    2. Measures the printed rule's empirical intensity dip profile L(dy) in confirmed empty stretches.
    3. Subtracts the line absorption dip using Lambert-Beer optical density D_ink = max(0, D_tot - D_rule),
       lifting the printed rule back up to paper background level while preserving darker/wider
       handwriting strokes, letters, and Shirorekha headlines solid without hollowing or splitting.
    4. Applies Otsu thresholding on the line-subtracted grayscale image.
    5. Zeros residual sub-threshold line artifacts strictly in empty gaps and margins (+/- 4px).
    6. Removes vertical notebook margin lines and filters thin line filaments (h <= 3, w >= 4) and speckles.

    Returns:
        ink_clean: binary ink mask with rules removed and handwriting preserved intact
        rule_mask: binary mask of detected rule pixels that were removed
    """
    binary_ink = (binary_ink > 0).astype(np.uint8)
    h, w = binary_ink.shape[:2]
    x_all = np.arange(w)

    if grayscale_image is not None and grayscale_image.shape[:2] == (h, w):
        gray_clean = grayscale_image.astype(float).copy()

        # Vertical character support: genuine characters have vertical extent >= 10px
        vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        vert_support = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, vert_k)
        vert_has_stroke = (vert_support > 0)

        dy_range = np.arange(-5, 6)
        line_trajectories = []

        for line in ruled_lines:
            s = line.get("slope", 0.0)
            intercept = line.get("intercept", line.get("y_center", 0.0))
            y_guide = np.clip(np.round(s * x_all + intercept).astype(int), 0, h - 1)

            # Fast window check for confirmed empty columns (no vertical letter strokes)
            p = int(round(np.median(y_guide)))
            y1 = max(0, p - 8)
            y2 = min(h, p + 9)
            empty_mask = (vert_has_stroke[y1:y2, :].sum(axis=0) == 0)
            empty_cols = np.where(empty_mask)[0]

            if len(empty_cols) < 80:
                continue

            # Vectorized line minimum in empty columns
            sub_strip = grayscale_image[y1:y2, :]
            min_rows = y1 + np.argmin(sub_strip[:, empty_cols], axis=0)

            # Fit smooth line model (degree 3 captures lens barrel distortion across 2000px without kinking)
            poly = np.polyfit(empty_cols, min_rows, deg=3)
            y_smooth = np.clip(np.polyval(poly, x_all), 0, h - 1)
            line_trajectories.append((y_smooth, empty_cols))

            # Sample line profile in confirmed empty cols
            yc_empty = np.clip(np.round(y_smooth[empty_cols]).astype(int), 0, h - 1)
            profiles = []
            for dy in dy_range:
                y_dy = np.clip(yc_empty + dy, 0, h - 1)
                profiles.append(grayscale_image[y_dy, empty_cols])
            profiles = np.array(profiles)

            prof_median = np.median(profiles, axis=1)
            bg = np.median([prof_median[0], prof_median[-1]])
            dip = np.maximum(0.0, bg - prof_median)

            if np.max(dip) < 12.0:
                continue

            # Optical density of the rule (Lambert-Beer absorbance)
            I_rule = np.maximum(5.0, bg - dip)
            D_rule = -np.log(I_rule / bg)

            # Vectorized optical density subtraction across ALL columns
            yc_all = np.clip(np.round(y_smooth).astype(int), 0, h - 1)
            for dy_idx, dy in enumerate(dy_range):
                d_r = D_rule[dy_idx]
                if d_r <= 0.03:
                    continue
                y_dy = np.clip(yc_all + dy, 0, h - 1)
                val = gray_clean[y_dy, x_all]
                mask_darker = (val < bg)
                if np.any(mask_darker):
                    val_m = val[mask_darker]
                    D_tot = -np.log(np.maximum(1.0, val_m) / bg)
                    D_ink = np.maximum(0.0, D_tot - d_r)
                    gray_clean[y_dy[mask_darker], x_all[mask_darker]] = bg * np.exp(-D_ink)

        # Threshold clean grayscale with Otsu
        gray_clean_uint8 = np.clip(gray_clean, 0, 255).astype(np.uint8)
        _, ink_clean = cv2.threshold(gray_clean_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Fast vectorized zeroing in confirmed empty columns (+/- 4px around smooth line)
        for y_smooth, empty_cols in line_trajectories:
            yc_empty = np.clip(np.round(y_smooth[empty_cols]).astype(int), 0, h - 1)
            for dy in range(-4, 5):
                y_dy = np.clip(yc_empty + dy, 0, h - 1)
                ink_clean[y_dy, empty_cols] = 0

    else:
        # Fallback when grayscale is not provided: 1D vertical run length check
        rule_mask_fallback = np.zeros((h, w), dtype=np.uint8)
        for line in ruled_lines:
            slope = line["slope"]
            intercept = line["intercept"]
            y_line = np.clip(np.round(slope * x_all + intercept).astype(int), 0, h - 1)

            active_x = np.where(binary_ink[y_line, x_all] == 1)[0]
            for x in active_x:
                yl = y_line[x]
                yt = yl
                while yt >= 0 and binary_ink[yt, x] == 1:
                    yt -= 1
                yb = yl
                while yb < h and binary_ink[yb, x] == 1:
                    yb += 1

                run_total = yb - yt - 1
                dist_below = yb - 1 - yl

                if dist_below >= 8:
                    continue
                if run_total >= 5:
                    continue
                if run_total <= 4:
                    rule_mask_fallback[yt + 1 : yb, x] = 1

        ink_clean = np.where(rule_mask_fallback == 1, 0, binary_ink).astype(np.uint8)

    # Detect and remove long vertical margin lines (>= 120px tall)
    kernel_vert_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120))
    v_lines = cv2.morphologyEx(binary_ink, cv2.MORPH_OPEN, kernel_vert_line)
    v_lines_dilated = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    ink_clean = np.where(v_lines_dilated == 1, 0, ink_clean).astype(np.uint8)

    # Remove thin filament fragments (h <= 3, w >= 4) and tiny speckles (area <= 6)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_clean)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        cw = stats[lbl, cv2.CC_STAT_WIDTH]
        ch = stats[lbl, cv2.CC_STAT_HEIGHT]
        if (ch <= 3 and cw >= 4) or area <= 6 or (ch <= 2 and cw >= 2):
            ink_clean[labels == lbl] = 0

    rule_mask = np.where((binary_ink > 0) & (ink_clean == 0), 1, 0).astype(np.uint8)
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

    # Remove rules using Grayscale Line Modeling & Profile Subtraction
    ink_clean, rule_mask = remove_rules_and_restore_strokes(
        ink_deskewed,
        lines_deskewed,
        median_spacing_r=r,
        rule_thickness=rule_thickness,
        grayscale_image=gray_deskewed,
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


"""Page preprocessing and normalization module for Workstream A.

Implements:
1. Auto-orientation to portrait.
2. Quadrilateral contour detection and perspective warping to fixed 2000px width (with margin fallback).
3. Clean grayscale extraction without distortion or illumination flattening.
4. Clean Otsu binarization on slightly blurred grayscale to produce a solid binary ink mask.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


def auto_orient_portrait(image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Ensure image is in portrait orientation and right-side up.

    Evaluates candidate rotations (0°, 90° CW, 180°, 270° CW) using:
    1. Vertical red margin line position (must be on the LEFT of the page).
    2. Handwriting ink & header density (concentrated in TOP 40% vs bottom 40%).

    Returns:
        oriented_image, was_rotated
    """
    h, w = image_bgr.shape[:2]
    if w > h:
        candidates = [
            ("90_cw", cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), True),
            ("270_cw", cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), True),
        ]
    else:
        candidates = [
            ("0_deg", image_bgr, False),
            ("180_deg", cv2.rotate(image_bgr, cv2.ROTATE_180), True),
        ]

    best_cand = candidates[0]
    best_score = -1e9

    for name, rot, is_rot in candidates:
        rh, rw = rot.shape[:2]
        # 1. Red margin line detection
        hsv = cv2.cvtColor(rot, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 35, 35]), np.array([180, 255, 255]))
        red = mask1 | mask2

        # Vertical projection of red ink in left 25% vs right 25%
        v_left = float(red[:, : int(rw * 0.25)].sum(axis=0).max()) if rw > 0 else 0.0
        v_right = float(red[:, int(rw * 0.75) :].sum(axis=0).max()) if rw > 0 else 0.0

        # 2. Handwriting ink distribution (middle column, top 40% vs bottom 40%)
        gray_small = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        scale_s = 600.0 / max(rh, rw)
        small_g = cv2.resize(gray_small, (int(rw * scale_s), int(rh * scale_s)), interpolation=cv2.INTER_AREA)
        adapt_s = cv2.adaptiveThreshold(
            small_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
        )
        sh, sw = adapt_s.shape
        mid_strip = adapt_s[:, int(sw * 0.20) : int(sw * 0.85)]
        top_ink = float(mid_strip[: int(sh * 0.40), :].sum())
        bot_ink = float(mid_strip[int(sh * 0.60) :, :].sum())

        score = 0.0
        if v_left > 0 or v_right > 0:
            if v_left >= v_right:
                score += 15.0 * (v_left / max(v_right, 1.0))
            else:
                score -= 15.0 * (v_right / max(v_left, 1.0))

        if top_ink >= bot_ink:
            score += 5.0 * (top_ink / max(bot_ink, 1.0))
        else:
            score -= 5.0 * (bot_ink / max(top_ink, 1.0))

        if score > best_score:
            best_score = score
            best_cand = (name, rot, is_rot)

    return best_cand[1], best_cand[2]


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest diff
    return rect


def detect_page_quad(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Detect 4-point page contour. Returns ordered corners in original image space, or None."""
    orig_h, orig_w = image_bgr.shape[:2]
    # Downscale for fast and robust edge detection
    scale = 800.0 / max(orig_h, orig_w)
    small_w = int(orig_w * scale)
    small_h = int(orig_h * scale)
    small = cv2.resize(image_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 120)

    # Dilate slightly to close edge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = small_w * small_h

    for c in contours[:5]:
        area = cv2.contourArea(c)
        if area < 0.35 * img_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32") / scale
            return order_quad_points(pts)

    return None


def warp_page(
    image_bgr: np.ndarray,
    quad_corners: Optional[np.ndarray],
    target_width: int = 2000,
    fallback_margin_pct: float = 0.02,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Perspective warp quadrilateral to target_width, or apply margin crop fallback.

    Uses high-fidelity interpolation (INTER_CUBIC / INTER_LANCZOS4) to preserve ink stroke crispness.

    Returns:
        warped_image (BGR), ordered_quad_corners
    """
    orig_h, orig_w = image_bgr.shape[:2]

    if quad_corners is not None:
        tl, tr, br, bl = quad_corners
        width_top = np.linalg.norm(tr - tl)
        width_bot = np.linalg.norm(br - bl)
        avg_w = (width_top + width_bot) / 2.0

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        avg_h = (height_left + height_right) / 2.0

        target_height = int(target_width * (avg_h / max(avg_w, 1.0)))
        # Clamp height to reasonable portrait limits
        target_height = max(int(target_width * 1.1), min(target_height, int(target_width * 1.8)))

        dst = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(quad_corners, dst)
        warped = cv2.warpPerspective(image_bgr, matrix, (target_width, target_height), flags=cv2.INTER_CUBIC)
        return warped, quad_corners

    # Fallback: Trim 2% border and resize with high-fidelity interpolation
    m_x = int(orig_w * fallback_margin_pct)
    m_y = int(orig_h * fallback_margin_pct)
    cropped = image_bgr[m_y : orig_h - m_y, m_x : orig_w - m_x]

    crop_h, crop_w = cropped.shape[:2]
    target_height = int(crop_h * (target_width / crop_w))
    interp = cv2.INTER_LANCZOS4 if target_width >= crop_w else cv2.INTER_AREA
    warped = cv2.resize(cropped, (target_width, target_height), interpolation=interp)

    return warped, None


def binarize_image(grayscale: np.ndarray, block_size: int = 51, c: int = 15) -> np.ndarray:
    """Binarize grayscale image to extract ink mask (1 = ink, 0 = paper).

    Uses local adaptive Gaussian thresholding, robust to phone camera lighting
    gradients and ambient paper shadows without generating false ink.
    """
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c
    )
    binary_ink = (adapt > 0).astype(np.uint8)
    return binary_ink


def preprocess_page(
    image_bgr: np.ndarray,
    target_width: int = 2000,
    **kwargs,
) -> dict:
    """Complete page preprocessing pipeline.

    Returns dict with:
        - grayscale: original clean grayscale image (H, W) uint8
        - grayscale_norm: alias for grayscale
        - binary_ink: binary ink mask (H, W) uint8 (1=ink, 0=bg)
        - warped_bgr: warped BGR image (H, W, 3) uint8
        - quad_corners: list of 4 ordered corners or None
        - was_rotated: bool
    """
    oriented, was_rotated = auto_orient_portrait(image_bgr)
    quad_corners = detect_page_quad(oriented)
    warped, quad_pts = warp_page(oriented, quad_corners, target_width=target_width)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    binary_ink = binarize_image(gray)

    return {
        "grayscale": gray,
        "grayscale_norm": gray,
        "binary_ink": binary_ink,
        "warped_bgr": warped,
        "quad_corners": quad_pts.tolist() if quad_pts is not None else None,
        "was_rotated": was_rotated,
    }

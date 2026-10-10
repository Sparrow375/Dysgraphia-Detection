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

    If landscape (w > h), determines whether top of page is on the right or left
    by counting ruled-line density (the notebook footer has dense ruled lines,
    while the header box region has few/none). Rotates CCW or CW accordingly.

    Returns:
        oriented_image, was_rotated
    """
    h, w = image_bgr.shape[:2]
    if w > h:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
        edges_left = cv2.morphologyEx(cv2.Canny(gray[:, :int(w * 0.4)], 50, 150), cv2.MORPH_OPEN, kh).sum()
        edges_right = cv2.morphologyEx(cv2.Canny(gray[:, int(w * 0.6):], 50, 150), cv2.MORPH_OPEN, kh).sum()

        if edges_left > edges_right:
            # Dense ruled lines on left => footer is left => header is right => rotate CCW
            rotated = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Dense ruled lines on right => footer is right => header is left => rotate CW
            rotated = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        return rotated, True
    return image_bgr, False



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
        warped = cv2.warpPerspective(image_bgr, matrix, (target_width, target_height), flags=cv2.INTER_LINEAR)
        return warped, quad_corners

    # Fallback: Trim 2% border and resize to target_width
    m_x = int(orig_w * fallback_margin_pct)
    m_y = int(orig_h * fallback_margin_pct)
    cropped = image_bgr[m_y : orig_h - m_y, m_x : orig_w - m_x]

    crop_h, crop_w = cropped.shape[:2]
    target_height = int(crop_h * (target_width / crop_w))
    warped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return warped, None


def normalize_background(grayscale: np.ndarray) -> np.ndarray:
    """Normalize illumination gradients (shadows from phone cameras) without losing stroke sharpness.

    Estimates the paper background surface via large-kernel morphological dilation
    and median filtering, then normalizes the grayscale image to a consistent white paper baseline.
    """
    bg = cv2.morphologyEx(grayscale, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    bg = cv2.medianBlur(bg, 21)
    norm = np.clip((grayscale.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0, 0, 255).astype(np.uint8)
    return norm


def binarize_image(grayscale: np.ndarray) -> np.ndarray:
    """Binarize grayscale image to extract ink mask (1 = ink, 0 = paper).

    Uses Gaussian blur + Otsu thresholding, avoiding paper texture artifacts.
    """
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_ink = (otsu_inv > 0).astype(np.uint8)
    return binary_ink


def preprocess_page(
    image_bgr: np.ndarray,
    target_width: int = 2000,
    **kwargs,
) -> dict:
    """Complete page preprocessing pipeline.

    Returns dict with:
        - grayscale: original clean grayscale image (H, W) uint8
        - grayscale_norm: illumination-normalized grayscale image (H, W) uint8
        - binary_ink: binary ink mask (H, W) uint8 (1=ink, 0=bg)
        - warped_bgr: warped BGR image (H, W, 3) uint8
        - quad_corners: list of 4 ordered corners or None
        - was_rotated: bool
    """
    oriented, was_rotated = auto_orient_portrait(image_bgr)
    quad_corners = detect_page_quad(oriented)
    warped, quad_pts = warp_page(oriented, quad_corners, target_width=target_width)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray_norm = normalize_background(gray)
    binary_ink = binarize_image(gray_norm)

    return {
        "grayscale": gray_norm,
        "grayscale_norm": gray_norm,
        "binary_ink": binary_ink,
        "warped_bgr": warped,
        "quad_corners": quad_pts.tolist() if quad_pts is not None else None,
        "was_rotated": was_rotated,
    }


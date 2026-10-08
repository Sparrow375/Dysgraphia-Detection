"""Page preprocessing and normalization module for Workstream A.

Implements:
1. Auto-orientation to portrait.
2. Quadrilateral contour detection and perspective warping to fixed 2000px width (with margin fallback).
3. Background illumination flattening (division by large-kernel background estimate).
4. Sauvola binarization (W=31, k=0.2) producing dual grayscale and binary ink layers.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def auto_orient_portrait(image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Ensure image is in portrait orientation (height >= width).

    Returns:
        oriented_image, was_rotated
    """
    h, w = image_bgr.shape[:2]
    if w > h:
        # Rotate 90 degrees clockwise to make portrait
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


def flatten_illumination(grayscale: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """Flatten uneven lighting and shadows by dividing by large-kernel background estimate."""
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Large morphological closing estimates background paper illumination
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(grayscale, cv2.MORPH_CLOSE, kernel)

    # Avoid zero division
    bg_float = np.maximum(background.astype(np.float32), 1.0)
    gray_float = grayscale.astype(np.float32)

    # Normalized division scaled to [0, 255]
    flattened = np.clip((gray_float / bg_float) * 255.0, 0, 255).astype(np.uint8)
    return flattened


def binarize_sauvola(
    grayscale_norm: np.ndarray,
    window_size: int = 31,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    """Sauvola adaptive thresholding. Returns binary ink mask (1 = ink, 0 = paper)."""
    if window_size % 2 == 0:
        window_size += 1

    thresh = threshold_sauvola(grayscale_norm, window_size=window_size, k=k, r=r)
    # Ink pixels are darker than local threshold
    binary_ink = (grayscale_norm < thresh).astype(np.uint8)
    return binary_ink


def preprocess_page(
    image_bgr: np.ndarray,
    target_width: int = 2000,
    sauvola_window: int = 31,
    sauvola_k: float = 0.2,
) -> dict:
    """Complete page preprocessing pipeline.

    Returns dict with:
        - grayscale_norm: normalized grayscale image (H, W) uint8
        - binary_ink: binary ink mask (H, W) uint8 (1=ink, 0=bg)
        - quad_corners: list of 4 ordered corners or None
        - was_rotated: bool
    """
    oriented, was_rotated = auto_orient_portrait(image_bgr)
    quad_corners = detect_page_quad(oriented)
    warped, quad_pts = warp_page(oriented, quad_corners, target_width=target_width)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray_norm = flatten_illumination(gray, kernel_size=51)
    binary_ink = binarize_sauvola(gray_norm, window_size=sauvola_window, k=sauvola_k)

    return {
        "grayscale_norm": gray_norm,
        "binary_ink": binary_ink,
        "warped_bgr": warped,
        "quad_corners": quad_pts.tolist() if quad_pts is not None else None,
        "was_rotated": was_rotated,
    }

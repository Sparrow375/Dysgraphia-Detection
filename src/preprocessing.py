"""
Image Preprocessing Module for Dysgraphia Detection — Production Universal Pipeline.
Handles:
1. Landscape/portrait auto-orientation
2. Illumination gradient normalization (<30ms downsampled background division)
3. Polarity auto-detection (white-on-black photocopies vs black-on-white paper)
4. Red chroma margin line and header box suppression
5. Printed ruling line subtraction with descender healing (g, j, p, q, y)
6. Active notebook boundary cropping (drops empty paper at bottom)
7. Microscopic sensor speckle and border shadow removal
"""

from __future__ import annotations
import logging
from typing import Tuple, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def detect_text_line_orientation(img: np.ndarray) -> str:
    """
    Detects whether text lines are horizontal or vertical using projection profile energy.
    In horizontal writing, row projection (sum across columns) has sharp peaks and valleys,
    so its normalized standard deviation is significantly higher than column projection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = 500.0 / max(gray.shape) if max(gray.shape) > 500 else 1.0
    small = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_prof = np.sum(binary == 255, axis=1).astype(float)
    col_prof = np.sum(binary == 255, axis=0).astype(float)

    row_energy = np.std(row_prof) / (np.mean(row_prof) + 1e-5)
    col_energy = np.std(col_prof) / (np.mean(col_prof) + 1e-5)

    # If column energy is distinctly higher than row energy, text lines are running vertically
    if col_energy > 1.35 * row_energy:
        return "vertical"
    return "horizontal"


def orient_handwriting_image(
    img: np.ndarray,
    rotation: Optional[Union[int, str]] = None,
    auto_orient: bool = False,
) -> Tuple[np.ndarray, bool, int]:
    """
    Corrects handwriting image orientation without blindly rotating wide landscape pages.
    Wide notebook spreads, single-sentence crops, and exams are naturally horizontal (w > h).

    Args:
        img: Input BGR or Grayscale image.
        rotation: Optional manual rotation angle: 0, 90, 180, 270 (or descriptive strings).
        auto_orient: If True, uses projection energy to rotate vertical text lines to horizontal.

    Returns:
        (oriented_img, was_rotated, rotation_angle)
    """
    # 1. Manual rotation overrides
    if rotation is not None:
        rot_str = str(rotation).strip().lower()
        if rot_str in ["90", "90°", "rotate 90° clockwise", "rotate 90°", "90_cw", "90 deg"]:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True, 90
        elif rot_str in ["180", "180°", "rotate 180°", "180 deg"]:
            return cv2.rotate(img, cv2.ROTATE_180), True, 180
        elif rot_str in ["270", "270°", "rotate 270° counter-clockwise", "rotate 270°", "-90", "90_ccw", "270 deg"]:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), True, 270
        elif rot_str in ["0", "0°", "no rotation (0°)", "none", "false", "0 deg", ""]:
            return img, False, 0
        elif rot_str in ["auto", "auto-detect lines", "auto-detect"]:
            auto_orient = True

    # 2. Projection profile energy auto-orientation (only rotates if text is verified vertical)
    if auto_orient:
        text_dir = detect_text_line_orientation(img)
        if text_dir == "vertical":
            logger.info("Auto-orientation: detected vertical text lines. Rotating 90° clockwise to horizontal.")
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True, 90

    return img, False, 0


def preprocess_handwriting_image(
    img_input: Union[str, Path, np.ndarray],
    rotation: Optional[Union[int, str]] = None,
    auto_orient: bool = False,
    return_metadata: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Robust universal preprocessing pipeline for any handwritten paper input:
    - Scanned or photographed school notebook sheets
    - Unruled white paper photos taken on smartphones
    - Pre-inverted binary datasets (white text on dark background)

    Args:
        img_input: File path (str/Path) or numpy array (BGR/RGB or Grayscale).
        rotation: Manual rotation override (0, 90, 180, 270).
        auto_orient: If True, uses text-line energy detection before processing.
        return_metadata: If True, returns additional pipeline metadata dict.

    Returns:
        (binary_text_mask, preprocessed_vis_img) or (binary_text_mask, preprocessed_vis_img, metadata)
    """
    if isinstance(img_input, (str, Path)):
        p = Path(img_input)
        if not p.exists():
            raise FileNotFoundError(f"Could not read image from path: {img_input}")
        try:
            from PIL import Image, ImageOps
            pil_img = Image.open(str(p))
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode == "RGB":
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            elif pil_img.mode == "RGBA":
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
            else:
                img = np.array(pil_img)
        except Exception:
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(f"Could not read image from path: {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
    else:
        img = np.array(img_input)

    # 1. Orientation handling (NEVER blindly rotates simply because w > h)
    img, rotated, rot_angle = orient_handwriting_image(img, rotation=rotation, auto_orient=auto_orient)
    h, w = img.shape[:2]

    # 2. Extract Color Channels & Grayscale
    has_color = False
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        b, g, r = cv2.split(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        has_color = True
    else:
        gray = img.copy()

    median_val = float(np.median(gray))
    crop_box = (0, 0, w, h)

    if median_val > 60:
        # -------------------------------------------------------------------
        # Light Paper (Photos / Scans with Dark Ink)
        # -------------------------------------------------------------------
        # A. Fast downsampled illumination normalization (handles extreme shadows in <30ms)
        scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
        small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ksize = 31
        small_bg = cv2.GaussianBlur(small_gray, (ksize, ksize), 0)
        bg = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)
        norm = cv2.divide(gray, bg, scale=255)

        # B. Binarization of student ink
        inv = 255 - norm
        _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # C. Red Chroma Teacher Margin / Header Box Suppression
        if has_color:
            is_red = (r.astype(int) - g.astype(int) > 25) & (r.astype(int) - b.astype(int) > 25)
            binary[is_red] = 0

        # D. Clean Page Edge Borders (removes dark framing shadows from phone sensors)
        border_px = max(2, int(min(h, w) * 0.012))
        binary[:border_px, :] = 0
        binary[-border_px:, :] = 0
        binary[:, :border_px] = 0
        binary[:, -border_px:] = 0

        # E. Printed Horizontal Ruling Line Removal with Descender Preservation
        line_min_w = max(24, int(w * 0.04))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_w, 1))
        horizontal_ruling = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        if np.count_nonzero(horizontal_ruling) > 0:
            dilated_ruling = cv2.dilate(horizontal_ruling, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
            binary = cv2.subtract(binary, dilated_ruling)
            # Reconnect severed letter descenders ('g', 'j', 'p', 'q', 'y')
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))

        # F. Active Handwriting Zone Boundary Cropping (Drop empty bottom of notebook)
        row_sum = np.sum(binary, axis=1) / 255.0
        valid_rows = np.where(row_sum > 8)[0]
        if len(valid_rows) > 0:
            top_y = max(0, valid_rows[0] - 20)
            bot_y = min(h, valid_rows[-1] + 30)
        else:
            top_y, bot_y = 0, h

        col_sum = np.sum(binary[top_y:bot_y, :], axis=0) / 255.0
        valid_cols = np.where(col_sum > 5)[0]
        # Skip extreme margin lines if any remain on left
        content_cols = [c for c in valid_cols if c > int(w * 0.05)] if len(valid_cols) > 0 else []
        if content_cols:
            left_x = max(0, content_cols[0] - 20)
            right_x = min(w, content_cols[-1] + 20)
        else:
            left_x, right_x = 0, w

        clean_mask = binary[top_y:bot_y, left_x:right_x]
        clean_color = img[top_y:bot_y, left_x:right_x]
        crop_box = (left_x, top_y, right_x - left_x, bot_y - top_y)
    else:
        # -------------------------------------------------------------------
        # Dark Background / Pre-Inverted Images (Malay / Slovak Datasets)
        # -------------------------------------------------------------------
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, clean_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        clean_color = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)

        # Remove horizontal guide lines if present
        line_min_w = max(30, int(w * 0.25))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_w, 1))
        horizontal_lines = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        if np.count_nonzero(horizontal_lines) > 0:
            dilated = cv2.dilate(horizontal_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
            clean_mask = cv2.subtract(clean_mask, dilated)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    # 3. Filter Microscopic Noise Specks (< 12 pixels or page-spanning line remnants)
    nh, nw = clean_mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    filtered_mask = np.zeros_like(clean_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 12:
            continue
        # Drop ruling line remnants spanning > 80% width with tiny height
        if cw > 0.85 * nw and ch < 12:
            continue
        # Drop thin horizontal ruling line fragments from lined paper
        if ch <= 4 and cw >= 15:
            continue
        if ch <= 3 and cw >= 8:
            continue
        filtered_mask[labels == i] = 255

    vis_img = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2RGB)

    if return_metadata:
        metadata = {
            "rotated": rotated,
            "rotation_angle": rot_angle,
            "has_color": has_color,
            "crop_box": crop_box,
            "clean_color": clean_color,
            "ink_pixels": int(np.count_nonzero(filtered_mask)),
        }
        return filtered_mask, vis_img, metadata

    return filtered_mask, vis_img


def remove_guide_lines(binary_img: np.ndarray) -> np.ndarray:
    """Convenience alias for guide line removal."""
    h, w = binary_img.shape
    line_min_width = max(30, int(w * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    if np.sum(horizontal_lines > 0) > 0:
        dilated = cv2.dilate(horizontal_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
        binary_img = cv2.bitwise_and(binary_img, cv2.bitwise_not(dilated))
    return binary_img

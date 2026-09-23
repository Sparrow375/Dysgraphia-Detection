"""
Image Preprocessing Module for Dysgraphia Detection
Handles illumination normalization, polarity auto-detection (white-on-black vs black-on-white),
binarization, and baseline / guide line removal.
"""

import numpy as np
import cv2


def preprocess_handwriting_image(img_input) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust preprocessing pipeline for both scanned/photographed paper (black ink on light paper)
    and pre-inverted datasets (white ink on black background).
    
    Args:
        img_input: File path (str), numpy array (BGR/RGB or Grayscale).
        
    Returns:
        tuple (binary_text_mask, preprocessed_vis_img)
        - binary_text_mask: uint8 array where ink = 255, background = 0
        - preprocessed_vis_img: RGB image suitable for visualization
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
    else:
        img = np.array(img_input)
        
    # Convert to grayscale
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape

    # Polarity & Illumination normalization:
    # In any handwriting document, background represents > 75% of the surface area.
    median_val = float(np.median(gray))

    if median_val > 60:
        # Document photograph or scan on light paper (often has uneven illumination/shadows)
        # Background division to normalize lighting gradients:
        blur_kernel_size = max(21, (int(min(h, w) * 0.15) // 2) * 2 + 1)
        bg = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
        norm = cv2.divide(gray, bg, scale=255)
        
        # Invert so ink is white on black background
        inv = 255 - norm
        _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Already pre-inverted image (like the Malay dataset: white text on dark background)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean borders (removes dark edge shadows from camera sensors or framing borders)
    border_px = max(2, int(min(h, w) * 0.015))
    binary[:border_px, :] = 0
    binary[-border_px:, :] = 0
    binary[:, :border_px] = 0
    binary[:, -border_px:] = 0

    # Guide line removal: detect horizontal ruling lines spanning a large portion of the width
    line_min_width = max(30, int(w * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    if np.sum(horizontal_lines > 0) > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated_lines = cv2.dilate(horizontal_lines, dilate_kernel, iterations=1)
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(dilated_lines))
        # Bridge any characters intersected by the line
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # Filter spurious noise: components < 12 pixels or full-width remnants
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        comp_area = stats[i, cv2.CC_STAT_AREA]
        
        if comp_area < 12:
            continue
        if comp_w > 0.80 * w and comp_h < 15:
            continue
        clean_mask[labels == i] = 255

    vis_img = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB)
    return clean_mask, vis_img


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

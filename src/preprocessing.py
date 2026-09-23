"""
Image Preprocessing Module for Dysgraphia Detection
Handles polarity detection (white-on-black vs black-on-white), binarization,
noise reduction, and baseline / guide line removal.
"""

import numpy as np
import cv2


def detect_and_normalize_polarity(gray_img: np.ndarray) -> np.ndarray:
    """
    Detects whether the image has dark text on light background (e.g. paper scan)
    or light text on dark background (e.g. pre-inverted dataset images).
    Normalizes so that FOREGROUND INK is 255 (white) and BACKGROUND is 0 (black).
    """
    # Sample border pixels to estimate background intensity
    h, w = gray_img.shape
    border_pixels = np.concatenate([
        gray_img[0, :],
        gray_img[-1, :],
        gray_img[:, 0],
        gray_img[:, -1]
    ])
    median_border = np.median(border_pixels)
    
    # If border is bright (> 127), background is light and ink is dark
    if median_border > 127:
        # Invert so ink becomes white (255) on black background (0)
        norm_img = 255 - gray_img
    else:
        # Background is already dark, ink is light
        norm_img = gray_img.copy()
        
    return norm_img


def binarize_image(norm_img: np.ndarray) -> np.ndarray:
    """
    Binarizes normalized grayscale image (where ink is bright) using adaptive thresholding
    with Otsu fallback.
    """
    # Slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(norm_img, (3, 3), 0)
    
    # Check dynamic range
    if np.max(blurred) - np.min(blurred) < 20:
        # Low contrast image
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    else:
        # Otsu thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    return binary


def remove_guide_lines(binary_img: np.ndarray) -> np.ndarray:
    """
    Detects and suppresses long horizontal lines (such as notebook guide lines
    or ruling lines) that can distort character contour and connected component analysis.
    """
    h, w = binary_img.shape
    cleaned = binary_img.copy()
    
    # Find horizontal lines using a morphological kernel
    line_min_width = max(30, int(w * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    
    # If substantial horizontal lines detected, subtract them gently
    if np.sum(horizontal_lines > 0) > 0:
        # Dilate line slightly in height to cover line width
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated_lines = cv2.dilate(horizontal_lines, dilate_kernel, iterations=1)
        
        # Remove guide lines
        cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(dilated_lines))
        
        # Morphological close to bridge any characters intersected by the line
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
        
    return cleaned


def filter_spurious_components(binary_img: np.ndarray) -> np.ndarray:
    """
    Filters out extreme noise: components that are too tiny (< 5 pixels)
    or span across the entire image width (> 85% width) which are residual lines.
    """
    h, w = binary_img.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    clean_mask = np.zeros_like(binary_img)
    
    for i in range(1, num_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        comp_area = stats[i, cv2.CC_STAT_AREA]
        
        # Discard tiny speckles
        if comp_area < 8:
            continue
            
        # Discard full-width line remnants
        if comp_w > 0.80 * w and comp_h < 15:
            continue
            
        clean_mask[labels == i] = 255
        
    return clean_mask


def preprocess_handwriting_image(img_input) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline.
    
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
        # Convert PIL if passed
        img = np.array(img_input)
        
    # Convert to grayscale
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # 1. Normalize polarity (so ink is 255, background is 0)
    norm = detect_and_normalize_polarity(gray)
    
    # 2. Binarize
    binary = binarize_image(norm)
    
    # 3. Guide line removal
    no_lines = remove_guide_lines(binary)
    
    # 4. Filter spurious noise
    clean_mask = filter_spurious_components(no_lines)
    
    # Create an aesthetically pleasing RGB visualization
    vis_img = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB)
    
    return clean_mask, vis_img

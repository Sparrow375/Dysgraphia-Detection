"""
Robust Preprocessing Pipeline for Unconstrained Handwriting Ingestion.
=====================================================================
Handles real-world camera artifacts:
  1. EXIF orientation transposition (smartphone camera rotation)
  2. Severe non-uniform shadows and lighting normalization
  3. Blur estimation via Laplacian variance
  4. Deskew estimation via projection profile variance / moments
  5. Resolution and ink-density sanity checks
"""

from typing import Union, Tuple, Dict, Any, Optional
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, laplace, rotate


def apply_exif_orientation(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """
    Loads image and applies EXIF orientation transpose if present.
    Ensures camera photos taken vertically or inverted are oriented correctly.
    """
    if isinstance(image_input, (str, Path)):
        img_pil = Image.open(str(image_input))
    elif isinstance(image_input, Image.Image):
        img_pil = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            img_pil = Image.fromarray(image_input.astype(np.uint8))
        else:
            img_pil = Image.fromarray(image_input.astype(np.uint8), mode="L")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    try:
        transposed = ImageOps.exif_transpose(img_pil)
        return transposed if transposed is not None else img_pil
    except Exception:
        return img_pil


def estimate_blur_score(gray: np.ndarray) -> Tuple[float, bool]:
    """
    Measures image blur using the variance of the discrete 2D Laplacian operator.
    Lower values indicate soft focus or motion blur.
    Returns (variance, is_blurry).
    """
    if gray.size == 0:
        return 0.0, True
    try:
        val = float(laplace(gray.astype(np.float32)).var())
        is_blurry = val < 75.0
        return round(val, 2), is_blurry
    except Exception:
        return 100.0, False


def estimate_skew_angle(binary: np.ndarray, max_angle_deg: float = 12.0, num_steps: int = 49) -> float:
    """
    Estimates horizontal baseline skew by searching for the rotation angle
    that maximizes the variance of horizontal projection profile (Radon-like).
    """
    h, w = binary.shape
    if h < 20 or w < 20 or np.sum(binary) < 50:
        return 0.0

    # Downsample large images for rapid search
    scale = min(1.0, 600.0 / max(h, w))
    if scale < 1.0:
        from scipy.ndimage import zoom
        small = zoom(binary.astype(np.float32), scale, order=0) > 0.5
    else:
        small = binary

    angles = np.linspace(-max_angle_deg, max_angle_deg, num_steps)
    best_score = -1.0
    best_angle = 0.0

    for ang in angles:
        if abs(ang) < 0.1:
            rotated = small
        else:
            rotated = rotate(small.astype(np.float32), ang, reshape=False, order=0) > 0.5
        # Horizontal projection profile: sum along width
        proj = np.sum(rotated, axis=1)
        score = float(np.var(proj))
        if score > best_score:
            best_score = score
            best_angle = float(ang)

    return round(best_angle, 2)


def deskew_image(binary: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates binary mask by angle_deg to correct slant/skew.
    """
    if abs(angle_deg) < 0.5:
        return binary
    rotated = rotate(binary.astype(np.float32), angle_deg, reshape=False, order=0) > 0.5
    return rotated.astype(np.uint8)


def normalize_and_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Removes severe shadows and lighting gradients via adaptive background division,
    followed by Otsu thresholding on the background-subtracted difference map.
    """
    h, w = gray.shape
    # Local background estimation via Gaussian filter
    sigma = max(h, w) / 28.0
    bg = gaussian_filter(gray, sigma=sigma)

    # Auto-detect inverted polarity (light ink on dark background)
    bg_mean = float(bg.mean())
    is_inverted = bg_mean < 85.0

    if is_inverted:
        diff = gray - bg          # bright ink on dark bg
    else:
        diff = bg - gray          # dark ink on light bg  (standard)

    diff_clipped = np.clip(diff, 0, 255)

    # Otsu thresholding on difference map
    hist, _ = np.histogram(diff_clipped, bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)

    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sb = (mu_t * omega - mu)**2 / denom
    best_thresh = np.nanargmax(sb) if not np.all(np.isnan(sb)) else 20.0

    min_thresh = 8.0 if is_inverted else 14.0
    thresh = max(float(best_thresh), min_thresh)
    binary = (diff_clipped >= thresh).astype(np.uint8)

    # Clean isolated speckles: remove components with < 5 pixels
    from scipy.ndimage import label, sum_labels
    labeled, num_features = label(binary)
    if num_features > 0:
        sizes = sum_labels(binary, labeled, range(1, num_features + 1))
        keep = np.zeros(num_features + 1, dtype=bool)
        keep[1:] = sizes >= 5
        binary = keep[labeled].astype(np.uint8)

    return binary


def robust_load_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    auto_deskew: bool = True,
    max_deskew_angle: float = 12.0,
) -> Dict[str, Any]:
    """
    Top-level robust loader for unconstrained handwriting photos.
    Returns:
      binary_mask: np.ndarray (uint8, ink=1, bg=0)
      raw_pil: PIL.Image
      gray_image: np.ndarray (float32)
      quality_flags: dict with low_ink, low_resolution, blur, skew_corrected
      skew_angle_deg: float
    """
    img_pil = apply_exif_orientation(image_input)
    gray_pil = img_pil.convert("L")
    gray = np.array(gray_pil, dtype=np.float32)
    h, w = gray.shape

    # 1. Blur check
    blur_score, is_blurry = estimate_blur_score(gray)

    # 2. Lighting normalization & binarization
    binary = normalize_and_binarize(gray)

    # 3. Deskew check
    skew_angle = 0.0
    skew_corrected = False
    if auto_deskew:
        angle = estimate_skew_angle(binary, max_angle_deg=max_deskew_angle)
        if abs(angle) >= 1.0:
            binary = deskew_image(binary, angle)
            skew_angle = angle
            skew_corrected = True

    ink_px = int(np.sum(binary))
    ink_density = ink_px / max(h * w, 1)

    # Resolution & ink checks
    low_res = (h < 250 or w < 250)
    low_ink = (ink_px < 60 or ink_density < 0.001)

    quality_flags = {
        "low_ink": low_ink,
        "low_resolution": low_res,
        "blur": is_blurry,
        "blur_score": blur_score,
        "skew_corrected": skew_corrected,
        "skew_angle_deg": skew_angle,
    }

    return {
        "binary_mask": binary,
        "raw_pil": img_pil,
        "gray_image": gray,
        "quality_flags": quality_flags,
        "dimensions": (w, h),
        "ink_pixels": ink_px,
    }

"""
Grayscale Photometric Pressure Modeling (Solution 3)
Extracts optical ink absorbance and combines it with EDT line thickness
to reconstruct physical stylus down-force from static images.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation


def compute_photometric_pressure(
    gray_image: np.ndarray,
    stroke_pts: np.ndarray,
    dist_map: np.ndarray,
    h_med: float = 25.0,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Computes calibrated optical down-force proxy along a stroke path.
    Combines:
    1. Beer-Lambert Optical Absorbance: A = -log10(I_ink / I_bg)
    2. Optical Stroke Width from distance transform: W = (2 * EDT) / H_med

    Args:
        gray_image: 2D uint8 or float grayscale image (0-255).
        stroke_pts: (M, 2) array of (x, y) stroke coordinates.
        dist_map: 2D float distance transform map.
        h_med: median character height in pixels.
        alpha: weight for optical absorbance.
        beta: weight for normalized stroke width.

    Returns:
        1D float64 array of estimated pressure values along the stroke.
    """
    if len(stroke_pts) == 0:
        return np.array([], dtype=np.float64)

    h, w = gray_image.shape
    gray_float = gray_image.astype(np.float64)

    # Estimate local background intensity via grayscale dilation
    bg_estimate = grey_dilation(gray_float, size=(15, 15))

    pressures = []
    for pt in stroke_pts:
        x, y = pt[0], pt[1]
        ix = int(np.clip(np.round(x), 0, w - 1))
        iy = int(np.clip(np.round(y), 0, h - 1))

        # Sample ink intensity and local background
        i_ink = float(gray_float[iy, ix])
        i_bg = max(float(bg_estimate[iy, ix]), 1.0)

        # Optical Absorbance (Beer-Lambert Law)
        # Higher pressure -> darker ink -> higher absorbance
        transmission = np.clip((i_ink + 1.0) / (i_bg + 1.0), 1e-4, 1.0)
        absorbance = -np.log10(transmission)

        # Normalized stroke width from distance map
        stroke_width_norm = (2.0 * float(dist_map[iy, ix])) / max(float(h_med), 1.0)

        # Bivariate combination
        p_est = alpha * absorbance + beta * stroke_width_norm
        pressures.append(p_est)

    p_arr = np.array(pressures, dtype=np.float64)
    # Slight 1D Gaussian smoothing along continuous stroke
    p_smooth = gaussian_filter(p_arr, sigma=1.0) if len(p_arr) > 3 else p_arr
    return np.maximum(p_smooth, 0.01)

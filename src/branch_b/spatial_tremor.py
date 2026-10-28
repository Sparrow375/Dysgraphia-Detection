"""
Dual-Band Spatial Tremor Decomposition (Solution 2)
Decomposes stroke trajectories into macro-geometry (letter shape)
and micro-residual ripple (motor tremor) to extract true 4-8 Hz tremor power.
"""

from typing import List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch


def decompose_stroke_spatial_residual(
    stroke_pts: np.ndarray,
    h_med: float = 25.0,
    resample_step_px: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes a 2D stroke into macro backbone and high-frequency spatial ripple.

    Args:
        stroke_pts: (M, 2) array of (x, y) coordinates.
        h_med: character height in pixels.
        resample_step_px: arc-length step for spatial resampling.

    Returns:
        s_grid: 1D array of cumulative arc-length coordinates.
        macro_pts: (N, 2) smoothed macro backbone.
        ripple_mag: 1D array of perpendicular displacement magnitudes (micro-residual).
    """
    if len(stroke_pts) < 4:
        return np.array([]), np.array([]), np.array([])

    # Compute cumulative arc-length
    diffs = np.diff(stroke_pts, axis=0)
    seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = s_cum[-1]

    if total_len < 3.0:
        return np.array([]), np.array([]), np.array([])

    # Resample to fine uniform spatial grid
    n_pts = max(int(np.round(total_len / resample_step_px)), 8)
    s_grid = np.linspace(0.0, total_len, n_pts)

    x_res = np.interp(s_grid, s_cum, stroke_pts[:, 0])
    y_res = np.interp(s_grid, s_cum, stroke_pts[:, 1])
    raw_res = np.column_stack([x_res, y_res])

    # Macro backbone via Gaussian filter with spatial scale sigma = 0.15 * H_med
    sigma_spatial = max(0.15 * float(h_med) / resample_step_px, 2.0)
    x_macro = gaussian_filter1d(x_res, sigma=sigma_spatial, mode='nearest')
    y_macro = gaussian_filter1d(y_res, sigma=sigma_spatial, mode='nearest')
    macro_pts = np.column_stack([x_macro, y_macro])

    # Micro-residual vector and magnitude (perpendicular ripple)
    residual_vec = raw_res - macro_pts
    ripple_mag = np.sqrt(residual_vec[:, 0]**2 + residual_vec[:, 1]**2)

    return s_grid, macro_pts, ripple_mag


def compute_calibrated_tremor_index(
    strokes: List[np.ndarray],
    h_med: float = 25.0,
    estimated_speed_px_s: float = 120.0,
) -> float:
    """
    Computes the relative Power Spectral Density (PSD) in the neuromuscular tremor band (4-8 Hz)
    using spatial micro-residuals mapped through the estimated writing velocity.

    Args:
        strokes: List of (M, 2) stroke arrays.
        h_med: character height in pixels.
        estimated_speed_px_s: estimated average writing speed in px/s.

    Returns:
        Tremor index ratio in [0, 1].
    """
    all_ripples = []
    resample_step = 0.5  # 0.5 px step

    for stk in strokes:
        if len(stk) < 5:
            continue
        _, _, ripple = decompose_stroke_spatial_residual(stk, h_med=h_med, resample_step_px=resample_step)
        if len(ripple) >= 16:
            all_ripples.append(ripple)

    if not all_ripples:
        return 0.0

    # Concatenate residual series
    ripple_series = np.concatenate(all_ripples)
    if len(ripple_series) < 32:
        return 0.0

    # Convert spatial sampling rate to temporal sampling rate using estimated speed:
    # dt = resample_step / speed  =>  fs_eff = speed / resample_step
    speed = max(float(estimated_speed_px_s), 20.0)
    fs_eff = speed / resample_step

    # Welch PSD
    nperseg = min(len(ripple_series), 128)
    freqs, psd = welch(ripple_series - np.mean(ripple_series), fs=fs_eff, nperseg=nperseg)

    # 4-8 Hz physiological tremor band vs 0.5-20 Hz total band
    tremor_mask = (freqs >= 4.0) & (freqs <= 8.0)
    total_mask = (freqs >= 0.5) & (freqs <= 20.0)

    t_power = float(np.sum(psd[tremor_mask]))
    tot_power = float(np.sum(psd[total_mask]))

    if tot_power <= 1e-9:
        return 0.0

    return float(np.clip(t_power / tot_power, 0.0, 1.0))

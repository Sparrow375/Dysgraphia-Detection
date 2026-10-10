"""Curvature, stroke dynamics, and jerk proxy feature extraction.

Definitions:
- Smooth stroke paths with Savitzky-Golay (scale ~0.05h).
- Resample by arclength at 0.05h.
- Compute tangent angle theta(s) and curvature kappa = d(theta)/ds.
- Features:
  - jerk_proxy: mean |d(kappa)/ds| * h^2 (dimensionless jerk proxy).
  - tangent_variance_short_wavelength: fraction of tangent-angle variance at wavelengths < 0.5h.
  - kappa_sign_changes_per_h: kappa sign changes per h of path (with deadband).
  - tortuosity_median: median (path length / chord) per stroke segment.
- Shirorekha headline is masked out for Devanagari script.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from skan.csr import Skeleton

from features.utils import get_x_height as _get_x_height_shared, load_or_binarize_ink as _load_or_binarize_ink_shared


def _get_x_height(sentence_json: Dict[str, Any], default: float = 40.0) -> float:
    """Retrieve or estimate x-height h."""
    return _get_x_height_shared(sentence_json, default=default)


def _load_or_binarize_ink(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Retrieve or compute binary ink mask for sentence crop."""
    return _load_or_binarize_ink_shared(sentence_json, image_crop)


def mask_shirorekha(ink_mask: np.ndarray, h: float) -> np.ndarray:
    """Mask out the shirorekha (horizontal headline) for Devanagari script."""
    # Horizontal opening to detect long horizontal runs characteristic of shirorekha
    kernel_len = max(15, int(0.7 * h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    shiro_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)
    # Dilate vertically slightly to cover headline junction zone
    v_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    shiro_dilated = cv2.dilate(shiro_mask, v_dilate)

    clean_mask = ink_mask.copy()
    clean_mask[shiro_dilated > 0] = 0
    return clean_mask


def resample_path_arclength(
    coords: np.ndarray,
    step_ds: float,
) -> Optional[np.ndarray]:
    """Resample 2D path coordinates by uniform arclength step."""
    if len(coords) < 3:
        return None

    # coords is (y, x) or (x, y); treat col 0 as y, col 1 as x
    diffs = np.diff(coords, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = s[-1]

    if total_len < 3.0 * step_ds or total_len <= 0.0:
        return None

    # Unique s for interpolation
    unique_mask = np.concatenate([[True], seg_lens > 1e-4])
    s_unique = s[unique_mask]
    coords_unique = coords[unique_mask]

    if len(s_unique) < 3:
        return None

    s_uniform = np.arange(0.0, total_len, step_ds)
    if len(s_uniform) < 4:
        return None

    interp_fn = interp1d(s_unique, coords_unique, axis=0, kind="linear", fill_value="extrapolate")
    return interp_fn(s_uniform)


def compute_curvature_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute curvature, jerk proxy, high-frequency spectral variance, and tortuosity.

    Returns:
        dict with keys:
            - jerk_proxy: float | NaN
            - tangent_variance_short_wavelength: float | NaN
            - kappa_sign_changes_per_h: float | NaN
            - tortuosity_median: float | NaN
    """
    h = _get_x_height(sentence_json)
    ink_mask = _load_or_binarize_ink(sentence_json, image_crop)

    nan_dict = {
        "jerk_proxy": float("nan"),
        "tangent_variance_short_wavelength": float("nan"),
        "kappa_sign_changes_per_h": float("nan"),
        "tortuosity_median": float("nan"),
    }

    if ink_mask is None or ink_mask.sum() < 20 or h <= 5.0:
        return nan_dict

    # Mask shirorekha for Devanagari script
    script = str(sentence_json.get("script", "devanagari")).lower()
    if script == "devanagari":
        ink_to_skel = mask_shirorekha(ink_mask, h)
    else:
        ink_to_skel = ink_mask

    if ink_to_skel.sum() < 15:
        return nan_dict

    skel = skeletonize(ink_to_skel > 0).astype(np.uint8)
    if skel.sum() < 10:
        return nan_dict

    try:
        skan_obj = Skeleton(skel.astype(bool))
        n_paths = int(skan_obj.n_paths)
    except Exception:
        return nan_dict

    if n_paths == 0:
        return nan_dict

    step_ds = max(1.0, 0.05 * h)
    all_dkappa_ds: List[float] = []
    all_tortuosities: List[float] = []
    total_sign_changes = 0
    total_analyzed_length = 0.0
    spectral_short_var_weights: List[Tuple[float, float]] = []  # (fraction, length)

    for i in range(min(n_paths, 300)):
        coords = skan_obj.path_coordinates(i)  # (N, 2): [y, x]
        if len(coords) < 4:
            continue

        resampled = resample_path_arclength(coords, step_ds)
        if resampled is None or len(resampled) < 6:
            continue

        y_res = resampled[:, 0]
        x_res = resampled[:, 1]
        n_pts = len(resampled)
        path_len = (n_pts - 1) * step_ds

        # Chord distance
        chord = float(np.hypot(x_res[-1] - x_res[0], y_res[-1] - y_res[0]))
        if chord > 0.1 * h:
            tortuosity = float(path_len / max(chord, 1e-4))
            all_tortuosities.append(tortuosity)

        # Savitzky-Golay smoothing (window ~ 0.2h / step_ds)
        # Ensure window is odd and <= n_pts
        target_win = max(5, int(round(0.2 * h / step_ds)) | 1)
        win_size = min(target_win, n_pts if n_pts % 2 == 1 else n_pts - 1)

        if win_size >= 5:
            try:
                x_smooth = savgol_filter(x_res, window_length=win_size, polyorder=2)
                y_smooth = savgol_filter(y_res, window_length=win_size, polyorder=2)
            except Exception:
                x_smooth, y_smooth = x_res, y_res
        else:
            x_smooth, y_smooth = x_res, y_res

        # Tangent vector dx/ds, dy/ds
        dx = np.gradient(x_smooth, step_ds)
        dy = np.gradient(y_smooth, step_ds)
        speed = np.hypot(dx, dy)
        theta = np.unwrap(np.arctan2(dy, dx))

        # Curvature kappa = d(theta)/ds normalized by local speed
        kappa = np.gradient(theta, step_ds) / np.clip(speed, 0.5, None)

        # d(kappa)/ds
        dkappa_ds = np.gradient(kappa, step_ds) / np.clip(speed, 0.5, None)

        # Trim boundary margin points to eliminate Savitzky-Golay / finite-difference edge artifacts
        trim = min(win_size // 2, len(dkappa_ds) // 4)
        if trim > 0 and len(dkappa_ds) > 2 * trim:
            interior_dkappa = dkappa_ds[trim:-trim]
        else:
            interior_dkappa = dkappa_ds
        all_dkappa_ds.extend(np.abs(interior_dkappa).tolist())

        # Kappa sign changes with deadband (|kappa| > 0.02 / h)
        deadband = 0.02 / h
        active_signs = np.zeros_like(kappa)
        active_signs[kappa > deadband] = 1
        active_signs[kappa < -deadband] = -1

        # Remove zeroes to check actual polarity reversals
        non_zero_signs = active_signs[active_signs != 0]
        if len(non_zero_signs) >= 2:
            sign_flips = int(np.sum(non_zero_signs[:-1] != non_zero_signs[1:]))
            total_sign_changes += sign_flips

        total_analyzed_length += path_len

        # Short-wavelength tangent variance (< 0.5h, i.e. spatial freq f > 2 / h)
        if n_pts >= 16:
            # Detrend theta
            theta_detrend = theta - np.polyval(np.polyfit(np.arange(n_pts), theta, 1), np.arange(n_pts))
            total_var = float(np.var(theta_detrend))
            if total_var > 1e-6:
                # FFT power
                fft_vals = np.fft.rfft(theta_detrend)
                freqs = np.fft.rfftfreq(n_pts, d=step_ds)  # cycles per pixel
                cutoff_freq = 1.0 / (0.5 * h)  # f > 2/h
                high_mask = freqs > cutoff_freq
                if np.any(high_mask):
                    high_power = float(np.sum(np.abs(fft_vals[high_mask]) ** 2))
                    total_power = float(np.sum(np.abs(fft_vals) ** 2))
                    ratio = min(1.0, max(0.0, high_power / max(total_power, 1e-8)))
                    spectral_short_var_weights.append((ratio, path_len))

    # Check minimum path length requirement (at least 2 * h analyzed)
    if total_analyzed_length < 2.0 * h or not all_dkappa_ds:
        return nan_dict

    # 1. Dimensionless jerk proxy: mean |d(kappa)/ds| * h^2
    jerk_proxy = float(np.mean(all_dkappa_ds) * (h ** 2))

    # 2. Kappa sign changes per h of path
    h_units = total_analyzed_length / h
    sign_changes_per_h = float(total_sign_changes / max(h_units, 1e-4))

    # 3. Median tortuosity
    tortuosity_med = float(np.median(all_tortuosities)) if all_tortuosities else float("nan")

    # 4. Short-wavelength spectral variance fraction
    if spectral_short_var_weights:
        weights = np.array([w[1] for w in spectral_short_var_weights], dtype=np.float64)
        vals = np.array([w[0] for w in spectral_short_var_weights], dtype=np.float64)
        short_wavelength_var = float(np.sum(vals * weights) / max(np.sum(weights), 1e-8))
    else:
        short_wavelength_var = float("nan")

    return {
        "jerk_proxy": jerk_proxy,
        "tangent_variance_short_wavelength": short_wavelength_var,
        "kappa_sign_changes_per_h": sign_changes_per_h,
        "tortuosity_median": tortuosity_med,
    }

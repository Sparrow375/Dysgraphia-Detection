"""Stroke-width and grayscale shading micro-texture feature extraction (Stream 2).

Extracts physical pen/pencil pressure, stroke thickness fluctuations,
hesitation bloats, pinch breaks, and paper erasure/smudge halos that
binary Otsu masks destroy.

Zero deep learning, pure spatial computer vision (<10ms execution, <10MB RAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import stats
from skimage.morphology import skeletonize
from skan.csr import Skeleton

from features.utils import get_x_height as _get_x_height_shared, load_or_binarize_ink as _load_or_binarize_ink_shared


def _get_x_height(sentence_json: Dict[str, Any], default: float = 40.0) -> float:
    """Retrieve or robustly estimate x-height h."""
    return _get_x_height_shared(sentence_json, default=default)


def _load_or_binarize_ink(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
    data_root: str = "data/processed",
) -> Optional[np.ndarray]:
    """Retrieve clean binary ink mask (1=ink, 0=paper)."""
    return _load_or_binarize_ink_shared(sentence_json, image_crop, data_root=data_root)


def _load_grayscale_crop(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
    data_root: str = "data/processed",
) -> Optional[np.ndarray]:
    """Load or extract natural grayscale crop (paper=high, ink=low)."""
    if image_crop is not None:
        if len(image_crop.shape) == 3:
            return cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        return image_crop.copy()

    crop_fn = sentence_json.get("crop_filename")
    sid = sentence_json.get("student_id")
    school = sentence_json.get("school", "school_a")

    if crop_fn and sid:
        path = Path(data_root) / school / sid / crop_fn
        if path.exists():
            return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    return None


def compute_stroke_width_texture_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
    data_root: str = "data/processed",
) -> Dict[str, float]:
    """Compute Stream 2: Stroke-Width & Grayscale Shading micro-texture features.

    Features computed:
      1. Stroke-Width Profiling (Euclidean Distance Transform on clean ink):
         - stroke_width_mean_norm: Mean stroke thickness / x-height h
         - stroke_width_cv: Coefficient of variation of stroke thickness
         - stroke_width_jitter: Mean normalized step difference |w_{k+1} - w_k| / mean(w)
         - stroke_width_pinch_rate: Fraction of skeleton points with thickness < 0.5 * mean(w)
         - stroke_width_bloat_rate: Fraction of skeleton points with thickness > 1.5 * mean(w)

      2. Grayscale Shading & Pencil Pressure Dynamics:
         - pressure_intensity_mean: Mean optical density (0.0=white paper, 1.0=pitch black)
         - pressure_intensity_cv: Coefficient of variation of pencil pressure along skeleton
         - pressure_gradient_mean: Mean normalized local pressure step along skeleton paths
         - pressure_faint_rate: Fraction of stroke points with faint pressure (<0.25 * mean)
         - pressure_heavy_rate: Fraction of stroke points with heavy pressure (>1.6 * mean)
         - pressure_width_corr: Spearman correlation between local thickness and optical density

      3. Erasure & Smudge Halo Dynamics:
         - smudge_halo_darkness: Darkness of non-ink paper halo immediately surrounding words
         - background_roughness_std: Grayscale intensity standard deviation in halo zone (abrasion)
         - erasure_patch_ratio: Fraction of halo pixels significantly darker than clean paper

    Args:
        sentence_json: Loaded sentence JSON dictionary.
        image_crop: Optional pre-loaded grayscale or BGR crop image array.
        data_root: Base path for processed data resolution.

    Returns:
        Dict of feature name -> float (or NaN if extraction fails).
    """
    nan_dict: Dict[str, float] = {
        "stroke_width_mean_norm": float("nan"),
        "stroke_width_cv": float("nan"),
        "stroke_width_jitter": float("nan"),
        "stroke_width_pinch_rate": float("nan"),
        "stroke_width_bloat_rate": float("nan"),
        "pressure_intensity_mean": float("nan"),
        "pressure_intensity_cv": float("nan"),
        "pressure_gradient_mean": float("nan"),
        "pressure_faint_rate": float("nan"),
        "pressure_heavy_rate": float("nan"),
        "pressure_width_corr": float("nan"),
        "smudge_halo_darkness": float("nan"),
        "background_roughness_std": float("nan"),
        "erasure_patch_ratio": float("nan"),
    }

    h = _get_x_height(sentence_json)
    if h <= 5.0:
        return nan_dict

    ink_mask = _load_or_binarize_ink(sentence_json, image_crop, data_root=data_root)
    gray = _load_grayscale_crop(sentence_json, image_crop, data_root=data_root)

    if ink_mask is None or gray is None or ink_mask.sum() < 25:
        return nan_dict

    # Ensure mask and gray match dimensions
    if ink_mask.shape != gray.shape:
        ink_mask = cv2.resize(ink_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Standardize grayscale polarity: paper should be bright (>128), ink dark (<128)
    if np.mean(gray) < 128:
        gray = (255 - gray).astype(np.uint8)

    # ---------------------------------------------------------
    # 1. Background Paper Level & Normalized Optical Density
    # ---------------------------------------------------------
    non_ink_pixels = gray[ink_mask == 0]
    if len(non_ink_pixels) > 50:
        bg_level = float(np.percentile(non_ink_pixels, 90))
        bg_std = float(np.std(non_ink_pixels))
    else:
        bg_level = 255.0
        bg_std = 5.0

    bg_level = max(128.0, bg_level)
    bg_std = max(1.0, bg_std)

    # Optical density P: 0.0 = background paper, 1.0 = pitch black ink
    optical_density = np.clip((bg_level - gray.astype(np.float32)) / bg_level, 0.0, 1.0)

    # ---------------------------------------------------------
    # 2. Euclidean Distance Transform & Skeleton Centerline
    # ---------------------------------------------------------
    # Distance from each ink pixel to background (radius of stroke)
    dist_map = cv2.distanceTransform(ink_mask.astype(np.uint8), cv2.DIST_L2, 5)

    # Skeletonize ink
    skel = skeletonize(ink_mask > 0).astype(np.uint8)
    skel_ys, skel_xs = np.where(skel > 0)

    if len(skel_ys) < 15:
        return nan_dict

    # Local stroke diameter w = 2 * distance_transform
    stroke_widths = 2.0 * dist_map[skel_ys, skel_xs]
    densities = optical_density[skel_ys, skel_xs]

    w_mean = float(np.mean(stroke_widths))
    w_std = float(np.std(stroke_widths))
    w_cv = float(w_std / w_mean) if w_mean > 1e-4 else 0.0

    nan_dict["stroke_width_mean_norm"] = float(w_mean / h)
    nan_dict["stroke_width_cv"] = float(w_cv)

    # Pinch and bloat rates
    pinch_count = np.sum(stroke_widths < 0.5 * w_mean)
    bloat_count = np.sum(stroke_widths > 1.5 * w_mean)
    nan_dict["stroke_width_pinch_rate"] = float(pinch_count / len(stroke_widths))
    nan_dict["stroke_width_bloat_rate"] = float(bloat_count / len(stroke_widths))

    # ---------------------------------------------------------
    # 3. Path-Based Stroke Jitter & Pressure Gradients
    # ---------------------------------------------------------
    # Use skan to trace ordered coordinates along continuous stroke paths
    step_diffs_w: List[float] = []
    step_diffs_p: List[float] = []

    try:
        skan_obj = Skeleton(skel.astype(bool))
        n_paths = int(skan_obj.n_paths)
        for i in range(min(n_paths, 300)):
            coords = skan_obj.path_coordinates(i)  # (N, 2) in (y, x)
            if len(coords) >= 2:
                cy = coords[:, 0].astype(int)
                cx = coords[:, 1].astype(int)
                # Clip to image bounds
                cy = np.clip(cy, 0, gray.shape[0] - 1)
                cx = np.clip(cx, 0, gray.shape[1] - 1)

                p_w = 2.0 * dist_map[cy, cx]
                p_d = optical_density[cy, cx]

                dw = np.abs(np.diff(p_w))
                dp = np.abs(np.diff(p_d))

                step_diffs_w.extend(dw.tolist())
                step_diffs_p.extend(dp.tolist())
    except Exception:
        pass

    # If skan was unable to extract paths, fallback to coordinate sorting
    if len(step_diffs_w) == 0:
        sort_idx = np.lexsort((skel_xs, skel_ys))
        p_w_sorted = stroke_widths[sort_idx]
        p_d_sorted = densities[sort_idx]
        dw = np.abs(np.diff(p_w_sorted))
        dp = np.abs(np.diff(p_d_sorted))
        step_diffs_w = dw[dw < 10.0].tolist()
        step_diffs_p = dp.tolist()

    if len(step_diffs_w) > 0 and w_mean > 1e-4:
        nan_dict["stroke_width_jitter"] = float(np.mean(step_diffs_w) / w_mean)
    else:
        nan_dict["stroke_width_jitter"] = 0.0

    # ---------------------------------------------------------
    # 4. Pressure / Optical Density Distribution
    # ---------------------------------------------------------
    p_mean = float(np.mean(densities))
    p_std = float(np.std(densities))
    p_cv = float(p_std / p_mean) if p_mean > 1e-4 else 0.0

    nan_dict["pressure_intensity_mean"] = float(p_mean)
    nan_dict["pressure_intensity_cv"] = float(p_cv)

    if len(step_diffs_p) > 0 and p_mean > 1e-4:
        nan_dict["pressure_gradient_mean"] = float(np.mean(step_diffs_p) / p_mean)
    else:
        nan_dict["pressure_gradient_mean"] = 0.0

    # Faint contact (< 75% of mean) and heavy digging (> 125% of mean)
    faint_thresh = 0.75 * p_mean
    heavy_thresh = 1.25 * p_mean
    faint_count = np.sum(densities < faint_thresh)
    heavy_count = np.sum(densities > heavy_thresh)
    nan_dict["pressure_faint_rate"] = float(faint_count / len(densities))
    nan_dict["pressure_heavy_rate"] = float(heavy_count / len(densities))

    # Pressure-width mechanical coupling
    if len(stroke_widths) >= 20 and w_std > 1e-4 and p_std > 1e-4:
        corr, _ = stats.spearmanr(stroke_widths, densities)
        nan_dict["pressure_width_corr"] = float(corr) if not np.isnan(corr) else 0.0
    else:
        nan_dict["pressure_width_corr"] = 0.0

    # ---------------------------------------------------------
    # 5. Erasure & Smudge Halo Dynamics
    # ---------------------------------------------------------
    # Inner exclusion radius ~3px to skip antialiased stroke edges
    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dil_inner = cv2.dilate(ink_mask.astype(np.uint8), k_inner)

    # Outer halo radius ~15px (or ~0.35h) surrounding words
    r_outer = max(11, min(31, int(0.35 * h) * 2 + 1))
    k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_outer, r_outer))
    dil_outer = cv2.dilate(ink_mask.astype(np.uint8), k_outer)

    # Halo zone is the non-ink paper ring immediately surrounding strokes
    halo_mask = (dil_outer > 0) & (dil_inner == 0)

    halo_pixels = gray[halo_mask]

    if len(halo_pixels) > 50:
        halo_optical_density = np.clip((bg_level - halo_pixels.astype(np.float32)) / bg_level, 0.0, 1.0)
        halo_std = float(np.std(halo_pixels))

        # Smudge halo darkness: mean optical density of graphite smearing in the halo ring
        nan_dict["smudge_halo_darkness"] = float(np.mean(halo_optical_density))
        nan_dict["background_roughness_std"] = float(halo_std)

        # Erasure patches: fraction of halo pixels significantly darker than clean paper (> 2*sigma)
        erasure_thresh = bg_level - 2.0 * bg_std
        erasure_count = np.sum(halo_pixels < erasure_thresh)
        nan_dict["erasure_patch_ratio"] = float(erasure_count / len(halo_pixels))
    else:
        nan_dict["smudge_halo_darkness"] = 0.0
        nan_dict["background_roughness_std"] = 0.0
        nan_dict["erasure_patch_ratio"] = 0.0

    return nan_dict


def classify_writing_tool(
    image_gray_or_path: Any,
    bg_level: Optional[float] = None,
) -> Dict[str, Any]:
    """Classify writing tool (pen vs pencil) from 3 pure-math signals on page_normalized.png:
    
    1. Foreground Optical Density Spread:
       Pencils have lower mean darkness and wider gray-level variance;
       Ballpoint/gel pens have saturated dark cores.
    2. Stroke Edge Gradient:
       Pen ink creates a sharp step-edge against paper;
       Graphite creates a softer, grainy edge falloff.
    3. Median Stroke Width (w_med):
       Ballpoint nibs cluster tightly around ~2 px; worn pencils are wider.

    Parameters
    ----------
    image_gray_or_path : str, Path, or np.ndarray
        Page image (grayscale array or filepath to page_normalized.png).
    bg_level : float, optional
        Estimated clean paper background level (default estimated from 90th percentile).

    Returns
    -------
    dict
        'tool': 'pencil' or 'pen'
        'is_pencil': 1.0 if pencil else 0.0
        'w_med': float (median stroke width in px)
        'od_mean': float (mean foreground optical density in [0, 1])
        'sat_ratio': float (fraction of saturated dark pixels I < 60)
        'edge_grad': float (mean stroke edge gradient magnitude)
    """
    if isinstance(image_gray_or_path, (str, Path)):
        img = cv2.imread(str(image_gray_or_path), cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_gray_or_path, np.ndarray):
        img = image_gray_or_path if len(image_gray_or_path.shape) == 2 else cv2.cvtColor(image_gray_or_path, cv2.COLOR_BGR2GRAY)
    else:
        img = None

    if img is None or img.size == 0:
        return {
            "tool": "pen",
            "is_pencil": 0.0,
            "w_med": 2.0,
            "od_mean": 0.5,
            "sat_ratio": 0.1,
            "edge_grad": 400.0,
        }

    if bg_level is None:
        bg_level = float(np.percentile(img, 90))

    # Ink mask via Otsu thresholding
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Median stroke width via Euclidean Distance Transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    w_med = float(np.median(dist[dist > 0]) * 2.0) if np.any(dist > 0) else 2.0

    # 2. Foreground Optical Density and Core Darkness Saturation
    fg = img[mask > 0]
    if len(fg) > 0:
        od = np.clip((bg_level - fg.astype(np.float32)) / max(bg_level, 1e-3), 0.0, 1.0)
        od_mean = float(np.mean(od))
        od_std = float(np.std(od))
        sat_ratio = float(np.mean(fg < 60))
    else:
        od_mean = 0.5
        od_std = 0.1
        sat_ratio = 0.1

    # 3. Stroke Edge Gradient
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)
    k_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_zone = cv2.dilate(mask, k_edge) ^ cv2.erode(mask, k_edge)
    edge_grad = float(np.mean(grad[edge_zone > 0])) if np.any(edge_zone > 0) else 350.0

    # Decision rule:
    # Pens have saturated dark cores (sat_ratio > 0.03 and edge_grad > 330)
    # or narrow crisp strokes (w_med <= 2.2 and edge_grad > 350) or high overall OD (> 0.65)
    is_pen = (
        (sat_ratio > 0.03 and edge_grad > 330.0)
        or (w_med <= 2.2 and edge_grad > 350.0)
        or (od_mean > 0.65)
    )
    tool = "pen" if is_pen else "pencil"
    is_pencil = 0.0 if is_pen else 1.0

    return {
        "tool": tool,
        "is_pencil": is_pencil,
        "w_med": float(w_med),
        "od_mean": float(od_mean),
        "od_std": float(od_std),
        "sat_ratio": float(sat_ratio),
        "edge_grad": float(edge_grad),
    }


def calibrate_stream2_by_tool(
    df: pd.DataFrame,
    features_to_calibrate: Optional[List[str]] = None,
    train_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Tuple[float, float]]]]:
    """Normalize Stream 2 features strictly within tool (pen vs pencil) control baseline distribution.
    
    Formula:
        z_{tool} = (x - median(tool_controls)) / (1.4826 * max(MAD(tool_controls), 1e-6))
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing features and 'is_pencil'.
    features_to_calibrate : list of str, optional
        Features to normalize (defaults to core Stream 2 features).
    train_df : pd.DataFrame, optional
        Training set to fit baselines on (if None, fits on df).

    Returns
    -------
    tuple of (calibrated_df, params_dict)
    """
    if features_to_calibrate is None:
        features_to_calibrate = [
            "pressure_intensity_cv",
            "stroke_width_jitter",
            "stroke_width_bloat_rate",
            "pressure_faint_rate",
            "pressure_heavy_rate",
        ]

    fit_source = train_df if train_df is not None else df
    # Filter to controls for calibration baseline if label column exists
    if "label" in fit_source.columns and (fit_source["label"] == 0).any():
        control_source = fit_source[fit_source["label"] == 0]
    else:
        control_source = fit_source

    res_df = df.copy()
    params: Dict[str, Dict[str, Tuple[float, float]]] = {"pencil": {}, "pen": {}}

    for feat in features_to_calibrate:
        if feat not in df.columns:
            continue

        # Fit on pencil controls
        p_vals = control_source[control_source.get("is_pencil", 1.0) == 1.0][feat].dropna()
        p_med = float(p_vals.median()) if len(p_vals) > 0 else 0.0
        p_mad = float(np.median(np.abs(p_vals - p_med))) if len(p_vals) > 0 else 1.0
        p_scale = max(1.4826 * p_mad, 1e-4)
        params["pencil"][feat] = (p_med, p_scale)

        # Fit on pen controls
        pen_vals = control_source[control_source.get("is_pencil", 1.0) == 0.0][feat].dropna()
        pen_med = float(pen_vals.median()) if len(pen_vals) > 0 else p_med
        pen_mad = float(np.median(np.abs(pen_vals - pen_med))) if len(pen_vals) > 0 else p_mad
        pen_scale = max(1.4826 * pen_mad, 1e-4)
        params["pen"][feat] = (pen_med, pen_scale)

        # Apply calibration
        norm_col = f"{feat}_tool_norm"
        pencil_mask = res_df.get("is_pencil", 1.0) == 1.0
        pen_mask = ~pencil_mask

        res_df.loc[pencil_mask, norm_col] = (res_df.loc[pencil_mask, feat] - p_med) / p_scale
        res_df.loc[pen_mask, norm_col] = (res_df.loc[pen_mask, feat] - pen_med) / pen_scale

        # Clip z-scores to [-5.0, 5.0]
        res_df[norm_col] = res_df[norm_col].clip(-5.0, 5.0)

    return res_df, params


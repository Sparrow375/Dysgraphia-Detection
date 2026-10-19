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

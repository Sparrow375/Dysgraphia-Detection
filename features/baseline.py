"""Baseline residual and line trajectory features.

Definitions:
- Fit y = a*x + b per line (using RANSAC / robust linear regression).
- e_i = word-baseline residual.
- baseline_rmse_norm = sqrt(mean(e^2)) / h.
- baseline_slope_mean = mean baseline slope across lines.
- baseline_slope_std = standard deviation of baseline slopes across lines (NaN if < 2 lines).
- NaN if fewer than 3 words are available for sentence statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from features.utils import get_x_height as _get_x_height_shared


def _get_lines(sentence_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Retrieve physical line structures from sentence JSON schema."""
    lines = sentence_json.get("physical_lines") or sentence_json.get("lines") or []
    return lines


def _get_x_height(sentence_json: Dict[str, Any], default: float = 40.0) -> float:
    """Retrieve or estimate x-height h from sentence JSON."""
    return _get_x_height_shared(sentence_json, default=default)


def _fit_line_ransac(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Fit robust line y = a*x + b using RANSAC with linear regression fallback."""
    x_2d = x.reshape(-1, 1)
    if len(x) >= 3:
        try:
            ransac = RANSACRegressor(min_samples=2, residual_threshold=10.0, random_state=42)
            ransac.fit(x_2d, y)
            slope = float(ransac.estimator_.coef_[0])
            intercept = float(ransac.estimator_.intercept_)
            residuals = y - (slope * x + intercept)
            return slope, intercept, residuals
        except Exception:
            pass

    # Fallback to standard OLS
    lr = LinearRegression()
    lr.fit(x_2d, y)
    slope = float(lr.coef_[0])
    intercept = float(lr.intercept_)
    residuals = y - (slope * x + intercept)
    return slope, intercept, residuals


def compute_baseline_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute baseline residual and line alignment features.

    Returns:
        dict with keys:
            - baseline_rmse_norm: float | NaN
            - baseline_slope_mean: float | NaN
            - baseline_slope_std: float | NaN
    """
    lines = _get_lines(sentence_json)
    h = _get_x_height(sentence_json)

    # Collect points per line
    line_residuals: List[float] = []
    line_slopes: List[float] = []
    total_words = 0

    for line in lines:
        pts: List[Tuple[float, float]] = []
        # Check precomputed word baseline points
        if line.get("word_baseline_points"):
            for pt in line["word_baseline_points"]:
                pts.append((float(pt[0]), float(pt[1])))
        elif line.get("words"):
            for wd in line["words"]:
                # Point is word center-bottom
                if "cx" in wd and "bottom" in wd:
                    pts.append((float(wd["cx"]), float(wd["bottom"])))
                elif "bbox" in wd and len(wd["bbox"]) == 4:
                    wx, wy, ww, wh = wd["bbox"]
                    pts.append((float(wx + ww / 2.0), float(wy + wh)))

        total_words += len(pts)

        if len(pts) >= 2:
            x_arr = np.array([p[0] for p in pts], dtype=np.float64)
            y_arr = np.array([p[1] for p in pts], dtype=np.float64)
            slope, intercept, res = _fit_line_ransac(x_arr, y_arr)
            line_slopes.append(slope)
            line_residuals.extend(res.tolist())
        elif len(pts) == 1:
            # Single word has 0 residual from itself
            line_residuals.append(0.0)

    # Fallback if lines wasn't populated but top-level word_baseline_points exists
    if not line_residuals and sentence_json.get("word_baseline_points"):
        pts_top = sentence_json["word_baseline_points"]
        total_words = len(pts_top)
        if len(pts_top) >= 2:
            x_arr = np.array([p[0] for p in pts_top], dtype=np.float64)
            y_arr = np.array([p[1] for p in pts_top], dtype=np.float64)
            slope, intercept, res = _fit_line_ransac(x_arr, y_arr)
            line_slopes.append(slope)
            line_residuals.extend(res.tolist())

    # Specification: NaN if fewer than 3 words
    if total_words < 3 or not line_residuals:
        return {
            "baseline_rmse_norm": float("nan"),
            "baseline_slope_mean": float("nan"),
            "baseline_slope_std": float("nan"),
        }

    rmse = float(np.sqrt(np.mean(np.array(line_residuals, dtype=np.float64) ** 2)))
    rmse_norm = float(rmse / h) if h > 0 else float("nan")

    slope_mean = float(np.mean(line_slopes)) if line_slopes else float("nan")
    slope_std = float(np.std(line_slopes, ddof=1)) if len(line_slopes) >= 2 else float("nan")

    return {
        "baseline_rmse_norm": rmse_norm,
        "baseline_slope_mean": slope_mean,
        "baseline_slope_std": slope_std,
    }

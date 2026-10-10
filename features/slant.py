"""Slant feature extraction using orientation tensor on skeleton tangents.

Definition:
- Orientation tensor on skeleton tangents within 45 degrees of vertical.
- Uses doubled-angle circular mean:
  phi_k = 2 * theta_k
  C = mean(cos(phi_k)), S = mean(sin(phi_k))
  R = sqrt(C^2 + S^2)
  theta_mean = 0.5 * atan2(S, C)
  circ_std = 0.5 * sqrt(-2 * ln(max(R, 1e-8)))
- Reports:
  - slant_mean_deg: float | NaN
  - slant_circular_std_deg: float | NaN
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from features.utils import get_x_height as _get_x_height_shared, load_or_binarize_ink as _load_or_binarize_ink_shared


def _load_or_binarize_ink(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Retrieve or compute binary ink mask for sentence crop."""
    return _load_or_binarize_ink_shared(sentence_json, image_crop)


def extract_skeleton_tangents(
    ink_mask: np.ndarray,
    max_vertical_angle_rad: float = np.pi / 4,
) -> np.ndarray:
    """Extract tangent angles of skeleton stroke segments within +/- 45 deg of vertical.

    Returns:
        1D array of tangent angles in radians relative to vertical (0 = vertical,
        >0 = rightward slant, <0 = leftward slant).
    """
    if ink_mask.sum() < 20:
        return np.array([], dtype=np.float64)

    skel = skeletonize(ink_mask > 0).astype(np.uint8)
    coords = np.column_stack(np.where(skel > 0))  # (row, col) = (y, x)
    if len(coords) < 10:
        return np.array([], dtype=np.float64)

    # Use 5x5 local neighborhood covariance / eigenvectors for robust tangent estimation
    h, w = skel.shape
    tangents: List[float] = []

    # Fast sampling of skeleton coordinates
    step = 1 if len(coords) < 1500 else len(coords) // 1500
    sample_coords = coords[::step]

    for y, x in sample_coords:
        y_min, y_max = max(0, y - 2), min(h, y + 3)
        x_min, x_max = max(0, x - 2), min(w, x + 3)
        patch = skel[y_min:y_max, x_min:x_max]

        py, px = np.where(patch > 0)
        if len(px) < 3:
            continue

        # Covariance matrix of local skeleton points
        pts = np.column_stack([px - (x - x_min), py - (y - y_min)])
        cov = np.cov(pts, rowvar=False)

        if cov.shape == (2, 2) and not np.isnan(cov).any():
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Principal direction is eigenvector corresponding to larger eigenvalue
            dx, dy = eigvecs[:, -1]
            if dy < 0:
                dx, dy = -dx, -dy  # Orient downward (positive dy)

            if dy != 0:
                angle_from_vert = float(np.arctan2(dx, dy))
                # Restrict to +/- 45 degrees from vertical
                if abs(angle_from_vert) <= max_vertical_angle_rad:
                    tangents.append(angle_from_vert)

    return np.array(tangents, dtype=np.float64)


def compute_slant_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute stroke slant circular mean and circular standard deviation.

    Returns:
        dict with keys:
            - slant_mean_deg: float | NaN
            - slant_circular_std_deg: float | NaN
    """
    ink_mask = _load_or_binarize_ink(sentence_json, image_crop)
    if ink_mask is None:
        return {
            "slant_mean_deg": float("nan"),
            "slant_circular_std_deg": float("nan"),
        }

    tangents = extract_skeleton_tangents(ink_mask, max_vertical_angle_rad=np.pi / 4)
    if len(tangents) < 10:
        return {
            "slant_mean_deg": float("nan"),
            "slant_circular_std_deg": float("nan"),
        }

    # Doubled-angle circular mean
    phi = 2.0 * tangents
    c = float(np.mean(np.cos(phi)))
    s = float(np.mean(np.sin(phi)))
    r = float(np.sqrt(c * c + s * s))

    # Circular mean angle in radians
    mean_angle_rad = 0.5 * np.arctan2(s, c)
    mean_angle_deg = float(np.degrees(mean_angle_rad))

    # Circular standard deviation in radians
    # sqrt(-2 * ln(R)) clamped
    r_clamped = min(0.999999, max(1e-6, r))
    circ_std_rad = float(0.5 * np.sqrt(-2.0 * np.log(r_clamped)))
    circ_std_deg = float(np.degrees(circ_std_rad))

    return {
        "slant_mean_deg": mean_angle_deg,
        "slant_circular_std_deg": circ_std_deg,
    }

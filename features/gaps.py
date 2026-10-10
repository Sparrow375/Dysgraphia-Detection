"""Inter-word gap spacing features for Workstream A.

Definition:
- g_k / h for consecutive words in the same line.
- Reports:
  - gap_mean: float | NaN
  - gap_cv: float | NaN (std / mean)
  - gap_fraction_below_0_3h: float | NaN
  - gap_fraction_above_2h: float | NaN
- NaN if fewer than 3 words (or fewer than 2 gaps).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _get_x_height(sentence_json: Dict[str, Any], default: float = 40.0) -> float:
    """Retrieve or estimate x-height h."""
    lines = sentence_json.get("physical_lines") or sentence_json.get("lines") or []
    h_vals = [float(l["x_height_h"]) for l in lines if l.get("x_height_h") and float(l["x_height_h"]) > 5.0]
    if h_vals:
        return float(np.median(h_vals))
    bbox = sentence_json.get("crop_bbox", [0, 0, 100, 100])
    bh = bbox[3] if len(bbox) >= 4 else 100
    return max(15.0, float(bh * 0.4))


def compute_gap_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute inter-word gap statistics.

    Returns:
        dict with keys:
            - gap_mean: float | NaN
            - gap_cv: float | NaN
            - gap_fraction_below_0_3h: float | NaN
            - gap_fraction_above_2h: float | NaN
    """
    h = _get_x_height(sentence_json)
    lines = sentence_json.get("physical_lines") or sentence_json.get("lines") or []

    all_normalized_gaps: List[float] = []

    for line in lines:
        words = line.get("words", [])
        if len(words) < 2:
            continue

        # Sort words left-to-right
        sorted_words = sorted(words, key=lambda w: w["bbox"][0] if "bbox" in w else 0)

        for i in range(len(sorted_words) - 1):
            w_curr = sorted_words[i]
            w_next = sorted_words[i + 1]

            if "bbox" not in w_curr or "bbox" not in w_next:
                continue

            wx_curr, _, ww_curr, _ = w_curr["bbox"]
            wx_next, _, _, _ = w_next["bbox"]

            gap_px = wx_next - (wx_curr + ww_curr)
            norm_gap = float(gap_px / h) if h > 0 else float("nan")
            all_normalized_gaps.append(norm_gap)

    # Specification: NaN if fewer than 3 words (at least 2 gaps)
    if len(all_normalized_gaps) < 2 or h <= 0:
        return {
            "gap_mean": float("nan"),
            "gap_cv": float("nan"),
            "gap_fraction_below_0_3h": float("nan"),
            "gap_fraction_above_2h": float("nan"),
        }

    arr = np.array(all_normalized_gaps, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    cv_val = float(std_val / mean_val) if abs(mean_val) > 1e-6 else float("nan")

    frac_below_03 = float(np.mean(arr < 0.3))
    frac_above_2 = float(np.mean(arr > 2.0))

    return {
        "gap_mean": mean_val,
        "gap_cv": cv_val,
        "gap_fraction_below_0_3h": frac_below_03,
        "gap_fraction_above_2h": frac_above_2,
    }

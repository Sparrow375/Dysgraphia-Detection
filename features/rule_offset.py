"""Rule offset features for Workstream A.

Definition:
- o_i = (y_i - y_rule(x_i)) / r.
- Reports:
  - rule_offset_mean: float | NaN (mean floating distance above or below the rule).
  - rule_offset_std: float | NaN (vertical adherence variability).
- NaN if fewer than 3 words or r is unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_rule_offset_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute ruled-line offset mean and standard deviation.

    Returns:
        dict with keys:
            - rule_offset_mean: float | NaN
            - rule_offset_std: float | NaN
    """
    r = float(sentence_json.get("median_rule_spacing_r", 0.0))
    if r <= 10.0:
        # Check if line height or bbox can estimate r
        bbox = sentence_json.get("crop_bbox", [0, 0, 100, 100])
        lines_cnt = max(1, len(sentence_json.get("lines") or sentence_json.get("physical_lines") or []))
        bh = bbox[3] if len(bbox) >= 4 else 100
        r = float(bh / lines_cnt) if lines_cnt > 0 else 85.0

    lines = sentence_json.get("physical_lines") or sentence_json.get("lines") or []

    offsets: List[float] = []

    for line in lines:
        pts: List[Tuple[float, float]] = []
        if line.get("word_baseline_points"):
            for pt in line["word_baseline_points"]:
                pts.append((float(pt[0]), float(pt[1])))
        elif line.get("words"):
            for wd in line["words"]:
                if "cx" in wd and "bottom" in wd:
                    pts.append((float(wd["cx"]), float(wd["bottom"])))
                elif "bbox" in wd and len(wd["bbox"]) == 4:
                    wx, wy, ww, wh = wd["bbox"]
                    pts.append((float(wx + ww / 2.0), float(wy + wh)))

        if not pts:
            continue

        # Target rule baseline for this line: line baseline_y or median y of points
        ref_y = float(line.get("baseline_y", np.median([p[1] for p in pts])))

        # Compute offset of each word bottom from the designated line baseline / rule
        for p in pts:
            # Offset normalized by spacing r
            diff = p[1] - ref_y
            # Wrap within [-0.5*r, 0.5*r] if comparing against periodic ruling lines
            diff_wrapped = ((diff + 0.5 * r) % r) - 0.5 * r
            offsets.append(diff_wrapped / r)

    # Fallback to top-level points if line points empty
    if not offsets and sentence_json.get("word_baseline_points"):
        pts_top = sentence_json["word_baseline_points"]
        if pts_top:
            ref_y = float(np.median([p[1] for p in pts_top]))
            for p in pts_top:
                diff = p[1] - ref_y
                diff_wrapped = ((diff + 0.5 * r) % r) - 0.5 * r
                offsets.append(diff_wrapped / r)

    if len(offsets) < 3 or r <= 0:
        return {
            "rule_offset_mean": float("nan"),
            "rule_offset_std": float("nan"),
        }

    arr = np.array(offsets, dtype=np.float64)
    return {
        "rule_offset_mean": float(np.mean(arr)),
        "rule_offset_std": float(np.std(arr, ddof=1)),
    }

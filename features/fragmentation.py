"""Fragmentation, stroke graph density, and completion features.

Definitions:
- components_per_unit_width: Connected ink components / (sentence_width / h).
- junctions_per_unit_width: Skeleton branch junctions / (sentence_width / h).
- endpoints_per_unit_width: Skeleton endpoints / (sentence_width / h).
- words_written_ratio: Detected words / expected words (for copy/dictation; NaN for own writing).
- lines_used: Number of physical lines used by sentence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

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


# Standardized prompt word counts
EXPECTED_WORD_COUNTS: Dict[str, int] = {
    "copy_hindi": 7,
    "copy_english": 14,
    "dictated_hindi": 6,
    "dictated_english": 7,
}

def compute_fragmentation_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute fragmentation, graph topological density, and task completion metrics.

    Returns:
        dict with keys:
            - components_per_unit_width: float | NaN
            - junctions_per_unit_width: float | NaN
            - endpoints_per_unit_width: float | NaN
            - words_written_ratio: float | NaN
            - lines_used: float | NaN
    """
    h = _get_x_height(sentence_json)
    bbox = sentence_json.get("crop_bbox", [0, 0, 100, 100])
    w_px = float(bbox[2]) if len(bbox) >= 4 else 100.0
    w_unit = float(w_px / h) if h > 0 else 1.0

    lines = sentence_json.get("physical_lines") or sentence_json.get("lines") or []
    lines_used = float(max(1, len(lines)))

    # Graph statistics from schema or image
    skel_info = sentence_json.get("skeleton_graph", {})
    junctions = float(skel_info.get("junction_count", 0))
    endpoints = float(skel_info.get("endpoint_count", 0))

    # Connected components
    ink_mask = _load_or_binarize_ink(sentence_json, image_crop)
    if ink_mask is not None and ink_mask.sum() > 10:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        # Filter background and tiny noise specks (< 4px)
        comp_count = float(sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 4))
    else:
        # Fallback to word components sum if ink mask not loaded
        all_words = []
        for l in lines:
            all_words.extend(l.get("words", []))
        comp_count = float(max(len(all_words), 1))

    components_per_unit_width = float(comp_count / w_unit) if w_unit > 0 else float("nan")
    # NOTE: Do NOT guard on junctions/endpoints > 0. Zero junctions means fully disconnected
    # writing (no branching strokes), which is a clinically meaningful dysgraphic signal.
    # The old `and junctions > 0` guard converted the most fragmented writers to NaN,
    # which then got imputed to the population median — erasing the strongest signal.
    junctions_per_unit_width = float(junctions / w_unit) if w_unit > 0 else float("nan")
    endpoints_per_unit_width = float(endpoints / w_unit) if w_unit > 0 else float("nan")

    # Words written ratio (for copy and dictation)
    task_name = str(sentence_json.get("task_name", "")).lower()
    matched_task = None
    for k in EXPECTED_WORD_COUNTS:
        if k in task_name:
            matched_task = k
            break

    detected_words = float(sentence_json.get("qa_flags", {}).get("word_count_detected", 0))
    if detected_words == 0:
        all_words = []
        for l in lines:
            all_words.extend(l.get("words", []))
        detected_words = float(len(all_words))

    if matched_task and matched_task in EXPECTED_WORD_COUNTS:
        exp_words = EXPECTED_WORD_COUNTS[matched_task]
        words_written_ratio = float(detected_words / exp_words) if exp_words > 0 else float("nan")
    else:
        words_written_ratio = float("nan")

    return {
        "components_per_unit_width": components_per_unit_width,
        "junctions_per_unit_width": junctions_per_unit_width,
        "endpoints_per_unit_width": endpoints_per_unit_width,
        "words_written_ratio": words_written_ratio,
        "lines_used": lines_used,
    }

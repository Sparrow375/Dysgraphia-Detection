"""Segmentation module for Workstream A.

Implements:
1. Ink assignment to line bands between ruled lines.
2. Script classification per line (Devanagari vs. Latin) via shirorekha ratio.
3. Word segmentation (horizontal closing for Hindi, inter-word gap clustering for English).
4. Baseline and x-height estimation.
5. Sequential sentence assembly into the 6 standard task blocks:
   - sentence_01_copy_hindi
   - sentence_02_copy_english
   - sentence_03_dictated_hindi
   - sentence_04_dictated_english
   - sentence_05_own_hindi
   - sentence_06_own_english
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


TASK_TEMPLATE = [
    {"task_id": "sentence_01_copy_hindi", "task_name": "copy_hindi", "expected_script": "devanagari"},
    {"task_id": "sentence_02_copy_english", "task_name": "copy_english", "expected_script": "latin"},
    {"task_id": "sentence_03_dictated_hindi", "task_name": "dictated_hindi", "expected_script": "devanagari"},
    {"task_id": "sentence_04_dictated_english", "task_name": "dictated_english", "expected_script": "latin"},
    {"task_id": "sentence_05_own_hindi", "task_name": "own_hindi", "expected_script": "devanagari"},
    {"task_id": "sentence_06_own_english", "task_name": "own_english", "expected_script": "latin"},
]


def classify_line_script(line_ink: np.ndarray, threshold: float = 0.70) -> Tuple[str, float]:
    """Classify line script as 'devanagari' or 'latin' based on word-level shirorekha ratio.

    Returns:
        script ('devanagari' or 'latin'), shirorekha_score (float)
    """
    h, w = line_ink.shape[:2]
    if line_ink.sum() < 300:
        return "latin", 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_ink)
    comp_ratios = []

    for i in range(1, num_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter thin horizontal rule line remnants (aspect ratio > 10) and noise
        if (comp_w / max(comp_h, 1)) > 10.0 or area < 80 or comp_h < 18 or comp_w < 25:
            continue

        c_x = stats[i, cv2.CC_STAT_LEFT]
        c_y = stats[i, cv2.CC_STAT_TOP]
        comp_crop = line_ink[c_y : c_y + comp_h, c_x : c_x + comp_w]

        # Top 35% where the shirorekha headline is located
        top_slice = comp_crop[: max(2, int(comp_h * 0.35)), :]
        top_proj = (top_slice.sum(axis=0) > 0).astype(int)

        if top_proj.sum() == 0:
            comp_ratios.append(0.0)
            continue

        padded = np.pad(top_proj, (1, 1), "constant")
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        max_run = (ends - starts).max() if len(starts) > 0 else 0

        comp_ratios.append(max_run / comp_w)

    if not comp_ratios:
        return "latin", 0.0

    median_ratio = float(np.median(comp_ratios))
    # Threshold at 0.70 per plan
    if median_ratio >= threshold:
        return "devanagari", median_ratio

    return "latin", median_ratio


def estimate_line_baseline_and_xheight(
    line_ink: np.ndarray,
    script: str,
) -> Tuple[float, float]:
    """Estimate baseline y-coordinate and x-height h for a line crop.

    Returns:
        baseline_y (relative to line top), x_height_h
    """
    h, w = line_ink.shape[:2]
    if h < 10 or line_ink.sum() < 50:
        return float(h * 0.8), float(h * 0.5)

    v_proj = line_ink.sum(axis=1).astype(float)
    if v_proj.max() == 0:
        return float(h * 0.8), float(h * 0.5)

    # Smooth vertical profile
    kernel = np.ones(5) / 5.0
    smooth_proj = np.convolve(v_proj, kernel, mode="same")

    if script == "devanagari":
        # Headline is the peak in upper 40%
        upper_limit = max(3, int(h * 0.40))
        y_head = float(np.argmax(smooth_proj[:upper_limit]))

        # Baseline is the lower boundary of the core body
        core_region = smooth_proj[int(y_head) : int(h * 0.85)]
        if len(core_region) > 5:
            # Baseline where ink density drops below 25% of core peak
            core_peak = core_region.max()
            drops = np.where(core_region < 0.25 * core_peak)[0]
            if len(drops) > 0:
                y_base = y_head + float(drops[0])
            else:
                y_base = y_head + float(len(core_region))
        else:
            y_base = float(h * 0.75)

        x_height = max(15.0, y_base - y_head)
        return y_base, x_height
    else:
        # Latin: Find densest row band
        densest_idx = int(np.argmax(smooth_proj))
        peak_val = smooth_proj[densest_idx]

        # Scan downwards from peak to find baseline
        down_drops = np.where(smooth_proj[densest_idx:] < 0.30 * peak_val)[0]
        y_base = float(densest_idx + down_drops[0]) if len(down_drops) > 0 else float(h * 0.8)

        # Scan upwards from peak to find mean line
        up_drops = np.where(smooth_proj[:densest_idx] < 0.30 * peak_val)[0]
        y_top = float(up_drops[-1]) if len(up_drops) > 0 else float(h * 0.3)

        x_height = max(15.0, y_base - y_top)
        return y_base, x_height


def segment_words_devanagari(line_ink: np.ndarray) -> List[Dict[str, Any]]:
    """Segment Devanagari words using connected components after small horizontal closing."""
    # Close small shirorekha breaks (1x9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    closed = cv2.morphologyEx(line_ink, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    words: List[Dict[str, Any]] = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        # Filter stray ink dots
        if area < 50 or w < 15 or h < 12:
            continue

        words.append(
            {
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(area),
            }
        )

    # Sort left-to-right
    words.sort(key=lambda w_dict: w_dict["bbox"][0])
    for idx, w_dict in enumerate(words):
        w_dict["word_index"] = idx

    return words


def segment_words_english(line_ink: np.ndarray, x_height_h: float) -> List[Dict[str, Any]]:
    """Segment English words by clustering inter-character gaps using 2-component GMM."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_ink)
    components = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        if area < 25 or w < 4 or h < 8:
            continue
        components.append({"x": x, "y": y, "w": w, "h": h, "r": x + w, "b": y + h, "area": area})

    if not components:
        return []

    # Sort left to right
    components.sort(key=lambda c: c["x"])

    # Compute gaps between consecutive components
    gaps = []
    for j in range(len(components) - 1):
        gap = components[j + 1]["x"] - components[j]["r"]
        gaps.append(max(0, gap))

    # GMM clustering on gaps
    threshold = 0.5 * x_height_h
    pos_gaps = [g for g in gaps if g > 2]
    if len(pos_gaps) >= 6:
        try:
            X = np.array(pos_gaps).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)
            means = gmm.means_.flatten()
            if abs(means[0] - means[1]) > 5.0:
                # Separation threshold between the two cluster centers
                threshold = float(np.mean(means))
        except Exception:
            threshold = 0.5 * x_height_h

    # Group components separated by gaps < threshold into words
    words: List[Dict[str, Any]] = []
    current_comps = [components[0]]

    for j, gap in enumerate(gaps):
        if gap > threshold:
            # Word boundary
            min_x = min(c["x"] for c in current_comps)
            min_y = min(c["y"] for c in current_comps)
            max_r = max(c["r"] for c in current_comps)
            max_b = max(c["b"] for c in current_comps)
            tot_area = sum(c["area"] for c in current_comps)

            words.append(
                {
                    "bbox": [int(min_x), int(min_y), int(max_r - min_x), int(max_b - min_y)],
                    "area": int(tot_area),
                }
            )
            current_comps = [components[j + 1]]
        else:
            current_comps.append(components[j + 1])

    if current_comps:
        min_x = min(c["x"] for c in current_comps)
        min_y = min(c["y"] for c in current_comps)
        max_r = max(c["r"] for c in current_comps)
        max_b = max(c["b"] for c in current_comps)
        tot_area = sum(c["area"] for c in current_comps)
        words.append(
            {
                "bbox": [int(min_x), int(min_y), int(max_r - min_x), int(max_b - min_y)],
                "area": int(tot_area),
            }
        )

    for idx, w_dict in enumerate(words):
        w_dict["word_index"] = idx

    return words


def extract_lines_and_segment(
    ink_clean: np.ndarray,
    ruled_lines: List[Dict[str, Any]],
    median_spacing_r: float,
) -> List[Dict[str, Any]]:
    """Segment physical lines between ruled lines and extract words and metrics.

    Returns:
        List of line dicts ordered vertically from top to bottom.
    """
    h, w = ink_clean.shape[:2]
    y_rules = sorted([float(l["y_center"]) for l in ruled_lines])

    if len(y_rules) < 2:
        return []

    lines_out: List[Dict[str, Any]] = []

    for b in range(len(y_rules) - 1):
        y_top = max(0, int(y_rules[b]))
        y_bot = min(h, int(y_rules[b + 1]))

        # Expand line crop slightly to catch ascenders and descenders
        pad = int(median_spacing_r * 0.15)
        crop_top = max(0, y_top - pad)
        crop_bot = min(h, y_bot + pad)

        line_crop = ink_clean[crop_top:crop_bot, :]
        ink_pixels = int(line_crop.sum())

        # Skip blank/empty ruled lines
        if ink_pixels < 400:
            continue

        # Classify script
        script, score = classify_line_script(line_crop)

        # Baseline and x-height relative to line_crop top
        base_rel, x_height = estimate_line_baseline_and_xheight(line_crop, script)
        global_baseline = float(crop_top + base_rel)

        # Word segmentation
        if script == "devanagari":
            words = segment_words_devanagari(line_crop)
        else:
            words = segment_words_english(line_crop, x_height)

        # Shift word bboxes to global page coordinates
        for wd in words:
            wd["bbox"][1] += crop_top

        lines_out.append(
            {
                "rule_band_index": b,
                "bbox": [0, crop_top, w, crop_bot - crop_top],
                "y_top": crop_top,
                "y_bot": crop_bot,
                "ink_pixels": ink_pixels,
                "script": script,
                "script_confidence": score,
                "baseline_y": global_baseline,
                "x_height_h": float(x_height),
                "words": words,
                "word_count": len(words),
            }
        )

    return lines_out


def group_lines_into_sentence_blocks(
    lines: List[Dict[str, Any]],
    page_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Assemble segmented lines into the 6 canonical task sentences using a state machine."""
    if not lines:
        return []

    # Sequential state machine matching:
    # 0: Hindi Copy -> 1: English Copy -> 2: Hindi Dictated -> 3: English Dictated -> 4: Hindi Own -> 5: English Own
    sentence_blocks: List[Dict[str, Any]] = []
    current_task_idx = 0

    curr_lines = [lines[0]]
    curr_script = lines[0]["script"]

    for next_line in lines[1:]:
        next_script = next_line["script"]

        # If script changes, transition to the next task block
        if next_script != curr_script and current_task_idx < len(TASK_TEMPLATE) - 1:
            template = TASK_TEMPLATE[current_task_idx]
            sentence_blocks.append(
                _create_sentence_block(
                    template=template,
                    lines=curr_lines,
                    script=curr_script,
                    page_shape=page_shape,
                )
            )
            current_task_idx += 1
            curr_lines = [next_line]
            curr_script = next_script
        else:
            curr_lines.append(next_line)

    # Add final block
    if curr_lines:
        template = TASK_TEMPLATE[min(current_task_idx, len(TASK_TEMPLATE) - 1)]
        sentence_blocks.append(
            _create_sentence_block(
                template=template,
                lines=curr_lines,
                script=curr_script,
                page_shape=page_shape,
            )
        )

    return sentence_blocks


def _create_sentence_block(
    template: Dict[str, str],
    lines: List[Dict[str, Any]],
    script: str,
    page_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Helper to assemble a single sentence block dict."""
    h_page, w_page = page_shape[:2]
    y_min = max(0, min(l["y_top"] for l in lines) - 10)
    y_max = min(h_page, max(l["y_bot"] for l in lines) + 10)

    all_words = []
    for l in lines:
        all_words.extend(l["words"])

    # Bounding box around all ink words in the block
    if all_words:
        x_min = max(0, min(w["bbox"][0] for w in all_words) - 15)
        x_max = min(w_page, max(w["bbox"][0] + w["bbox"][2] for w in all_words) + 15)
    else:
        x_min, x_max = 50, w_page - 50

    crop_bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

    return {
        "task_id": template["task_id"],
        "task_name": template["task_name"],
        "script": script,
        "crop_bbox": crop_bbox,
        "lines": lines,
        "word_count": len(all_words),
    }

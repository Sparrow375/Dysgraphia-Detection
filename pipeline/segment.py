"""Segmentation module for Workstream A.

Implements:
1. Connected component analysis and filtering on cleaned ink mask.
2. Grouping components into physical text lines using nearest-neighbor y-clustering.
3. Script classification per line (Devanagari vs. Latin) via shirorekha ratio.
4. Word segmentation (horizontal closing for Hindi, inter-character gap clustering for English).
5. Per-word baseline sample mapping (bottom of bounding box) and robust linear baseline fitting.
6. Sequential assembly of physical lines into the 6 canonical task sentences:
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


def classify_line_script(line_ink: np.ndarray, threshold: float = 0.50) -> Tuple[str, float]:
    """Classify line script as 'devanagari' or 'latin' based on word-level shirorekha ratio.

    Returns:
        script ('devanagari' or 'latin'), shirorekha_score (float)
    """
    h, w = line_ink.shape[:2]
    if line_ink.sum() < 200:
        return "latin", 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_ink)
    comp_ratios = []

    for i in range(1, num_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter thin rule remnants and small noise
        if (comp_w / max(comp_h, 1)) > 8.0 or area < 60 or comp_h < 14 or comp_w < 18:
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

    kernel = np.ones(5) / 5.0
    smooth_proj = np.convolve(v_proj, kernel, mode="same")

    if script == "devanagari":
        upper_limit = max(3, int(h * 0.40))
        y_head = float(np.argmax(smooth_proj[:upper_limit]))
        core_region = smooth_proj[int(y_head) : int(h * 0.85)]
        if len(core_region) > 5:
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
        densest_idx = int(np.argmax(smooth_proj))
        peak_val = smooth_proj[densest_idx]

        down_drops = np.where(smooth_proj[densest_idx:] < 0.30 * peak_val)[0]
        y_base = float(densest_idx + down_drops[0]) if len(down_drops) > 0 else float(h * 0.8)

        up_drops = np.where(smooth_proj[:densest_idx] < 0.30 * peak_val)[0]
        y_top = float(up_drops[-1]) if len(up_drops) > 0 else float(h * 0.3)

        x_height = max(15.0, y_base - y_top)
        return y_base, x_height


def segment_words_devanagari(line_ink: np.ndarray) -> List[Dict[str, Any]]:
    """Segment Devanagari words using connected components after small horizontal closing."""
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

        if area < 40 or w < 12 or h < 10:
            continue

        words.append(
            {
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(area),
                "bottom": int(y + h),
                "cx": float(x + w / 2.0),
            }
        )

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

        if area < 20 or w < 3 or h < 8:
            continue
        components.append({"x": x, "y": y, "w": w, "h": h, "r": x + w, "b": y + h, "area": area})

    if not components:
        return []

    components.sort(key=lambda c: c["x"])

    gaps = []
    for j in range(len(components) - 1):
        gap = components[j + 1]["x"] - components[j]["r"]
        gaps.append(max(0, gap))

    threshold = max(12.0, 0.45 * x_height_h)
    pos_gaps = [g for g in gaps if g > 2]
    if len(pos_gaps) >= 6:
        try:
            X = np.array(pos_gaps).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)
            means = gmm.means_.flatten()
            if abs(means[0] - means[1]) > 5.0:
                threshold = float(np.mean(means))
        except Exception:
            pass

    words: List[Dict[str, Any]] = []
    current_comps = [components[0]]

    for j, gap in enumerate(gaps):
        if gap > threshold:
            min_x = min(c["x"] for c in current_comps)
            min_y = min(c["y"] for c in current_comps)
            max_r = max(c["r"] for c in current_comps)
            max_b = max(c["b"] for c in current_comps)
            tot_area = sum(c["area"] for c in current_comps)

            words.append(
                {
                    "bbox": [int(min_x), int(min_y), int(max_r - min_x), int(max_b - min_y)],
                    "area": int(tot_area),
                    "bottom": int(max_b),
                    "cx": float((min_x + max_r) / 2.0),
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
                "bottom": int(max_b),
                "cx": float((min_x + max_r) / 2.0),
            }
        )

    for idx, w_dict in enumerate(words):
        w_dict["word_index"] = idx

    return words


def extract_lines_and_segment(
    ink_clean: np.ndarray,
    ruled_lines: List[Dict[str, Any]],
    median_spacing_r: float,
    rule_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Segment physical text lines and extract words, components, and baseline models.

    Uses connected component clustering by y-centroid to assemble physical lines,
    filtering header tables, page boundaries, and ruled-line fragments.

    When rule_mask is provided (binary mask of ruled line positions), components
    whose pixels overlap >60% with rule pixels are discarded as ruled-line remnants.

    Returns:
        List of line dicts ordered vertically from top to bottom.
    """
    h_page, w_page = ink_clean.shape[:2]

    # Mask out outer page margins and top header table region
    clean_mask = ink_clean.copy()
    clean_mask[:320, :] = 0        # Skip header box at page top
    clean_mask[:, :225] = 0        # Skip left vertical notebook margin line
    clean_mask[:, 1950:] = 0       # Skip right margin edge

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask)
    comps = []

    for i in range(1, num_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        carea = stats[i, cv2.CC_STAT_AREA]
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        ccx, ccy = centroids[i]

        # --- Filter 1: Noise removal ---
        if carea < 25 or ch < 8 or cw < 6:
            continue

        # --- Filter 2: Reject giant border/margin line artifacts spanning multiple lines ---
        if ch > 1.8 * median_spacing_r or cw > 1200:
            continue

        # --- Filter 3: Reject thin horizontal ruled-line fragments ---
        aspect = cw / max(ch, 1)
        if aspect > 12.0 and ch < 20:
            continue

        comps.append(
            {
                "x": cx,
                "y": cy,
                "w": cw,
                "h": ch,
                "cx": float(ccx),
                "cy": float(ccy),
                "area": int(carea),
                "bottom": int(cy + ch),
            }
        )

    if not comps:
        return []


    # Sort components top to bottom
    comps.sort(key=lambda c: c["cy"])
    lines_raw = []
    y_thresh = median_spacing_r * 0.65

    for c in comps:
        assigned = False
        for line in lines_raw:
            line_cy = np.mean([it["cy"] for it in line])
            if abs(c["cy"] - line_cy) < y_thresh:
                line.append(c)
                assigned = True
                break
        if not assigned:
            lines_raw.append([c])

    # Keep lines with at least 2 components and > 450 px of ink
    valid_lines_raw = [l for l in lines_raw if len(l) >= 2 and sum(it["area"] for it in l) > 450]
    valid_lines_raw.sort(key=lambda line: np.mean([it["cy"] for it in line]))

    lines_out: List[Dict[str, Any]] = []

    for l_idx, line_comps in enumerate(valid_lines_raw):
        y_min = max(0, min(it["y"] for it in line_comps) - 10)
        y_max = min(h_page, max(it["bottom"] for it in line_comps) + 10)
        x_min = max(0, min(it["x"] for it in line_comps) - 15)
        x_max = min(w_page, max(it["x"] + it["w"] for it in line_comps) + 15)

        line_crop = ink_clean[y_min:y_max, x_min:x_max]

        # If we have a rule_mask, mask out ruled-line pixels from the line crop
        # so script classification and word segmentation operate on ink-only pixels
        if rule_mask is not None:
            rule_crop = rule_mask[y_min:y_max, x_min:x_max]
            line_crop_clean = np.where(rule_crop > 0, 0, line_crop).astype(np.uint8)
        else:
            line_crop_clean = line_crop

        ink_pixels = int(line_crop_clean.sum())

        script, score = classify_line_script(line_crop_clean)
        base_rel, x_height = estimate_line_baseline_and_xheight(line_crop_clean, script)
        global_baseline = float(y_min + base_rel)

        if script == "devanagari":
            words = segment_words_devanagari(line_crop_clean)
        else:
            words = segment_words_english(line_crop_clean, x_height)

        # Shift word bboxes to page coordinates
        for wd in words:
            wd["bbox"][0] += x_min
            wd["bbox"][1] += y_min
            wd["bottom"] += y_min
            wd["cx"] += x_min

        # Per-word baseline mapping (bottom of word bbox)
        word_bottoms = [float(wd["bottom"]) for wd in words]
        word_cxs = [float(wd["cx"]) for wd in words]

        # Fit robust line through word bottoms
        if len(word_cxs) >= 2:
            try:
                p = np.polyfit(word_cxs, word_bottoms, 1)
                slope = float(p[0])
                intercept = float(p[1])
                residuals = [float(b - (slope * x + intercept)) for x, b in zip(word_cxs, word_bottoms)]
                baseline_rmse = float(np.std(residuals))
            except Exception:
                slope = 0.0
                intercept = float(np.mean(word_bottoms)) if word_bottoms else global_baseline
                residuals = [0.0] * len(word_bottoms)
                baseline_rmse = 0.0
        else:
            slope = 0.0
            intercept = float(np.mean(word_bottoms)) if word_bottoms else global_baseline
            residuals = [0.0] * len(word_bottoms)
            baseline_rmse = 0.0

        lines_out.append(
            {
                "line_index": l_idx,
                "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                "y_top": int(y_min),
                "y_bot": int(y_max),
                "cy_mean": float(np.mean([it["cy"] for it in line_comps])),
                "ink_pixels": ink_pixels,
                "script": script,
                "script_confidence": score,
                "baseline_y": global_baseline,
                "baseline_slope": slope,
                "baseline_intercept": intercept,
                "baseline_rmse": baseline_rmse,
                "x_height_h": float(x_height),
                "words": words,
                "word_count": len(words),
                "word_baseline_points": list(zip(word_cxs, word_bottoms)),
                "word_baseline_residuals": residuals,
            }
        )

    return lines_out


def group_lines_into_sentence_blocks(
    lines: List[Dict[str, Any]],
    page_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Assemble physical lines into the 6 canonical task sentences.

    Follows the 6-task protocol:
    - Task 1: sentence_01_copy_hindi (1 line)
    - Task 2: sentence_02_copy_english (1-3 lines)
    - Task 3: sentence_03_dictated_hindi (1 line)
    - Task 4: sentence_04_dictated_english (1 line)
    - Task 5: sentence_05_own_hindi (1 line)
    - Task 6: sentence_06_own_english (1 line)
    """
    if not lines:
        return []

    h_page, w_page = page_shape[:2]
    N = len(lines)
    task_to_lines: Dict[int, List[Dict[str, Any]]] = {}

    if N >= 6:
        task_to_lines[0] = [lines[0]]
        # Task 2 absorbs multi-line English copy between line 0 and the final 4 single-line tasks
        task_to_lines[1] = lines[1 : N - 4]
        task_to_lines[2] = [lines[N - 4]]
        task_to_lines[3] = [lines[N - 3]]
        task_to_lines[4] = [lines[N - 2]]
        task_to_lines[5] = [lines[N - 1]]
    else:
        for idx, l in enumerate(lines):
            task_to_lines[idx] = [l]

    sentence_blocks: List[Dict[str, Any]] = []

    for t_idx, template in enumerate(TASK_TEMPLATE):
        if t_idx not in task_to_lines:
            continue

        assigned_lines = task_to_lines[t_idx]
        if not assigned_lines:
            continue

        y_min = max(0, min(l["y_top"] for l in assigned_lines) - 10)
        y_max = min(h_page, max(l["y_bot"] for l in assigned_lines) + 10)

        all_words = []
        for l in assigned_lines:
            all_words.extend(l["words"])

        if all_words:
            x_min = max(0, min(w["bbox"][0] for w in all_words) - 15)
            x_max = min(w_page, max(w["bbox"][0] + w["bbox"][2] for w in all_words) + 15)
        else:
            x_min = max(0, min(l["bbox"][0] for l in assigned_lines) - 15)
            x_max = min(w_page, max(l["bbox"][0] + l["bbox"][2] for l in assigned_lines) + 15)

        crop_bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

        # Collect per-word baseline points across the sentence
        all_word_cxs = [float(w["cx"]) for w in all_words]
        all_word_bottoms = [float(w["bottom"]) for w in all_words]

        if len(all_word_cxs) >= 2:
            try:
                p = np.polyfit(all_word_cxs, all_word_bottoms, 1)
                s_slope = float(p[0])
                s_intercept = float(p[1])
                s_residuals = [float(b - (s_slope * x + s_intercept)) for x, b in zip(all_word_cxs, all_word_bottoms)]
                s_rmse = float(np.std(s_residuals))
            except Exception:
                s_slope = 0.0
                s_intercept = float(np.mean(all_word_bottoms)) if all_word_bottoms else float(y_min)
                s_residuals = [0.0] * len(all_word_cxs)
                s_rmse = 0.0
        else:
            s_slope = 0.0
            s_intercept = float(np.mean(all_word_bottoms)) if all_word_bottoms else float(y_min)
            s_residuals = [0.0] * len(all_word_cxs)
            s_rmse = 0.0

        sentence_blocks.append(
            {
                "task_id": template["task_id"],
                "task_name": template["task_name"],
                "script": template["expected_script"],
                "crop_bbox": crop_bbox,
                "lines": assigned_lines,
                "word_count": len(all_words),
                "sentence_baseline_slope": s_slope,
                "sentence_baseline_intercept": s_intercept,
                "sentence_baseline_rmse": s_rmse,
                "word_baseline_points": list(zip(all_word_cxs, all_word_bottoms)),
                "word_baseline_residuals": s_residuals,
            }
        )

    return sentence_blocks

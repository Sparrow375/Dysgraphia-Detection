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


def classify_line_script(line_ink: np.ndarray, threshold: float = 0.14) -> Tuple[str, float]:
    """Classify line script as 'devanagari' or 'latin' based on shirorekha coverage ratio.

    In Devanagari handwriting, characters hang from a continuous headline (shirorekha),
    so words contain horizontal runs of >= 36px in the upper 35% of components, covering
    >= 14% of the total line ink. In Latin handwriting, characters sit on the baseline
    and words are split into separate letters without long continuous headlines.
    """
    h, w = line_ink.shape[:2]
    tot_ink = int(line_ink.sum())
    if tot_ink < 100 or h < 8:
        return "latin", 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_ink)
    shiro_ink = 0

    for i in range(1, num_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        carea = stats[i, cv2.CC_STAT_AREA]

        if comp_w >= 42 and comp_h >= 12 and (comp_w / max(comp_h, 1)) < 20.0:
            c_x = stats[i, cv2.CC_STAT_LEFT]
            c_y = stats[i, cv2.CC_STAT_TOP]
            comp_crop = line_ink[c_y : c_y + comp_h, c_x : c_x + comp_w]

            top_slice = comp_crop[: max(2, int(comp_h * 0.35)), :]
            top_proj = (top_slice.sum(axis=0) > 0).astype(int)

            if top_proj.sum() > 0:
                padded = np.pad(top_proj, (1, 1), "constant")
                diffs = np.diff(padded)
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                max_run = (ends - starts).max() if len(starts) > 0 else 0
                if max_run >= 36:
                    shiro_ink += carea

    ratio = shiro_ink / max(tot_ink, 1.0)
    is_dev = ratio >= threshold
    script = "devanagari" if is_dev else "latin"
    confidence = float(ratio if is_dev else (1.0 - ratio))

    return script, confidence


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
) -> List[Dict[str, Any]]:
    """Segment physical text lines and extract words, components, and baseline models.

    Uses connected component clustering with altitude proximity and a secondary line-merge
    pass to group words on the same line even across large horizontal gaps.

    Returns:
        List of line dicts ordered vertically from top to bottom.
    """
    h_page, w_page = ink_clean.shape[:2]

    # Mask out outer page margins and top header table region
    clean_mask = ink_clean.copy()
    clean_mask[:320, :] = 0        # Skip header box at page top
    clean_mask[:, :210] = 0        # Skip left vertical margin line and punch holes
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

        # Ignore tiny specks while preserving wide cursive and underlined strokes
        if carea >= 25 and ch >= 8 and (cw / max(ch, 1)) < 25.0:
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
    lines_raw: List[List[Dict[str, Any]]] = []
    y_thresh = median_spacing_r * 0.35

    for c in comps:
        best_line = None
        best_dist = 1e9
        for line in lines_raw:
            line_cy = np.mean([it["cy"] for it in line])
            dist = abs(c["cy"] - line_cy)
            if dist < y_thresh and dist < best_dist:
                best_dist = dist
                best_line = line
        if best_line is not None:
            best_line.append(c)
        else:
            lines_raw.append([c])

    # Secondary merge pass: merge raw line clusters that share the exact same altitude
    # Handles words at the same altitude separated by large horizontal gaps (e.g. 'my baber' and rest of sentence)
    # Constrained by maximum combined vertical span (<= 1.25 * r) so adjacent stacked lines never chain together
    merged = True
    while merged:
        merged = False
        for i in range(len(lines_raw)):
            for j in range(i + 1, len(lines_raw)):
                cy_i = np.mean([it["cy"] for it in lines_raw[i]])
                cy_j = np.mean([it["cy"] for it in lines_raw[j]])
                if abs(cy_i - cy_j) < 0.28 * median_spacing_r:
                    comb_y_min = min(min(it["y"] for it in lines_raw[i]), min(it["y"] for it in lines_raw[j]))
                    comb_y_max = max(max(it["bottom"] for it in lines_raw[i]), max(it["bottom"] for it in lines_raw[j]))
                    if (comb_y_max - comb_y_min) <= 1.25 * median_spacing_r:
                        lines_raw[i].extend(lines_raw[j])
                        lines_raw.pop(j)
                        merged = True
                        break
            if merged:
                break

    # Keep lines with at least 2 components and > 500 px of ink (filters empty ruled-line noise)
    valid_lines_raw = [l for l in lines_raw if len(l) >= 2 and sum(it["area"] for it in l) > 500]
    valid_lines_raw.sort(key=lambda line: np.mean([it["cy"] for it in line]))

    lines_out: List[Dict[str, Any]] = []

    for l_idx, line_comps in enumerate(valid_lines_raw):
        y_min = max(0, min(it["y"] for it in line_comps) - 10)
        y_max = min(h_page, max(it["bottom"] for it in line_comps) + 10)
        x_min = max(0, min(it["x"] for it in line_comps) - 15)
        x_max = min(w_page, max(it["x"] + it["w"] for it in line_comps) + 15)

        # Construct isolated line crop containing only line_comps to avoid bleeding from adjacent lines
        line_crop = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        for it in line_comps:
            cx, cy, cw, ch = it["x"], it["y"], it["w"], it["h"]
            comp_pixels = ink_clean[cy : cy + ch, cx : cx + cw]
            line_crop[cy - y_min : cy - y_min + ch, cx - x_min : cx - x_min + cw] |= comp_pixels

        ink_pixels = int(line_crop.sum())
        if ink_pixels < 350:
            continue

        script, score = classify_line_script(line_crop)
        base_rel, x_height = estimate_line_baseline_and_xheight(line_crop, script)
        global_baseline = float(y_min + base_rel)

        if script == "devanagari":
            words = segment_words_devanagari(line_crop)
        else:
            words = segment_words_english(line_crop, x_height)

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
    reference_rules: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Assemble physical lines into single-language sentence blocks.

    Language-Aware Grouping Rules:
    1. Each cropped sentence block contains strictly ONE language.
    2. Language changes (Hindi <-> English) mark sentence boundaries.
    3. Consecutive lines of the same language merge into the same multi-line sentence.
    4. Sentences are mapped to the canonical 6-task protocol.
    5. Baseline regression and rule-offset metrics are computed against the reference ruled lines layer.
    """
    if not lines:
        return []

    h_page, w_page = page_shape[:2]

    # 1. Group consecutive lines with the same script into single-language sentence blocks
    line_groups: List[List[Dict[str, Any]]] = []
    for l in lines:
        if not line_groups:
            line_groups.append([l])
        else:
            prev_script = line_groups[-1][-1]["script"]
            if l["script"] == prev_script:
                # Same language -> continuation of upper sentence
                line_groups[-1].append(l)
            else:
                # Language transition -> start new sentence block
                line_groups.append([l])

    sentence_blocks: List[Dict[str, Any]] = []

    for g_idx, assigned_lines in enumerate(line_groups):
        if g_idx < len(TASK_TEMPLATE):
            template = TASK_TEMPLATE[g_idx]
        else:
            template = {
                "task_id": f"sentence_{g_idx + 1:02d}_extra",
                "task_name": f"extra_task_{g_idx + 1}",
                "expected_script": assigned_lines[0]["script"],
            }

        all_words = []
        for l in assigned_lines:
            all_words.extend(l["words"])

        if all_words:
            x_min = max(0, min(w["bbox"][0] for w in all_words) - 15)
            x_max = min(w_page, max(w["bbox"][0] + w["bbox"][2] for w in all_words) + 15)
            y_min = max(0, min(w["bbox"][1] for w in all_words) - 15)
            y_max = min(h_page, max(w["bbox"][1] + w["bbox"][3] for w in all_words) + 15)
        else:
            x_min = max(0, min(l["bbox"][0] for l in assigned_lines) - 15)
            x_max = min(w_page, max(l["bbox"][0] + l["bbox"][2] for l in assigned_lines) + 15)
            y_min = max(0, min(l["y_top"] for l in assigned_lines) - 15)
            y_max = min(h_page, max(l["y_bot"] for l in assigned_lines) + 15)

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

        # Compute baseline-to-rule offset against reference ruled lines
        rule_offset_px = 0.0
        nearest_rule_y = float(y_min)
        rule_slope_diff = float(s_slope)
        per_word_rule_offsets: List[float] = []

        if reference_rules:
            mid_x = (x_min + x_max) / 2.0
            base_mid_y = s_slope * mid_x + s_intercept
            rule_dists = [
                abs((r.get("slope", 0.0) * mid_x + r.get("intercept", r["y_center"])) - base_mid_y)
                for r in reference_rules
            ]
            best_r_idx = int(np.argmin(rule_dists))
            nearest_rule = reference_rules[best_r_idx]
            nearest_rule_y = float(nearest_rule.get("slope", 0.0) * mid_x + nearest_rule.get("intercept", nearest_rule["y_center"]))
            rule_offset_px = float(base_mid_y - nearest_rule_y)
            rule_slope_diff = float(s_slope - nearest_rule.get("slope", 0.0))

            for cx, bot in zip(all_word_cxs, all_word_bottoms):
                r_y = nearest_rule.get("slope", 0.0) * cx + nearest_rule.get("intercept", nearest_rule["y_center"])
                per_word_rule_offsets.append(float(bot - r_y))

        rule_offset_rmse = float(np.std(per_word_rule_offsets)) if per_word_rule_offsets else 0.0

        sentence_blocks.append(
            {
                "task_id": template["task_id"],
                "task_name": template["task_name"],
                "script": assigned_lines[0]["script"],
                "crop_bbox": crop_bbox,
                "lines": assigned_lines,
                "word_count": len(all_words),
                "sentence_baseline_slope": s_slope,
                "sentence_baseline_intercept": s_intercept,
                "sentence_baseline_rmse": s_rmse,
                "word_baseline_points": list(zip(all_word_cxs, all_word_bottoms)),
                "word_baseline_residuals": s_residuals,
                "nearest_rule_y": nearest_rule_y,
                "rule_offset_px": rule_offset_px,
                "rule_slope_diff": rule_slope_diff,
                "per_word_rule_offsets": per_word_rule_offsets,
                "rule_offset_rmse": rule_offset_rmse,
            }
        )

    return sentence_blocks

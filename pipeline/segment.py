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


_OCR_READER = None


def get_ocr_reader():
    """Lazily initialize and cache EasyOCR reader for Hindi and English."""
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            _OCR_READER = easyocr.Reader(["hi", "en"], gpu=False, verbose=False)
        except Exception:
            _OCR_READER = False
    return _OCR_READER if _OCR_READER is not False else None


def classify_line_script(
    line_ink: np.ndarray,
    rule_crop: Optional[np.ndarray] = None,
    gray_crop: Optional[np.ndarray] = None,
    threshold: float = 0.50,
) -> Tuple[str, float]:
    """Classify text line as 'devanagari' or 'latin' using OCR with shirorekha fallback.

    Uses deep learning OCR (EasyOCR hi+en) to count recognized Devanagari vs Latin characters.
    Falls back to headline (shirorekha) continuity if OCR returns no characters.

    Returns:
        script ('devanagari' or 'latin'), confidence_logit (float)
    """
    # --- Priority 1: High-Confidence OCR Script Identification ---
    reader = get_ocr_reader()
    if reader is not None:
        try:
            if gray_crop is not None and gray_crop.size > 0:
                img_ocr = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB) if len(gray_crop.shape) == 2 else gray_crop
            else:
                img_ocr = cv2.cvtColor((255 - line_ink * 255), cv2.COLOR_GRAY2RGB)

            # Slicing line width to first 800px delivers 2.5x faster inference while capturing first 2-3 words
            img_slice = img_ocr[:, :min(img_ocr.shape[1], 800)]
            results = reader.readtext(img_slice)
            all_text = " ".join([r[1] for r in results])
            n_dev = sum(1 for c in all_text if "\u0900" <= c <= "\u097F")
            n_lat = sum(1 for c in all_text if c.isascii() and c.isalpha())

            if n_dev > 0 or n_lat > 0:
                if n_dev > n_lat:
                    conf = 4.0 + min(6.0, (n_dev - n_lat) * 0.5)
                    return "devanagari", float(conf)
                else:
                    conf = -4.0 - min(6.0, (n_lat - n_dev) * 0.5)
                    return "latin", float(conf)
        except Exception:
            pass

    # --- Priority 2: Morphological Shirorekha & Projection Fallback ---
    if rule_crop is not None:
        ink = np.where(rule_crop > 0, 0, line_ink).astype(np.uint8)
    else:
        ink = line_ink.copy()

    h, w = ink.shape[:2]
    if ink.sum() < 200:
        return "latin", -3.0

    # Filter thin horizontal rule remnants (aspect > 5.5 and height < 32)
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    clean = ink.copy()
    for i in range(1, num_cc):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if (cw / max(ch, 1)) > 5.5 and ch < 32:
            clean[labels == i] = 0

    # Filter out long continuous rule lines (> 160px)
    kernel_rule = cv2.getStructuringElement(cv2.MORPH_RECT, (160, 1))
    long_rules = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_rule)
    if long_rules.sum() > 0:
        clean = np.where(
            cv2.dilate(long_rules, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))) > 0,
            0,
            clean,
        ).astype(np.uint8)

    proj_y = clean.sum(axis=1)
    nz = np.where(proj_y > 0)[0]
    if len(nz) < 10:
        return "latin", -2.0

    y1, y2 = nz[0], nz[-1] + 1
    clean_t = clean[y1:y2, :]
    th = y2 - y1
    p_trim = proj_y[y1:y2]

    # Group into words for shirorekha continuity analysis
    kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    dilated = cv2.dilate(clean_t, kernel_word)
    num_w, labels_w, stats_w, _ = cv2.connectedComponentsWithStats(dilated)

    total_w = 0
    total_run = 0
    strong_dev = 0

    for wi in range(1, num_w):
        ww = stats_w[wi, cv2.CC_STAT_WIDTH]
        wh = stats_w[wi, cv2.CC_STAT_HEIGHT]
        wx = stats_w[wi, cv2.CC_STAT_LEFT]
        wy = stats_w[wi, cv2.CC_STAT_TOP]
        if ww < 45 or wh < 15:
            continue
        crop_w = clean_t[wy : wy + wh, wx : wx + ww]
        top_slice = crop_w[: max(2, int(wh * 0.42)), :]
        top_proj = (top_slice.sum(axis=0) > 0).astype(int)
        padded = np.pad(top_proj, (1, 1), "constant")
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        mr = (ends - starts).max() if len(starts) > 0 else 0
        total_w += ww
        total_run += mr
        # Word has strong shirorekha if run >= 45px, covers >= 40% of word, and <= 350px
        if 45 <= mr <= 350 and (mr / float(ww)) >= 0.40:
            strong_dev += 1

    ratio = (total_run / float(total_w)) if total_w > 0 else 0.0
    top_40_pct = p_trim[: max(2, int(th * 0.40))].sum() / max(p_trim.sum(), 1)
    logit = strong_dev * 2.5 + (ratio - 0.28) * 10.0 + (top_40_pct - 0.35) * 5.0
    prob_dev = float(1.0 / (1.0 + np.exp(-logit)))

    is_dev = prob_dev >= threshold
    return ("devanagari" if is_dev else "latin"), float(logit)



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
    grayscale_deskewed: Optional[np.ndarray] = None,
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

    # Keep lines with at least 2 components OR substantial text area (> 800px)
    valid_lines_raw = [l for l in lines_raw if len(l) >= 2 or sum(it["area"] for it in l) > 800]
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
            rule_crop = None
            line_crop_clean = line_crop

        if grayscale_deskewed is not None:
            gray_crop = grayscale_deskewed[y_min:y_max, x_min:x_max]
        else:
            gray_crop = None

        ink_pixels = int(line_crop_clean.sum())

        script, score = classify_line_script(line_crop, rule_crop, gray_crop=gray_crop)
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
                "p_dev": float(1.0 / (1.0 + np.exp(-score))),
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
    grade: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Assemble physical lines into canonical task sentences.

    Enforces the fundamental task protocol:
    1. Tasks strictly alternate in language: Hindi -> English -> Hindi -> English (-> Hindi -> English).
    2. Grade 3 students wrote 4 sentences (Tasks 1-4).
    3. Grades 4-7 students wrote up to 6 sentences (Tasks 1-6).
    4. Multi-line sentences (e.g. English copy spanning 2-3 lines) are globally partitioned
       using dynamic programming to guarantee zero language swallowing.
    """
    if not lines:
        return []

    h_page, w_page = page_shape[:2]

    # Filter out page footer artifacts and blank ruled-line noise
    cutoff_y = 1250 if (grade == 3) else 2400
    filtered = [
        l for l in lines
        if l["y_top"] < cutoff_y and not (l.get("word_count", 0) <= 1 and l.get("ink_pixels", 0) < 6000)
    ]
    if not filtered:
        filtered = lines

    K = len(filtered)
    # Target task count: Grade 3 is strictly 4 tasks. Grades 4-7 is up to 6 tasks.
    if grade == 3 or K < 6:
        T = min(4, K)
    else:
        T = min(6, K)

    target_templates = TASK_TEMPLATE[:T]
    target_scripts = [t["expected_script"] for t in target_templates]

    # Dynamic Programming Alternating Partition
    eps = 1e-4
    dp = np.full((T + 1, K + 1), 1e9)
    parent = np.zeros((T + 1, K + 1), dtype=int)
    dp[0, 0] = 0.0

    for t in range(1, T + 1):
        is_dev = (target_scripts[t - 1] == "devanagari")
        for i in range(t, K + 1):
            for k in range(t - 1, i):
                n_lines = i - k
                reg_penalty = 0.0
                # Task 2 (1-indexed) is English copy, which naturally spans multiple lines.
                # Mild penalty for grouping multiple lines into single-line tasks.
                if t != 2 and n_lines > 1:
                    reg_penalty = 0.5 * (n_lines - 1)
                cost = reg_penalty
                for l_idx in range(k, i):
                    p_dev = filtered[l_idx].get("p_dev")
                    if p_dev is None:
                        sc = filtered[l_idx].get("script_confidence", 0.0)
                        p_dev = float(1.0 / (1.0 + np.exp(-sc)))
                    p = p_dev if is_dev else (1.0 - p_dev)
                    cost -= np.log(max(p, eps))

                if dp[t - 1, k] + cost < dp[t, i]:
                    dp[t, i] = dp[t - 1, k] + cost
                    parent[t, i] = k

    split_indices = [K]
    cur = K
    for t in range(T, 0, -1):
        prev = int(parent[t, cur])
        split_indices.append(prev)
        cur = prev
    split_indices.reverse()

    sentence_blocks: List[Dict[str, Any]] = []

    for t in range(T):
        s_idx, e_idx = split_indices[t], split_indices[t + 1]
        assigned_lines = filtered[s_idx:e_idx]
        if not assigned_lines:
            continue
        template = target_templates[t]

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

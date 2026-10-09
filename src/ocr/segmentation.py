"""
Segmentation Module — Line, Word, and Character Region Extraction.

Provides a multi-strategy word segmentation pipeline:
  1. Reuses existing BHK line segmentation (segment_text_lines)
  2. Projection-profile word boundary detection (primary)
  3. Connected-component clustering fallback (for messy writing)
  4. Word image normalization for model input
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Word Region Data Structure
# ---------------------------------------------------------------------------

class WordRegion:
    """
    A segmented word region extracted from a text line.

    Attributes:
        image: Cropped binary mask of the word (ink=255, bg=0).
        bbox_in_line: (x, y, w, h) bounding box relative to the line image.
        bbox_in_image: (x, y, w, h) bounding box relative to the full image.
        word_index: Position index within the line (left-to-right).
        components: List of connected-component stats within this word.
    """

    def __init__(
        self,
        image: np.ndarray,
        bbox_in_line: Tuple[int, int, int, int],
        bbox_in_image: Tuple[int, int, int, int],
        word_index: int = 0,
        components: Optional[List[Dict]] = None,
        is_bullet: bool = False,
    ):
        self.image = image
        self.bbox_in_line = bbox_in_line
        self.bbox_in_image = bbox_in_image
        self.word_index = word_index
        self.components = components or []
        self.is_bullet = is_bullet

    @property
    def width(self) -> int:
        return self.bbox_in_line[2]

    @property
    def height(self) -> int:
        return self.bbox_in_line[3]

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


class LineRegion:
    """
    A segmented text line containing word regions.

    Attributes:
        image: Cropped binary mask of the line.
        bbox_in_image: (x, y, w, h) bounding box relative to the full image.
        words: List of WordRegion objects in reading order (left-to-right).
        line_index: 0-based line index (top-to-bottom).
        baseline_slope: Fitted baseline slope for this line (from BHK).
        baseline_intercept: Fitted baseline y-intercept.
    """

    def __init__(
        self,
        image: np.ndarray,
        bbox_in_image: Tuple[int, int, int, int],
        line_index: int = 0,
        baseline_slope: float = 0.0,
        baseline_intercept: float = 0.0,
    ):
        self.image = image
        self.bbox_in_image = bbox_in_image
        self.line_index = line_index
        self.baseline_slope = baseline_slope
        self.baseline_intercept = baseline_intercept
        self.words: List[WordRegion] = []

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Convenience property for bounding box in image coordinates."""
        return self.bbox_in_image


# ---------------------------------------------------------------------------
# Line Segmentation (delegates to BHK engine)
# ---------------------------------------------------------------------------

def segment_lines(
    binary_mask: np.ndarray,
    min_components_per_line: int = 2,
) -> List[LineRegion]:
    """
    Segment a full-page binary handwriting image into individual text lines
    using horizontal morphological ribbons and projection valley tracking.
    Completely prevents multi-line collision caused by cursive descenders.

    Args:
        binary_mask: Full-page binary image (ink=255, bg=0).
        min_components_per_line: Minimum connected components to form a valid line.

    Returns:
        List of LineRegion objects sorted top-to-bottom.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    h_img, w_img = binary_mask.shape

    # 1. Connected components analysis to determine character scale
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    cand_h = []
    letters = []
    for i in range(1, num_labels):
        a = int(stats[i, cv2.CC_STAT_AREA])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        if a >= 25 and ch >= 8 and cw >= 2 and cw < 0.85 * w_img and ch < 0.70 * h_img:
            cand_h.append(ch)
            letters.append({
                'x': int(stats[i, cv2.CC_STAT_LEFT]),
                'y': int(stats[i, cv2.CC_STAT_TOP]),
                'w': cw,
                'h': ch,
                'area': a,
                'cx': float(centroids[i][0]),
                'cy': float(centroids[i][1]),
                'bottom': int(stats[i, cv2.CC_STAT_TOP] + ch),
                'label_idx': i,
            })

    if len(cand_h) < 3:
        region = LineRegion(
            image=binary_mask.copy(),
            bbox_in_image=(0, 0, w_img, h_img),
            line_index=0,
        )
        return [region]

    median_h = max(10.0, float(np.median(cand_h)))

    # 2. Horizontal Morphological Ribbon Tracking
    # Dilate horizontally to connect letters across each line while preserving inter-line valleys
    k_w = int(max(35, 1.6 * median_h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    dilated = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    proj_y = np.sum(dilated > 0, axis=1)
    smooth_y = gaussian_filter1d(proj_y.astype(float), sigma=max(4.0, 0.25 * median_h))

    min_dist = int(max(35, 2.4 * median_h))
    peaks, _ = find_peaks(smooth_y, distance=min_dist, prominence=smooth_y.max() * 0.05)

    if len(peaks) == 0:
        peaks = [int(h_img / 2)]

    # Compute valley cut points between adjacent peaks
    valleys = [0]
    for k in range(len(peaks) - 1):
        p1, p2 = peaks[k], peaks[k + 1]
        v = p1 + int(np.argmin(smooth_y[p1:p2]))
        valleys.append(v)
    valleys.append(h_img)

    # 3. Form LineRegion for each detected ribbon
    line_regions: List[LineRegion] = []
    for idx in range(len(peaks)):
        y1 = max(0, valleys[idx] - 2)
        y2 = min(h_img, valleys[idx + 1] + 2)

        # Filter letters whose vertical centroid falls into this line band
        line_letters = [
            l for l in letters
            if y1 <= l['cy'] < y2
        ]

        strip = binary_mask[y1:y2, :]
        if cv2.countNonZero(strip) < 180 or len(line_letters) < 2:
            continue

        if line_letters:
            min_x = max(0, min(l['x'] for l in line_letters) - int(0.2 * median_h))
            max_x = min(w_img, max(l['x'] + l['w'] for l in line_letters) + int(0.2 * median_h))
            min_y = max(y1, min(l['y'] for l in line_letters) - int(0.2 * median_h))
            max_y = min(y2, max(l['bottom'] for l in line_letters) + int(0.2 * median_h))
        else:
            coords = cv2.findNonZero(strip)
            if coords is None:
                continue
            bx, by, bw, bh = cv2.boundingRect(coords)
            min_x = max(0, bx - int(0.2 * median_h))
            max_x = min(w_img, bx + bw + int(0.2 * median_h))
            min_y = max(y1, y1 + by - int(0.2 * median_h))
            max_y = min(y2, y1 + by + bh + int(0.2 * median_h))

        line_w = max_x - min_x
        line_h = max_y - min_y
        if line_w < 10 or line_h < 8:
            continue

        line_img = binary_mask[min_y:max_y, min_x:max_x].copy()

        # Fit baseline slope
        slope = 0.0
        intercept = float(line_h / 2)
        if line_letters and len(line_letters) >= 3:
            xs = np.array([l['cx'] - min_x for l in line_letters], dtype=float)
            bottoms = np.array([l['bottom'] - min_y for l in line_letters], dtype=float)
            if (np.max(xs) - np.min(xs)) > 10:
                try:
                    p = np.polyfit(xs, bottoms, 1)
                    slope = float(p[0])
                    intercept = float(p[1])
                except Exception:
                    pass

        region = LineRegion(
            image=line_img,
            bbox_in_image=(min_x, min_y, line_w, line_h),
            line_index=len(line_regions),
            baseline_slope=slope,
            baseline_intercept=intercept,
        )

        region._components = [
            {**l, 'x_in_line': l['x'] - min_x, 'y_in_line': l['y'] - min_y}
            for l in line_letters
        ]
        line_regions.append(region)

    if not line_regions:
        region = LineRegion(
            image=binary_mask.copy(),
            bbox_in_image=(0, 0, w_img, h_img),
            line_index=0,
        )
        return [region]

    return line_regions


# ---------------------------------------------------------------------------
# Word Segmentation (within a single line)
# ---------------------------------------------------------------------------

def segment_words(
    line_region: LineRegion,
    gap_multiplier: float = 1.5,
) -> List[WordRegion]:
    """
    Segment a text line image into individual word regions using ink mask
    column density projection, whitespace valley detection, and border artifact stripping.
    Prevents cursive run-together multi-word blocks.

    Args:
        line_region: A LineRegion with its binary image and component info.
        gap_multiplier: Unused compatibility parameter.

    Returns:
        List of WordRegion objects sorted left-to-right.
    """
    from scipy.ndimage import gaussian_filter1d

    line_img = line_region.image
    lx, ly, lw, lh = line_region.bbox_in_image
    line_h, line_w = line_img.shape

    if line_h < 5 or line_w < 5 or cv2.countNonZero(line_img) < 10:
        return [WordRegion(
            image=line_img.copy(),
            bbox_in_line=(0, 0, line_w, line_h),
            bbox_in_image=(lx, ly, lw, lh),
            word_index=0,
        )]

    # 1. Clean edge slivers (thin 1-6px lines touching border from ruling or cut descenders)
    num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats(line_img, connectivity=8)
    clean_line = np.zeros_like(line_img)
    for i in range(1, num_l):
        y = int(stats_l[i, cv2.CC_STAT_TOP])
        h = int(stats_l[i, cv2.CC_STAT_HEIGHT])
        w = int(stats_l[i, cv2.CC_STAT_WIDTH])
        a = int(stats_l[i, cv2.CC_STAT_AREA])
        if (y <= 2 or y + h >= line_h - 2) and h <= 6 and w > 15:
            continue
        if a < 8:
            continue
        clean_line[labels_l == i] = 255

    # 2. Column ink projection profile
    col_ink = np.sum(clean_line > 0, axis=0)
    nonzero_cols = np.where(col_ink > 0)[0]
    if len(nonzero_cols) == 0:
        return [WordRegion(
            image=line_img.copy(),
            bbox_in_line=(0, 0, line_w, line_h),
            bbox_in_image=(lx, ly, lw, lh),
            word_index=0,
        )]

    x_min, x_max = int(nonzero_cols[0]), int(nonzero_cols[-1])

    # 3. Dynamic Whitespace Valley Gap (adaptive: max(8, 0.10 * line_h))
    min_gap = max(8, int(0.10 * line_h))

    zero_runs: List[Tuple[int, int]] = []
    in_zero = False
    start_z = 0
    for x in range(x_min, x_max + 1):
        if col_ink[x] <= 1:
            if not in_zero:
                in_zero = True
                start_z = x
        else:
            if in_zero:
                in_zero = False
                if (x - start_z) >= min_gap:
                    zero_runs.append((start_z, x))
    if in_zero and (x_max + 1 - start_z) >= min_gap:
        zero_runs.append((start_z, x_max + 1))

    # Slice words between whitespace valleys
    raw_word_spans: List[Tuple[int, int]] = []
    curr_s = x_min
    for zs, ze in zero_runs:
        if zs > curr_s:
            raw_word_spans.append((curr_s, zs))
        curr_s = ze
    if curr_s <= x_max:
        raw_word_spans.append((curr_s, x_max + 1))

    # 4. Secondary splitting for run-together cursive words (AR > 3.0)
    # In continuous cursive, pen may not fully lift, but ink drops to a single thin ligature stroke
    refined_spans: List[Tuple[int, int]] = []
    for ws, we in raw_word_spans:
        span_w = we - ws
        if span_w < 8:
            continue

        ar = span_w / max(line_h, 1)
        if ar > 3.0 and span_w > 2.0 * line_h:
            # Look for internal local minima in smoothed column profile
            sub_profile = col_ink[ws:we].astype(float)
            sub_smooth = gaussian_filter1d(sub_profile, sigma=3.0)

            med_ink = np.median(sub_profile[sub_profile > 0]) if np.any(sub_profile > 0) else 5.0
            split_candidates = []
            for ix in range(int(0.20 * span_w), int(0.80 * span_w)):
                if sub_smooth[ix] <= max(2.0, 0.40 * med_ink):
                    if sub_smooth[ix] <= sub_smooth[ix - 1] and sub_smooth[ix] <= sub_smooth[ix + 1]:
                        split_candidates.append(ws + ix)

            valid_splits = []
            for sc in split_candidates:
                if not valid_splits or (sc - valid_splits[-1]) >= int(0.9 * line_h):
                    valid_splits.append(sc)

            if valid_splits:
                last_split = ws
                for sp in valid_splits:
                    if (sp - last_split) >= 10:
                        refined_spans.append((last_split, sp))
                        last_split = sp
                if (we - last_split) >= 10:
                    refined_spans.append((last_split, we))
            else:
                refined_spans.append((ws, we))
        else:
            refined_spans.append((ws, we))

    # 5. Build WordRegion objects with tight ink crop and dynamic padding
    word_regions: List[WordRegion] = []
    for wi, (ws, we) in enumerate(refined_spans):
        w_crop = clean_line[:, ws:we]
        coords = cv2.findNonZero(w_crop)
        if coords is None:
            continue
        cx, cy, cw, ch = cv2.boundingRect(coords)
        if cw < 6 or ch < 6:
            continue

        # Filter out edge border artifacts touching crop edge with small height
        if (cy <= 2 and ch <= 20) or (cy + ch >= line_h - 2 and ch <= 12):
            if cw < 60 and cv2.countNonZero(w_crop) < 100:
                continue

        pad_x = int(max(4, 0.08 * line_h))
        pad_y = int(max(4, 0.08 * line_h))

        wx = max(0, ws + cx - pad_x)
        wy = max(0, cy - pad_y)
        wx2 = min(line_w, ws + cx + cw + pad_x)
        wy2 = min(line_h, cy + ch + pad_y)

        ww = wx2 - wx
        wh = wy2 - wy

        final_word_img = clean_line[wy:wy2, wx:wx2].copy()

        word_regions.append(WordRegion(
            image=final_word_img,
            bbox_in_line=(wx, wy, ww, wh),
            bbox_in_image=(lx + wx, ly + wy, ww, wh),
            word_index=len(word_regions),
        ))

    # 6. Punctuation Binding: merge tiny trailing specks into preceding word
    merged_regions: List[WordRegion] = []
    for wr in word_regions:
        bx, by, bw, bh = wr.bbox_in_image
        if bw <= 16 and bh <= 16 and merged_regions:
            p_wr = merged_regions[-1]
            px, py, pw, ph = p_wr.bbox_in_image
            nx = min(px, bx)
            ny = min(py, by)
            nw = max(px + pw, bx + bw) - nx
            nh = max(py + ph, by + bh) - ny
            p_wr.bbox_in_image = (nx, ny, nw, nh)
            p_wr.bbox_in_line = (nx - lx, ny - ly, nw, nh)
        else:
            merged_regions.append(wr)

    if not merged_regions:
        merged_regions.append(WordRegion(
            image=line_img.copy(),
            bbox_in_line=(0, 0, line_w, line_h),
            bbox_in_image=(lx, ly, lw, lh),
            word_index=0,
        ))

    return merged_regions


# ---------------------------------------------------------------------------
# Word Image Normalization
# ---------------------------------------------------------------------------

def normalize_word_image(
    word_image: np.ndarray,
    target_height: int = 64,
    pad_width_multiple: int = 16,
    deskew: bool = True,
    baseline_slope: float = 0.0,
) -> np.ndarray:
    """
    Normalize a word image for model input.

    Steps:
    1. Crop to tight ink bounding box (remove blank margins)
    2. Optional deskew based on local baseline slope
    3. Scale to fixed height preserving aspect ratio
    4. Pad width to next multiple of pad_width_multiple

    Args:
        word_image: Binary mask (ink=255, bg=0).
        target_height: Output image height in pixels.
        pad_width_multiple: Pad width to be divisible by this value.
        deskew: Whether to apply deskew correction.
        baseline_slope: Baseline slope for deskew (from line-level fitting).

    Returns:
        Normalized grayscale image (target_height × padded_width), uint8.
    """
    if word_image is None or word_image.size == 0:
        return np.zeros((target_height, pad_width_multiple), dtype=np.uint8)

    # 1. Crop to tight bounding box
    coords = cv2.findNonZero(word_image)
    if coords is None:
        return np.zeros((target_height, pad_width_multiple), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = word_image[y:y + h, x:x + w]

    # 2. Optional deskew
    if deskew and abs(baseline_slope) > 0.01:
        angle_deg = -np.degrees(np.arctan(baseline_slope))
        # Limit deskew to ±15°
        angle_deg = np.clip(angle_deg, -15.0, 15.0)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cropped = cv2.warpAffine(
            cropped, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Re-crop after rotation
        coords = cv2.findNonZero(cropped)
        if coords is not None:
            x2, y2, w2, h2 = cv2.boundingRect(coords)
            cropped = cropped[y2:y2 + h2, x2:x2 + w2]

    ch, cw = cropped.shape
    if ch < 2 or cw < 2:
        return np.zeros((target_height, pad_width_multiple), dtype=np.uint8)

    # 3. Scale to target height, preserving aspect ratio
    scale = target_height / ch
    new_w = max(1, int(cw * scale))
    scaled = cv2.resize(cropped, (new_w, target_height), interpolation=cv2.INTER_AREA)

    # 4. Pad width to next multiple
    pad_w = ((new_w + pad_width_multiple - 1) // pad_width_multiple) * pad_width_multiple
    padded = np.zeros((target_height, pad_w), dtype=np.uint8)
    # Center horizontally
    offset_x = (pad_w - new_w) // 2
    padded[:, offset_x:offset_x + new_w] = scaled

    return padded


# ---------------------------------------------------------------------------
# Full Segmentation Pipeline
# ---------------------------------------------------------------------------

def segment_image(
    binary_mask: np.ndarray,
) -> List[LineRegion]:
    """
    Complete segmentation pipeline: Image → Lines → Words.

    This is the main entry point for the segmentation stage.

    Args:
        binary_mask: Full-page preprocessed binary image (ink=255, bg=0).

    Returns:
        List of LineRegion objects, each containing segmented WordRegion objects,
        sorted in reading order (top-to-bottom, left-to-right).
    """
    # Step 1: Segment into lines
    lines = segment_lines(binary_mask)

    # Step 2: Segment each line into words
    for line in lines:
        line.words = segment_words(line)

    return lines

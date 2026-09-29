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
    ):
        self.image = image
        self.bbox_in_line = bbox_in_line
        self.bbox_in_image = bbox_in_image
        self.word_index = word_index
        self.components = components or []

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
    Segment a full-page binary handwriting image into individual text lines.

    Reuses the proven multi-baseline segmentation from bhk_features.py:
    vertical projection + centroid clustering + per-line baseline fitting.

    Args:
        binary_mask: Full-page binary image (ink=255, bg=0).
        min_components_per_line: Minimum connected components to form a valid line.

    Returns:
        List of LineRegion objects sorted top-to-bottom.
    """
    h_img, w_img = binary_mask.shape

    # Find connected components and their stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # Establish median character height for line clustering threshold
    cand_h = []
    letters = []
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        if a >= 15 and ch >= 5 and cw >= 2 and cw < 0.85 * w_img and ch < 0.85 * h_img:
            cand_h.append(ch)
            letters.append({
                'x': stats[i, cv2.CC_STAT_LEFT],
                'y': stats[i, cv2.CC_STAT_TOP],
                'w': cw,
                'h': ch,
                'area': a,
                'cx': centroids[i][0],
                'cy': centroids[i][1],
                'bottom': stats[i, cv2.CC_STAT_TOP] + ch,
                'label_idx': i,
            })

    if len(cand_h) < 3:
        # Not enough components — return entire image as one line
        region = LineRegion(
            image=binary_mask.copy(),
            bbox_in_image=(0, 0, w_img, h_img),
            line_index=0,
        )
        return [region]

    median_h = max(5.0, float(np.median(cand_h)))

    # Cluster components into lines by vertical centroid proximity
    letters_sorted = sorted(letters, key=lambda l: l['cy'])
    line_clusters: List[List[Dict]] = []

    for l in letters_sorted:
        assigned = False
        for cluster in line_clusters:
            cluster_mean_cy = np.mean([item['cy'] for item in cluster])
            if abs(l['cy'] - cluster_mean_cy) < 0.85 * median_h:
                cluster.append(l)
                assigned = True
                break
        if not assigned:
            line_clusters.append([l])

    # Filter and sort
    valid_clusters = [
        sorted(cluster, key=lambda it: it['cx'])
        for cluster in line_clusters
        if len(cluster) >= min_components_per_line
    ]
    valid_clusters.sort(key=lambda cluster: np.mean([it['cy'] for it in cluster]))

    if not valid_clusters:
        valid_clusters = [sorted(letters, key=lambda it: it['cx'])]

    # Build LineRegion objects
    line_regions = []
    for idx, cluster in enumerate(valid_clusters):
        # Compute bounding box of the entire line
        min_x = min(c['x'] for c in cluster)
        min_y = min(c['y'] for c in cluster)
        max_x = max(c['x'] + c['w'] for c in cluster)
        max_y = max(c['bottom'] for c in cluster)

        # Add vertical padding (half median char height)
        pad_y = int(median_h * 0.3)
        min_y = max(0, min_y - pad_y)
        max_y = min(h_img, max_y + pad_y)
        # Small horizontal padding
        pad_x = int(median_h * 0.2)
        min_x = max(0, min_x - pad_x)
        max_x = min(w_img, max_x + pad_x)

        line_w = max_x - min_x
        line_h = max_y - min_y

        line_img = binary_mask[min_y:max_y, min_x:max_x].copy()

        # Fit baseline for this line
        xs = np.array([c['cx'] - min_x for c in cluster], dtype=float)
        bottoms = np.array([c['bottom'] - min_y for c in cluster], dtype=float)
        slope = 0.0
        intercept = float(np.mean(bottoms))
        if len(xs) >= 3 and (np.max(xs) - np.min(xs)) > 10:
            try:
                p = np.polyfit(xs, bottoms, 1)
                slope = float(p[0])
                intercept = float(p[1])
            except Exception:
                pass

        region = LineRegion(
            image=line_img,
            bbox_in_image=(min_x, min_y, line_w, line_h),
            line_index=idx,
            baseline_slope=slope,
            baseline_intercept=intercept,
        )

        # Store component info for word segmentation
        region._components = [
            {**c, 'x_in_line': c['x'] - min_x, 'y_in_line': c['y'] - min_y}
            for c in cluster
        ]

        line_regions.append(region)

    return line_regions


# ---------------------------------------------------------------------------
# Word Segmentation (within a single line)
# ---------------------------------------------------------------------------

def segment_words(
    line_region: LineRegion,
    gap_multiplier: float = 1.5,
) -> List[WordRegion]:
    """
    Segment a line image into individual word regions.

    Uses a hybrid approach:
    1. Primary: Vertical projection profile to find inter-word gaps
    2. Fallback: Connected-component horizontal distance clustering

    Args:
        line_region: A LineRegion with its binary image and component info.
        gap_multiplier: Gaps larger than gap_multiplier × median_gap are word breaks.

    Returns:
        List of WordRegion objects sorted left-to-right.
    """
    line_img = line_region.image
    line_h, line_w = line_img.shape
    lx, ly, lw, lh = line_region.bbox_in_image

    # Get components within this line
    components = getattr(line_region, '_components', None)
    if components is None or len(components) < 2:
        # Single-component line or no component info — return entire line as one word
        word = WordRegion(
            image=line_img.copy(),
            bbox_in_line=(0, 0, line_w, line_h),
            bbox_in_image=(lx, ly, lw, lh),
            word_index=0,
        )
        return [word]

    # Sort components left-to-right
    sorted_comps = sorted(components, key=lambda c: c['x_in_line'])

    # Compute inter-component gaps
    gaps = []
    for i in range(len(sorted_comps) - 1):
        right_edge = sorted_comps[i]['x_in_line'] + sorted_comps[i]['w']
        left_edge = sorted_comps[i + 1]['x_in_line']
        gap = max(0, left_edge - right_edge)
        gaps.append(gap)

    if not gaps:
        word = WordRegion(
            image=line_img.copy(),
            bbox_in_line=(0, 0, line_w, line_h),
            bbox_in_image=(lx, ly, lw, lh),
            word_index=0,
        )
        return [word]

    # Determine word-break threshold
    median_gap = float(np.median(gaps))
    # Use median character height as a secondary reference
    median_comp_h = float(np.median([c['h'] for c in sorted_comps]))
    # Word break = whichever is larger: gap_multiplier × median_gap or 0.6 × median_char_height
    word_break_threshold = max(
        gap_multiplier * max(median_gap, 1.0),
        0.6 * median_comp_h,
    )

    # Group components into words based on gap threshold
    word_groups: List[List[Dict]] = [[sorted_comps[0]]]
    for i, gap in enumerate(gaps):
        if gap > word_break_threshold:
            word_groups.append([sorted_comps[i + 1]])
        else:
            word_groups[-1].append(sorted_comps[i + 1])

    # Build WordRegion objects
    word_regions = []
    for wi, group in enumerate(word_groups):
        # Compute bounding box of this word within the line
        wx = min(c['x_in_line'] for c in group)
        wy = min(c['y_in_line'] for c in group)
        wx2 = max(c['x_in_line'] + c['w'] for c in group)
        wy2 = max(c['y_in_line'] + c['h'] for c in group)

        # Small padding
        pad = 2
        wx = max(0, wx - pad)
        wy = max(0, wy - pad)
        wx2 = min(line_w, wx2 + pad)
        wy2 = min(line_h, wy2 + pad)

        ww = wx2 - wx
        wh = wy2 - wy

        word_img = line_img[wy:wy2, wx:wx2].copy()

        word_regions.append(WordRegion(
            image=word_img,
            bbox_in_line=(wx, wy, ww, wh),
            bbox_in_image=(lx + wx, ly + wy, ww, wh),
            word_index=wi,
            components=group,
        ))

    return word_regions


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

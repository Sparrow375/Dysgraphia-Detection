"""
Segmentation module for static handwriting.
Segments binary handwriting masks into text lines and individual character/letter units
using connected components and horizontal spacing analysis.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.ndimage import label, find_objects


@dataclass
class LetterComponent:
    label_id: int
    bbox: Tuple[int, int, int, int]  # (y_min, y_max, x_min, x_max)
    width: int
    height: int
    area: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    x_center: float
    y_center: float
    y_bottom: float  # Baseline contact point (y_max)


@dataclass
class TextLine:
    line_index: int
    components: List[LetterComponent]
    median_height: float
    mean_baseline: float
    gaps: List[float]  # Horizontal distances between consecutive components


def segment_handwriting(
    binary_img: np.ndarray,
    min_area: int = 15,
    min_height: int = 6,
    line_overlap_thresh: float = 0.3
) -> List[TextLine]:
    """
    Performs connected component segmentation and clusters components into ordered text lines.
    Within each text line, components are sorted from left to right.
    """
    labeled, num_features = label(binary_img)
    if num_features == 0:
        return []

    slices = find_objects(labeled)
    components: List[LetterComponent] = []

    for i, slc in enumerate(slices):
        if slc is None:
            continue
        y_slice, x_slice = slc
        comp_mask = (labeled[y_slice, x_slice] == (i + 1))
        area = int(np.sum(comp_mask))

        h = y_slice.stop - y_slice.start
        w = x_slice.stop - x_slice.start

        if area < min_area or h < min_height:
            continue

        y_min, y_max = y_slice.start, y_slice.stop - 1
        x_min, x_max = x_slice.start, x_slice.stop - 1

        components.append(LetterComponent(
            label_id=i + 1,
            bbox=(y_min, y_max, x_min, x_max),
            width=w,
            height=h,
            area=area,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_center=float(x_min + x_max) / 2.0,
            y_center=float(y_min + y_max) / 2.0,
            y_bottom=float(y_max),
        ))

    if not components:
        return []

    # Sort all components vertically by y_center
    components.sort(key=lambda c: c.y_center)

    # Group into text lines based on vertical overlap
    lines_raw: List[List[LetterComponent]] = []
    for comp in components:
        placed = False
        for line in lines_raw:
            # Check overlap with line's vertical span
            line_y_mins = [c.y_min for c in line]
            line_y_maxs = [c.y_max for c in line]
            l_min, l_max = min(line_y_mins), max(line_y_maxs)
            line_height = l_max - l_min + 1

            overlap = max(0, min(comp.y_max, l_max) - max(comp.y_min, l_min))
            if overlap / max(comp.height, 1) > line_overlap_thresh or overlap / max(line_height, 1) > line_overlap_thresh:
                line.append(comp)
                placed = True
                break
        if not placed:
            lines_raw.append([comp])

    # Sort lines by median y_center
    lines_raw.sort(key=lambda l: np.median([c.y_center for c in l]))

    text_lines: List[TextLine] = []
    for idx, raw_line in enumerate(lines_raw):
        # Sort components left to right
        raw_line.sort(key=lambda c: c.x_min)

        # Compute horizontal gaps between consecutive letters: gap = x_min[i+1] - x_max[i]
        gaps = []
        for i in range(len(raw_line) - 1):
            gap = float(raw_line[i + 1].x_min - raw_line[i].x_max)
            gaps.append(gap)

        heights = [c.height for c in raw_line]
        baselines = [c.y_bottom for c in raw_line]

        text_lines.append(TextLine(
            line_index=idx,
            components=raw_line,
            median_height=float(np.median(heights)),
            mean_baseline=float(np.mean(baselines)),
            gaps=gaps,
        ))

    return text_lines

"""
BHK static feature extraction module.
Extracts 6 core clinical BHK-style features from preprocessed and segmented handwriting:
  1. Size Covariance (CV of height and area across letters)
  2. Height-Ratio Consistency (IQR / median height)
  3. Baseline Drift (Linear trend slope & residual variance of line bottom bounds)
  4. Inter-Letter Spacing Entropy (Shannon entropy of normalized gaps)
  5. Stroke-Width Variance (CV and variance of distance-transform values on skeleton)
  6. Telescoping / Overlap (Proportion and depth of colliding adjacent characters)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from src.branch_a.preprocessing import load_and_binarize, zhang_suen_skeletonize, compute_stroke_width_map
from src.branch_a.segmentation import segment_handwriting, TextLine


def compute_size_covariance(text_lines: List[TextLine]) -> Dict[str, float]:
    """
    Computes size variation across segmented letter units.
    In dysgraphic handwriting, letter height and area fluctuate erratically.
    """
    all_heights = []
    all_areas = []

    for line in text_lines:
        for comp in line.components:
            all_heights.append(comp.height)
            all_areas.append(comp.area)

    if len(all_heights) < 2:
        return {"size_covariance_height": 0.0, "size_covariance_area": 0.0, "size_covariance_score": 0.0}

    h_arr = np.array(all_heights, dtype=np.float64)
    a_arr = np.array(all_areas, dtype=np.float64)

    cv_h = float(np.std(h_arr) / max(np.mean(h_arr), 1e-4))
    cv_a = float(np.std(a_arr) / max(np.mean(a_arr), 1e-4))
    score = float(0.6 * cv_h + 0.4 * cv_a)

    return {
        "size_covariance_height": cv_h,
        "size_covariance_area": cv_a,
        "size_covariance_score": score,
    }


def compute_height_ratio_consistency(text_lines: List[TextLine]) -> Dict[str, float]:
    """
    Measures height dispersion using the normalized Interquartile Range (IQR / median).
    Typical handwriting clusters tightly into body (x-height) and ascenders;
    dysgraphia exhibits uncoordinated height dispersion.
    """
    all_heights = []
    for line in text_lines:
        for comp in line.components:
            all_heights.append(comp.height)

    if len(all_heights) < 3:
        return {"height_iqr_ratio": 0.0, "height_std_ratio": 0.0}

    h_arr = np.array(all_heights, dtype=np.float64)
    med = float(np.median(h_arr))
    q75, q25 = np.percentile(h_arr, [75, 25])
    iqr = float(q75 - q25)

    iqr_ratio = float(iqr / max(med, 1.0))
    std_ratio = float(np.std(h_arr) / max(med, 1.0))

    return {
        "height_iqr_ratio": iqr_ratio,
        "height_std_ratio": std_ratio,
    }


def compute_baseline_drift(text_lines: List[TextLine]) -> Dict[str, float]:
    """
    Computes both macro baseline tilt (|slope|) and micro baseline wobble (residual std)
    by fitting a linear regression line to the bottom contacts (y_bottom) of letters.
    """
    line_slopes = []
    line_residuals = []

    for line in text_lines:
        if len(line.components) < 3:
            continue

        x_coords = np.array([c.x_center for c in line.components], dtype=np.float64)
        y_bottoms = np.array([c.y_bottom for c in line.components], dtype=np.float64)
        med_h = max(line.median_height, 1.0)

        # Normalize x to span [0, 1] so slope is directly comparable
        span_x = np.max(x_coords) - np.min(x_coords)
        if span_x < 10.0:
            continue
        x_norm = (x_coords - np.min(x_coords)) / span_x

        # Linear fit: y = beta_0 + beta_1 * x_norm
        poly = np.polyfit(x_norm, y_bottoms, deg=1)
        slope = float(abs(poly[0]) / med_h)  # Normalized vertical drift over the line

        # Residual variation around linear baseline
        y_pred = np.polyval(poly, x_norm)
        residual_std = float(np.std(y_bottoms - y_pred) / med_h)

        line_slopes.append(slope)
        line_residuals.append(residual_std)

    if not line_slopes:
        return {"baseline_drift_slope": 0.0, "baseline_drift_residual": 0.0, "baseline_drift_score": 0.0}

    mean_slope = float(np.mean(line_slopes))
    mean_resid = float(np.mean(line_residuals))
    score = float(mean_slope + 2.0 * mean_resid)

    return {
        "baseline_drift_slope": mean_slope,
        "baseline_drift_residual": mean_resid,
        "baseline_drift_score": score,
    }


def compute_spacing_entropy(text_lines: List[TextLine], n_bins: int = 8) -> Dict[str, float]:
    """
    Calculates Shannon entropy of normalized inter-character gaps.
    In consistent writing, gaps have low entropy (concentrated around a regular rhythm).
    In dysgraphic writing, gaps are chaotic and irregular, resulting in high entropy.
    """
    all_gaps = []
    all_widths = []

    for line in text_lines:
        for c in line.components:
            all_widths.append(c.width)
        for g in line.gaps:
            all_gaps.append(g)

    if len(all_gaps) < 3 or not all_widths:
        return {"spacing_entropy": 0.0, "spacing_cv": 0.0}

    med_w = float(np.median(all_widths)) if all_widths else 20.0
    med_w = max(med_w, 1.0)

    # Normalize gaps by median character width
    norm_gaps = np.array(all_gaps, dtype=np.float64) / med_w

    # Discard extreme multi-word spacing outliers for letter-level entropy
    valid_gaps = norm_gaps[(norm_gaps >= -1.0) & (norm_gaps <= 3.0)]
    if len(valid_gaps) < 3:
        return {"spacing_entropy": 0.0, "spacing_cv": 0.0}

    # Histogram of gaps
    counts, _ = np.histogram(valid_gaps, bins=n_bins, range=(-1.0, 3.0))
    probs = counts.astype(np.float64) / (np.sum(counts) + 1e-10)
    probs = probs[probs > 0]

    # Shannon entropy
    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(n_bins)
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Coefficient of variation of positive gaps
    pos_gaps = valid_gaps[valid_gaps > 0]
    spacing_cv = float(np.std(pos_gaps) / max(np.mean(pos_gaps), 1e-3)) if len(pos_gaps) > 1 else 0.0

    return {
        "spacing_entropy": norm_entropy,
        "spacing_cv": spacing_cv,
    }


def compute_stroke_width_stats(skeleton_widths: np.ndarray) -> Dict[str, float]:
    """
    Analyzes local stroke widths sampled along the morphological skeleton.
    Measures stroke-width variance, coefficient of variation, and local thickness tremors.
    """
    if len(skeleton_widths) < 5:
        return {"stroke_width_mean": 0.0, "stroke_width_variance": 0.0, "stroke_width_cv": 0.0}

    mean_w = float(np.mean(skeleton_widths))
    var_w = float(np.var(skeleton_widths))
    cv_w = float(np.std(skeleton_widths) / max(mean_w, 1e-4))

    return {
        "stroke_width_mean": mean_w,
        "stroke_width_variance": var_w,
        "stroke_width_cv": cv_w,
    }


def compute_telescoping_overlap(text_lines: List[TextLine]) -> Dict[str, float]:
    """
    Measures letter telescoping (characters intruding into or overlapping adjacent character space).
    Telescoping is defined by negative horizontal gaps (gap_i < 0).
    """
    total_pairs = 0
    overlapping_pairs = 0
    overlap_depths = []
    all_widths = []

    for line in text_lines:
        for c in line.components:
            all_widths.append(c.width)
        for g in line.gaps:
            total_pairs += 1
            if g < 0:
                overlapping_pairs += 1
                overlap_depths.append(abs(g))

    if total_pairs == 0:
        return {"telescoping_ratio": 0.0, "telescoping_mean_depth": 0.0, "telescoping_score": 0.0}

    med_w = float(np.median(all_widths)) if all_widths else 20.0
    med_w = max(med_w, 1.0)

    telescoping_ratio = float(overlapping_pairs) / float(total_pairs)
    mean_depth = float(np.mean(overlap_depths) / med_w) if overlap_depths else 0.0
    score = float(telescoping_ratio + mean_depth)

    return {
        "telescoping_ratio": telescoping_ratio,
        "telescoping_mean_depth": mean_depth,
        "telescoping_score": score,
    }


def compute_acute_turns(
    recovered_strokes: List[np.ndarray],
    min_angle_deg: float = 110.0
) -> Dict[str, float]:
    """
    BHK Item 5: Acute turns and motor tremor.
    Measures abrupt directional deviations (> min_angle_deg) along continuous skeleton strokes,
    capturing fine motor jerkiness and tremor.
    """
    total_turns = 0
    total_arc_len = 0.0
    min_rad = np.radians(min_angle_deg)

    for s in recovered_strokes:
        if len(s) < 4:
            continue
        dx = np.diff(s[:, 0])
        dy = np.diff(s[:, 1])
        seg_lens = np.sqrt(dx**2 + dy**2)
        total_arc_len += float(np.sum(seg_lens))

        angles = np.arctan2(dy, dx)
        d_theta = np.abs(np.diff(angles))
        d_theta = np.where(d_theta > np.pi, 2 * np.pi - d_theta, d_theta)

        sharp = d_theta >= min_rad
        total_turns += int(np.sum(sharp))

    n_strokes = max(len(recovered_strokes), 1)
    rate_per_stroke = float(total_turns) / n_strokes
    rate_per_100px = float(total_turns) / max(total_arc_len / 100.0, 0.1)

    return {
        "acute_turns_count": float(total_turns),
        "acute_turns_per_stroke": rate_per_stroke,
        "acute_turns_per_100px": rate_per_100px,
        "acute_turns_score": float(np.clip(rate_per_stroke, 0.0, 5.0)),
    }


def compute_left_margin_drift(text_lines: List[TextLine]) -> Dict[str, float]:
    """
    BHK Item 2: Left margin alignment and drift.
    Measures horizontal drift and variance of line starting coordinates (x_start).
    """
    if len(text_lines) < 2:
        return {"left_margin_drift_slope": 0.0, "left_margin_std": 0.0, "left_margin_score": 0.0}

    left_coords = []
    med_heights = []
    for line in text_lines:
        if not line.components:
            continue
        xs = [c.x_center - c.width / 2.0 for c in line.components]
        left_coords.append(min(xs))
        med_heights.append(line.median_height)

    if len(left_coords) < 2:
        return {"left_margin_drift_slope": 0.0, "left_margin_std": 0.0, "left_margin_score": 0.0}

    x_arr = np.array(left_coords, dtype=float)
    h_med = float(np.median(med_heights)) if med_heights else 20.0
    h_med = max(h_med, 1.0)

    # Normalize by median line height
    x_norm = x_arr / h_med
    line_indices = np.arange(len(x_arr), dtype=float)

    poly = np.polyfit(line_indices, x_norm, deg=1)
    slope = float(abs(poly[0]))
    std_drift = float(np.std(x_norm))
    score = float(slope + 0.5 * std_drift)

    return {
        "left_margin_drift_slope": slope,
        "left_margin_std": std_drift,
        "left_margin_score": score,
    }


def compute_line_collisions(text_lines: List[TextLine]) -> Dict[str, float]:
    """
    BHK Item 13: Inter-line spacing consistency and collisions.
    Measures vertical overlap and distance variance between consecutive lines.
    """
    if len(text_lines) < 2:
        return {"line_collision_ratio": 0.0, "interline_spacing_cv": 0.0, "line_collision_score": 0.0}

    collisions = 0
    spacings = []
    med_heights = [l.median_height for l in text_lines if l.components]
    h_med = float(np.median(med_heights)) if med_heights else 20.0
    h_med = max(h_med, 1.0)

    for i in range(len(text_lines) - 1):
        l1 = text_lines[i]
        l2 = text_lines[i + 1]
        if not l1.components or not l2.components:
            continue

        y1_max = max(c.y_bottom for c in l1.components)
        y2_min = min(c.y_center - c.height / 2.0 for c in l2.components)

        gap = y2_min - y1_max
        if gap < 0:
            collisions += 1

        y1_base = np.median([c.y_bottom for c in l1.components])
        y2_base = np.median([c.y_bottom for c in l2.components])
        spacings.append(abs(y2_base - y1_base))

    n_pairs = max(len(text_lines) - 1, 1)
    collision_ratio = float(collisions) / float(n_pairs)
    spacing_cv = float(np.std(spacings) / max(np.mean(spacings), 1e-3)) if len(spacings) > 1 else 0.0
    score = float(collision_ratio + 0.5 * spacing_cv)

    return {
        "line_collision_ratio": collision_ratio,
        "interline_spacing_cv": spacing_cv,
        "line_collision_score": score,
    }


def extract_bhk_features(image_input) -> Dict[str, Any]:
    """
    Full Branch A extraction pipeline:
      Input: Image (filepath, PIL Image, or numpy array)
      Output: Dictionary containing all 9 BHK clinical features, sub-metrics, and segmentation stats.
    """
    from src.branch_b.stroke_recovery import recover_handwriting_trajectory

    binary_img = load_and_binarize(image_input)
    skeleton = zhang_suen_skeletonize(binary_img)
    _, skeleton_widths, _ = compute_stroke_width_map(binary_img, skeleton)

    text_lines = segment_handwriting(binary_img)
    recovered_strokes = recover_handwriting_trajectory(skeleton)

    # 1. Size Covariance (BHK #4)
    f_size = compute_size_covariance(text_lines)
    # 2. Height Ratio Consistency (BHK #3)
    f_height = compute_height_ratio_consistency(text_lines)
    # 3. Baseline Drift (BHK #1)
    f_drift = compute_baseline_drift(text_lines)
    # 4. Spacing Entropy (BHK #8)
    f_spacing = compute_spacing_entropy(text_lines)
    # 5. Stroke Width Variance (BHK #6)
    f_width = compute_stroke_width_stats(skeleton_widths)
    # 6. Telescoping Overlap (BHK #7)
    f_telescope = compute_telescoping_overlap(text_lines)
    # 7. Acute Turns / Curvature Tremor (BHK #5)
    f_turns = compute_acute_turns(recovered_strokes)
    # 8. Left Margin Drift (BHK #2)
    f_margin = compute_left_margin_drift(text_lines)
    # 9. Line Collisions & Interline Spacing (BHK #13)
    f_lines = compute_line_collisions(text_lines)

    total_letters = sum(len(l.components) for l in text_lines)
    total_lines = len(text_lines)

    # Expanded 9-BHK Clinical Feature Vector
    feature_vector = np.array([
        f_size["size_covariance_score"],
        f_height["height_iqr_ratio"],
        f_drift["baseline_drift_score"],
        f_spacing["spacing_entropy"],
        f_width["stroke_width_cv"],
        f_telescope["telescoping_score"],
        f_turns["acute_turns_score"],
        f_margin["left_margin_score"],
        f_lines["line_collision_score"],
    ], dtype=np.float64)

    feature_names = [
        "size_covariance",
        "height_ratio_consistency",
        "baseline_drift",
        "spacing_entropy",
        "stroke_width_variance",
        "telescoping_overlap",
        "acute_turns",
        "left_margin_drift",
        "line_collisions",
    ]

    return {
        "feature_vector": feature_vector,
        "feature_names": feature_names,
        "metrics": {
            **f_size,
            **f_height,
            **f_drift,
            **f_spacing,
            **f_width,
            **f_telescope,
            **f_turns,
            **f_margin,
            **f_lines,
            "total_letters": total_letters,
            "total_lines": total_lines,
            "recovered_strokes_count": len(recovered_strokes),
        },
        "binary_mask": binary_img,
        "skeleton": skeleton,
        "text_lines": text_lines,
        "recovered_strokes": recovered_strokes,
    }

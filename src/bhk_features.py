"""
BHK Feature Extraction Module for Dysgraphia Detection
Extracts scale-invariant quantitative approximations of the BHK criteria
from 2D offline handwriting images with multi-baseline line detection and
cursive-aware script disentanglement.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import cv2


# Core 13 scale-invariant BHK features matching base model bundle
FEATURE_NAMES = [
    "letter_size_cv",              # BHK #8: Inconsistent letter size (CoV = std/mean)
    "letter_area_cv",              # BHK #8: Inconsistent letter area (CoV)
    "aspect_ratio_mean",           # Mean component aspect ratio (w/h, cursive-compensated)
    "aspect_ratio_std",            # Aspect ratio variation
    "baseline_drift_slope",        # BHK #3: Average baseline drift slope across lines (|dy/dx|)
    "baseline_drift_residual_norm", # BHK #3: Average baseline waviness RMSE across lines (normalized)
    "inter_component_gap_norm",    # BHK #4: Inter-component / inter-word spacing normalized
    "inter_component_gap_cv",      # BHK #4: Spacing irregularity (CoV = std/mean)
    "letter_collision_ratio",      # BHK #7: Collision / overlap ratio
    "relative_height_ratio",       # BHK #9: Ascenders/descenders proportionality (P90/P50)
    "trace_unsteadiness_mean",     # BHK #13: Trace shakiness / curvature variance
    "ink_density",                 # Stroke thickness / pressure proxy
    "component_count",             # Number of detected letter / word units
]

# Extended features including multi-line geometry and clinical subtype indicators
EXTENDED_FEATURE_NAMES = FEATURE_NAMES + [
    "line_count",                  # Total detected text lines
    "components_per_line",         # Line-density normalized component count
    "line_parallelism_std",        # Standard deviation of baseline slopes across lines (Spatial)
    "line_spacing_cv",             # Inter-line vertical spacing irregularity CoV (Spatial)
    "cursive_index",               # Ratio of median component width to x-height (Cursive detection)
    "is_cursive",                  # Boolean flag (1.0 or 0.0) indicating connected script
    "slant_angle_mean",            # Dominant stroke slant angle (degrees from horizontal)
    "slant_angle_std",             # Stroke slant irregularity (Motor dysgraphia indicator)
    "stroke_tremor_high_freq",     # High-frequency neuromotor micro-tremor along strokes (Motor)
    "spatial_dysgraphia_score",    # Visuospatial layout impairment index (0.0 to 1.0)
    "motor_dysgraphia_score",      # Fine-motor graphomotor impairment index (0.0 to 1.0)
    "dyslexic_risk_score",         # Linguistic / phonological layout risk index (0.0 to 1.0)
    "cursive_fluidity_index",      # Fluidity and consistency of cursive execution (0.0 to 1.0)
]


def get_feature_names() -> List[str]:
    """Returns the ordered list of core 13 feature names."""
    return FEATURE_NAMES.copy()


def get_extended_feature_names() -> List[str]:
    """Returns the ordered list of all extended feature names."""
    return EXTENDED_FEATURE_NAMES.copy()


def calculate_trace_curvature(contour: np.ndarray) -> float:
    """Computes curvature variation along stroke contour (BHK #13 proxy)."""
    if len(contour) < 10:
        return 0.0
    pts = contour.reshape(-1, 2)
    step = max(1, len(pts) // 30)
    sampled = pts[::step]
    if len(sampled) < 5:
        return 0.0
        
    v1 = sampled[1:-1] - sampled[:-2]
    v2 = sampled[2:] - sampled[1:-1]
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    valid_mask = (norm1 > 1e-4) & (norm2 > 1e-4)
    if np.sum(valid_mask) < 3:
        return 0.0
        
    dot_products = np.sum(v1[valid_mask] * v2[valid_mask], axis=1)
    cos_angles = np.clip(dot_products / (norm1[valid_mask] * norm2[valid_mask]), -1.0, 1.0)
    angles = np.arccos(cos_angles)
    return float(np.var(angles))


def segment_text_lines(letters: List[Dict], median_h: float) -> List[List[Dict]]:
    """
    Groups handwriting components into distinct horizontal text lines.
    Handles multi-line documents, preventing multi-line diagonal regression artifacts.
    """
    if not letters:
        return []

    # Sort components top-to-bottom by vertical centroid
    letters_sorted = sorted(letters, key=lambda l: l['cy'])
    line_clusters: List[List[Dict]] = []
    thresh = 1.6 * median_h

    for l in letters_sorted:
        best_ci = -1
        best_dist = 9999.0
        for ci, cluster in enumerate(line_clusters):
            c_cy = float(np.mean([item['cy'] for item in cluster]))
            dist = abs(l['cy'] - c_cy)
            if dist < thresh and dist < best_dist:
                best_dist = dist
                best_ci = ci
        if best_ci != -1:
            line_clusters[best_ci].append(l)
        else:
            line_clusters.append([l])

    # Filter lines with at least 3 components (or 2 if components span a wider width)
    valid_clusters = [
        sorted(cluster, key=lambda it: it['cx'])
        for cluster in line_clusters
        if len(cluster) >= 3 or (len(cluster) >= 2 and (max(c['x'] + c['w'] for c in cluster) - min(c['x'] for c in cluster) > 2.0 * median_h))
    ]

    # Merge line clusters that share vertical overlap
    merged = True
    while merged and len(valid_clusters) > 1:
        merged = False
        for i in range(len(valid_clusters)):
            for j in range(i + 1, len(valid_clusters)):
                c1, c2 = valid_clusters[i], valid_clusters[j]
                y1_min, y1_max = min(c['y'] for c in c1), max(c['bottom'] for c in c1)
                y2_min, y2_max = min(c['y'] for c in c2), max(c['bottom'] for c in c2)
                overlap = min(y1_max, y2_max) - max(y1_min, y2_min)
                h_min = min(y1_max - y1_min, y2_max - y2_min)
                if overlap > 0.40 * h_min:
                    valid_clusters[i] = sorted(c1 + c2, key=lambda it: it['cx'])
                    valid_clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    valid_clusters.sort(key=lambda cluster: np.mean([it['cy'] for it in cluster]))

    if not valid_clusters and letters:
        valid_clusters = [sorted(letters, key=lambda it: it['cx'])]

    return valid_clusters


def fit_line_baselines(lines: List[List[Dict]], median_h: float) -> List[Dict[str, Any]]:
    """
    Fits independent robust linear baselines (y = m*x + c) for each detected text line
    with descender outlier rejection (ignoring hanging tails of 'g', 'y', 'p').
    """
    line_models = []
    for line_idx, line in enumerate(lines):
        xs = np.array([it['cx'] for it in line], dtype=float)
        bottoms = np.array([it['bottom'] for it in line], dtype=float)

        if len(xs) >= 3 and (np.max(xs) - np.min(xs) > 10):
            try:
                # Initial linear fit
                p = np.polyfit(xs, bottoms, 1)
                res = bottoms - np.polyval(p, xs)
                # Inlier threshold: filter descenders (res > 0.75 * median_h) and ascender tops
                inliers = (res <= 0.75 * median_h) & (res >= -1.5 * median_h)
                if np.sum(inliers) >= 2:
                    p_refined = np.polyfit(xs[inliers], bottoms[inliers], 1)
                else:
                    p_refined = p

                slope = float(p_refined[0])
                intercept = float(p_refined[1])
                pred = np.polyval(p_refined, xs)
                rmse = float(np.std(bottoms - pred) / max(median_h, 1e-4))
            except Exception:
                slope = 0.0
                intercept = float(np.mean(bottoms))
                rmse = float(np.std(bottoms) / max(median_h, 1e-4))
        elif len(xs) == 2:
            dx = max(xs[1] - xs[0], 1.0)
            slope = float((bottoms[1] - bottoms[0]) / dx)
            intercept = float(bottoms[0] - slope * xs[0])
            rmse = 0.0
        else:
            slope = 0.0
            intercept = float(bottoms[0]) if len(bottoms) > 0 else 0.0
            rmse = 0.0

        line_models.append({
            'line_idx': line_idx + 1,
            'letters': line,
            'slope': slope,
            'abs_slope': abs(slope),
            'intercept': intercept,
            'residual_norm': rmse,
            'min_x': float(np.min(xs)),
            'max_x': float(np.max(xs)),
            'mean_y': float(np.mean(bottoms))
        })
    return line_models


def calculate_slant_and_tremor(binary_mask: np.ndarray, median_h: float) -> Tuple[float, float, float]:
    """
    Computes stroke slant orientation uniformity (sigma_slant) and isolates
    high-frequency neuromotor micro-tremor from intentional smooth cursive loops.
    
    Returns:
        (mean_slant_deg, slant_std_deg, stroke_tremor_high_freq)
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_slants = []
    tremor_scores = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 25 or len(cnt) < 15:
            continue
        pts = cnt.reshape(-1, 2).astype(float)

        # 1. Slant analysis on vertical stroke paths
        step = max(2, len(pts) // 35)
        p_prev = pts[:-step:step]
        p_next = pts[step::step]
        dx = p_next[:, 0] - p_prev[:, 0]
        dy = p_next[:, 1] - p_prev[:, 1]

        # Consider strokes with vertical progression (|dy| > 3.0)
        vert_mask = np.abs(dy) > 3.0
        if np.sum(vert_mask) >= 3:
            dx_v = dx[vert_mask]
            dy_v = dy[vert_mask]
            angles = np.arctan2(np.abs(dy_v), dx_v) * 180.0 / np.pi
            # Filter near-horizontal connecting ligatures (keep 35 to 145 degrees)
            angles_clean = angles[(angles >= 35.0) & (angles <= 145.0)]
            if len(angles_clean) > 0:
                all_slants.extend(angles_clean.tolist())

        # 2. High-frequency neuromotor micro-tremor (bandpass filtering)
        # Separates intentional cursive bezier loops (low frequency) from shaky micro-tremors (high frequency)
        if len(pts) >= 15:
            dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
            cum_dist = np.insert(np.cumsum(dists), 0, 0)
            total_len = cum_dist[-1]
            if total_len > 25:
                sample_step = 3.0  # 3 pixels uniform step
                n_samples = max(6, int(total_len / sample_step))
                interp_dists = np.linspace(0, total_len, n_samples)
                sample_x = np.interp(interp_dists, cum_dist, pts[:, 0])
                sample_y = np.interp(interp_dists, cum_dist, pts[:, 1])
                sample_pts = np.column_stack([sample_x, sample_y])

                # 5-point moving average represents the intentional stroke trajectory
                smooth_x = np.convolve(sample_x, np.ones(5) / 5.0, mode='valid')
                smooth_y = np.convolve(sample_y, np.ones(5) / 5.0, mode='valid')
                smooth_pts = np.column_stack([smooth_x, smooth_y])

                # Residual difference isolates high-frequency micro-shakiness
                raw_subset = sample_pts[2:-2]
                residual = np.sqrt(np.sum((raw_subset - smooth_pts)**2, axis=1))
                tremor_scores.append(float(np.mean(residual) / max(median_h, 1e-4)))

    mean_slant = float(np.mean(all_slants)) if len(all_slants) >= 8 else 90.0
    slant_std = float(np.std(all_slants)) if len(all_slants) >= 8 else 18.0
    mean_tremor = float(np.mean(tremor_scores)) if len(tremor_scores) > 0 else 0.005

    return mean_slant, slant_std, mean_tremor


def extract_bhk_features(binary_mask: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Extracts scale-invariant BHK proxy features from a preprocessed binary mask.
    Features are normalized by median character height (x-height) and include
    multi-baseline line modeling, cursive script compensation, and clinical subtype scores.

    Args:
        binary_mask: 2D uint8 array where ink is 255 and background is 0.

    Returns:
        tuple (features_dict, feature_vector_1d_array_13d)
    """
    h_img, w_img = binary_mask.shape
    features = {name: 0.0 for name in EXTENDED_FEATURE_NAMES}

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Pass 1: Find candidate handwriting components to establish median character height (x-height)
    cand_h = []
    cand_w = []
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        if a >= 15 and h >= 5 and w >= 2 and w < 0.85 * w_img and h < 0.85 * h_img:
            cand_h.append(h)
            cand_w.append(w)

    if len(cand_h) < 3:
        vec = np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)
        return features, vec

    median_h = max(5.0, float(np.median(cand_h)))
    median_w = max(4.0, float(np.median(cand_w)))

    # Cursive Script Detection:
    # In cursive handwriting, letters are linked into wide connected words.
    cursive_index = float(median_w / median_h)
    is_cursive = bool(cursive_index >= 1.6)
    features["cursive_index"] = cursive_index
    features["is_cursive"] = 1.0 if is_cursive else 0.0

    # Pass 2: Filter valid text unit components
    letters = []
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        # Keep components that conform to valid handwriting units
        if a >= 15 and h >= 0.35 * median_h and h <= 4.0 * median_h and w >= 2 and w < 0.80 * w_img:
            letters.append({
                'x': stats[i, cv2.CC_STAT_LEFT],
                'y': stats[i, cv2.CC_STAT_TOP],
                'w': w,
                'h': h,
                'area': a,
                'cx': centroids[i][0],
                'cy': centroids[i][1],
                'bottom': stats[i, cv2.CC_STAT_TOP] + h
            })

    if len(letters) < 3:
        features["component_count"] = float(len(letters))
        vec = np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)
        return features, vec

    heights = np.array([l['h'] for l in letters], dtype=float)
    widths = np.array([l['w'] for l in letters], dtype=float)
    areas = np.array([l['area'] for l in letters], dtype=float)
    x_coords = np.array([l['x'] for l in letters], dtype=float)
    bottoms = np.array([l['bottom'] for l in letters], dtype=float)

    # 1. Letter size & area consistency (BHK #1, #8 - Scale Invariant CoV)
    mean_h = float(np.mean(heights))
    cv_h = float(np.std(heights) / max(mean_h, 1e-4))

    # Cursive area compensation: in connected script, normalize component area by estimated letter units (w / median_h)
    if is_cursive:
        estimated_units = np.maximum(1.0, widths / median_h)
        normalized_areas = areas / estimated_units
        cv_area = float(np.std(normalized_areas) / max(np.mean(normalized_areas), 1e-4))
    else:
        cv_area = float(np.std(areas) / max(np.mean(areas), 1e-4))

    features["letter_size_cv"] = cv_h
    features["letter_area_cv"] = cv_area

    # 2. Aspect ratio statistics (Cursive-Compensated)
    if is_cursive:
        # In cursive, each component can span multiple letters. Normalize aspect ratio per letter unit.
        char_aspect_ratios = (widths / np.maximum(1.0, widths / median_h)) / np.maximum(heights, 1.0)
    else:
        char_aspect_ratios = widths / np.maximum(heights, 1.0)

    features["aspect_ratio_mean"] = float(np.mean(char_aspect_ratios))
    features["aspect_ratio_std"] = float(np.std(char_aspect_ratios))

    # 3. Multi-Baseline Detection & Per-Line Alignment (BHK #3)
    # Solves the multi-line diagonal slash problem by clustering components into text lines
    lines = segment_text_lines(letters, median_h)
    line_models = fit_line_baselines(lines, median_h)

    if line_models:
        mean_drift_slope = float(np.mean([m['abs_slope'] for m in line_models]))
        mean_drift_res = float(np.mean([m['residual_norm'] for m in line_models]))
        features["baseline_drift_slope"] = mean_drift_slope
        features["baseline_drift_residual_norm"] = mean_drift_res
        features["line_count"] = float(len(line_models))

        if len(line_models) > 1:
            slopes = [m['slope'] for m in line_models]
            features["line_parallelism_std"] = float(np.std(slopes))
            # Inter-line vertical spacing irregularity
            line_spacings = [line_models[i + 1]['mean_y'] - line_models[i]['mean_y'] for i in range(len(line_models) - 1)]
            mean_sp = float(np.mean(line_spacings))
            features["line_spacing_cv"] = float(np.std(line_spacings) / max(mean_sp, 1e-4))
        else:
            features["line_parallelism_std"] = 0.0
            features["line_spacing_cv"] = 0.0
    else:
        features["baseline_drift_slope"] = 0.0
        features["baseline_drift_residual_norm"] = 0.0
        features["line_count"] = 1.0
        features["line_parallelism_std"] = 0.0
        features["line_spacing_cv"] = 0.0

    # 4. Inter-component gap & collisions (BHK #4, #7 - Per Line Analysis)
    # In multi-line or cursive documents, measure gaps along lines to avoid cross-line collisions
    line_gaps = []
    total_gaps_count = 0
    total_collisions = 0

    for line in lines:
        if len(line) < 2:
            continue
        line_xs = [item['x'] for item in line]
        line_ws = [item['w'] for item in line]
        sort_order = np.argsort(line_xs)
        s_xs = np.array(line_xs)[sort_order]
        s_ws = np.array(line_ws)[sort_order]

        for idx in range(len(s_xs) - 1):
            g = s_xs[idx + 1] - (s_xs[idx] + s_ws[idx])
            total_gaps_count += 1
            if g < 0:
                total_collisions += 1
                line_gaps.append(0.0)
            else:
                line_gaps.append(float(g))

    if total_gaps_count > 0:
        gaps_arr = np.array(line_gaps)
        mean_gap_norm = float(np.mean(gaps_arr) / median_h)
        gap_cv = float(np.std(gaps_arr) / max(np.mean(gaps_arr), 1e-4))
        collision_ratio = float(total_collisions / total_gaps_count)
    else:
        mean_gap_norm = 0.0
        gap_cv = 0.0
        collision_ratio = 0.0

    features["inter_component_gap_norm"] = mean_gap_norm
    features["inter_component_gap_cv"] = gap_cv
    features["letter_collision_ratio"] = collision_ratio

    # 5. Relative height ratio (BHK #9 - P90/P50)
    rel_height = float(np.percentile(heights, 90) / max(float(np.median(heights)), 1e-4))
    features["relative_height_ratio"] = rel_height

    # 6. Trace unsteadiness (BHK #13)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    unsteadiness_vals = [calculate_trace_curvature(c) for c in contours if cv2.contourArea(c) > 20]
    features["trace_unsteadiness_mean"] = float(np.mean(unsteadiness_vals)) if len(unsteadiness_vals) > 0 else 0.0

    # 7. Stroke slant uniformity and high-frequency neuromotor micro-tremor
    mean_slant, slant_std, tremor_hf = calculate_slant_and_tremor(binary_mask, median_h)
    features["slant_angle_mean"] = mean_slant
    features["slant_angle_std"] = slant_std
    features["stroke_tremor_high_freq"] = tremor_hf

    # 8. Ink density (Stroke ink ratio within writing bounding box area)
    total_ink = float(np.sum(binary_mask > 0))
    min_x, max_x = np.min(x_coords), np.max(x_coords + widths)
    min_y, max_y = np.min(np.array([l['y'] for l in letters])), np.max(bottoms)
    bbox_area = max(1.0, (max_x - min_x) * (max_y - min_y))
    features["ink_density"] = float(total_ink / bbox_area)

    # 9. Valid component count & line-density normalization
    features["component_count"] = float(len(letters))
    features["components_per_line"] = float(len(letters)) / max(1.0, features.get("line_count", 1.0))

    # 10. Clinical Subtype Diagnostic Indices
    # Cursive Fluidity Index: high fluidity = neurotypical cursive (suppresses false dysgraphia alarms)
    slant_uniformity_score = max(0.0, 1.0 - (slant_std / 35.0))
    tremor_clean_score = max(0.0, 1.0 - (tremor_hf / 0.025))
    baseline_clean_score = max(0.0, 1.0 - (features["baseline_drift_residual_norm"] / 0.35))
    cursive_fluidity = float(np.clip(
        (0.35 * slant_uniformity_score) +
        (0.35 * tremor_clean_score) +
        (0.30 * baseline_clean_score),
        0.0, 1.0
    ))
    features["cursive_fluidity_index"] = cursive_fluidity

    # Spatial Dysgraphia Index (Layout, line collisions, erratic gaps, baseline wandering)
    spatial_score = float(np.clip(
        (min(features["baseline_drift_residual_norm"], 0.5) / 0.5 * 0.30) +
        (min(features["baseline_drift_slope"], 0.2) / 0.2 * 0.25) +
        (min(features["line_parallelism_std"], 0.15) / 0.15 * 0.20) +
        (min(features["inter_component_gap_cv"], 1.5) / 1.5 * 0.15) +
        (min(features["letter_collision_ratio"], 0.3) / 0.3 * 0.10),
        0.0, 1.0
    ))
    features["spatial_dysgraphia_score"] = spatial_score

    # Motor Dysgraphia Index (Micro-tremor, erratic slant, letter size inconsistency)
    # Calibrated against NIST SD19 neurotypical benchmark (control mean tremor_hf = 0.038, slant_std = 20.9°)
    raw_motor_score = (
        (min(tremor_hf, 0.065) / 0.065 * 0.35) +
        (min(slant_std, 40.0) / 40.0 * 0.25) +
        (min(cv_h, 0.6) / 0.6 * 0.25) +
        (min(features["trace_unsteadiness_mean"], 0.5) / 0.5 * 0.15)
    )
    if is_cursive and cursive_fluidity > 0.65:
        # Fluid cursive discount
        motor_score = float(np.clip(raw_motor_score * 0.65, 0.0, 1.0))
    else:
        motor_score = float(np.clip(raw_motor_score, 0.0, 1.0))
    features["motor_dysgraphia_score"] = motor_score

    # Dyslexic / Linguistic Risk Index (Extreme aspect ratio outliers and letter height disparity)
    dyslexic_score = float(np.clip(
        (min(abs(features["aspect_ratio_mean"] - 1.0), 1.0) * 0.40) +
        (max(0.0, features["relative_height_ratio"] - 1.5) / 2.0 * 0.30) +
        (min(features["inter_component_gap_cv"], 1.5) / 1.5 * 0.30),
        0.0, 1.0
    ))
    features["dyslexic_risk_score"] = dyslexic_score

    # Construct standard 13-D feature vector for model bundle
    feature_vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    return features, feature_vector


def generate_feature_visualization(binary_mask: np.ndarray, base_rgb_img: np.ndarray = None) -> np.ndarray:
    """
    Renders visual explainability overlay:
    - Independent multi-line baselines for EACH detected text line with color coding
    - Character / word bounding boxes (green)
    - Centroids (amber)
    - Line index labels (L1, L2, ...)
    - Cursive compensation status badge
    """
    h_img, w_img = binary_mask.shape
    if base_rgb_img is None:
        vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    else:
        vis = cv2.resize(base_rgb_img, (w_img, h_img)).copy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Establish median scale
    cand_h = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)
              if stats[i, cv2.CC_STAT_AREA] >= 15 and stats[i, cv2.CC_STAT_HEIGHT] >= 5 and stats[i, cv2.CC_STAT_WIDTH] < 0.85 * w_img]
    median_h = max(5.0, float(np.median(cand_h))) if len(cand_h) > 0 else 10.0

    letters = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]

        if area >= 15 and comp_h >= 0.35 * median_h and comp_h <= 4.0 * median_h and comp_w < 0.80 * w_img:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            # Green bounding box around detected unit
            cv2.rectangle(vis, (x, y), (x + comp_w, y + comp_h), (0, 220, 100), 2)

            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            cv2.circle(vis, (cx, cy), 3, (255, 200, 0), -1)

            letters.append({
                'x': x, 'y': y, 'w': comp_w, 'h': comp_h,
                'cx': centroids[i][0], 'cy': centroids[i][1],
                'bottom': y + comp_h
            })

    # Segment and fit multi-line baselines
    lines = segment_text_lines(letters, median_h)
    line_models = fit_line_baselines(lines, median_h)

    # Draw distinct baseline rays for each detected text line
    for idx, model in enumerate(line_models):
        slope = model['slope']
        intercept = model['intercept']
        abs_slope = model['abs_slope']
        min_x = max(10, int(model['min_x'] - 10))
        max_x = min(w_img - 10, int(model['max_x'] + 10))

        y_start = int(slope * min_x + intercept)
        y_end = int(slope * max_x + intercept)

        # Clamp within image bounds
        y_start = np.clip(y_start, 0, h_img - 1)
        y_end = np.clip(y_end, 0, h_img - 1)

        # Baseline color coding based on drift slope severity:
        # Turquoise green for stable (<0.06), Amber for mild drift (0.06-0.12), Red for severe drift (>0.12)
        if abs_slope < 0.06:
            line_color = (0, 230, 180)    # Turquoise green (Stable)
        elif abs_slope < 0.12:
            line_color = (0, 180, 255)    # Amber (Mild drift)
        else:
            line_color = (255, 60, 60)    # Red (Severe drift)

        # Draw baseline
        cv2.line(vis, (min_x, y_start), (max_x, y_end), line_color, 2, cv2.LINE_AA)

        # Draw line badge label (e.g. L1, L2)
        badge_text = f"L{idx + 1}"
        badge_pos = (max(5, min_x - 30), max(18, y_start - 4))
        cv2.putText(vis, badge_text, badge_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)

    # Check cursive status
    cand_w = [l['w'] for l in letters]
    med_w = float(np.median(cand_w)) if cand_w else 10.0
    if (med_w / median_h) >= 1.6:
        # Render subtle cursive compensation badge
        cv2.putText(vis, "[Cursive Script Compensated]", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 0), 2, cv2.LINE_AA)

    return vis

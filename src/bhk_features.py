"""
BHK Feature Extraction Module for Dysgraphia Detection
Extracts scale-invariant quantitative approximations of the BHK criteria
from 2D offline handwriting images.
"""

from typing import Dict, List, Tuple
import numpy as np
import cv2


FEATURE_NAMES = [
    "letter_size_cv",              # BHK #8: Inconsistent letter size (CoV = std/mean)
    "letter_area_cv",              # BHK #8: Inconsistent letter area (CoV)
    "aspect_ratio_mean",           # Mean component aspect ratio (w/h)
    "aspect_ratio_std",            # Aspect ratio variation
    "baseline_drift_slope",        # BHK #3: Baseline alignment slope (|dy/dx|)
    "baseline_drift_residual_norm", # BHK #3: Baseline waviness RMSE normalized by character scale
    "inter_component_gap_norm",    # BHK #4: Inter-character spacing normalized by character scale
    "inter_component_gap_cv",      # BHK #4: Spacing irregularity (CoV = std/mean)
    "letter_collision_ratio",      # BHK #7: Collision / overlap ratio
    "relative_height_ratio",       # BHK #9: Ascenders/descenders proportionality (P90/P50)
    "trace_unsteadiness_mean",     # BHK #13: Trace shakiness (contour curvature variance)
    "ink_density",                 # Stroke thickness / pressure proxy
    "component_count",             # Number of detected letter units
]


def get_feature_names() -> List[str]:
    """Returns the ordered list of feature names."""
    return FEATURE_NAMES.copy()


def calculate_trace_curvature(contour: np.ndarray) -> float:
    """Computes curvature variation along stroke contour to quantify shakiness (BHK #13)."""
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


def extract_bhk_features(binary_mask: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Extracts scale-invariant BHK proxy features from a preprocessed binary mask.
    Metrics are normalized by the median character height (x-height) to guarantee
    invariance to camera zoom, image resolution, and paper distance.
    
    Args:
        binary_mask: 2D uint8 array where ink is 255 and background is 0.
        
    Returns:
        tuple (features_dict, feature_vector_1d_array)
    """
    h_img, w_img = binary_mask.shape
    features = {name: 0.0 for name in FEATURE_NAMES}
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Pass 1: Find candidate handwriting components to establish median character height (scale)
    cand_h = []
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        if a >= 15 and h >= 5 and w >= 2 and w < 0.80 * w_img and h < 0.85 * h_img:
            cand_h.append(h)
            
    if len(cand_h) < 3:
        vec = np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)
        return features, vec
        
    median_h = max(5.0, float(np.median(cand_h)))
    
    # Pass 2: Filter letter body components
    # Exclude diacritics/dots on 'i' (h < 0.35 * median_h) and giant page-spanning lines
    letters = []
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        if a >= 15 and h >= 0.35 * median_h and h <= 3.5 * median_h and w >= 2 and w < 0.75 * w_img:
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
    centroids_x = np.array([l['cx'] for l in letters], dtype=float)

    # 1. Letter size & area consistency (BHK #1, #8 - Scale Invariant CoV)
    mean_h = float(np.mean(heights))
    cv_h = float(np.std(heights) / max(mean_h, 1e-4))
    cv_area = float(np.std(areas) / max(np.mean(areas), 1e-4))
    features["letter_size_cv"] = cv_h
    features["letter_area_cv"] = cv_area

    # 2. Aspect ratio statistics
    aspect_ratios = widths / np.maximum(heights, 1.0)
    features["aspect_ratio_mean"] = float(np.mean(aspect_ratios))
    features["aspect_ratio_std"] = float(np.std(aspect_ratios))

    # 3. Baseline drift & alignment (BHK #3 - Dimensionless slope & scale-normalized residual RMSE)
    if len(letters) >= 3 and (np.max(centroids_x) - np.min(centroids_x) > 10):
        try:
            poly = np.polyfit(centroids_x, bottoms, 1)
            slope = abs(float(poly[0]))
            pred_b = np.polyval(poly, centroids_x)
            residual_norm = float(np.std(bottoms - pred_b) / median_h)
        except Exception:
            slope = 0.0
            residual_norm = float(np.std(bottoms) / median_h)
    else:
        slope = 0.0
        residual_norm = float(np.std(bottoms) / median_h)
    features["baseline_drift_slope"] = slope
    features["baseline_drift_residual_norm"] = residual_norm

    # 4. Inter-component gap & collisions (BHK #4, #7 - Scale normalized gap & collision ratio)
    sort_idx = np.argsort(x_coords)
    sorted_x = x_coords[sort_idx]
    sorted_w = widths[sort_idx]
    gaps = []
    collisions = 0
    for idx in range(len(sorted_x) - 1):
        g = sorted_x[idx + 1] - (sorted_x[idx] + sorted_w[idx])
        if g < 0:
            collisions += 1
            gaps.append(0.0)
        else:
            gaps.append(float(g))

    if len(gaps) > 0:
        gaps_arr = np.array(gaps)
        mean_gap_norm = float(np.mean(gaps_arr) / median_h)
        gap_cv = float(np.std(gaps_arr) / max(np.mean(gaps_arr), 1e-4))
        collision_ratio = float(collisions / len(gaps))
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

    # 7. Ink density (Ratio of stroke ink to total writing bounding box area)
    total_ink = float(np.sum(binary_mask > 0))
    min_x, max_x = np.min(x_coords), np.max(x_coords + widths)
    min_y, max_y = np.min(np.array([l['y'] for l in letters])), np.max(bottoms)
    bbox_area = max(1.0, (max_x - min_x) * (max_y - min_y))
    features["ink_density"] = float(total_ink / bbox_area)

    # 8. Valid component count
    features["component_count"] = float(len(letters))

    feature_vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    return features, feature_vector


def generate_feature_visualization(binary_mask: np.ndarray, base_rgb_img: np.ndarray = None) -> np.ndarray:
    """Renders visual overlay of character bounding boxes, centroids, and baseline fit."""
    h_img, w_img = binary_mask.shape
    if base_rgb_img is None:
        vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    else:
        vis = cv2.resize(base_rgb_img, (w_img, h_img)).copy()
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Establish median scale
    cand_h = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)
              if stats[i, cv2.CC_STAT_AREA] >= 15 and stats[i, cv2.CC_STAT_HEIGHT] >= 5 and stats[i, cv2.CC_STAT_WIDTH] < 0.80 * w_img]
    median_h = max(5.0, float(np.median(cand_h))) if len(cand_h) > 0 else 10.0

    valid_pts_x = []
    valid_bottoms_y = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        
        if area >= 15 and comp_h >= 0.35 * median_h and comp_h <= 3.5 * median_h and comp_w < 0.75 * w_img:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            
            # Green bounding box around detected letter
            cv2.rectangle(vis, (x, y), (x + comp_w, y + comp_h), (0, 220, 100), 2)
            
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            cv2.circle(vis, (cx, cy), 3, (255, 200, 0), -1)
            
            valid_pts_x.append(cx)
            valid_bottoms_y.append(y + comp_h)
            
    if len(valid_pts_x) >= 3 and (max(valid_pts_x) - min(valid_pts_x) > 20):
        try:
            poly = np.polyfit(valid_pts_x, valid_bottoms_y, 1)
            x_start = int(min(valid_pts_x))
            x_end = int(max(valid_pts_x))
            y_start = int(np.polyval(poly, x_start))
            y_end = int(np.polyval(poly, x_end))
            cv2.line(vis, (x_start, y_start), (x_end, y_end), (255, 60, 60), 2)
        except Exception:
            pass
            
    return vis

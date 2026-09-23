"""
BHK Feature Extraction Module for Dysgraphia Detection
Extracts quantitative approximations of the 13 BHK (Beknopte Beoordelingsmethode
voor Kinderhandschriften) criteria from 2D offline handwriting images.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import cv2


FEATURE_NAMES = [
    "letter_size_mean",           # Mean character height
    "letter_size_std",            # BHK #1: Letter size consistency
    "letter_size_cv",             # BHK #8: Inconsistent letter size (CoV)
    "letter_area_cv",             # BHK #8: Inconsistent letter area (CoV)
    "aspect_ratio_mean",          # Mean component aspect ratio (w/h)
    "aspect_ratio_std",           # Aspect ratio variation
    "left_margin_std",            # BHK #2: Left-hand margin variation
    "baseline_drift_slope",       # BHK #3: Word alignment / baseline slope
    "baseline_drift_residual",    # BHK #3: Baseline waviness / residual variance
    "inter_component_gap_mean",   # BHK #4: Word and letter spacing mean
    "inter_component_gap_cv",     # BHK #4: Spacing irregularity (CoV)
    "letter_collision_ratio",     # BHK #7: Overlapping / collision of letters
    "relative_height_ratio",      # BHK #9: Incorrect relative height (P90 / P50)
    "trace_unsteadiness_mean",    # BHK #13: Unsteady writing trace (contour curvature variance)
    "ink_density",                # Ink concentration within handwriting bounding region
    "component_count",            # Number of detected handwriting components
]


def get_feature_names() -> List[str]:
    """Returns the ordered list of feature names."""
    return FEATURE_NAMES.copy()


def calculate_contour_curvature_variance(contour: np.ndarray) -> float:
    """
    Computes curvature variation along a contour to quantify trace shakiness / unsteadiness (BHK #13).
    A smooth stroke has slowly changing angles; a shaky or hesitant trace has high angle variance.
    """
    if len(contour) < 10:
        return 0.0
        
    pts = contour.reshape(-1, 2)
    # Subsample points if contour is dense
    step = max(1, len(pts) // 30)
    sampled = pts[::step]
    
    if len(sampled) < 5:
        return 0.0
        
    # Compute consecutive segment vectors
    v1 = sampled[1:-1] - sampled[:-2]
    v2 = sampled[2:] - sampled[1:-1]
    
    # Compute angles between consecutive segments
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    
    valid_mask = (norm1 > 1e-4) & (norm2 > 1e-4)
    if np.sum(valid_mask) < 3:
        return 0.0
        
    dot_products = np.sum(v1[valid_mask] * v2[valid_mask], axis=1)
    cos_angles = np.clip(dot_products / (norm1[valid_mask] * norm2[valid_mask]), -1.0, 1.0)
    angles = np.arccos(cos_angles)
    
    # Curvature variance (fluctuation of turn sharpness)
    return float(np.var(angles))


def extract_bhk_features(binary_mask: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Extracts numerical BHK proxy features from a preprocessed binary mask.
    
    Args:
        binary_mask: 2D uint8 array where ink is 255 and background is 0.
        
    Returns:
        tuple (features_dict, feature_vector_1d_array)
    """
    h_img, w_img = binary_mask.shape
    
    # Default feature values in case of blank or degenerate image
    features = {name: 0.0 for name in FEATURE_NAMES}
    
    # Find connected components with statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Filter valid character-like components (exclude tiny speckles or huge borders)
    valid_indices = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Valid handwriting stroke/character filter
        if area >= 12 and h >= 5 and w >= 3 and w < 0.75 * w_img and h < 0.85 * h_img:
            valid_indices.append(i)
            
    if len(valid_indices) < 2:
        # Insufficient components to calculate variance/spacing
        features["component_count"] = float(len(valid_indices))
        vec = np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)
        return features, vec
        
    # Extract bounding box statistics
    heights = np.array([stats[i, cv2.CC_STAT_HEIGHT] for i in valid_indices], dtype=np.float64)
    widths = np.array([stats[i, cv2.CC_STAT_WIDTH] for i in valid_indices], dtype=np.float64)
    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in valid_indices], dtype=np.float64)
    x_coords = np.array([stats[i, cv2.CC_STAT_LEFT] for i in valid_indices], dtype=np.float64)
    y_coords = np.array([stats[i, cv2.CC_STAT_TOP] for i in valid_indices], dtype=np.float64)
    bottoms = y_coords + heights
    centroids_x = np.array([centroids[i][0] for i in valid_indices], dtype=np.float64)
    centroids_y = np.array([centroids[i][1] for i in valid_indices], dtype=np.float64)
    
    aspect_ratios = widths / np.maximum(heights, 1.0)
    
    # 1. Letter size mean & consistency (BHK #1, #8)
    mean_h = float(np.mean(heights))
    std_h = float(np.std(heights))
    cv_h = std_h / max(mean_h, 1e-4)
    cv_area = float(np.std(areas) / max(np.mean(areas), 1e-4))
    
    features["letter_size_mean"] = mean_h
    features["letter_size_std"] = std_h
    features["letter_size_cv"] = cv_h
    features["letter_area_cv"] = cv_area
    
    # 2. Aspect ratio statistics
    features["aspect_ratio_mean"] = float(np.mean(aspect_ratios))
    features["aspect_ratio_std"] = float(np.std(aspect_ratios))
    
    # 3. Left-hand margin variation (BHK #2)
    # Sort components by Y coordinate to cluster into lines, then find leftmost X
    features["left_margin_std"] = float(np.std(x_coords[:min(5, len(x_coords))]))
    
    # 4. Baseline drift & alignment (BHK #3)
    # Fit linear regression line across component bottoms: bottom_y = m * x + c
    if len(valid_indices) >= 3 and (np.max(centroids_x) - np.min(centroids_x) > 10):
        try:
            poly_fit = np.polyfit(centroids_x, bottoms, deg=1)
            slope = abs(float(poly_fit[0]))
            predicted_bottoms = np.polyval(poly_fit, centroids_x)
            residual_std = float(np.std(bottoms - predicted_bottoms))
        except Exception:
            slope = 0.0
            residual_std = float(np.std(bottoms))
    else:
        slope = 0.0
        residual_std = float(np.std(bottoms))
        
    features["baseline_drift_slope"] = slope
    features["baseline_drift_residual"] = residual_std
    
    # 5. Inter-component gap / spacing (BHK #4)
    # Sort components horizontally by left edge
    sorted_order = np.argsort(x_coords)
    sorted_x = x_coords[sorted_order]
    sorted_w = widths[sorted_order]
    
    gaps = []
    collisions = 0
    for idx in range(len(sorted_x) - 1):
        gap = sorted_x[idx + 1] - (sorted_x[idx] + sorted_w[idx])
        if gap < 0:
            collisions += 1
            gaps.append(0.0)
        else:
            gaps.append(float(gap))
            
    if len(gaps) > 0:
        gaps_arr = np.array(gaps)
        mean_gap = float(np.mean(gaps_arr))
        std_gap = float(np.std(gaps_arr))
        cv_gap = std_gap / max(mean_gap, 1e-4)
        features["inter_component_gap_mean"] = mean_gap
        features["inter_component_gap_cv"] = cv_gap
        features["letter_collision_ratio"] = float(collisions / len(gaps))
    else:
        features["inter_component_gap_mean"] = 0.0
        features["inter_component_gap_cv"] = 0.0
        features["letter_collision_ratio"] = 0.0
        
    # 6. Relative height ratio (BHK #9: Ascenders/descenders proportionality)
    p90_h = float(np.percentile(heights, 90))
    p50_h = float(np.median(heights))
    features["relative_height_ratio"] = p90_h / max(p50_h, 1e-4)
    
    # 7. Trace unsteadiness (BHK #13: Shakiness)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    unsteadiness_scores = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            var = calculate_contour_curvature_variance(cnt)
            unsteadiness_scores.append(var)
            
    if len(unsteadiness_scores) > 0:
        features["trace_unsteadiness_mean"] = float(np.mean(unsteadiness_scores))
    else:
        features["trace_unsteadiness_mean"] = 0.0
        
    # 8. Ink density within writing bounding box
    total_ink_pixels = float(np.sum(binary_mask > 0))
    min_x, max_x = np.min(x_coords), np.max(x_coords + widths)
    min_y, max_y = np.min(y_coords), np.max(bottoms)
    bbox_area = max(1.0, (max_x - min_x) * (max_y - min_y))
    features["ink_density"] = float(total_ink_pixels / bbox_area)
    
    # 9. Component count
    features["component_count"] = float(len(valid_indices))
    
    feature_vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    return features, feature_vector


def generate_feature_visualization(binary_mask: np.ndarray, base_rgb_img: np.ndarray = None) -> np.ndarray:
    """
    Renders an overlay highlighting character bounding boxes, centroids,
    and the fitted baseline regression line to visually explain BHK feature extraction.
    """
    h_img, w_img = binary_mask.shape
    if base_rgb_img is None:
        vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    else:
        vis = cv2.resize(base_rgb_img, (w_img, h_img)).copy()
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    valid_pts_x = []
    valid_bottoms_y = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        if area >= 12 and h >= 5 and w >= 3 and w < 0.75 * w_img and h < 0.85 * h_img:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            
            # Draw green bounding box around character component
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 100), 1)
            
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            # Draw centroid in cyan
            cv2.circle(vis, (cx, cy), 2, (255, 200, 0), -1)
            
            valid_pts_x.append(cx)
            valid_bottoms_y.append(y + h)
            
    # Draw fitted baseline regression line if possible
    if len(valid_pts_x) >= 3 and (max(valid_pts_x) - min(valid_pts_x) > 20):
        try:
            poly = np.polyfit(valid_pts_x, valid_bottoms_y, 1)
            x_start = int(min(valid_pts_x))
            x_end = int(max(valid_pts_x))
            y_start = int(np.polyval(poly, x_start))
            y_end = int(np.polyval(poly, x_end))
            # Draw fitted baseline in red/coral
            cv2.line(vis, (x_start, y_start), (x_end, y_end), (255, 60, 60), 2)
        except Exception:
            pass
            
    return vis

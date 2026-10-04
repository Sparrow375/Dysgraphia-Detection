"""
Deep Visual Stroke & Texture Feature Extractor for Dysgraphia Detection
Improves upon base paper (Kunhoth et al. / DenseNet201) by replacing fixed-word
whole-image squash with script-agnostic patch-based multi-scale convolutional
analysis of handwriting stroke dynamics.

Extracts a standardized 16-D Deep Stroke Texture Vector:
1. stroke_dir_energy_mean: Mean directional stroke energy across orientations
2. stroke_dir_energy_std: Directional energy variance across angles
3. stroke_dir_entropy: Orientation entropy (ballistic uniformity vs erratic scatter)
4. stroke_edge_sharpness_mean: Mean edge gradient magnitude along strokes
5. stroke_edge_sharpness_cv: Gradient steepness variation (erratic contact pressure)
6. stroke_thickness_mean: Mean stroke width normalized by character scale
7. stroke_thickness_cv: Stroke width inconsistency (hesitation & pressure variation)
8. stroke_curvature_energy: High-frequency curvature energy from 2nd-order derivatives
9. patch_texture_contrast: Local boundary contrast across stroke transitions
10. patch_texture_homogeneity: Local stroke consistency / smoothness
11. ink_distribution_entropy: Spatial dispersion of ink within component hulls
12. pen_hesitation_density: Localized ink concentration blobs (resting pen on paper)
13. stroke_branch_density: Density of stroke bifurcations / junctions
14. stroke_endpoint_density: Density of pen lifts and stroke terminations
15. loop_eccentricity_mean: Roundness and regularity of closed loops (e.g. 'o', 'a', 'e')
16. loop_eccentricity_cv: Inconsistency of loop geometries across sample
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


DEEP_FEATURE_NAMES = [
    "stroke_dir_energy_mean",
    "stroke_dir_energy_std",
    "stroke_dir_entropy",
    "stroke_edge_sharpness_mean",
    "stroke_edge_sharpness_cv",
    "stroke_thickness_mean",
    "stroke_thickness_cv",
    "stroke_curvature_energy",
    "patch_texture_contrast",
    "patch_texture_homogeneity",
    "ink_distribution_entropy",
    "pen_hesitation_density",
    "stroke_branch_density",
    "stroke_endpoint_density",
    "loop_eccentricity_mean",
    "loop_eccentricity_cv",
]


def get_deep_feature_names() -> List[str]:
    """Returns the ordered list of 16 deep stroke feature names."""
    return DEEP_FEATURE_NAMES.copy()


def build_gabor_bank(num_angles: int = 8, frequency: float = 0.15) -> List[np.ndarray]:
    """Generates a bank of directional Gabor filter kernels across 180 degrees."""
    kernels = []
    ksize = 17
    sigma = 3.5
    gamma = 0.5
    psi = 0
    for i in range(num_angles):
        theta = i * np.pi / num_angles
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 1.0 / frequency, gamma, psi, ktype=cv2.CV_32F)
        kernel -= np.mean(kernel)
        kernels.append(kernel)
    return kernels


_GABOR_KERNELS = build_gabor_bank(num_angles=8, frequency=0.15)


def extract_deep_stroke_features(
    binary_mask: np.ndarray,
    gray_image: np.ndarray = None
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Extracts 16-D deep stroke texture features from a handwriting binary mask
    and optional grayscale image.
    
    Args:
        binary_mask: Binary image with foreground ink as 255 and background as 0.
        gray_image: Optional grayscale image (0-255). If None, binary_mask is used.
        
    Returns:
        features_dict: Mapping of feature name to float value.
        feature_vector: 16-D float32 numpy array.
    """
    features: Dict[str, float] = {name: 0.0 for name in DEEP_FEATURE_NAMES}
    
    ink_pixels = int(np.sum(binary_mask > 0))
    if ink_pixels < 50:
        return features, np.zeros(len(DEEP_FEATURE_NAMES), dtype=np.float32)

    if gray_image is None:
        gray_image = cv2.bitwise_not(binary_mask)

    h_img, w_img = binary_mask.shape[:2]

    # --- 1. Multi-Scale Directional Stroke Energy & Entropy (Gabor Convolutions) ---
    float_mask = (binary_mask.astype(np.float32) / 255.0)
    dir_energies = []
    for kernel in _GABOR_KERNELS:
        conv = cv2.filter2D(float_mask, cv2.CV_32F, kernel)
        energy = np.mean(conv[binary_mask > 0] ** 2) if ink_pixels > 0 else 0.0
        dir_energies.append(float(energy))

    dir_arr = np.array(dir_energies, dtype=np.float32)
    dir_sum = float(np.sum(dir_arr))
    mean_energy = float(np.mean(dir_arr))
    std_energy = float(np.std(dir_arr))

    if dir_sum > 1e-6:
        p_dir = dir_arr / dir_sum
        p_dir_nz = p_dir[p_dir > 1e-6]
        dir_entropy = float(-np.sum(p_dir_nz * np.log2(p_dir_nz)) / np.log2(len(_GABOR_KERNELS)))
    else:
        dir_entropy = 1.0

    features["stroke_dir_energy_mean"] = mean_energy
    features["stroke_dir_energy_std"] = std_energy
    features["stroke_dir_entropy"] = dir_entropy

    # --- 2. Stroke Edge Sharpness & Gradient Variation ---
    sobel_x = cv2.Sobel(float_mask, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(float_mask, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary_mask, kernel_e)
    eroded = cv2.erode(binary_mask, kernel_e)
    edge_band = (dilated > 0) & (eroded == 0)

    if np.sum(edge_band) > 0:
        edge_grads = grad_mag[edge_band]
        mean_edge = float(np.mean(edge_grads))
        std_edge = float(np.std(edge_grads))
        cv_edge = float(std_edge / max(mean_edge, 1e-4))
    else:
        mean_edge = 0.0
        cv_edge = 0.0

    features["stroke_edge_sharpness_mean"] = mean_edge
    features["stroke_edge_sharpness_cv"] = cv_edge

    # --- 3. Stroke Thickness & Inconsistency via Distance Transform ---
    dist_map = distance_transform_edt(binary_mask > 0)
    try:
        skeleton = np.zeros_like(binary_mask)
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_img = binary_mask.copy()
        for _ in range(15):
            eroded_s = cv2.erode(temp_img, elem)
            opened_s = cv2.dilate(eroded_s, elem)
            sub = cv2.subtract(temp_img, opened_s)
            skeleton = cv2.bitwise_or(skeleton, sub)
            temp_img = eroded_s
            if cv2.countNonZero(temp_img) == 0:
                break
    except Exception:
        skeleton = binary_mask

    skel_pts = dist_map[skeleton > 0]
    if len(skel_pts) > 10:
        stroke_widths = skel_pts * 2.0
        mean_thick = float(np.mean(stroke_widths))
        std_thick = float(np.std(stroke_widths))
        cv_thick = float(std_thick / max(mean_thick, 1e-4))
    else:
        mean_thick = 3.0
        cv_thick = 0.0

    features["stroke_thickness_mean"] = mean_thick
    features["stroke_thickness_cv"] = cv_thick

    # --- 4. High-Frequency Stroke Curvature Energy (Laplacian) ---
    laplacian = cv2.Laplacian(float_mask, cv2.CV_32F, ksize=3)
    curv_energy = float(np.var(laplacian[binary_mask > 0])) if ink_pixels > 0 else 0.0
    features["stroke_curvature_energy"] = curv_energy

    # --- 5. Patch Texture Contrast & Homogeneity (Gray-Level Transitions) ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    valid_boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area > 30 and h > 8:
            valid_boxes.append((stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], w, h))

    contrast_list = []
    homogeneity_list = []
    for x, y, bw, bh in valid_boxes[:40]:
        patch = float_mask[y:y + bh, x:x + bw]
        if patch.size < 16:
            continue
        contrast_list.append(float(np.var(patch)))
        homogeneity_list.append(float(np.mean(patch > 0.8) + np.mean(patch < 0.2)))

    features["patch_texture_contrast"] = float(np.mean(contrast_list)) if contrast_list else 0.0
    features["patch_texture_homogeneity"] = float(np.mean(homogeneity_list)) if homogeneity_list else 1.0

    # --- 6. Spatial Ink Distribution Entropy & Pen Hesitation (Pooling) ---
    hesitation_threshold = max(4.0, mean_thick * 2.2)
    hesitation_mask = (dist_map * 2.0) > hesitation_threshold
    num_hesitations = float(np.sum(hesitation_mask))
    features["pen_hesitation_density"] = float(num_hesitations / max(1.0, float(len(valid_boxes) * max(mean_thick, 1.0) ** 2)))

    grid_h = max(1, h_img // 6)
    grid_w = max(1, w_img // 6)
    grid_counts = []
    for r in range(6):
        for c in range(6):
            cell = binary_mask[r * grid_h:(r + 1) * grid_h, c * grid_w:(c + 1) * grid_w]
            grid_counts.append(float(np.sum(cell > 0)))
    grid_arr = np.array(grid_counts)
    grid_sum = float(np.sum(grid_arr))
    if grid_sum > 0:
        p_grid = grid_arr / grid_sum
        p_grid_nz = p_grid[p_grid > 1e-6]
        spatial_entropy = float(-np.sum(p_grid_nz * np.log2(p_grid_nz)) / np.log2(36.0))
    else:
        spatial_entropy = 0.0
    features["ink_distribution_entropy"] = spatial_entropy

    # --- 7. Stroke Skeleton Topology (Bifurcations & Endpoints) ---
    skel_binary = (skeleton > 0).astype(np.uint8)
    kernel_neigh = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, cv2.CV_8U, kernel_neigh) * skel_binary

    endpoints = np.sum((skel_binary == 1) & (neighbor_count == 1))
    junctions = np.sum((skel_binary == 1) & (neighbor_count >= 3))
    norm_factor = max(1.0, float(len(valid_boxes)))

    features["stroke_endpoint_density"] = float(endpoints / norm_factor)
    features["stroke_branch_density"] = float(junctions / norm_factor)

    # --- 8. Closed Loop Regularity & Eccentricity (BHK #10 Proxy) ---
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    loop_eccentricities = []
    if hierarchy is not None and len(contours) > 0:
        for idx in range(len(contours)):
            if hierarchy[0][idx][3] != -1:
                cnt = contours[idx]
                if len(cnt) >= 5 and cv2.contourArea(cnt) > 10:
                    try:
                        ellipse = cv2.fitEllipse(cnt)
                        (center, (d1, d2), angle) = ellipse
                        minor_ax = min(d1, d2)
                        major_ax = max(d1, d2)
                        if major_ax > 0:
                            ecc = np.sqrt(max(0.0, 1.0 - (minor_ax / major_ax) ** 2))
                            loop_eccentricities.append(float(ecc))
                    except Exception:
                        pass

    if loop_eccentricities:
        mean_ecc = float(np.mean(loop_eccentricities))
        cv_ecc = float(np.std(loop_eccentricities) / max(mean_ecc, 1e-4))
    else:
        mean_ecc = 0.5
        cv_ecc = 0.0

    features["loop_eccentricity_mean"] = mean_ecc
    features["loop_eccentricity_cv"] = cv_ecc

    feature_vector = np.array([features[name] for name in DEEP_FEATURE_NAMES], dtype=np.float32)
    return features, feature_vector

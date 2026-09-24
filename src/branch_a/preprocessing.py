"""
Preprocessing module for static handwriting images.
Performs:
  1. Grayscale & adaptive binarization (foreground ink = 1, background = 0)
  2. Zhang-Suen morphological skeletonization (1-pixel centerline)
  3. Distance-transform stroke width estimation and normalization
"""

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter


def load_and_binarize(image_input, otsu_fallback: bool = True) -> np.ndarray:
    """
    Loads an image (file path, PIL Image, or numpy array) and converts it
    to a binary mask where ink=1 and background=0.
    Handles uneven illumination, camera shadows, and noisy background.
    """
    if isinstance(image_input, str):
        img_pil = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("L")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            img_pil = Image.fromarray(image_input).convert("L")
        else:
            img_pil = Image.fromarray(image_input.astype(np.uint8))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    img_gray = np.array(img_pil, dtype=np.float32)

    # Local background estimation via large Gaussian blur
    # Handles paper texture, gradients, and shadows
    sigma = max(img_gray.shape) / 30.0
    bg_estimate = gaussian_filter(img_gray, sigma=sigma)

    # Relative difference: ink is darker than estimated local background
    diff = bg_estimate - img_gray

    # Otsu thresholding on the difference map
    diff_clipped = np.clip(diff, 0, 255)
    hist, bin_edges = np.histogram(diff_clipped, bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)

    # Compute Otsu's threshold
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]

    # Between-class variance
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sb = (mu_t * omega - mu)**2 / denom
    best_thresh = np.nanargmax(sb) if not np.all(np.isnan(sb)) else 25.0

    # Ensure a reasonable minimum threshold to ignore flat background noise
    thresh = max(best_thresh, 15.0)
    binary = (diff_clipped >= thresh).astype(np.uint8)

    # Clean isolated speckles: remove tiny 1-2 pixel components
    from scipy.ndimage import label, sum_labels
    labeled, num_features = label(binary)
    if num_features > 0:
        sizes = sum_labels(binary, labeled, range(1, num_features + 1))
        # Keep components with size >= 5 pixels
        keep = np.zeros(num_features + 1, dtype=bool)
        keep[1:] = sizes >= 5
        binary = keep[labeled].astype(np.uint8)

    return binary


def zhang_suen_skeletonize(binary_img: np.ndarray) -> np.ndarray:
    """
    Implements the Zhang-Suen thinning algorithm in NumPy to produce a
    topology-preserving 1-pixel centerline skeleton of the binary ink.
    Input: binary_img (0=bg, 1=ink)
    Output: skeleton (0=bg, 1=skeleton)
    """
    img = (binary_img > 0).astype(np.uint8)
    # Pad with 1 pixel of zero to avoid boundary conditions
    padded = np.pad(img, 1, mode="constant", constant_values=0)

    changed = True
    while changed:
        changed = False
        for step in (1, 2):
            # 8-neighbors:
            # P9 P2 P3
            # P8 P1 P4
            # P7 P6 P5
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            p1 = padded[1:-1, 1:-1]

            # Condition 1: 2 <= B(P1) <= 6 (number of nonzero neighbors)
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            c1 = (b >= 2) & (b <= 6)

            # Condition 2: A(P1) == 1 (0 -> 1 transitions in cyclic order)
            # Order: p2, p3, p4, p5, p6, p7, p8, p9, p2
            a = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8) +
                ((p3 == 0) & (p4 == 1)).astype(np.uint8) +
                ((p4 == 0) & (p5 == 1)).astype(np.uint8) +
                ((p5 == 0) & (p6 == 1)).astype(np.uint8) +
                ((p6 == 0) & (p7 == 1)).astype(np.uint8) +
                ((p7 == 0) & (p8 == 1)).astype(np.uint8) +
                ((p8 == 0) & (p9 == 1)).astype(np.uint8) +
                ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            c2 = (a == 1)

            if step == 1:
                # Sub-iteration 1: P2*P4*P6 == 0 and P4*P6*P8 == 0
                c3 = (p2 * p4 * p6 == 0)
                c4 = (p4 * p6 * p8 == 0)
            else:
                # Sub-iteration 2: P2*P4*P8 == 0 and P2*P6*P8 == 0
                c3 = (p2 * p4 * p8 == 0)
                c4 = (p2 * p6 * p8 == 0)

            # Points to delete
            candidates = (p1 == 1) & c1 & c2 & c3 & c4
            if np.any(candidates):
                padded[1:-1, 1:-1][candidates] = 0
                changed = True

    return padded[1:-1, 1:-1].astype(np.uint8)


def compute_stroke_width_map(binary_img: np.ndarray, skeleton: np.ndarray):
    """
    Computes Euclidean Distance Transform of binary ink mask.
    At skeleton pixels, distance value represents local stroke radius.
    Local stroke width = 2.0 * distance_transform.
    Returns:
      dist_map: 2D float array of distances to nearest background
      skeleton_widths: 1D array of local stroke widths sampled at skeleton pixels
      width_map: 2D array where skeleton pixels store their local stroke width
    """
    dist_map = distance_transform_edt(binary_img)
    skel_mask = skeleton > 0

    if np.any(skel_mask):
        skeleton_widths = 2.0 * dist_map[skel_mask]
    else:
        skeleton_widths = np.array([1.0], dtype=np.float64)

    width_map = np.zeros_like(dist_map)
    width_map[skel_mask] = 2.0 * dist_map[skel_mask]

    return dist_map, skeleton_widths, width_map

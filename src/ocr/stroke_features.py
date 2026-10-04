"""
Stroke Feature Extraction — Topological Skeleton Analysis.

Extracts structural stroke primitives (ascenders, descenders, loops, crossings,
junctions) from handwriting that survive even severe dysgraphic distortion.

These primitives produce a soft character-likelihood prior P(char | strokes) that
supplements the CNN visual model's output during beam search decoding.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2

from src.ocr.utils import StrokePrimitive, StrokeSegment, StrokeAnalysis


# ---------------------------------------------------------------------------
# Character ↔ Stroke Primitive Lookup Table
# ---------------------------------------------------------------------------
# Maps characters to the stroke primitives they typically contain.
# Used in reverse to compute P(char | observed_primitives).

CHAR_STROKE_MAP: Dict[str, List[StrokePrimitive]] = {
    # Ascender characters
    'b': [StrokePrimitive.ASCENDER, StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.CLOSED_LOOP],
    'd': [StrokePrimitive.ASCENDER, StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.CLOSED_LOOP],
    'f': [StrokePrimitive.ASCENDER, StrokePrimitive.DESCENDER, StrokePrimitive.HORIZONTAL_CROSS],
    'h': [StrokePrimitive.ASCENDER, StrokePrimitive.VERTICAL_STROKE],
    'k': [StrokePrimitive.ASCENDER, StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.DIAGONAL_RIGHT, StrokePrimitive.JUNCTION],
    'l': [StrokePrimitive.ASCENDER, StrokePrimitive.VERTICAL_STROKE],
    't': [StrokePrimitive.ASCENDER, StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.HORIZONTAL_CROSS],

    # Descender characters
    'g': [StrokePrimitive.DESCENDER, StrokePrimitive.CLOSED_LOOP],
    'j': [StrokePrimitive.DESCENDER, StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.DOT],
    'p': [StrokePrimitive.DESCENDER, StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.CLOSED_LOOP],
    'q': [StrokePrimitive.DESCENDER, StrokePrimitive.CLOSED_LOOP],
    'y': [StrokePrimitive.DESCENDER, StrokePrimitive.DIAGONAL_RIGHT, StrokePrimitive.DIAGONAL_LEFT],

    # Loop / round characters (x-height zone)
    'a': [StrokePrimitive.CLOSED_LOOP, StrokePrimitive.VERTICAL_STROKE],
    'c': [StrokePrimitive.OPEN_CURVE_RIGHT],
    'e': [StrokePrimitive.CLOSED_LOOP, StrokePrimitive.OPEN_CURVE_RIGHT],
    'o': [StrokePrimitive.CLOSED_LOOP],

    # Vertical strokes (x-height zone)
    'i': [StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.DOT],
    'n': [StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.VERTICAL_STROKE],
    'm': [StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.VERTICAL_STROKE],
    'r': [StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.OPEN_CURVE_RIGHT],
    'u': [StrokePrimitive.VERTICAL_STROKE, StrokePrimitive.VERTICAL_STROKE],

    # Diagonal characters
    'v': [StrokePrimitive.DIAGONAL_LEFT, StrokePrimitive.DIAGONAL_RIGHT],
    'w': [StrokePrimitive.DIAGONAL_LEFT, StrokePrimitive.DIAGONAL_RIGHT, StrokePrimitive.DIAGONAL_LEFT, StrokePrimitive.DIAGONAL_RIGHT],
    'x': [StrokePrimitive.DIAGONAL_LEFT, StrokePrimitive.DIAGONAL_RIGHT, StrokePrimitive.JUNCTION],
    'z': [StrokePrimitive.HORIZONTAL_CROSS, StrokePrimitive.DIAGONAL_LEFT, StrokePrimitive.HORIZONTAL_CROSS],

    # Special
    's': [StrokePrimitive.OPEN_CURVE_RIGHT, StrokePrimitive.OPEN_CURVE_LEFT],
}

# Include uppercase (same primitives, often with ASCENDER added)
for _c, _prims in list(CHAR_STROKE_MAP.items()):
    upper = _c.upper()
    if upper not in CHAR_STROKE_MAP:
        CHAR_STROKE_MAP[upper] = [StrokePrimitive.ASCENDER] + _prims


# ---------------------------------------------------------------------------
# Skeletonization
# ---------------------------------------------------------------------------

def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute the morphological skeleton (medial axis) of a binary image.

    Uses iterative thinning via OpenCV morphology. Falls back to a simple
    iterative approach if scikit-image is not available.

    Args:
        binary_mask: Binary image (ink=255, bg=0), dtype uint8.

    Returns:
        Skeleton image, same size, dtype uint8 (skeleton=255, bg=0).
    """
    try:
        from skimage.morphology import skeletonize as ski_skeletonize
        # skimage expects boolean input
        bool_mask = binary_mask > 0
        skel = ski_skeletonize(bool_mask).astype(np.uint8) * 255
        return skel
    except ImportError:
        pass

    # Fallback: OpenCV iterative thinning (Zhang-Suen approximation)
    img = binary_mask.copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


# ---------------------------------------------------------------------------
# Skeleton Graph Analysis
# ---------------------------------------------------------------------------

def _get_neighbors(skel: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    """Get 8-connected skeleton neighbor coordinates."""
    h, w = skel.shape
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0:
                neighbors.append((ny, nx))
    return neighbors


def find_skeleton_keypoints(
    skel: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Find endpoints and junction points on a skeleton image.

    Endpoints: pixels with exactly 1 skeleton neighbor.
    Junctions: pixels with 3+ skeleton neighbors.

    Args:
        skel: Skeleton binary image (255/0).

    Returns:
        (endpoints, junctions) as lists of (y, x) coordinates.
    """
    h, w = skel.shape
    endpoints = []
    junctions = []

    ys, xs = np.where(skel > 0)
    for y, x in zip(ys, xs):
        n_neighbors = len(_get_neighbors(skel, y, x))
        if n_neighbors == 1:
            endpoints.append((y, x))
        elif n_neighbors >= 3:
            junctions.append((y, x))

    return endpoints, junctions


def trace_stroke_segments(
    skel: np.ndarray,
    endpoints: List[Tuple[int, int]],
    junctions: List[Tuple[int, int]],
) -> List[np.ndarray]:
    """
    Trace individual stroke segments along the skeleton.

    A segment runs from one keypoint (endpoint or junction) to another,
    following the skeleton path. This decomposes the skeleton into
    individual "strokes" that can be independently classified.

    Args:
        skel: Skeleton binary image.
        endpoints: List of (y, x) endpoint coordinates.
        junctions: List of (y, x) junction coordinates.

    Returns:
        List of stroke paths, each as an (N, 2) array of (x, y) coordinates.
    """
    keypoints = set(endpoints + junctions)
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    segments = []

    # Start tracing from each endpoint
    start_points = endpoints if endpoints else junctions
    for start in start_points:
        sy, sx = start
        if visited[sy, sx] and start not in junctions:
            continue

        # Trace path from this start point
        path = [(sx, sy)]
        visited[sy, sx] = True
        current = start

        while True:
            cy, cx = current
            neighbors = _get_neighbors(skel, cy, cx)

            # Find unvisited neighbor to continue (or junction to stop at)
            next_point = None
            for ny, nx in neighbors:
                if not visited[ny, nx]:
                    next_point = (ny, nx)
                    break
                elif (ny, nx) in keypoints and (ny, nx) != start and len(path) > 2:
                    # Reached another keypoint — end segment here
                    path.append((nx, ny))
                    next_point = None
                    break

            if next_point is None:
                break

            ny, nx = next_point
            path.append((nx, ny))
            visited[ny, nx] = True
            current = next_point

            # Stop if we've reached a keypoint (other than start)
            if current in keypoints and current != start:
                break

        if len(path) >= 3:
            segments.append(np.array(path, dtype=np.float32))

    return segments


# ---------------------------------------------------------------------------
# Stroke Classification
# ---------------------------------------------------------------------------

def _compute_dominant_angle(points: np.ndarray) -> float:
    """
    Compute the dominant orientation angle of a stroke segment.

    Uses PCA (first principal component) on the point cloud.

    Returns:
        Angle in degrees from horizontal (0° = horizontal, 90° = vertical).
    """
    if len(points) < 3:
        return 0.0

    centered = points - np.mean(points, axis=0)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        # First principal component direction
        dx, dy = Vt[0]
        angle = np.abs(np.degrees(np.arctan2(dy, dx)))
        # Normalize to 0–90 range
        if angle > 90:
            angle = 180 - angle
        return float(angle)
    except np.linalg.LinAlgError:
        return 0.0


def classify_stroke_segment(
    points: np.ndarray,
    x_height_top: float,
    baseline_y: float,
    median_char_h: float,
) -> StrokeSegment:
    """
    Classify a traced stroke segment into a StrokePrimitive type.

    Uses the segment's geometry, position relative to x-height/baseline zones,
    and dominant orientation angle.

    Args:
        points: (N, 2) array of (x, y) coordinates along the stroke.
        x_height_top: Y-coordinate of the top of the x-height zone.
        baseline_y: Y-coordinate of the baseline.
        median_char_h: Median character height for normalization.

    Returns:
        StrokeSegment with classified primitive type.
    """
    if len(points) < 2:
        return StrokeSegment(
            primitive=StrokePrimitive.DOT,
            points=points,
        )

    # Geometric properties
    angle = _compute_dominant_angle(points)
    ys = points[:, 1]
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))
    length = float(np.sum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))))
    length_norm = length / max(median_char_h, 1.0)

    # Zone classification
    extends_above = min_y < (x_height_top - 0.15 * median_char_h)
    extends_below = max_y > (baseline_y + 0.15 * median_char_h)

    start_y = float(ys[0])
    end_y = float(ys[-1])
    start_zone = "above" if start_y < x_height_top else ("below" if start_y > baseline_y else "x-height")
    end_zone = "above" if end_y < x_height_top else ("below" if end_y > baseline_y else "x-height")

    # Classify primitive type
    if length_norm < 0.15:
        primitive = StrokePrimitive.DOT
    elif extends_above and extends_below:
        # Spans the full height — likely an ascender+descender like 'f'
        primitive = StrokePrimitive.ASCENDER  # Primary classification
    elif extends_above:
        primitive = StrokePrimitive.ASCENDER
    elif extends_below:
        primitive = StrokePrimitive.DESCENDER
    elif angle > 70:
        primitive = StrokePrimitive.VERTICAL_STROKE
    elif angle < 20:
        primitive = StrokePrimitive.HORIZONTAL_CROSS
    elif angle >= 20 and angle <= 70:
        # Determine diagonal direction from start-to-end x movement
        dx = points[-1, 0] - points[0, 0]
        primitive = StrokePrimitive.DIAGONAL_RIGHT if dx > 0 else StrokePrimitive.DIAGONAL_LEFT
    else:
        primitive = StrokePrimitive.VERTICAL_STROKE

    return StrokeSegment(
        primitive=primitive,
        points=points,
        angle_deg=angle,
        length_norm=length_norm,
        start_zone=start_zone,
        end_zone=end_zone,
    )


def detect_closed_loops(binary_mask: np.ndarray) -> int:
    """
    Count closed loops (enclosed regions) in a binary character image.

    Uses contour hierarchy to find inner contours (holes in the ink),
    which correspond to loops in letters like o, a, d, g, b, p, q, e.

    Args:
        binary_mask: Binary image of a single character or word region.

    Returns:
        Number of detected closed loops.
    """
    # Invert so holes become white regions
    inverted = cv2.bitwise_not(binary_mask)

    contours, hierarchy = cv2.findContours(
        inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return 0

    n_loops = 0
    for i in range(len(contours)):
        # Inner contours have a parent (hierarchy[0][i][3] >= 0)
        if hierarchy[0][i][3] >= 0:
            area = cv2.contourArea(contours[i])
            # Filter tiny noise holes
            if area >= 15:
                n_loops += 1

    return n_loops


# ---------------------------------------------------------------------------
# Complete Stroke Analysis
# ---------------------------------------------------------------------------

def analyze_strokes(
    char_image: np.ndarray,
    x_height_top: Optional[float] = None,
    baseline_y: Optional[float] = None,
    median_char_h: Optional[float] = None,
) -> StrokeAnalysis:
    """
    Complete stroke analysis for a character or small word region.

    Pipeline:
    1. Skeletonize the binary image
    2. Find endpoints and junctions on the skeleton
    3. Trace individual stroke segments
    4. Classify each segment
    5. Detect closed loops
    6. Compute character likelihood prior P(char | strokes)

    Args:
        char_image: Binary image (ink=255, bg=0) of a single character/word.
        x_height_top: Y-coordinate of x-height top (default: estimated from image).
        baseline_y: Y-coordinate of baseline (default: estimated from image).
        median_char_h: Median character height (default: estimated from image).

    Returns:
        StrokeAnalysis with all detected primitives and character prior.
    """
    h_img, w_img = char_image.shape

    # Estimate zone boundaries if not provided
    if median_char_h is None:
        # Use image height as rough estimate
        coords = cv2.findNonZero(char_image)
        if coords is not None:
            _, _, _, ch = cv2.boundingRect(coords)
            median_char_h = float(ch)
        else:
            median_char_h = float(h_img)

    if x_height_top is None:
        x_height_top = h_img * 0.25  # Top 25% = ascender zone

    if baseline_y is None:
        baseline_y = h_img * 0.75  # Bottom 25% = descender zone

    # Step 1: Skeletonize
    skel = skeletonize(char_image)

    # Step 2: Find keypoints
    endpoints, junctions = find_skeleton_keypoints(skel)

    # Step 3: Trace segments
    raw_segments = trace_stroke_segments(skel, endpoints, junctions)

    # Step 4: Classify segments
    classified_segments = [
        classify_stroke_segment(seg, x_height_top, baseline_y, median_char_h)
        for seg in raw_segments
    ]

    # Step 5: Detect closed loops
    n_loops = detect_closed_loops(char_image)

    # Add CLOSED_LOOP primitives to the analysis
    for _ in range(n_loops):
        classified_segments.append(StrokeSegment(
            primitive=StrokePrimitive.CLOSED_LOOP,
            points=np.array([], dtype=np.float32).reshape(0, 2),
        ))

    # Assemble analysis
    has_ascender = any(s.primitive == StrokePrimitive.ASCENDER for s in classified_segments)
    has_descender = any(s.primitive == StrokePrimitive.DESCENDER for s in classified_segments)

    analysis = StrokeAnalysis(
        segments=classified_segments,
        n_endpoints=len(endpoints),
        n_junctions=len(junctions),
        n_loops=n_loops,
        has_ascender=has_ascender,
        has_descender=has_descender,
    )

    # Step 6: Compute character prior
    analysis.char_prior = compute_char_prior(analysis)

    return analysis


# ---------------------------------------------------------------------------
# Character Prior from Stroke Primitives
# ---------------------------------------------------------------------------

def compute_char_prior(
    analysis: StrokeAnalysis,
    smoothing: float = 0.01,
) -> Dict[str, float]:
    """
    Compute P(char | observed_stroke_primitives) using a simple
    Jaccard-like similarity between observed and expected primitives.

    For each character in CHAR_STROKE_MAP, compute:
        similarity = |observed ∩ expected| / |observed ∪ expected|

    Then normalize to a probability distribution.

    Args:
        analysis: StrokeAnalysis with detected primitives.
        smoothing: Laplace smoothing to avoid zero probabilities.

    Returns:
        Dict mapping characters to probabilities (sums to 1.0).
    """
    observed = set(analysis.primitives)

    if not observed:
        # No strokes detected — uniform prior
        n_chars = len(CHAR_STROKE_MAP)
        return {c: 1.0 / n_chars for c in CHAR_STROKE_MAP}

    scores: Dict[str, float] = {}

    for char, expected_primitives in CHAR_STROKE_MAP.items():
        expected = set(expected_primitives)

        # Jaccard similarity
        intersection = len(observed & expected)
        union = len(observed | expected)
        jaccard = intersection / max(union, 1)

        # Bonus for matching ascender/descender presence
        bonus = 0.0
        if analysis.has_ascender and StrokePrimitive.ASCENDER in expected:
            bonus += 0.1
        if analysis.has_descender and StrokePrimitive.DESCENDER in expected:
            bonus += 0.1
        if analysis.n_loops > 0 and StrokePrimitive.CLOSED_LOOP in expected:
            bonus += 0.1

        # Penalty for missing expected primitives
        missing = expected - observed
        penalty = 0.05 * len(missing)

        scores[char] = max(smoothing, jaccard + bonus - penalty)

    # Normalize to probability distribution
    total = sum(scores.values())
    if total > 0:
        scores = {c: s / total for c, s in scores.items()}

    return scores


def stroke_prior_to_ctc_bias(
    char_prior: Dict[str, float],
    alphabet: str,
    weight: float = 0.15,
) -> np.ndarray:
    """
    Convert a character prior distribution into a log-probability bias
    vector for CTC beam search augmentation.

    Args:
        char_prior: P(char | strokes) from compute_char_prior().
        alphabet: String of all characters in the CTC alphabet.
        weight: How much to weight the stroke prior (0 = ignore, 1 = fully trust).

    Returns:
        (|alphabet| + 1,) array of log-probability biases.
        Index 0 is the CTC blank token (no bias applied).
    """
    bias = np.zeros(len(alphabet) + 1, dtype=np.float32)

    for i, char in enumerate(alphabet):
        if char in char_prior:
            # Convert probability to log-space and scale by weight
            bias[i + 1] = weight * np.log(max(char_prior[char], 1e-10))

    return bias


# Alias for pipeline consistency
extract_stroke_features = analyze_strokes

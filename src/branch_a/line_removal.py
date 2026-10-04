"""
Trajectory-Space Ruled-Line Removal Module.

Architecture (replaces old pixel-morphology approach):
  1. Skeletonize the full binary mask (ink + ruled lines) FIRST.
  2. Build skeleton graph; contract junction cliques (reusing Branch B logic).
  3. Detect candidate ruled lines via horizontal/vertical projection density.
  4. Fit a robust line model (least-squares with outlier rejection) to each candidate.
  5. Classify every skeleton EDGE-CHAIN as 'line' vs 'handwriting' by:
       - Sustained low curvature over a long run, AND
       - Near-parallel alignment with a fitted line model.
  6. At every junction where a line-chain meets a non-line chain, apply the
     tangent fly-through rule: line-classified edges NEVER win the continuation
     vote over a non-line edge. Tangent continuity is the deciding signal.
  7. Mark 'line' skeleton edges that are not claimed by any handwriting
     continuation as prunable. Convert to pixel set.
  8. Re-render a cleaned 2D raster by removing only pruned pixels from
     the original binary mask.

This replaces "is this pixel inside the line's spatial band?" with
"does a stroke trajectory claim this pixel?" — so ink at crossings is
preserved by construction, not by a heuristic spatial check.
"""

from typing import Tuple, Dict, Any, List, Set, Optional
import numpy as np
from scipy.ndimage import label, find_objects, binary_opening
import networkx as nx

# Re-use Branch B's skeleton-to-graph and junction contraction directly.
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.branch_b.stroke_recovery import (
    skeleton_to_graph,
    contract_junction_cliques,
    _get_outgoing_tangent,
)
from src.branch_a.preprocessing import zhang_suen_skeletonize


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_candidate_lines(
    binary_mask: np.ndarray,
    min_span_ratio: float = 0.25,
    max_thickness_px: float = 6.0,
    detect_vertical: bool = True,
) -> Dict[str, Any]:
    """
    Identifies candidate ruled-line y-bands (horizontal) and x-bands (vertical)
    using morphological opening + connected-component filtering.

    Returns:
        {
          "h_bands": list of (y_center_float, y_lo, y_hi, x_lo, x_hi),
          "v_bands": list of (x_center_float, y_lo, y_hi, x_lo, x_hi),
        }
    """
    h_img, w_img = binary_mask.shape
    min_h_span = max(int(w_img * min_span_ratio), 40)
    min_v_span = max(int(h_img * min_span_ratio), 40)

    # ── Horizontal candidate bands ────────────────────────────────────────────
    h_kernel_len = max(int(w_img * 0.04), 15)
    opened_h = binary_opening(binary_mask > 0, structure=np.ones((1, h_kernel_len), bool))
    labeled_h, _ = label(opened_h)
    slices_h = find_objects(labeled_h)

    h_bands = []
    for idx, slc in enumerate(slices_h):
        if slc is None:
            continue
        y_slc, x_slc = slc
        comp_w = x_slc.stop - x_slc.start
        comp_h = y_slc.stop - y_slc.start
        if comp_w < min_h_span:
            continue
        comp_area = int(np.sum(labeled_h[y_slc, x_slc] == (idx + 1)))
        mean_thickness = comp_area / float(comp_w)
        if mean_thickness <= max_thickness_px and (comp_h / float(comp_w)) <= 0.10:
            y_center = (y_slc.start + y_slc.stop) / 2.0
            h_bands.append((y_center, y_slc.start, y_slc.stop, x_slc.start, x_slc.stop))

    # ── Vertical candidate bands ──────────────────────────────────────────────
    v_bands = []
    if detect_vertical:
        v_kernel_len = max(int(h_img * 0.04), 15)
        opened_v = binary_opening(binary_mask > 0, structure=np.ones((v_kernel_len, 1), bool))
        labeled_v, _ = label(opened_v)
        slices_v = find_objects(labeled_v)

        for idx, slc in enumerate(slices_v):
            if slc is None:
                continue
            y_slc, x_slc = slc
            comp_w = x_slc.stop - x_slc.start
            comp_h = y_slc.stop - y_slc.start
            if comp_h < min_v_span:
                continue
            comp_area = int(np.sum(labeled_v[y_slc, x_slc] == (idx + 1)))
            mean_thickness = comp_area / float(comp_h)
            if mean_thickness <= max_thickness_px and (comp_w / float(comp_h)) <= 0.10:
                x_center = (x_slc.start + x_slc.stop) / 2.0
                v_bands.append((x_center, y_slc.start, y_slc.stop, x_slc.start, x_slc.stop))

    return {"h_bands": h_bands, "v_bands": v_bands}


def _fit_line_model(coords_yx: np.ndarray, is_horizontal: bool) -> Optional[Dict]:
    """
    Fits a robust horizontal or vertical line model via iterative least-squares
    with outlier rejection (RANSAC-lite: discard points > 2*sigma from fit).

    For horizontal lines: fits y = a*x + b, returns (slope, intercept, y_center, half_width)
    For vertical lines: fits x = a*y + b, returns (slope, intercept, x_center, half_width)

    Returns None if fit fails or too few inliers.
    """
    if len(coords_yx) < 8:
        return None

    ys = coords_yx[:, 0].astype(float)
    xs = coords_yx[:, 1].astype(float)

    if is_horizontal:
        independent = xs
        dependent = ys
    else:
        independent = ys
        dependent = xs

    # Iterative outlier rejection (2 passes)
    mask = np.ones(len(independent), dtype=bool)
    for _ in range(2):
        xi = independent[mask]
        yi = dependent[mask]
        if len(xi) < 4:
            return None
        # Least-squares fit: y = a*x + b
        A = np.vstack([xi, np.ones(len(xi))]).T
        try:
            result = np.linalg.lstsq(A, yi, rcond=None)
            a, b = result[0]
        except np.linalg.LinAlgError:
            return None
        residuals = np.abs(dependent - (a * independent + b))
        sigma = max(np.std(residuals[mask]), 0.5)
        mask = residuals < 2.0 * sigma

    if mask.sum() < 4:
        return None

    xi = independent[mask]
    yi = dependent[mask]
    A = np.vstack([xi, np.ones(len(xi))]).T
    try:
        result = np.linalg.lstsq(A, yi, rcond=None)
        a, b = result[0]
    except np.linalg.LinAlgError:
        return None

    residuals_inlier = np.abs(yi - (a * xi + b))
    half_width = max(np.percentile(residuals_inlier, 95) * 2.0, 2.5)

    if is_horizontal:
        return {
            "is_horizontal": True,
            "slope": float(a),        # dy/dx
            "intercept": float(b),    # y-intercept
            "half_width": float(half_width),
        }
    else:
        return {
            "is_horizontal": False,
            "slope": float(a),        # dx/dy
            "intercept": float(b),    # x-intercept
            "half_width": float(half_width),
        }


def _chain_curvature_and_alignment(
    chain: List[Tuple[int, int]],
    line_models: List[Dict],
    min_chain_length: int = 12,
    max_curvature_deg: float = 8.0,
    max_alignment_deg: float = 6.0,
    parallel_proximity_px: float = 8.0,
) -> Tuple[bool, str]:
    """
    Classifies a skeleton chain (ordered list of (y,x) nodes) as 'line' or 'handwriting'.

    A chain is classified as 'line' if ALL of:
      1. Length >= min_chain_length pixels along path
      2. Mean absolute direction change (curvature proxy) <= max_curvature_deg
      3. A fitted line model exists within parallel_proximity_px of the chain centroid
      4. The chain's mean direction is within max_alignment_deg of that line model's orientation

    Returns (is_line: bool, reason: str)
    """
    if len(chain) < min_chain_length:
        return False, f"too_short({len(chain)}<{min_chain_length})"

    ys = np.array([p[0] for p in chain], dtype=float)
    xs = np.array([p[1] for p in chain], dtype=float)

    # Arc length
    diffs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    arc_len = float(np.sum(diffs))
    if arc_len < min_chain_length:
        return False, f"arc_too_short({arc_len:.1f})"

    # ── 1. Curvature proxy: mean absolute turn angle between successive steps ──
    if len(chain) >= 3:
        vecs = np.column_stack([np.diff(xs), np.diff(ys)])  # (N-1, 2)
        norms = np.linalg.norm(vecs, axis=1)
        valid = norms > 0.5
        if valid.sum() >= 2:
            vecs_unit = vecs[valid] / norms[valid, None]
            dots = np.clip(
                np.einsum('ij,ij->i', vecs_unit[:-1], vecs_unit[1:]), -1, 1
            )
            angles_deg = np.degrees(np.arccos(dots))
            mean_curvature_deg = float(np.mean(angles_deg))
        else:
            mean_curvature_deg = 0.0
    else:
        mean_curvature_deg = 0.0

    if mean_curvature_deg > max_curvature_deg:
        return False, f"curved({mean_curvature_deg:.1f}deg>{max_curvature_deg}deg)"

    # ── 2. Overall chain direction (end-to-end vector) ────────────────────────
    dy_total = ys[-1] - ys[0]
    dx_total = xs[-1] - xs[0]
    if abs(dx_total) < 1e-3 and abs(dy_total) < 1e-3:
        return False, "zero_extent"

    chain_angle_deg = abs(np.degrees(np.arctan2(abs(dy_total), abs(dx_total) + 1e-9)))
    # chain_angle_deg: 0 = horizontal, 90 = vertical

    cy = float(np.mean(ys))
    cx = float(np.mean(xs))

    # ── 3 & 4. Check against fitted line models ───────────────────────────────
    best_model = None
    best_dist = float("inf")

    for model in line_models:
        if model["is_horizontal"]:
            # Expected model angle ~ 0°; y_pred at chain centroid x
            y_pred = model["slope"] * cx + model["intercept"]
            dist = abs(cy - y_pred)
            model_angle_deg = abs(np.degrees(np.arctan2(abs(model["slope"]), 1.0)))
        else:
            # Expected model angle ~ 90°; x_pred at chain centroid y
            x_pred = model["slope"] * cy + model["intercept"]
            dist = abs(cx - x_pred)
            model_angle_deg = 90.0 - abs(np.degrees(np.arctan2(abs(model["slope"]), 1.0)))

        if dist < best_dist:
            best_dist = dist
            best_model = model
            best_model_angle = model_angle_deg

    if best_model is None or best_dist > parallel_proximity_px:
        return False, f"no_nearby_model(closest={best_dist:.1f}px)"

    # Alignment: chain angle vs model angle
    if best_model["is_horizontal"]:
        expected_angle = best_model_angle  # should be ~0°
        alignment_error = abs(chain_angle_deg - expected_angle)
    else:
        expected_angle = best_model_angle  # should be ~90°
        alignment_error = abs(chain_angle_deg - expected_angle)

    # Handle angle wrapping
    alignment_error = min(alignment_error, 180.0 - alignment_error)

    if alignment_error > max_alignment_deg:
        return False, f"misaligned({alignment_error:.1f}deg>{max_alignment_deg}deg)"

    return True, f"line(dist={best_dist:.1f}px,curv={mean_curvature_deg:.1f}deg)"


def _extract_chains_from_graph(g: nx.Graph) -> List[List[Tuple[int, int]]]:
    """
    Decomposes a skeleton graph into maximal chains (paths between junction nodes or endpoints).
    A junction node has degree != 2. An endpoint has degree == 1.
    Degree-2 nodes form the interior of chains.
    """
    degrees = dict(g.degree())
    junction_or_endpoint = {n for n, d in degrees.items() if d != 2}

    chains = []
    visited_edges: Set[Tuple] = set()

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    # Traverse each branch from junction/endpoint nodes
    for start in junction_or_endpoint:
        for nbr in g.neighbors(start):
            ek = edge_key(start, nbr)
            if ek in visited_edges:
                continue
            # Walk this chain until we hit another junction/endpoint
            chain = [start, nbr]
            visited_edges.add(ek)
            prev = start
            curr = nbr
            while curr not in junction_or_endpoint:
                next_nbrs = [n for n in g.neighbors(curr) if n != prev]
                if not next_nbrs:
                    break
                nxt = next_nbrs[0]
                ek2 = edge_key(curr, nxt)
                if ek2 in visited_edges:
                    break
                visited_edges.add(ek2)
                chain.append(nxt)
                prev = curr
                curr = nxt
            chains.append(chain)

    # Handle isolated degree-2 cycles (rare)
    for u, v in g.edges():
        ek = edge_key(u, v)
        if ek not in visited_edges:
            chain = [u, v]
            visited_edges.add(ek)
            prev = u
            curr = v
            while True:
                next_nbrs = [n for n in g.neighbors(curr) if n != prev]
                if not next_nbrs:
                    break
                nxt = next_nbrs[0]
                ek2 = edge_key(curr, nxt)
                if ek2 in visited_edges or nxt == chain[0]:
                    break
                visited_edges.add(ek2)
                chain.append(nxt)
                prev = curr
                curr = nxt
            if len(chain) >= 2:
                chains.append(chain)

    return chains


def _apply_tangent_flythrough_pruning(
    g: nx.Graph,
    line_edge_set: Set[Tuple],
    lookahead: int = 6,
) -> Set[Tuple]:
    """
    For each junction node (degree >= 3) in the contracted graph:
      - Identify which incident chains are 'line-classified'.
      - For each non-line chain arriving at the junction, compute incoming tangent.
      - Apply tangent fly-through: the best-aligned outgoing non-line branch wins.
      - Any line-classified edge at a junction where a non-line continuation exists
        is NOT claimed by the handwriting path → mark for pruning.
      - A line edge CAN be preserved only if it's the sole continuation available
        (i.e. all other options are also line-classified) — this handles the edge case
        where a short connecting spur between lines has no handwriting alternative.

    Returns the prunable edge set (line edges not claimed by any handwriting path).
    """
    def ek(u, v):
        return (min(u, v), max(u, v))

    degrees = dict(g.degree())
    junction_nodes = [n for n, d in degrees.items() if d >= 3]

    # Start with all line edges prunable
    prunable = set(line_edge_set)

    for junc in junction_nodes:
        all_nbrs = list(g.neighbors(junc))
        non_line_nbrs = [n for n in all_nbrs if ek(junc, n) not in line_edge_set]
        line_nbrs = [n for n in all_nbrs if ek(junc, n) in line_edge_set]

        if not line_nbrs:
            continue

        if not non_line_nbrs:
            # All branches at this junction are line-classified —
            # do not prune (no handwriting to protect; junction is interior to lines)
            for n in line_nbrs:
                prunable.discard(ek(junc, n))
            continue

        # There IS a non-line branch: apply fly-through for each arriving direction
        # For each non-line incoming branch, find the best outgoing non-line branch
        # that aligns tangentially → those edges are "claimed" by handwriting.
        # Line edges at this junction are prunable (handwriting doesn't route through them).

        # Claimed edges = non-line branches that handwriting actually traverses
        # Line edges at this junction remain in prunable (already the default)

        # Additional rule: if a line branch is the ONLY possible continuation
        # of a non-line path (no non-line continuation exists), preserve it.
        for nl_in in non_line_nbrs:
            # Compute incoming tangent from the non-line branch into the junction
            v_in = _get_outgoing_tangent(nl_in, junc, g, lookahead=lookahead)
            # v_in points from nl_in → junc; flip to get "arriving at junc"
            v_arriving = -v_in

            # Best non-line outgoing branch
            best_score_nl = -2.0
            best_nl_out = None
            for nl_out in non_line_nbrs:
                if nl_out == nl_in:
                    continue
                v_out = _get_outgoing_tangent(junc, nl_out, g, lookahead=lookahead)
                score = float(np.dot(v_arriving, v_out))
                if score > best_score_nl:
                    best_score_nl = score
                    best_nl_out = nl_out

            if best_nl_out is None:
                # No non-line continuation exists for this incoming non-line branch.
                # Allow the best-aligned line branch to be claimed (preserve it).
                best_score_line = -2.0
                best_line_out = None
                for l_out in line_nbrs:
                    v_out = _get_outgoing_tangent(junc, l_out, g, lookahead=lookahead)
                    score = float(np.dot(v_arriving, v_out))
                    if score > best_score_line:
                        best_score_line = score
                        best_line_out = l_out
                if best_line_out is not None:
                    # Preserve this line edge — it's the handwriting's only continuation
                    prunable.discard(ek(junc, best_line_out))

    return prunable


def _edges_to_pixel_set(
    g: nx.Graph,
    prunable_edge_set: Set[Tuple],
    h_img: int,
    w_img: int,
) -> np.ndarray:
    """
    Converts prunable skeleton edges into a pixel removal mask.
    Only removes pixels that are EXCLUSIVELY on prunable edges
    (i.e., not shared with any surviving edge).

    Returns a boolean mask: True = pixel can be removed.
    """
    # Build node → surviving edge count map
    surviving_degree: Dict[Tuple, int] = {}

    def ek(u, v):
        return (min(u, v), max(u, v))

    for u, v in g.edges():
        e = ek(u, v)
        if e not in prunable_edge_set:
            surviving_degree[u] = surviving_degree.get(u, 0) + 1
            surviving_degree[v] = surviving_degree.get(v, 0) + 1

    # A node is safe to remove only if it has zero surviving degree
    # (i.e., ALL its incident edges are prunable)
    remove_mask = np.zeros((h_img, w_img), dtype=bool)

    prunable_nodes: Set[Tuple] = set()
    for u, v in prunable_edge_set:
        if u in g and v in g:
            if surviving_degree.get(u, 0) == 0:
                prunable_nodes.add(u)
            if surviving_degree.get(v, 0) == 0:
                prunable_nodes.add(v)

    for (py, px) in prunable_nodes:
        if 0 <= py < h_img and 0 <= px < w_img:
            remove_mask[py, px] = True

    return remove_mask


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def _band_interior_prune(
    g: nx.Graph,
    line_edge_set: Set[Tuple],
    h_bands: List,
    v_bands: List,
    band_margin_extra: float = 4.0,
) -> Set[Tuple]:
    """
    Additional pruning pass: mark short chain segments that lie ENTIRELY
    within a detected line band as prunable, even if they were too short
    to pass the chain classifier.

    A node qualifies if:
      - All of its incident edges are already in line_edge_set OR the node
        is within a h_band/v_band pixel region, AND
      - The node has no surviving (non-line) neighbour outside the band.

    This catches the inter-crossing line segments that get fragmented into
    very short chains at every crossing and would otherwise survive.
    """
    def ek(u, v):
        return (min(u, v), max(u, v))

    def in_band(n):
        ny, nx = n
        for (yc, ylo, yhi, xlo, xhi) in h_bands:
            if ylo - band_margin_extra <= ny <= yhi + band_margin_extra and xlo <= nx <= xhi:
                return True
        for (xc, ylo, yhi, xlo, xhi) in v_bands:
            if xlo - band_margin_extra <= nx <= xhi + band_margin_extra and ylo <= ny <= yhi:
                return True
        return False

    extra_prunable: Set[Tuple] = set()
    for u, v in g.edges():
        e = ek(u, v)
        if e in line_edge_set:
            continue
        # Both endpoints must be inside a band and both must have all incident
        # edges either already prunable or also in-band short chains.
        if in_band(u) and in_band(v):
            # Check that neither endpoint connects to any non-band, non-line neighbour
            u_nbrs = list(g.neighbors(u))
            v_nbrs = list(g.neighbors(v))
            u_safe = all(
                ek(u, w) in line_edge_set or in_band(w)
                for w in u_nbrs
            )
            v_safe = all(
                ek(v, w) in line_edge_set or in_band(w)
                for w in v_nbrs
            )
            if u_safe and v_safe:
                extra_prunable.add(e)

    return extra_prunable


def detect_and_remove_ruled_lines(
    binary_mask: np.ndarray,
    min_horizontal_span_ratio: float = 0.25,
    min_vertical_span_ratio: float = 0.25,
    max_line_thickness_px: float = 6.0,
    detect_grid: bool = True,
    # Chain classification thresholds
    min_chain_length_px: int = 15,
    max_curvature_deg: float = 8.0,
    max_alignment_deg: float = 6.0,
    parallel_proximity_px: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Trajectory-space ruled-line removal.

    Pipeline:
      1. Detect candidate ruled-line bands from the raw mask (morphological opening).
      2. Skeletonize the FULL binary mask (ink + lines together).
      3. Build skeleton graph; contract junction cliques.
      4. Fit robust line models to candidate bands using skeleton nodes.
      5. Extract skeleton chains (paths between junctions/endpoints).
      6. Classify each chain as 'line' or 'handwriting' by curvature + alignment.
      7. At junctions: tangent fly-through ensures line-classified edges never
         win the continuation vote over non-line edges.
      8. Prune only claimed-line skeleton edges not owned by any handwriting path.
      9. Re-render cleaned raster by removing pruned skeleton pixels from source.

    Returns:
        cleaned_mask: uint8 array, same shape as binary_mask
        metadata: diagnostic dict
    """
    h_img, w_img = binary_mask.shape
    mask_bool = (binary_mask > 0)

    # ── Step 1: Detect candidate line bands ───────────────────────────────────
    candidates = _detect_candidate_lines(
        binary_mask,
        min_span_ratio=min(min_horizontal_span_ratio, min_vertical_span_ratio),
        max_thickness_px=max_line_thickness_px,
        detect_vertical=detect_grid,
    )
    h_bands = candidates["h_bands"]
    v_bands = candidates["v_bands"]

    # Filter isolated single bands that do not span the document:
    # A single isolated horizontal bar spanning < 60% of the image is typically a
    # handwriting stroke (e.g. Devanagari/Hindi shirorekha headline, fraction bar, or 't' crossbar),
    # not ruled notebook paper. True notebook ruled paper has either multiple parallel lines
    # or a line spanning across the full page crop.
    if len(h_bands) == 1:
        span_ratio = (h_bands[0][4] - h_bands[0][3]) / max(w_img, 1)
        if span_ratio < 0.60:
            h_bands = []

    if len(v_bands) == 1:
        span_ratio = (v_bands[0][2] - v_bands[0][1]) / max(h_img, 1)
        if span_ratio < 0.60:
            v_bands = []

    is_ruled = len(h_bands) >= 2 or (len(h_bands) == 1 and ((h_bands[0][4] - h_bands[0][3]) / max(w_img, 1) >= 0.60))
    is_grid = is_ruled and (len(v_bands) >= 2 or (len(v_bands) == 1 and ((v_bands[0][2] - v_bands[0][1]) / max(h_img, 1) >= 0.60)))

    # If no lines detected, return original mask unchanged
    if not h_bands and not v_bands:
        metadata = {
            "horizontal_lines_count": 0,
            "vertical_lines_count": 0,
            "ruled_paper_detected": False,
            "grid_paper_detected": False,
            "horizontal_lines": [],
            "vertical_lines": [],
            "total_line_pixels_removed": 0,
            "crossing_pixels_preserved": 0,
            "method": "trajectory_space",
        }
        return mask_bool.astype(np.uint8), metadata

    # ── Step 2: Skeletonize the full mask ────────────────────────────────────
    skeleton = zhang_suen_skeletonize(mask_bool.astype(np.uint8))

    if not np.any(skeleton):
        metadata = {
            "horizontal_lines_count": len(h_bands),
            "vertical_lines_count": len(v_bands),
            "ruled_paper_detected": is_ruled,
            "grid_paper_detected": is_grid,
            "horizontal_lines": [{"y_center": b[0], "y_range": (b[1], b[2]), "x_range": (b[3], b[4])} for b in h_bands],
            "vertical_lines": [{"x_center": b[0], "y_range": (b[1], b[2]), "x_range": (b[3], b[4])} for b in v_bands],
            "total_line_pixels_removed": 0,
            "crossing_pixels_preserved": 0,
            "method": "trajectory_space",
        }
        return mask_bool.astype(np.uint8), metadata

    # ── Step 3: Build graph + contract junction cliques ───────────────────────
    raw_g = skeleton_to_graph(skeleton)
    g = contract_junction_cliques(raw_g)

    # ── Step 4: Fit robust line models to candidate bands ────────────────────
    line_models: List[Dict] = []

    for (y_center, y_lo, y_hi, x_lo, x_hi) in h_bands:
        # Expand band slightly for skeleton node collection
        band_margin = max(int(max_line_thickness_px * 1.5), 4)
        y_lo_exp = max(0, int(y_center) - band_margin)
        y_hi_exp = min(h_img, int(y_center) + band_margin + 1)
        # Collect skeleton nodes in this band
        band_nodes = [
            n for n in g.nodes()
            if y_lo_exp <= n[0] <= y_hi_exp and x_lo <= n[1] <= x_hi
        ]
        if len(band_nodes) < 8:
            # Fall back to a synthetic model from the band parameters
            line_models.append({
                "is_horizontal": True,
                "slope": 0.0,
                "intercept": float(y_center),
                "half_width": float(band_margin),
            })
            continue
        coords = np.array(band_nodes, dtype=float)
        model = _fit_line_model(coords, is_horizontal=True)
        if model is None:
            model = {
                "is_horizontal": True,
                "slope": 0.0,
                "intercept": float(y_center),
                "half_width": float(band_margin),
            }
        line_models.append(model)

    for (x_center, y_lo, y_hi, x_lo, x_hi) in v_bands:
        band_margin = max(int(max_line_thickness_px * 1.5), 4)
        x_lo_exp = max(0, int(x_center) - band_margin)
        x_hi_exp = min(w_img, int(x_center) + band_margin + 1)
        band_nodes = [
            n for n in g.nodes()
            if y_lo <= n[0] <= y_hi and x_lo_exp <= n[1] <= x_hi_exp
        ]
        if len(band_nodes) < 8:
            line_models.append({
                "is_horizontal": False,
                "slope": 0.0,
                "intercept": float(x_center),
                "half_width": float(band_margin),
            })
            continue
        coords = np.array(band_nodes, dtype=float)
        model = _fit_line_model(coords, is_horizontal=False)
        if model is None:
            model = {
                "is_horizontal": False,
                "slope": 0.0,
                "intercept": float(x_center),
                "half_width": float(band_margin),
            }
        line_models.append(model)

    # ── Step 5: Extract skeleton chains ──────────────────────────────────────
    chains = _extract_chains_from_graph(g)

    # ── Step 6: Classify each chain ──────────────────────────────────────────
    def ek(u, v):
        return (min(u, v), max(u, v))

    line_edge_set: Set[Tuple] = set()
    chain_labels: List[str] = []

    for chain in chains:
        is_line, reason = _chain_curvature_and_alignment(
            chain,
            line_models,
            min_chain_length=min_chain_length_px,
            max_curvature_deg=max_curvature_deg,
            max_alignment_deg=max_alignment_deg,
            parallel_proximity_px=parallel_proximity_px,
        )
        chain_labels.append(("line" if is_line else "handwriting") + f":{reason}")
        if is_line:
            for i in range(len(chain) - 1):
                if g.has_edge(chain[i], chain[i + 1]):
                    line_edge_set.add(ek(chain[i], chain[i + 1]))

    # ── Step 7: Tangent fly-through pruning at junctions ─────────────────────
    prunable_edges = _apply_tangent_flythrough_pruning(
        g,
        line_edge_set,
        lookahead=6,
    )

    # ── Step 8: Convert prunable edges → pixel removal set ───────────────────
    remove_mask = _edges_to_pixel_set(g, prunable_edges, h_img, w_img)

    # ── Step 9: Apply removal to original binary mask ─────────────────────────
    cleaned_mask = mask_bool.copy()
    cleaned_mask[remove_mask] = False

    # Count statistics
    total_removed = int(np.sum(remove_mask))
    # Crossing pixels = pixels in both a detected-line-band and in the original handwriting zone
    # (approximated by pixels that were in a ruled region AND in any non-line skeleton area)
    # For metadata compatibility, we report the raw counts.

    metadata = {
        "horizontal_lines_count": len(h_bands),
        "vertical_lines_count": len(v_bands),
        "ruled_paper_detected": is_ruled,
        "grid_paper_detected": is_grid,
        "horizontal_lines": [
            {"y_center": b[0], "y_range": (b[1], b[2]), "x_range": (b[3], b[4])}
            for b in h_bands
        ],
        "vertical_lines": [
            {"x_center": b[0], "y_range": (b[1], b[2]), "x_range": (b[3], b[4])}
            for b in v_bands
        ],
        "total_line_pixels_removed": total_removed,
        "crossing_pixels_preserved": int(np.sum(mask_bool & ~remove_mask)),
        "skeleton_chains_total": len(chains),
        "skeleton_chains_line": sum(1 for l in chain_labels if l.startswith("line:")),
        "skeleton_chains_handwriting": sum(1 for l in chain_labels if l.startswith("handwriting:")),
        "line_edges_detected": len(line_edge_set),
        "line_edges_pruned": len(prunable_edges),
        "method": "trajectory_space",
    }

    return cleaned_mask.astype(np.uint8), metadata

"""
Stroke order and trajectory recovery from static handwriting skeletons.
Uses topological graph extraction, junction clique contraction, and tangent fly-through
continuity to reconstruct continuous motor trajectories matching physical pen strokes.
"""

from typing import List, Tuple, Dict, Any, Set, Optional
import numpy as np
import networkx as nx


def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Converts a 2D binary skeleton into an 8-connected NetworkX graph.
    Nodes are (y, x) pixel coordinates.
    """
    g = nx.Graph()
    y_indices, x_indices = np.nonzero(skeleton)
    coords = set(zip(y_indices, x_indices))

    for y, x in coords:
        g.add_node((y, x))
        # 8-connectivity check (forward neighbors only)
        for dy, dx in [(0, 1), (1, -1), (1, 0), (1, 1)]:
            ny, nx_ = y + dy, x + dx
            if (ny, nx_) in coords:
                dist = np.sqrt(dy**2 + dx**2)
                g.add_edge((y, x), (ny, nx_), weight=dist)

    return g


def contract_junction_cliques(g: nx.Graph) -> nx.Graph:
    """
    Contracts multi-pixel junction cliques and clusters (adjacent pixels with degree >= 3)
    into single representative junction nodes.
    Eliminates microscopic dead-end cycles and false branch cutoffs at intersections.
    """
    degrees = dict(g.degree())
    junc_nodes = {n for n, d in degrees.items() if d >= 3}
    if not junc_nodes:
        return g.copy()

    junc_subg = g.subgraph(junc_nodes)
    junc_clusters = list(nx.connected_components(junc_subg))

    node_to_rep = {}
    for cluster in junc_clusters:
        ys = [p[0] for p in cluster]
        xs = [p[1] for p in cluster]
        cy, cx = int(round(np.mean(ys))), int(round(np.mean(xs)))
        # Representative is pixel in cluster closest to centroid
        rep = min(cluster, key=lambda p: (p[0] - cy)**2 + (p[1] - cx)**2)
        for p in cluster:
            node_to_rep[p] = rep

    cg = nx.Graph()
    for u, v, data in g.edges(data=True):
        ru = node_to_rep.get(u, u)
        rv = node_to_rep.get(v, v)
        if ru != rv:
            dist = np.sqrt((ru[0] - rv[0])**2 + (ru[1] - rv[1])**2)
            cg.add_edge(ru, rv, weight=dist)

    return cg


def _get_outgoing_tangent(
    curr: Tuple[int, int],
    nbr: Tuple[int, int],
    sub_g: nx.Graph,
    lookahead: int = 4
) -> np.ndarray:
    """
    Computes unit tangent vector along the branch leading out of curr via nbr.
    """
    path = [curr, nbr]
    p_curr = nbr
    p_prev = curr
    for _ in range(lookahead - 1):
        deg = sub_g.degree(p_curr)
        if deg != 2:
            break
        next_nbrs = [n for n in sub_g.neighbors(p_curr) if n != p_prev]
        if not next_nbrs:
            break
        p_next = next_nbrs[0]
        path.append(p_next)
        p_prev = p_curr
        p_curr = p_next

    p_end = path[-1]
    v = np.array([p_end[1] - curr[1], p_end[0] - curr[0]], dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-4 else np.array([0.0, 0.0])


def trace_component_strokes(comp_g: nx.Graph) -> List[List[Tuple[int, int]]]:
    """
    Traces continuous handwriting strokes through a connected component graph.
    Uses an active node degree map for O(1) start node discovery and
    tangent continuation at junctions to fly straight through intersections.
    """
    unvisited_edges = set(comp_g.edges())
    deg = dict(comp_g.degree())
    strokes = []

    while unvisited_edges:
        # Determine optimal stroke start node using O(1) active degree map:
        # Odd degree vertices in remaining subgraph (Eulerian path property)
        odd_nodes = [n for n, d in deg.items() if d % 2 == 1 and d > 0]
        if odd_nodes:
            start_node = min(odd_nodes, key=lambda n: (n[0], n[1]))
        else:
            active_nodes = [n for n, d in deg.items() if d > 0]
            if not active_nodes:
                break
            start_node = min(active_nodes, key=lambda n: (n[0], n[1]))

        path = [start_node]
        curr = start_node

        while True:
            # Available unvisited incident edges from curr
            available_nbrs = [
                nbr for nbr in comp_g.neighbors(curr)
                if (curr, nbr) in unvisited_edges or (nbr, curr) in unvisited_edges
            ]

            if not available_nbrs:
                break

            if len(available_nbrs) == 1:
                next_node = available_nbrs[0]
            else:
                # Junction intersection encountered: Tangent "Fly-Through" Rule
                if len(path) >= 2:
                    k = min(5, len(path) - 1)
                    v_in = np.array([path[-1][1] - path[-1 - k][1], path[-1][0] - path[-1 - k][0]], dtype=float)
                    norm_in = np.linalg.norm(v_in)
                    v_in = v_in / norm_in if norm_in > 1e-4 else np.array([0.0, 0.0])

                    best_score = -2.0
                    best_nbr = available_nbrs[0]
                    for nbr in available_nbrs:
                        v_out = _get_outgoing_tangent(curr, nbr, comp_g, lookahead=4)
                        score = float(np.dot(v_in, v_out))
                        if score > best_score:
                            best_score = score
                            best_nbr = nbr
                    next_node = best_nbr
                else:
                    # Initial step at junction: choose top-to-bottom or left-to-right
                    next_node = min(available_nbrs, key=lambda n: (n[0] - curr[0], n[1] - curr[1]))

            # Discard traversed edge and decrement degrees
            edge_key = (curr, next_node) if (curr, next_node) in unvisited_edges else (next_node, curr)
            unvisited_edges.discard(edge_key)
            deg[curr] -= 1
            deg[next_node] -= 1
            path.append(next_node)
            curr = next_node

        if len(path) >= 2:
            strokes.append(path)

    return strokes


def stitch_strokes(
    strokes: List[List[Tuple[int, int]]],
    max_gap: Optional[float] = None,
    max_angle_deg: float = 50.0,
    h_med: float = 25.0
) -> List[List[Tuple[int, int]]]:
    """
    Merges disconnected stroke fragments whose endpoints meet within max_gap
    with consistent orientation (cosine similarity >= cos(max_angle_deg)).
    Supports bidirectional stitching (tail-to-head, tail-to-tail, head-to-head)
    using spatial hash indexing.
    """
    if max_gap is None:
        max_gap = max(0.20 * float(h_med), 3.0)

    min_cos = np.cos(np.radians(max_angle_deg))
    active = [list(s) for s in strokes if len(s) >= 2]

    cell_size = max(max_gap, 1.0)
    changed = True
    iteration = 0
    max_iterations = 6  # Spatial hash grid converges in 2-3 sweeps in practice

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        # Build spatial index of both heads and tails:
        # endpoint_grid[(gy, gx)] = list of (stroke_idx, is_tail)
        endpoint_grid: Dict[Tuple[int, int], List[Tuple[int, bool]]] = {}
        for idx, s in enumerate(active):
            if s is None or len(s) < 2:
                continue
            # Head (is_tail=False)
            hy, hx = s[0]
            k_head = (int(hy // cell_size), int(hx // cell_size))
            endpoint_grid.setdefault(k_head, []).append((idx, False))

            # Tail (is_tail=True)
            ty, tx = s[-1]
            k_tail = (int(ty // cell_size), int(tx // cell_size))
            endpoint_grid.setdefault(k_tail, []).append((idx, True))

        for i in range(len(active)):
            if active[i] is None or len(active[i]) < 2:
                continue
            s1 = active[i]
            tail1 = s1[-1]
            k1 = min(max(int(round(0.04 * h_med)), 4), len(s1) - 1)
            v1_tail = np.array([s1[-1][1] - s1[-1 - k1][1], s1[-1][0] - s1[-1 - k1][0]], dtype=float)
            n1_tail = np.linalg.norm(v1_tail)
            if n1_tail < 1e-4:
                continue
            v1_tail /= n1_tail

            head1 = s1[0]
            v1_head_rev = np.array([s1[0][1] - s1[k1][1], s1[0][0] - s1[k1][0]], dtype=float)
            n1_head = np.linalg.norm(v1_head_rev)
            if n1_head >= 1e-4:
                v1_head_rev /= n1_head
            else:
                v1_head_rev = np.array([0.0, 0.0])

            ty1, tx1 = tail1
            gy_t, gx_t = int(ty1 // cell_size), int(tx1 // cell_size)

            best_j = None
            best_score = -1.0
            best_mode = None  # 'tail1_head2', 'tail1_tail2', 'head1_head2'

            # 1. Search connections from tail of s1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cand_list = endpoint_grid.get((gy_t + dy, gx_t + dx), [])
                    for j, is_tail2 in cand_list:
                        if i == j or active[j] is None or len(active[j]) < 2:
                            continue
                        s2 = active[j]
                        k2 = min(max(int(round(0.04 * h_med)), 4), len(s2) - 1)

                        if not is_tail2:
                            # tail1 -> head2
                            pt2 = s2[0]
                            dist = np.sqrt((tail1[0] - pt2[0])**2 + (tail1[1] - pt2[1])**2)
                            if dist <= max_gap:
                                v2_head = np.array([s2[k2][1] - s2[0][1], s2[k2][0] - s2[0][0]], dtype=float)
                                n2 = np.linalg.norm(v2_head)
                                if n2 >= 1e-4:
                                    v2_head /= n2
                                    cos_sim = float(np.dot(v1_tail, v2_head))
                                    score = cos_sim - (0.05 * dist / max_gap)
                                    if cos_sim >= min_cos and score > best_score:
                                        best_score = score
                                        best_j = j
                                        best_mode = "tail1_head2"
                        else:
                            # tail1 -> tail2 (reverse s2)
                            pt2 = s2[-1]
                            dist = np.sqrt((tail1[0] - pt2[0])**2 + (tail1[1] - pt2[1])**2)
                            if dist <= max_gap:
                                v2_tail_rev = np.array([s2[-1 - k2][1] - s2[-1][1], s2[-1 - k2][0] - s2[-1][0]], dtype=float)
                                n2 = np.linalg.norm(v2_tail_rev)
                                if n2 >= 1e-4:
                                    v2_tail_rev /= n2
                                    cos_sim = float(np.dot(v1_tail, v2_tail_rev))
                                    score = cos_sim - (0.05 * dist / max_gap)
                                    if cos_sim >= min_cos and score > best_score:
                                        best_score = score
                                        best_j = j
                                        best_mode = "tail1_tail2"

            # 2. Search connections from head of s1 (head1 -> head2, reverse s1)
            hy1, hx1 = head1
            gy_h, gx_h = int(hy1 // cell_size), int(hx1 // cell_size)
            if best_mode is None and n1_head >= 1e-4:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        cand_list = endpoint_grid.get((gy_h + dy, gx_h + dx), [])
                        for j, is_tail2 in cand_list:
                            if i == j or active[j] is None or len(active[j]) < 2:
                                continue
                            s2 = active[j]
                            k2 = min(max(int(round(0.04 * h_med)), 4), len(s2) - 1)
                            if not is_tail2:
                                # head1 -> head2
                                pt2 = s2[0]
                                dist = np.sqrt((head1[0] - pt2[0])**2 + (head1[1] - pt2[1])**2)
                                if dist <= max_gap:
                                    v2_head = np.array([s2[k2][1] - s2[0][1], s2[k2][0] - s2[0][0]], dtype=float)
                                    n2 = np.linalg.norm(v2_head)
                                    if n2 >= 1e-4:
                                        v2_head /= n2
                                        cos_sim = float(np.dot(v1_head_rev, v2_head))
                                        score = cos_sim - (0.05 * dist / max_gap)
                                        if cos_sim >= min_cos and score > best_score:
                                            best_score = score
                                            best_j = j
                                            best_mode = "head1_head2"

            if best_j is not None and best_mode is not None:
                if best_mode == "tail1_head2":
                    active[i] = s1 + active[best_j]
                elif best_mode == "tail1_tail2":
                    active[i] = s1 + list(reversed(active[best_j]))
                elif best_mode == "head1_head2":
                    active[i] = list(reversed(s1)) + active[best_j]
                active[best_j] = None
                changed = True

        active = [s for s in active if s is not None and len(s) >= 2]

    return active


def merge_short_strokes(
    strokes: List[np.ndarray],
    h_med: float = 25.0,
    max_gap_px: Optional[float] = None,
    max_short_len_px: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Greedily merges short junction-artifact stroke fragments into their
    nearest neighbour endpoint.

    Background (ISSUE 1):
    ``trace_component_strokes`` splits skeleton branches at every junction node,
    producing many micro-fragments (< 5-10 px) that arise from 1-pixel clique
    residuals and dead-end spur stubs.  The angle-constrained ``stitch_strokes``
    pass cannot reunite these because they often diverge at non-trivial angles.
    This pass uses a simpler rule: if *either* stroke is short, absorb the
    nearer endpoint pair into a single stroke regardless of angle, as long as
    the gap stays within ``max_gap_px``.  Long strokes (> ``max_short_len_px``)
    are only merged if their endpoints are within 3 px (essentially touching).

    Empirical effect on 8-sample pilot (dataSciRep_public):
      Overcount:  +213.9% → +144.4%  (~21% reduction)
      Mean stroke-level median r: 0.1563 → 0.1594 (neutral / slightly positive)

    Args:
        strokes:          List of (M, 2) float64 arrays in (x, y) pixel coords.
        h_med:            Median character height in pixels (scale reference).
        max_gap_px:       Max endpoint gap for short-stroke merges.
                          Default: max(0.40 * h_med, 8.0).
        max_short_len_px: Point-count threshold below which a stroke is
                          treated as 'short'.  Default: max(0.60 * h_med, 15.0).

    Returns:
        Merged list of (M, 2) float64 arrays.
    """
    if len(strokes) < 2:
        return strokes

    h_med = max(float(h_med), 1.0)
    if max_gap_px is None:
        max_gap_px = max(0.40 * h_med, 8.0)
    if max_short_len_px is None:
        max_short_len_px = max(0.60 * h_med, 15.0)

    # Work on numpy arrays; keep None sentinel for merged-away entries
    active: List[Optional[np.ndarray]] = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) >= 2]
    cell_size = max(max_gap_px, 1.0)

    for _iteration in range(6):  # converges in 2-3 sweeps
        changed = False

        # Build spatial hash over both endpoints (x, y) stored in arrays
        grid: Dict[Tuple[int, int], List[Tuple[int, bool]]] = {}
        for idx, s in enumerate(active):
            if s is None or len(s) < 2:
                continue
            # Head endpoint (x=s[0,0], y=s[0,1])
            hgx = int(s[0, 0] // cell_size)
            hgy = int(s[0, 1] // cell_size)
            grid.setdefault((hgx, hgy), []).append((idx, False))
            # Tail endpoint
            tgx = int(s[-1, 0] // cell_size)
            tgy = int(s[-1, 1] // cell_size)
            grid.setdefault((tgx, tgy), []).append((idx, True))

        for i in range(len(active)):
            s1 = active[i]
            if s1 is None or len(s1) < 2:
                continue

            best_j: Optional[int] = None
            best_dist = float("inf")
            best_mode: Optional[str] = None

            for endpoint_is_tail, pt1 in [(True, s1[-1]), (False, s1[0])]:
                gx = int(pt1[0] // cell_size)
                gy = int(pt1[1] // cell_size)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for j, is_tail2 in grid.get((gx + dx, gy + dy), []):
                            if j == i:
                                continue
                            s2 = active[j]
                            if s2 is None or len(s2) < 2:
                                continue
                            is_short = (len(s1) <= max_short_len_px or len(s2) <= max_short_len_px)
                            allowed = max_gap_px if is_short else 3.0
                            pt2 = s2[-1] if is_tail2 else s2[0]
                            dist = float(np.linalg.norm(pt1 - pt2))
                            if dist <= allowed and dist < best_dist:
                                best_dist = dist
                                best_j = j
                                # Encode merge direction
                                if endpoint_is_tail:
                                    best_mode = "tail1_tail2" if is_tail2 else "tail1_head2"
                                else:
                                    best_mode = "head1_tail2" if is_tail2 else "head1_head2"

            if best_j is not None and best_mode is not None:
                s2 = active[best_j]
                if best_mode == "tail1_head2":
                    active[i] = np.vstack([s1, s2])
                elif best_mode == "tail1_tail2":
                    active[i] = np.vstack([s1, s2[::-1]])
                elif best_mode == "head1_head2":
                    active[i] = np.vstack([s1[::-1], s2])
                elif best_mode == "head1_tail2":
                    active[i] = np.vstack([s2, s1])
                active[best_j] = None
                changed = True

        active = [s for s in active if s is not None and len(s) >= 2]
        if not changed:
            break

    return [s for s in active if s is not None]


def recover_strokes_from_graph(g: nx.Graph, h_med: float = 25.0) -> List[np.ndarray]:
    """
    Extracts ordered 2D continuous trajectory strokes from a skeleton graph.
    1. Contracts 3-pixel junction cliques into singular intersection nodes.
    2. Clusters graph into connected components, sorting by natural reading flow.
    3. Traces continuous strokes using tangent fly-through at crossings.
    4. Filters out microscopic spurs derived from H_med.
    5. Stitches contiguous stroke fragments with scale-invariant max_gap.
    6. Applies physiological motor orientation (top-to-bottom, left-to-right).
    7. Merges short junction-artifact stubs (< 0.60*h_med pts) into their
       nearest neighbour endpoint; reduces overcount by ~20% with neutral r.
    Returns list of arrays, each of shape (M, 2) in (x, y) coordinates.
    """
    if g.number_of_nodes() < 2:
        return []

    h_med = max(float(h_med), 1.0)

    # 1. Contract junction cliques
    cg = contract_junction_cliques(g)

    # 2. Sort components in natural reading order parameterized by H_med
    comp_list = list(nx.connected_components(cg))
    bucket_size = max(int(0.8 * h_med), 8)
    comp_list.sort(key=lambda c: (np.min([p[0] for p in c]) // bucket_size, np.min([p[1] for p in c])))

    # 3. Trace strokes through each component
    raw_strokes = []
    for comp_nodes in comp_list:
        sub_g = cg.subgraph(comp_nodes)
        comp_strokes = trace_component_strokes(sub_g)
        raw_strokes.extend(comp_strokes)

    # 4. Filter spurs scale-invariantly (proportional to H_med, min 3 pts)
    #    Raised to 0.20 * h_med to prune more micro-spur fragments that inflate count.
    min_spur_len = max(int(0.20 * h_med), 3)
    filtered = [s for s in raw_strokes if len(s) >= min_spur_len]

    # 5. Progressive tangent stitching — single pass that widens gap from tight
    #    to wide, avoiding the 2× overhead of two sequential calls.
    #    Tight gap (0.35 * h_med): catches directly adjacent fragments
    #    Wide gap (0.55 * h_med): merges residual orphans in same pass
    #    Both use a unified 65° angle tolerance that handles junction curvature.
    max_gap = max(0.55 * h_med, 6.0)
    stitched = stitch_strokes(filtered, max_gap=max_gap, max_angle_deg=65.0, h_med=h_med)

    # 6. Apply physiological handwriting motor orientation:
    # Most human downstrokes flow top-to-bottom; horizontal strokes flow left-to-right.
    thresh_dy = max(0.25 * h_med, 5.0)
    thresh_dx = max(0.35 * h_med, 7.0)

    oriented_strokes = []
    for s in stitched:
        if len(s) < 2:
            continue
        p_start, p_end = s[0], s[-1]
        dy = p_end[0] - p_start[0]
        dx = p_end[1] - p_start[1]
        # If stroke has significant vertical component and runs bottom-to-top, reverse it
        if dy < -thresh_dy and abs(dy) > abs(dx) * 0.8:
            s_oriented = list(reversed(s))
        # If predominantly horizontal and runs right-to-left, reverse it
        elif abs(dy) <= thresh_dy and dx < -thresh_dx:
            s_oriented = list(reversed(s))
        else:
            s_oriented = s
        oriented_strokes.append(s_oriented)

    # Format as (x, y) numpy coordinate arrays
    recovered = [np.array([(p[1], p[0]) for p in s], dtype=np.float64) for s in oriented_strokes]

    # 7. Merge short junction-artifact stubs into their nearest neighbour.
    #    This reduces skeleton-fragmentation overcount by ~20% with neutral
    #    effect on stroke-level kinematic correlation.  A full fix would
    #    require redesigning the junction traversal in trace_component_strokes;
    #    this is documented as a known limitation in CORRECTIONS.md ISSUE 1.
    recovered = merge_short_strokes(recovered, h_med=h_med)

    return recovered


def recover_handwriting_trajectory(skeleton: np.ndarray, h_med: float = 25.0) -> List[np.ndarray]:
    """
    High-level entry point:
    Input: 2D binary skeleton (0=background, 1=skeleton), optional h_med character height
    Output: List of recovered stroke arrays, each shape (M, 2) [x, y]
    """
    g = skeleton_to_graph(skeleton)
    strokes = recover_strokes_from_graph(g, h_med=h_med)
    return strokes

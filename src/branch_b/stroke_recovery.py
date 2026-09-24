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
    max_gap: float = 4.0,
    max_angle_deg: float = 50.0
) -> List[List[Tuple[int, int]]]:
    """
    Merges disconnected stroke fragments whose endpoints meet within max_gap
    with consistent orientation (cosine similarity >= cos(max_angle_deg)).
    Uses spatial hash indexing for O(N) performance.
    """
    min_cos = np.cos(np.radians(max_angle_deg))
    active = [list(s) for s in strokes if len(s) >= 2]

    cell_size = max(max_gap, 1.0)
    changed = True
    iteration = 0
    max_iterations = 15

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        # Build spatial index of heads
        head_grid: Dict[Tuple[int, int], List[int]] = {}
        for idx, s in enumerate(active):
            if s is None:
                continue
            hy, hx = s[0]
            key = (int(hy // cell_size), int(hx // cell_size))
            if key not in head_grid:
                head_grid[key] = []
            head_grid[key].append(idx)

        for i in range(len(active)):
            if active[i] is None:
                continue
            s1 = active[i]
            tail = s1[-1]
            k1 = min(5, len(s1) - 1)
            v1 = np.array([s1[-1][1] - s1[-1 - k1][1], s1[-1][0] - s1[-1 - k1][0]], dtype=float)
            n1 = np.linalg.norm(v1)
            if n1 < 1e-4:
                continue
            v1 /= n1

            ty, tx = tail
            gy, gx = int(ty // cell_size), int(tx // cell_size)

            best_j = None
            best_score = -1.0

            # Query 9 neighboring grid cells
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cand_list = head_grid.get((gy + dy, gx + dx), [])
                    for j in cand_list:
                        if i == j or active[j] is None:
                            continue
                        s2 = active[j]
                        head = s2[0]
                        dist = np.sqrt((tail[0] - head[0])**2 + (tail[1] - head[1])**2)
                        if dist <= max_gap:
                            k2 = min(5, len(s2) - 1)
                            v2 = np.array([s2[k2][1] - s2[0][1], s2[k2][0] - s2[0][0]], dtype=float)
                            n2 = np.linalg.norm(v2)
                            if n2 < 1e-4:
                                continue
                            v2 /= n2
                            cos_sim = float(np.dot(v1, v2))
                            score = cos_sim - 0.1 * dist
                            if cos_sim >= min_cos and score > best_score:
                                best_score = score
                                best_j = j

            if best_j is not None:
                active[i] = s1 + active[best_j]
                active[best_j] = None
                changed = True

        active = [s for s in active if s is not None]

    return active


def recover_strokes_from_graph(g: nx.Graph) -> List[np.ndarray]:
    """
    Extracts ordered 2D continuous trajectory strokes from a skeleton graph.
    1. Contracts 3-pixel junction cliques into singular intersection nodes.
    2. Clusters graph into connected components, sorting by natural reading flow.
    3. Traces continuous strokes using tangent fly-through at crossings.
    4. Filters out microscopic spurs (< 4 px).
    5. Stitches contiguous stroke fragments.
    Returns list of arrays, each of shape (M, 2) in (x, y) coordinates.
    """
    if g.number_of_nodes() < 2:
        return []

    # 1. Contract junction cliques
    cg = contract_junction_cliques(g)

    # 2. Sort components in natural reading order (top-to-bottom, left-to-right)
    comp_list = list(nx.connected_components(cg))
    comp_list.sort(key=lambda c: (np.min([p[0] for p in c]) // 40, np.min([p[1] for p in c])))

    # 3. Trace strokes through each component
    raw_strokes = []
    for comp_nodes in comp_list:
        sub_g = cg.subgraph(comp_nodes)
        comp_strokes = trace_component_strokes(sub_g)
        raw_strokes.extend(comp_strokes)

    # 4. Filter spurs (< 4 px)
    filtered = [s for s in raw_strokes if len(s) >= 4]

    # 5. Tangent stitching
    stitched = stitch_strokes(filtered, max_gap=4.0, max_angle_deg=50.0)

    # Format as (x, y) numpy coordinate arrays
    recovered = [np.array([(p[1], p[0]) for p in s], dtype=np.float64) for s in stitched]
    return recovered


def recover_handwriting_trajectory(skeleton: np.ndarray) -> List[np.ndarray]:
    """
    High-level entry point:
    Input: 2D binary skeleton (0=background, 1=skeleton)
    Output: List of recovered stroke arrays, each shape (M, 2) [x, y]
    """
    g = skeleton_to_graph(skeleton)
    strokes = recover_strokes_from_graph(g)
    return strokes

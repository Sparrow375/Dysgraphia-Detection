"""Skeletonization, spur pruning, and stroke graph module for Workstream A.

Implements:
1. Morphological skeletonization using skimage.morphology.
2. Spur pruning under 0.15 * x_height.
3. Stroke graph topology extraction via skan (nodes, branches, junctions, endpoints).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from skan.csr import Skeleton


def extract_skeleton(binary_ink: np.ndarray) -> np.ndarray:
    """Extract 1-pixel wide morphological skeleton from binary ink mask."""
    bool_ink = binary_ink > 0
    skel = skeletonize(bool_ink)
    return skel.astype(np.uint8)


def prune_spurs(skeleton: np.ndarray, max_spur_length: float) -> np.ndarray:
    """Prune short terminal spurs (branches ending in endpoints) under max_spur_length pixels."""
    if skeleton.sum() == 0 or max_spur_length <= 1.0:
        return skeleton

    pruned = skeleton.copy()
    max_steps = int(np.ceil(max_spur_length))

    # Kernel to count 8-connected neighbors
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    for _ in range(max_steps):
        # Count neighbors of each skeleton pixel
        neighbor_count = cv2.filter2D(pruned, -1, kernel)
        # Endpoints have exactly 1 neighbor
        endpoints = (pruned == 1) & (neighbor_count == 1)
        if not np.any(endpoints):
            break
        # Remove endpoints in this iteration
        pruned[endpoints] = 0

    return pruned


def build_stroke_graph(skeleton: np.ndarray) -> Dict[str, Any]:
    """Build graph from pruned skeleton and extract topological features.

    Returns:
        dict with:
            - num_nodes: int
            - num_edges: int
            - junction_count: int (degree >= 3)
            - endpoint_count: int (degree == 1)
            - total_path_length: float
            - mean_branch_length: float
            - branches: List of summary dicts for each branch
    """
    if skeleton.sum() < 5:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "junction_count": 0,
            "endpoint_count": 0,
            "total_path_length": 0.0,
            "mean_branch_length": 0.0,
            "branches": [],
        }

    try:
        skan_obj = Skeleton(skeleton.astype(bool))
        n_paths = int(skan_obj.n_paths)

        # Graph node degrees
        degrees = np.array(skan_obj.degrees)
        junction_count = int(np.sum(degrees >= 3))
        endpoint_count = int(np.sum(degrees == 1))

        # Path lengths
        path_lengths = [float(skan_obj.path_lengths()[i]) for i in range(n_paths)]
        total_len = float(sum(path_lengths))
        mean_len = float(np.mean(path_lengths)) if path_lengths else 0.0

        branches: List[Dict[str, Any]] = []
        for i in range(min(n_paths, 200)):  # Store up to 200 main branches to keep JSON compact
            coords = skan_obj.path_coordinates(i)
            # Sample coordinates if long
            if len(coords) > 20:
                step = len(coords) // 20
                coords_sampled = coords[::step]
            else:
                coords_sampled = coords

            branches.append(
                {
                    "branch_id": i,
                    "length": path_lengths[i],
                    "sample_coords": [[int(pt[0]), int(pt[1])] for pt in coords_sampled],
                }
            )

        return {
            "num_nodes": int(len(degrees)),
            "num_edges": n_paths,
            "junction_count": junction_count,
            "endpoint_count": endpoint_count,
            "total_path_length": total_len,
            "mean_branch_length": mean_len,
            "branches": branches,
        }

    except Exception:
        # Fallback if skan encounters disjoint singleton pixels
        return {
            "num_nodes": int(skeleton.sum()),
            "num_edges": 0,
            "junction_count": 0,
            "endpoint_count": 0,
            "total_path_length": float(skeleton.sum()),
            "mean_branch_length": 0.0,
            "branches": [],
        }


def process_sentence_skeleton(
    sentence_crop_ink: np.ndarray,
    x_height_h: float = 40.0,
    spur_ratio: float = 0.15,
) -> Dict[str, Any]:
    """Skeletonize sentence crop, prune spurs, and extract topological graph."""
    raw_skel = extract_skeleton(sentence_crop_ink)
    spur_len = max(2.0, spur_ratio * x_height_h)
    pruned_skel = prune_spurs(raw_skel, max_spur_length=spur_len)
    graph_dict = build_stroke_graph(pruned_skel)

    return {
        "skeleton_mask": pruned_skel,
        "graph": graph_dict,
    }

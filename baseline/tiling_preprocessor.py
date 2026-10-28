"""Tiling & Native Stroke-Scale Preprocessor for Experiment 7.

Performs:
1. White-Canvas Rendering (H x W x 3):
   - Ch1: Clean Inverted Binary Ink (255 background, 0 ink) with ruled lines removed
   - Ch2: Natural Grayscale Shading strictly inside ink (255 background, optical density inside ink)
   - Ch3: Clean Ink + 0.7 Gray Guideline Reference (255 background, 178 guideline, 0 ink)
2. Aspect-Preserving Height Normalization:
   - Rescales line strip proportionally so height = 224px (preserving 1:1 letter aspect ratio)
3. Overlapping Square Tiling:
   - Slides a 224x224 window horizontally with 50% stride (step = 112px)
   - Right-pads boundary tiles with white background (255)
   - Filters completely blank margin tiles (requiring min_ink_pixels >= 50)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def build_white_canvas_strip(
    clean_mask_path: Path,
    raw_gray_path: Path,
    json_path: Optional[Path] = None,
) -> np.ndarray:
    """Construct custom 3-channel white-canvas line strip with ruled lines removed.

    Returns:
        np.ndarray of shape (H, W, 3) with dtype uint8 (values 0..255).
        Background = 255 (pure white canvas).
        Ink = dark strokes (0..128).
    """
    # 1. Load clean ink mask (0 = ink, 255 = background/erased rules)
    mask_gray = cv2.imread(str(clean_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"Missing clean ink mask: {clean_mask_path}")
    ink_bin = (mask_gray < 128).astype(np.float32)

    # 2. Load raw grayscale crop
    raw_gray = cv2.imread(str(raw_gray_path), cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        raise FileNotFoundError(f"Missing raw gray crop: {raw_gray_path}")

    if ink_bin.shape != raw_gray.shape:
        ink_bin = cv2.resize(
            ink_bin, (raw_gray.shape[1], raw_gray.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # Standardize polarity (paper bright, ink dark)
    if np.mean(raw_gray) < 128:
        raw_gray = 255 - raw_gray

    h, w = raw_gray.shape

    # Channel 1: Clean Inverted Binary Ink (255 background, 0 ink)
    ch1 = np.where(ink_bin > 0, 0.0, 255.0).astype(np.float32)

    # Channel 2: Natural Normalized Grayscale Shading inside ink
    non_ink = raw_gray[ink_bin == 0]
    bg_level = float(np.percentile(non_ink, 90)) if len(non_ink) > 50 else 255.0
    bg_level = max(128.0, bg_level)
    optical_density = np.clip((bg_level - raw_gray.astype(np.float32)) / bg_level, 0.0, 1.0)
    ink_intensity = 255.0 * (1.0 - optical_density)
    ch2 = np.where(ink_bin > 0, ink_intensity, 255.0).astype(np.float32)

    # Channel 3: Clean Ink + 0.7 Gray Ruled Guideline Reference
    rule_map = np.zeros((h, w), dtype=np.float32)
    has_rule = False

    if json_path is not None and json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
            bbox = meta.get("crop_bbox", [0, 0, w, h])
            bx, by = bbox[0], bbox[1]
            lines = meta.get("physical_lines", [])
            for pl in lines:
                slope = pl.get("baseline_slope")
                intercept = pl.get("baseline_intercept")
                if slope is not None and intercept is not None:
                    xs = np.arange(w)
                    ys = np.round(slope * (xs + bx) + intercept - by).astype(int)
                    valid = (ys >= 0) & (ys < h)
                    for x_val, y_val in zip(xs[valid], ys[valid]):
                        y_lo = max(0, y_val - 1)
                        y_hi = min(h, y_val + 2)
                        rule_map[y_lo:y_hi, x_val] = 1.0
                    has_rule = True
        except Exception:
            pass

    if not has_rule or np.count_nonzero(rule_map) < 50:
        no_ink = raw_gray.copy()
        no_ink[ink_bin > 0] = 255
        inverted = 255 - no_ink
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1))
        opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        _, bin_rule = cv2.threshold(opened, 25, 1.0, cv2.THRESH_BINARY)
        rule_map = bin_rule.astype(np.float32)

    # Background = 255, guideline = 178 (0.7 intensity), ink = 0 over rule
    ch3 = np.full((h, w), 255.0, dtype=np.float32)
    ch3[rule_map > 0] = 178.0
    ch3[ink_bin > 0] = 0.0

    tensor = np.stack([ch1, ch2, ch3], axis=-1).clip(0.0, 255.0).astype(np.uint8)
    return tensor


def extract_tiles_from_strip(
    canvas_strip: np.ndarray,
    target_height: int = 224,
    stride: int = 112,
    min_ink_pixels: int = 40,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Scales line strip proportionally to target_height and slices overlapping square tiles.

    Args:
        canvas_strip: np.ndarray of shape (H, W, 3), background=255.
        target_height: Target height for 1:1 aspect ratio tiles (default 224).
        stride: Horizontal step size in pixels (default 112 for 50% overlap).
        min_ink_pixels: Minimum number of ink pixels (where ch1 < 128) to retain a tile.

    Returns:
        tiles: List of np.ndarray of shape (target_height, target_height, 3).
        coordinates: List of (x_start, x_end) in scaled coordinate space.
    """
    h, w, c = canvas_strip.shape
    scale = target_height / float(h)
    w_scaled = max(target_height, int(round(w * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_strip = cv2.resize(canvas_strip, (w_scaled, target_height), interpolation=interp)

    tiles: List[np.ndarray] = []
    coords: List[Tuple[int, int]] = []

    for x in range(0, w_scaled, stride):
        x_end = x + target_height
        if x_end <= w_scaled:
            tile = resized_strip[:, x:x_end, :].copy()
        else:
            # Right-pad with pure white canvas (255)
            tile_slice = resized_strip[:, x:w_scaled, :]
            pad_w = target_height - tile_slice.shape[1]
            pad = np.full((target_height, pad_w, c), 255, dtype=np.uint8)
            tile = np.concatenate([tile_slice, pad], axis=1)

        # Check ink presence (Channel 1 < 128 represents dark ink)
        ink_count = int(np.count_nonzero(tile[:, :, 0] < 128))
        if ink_count >= min_ink_pixels or len(tiles) == 0:
            tiles.append(tile)
            coords.append((x, min(x_end, w_scaled)))

    # Ensure at least one tile is returned
    if len(tiles) == 0:
        first_tile = resized_strip[:, :target_height, :].copy()
        if first_tile.shape[1] < target_height:
            pad = np.full((target_height, target_height - first_tile.shape[1], c), 255, dtype=np.uint8)
            first_tile = np.concatenate([first_tile, pad], axis=1)
        tiles.append(first_tile)
        coords.append((0, min(target_height, w_scaled)))

    return tiles, coords


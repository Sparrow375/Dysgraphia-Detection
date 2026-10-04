"""
Synthetic Dysgraphic Handwriting Augmentation Module.

Generates realistic dysgraphic handwriting distortions from clean or standard handwriting.
Each augmentation models a specific neuromotor or visuospatial deficit observed in dysgraphia:
  1. Baseline Wander (Spatial dysgraphia — unstable line orientation)
  2. Letter Size Jitter (Motor dysgraphia — inconsistent vertical scaling)
  3. Stroke Tremor (Neuromotor micro-tremors — elastic deformation along strokes)
  4. Spacing Irregularity (Visuospatial dysgraphia — erratic character spacing)
  5. Character Collision (Visuospatial dysgraphia — overlapping letter boundaries)
  6. Slant Variation (Motor dysgraphia — inconsistent pen tilt and slant angle)
  7. Ink Bleed / Pressure (Motor dysgraphia — excessive or uneven pen pressure)
  8. Stroke Fragmentation (Motor dysgraphia — pen lifts and broken strokes)
"""

from __future__ import annotations

import random
from typing import Tuple, List, Optional
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates


class DysgraphiaAugmenter:
    """
    Applies realistic neuromotor and visuospatial distortions to handwriting images.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def baseline_wander(
        self,
        img: np.ndarray,
        severity: float = 0.5,
    ) -> np.ndarray:
        """
        Sinusoidal and non-linear vertical displacement along the horizontal axis.
        Mimics: Visuospatial dysgraphia — inability to maintain a straight baseline.
        """
        h, w = img.shape[:2]
        if w < 10 or h < 10:
            return img

        # Generate smooth vertical displacement curve
        num_cycles = random.uniform(0.5, 2.5)
        amplitude = int(severity * (h * 0.25))
        x = np.linspace(0, num_cycles * 2 * np.pi, w)
        dy = (amplitude * np.sin(x + random.uniform(0, np.pi))).astype(np.float32)

        # Coordinate grid
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_y = grid_y + dy[np.newaxis, :]
        map_x = grid_x

        distorted = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return distorted

    def stroke_tremor(
        self,
        img: np.ndarray,
        severity: float = 0.5,
    ) -> np.ndarray:
        """
        High-frequency elastic deformation along stroke trajectories.
        Mimics: Motor dysgraphia — neuromotor micro-tremor and jerky pen movements.
        """
        h, w = img.shape[:2]
        if w < 10 or h < 10:
            return img

        # Generate random displacement fields smoothed by small sigma
        sigma = max(1.0, 3.5 - severity * 1.5)  # smaller sigma = higher frequency tremor
        alpha = severity * 6.0  # tremor amplitude

        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=sigma) * alpha

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        distorted = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return distorted

    def letter_size_jitter(
        self,
        img: np.ndarray,
        severity: float = 0.5,
    ) -> np.ndarray:
        """
        Locally varies vertical scale along horizontal segments of the word.
        Mimics: Inconsistent character sizing (high CV of character height).
        """
        h, w = img.shape[:2]
        if w < 20 or h < 10:
            return img

        # Piecewise scale modulation
        n_anchors = max(3, w // 25)
        anchor_scales = 1.0 + (np.random.rand(n_anchors) * 2 - 1) * (0.35 * severity)
        scale_curve = cv2.resize(
            anchor_scales[np.newaxis, :], (w, 1),
            interpolation=cv2.INTER_CUBIC,
        ).squeeze(0)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        center_y = h / 2.0
        map_y = center_y + (grid_y - center_y) / np.maximum(scale_curve[np.newaxis, :], 0.2)
        map_x = grid_x

        distorted = cv2.remap(
            img, map_x.astype(np.float32), map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return distorted

    def spacing_irregularity(
        self,
        img: np.ndarray,
        severity: float = 0.5,
    ) -> np.ndarray:
        """
        Locally expands or compresses horizontal spacing.
        Mimics: Visuospatial crowding and erratic spacing between letters.
        """
        h, w = img.shape[:2]
        if w < 20 or h < 10:
            return img

        n_anchors = max(3, w // 30)
        shifts = (np.random.rand(n_anchors) * 2 - 1) * (12.0 * severity)
        shift_curve = cv2.resize(
            shifts[np.newaxis, :], (w, 1),
            interpolation=cv2.INTER_LINEAR,
        ).squeeze(0)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + shift_curve[np.newaxis, :]
        map_y = grid_y

        distorted = cv2.remap(
            img, map_x.astype(np.float32), map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return distorted

    def slant_variation(
        self,
        img: np.ndarray,
        severity: float = 0.5,
    ) -> np.ndarray:
        """
        Applies non-uniform affine shear to alter slant angle.
        Mimics: Erratic pen inclination and inconsistent stroke slant.
        """
        h, w = img.shape[:2]
        shear_factor = (random.random() * 2 - 1) * (0.35 * severity)

        # Affine transformation matrix for horizontal shearing
        M = np.array([
            [1.0, shear_factor, -shear_factor * (h / 2.0)],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)

        distorted = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return distorted

    def ink_bleed(
        self,
        img: np.ndarray,
        severity: float = 0.5,
    ) -> np.ndarray:
        """
        Morphological dilation + light blur to simulate heavy pen pressure / ink smudge.
        """
        kernel_size = 2 if severity < 0.6 else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(img, kernel, iterations=1)

        # Blend with slight blur
        blurred = cv2.GaussianBlur(dilated, (3, 3), 0)
        return np.maximum(dilated, blurred)

    def stroke_fragmentation(
        self,
        img: np.ndarray,
        severity: float = 0.5,
    ) -> np.ndarray:
        """
        Randomly breaks continuous stroke segments to simulate pen lifts / poor ink flow.
        """
        h, w = img.shape[:2]
        # Create random speckle mask
        mask = (np.random.rand(h, w) > (0.08 * severity)).astype(np.uint8) * 255
        # Morphological opening on mask to create realistic stroke breaks rather than salt noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return cv2.bitwise_and(img, mask)

    def compose(
        self,
        img: np.ndarray,
        severity: float = 0.5,
        n_augmentations: int = 3,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Applies N random dysgraphic augmentations in sequence.

        Args:
            img: Input grayscale/binary image.
            severity: Degradation strength in [0.1, 1.0].
            n_augmentations: How many distortion transforms to chain.

        Returns:
            (augmented_image, list_of_applied_augmentations)
        """
        transforms = [
            ("baseline_wander", self.baseline_wander),
            ("stroke_tremor", self.stroke_tremor),
            ("letter_size_jitter", self.letter_size_jitter),
            ("spacing_irregularity", self.spacing_irregularity),
            ("slant_variation", self.slant_variation),
            ("ink_bleed", self.ink_bleed),
            ("stroke_fragmentation", self.stroke_fragmentation),
        ]

        selected = random.sample(transforms, min(n_augmentations, len(transforms)))
        applied_names: List[str] = []
        out = img.copy()

        for name, fn in selected:
            # Vary severity slightly per transform
            sev = max(0.1, min(1.0, severity * random.uniform(0.75, 1.25)))
            out = fn(out, severity=sev)
            applied_names.append(name)

        return out, applied_names

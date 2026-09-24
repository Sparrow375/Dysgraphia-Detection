"""
Consolidated Dysgraphia Feature Extraction Pipeline.
Integrates Branch A (BHK Static Spatial Features - 9 items) and
Branch B (Reconstructed Kinematic Fluency Features) into a unified, production-ready interface.
Input: Static handwriting image (file path, PIL Image, or NumPy array)
Output: Consolidated dictionary with BHK feature vector, Kinematic feature vector, and full metric dictionaries.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import time
import numpy as np
from PIL import Image

from src.branch_a.preprocessing import load_and_binarize, zhang_suen_skeletonize, compute_stroke_width_map
from src.branch_a.segmentation import segment_handwriting
from src.branch_a.features import (
    compute_size_covariance,
    compute_height_ratio_consistency,
    compute_baseline_drift,
    compute_spacing_entropy,
    compute_stroke_width_stats,
    compute_telescoping_overlap,
    compute_acute_turns,
    compute_left_margin_drift,
    compute_line_collisions,
)
from src.branch_b.stroke_recovery import recover_handwriting_trajectory
from src.branch_b.kinematics import extract_kinematic_features


class DysgraphiaFeaturePipeline:
    """
    Unified multimodal feature extractor for dysgraphia detection from static handwriting.
    """

    BHK_FEATURE_NAMES = [
        "size_covariance",
        "height_ratio_consistency",
        "baseline_drift",
        "spacing_entropy",
        "stroke_width_variance",
        "telescoping_overlap",
        "acute_turns",
        "left_margin_drift",
        "line_collisions",
    ]

    KINEMATIC_FEATURE_NAMES = [
        "mean_velocity",
        "peak_velocity",
        "velocity_skewness",
        "nvi_rate",
        "nvi_per_stroke",
        "jerk_metric",
        "pen_lift_count",
        "mean_stroke_length",
    ]

    def __init__(self, compute_kinematics: bool = True):
        self.compute_kinematics = compute_kinematics

    def extract(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        compute_kinematics: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Runs full feature extraction on an input handwriting image.
        Returns:
          - bhk_vector: (9,) NumPy array
          - bhk_metrics: Dict of individual static indicators
          - kinematic_vector: (8,) NumPy array (or empty if disabled)
          - kinematic_metrics: Dict of kinematic fluency indicators
          - combined_vector: Concatenated (17,) NumPy array
          - combined_feature_names: List of all 17 feature names
          - metadata: Processing time, image dimensions, component counts
        """
        t0 = time.time()
        do_kinematics = self.compute_kinematics if compute_kinematics is None else compute_kinematics

        # 1. Preprocessing: Binarize -> Skeletonize -> Distance Transform
        binary_mask = load_and_binarize(image_input)
        skeleton = zhang_suen_skeletonize(binary_mask)
        dist_map, skeleton_widths, _ = compute_stroke_width_map(binary_mask, skeleton)

        # 2. Recover trajectory (used for acute turns and kinematics)
        recovered_strokes = recover_handwriting_trajectory(skeleton) if np.any(skeleton > 0) else []

        # 3. Branch A: Text line segmentation & BHK feature extraction (9 features)
        text_lines = segment_handwriting(binary_mask)

        f_size = compute_size_covariance(text_lines)
        f_height = compute_height_ratio_consistency(text_lines)
        f_drift = compute_baseline_drift(text_lines)
        f_spacing = compute_spacing_entropy(text_lines)
        f_width = compute_stroke_width_stats(skeleton_widths)
        f_telescope = compute_telescoping_overlap(text_lines)
        f_turns = compute_acute_turns(recovered_strokes)
        f_margin = compute_left_margin_drift(text_lines)
        f_lines = compute_line_collisions(text_lines)

        bhk_vector = np.array([
            f_size["size_covariance_score"],
            f_height["height_iqr_ratio"],
            f_drift["baseline_drift_score"],
            f_spacing["spacing_entropy"],
            f_width["stroke_width_cv"],
            f_telescope["telescoping_score"],
            f_turns["acute_turns_score"],
            f_margin["left_margin_score"],
            f_lines["line_collision_score"],
        ], dtype=np.float64)

        bhk_metrics = {
            **f_size,
            **f_height,
            **f_drift,
            **f_spacing,
            **f_width,
            **f_telescope,
            **f_turns,
            **f_margin,
            **f_lines,
            "total_letters": sum(len(l.components) for l in text_lines),
            "total_lines": len(text_lines),
        }

        # 4. Branch B: Kinematic fluency modeling
        if do_kinematics and recovered_strokes:
            kin_res = extract_kinematic_features(recovered_strokes, dist_map)

            kinematic_vector = np.array([
                kin_res["mean_velocity"],
                kin_res["peak_velocity"],
                kin_res["velocity_skewness"],
                kin_res["nvi_rate"],
                kin_res["nvi_per_stroke"],
                kin_res["jerk_metric"],
                float(kin_res["pen_lift_count"]),
                kin_res["mean_stroke_length"],
            ], dtype=np.float64)

            kinematic_metrics = {
                k: v for k, v in kin_res.items()
                if k not in ("reconstructed_v_full", "reconstructed_p_full")
            }
        else:
            kinematic_vector = np.zeros(len(self.KINEMATIC_FEATURE_NAMES), dtype=np.float64)
            kinematic_metrics = {k: 0.0 for k in self.KINEMATIC_FEATURE_NAMES}

        # Combined feature representation
        if do_kinematics:
            combined_vector = np.concatenate([bhk_vector, kinematic_vector])
            combined_feature_names = self.BHK_FEATURE_NAMES + self.KINEMATIC_FEATURE_NAMES
        else:
            combined_vector = bhk_vector
            combined_feature_names = self.BHK_FEATURE_NAMES

        elapsed_sec = time.time() - t0

        return {
            "bhk_vector": bhk_vector,
            "bhk_feature_names": self.BHK_FEATURE_NAMES,
            "bhk_metrics": bhk_metrics,
            "kinematic_vector": kinematic_vector,
            "kinematic_feature_names": self.KINEMATIC_FEATURE_NAMES,
            "kinematic_metrics": kinematic_metrics,
            "combined_vector": combined_vector,
            "combined_feature_names": combined_feature_names,
            "metadata": {
                "elapsed_seconds": round(elapsed_sec, 3),
                "image_shape": binary_mask.shape,
                "ink_pixels": int(np.sum(binary_mask)),
                "skeleton_pixels": int(np.sum(skeleton)),
                "recovered_strokes_count": len(recovered_strokes),
                "computed_kinematics": do_kinematics,
            }
        }

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.branch_b.kinematics import (
    estimate_stroke_velocity,
    estimate_stroke_pressure_proxy,
    extract_kinematic_features,
)


def generate_synthetic_stroke_geometry(scale: float = 1.0):
    """
    Generates a continuous synthetic cursive loop trajectory across varying resolution scales.
    Coordinates, arc length, stroke thickness, and H_med scale strictly with 'scale'.
    """
    n_pts = int(round(120 * scale))
    t = np.linspace(0, 2 * np.pi, n_pts)
    x = scale * (30.0 * (1 - np.cos(t)) + 15.0 * t)
    y = scale * (40.0 * np.sin(t) - 10.0 * np.sin(2 * t))
    stroke_pts = np.column_stack([x, y])
    h_med = 50.0 * scale

    # Synthetic stroke width: 4.0 * scale pixels
    stroke_width_px = 4.0 * scale
    dist_map = np.full((int(250 * scale), int(400 * scale)), stroke_width_px / 2.0, dtype=np.float32)

    return stroke_pts, dist_map, h_med


class TestKinematicScaleInvariance(unittest.TestCase):
    """
    Unit test verifying that kinematic feature extraction is strictly scale-invariant
    across 0.5x, 1.0x, and 2.0x image zoom/resolution levels.
    Acceptance criterion: normalized kinematic values stay within small tolerance of each other.
    """

    def setUp(self):
        self.scales = [0.5, 1.0, 2.0]

    def test_kinematic_features_scale_invariance(self):
        results = {}
        for s in self.scales:
            pts, dist_map, h_med = generate_synthetic_stroke_geometry(s)
            results[s] = extract_kinematic_features([pts], dist_map, h_med=h_med)

        print("\n" + "=" * 90)
        print("UNIT TEST: KINEMATIC SCALE INVARIANCE VALIDATION (0.5x, 1.0x, 2.0x)")
        print("=" * 90)
        header = f"{'Metric':<25} | {'0.5x':<12} | {'1.0x':<12} | {'2.0x':<12} | {'Max % Diff':<10}"
        print(header)
        print("-" * len(header))

        metrics_to_test = {
            "mean_velocity": 5.0,         # Strict < 5% diff
            "peak_velocity": 5.0,         # Strict < 5% diff
            "velocity_skewness": 5.0,     # Strict < 5% diff
            "jerk_metric": 5.0,           # Strict < 5% diff (resampled arc-length)
            "dimensionless_jerk": 5.0,    # Strict < 5% diff (resampled arc-length)
            "ink_width_ratio_mean": 1.0,   # Strict < 1% diff (normalized stroke width)
        }

        diffs = {}
        for metric_name, max_allowed_diff in metrics_to_test.items():
            v05 = results[0.5][metric_name]
            v10 = results[1.0][metric_name]
            v20 = results[2.0][metric_name]
            vals = [v05, v10, v20]
            max_diff = (max(vals) - min(vals)) / max(abs(float(np.mean(vals))), 1e-6) * 100.0
            diffs[metric_name] = max_diff
            print(f"{metric_name:<25} | {v05:<12.4f} | {v10:<12.4f} | {v20:<12.4f} | {max_diff:<9.2f}%")
            self.assertLess(
                max_diff,
                max_allowed_diff,
                f"Metric {metric_name} failed scale invariance: {max_diff:.2f}% > {max_allowed_diff:.2f}%"
            )

        print("=" * 90)
        print("All kinematic scale-invariance tolerances passed successfully.")


if __name__ == "__main__":
    unittest.main()

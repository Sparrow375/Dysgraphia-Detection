"""
Calibrated Kinematics Reconstruction Engine (Solutions 1, 2, 3, 4)
Reconstructs physically calibrated 13D kinematics from static images:
  1. Execution Timing Calibration: predicts physical duration and scales nominal velocity
     profile so mean velocity, peak velocity, and NVI rate have true physical dimensions.
  2. Dual-band spatial micro-residual decomposition: extracts true 4-8 Hz tremor PSD ratio.
  3. Grayscale photometric optical absorbance: combines Beer-Lambert absorbance with stroke width
     to reconstruct calibrated stylus down-force.
  4. Unit-consistent acceleration & jerk differentials: computes a(t) and j(t) on calibrated
     time grid, ensuring Flash & Hogan dimensionless jerk is strictly unitless and jump-free.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from scipy.signal import find_peaks

from src.branch_b.timing_calibrator import extract_static_hesitation_features
from src.branch_b.spatial_tremor import compute_calibrated_tremor_index
from src.branch_b.photometric_pressure import compute_photometric_pressure

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "kinematic_calibrator.json"
)


class CalibratedKinematicsReconstructor:
    """
    Reconstructs continuous kinematic trajectories v(t), a(t), j(t), p(t)
    and canonical 13D scale-invariant features from static handwriting.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Calibrator model not found at {self.model_path}. Using fallback heuristics.")
            return

        try:
            with open(self.model_path, "r") as f:
                self.model_data = json.load(f)
            self.means_dur = np.array(self.model_data["means_dur"])
            self.stds_dur = np.array(self.model_data["stds_dur"])
            self.coefs_dur = np.array(self.model_data["coefs_dur"])
            self.intercept_dur = float(self.model_data["intercept_dur"])

            self.means_press = np.array(self.model_data["means_press"])
            self.stds_press = np.array(self.model_data["stds_press"])
            self.coefs_press = np.array(self.model_data["coefs_press"])
            self.intercept_press = float(self.model_data["intercept_press"])
            self.is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load calibrator model: {e}")

    def predict_duration(self, hes_feat: Dict[str, float], is_task_s2: float = 0.0) -> float:
        """Predicts true physical execution duration in seconds."""
        if not self.is_loaded:
            # Fallback heuristic: 0.12 s per normalized H_med
            return max(hes_feat.get("total_arc_norm", 10.0) * 0.12, 3.0)

        x_vec = np.array([
            hes_feat["total_arc_norm"],
            hes_feat["total_arc_px"],
            float(hes_feat["n_strokes"]),
            hes_feat["mean_tortuosity"],
            hes_feat["curvature_entropy"],
            hes_feat["terminal_hook_ratio"],
            hes_feat["h_med"],
            float(is_task_s2)
        ], dtype=np.float64)

        x_scaled = (x_vec - self.means_dur) / self.stds_dur
        pred = float(np.dot(x_scaled, self.coefs_dur) + self.intercept_dur)
        # Bounded by physical handwriting speed constraints (0.2 to 8.0 H_med/s)
        total_arc = max(hes_feat["total_arc_norm"], 1.0)
        min_dur = total_arc / 8.0
        max_dur = total_arc / 0.2
        return float(np.clip(pred, max(min_dur, 2.0), max_dur))

    def predict_calibrated_pressure(
        self, mean_absorbance: float, mean_width_norm: float, h_med: float
    ) -> float:
        """Predicts calibrated stylus down-force."""
        if not self.is_loaded:
            return 0.5 * mean_absorbance + 0.5 * mean_width_norm

        x_vec = np.array([mean_absorbance, mean_width_norm, float(h_med)], dtype=np.float64)
        x_scaled = (x_vec - self.means_press) / self.stds_press
        pred = float(np.dot(x_scaled, self.coefs_press) + self.intercept_press)
        return pred

    def extract_calibrated_features(
        self,
        recovered_strokes: List[np.ndarray],
        dist_map: np.ndarray,
        gray_image: Optional[np.ndarray] = None,
        h_med: float = 25.0,
        task: str = "S1",
        nominal_dt: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Extracts the 13 canonical kinematic features using calibrated models.
        """
        h_med = max(float(h_med), 1.0)
        n_strokes = len(recovered_strokes)

        if not recovered_strokes:
            return self._empty_result(h_med)

        # 1. Static Hesitation Features
        is_task_s2 = 1.0 if task == "S2" else 0.0
        hes_feat = extract_static_hesitation_features(recovered_strokes, h_med=h_med)
        total_arc_norm = hes_feat["total_arc_norm"]
        total_arc_px = hes_feat["total_arc_px"]

        # 2. Predicted Contact Duration & Implied Global Speed
        pred_duration = self.predict_duration(hes_feat, is_task_s2=is_task_s2)
        v_implied_norm = total_arc_norm / max(pred_duration, 0.1)
        v_implied_px_s = total_arc_px / max(pred_duration, 0.1)

        # 3. Two-Thirds Power Law Shape Reconstruction & Calibration
        # Compute unscaled shapes first
        all_v_uniform = []
        stroke_lens_norm = []
        stroke_durations = []
        stroke_dim_jerks = []
        stroke_mean_jerks = []
        total_inversions = 0

        # We allocate duration to each stroke proportional to its arc length
        for stk in recovered_strokes:
            if len(stk) < 3:
                continue

            diffs = np.diff(stk, axis=0)
            seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
            arc_px = float(np.sum(seg_lens))
            L_norm = arc_px / h_med
            stroke_lens_norm.append(L_norm)

            if total_arc_norm > 1e-4:
                stk_duration = pred_duration * (L_norm / total_arc_norm)
            else:
                stk_duration = pred_duration / max(n_strokes, 1)
            stk_duration = max(stk_duration, 0.05)
            stroke_durations.append(stk_duration)

            # Normalized arc grid u in [0, 1]
            n_samples = max(int(np.round(stk_duration / nominal_dt)), 6)
            u = np.linspace(0.0, 1.0, n_samples)
            s_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
            norm_s = s_cum / max(arc_px, 1e-6)

            x_norm = np.interp(u, norm_s, stk[:, 0] / h_med)
            y_norm = np.interp(u, norm_s, stk[:, 1] / h_med)

            # Curvature calculation
            dx = np.gradient(x_norm)
            dy = np.gradient(y_norm)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            speed_sq = dx**2 + dy**2
            denom = np.maximum(speed_sq**1.5, 1e-6)
            kappa_norm = np.clip(np.abs(dx * ddy - dy * ddx) / denom, 0.0, 50.0)

            # Base Two-Thirds Power Law speed shape: v_shape(u) ~ 1 / (1 + gamma * kappa^(1/3))
            gamma_c = 1.2
            v_shape = 1.0 / (1.0 + gamma_c * (kappa_norm**(1.0 / 3.0)))
            # Boundary envelope
            env = np.sin(np.pi * u)**0.4
            v_shape = v_shape * np.maximum(env, 0.15)

            # Scale v_shape so its arc-integral equals L_norm over stk_duration:
            # integral v(t) dt = L_norm  =>  mean(v) * stk_duration = L_norm
            mean_shape = float(np.mean(v_shape))
            scale_fac = (L_norm / stk_duration) / max(mean_shape, 1e-6)
            v_calib = v_shape * scale_fac
            v_calib = gaussian_filter1d(v_calib, sigma=1.5)
            all_v_uniform.append(v_calib)

            # NVI on calibrated velocity series
            if len(v_calib) >= 5:
                pks, _ = find_peaks(v_calib)
                trg, _ = find_peaks(-v_calib)
                total_inversions += (len(pks) + len(trg))

                # Unit-consistent acceleration and jerk
                stk_dt = stk_duration / (n_samples - 1)
                stk_acc = np.gradient(v_calib, stk_dt)
                stk_jerk = np.gradient(stk_acc, stk_dt)
                int_jerk_sq = np.sum(stk_jerk**2) * stk_dt

                # Flash & Hogan Dimensionless Jerk: (T^5 / L^2) * int(j^2 dt)
                dim_j = float((stk_duration**5 / (max(L_norm, 0.1)**2)) * int_jerk_sq * 1e-4)
                stroke_dim_jerks.append(dim_j)
                stroke_mean_jerks.append(float(np.mean(stk_jerk**2)))

        if not all_v_uniform:
            return self._empty_result(h_med)

        v_concat = np.concatenate(all_v_uniform)
        mean_v = float(np.mean(v_concat))
        peak_v = float(np.max(v_concat))
        from scipy.stats import skew
        v_skew = float(skew(v_concat)) if len(v_concat) > 2 else 0.0

        # Duration-weighted jerk
        if stroke_mean_jerks:
            jerk_val = float(np.average(stroke_mean_jerks, weights=stroke_durations))
        else:
            jerk_val = 0.0
        mean_dim_jerk = float(np.mean(stroke_dim_jerks)) if stroke_dim_jerks else 0.0

        # 4. Dual-Band Spatial Tremor Index (4-8 Hz)
        tremor_index = compute_calibrated_tremor_index(
            recovered_strokes, h_med=h_med, estimated_speed_px_s=v_implied_px_s
        )

        # 5. Photometric Pressure Calibration
        all_absorbance = []
        all_widths = []
        h_dm, w_dm = dist_map.shape if dist_map is not None else (100, 100)

        if gray_image is not None:
            gray_float = gray_image.astype(np.float64)
            h_im, w_im = gray_image.shape
            for stk in recovered_strokes:
                for pt in stk:
                    ix = int(np.clip(np.round(pt[0]), 0, w_im - 1))
                    iy = int(np.clip(np.round(pt[1]), 0, h_im - 1))
                    i_ink = float(gray_float[iy, ix])
                    abs_val = -np.log10(np.clip((i_ink + 1.0) / 256.0, 1e-4, 1.0))
                    all_absorbance.append(abs_val)
                    if dist_map is not None:
                        all_widths.append(2.0 * float(dist_map[min(iy, h_dm - 1), min(ix, w_dm - 1)]) / max(h_med, 1.0))

        mean_abs = float(np.mean(all_absorbance)) if all_absorbance else 0.5
        mean_w = float(np.mean(all_widths)) if all_widths else 0.5

        calib_press_mean = self.predict_calibrated_pressure(mean_abs, mean_w, h_med)
        # Pressure std proxy
        calib_press_std = float(np.std(all_absorbance)) if len(all_absorbance) > 2 else 0.1

        # 6. Physical NVI Rate & Normalized Metrics
        nvi_rate = float(total_inversions) / max(pred_duration, 0.1)
        nvi_per_stroke = float(total_inversions) / max(n_strokes, 1)
        nvi_per_h_med = float(total_inversions) / max(total_arc_norm, 0.1)

        return {
            "mean_velocity": mean_v,
            "peak_velocity": peak_v,
            "velocity_skewness": v_skew,
            "nvi_rate": nvi_rate,
            "nvi_per_stroke": nvi_per_stroke,
            "nvi_per_100px": nvi_per_h_med,
            "nvi_per_h_med": nvi_per_h_med,
            "total_nvi": int(total_inversions),
            "jerk_metric": jerk_val,
            "dimensionless_jerk": mean_dim_jerk,
            "spatial_roughness_4_8hz": tremor_index,
            "pen_lift_count": n_strokes,
            "mean_stroke_length": float(np.mean(stroke_lens_norm)) if stroke_lens_norm else 0.0,
            "ink_width_ratio_mean": calib_press_mean,
            "ink_width_ratio_std": calib_press_std,
            "predicted_contact_duration": pred_duration,
            "reconstructed_v_full": v_concat,
        }

    def _empty_result(self, h_med: float) -> Dict[str, Any]:
        return {
            "mean_velocity": 0.0,
            "peak_velocity": 0.0,
            "velocity_skewness": 0.0,
            "nvi_rate": 0.0,
            "nvi_per_stroke": 0.0,
            "nvi_per_100px": 0.0,
            "nvi_per_h_med": 0.0,
            "total_nvi": 0,
            "jerk_metric": 0.0,
            "dimensionless_jerk": 0.0,
            "spatial_roughness_4_8hz": 0.0,
            "pen_lift_count": 0,
            "mean_stroke_length": 0.0,
            "ink_width_ratio_mean": 0.0,
            "ink_width_ratio_std": 0.0,
            "predicted_contact_duration": 0.0,
            "reconstructed_v_full": np.array([]),
        }

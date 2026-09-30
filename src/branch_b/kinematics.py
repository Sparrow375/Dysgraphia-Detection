"""
Kinematic parameter estimation and ground-truth validation module.
Implements:
  1. Sigma-Lognormal and 2/3-power-law velocity profile modeling along recovered strokes
  2. Optical stroke-width pressure proxy estimation
  3. Neuromotor fluency feature extraction (NVI, jerk, peak velocity, velocity skewness)
  4. Quantitative correlation & error evaluation against true recorded tablet telemetry

Feature renaming (Step 4):
  tremor_index_4_8hz  →  spatial_roughness_4_8hz
  Rationale: the pipeline produces a 4–8 Hz PSD ratio over the ESTIMATED velocity
  profile — not an accelerometer or EEG tremor measurement.  "Spatial roughness"
  better describes what the feature actually measures: micro-oscillations in the
  reconstructed path that correlate with motor tremor but are not a direct measurement.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import pearsonr, skew
from src.loaders import SampleData, MIN_DT_S

logger = logging.getLogger(__name__)


def estimate_stroke_velocity(
    stroke_pts: np.ndarray,
    h_med: float = 25.0,
    v_nominal_norm: float = 4.0,
    gamma_curvature: float = 1.2,
    n_arc_resample: int = 100,
    return_resampled: bool = False
) -> Any:
    """
    Estimates a continuous scale-invariant velocity profile for an individual stroke using
    Plamondon's Kinematic Theory and the Two-Thirds Power Law of human motor control:
      v_norm(s) = v_nominal_norm / (1 + gamma * kappa_norm^(1/3)) * boundary_envelope

    Resamples each stroke onto a fixed number of normalized arc-length points before
    differentiating (independent of native pixel count), ensuring higher derivatives
    (acceleration, jerk) are strictly scale-invariant across camera resolutions.

    Returns:
      arc_lengths: cumulative distance array in pixels (M,)
      velocity: reconstructed velocity array in H_med / sec (M,)
      curvature: dimensionless local curvature array kappa * H_med (M,)
      [optional] v_resamp: velocity profile on normalized arc-length grid u (n_arc_resample,)
    """
    n_pts = len(stroke_pts)
    if n_pts < 2:
        zeros = np.zeros(n_pts)
        if return_resampled:
            return zeros, zeros, zeros, np.zeros(n_arc_resample)
        return zeros, zeros, zeros

    h_med = max(float(h_med), 1.0)

    # 1. Arc-length parameterization
    diffs = np.diff(stroke_pts, axis=0)
    seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = s[-1]

    if total_len < 1e-4 or n_pts < 4:
        const_v = np.ones(n_pts) * (0.25 * v_nominal_norm)
        zeros_k = np.zeros(n_pts)
        if return_resampled:
            return s, const_v, zeros_k, np.ones(n_arc_resample) * (0.25 * v_nominal_norm)
        return s, const_v, zeros_k

    # 2. Resample onto fixed normalized arc-length grid u in [0, 1]
    # This decouples discrete point density from image resolution
    u = np.linspace(0.0, 1.0, n_arc_resample)
    norm_s_orig = s / max(total_len, 1e-6)
    x_norm = np.interp(u, norm_s_orig, stroke_pts[:, 0] / h_med)
    y_norm = np.interp(u, norm_s_orig, stroke_pts[:, 1] / h_med)

    # Fixed scale-invariant smoothing along normalized arc length
    sigma_smooth = 2.0
    x_smooth = gaussian_filter1d(x_norm, sigma=sigma_smooth)
    y_smooth = gaussian_filter1d(y_norm, sigma=sigma_smooth)

    # First and second derivatives with respect to normalized arc index
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Dimensionless curvature: kappa * H_med
    speed_sq = dx**2 + dy**2
    denom = np.maximum(speed_sq**1.5, 1e-6)
    kappa_norm_resamp = np.clip(np.abs(dx * ddy - dy * ddx) / denom, 0.0, 50.0)

    # 3. Two-Thirds Power Law base speed in H_med / sec
    base_speed = v_nominal_norm / (1.0 + gamma_curvature * (kappa_norm_resamp**(1.0 / 3.0)))

    # 4. Asymmetric Plamondon Sigma-Lognormal Boundary Envelope
    p_exp, q_exp = 0.8, 1.4
    u_peak = p_exp / (p_exp + q_exp)
    peak_norm = (u_peak**p_exp) * ((1.0 - u_peak)**q_exp)
    raw_envelope = (u**p_exp) * ((1.0 - u + 1e-6)**q_exp)
    envelope = np.clip(raw_envelope / max(peak_norm, 1e-6), 0.0, 1.0)

    v_resamp = base_speed * envelope
    v_resamp = gaussian_filter1d(v_resamp, sigma=1.0)
    v_resamp = np.maximum(v_resamp, 0.05 * v_nominal_norm)

    # 5. Interpolate back to original stroke points for exact per-point fidelity
    reconstructed_v = np.interp(norm_s_orig, u, v_resamp)
    kappa_norm = np.interp(norm_s_orig, u, kappa_norm_resamp)

    if return_resampled:
        return s, reconstructed_v, kappa_norm, v_resamp
    return s, reconstructed_v, kappa_norm


def estimate_stroke_pressure_proxy(
    stroke_pts: np.ndarray,
    dist_map: np.ndarray,
    h_med: float = 25.0
) -> np.ndarray:
    """
    Extracts local stroke width from the distance transform along the recovered stroke
    and normalizes by H_med (relative stroke width W / H_med) as a scale-invariant
    optical proxy for stylus down-force pressure.
    """
    h_med = max(float(h_med), 1.0)
    h, w = dist_map.shape
    pressures = []
    for x, y in stroke_pts:
        ix = int(np.clip(np.round(x), 0, w - 1))
        iy = int(np.clip(np.round(y), 0, h - 1))
        # Stroke width normalized by H_med: (2 * distance to background) / H_med
        pw = (2.0 * float(dist_map[iy, ix])) / h_med
        pressures.append(pw)

    p_arr = np.array(pressures, dtype=np.float64)
    p_smooth = gaussian_filter1d(p_arr, sigma=1.5) if len(p_arr) > 3 else p_arr
    return np.maximum(p_smooth, 0.01)


def compute_tremor_spectral_power(
    velocity_series: np.ndarray,
    fs: float = 100.0,
    tremor_band: Tuple[float, float] = (4.0, 8.0),
    total_band: Tuple[float, float] = (0.5, 20.0)
) -> float:
    """
    Computes the relative Power Spectral Density (PSD) in the neuromuscular tremor band (4-8 Hz)
    versus total handwriting motor band (0.5-20 Hz).
    Dysgraphic children exhibit involuntary micro-oscillations in the 4-8 Hz band.
    """
    if len(velocity_series) < 32:
        return 0.0

    from scipy.signal import welch
    nperseg = min(len(velocity_series), 128)
    freqs, psd = welch(velocity_series - np.mean(velocity_series), fs=fs, nperseg=nperseg)

    tremor_mask = (freqs >= tremor_band[0]) & (freqs <= tremor_band[1])
    total_mask = (freqs >= total_band[0]) & (freqs <= total_band[1])

    tremor_power = float(np.sum(psd[tremor_mask]))
    total_power = float(np.sum(psd[total_mask]))

    if total_power <= 1e-9:
        return 0.0

    return float(np.clip(tremor_power / total_power, 0.0, 1.0))


def extract_kinematic_features(
    recovered_strokes: List[np.ndarray],
    dist_map: np.ndarray,
    h_med: float = 25.0,
    v_nominal_norm: float = 4.0,
    nominal_dt: float = 0.01
) -> Dict[str, Any]:
    """
    Reconstructs scale-invariant velocity and pressure profiles across all strokes and extracts
    neuromotor fluency and kinematic features normalized by median character height H_med:
    - Asymmetric Sigma-Lognormal velocity distributions (reported in H_med / sec)
    - Flash & Hogan Dimensionless Jerk
    - 4-8 Hz Neuromuscular Tremor Spectral Power Index
    - Normalized Velocity Inversions (NVI per stroke & per H_med unit)
    - Pressure proxy reported per H_med unit (relative stroke width)
    """
    h_med = max(float(h_med), 1.0)

    if not recovered_strokes:
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
            "reconstructed_v_full": np.array([]),
            "reconstructed_p_full": np.array([]),
        }

    all_v_pts = []
    all_v_uniform = []
    all_p = []
    total_inversions = 0
    stroke_lens_norm = []
    stroke_dimensionless_jerks = []
    stroke_mean_jerks = []
    stroke_durations = []

    for stroke in recovered_strokes:
        if len(stroke) < 3:
            continue
        s, v, _, v_resamp = estimate_stroke_velocity(
            stroke, h_med=h_med, v_nominal_norm=v_nominal_norm, return_resampled=True
        )
        p = estimate_stroke_pressure_proxy(stroke, dist_map, h_med=h_med)

        all_v_pts.append(v)
        all_p.append(p)
        L_norm = float(s[-1] / h_med)
        stroke_lens_norm.append(L_norm)

        # Scale-invariant temporal parameterization:
        # Directly utilize the normalized arc-length velocity profile (independent of discrete pixel count)
        n_resamp = len(v_resamp)
        ds_norm = (1.0 / max(n_resamp - 1, 1)) * L_norm
        v_mid = 0.5 * (v_resamp[:-1] + v_resamp[1:])
        dt_segs = ds_norm / np.maximum(v_mid, 1e-3)
        t_cum = np.concatenate([[0.0], np.cumsum(dt_segs)])
        total_duration = max(float(t_cum[-1]), 0.05)

        # Resample uniformly at fs = 100 Hz (dt = 0.01s)
        n_samples = max(int(np.round(total_duration / nominal_dt)), 5)
        t_uniform = np.linspace(0.0, total_duration, n_samples)
        v_uniform = np.interp(t_uniform, t_cum, v_resamp)
        v_uniform = gaussian_filter1d(v_uniform, sigma=2.0)
        all_v_uniform.append(v_uniform)

        # Number of Velocity Inversions (NVI): local peaks and troughs in speed
        if len(v_uniform) >= 5:
            peaks, _ = find_peaks(v_uniform)
            troughs, _ = find_peaks(-v_uniform)
            total_inversions += (len(peaks) + len(troughs))

            # Flash & Hogan dimensionless jerk per stroke: (T^5 / L^2) * integral(jerk^2 dt)
            stk_acc = np.gradient(v_uniform, nominal_dt)
            stk_jerk = np.gradient(stk_acc, nominal_dt)
            integral_jerk_sq = np.sum(stk_jerk**2) * nominal_dt
            # Strictly dimensionless normalization (length and time units cancel)
            dim_jerk = float((total_duration**5 / (max(L_norm, 0.1)**2)) * integral_jerk_sq * 1e-4)
            stroke_dimensionless_jerks.append(dim_jerk)
            stroke_mean_jerks.append(float(np.mean(stk_jerk**2)))
            stroke_durations.append(total_duration)

    if not all_v_pts:
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
            "pen_lift_count": len(recovered_strokes),
            "mean_stroke_length": 0.0,
            "ink_width_ratio_mean": 0.0,
            "ink_width_ratio_std": 0.0,
            "reconstructed_v_full": np.array([]),
            "reconstructed_p_full": np.array([]),
        }

    v_pts_concat = np.concatenate(all_v_pts)
    v_uniform_concat = np.concatenate(all_v_uniform)
    p_concat = np.concatenate(all_p)

    mean_v = float(np.mean(v_uniform_concat))
    peak_v = float(np.max(v_uniform_concat))
    v_skew = float(skew(v_uniform_concat)) if len(v_uniform_concat) > 2 else 0.0

    # Overall jerk metric: duration-weighted average of stroke jerks
    # (prevents artificial numerical spikes at discrete pen-lift seams)
    if stroke_mean_jerks:
        jerk_val = float(np.average(stroke_mean_jerks, weights=stroke_durations))
    else:
        jerk_val = 0.0

    # Mean dimensionless jerk
    mean_dim_jerk = float(np.mean(stroke_dimensionless_jerks)) if stroke_dimensionless_jerks else 0.0

    # 4-8 Hz Neuromuscular Tremor Spectral Power Index on uniform 100 Hz series
    fs_nominal = 1.0 / nominal_dt
    tremor_index = compute_tremor_spectral_power(v_uniform_concat, fs=fs_nominal)

    # Scale-invariant NVI metrics
    total_arc_norm = sum(stroke_lens_norm) if stroke_lens_norm else 0.0
    total_duration = len(v_uniform_concat) * nominal_dt
    nvi_rate = float(total_inversions) / max(total_duration, 0.1)
    nvi_per_stroke = float(total_inversions) / max(len(recovered_strokes), 1)
    nvi_per_h_med = float(total_inversions) / max(total_arc_norm, 0.1)
    nvi_per_100px = nvi_per_h_med  # Scale-invariant normalized count

    mean_press = float(np.mean(p_concat))
    std_press = float(np.std(p_concat))

    return {
        "mean_velocity": mean_v,
        "peak_velocity": peak_v,
        "velocity_skewness": v_skew,
        "nvi_rate": nvi_rate,
        "nvi_per_stroke": nvi_per_stroke,
        "nvi_per_100px": nvi_per_100px,
        "nvi_per_h_med": nvi_per_h_med,
        "total_nvi": int(total_inversions),
        "jerk_metric": jerk_val,
        "dimensionless_jerk": mean_dim_jerk,
        "spatial_roughness_4_8hz": tremor_index,
        "pen_lift_count": len(recovered_strokes),
        "mean_stroke_length": float(np.mean(stroke_lens_norm)) if stroke_lens_norm else 0.0,
        "ink_width_ratio_mean": mean_press,
        "ink_width_ratio_std": std_press,
        "reconstructed_v_full": v_pts_concat,
        "reconstructed_p_full": p_concat,
    }



def validate_kinematics_against_ground_truth(
    sample: SampleData,
    recovered_strokes: List[np.ndarray],
    dist_map: np.ndarray,
    h_med: float = 25.0,
    n_resample: int = 500
) -> Dict[str, Any]:
    """
    Compares reconstructed velocity and pressure profiles against true recorded tablet sensors:
      - Level 1: Single-Stroke correlation distribution (matches individual strokes, evaluates 2/3 Power Law)
      - Level 3: Whole-document concatenated progression grid [0, 1]
    """
    # 1. Ground truth on-surface velocity and pressure
    on_mask = (sample.pen_status > 0.5) & (sample.pressure > 0)
    if not np.any(on_mask) or len(sample.points) < 5:
        return {"status": "insufficient_ground_truth"}

    gt_pts = sample.points[on_mask]
    gt_pressure = sample.pressure[on_mask]

    # Compute true velocity
    dx = np.diff(gt_pts[:, 0])
    dy = np.diff(gt_pts[:, 1])
    dt = np.diff(gt_pts[:, 2])
    dt_safe = np.where(dt < MIN_DT_S, MIN_DT_S, dt)
    gt_v = np.sqrt(dx**2 + dy**2) / dt_safe
    gt_v = np.concatenate([[gt_v[0]], gt_v])
    q99 = np.percentile(gt_v, 99) if len(gt_v) > 0 else 1.0
    gt_v = np.clip(gt_v, 0, max(q99 * 1.5, 1.0))

    # 2. Reconstructed profiles
    kin = extract_kinematic_features(recovered_strokes, dist_map, h_med=h_med)
    rec_v = kin["reconstructed_v_full"]
    rec_p = kin["reconstructed_p_full"]

    if len(rec_v) < 5 or len(gt_v) < 5:
        return {"status": "insufficient_reconstructed_points"}

    # 3. Resample both onto uniform progression [0, 1]
    grid = np.linspace(0.0, 1.0, n_resample)
    gt_grid = np.linspace(0.0, 1.0, len(gt_v))
    rec_grid = np.linspace(0.0, 1.0, len(rec_v))

    gt_v_resamp = np.interp(grid, gt_grid, gt_v)
    rec_v_resamp = np.interp(grid, rec_grid, rec_v)

    gt_p_resamp = np.interp(grid, gt_grid, gt_pressure)
    rec_p_resamp = np.interp(grid, rec_grid, rec_p)

    def min_max(arr):
        mn, mx = np.min(arr), np.max(arr)
        return (arr - mn) / max(mx - mn, 1e-6)

    gt_v_norm = min_max(gt_v_resamp)
    rec_v_norm = min_max(rec_v_resamp)

    gt_p_norm = min_max(gt_p_resamp)
    rec_p_norm = min_max(rec_p_resamp)

    # 4. Whole-document metrics
    # Explicitly detect constant inputs to prevent ConstantInputWarning
    std_rec_v = float(np.std(rec_v_norm))
    std_gt_v = float(np.std(gt_v_norm))
    if std_rec_v > 1e-4 and std_gt_v > 1e-4:
        r_velocity, pval_v = pearsonr(rec_v_norm, gt_v_norm)
        r_velocity = float(r_velocity)
        pval_v = float(pval_v)
    else:
        r_velocity, pval_v = float('nan'), float('nan')
        logger.warning(
            f"[{sample.sample_id}] Whole-document velocity correlation excluded: "
            f"constant input signal (rec_std={std_rec_v:.2e}, gt_std={std_gt_v:.2e})"
        )

    std_rec_p = float(np.std(rec_p_norm))
    std_gt_p = float(np.std(gt_p_norm))
    if std_rec_p > 1e-4 and std_gt_p > 1e-4:
        r_pressure, pval_p = pearsonr(rec_p_norm, gt_p_norm)
        r_pressure = float(r_pressure)
        pval_p = float(pval_p)
    else:
        r_pressure, pval_p = float('nan'), float('nan')
        logger.warning(
            f"[{sample.sample_id}] Whole-document pressure correlation excluded: "
            f"constant input signal (rec_std={std_rec_p:.2e}, gt_std={std_gt_p:.2e})"
        )

    nrmse_v = float(np.sqrt(np.mean((rec_v_norm - gt_v_norm)**2)))
    nrmse_p = float(np.sqrt(np.mean((rec_p_norm - gt_p_norm)**2)))

    # 5. Level 1: Single-Stroke Ground-Truth Validation
    # Map GT strokes into pixel coordinate space
    stroke_level_corrs = []
    if sample.strokes and recovered_strokes:
        all_x = np.concatenate([s[:, 0] for s in sample.strokes])
        all_y = np.concatenate([s[:, 1] for s in sample.strokes])
        min_x, max_x = float(np.min(all_x)), float(np.max(all_x))
        min_y, max_y = float(np.min(all_y)), float(np.max(all_y))
        span_x = max(max_x - min_x, 1e-5)
        target_width = 1200
        padding = 40
        drawable_width = target_width - 2 * padding
        scale = drawable_width / span_x
        dt_sample = 1.0 / max(sample.sampling_rate_hz, 1.0)

        for s_gt in sample.strokes:
            pts_gt = s_gt[:, :2]
            if len(pts_gt) < 10:
                continue
            # Pixel transformation
            px = padding + (pts_gt[:, 0] - min_x) * scale
            py = padding + (max_y - pts_gt[:, 1]) * scale
            gt_px = np.column_stack([px, py])

            # GT stroke velocity
            d_diff = np.diff(gt_px, axis=0)
            d_dists = np.sqrt(d_diff[:, 0]**2 + d_diff[:, 1]**2)
            v_gt_s = d_dists / dt_sample
            v_gt_s = np.concatenate([[v_gt_s[0]], v_gt_s])
            if np.std(v_gt_s) < 1e-3:
                continue

            c_gt = np.mean(gt_px, axis=0)

            # Find matching recovered stroke
            best_d = float('inf')
            best_rec = None
            for s_rec in recovered_strokes:
                if len(s_rec) < 6:
                    continue
                c_rec = np.mean(s_rec, axis=0)
                d = np.linalg.norm(c_gt - c_rec)
                if d < best_d:
                    best_d = d
                    best_rec = s_rec

            if best_d < 30.0 and best_rec is not None:
                _, v_rec_s, _ = estimate_stroke_velocity(best_rec, h_med=h_med)
                if len(v_rec_s) >= 6 and np.std(v_rec_s) > 1e-4:
                    common_len = 50
                    grid_common = np.linspace(0, 1, common_len)
                    f_g = np.interp(grid_common, np.linspace(0, 1, len(v_gt_s)), v_gt_s)
                    f_r = np.interp(grid_common, np.linspace(0, 1, len(v_rec_s)), v_rec_s)
                    if np.std(f_g) > 1e-4 and np.std(f_r) > 1e-4:
                        r_fwd, _ = pearsonr(f_g, f_r)
                        r_rev, _ = pearsonr(f_g, f_r[::-1])
                        best_r = max(r_fwd, r_rev)
                        if not np.isnan(best_r):
                            stroke_level_corrs.append(float(best_r))

    stroke_r_arr = np.array(stroke_level_corrs) if stroke_level_corrs else np.array([])
    stroke_mean_r = float(np.mean(stroke_r_arr)) if len(stroke_r_arr) > 0 else 0.0
    stroke_median_r = float(np.median(stroke_r_arr)) if len(stroke_r_arr) > 0 else 0.0
    frac_gt_03 = float(np.sum(stroke_r_arr > 0.30) / len(stroke_r_arr)) if len(stroke_r_arr) > 0 else 0.0
    frac_gt_05 = float(np.sum(stroke_r_arr > 0.50) / len(stroke_r_arr)) if len(stroke_r_arr) > 0 else 0.0

    return {
        "status": "success",
        "sample_id": sample.sample_id,
        "dataset": sample.dataset_name,
        "task": sample.task_name,
        "pearson_r_velocity": float(r_velocity),
        "pvalue_velocity": float(pval_v),
        "pearson_r_pressure": float(r_pressure),
        "pvalue_pressure": float(pval_p),
        "nrmse_velocity": nrmse_v,
        "nrmse_pressure": nrmse_p,
        "stroke_level_count": len(stroke_level_corrs),
        "stroke_level_mean_r": stroke_mean_r,
        "stroke_level_median_r": stroke_median_r,
        "stroke_level_fraction_gt_03": frac_gt_03,
        "stroke_level_fraction_gt_05": frac_gt_05,
        "gt_stroke_count": len(sample.strokes),
        "recovered_stroke_count": len(recovered_strokes),
        "kinematic_features": {
            "mean_velocity": kin["mean_velocity"],
            "peak_velocity": kin["peak_velocity"],
            "velocity_skewness": kin["velocity_skewness"],
            "nvi_rate": kin["nvi_rate"],
            "nvi_per_stroke": kin["nvi_per_stroke"],
            "nvi_per_100px": kin["nvi_per_100px"],
            "total_nvi": kin["total_nvi"],
            "dimensionless_jerk": kin["dimensionless_jerk"],
            "jerk_metric": kin["jerk_metric"],
            "spatial_roughness_4_8hz": kin["spatial_roughness_4_8hz"],
            "pen_lift_count": kin["pen_lift_count"],
            "ink_width_ratio_mean": kin["ink_width_ratio_mean"],
        },
        "resampled_curves": {
            "grid": grid.tolist(),
            "gt_v_norm": gt_v_norm.tolist(),
            "rec_v_norm": rec_v_norm.tolist(),
            "gt_p_norm": gt_p_norm.tolist(),
            "rec_p_norm": rec_p_norm.tolist(),
        }
    }

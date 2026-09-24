"""
Kinematic parameter estimation and ground-truth validation module.
Implements:
  1. Sigma-Lognormal and 2/3-power-law velocity profile modeling along recovered strokes
  2. Optical stroke-width pressure proxy estimation
  3. Neuromotor fluency feature extraction (NVI, jerk, peak velocity, velocity skewness)
  4. Quantitative correlation & error evaluation against true recorded tablet telemetry
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import pearsonr, skew
from src.loaders import SampleData


def estimate_stroke_velocity(
    stroke_pts: np.ndarray,
    v_nominal: float = 100.0,
    gamma_curvature: float = 1.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates a continuous velocity profile for an individual stroke using
    Plamondon's Kinematic Theory and the Two-Thirds Power Law of human motor control:
      v(s) ~ v_max / (1 + gamma * kappa^(1/3)) * boundary_envelope
    Returns:
      arc_lengths: cumulative distance array (M,)
      velocity: reconstructed velocity array (M,)
      curvature: local curvature array (M,)
    """
    n_pts = len(stroke_pts)
    if n_pts < 2:
        return np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts)

    # 1. Arc-length parameterization
    diffs = np.diff(stroke_pts, axis=0)
    seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = s[-1]

    if total_len < 1e-4 or n_pts < 4:
        return s, np.ones(n_pts) * 10.0, np.zeros(n_pts)

    # 2. Smooth coordinates for stable differential calculus
    sigma_smooth = max(n_pts / 30.0, 1.5)
    x_smooth = gaussian_filter1d(stroke_pts[:, 0], sigma=sigma_smooth)
    y_smooth = gaussian_filter1d(stroke_pts[:, 1], sigma=sigma_smooth)

    # First and second derivatives with respect to index/step
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature: kappa = |dx*ddy - dy*ddx| / (dx^2 + dy^2)^(1.5)
    speed_sq = dx**2 + dy**2
    denom = np.maximum(speed_sq**1.5, 1e-6)
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    curvature = np.clip(curvature, 0.0, 5.0)

    # 3. Two-Thirds Power Law base speed: v ~ kappa^(-1/3)
    # Slows down at sharp corners; speeds up along straight trajectories
    base_speed = v_nominal / (1.0 + gamma_curvature * (curvature**(1.0 / 3.0)))

    # 4. Neuromuscular boundary envelope (smooth acceleration from zero at touchdown,
    # and smooth deceleration to zero at lift-off)
    norm_s = s / max(total_len, 1e-4)
    # Bell-shaped boundary ramp using sine envelope: sin(pi * s / L)
    envelope = np.sin(np.pi * norm_s) ** 0.5
    reconstructed_v = base_speed * envelope

    # Smooth the final velocity curve
    reconstructed_v = gaussian_filter1d(reconstructed_v, sigma=max(sigma_smooth / 2.0, 1.0))
    reconstructed_v = np.maximum(reconstructed_v, 1.0)

    return s, reconstructed_v, curvature


def estimate_stroke_pressure_proxy(
    stroke_pts: np.ndarray,
    dist_map: np.ndarray
) -> np.ndarray:
    """
    Extracts local stroke width from the distance transform along the recovered stroke
    as an optical proxy for stylus down-force pressure.
    """
    h, w = dist_map.shape
    pressures = []
    for x, y in stroke_pts:
        ix = int(np.clip(np.round(x), 0, w - 1))
        iy = int(np.clip(np.round(y), 0, h - 1))
        # Stroke width = 2 * distance to background
        pw = 2.0 * float(dist_map[iy, ix])
        pressures.append(pw)

    p_arr = np.array(pressures, dtype=np.float64)
    p_smooth = gaussian_filter1d(p_arr, sigma=1.5) if len(p_arr) > 3 else p_arr
    return np.maximum(p_smooth, 0.5)


def extract_kinematic_features(
    recovered_strokes: List[np.ndarray],
    dist_map: np.ndarray,
    nominal_dt: float = 0.01
) -> Dict[str, Any]:
    """
    Reconstructs velocity and pressure profiles across all strokes and extracts
    neuromotor fluency and kinematic features.
    """
    if not recovered_strokes:
        return {
            "mean_velocity": 0.0,
            "peak_velocity": 0.0,
            "velocity_skewness": 0.0,
            "nvi_rate": 0.0,
            "total_nvi": 0,
            "jerk_metric": 0.0,
            "pen_lift_count": 0,
            "mean_stroke_length": 0.0,
            "pressure_proxy_mean": 0.0,
            "pressure_proxy_std": 0.0,
            "reconstructed_v_full": np.array([]),
            "reconstructed_p_full": np.array([]),
        }

    all_v = []
    all_p = []
    total_inversions = 0
    stroke_lens = []

    for stroke in recovered_strokes:
        if len(stroke) < 3:
            continue
        s, v, _ = estimate_stroke_velocity(stroke)
        p = estimate_stroke_pressure_proxy(stroke, dist_map)

        all_v.append(v)
        all_p.append(p)
        stroke_lens.append(s[-1])

        # Number of Velocity Inversions (NVI): local peaks and troughs in speed
        if len(v) >= 5:
            peaks, _ = find_peaks(v)
            troughs, _ = find_peaks(-v)
            total_inversions += (len(peaks) + len(troughs))

    if not all_v:
        return {
            "mean_velocity": 0.0,
            "peak_velocity": 0.0,
            "velocity_skewness": 0.0,
            "nvi_rate": 0.0,
            "total_nvi": 0,
            "jerk_metric": 0.0,
            "pen_lift_count": len(recovered_strokes),
            "mean_stroke_length": 0.0,
            "pressure_proxy_mean": 0.0,
            "pressure_proxy_std": 0.0,
            "reconstructed_v_full": np.array([]),
            "reconstructed_p_full": np.array([]),
        }

    v_concat = np.concatenate(all_v)
    p_concat = np.concatenate(all_p)

    mean_v = float(np.mean(v_concat))
    peak_v = float(np.max(v_concat))
    v_skew = float(skew(v_concat)) if len(v_concat) > 2 else 0.0

    # Acceleration and jerk
    acc = np.gradient(v_concat, nominal_dt)
    jerk = np.gradient(acc, nominal_dt)
    jerk_val = float(np.mean(jerk**2))

    # NVI normalized by stroke count and arc length to prevent stroke length artifacts
    total_arc = sum(stroke_lens) if stroke_lens else 0.0
    nvi_rate = float(total_inversions) / max(len(v_concat) * nominal_dt, 0.1)
    nvi_per_stroke = float(total_inversions) / max(len(recovered_strokes), 1)
    nvi_per_100px = float(total_inversions) / max(total_arc / 100.0, 0.1)

    mean_press = float(np.mean(p_concat))
    std_press = float(np.std(p_concat))

    return {
        "mean_velocity": mean_v,
        "peak_velocity": peak_v,
        "velocity_skewness": v_skew,
        "nvi_rate": nvi_rate,
        "nvi_per_stroke": nvi_per_stroke,
        "nvi_per_100px": nvi_per_100px,
        "total_nvi": int(total_inversions),
        "jerk_metric": jerk_val,
        "pen_lift_count": len(recovered_strokes),
        "mean_stroke_length": float(np.mean(stroke_lens)) if stroke_lens else 0.0,
        "pressure_proxy_mean": mean_press,
        "pressure_proxy_std": std_press,
        "reconstructed_v_full": v_concat,
        "reconstructed_p_full": p_concat,
    }


def validate_kinematics_against_ground_truth(
    sample: SampleData,
    recovered_strokes: List[np.ndarray],
    dist_map: np.ndarray,
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
    dt_safe = np.where(dt <= 1e-5, 1e-5, dt)
    gt_v = np.sqrt(dx**2 + dy**2) / dt_safe
    gt_v = np.concatenate([[gt_v[0]], gt_v])
    q99 = np.percentile(gt_v, 99) if len(gt_v) > 0 else 1.0
    gt_v = np.clip(gt_v, 0, max(q99 * 1.5, 1.0))

    # 2. Reconstructed profiles
    kin = extract_kinematic_features(recovered_strokes, dist_map)
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
    r_velocity, pval_v = pearsonr(rec_v_norm, gt_v_norm)
    r_pressure, pval_p = pearsonr(rec_p_norm, gt_p_norm)

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
                _, v_rec_s, _ = estimate_stroke_velocity(best_rec)
                if len(v_rec_s) >= 6 and np.std(v_rec_s) > 1e-4:
                    common_len = 50
                    grid_common = np.linspace(0, 1, common_len)
                    f_g = np.interp(grid_common, np.linspace(0, 1, len(v_gt_s)), v_gt_s)
                    f_r = np.interp(grid_common, np.linspace(0, 1, len(v_rec_s)), v_rec_s)
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
            "pen_lift_count": kin["pen_lift_count"],
            "pressure_proxy_mean": kin["pressure_proxy_mean"],
        },
        "resampled_curves": {
            "grid": grid.tolist(),
            "gt_v_norm": gt_v_norm.tolist(),
            "rec_v_norm": rec_v_norm.tolist(),
            "gt_p_norm": gt_p_norm.tolist(),
            "rec_p_norm": rec_p_norm.tolist(),
        }
    }

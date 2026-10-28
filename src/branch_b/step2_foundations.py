"""
Step 2 Foundations: Physical Signal Extraction & Scale-Invariant Modeling
Implements:
  2a. Scale-invariant B-spline smoothing and tangent-continuity graph merging at degree-3+ skeleton junctions.
  2b. Space-to-Pseudo-Time (ds -> dt) resampling via Two-Thirds Power Law and 5-band H_med spatial spectrum with dynamic frequency shifting.
  2c. Photometric background shading division and perpendicular sub-pixel cross-section profiling.
  2d. Controlled synthetic ruling rendering and crossing ink preservation benchmark.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import networkx as nx
from scipy.interpolate import splprep, splev, interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter, distance_transform_edt
from scipy.signal import find_peaks, welch
from scipy.stats import skew


# =============================================================================
# 2a. Scale-Invariant Spline Smoothing & Tangent-Continuity Graph Merging
# =============================================================================

def scale_invariant_spline_smooth(
    stroke_pts: np.ndarray,
    h_med: float = 25.0,
    smoothing_factor: float = 0.05
) -> np.ndarray:
    """
    Fits scale-invariant cubic smoothing to skeleton pixel coordinates.
    The smoothing bandwidth is scaled strictly by H_med (in normalized units),
    eliminating 45/90-degree pixel staircase quantization noise before taking derivatives.
    """
    n_pts = len(stroke_pts)
    if n_pts < 4:
        return stroke_pts.copy()

    h_med = max(float(h_med), 1.0)
    pts_norm = stroke_pts / h_med

    # Filter out duplicate consecutive points
    diffs = np.diff(pts_norm, axis=0)
    dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    keep = np.concatenate([[True], dists > 1e-5])
    clean_pts = pts_norm[keep]
    n_clean = len(clean_pts)

    if n_clean < 4:
        return stroke_pts.copy()

    # Total arc length in H_med units
    clean_diffs = np.diff(clean_pts, axis=0)
    clean_dists = np.sqrt(clean_diffs[:, 0]**2 + clean_diffs[:, 1]**2)
    s_arc = np.concatenate([[0.0], np.cumsum(clean_dists)])
    total_arc = s_arc[-1]

    if total_arc < 0.05:
        return stroke_pts.copy()

    try:
        # Scale-invariant smoothing parameter s = smoothing_factor * n_clean
        # In H_med units, 1 px error is (1/h_med).
        s_param = max(smoothing_factor * n_clean * (1.0 / h_med)**2, 1e-4)
        k_order = min(3, n_clean - 1)
        tck, u = splprep([clean_pts[:, 0], clean_pts[:, 1]], u=s_arc / total_arc, s=s_param, k=k_order)
        u_fine = np.linspace(0.0, 1.0, max(n_pts, 10))
        x_smooth, y_smooth = splev(u_fine, tck)
        return np.column_stack([x_smooth * h_med, y_smooth * h_med])
    except Exception:
        # Robust fallback: Gaussian filter with sigma scaled by H_med (sigma_px = 0.08 * H_med)
        sigma_pts = max(0.08 * h_med, 1.0)
        xs = gaussian_filter1d(stroke_pts[:, 0], sigma=sigma_pts)
        ys = gaussian_filter1d(stroke_pts[:, 1], sigma=sigma_pts)
        return np.column_stack([xs, ys])


def merge_junction_tangent_continuity(
    cg: nx.Graph,
    h_med: float = 25.0,
    delta_theta_max_deg: float = 25.0
) -> List[List[Tuple[int, int]]]:
    """
    Traces skeleton branches and merges incoming/outgoing branches across degree >= 3
    junctions if their incoming/outgoing tangent angle difference is small (delta_theta < delta_theta_max_deg).
    Handles Hindi Shirorekha (horizontal header line) and English T-junctions without shattering strokes.
    """
    h_med = max(float(h_med), 1.0)
    cos_thresh = np.cos(np.radians(delta_theta_max_deg))

    # Identify key nodes (junctions degree >= 3, and endpoints degree 1)
    degrees = dict(cg.degree())
    key_nodes = {n for n, d in degrees.items() if d != 2}

    # For isolated simple cycles with no key nodes, pick one node to break cycle
    for comp in nx.connected_components(cg):
        comp_keys = set(comp) & key_nodes
        if not comp_keys and len(comp) >= 3:
            tl = min(comp, key=lambda n: (n[0], n[1]))
            key_nodes.add(tl)

    visited_edges = set()
    raw_branches = []

    # Trace all non-branching branch paths between key nodes
    for kn in sorted(key_nodes, key=lambda n: (n[0], n[1])):
        for nbr in cg.neighbors(kn):
            edge = tuple(sorted([kn, nbr]))
            if edge in visited_edges:
                continue

            path = [kn, nbr]
            visited_edges.add(edge)
            prev = kn
            curr = nbr

            while curr not in key_nodes:
                next_nbrs = [n for n in cg.neighbors(curr) if n != prev]
                if not next_nbrs:
                    break
                next_node = next_nbrs[0]
                edge_step = tuple(sorted([curr, next_node]))
                visited_edges.add(edge_step)
                path.append(next_node)
                prev = curr
                curr = next_node

            raw_branches.append(path)

    # Filter out micro-spurs (< 0.15 * H_med ending at dead-end degree 1)
    min_spur_len = max(int(0.15 * h_med), 3)
    clean_branches = []
    for b in raw_branches:
        if len(b) < min_spur_len:
            d0 = degrees.get(b[0], 0)
            d1 = degrees.get(b[-1], 0)
            if d0 == 1 or d1 == 1:
                continue
        clean_branches.append(b)

    if not clean_branches:
        return []

    # Tangent vector helper
    def get_end_tangent(pts, at_end=True, k=5):
        k = min(k, len(pts) - 1)
        if at_end:
            v = np.array([pts[-1][1] - pts[-1 - k][1], pts[-1][0] - pts[-1 - k][0]], dtype=float)
        else:
            v = np.array([pts[0][1] - pts[k][1], pts[0][0] - pts[k][0]], dtype=float)
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-4 else np.array([0.0, 0.0])

    # Build junction-continuity graph:
    # At each junction node J (degree >= 3), branches meet.
    # Pair incoming branch B_in with outgoing branch B_out maximizing cosine similarity > cos_thresh.
    strokes = [list(b) for b in clean_branches]
    changed = True
    iteration = 0
    max_gap_px = max(0.40 * h_med, 6.0)

    while changed and iteration < 40:
        changed = False
        iteration += 1

        # Map heads and tails
        heads: Dict[Tuple[int, int], List[int]] = {}
        for idx, s in enumerate(strokes):
            if s is None or len(s) < 2:
                continue
            heads.setdefault(s[0], []).append(idx)

        for i, s1 in enumerate(strokes):
            if s1 is None or len(s1) < 2:
                continue
            tail = s1[-1]
            v1_tail = get_end_tangent(s1, at_end=True, k=max(int(0.15 * h_med), 3))
            if np.linalg.norm(v1_tail) < 1e-4:
                continue

            # Find candidate branches starting at or near tail
            candidates = []
            if tail in heads:
                for j in heads[tail]:
                    if j != i and strokes[j] is not None:
                        candidates.append((j, 0.0))

            # Near-touching candidates within max_gap_px
            if not candidates:
                ty, tx = tail
                for h, indices in heads.items():
                    d = np.sqrt((ty - h[0])**2 + (tx - h[1])**2)
                    if d <= max_gap_px:
                        for j in indices:
                            if j != i and strokes[j] is not None:
                                candidates.append((j, d))

            if not candidates:
                continue

            best_j = None
            best_score = -1.0

            for j, dist in candidates:
                s2 = strokes[j]
                v2_head = -get_end_tangent(s2, at_end=False, k=max(int(0.15 * h_med), 3))
                cos_sim = float(np.dot(v1_tail, v2_head))
                score = cos_sim - (0.05 * dist / max_gap_px)

                # Through-continuation criterion: angle < delta_theta_max_deg
                if cos_sim >= cos_thresh and score > best_score:
                    best_score = score
                    best_j = j

            if best_j is not None:
                # Merge s1 and strokes[best_j]
                strokes[i] = s1 + strokes[best_j]
                strokes[best_j] = None
                changed = True
                break

        strokes = [s for s in strokes if s is not None and len(s) >= 2]

    return strokes


# =============================================================================
# 2b. Space-to-Pseudo-Time (ds -> dt) Resampling & Kinematics
# =============================================================================

def resample_stroke_space_to_time(
    stroke_pts: np.ndarray,
    h_med: float = 25.0,
    target_dt: float = 0.01,
    gamma: float = 1.0,
    eps_kappa: float = 0.05
) -> Dict[str, Any]:
    """
    Converts a spatial arc-length stroke (uniform ds) into a temporal trajectory (uniform dt)
    using the biophysical Two-Thirds Power Law: v(s) = (kappa(s) + eps)^(-1/3).
    Resampling to uniform dt BEFORE computing moments eliminates velocity skewness flip.
    """
    n_pts = len(stroke_pts)
    if n_pts < 3:
        return {
            "t": np.array([0.0]),
            "x": stroke_pts[:, 0] if n_pts else np.array([0.0]),
            "y": stroke_pts[:, 1] if n_pts else np.array([0.0]),
            "v": np.array([0.0]),
            "curvature": np.array([0.0]),
            "duration": 0.05,
            "arc_length": 0.0
        }

    h_med = max(float(h_med), 1.0)
    smooth_pts = scale_invariant_spline_smooth(stroke_pts, h_med=h_med)

    diffs = np.diff(smooth_pts, axis=0)
    seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_arc = float(s_cum[-1])

    if total_arc < 1e-4:
        return {
            "t": np.array([0.0]),
            "x": smooth_pts[:, 0],
            "y": smooth_pts[:, 1],
            "v": np.array([0.0]),
            "curvature": np.array([0.0]),
            "duration": 0.05,
            "arc_length": 0.0
        }

    # Curvature calculation in H_med units
    x_norm = smooth_pts[:, 0] / h_med
    y_norm = smooth_pts[:, 1] / h_med
    dx = np.gradient(x_norm)
    dy = np.gradient(y_norm)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    speed_sq = dx**2 + dy**2
    denom = np.maximum(speed_sq**1.5, 1e-6)
    kappa_norm = np.clip(np.abs(dx * ddy - dy * ddx) / denom, 0.0, 50.0)

    # Two-Thirds Power Law speed profile: v_rel ~ 1 / (1 + gamma * kappa^(1/3))
    v_rel = 1.0 / (1.0 + gamma * (kappa_norm**(1.0 / 3.0) + eps_kappa))

    # Biological boundary dampening (start and end acceleration glide)
    norm_s = s_cum / total_arc
    env = np.sin(np.pi * norm_s)**0.35
    v_rel = v_rel * np.maximum(env, 0.15)

    # Space-to-Pseudo-Time integration: dt = ds / v_rel
    # Integrate cumulative time: t(s) = integral ds / v_rel(s)
    # seg_lens has length n_pts - 1
    mid_v = 0.5 * (v_rel[:-1] + v_rel[1:])
    d_times = seg_lens / np.maximum(mid_v * h_med, 1e-3)
    t_cum = np.concatenate([[0.0], np.cumsum(d_times)])
    total_duration = float(t_cum[-1])

    # Resample onto uniform temporal grid dt = target_dt
    n_time_steps = max(int(np.round(total_duration / target_dt)), 8)
    t_uniform = np.linspace(0.0, total_duration, n_time_steps)

    # Interpolate x, y, v onto uniform time
    x_t = np.interp(t_uniform, t_cum, smooth_pts[:, 0] / h_med)
    y_t = np.interp(t_uniform, t_cum, smooth_pts[:, 1] / h_med)
    
    # Differentiate on uniform time
    dt_actual = t_uniform[1] - t_uniform[0] if len(t_uniform) > 1 else target_dt
    vx = np.gradient(x_t, dt_actual)
    vy = np.gradient(y_t, dt_actual)
    v_t = np.sqrt(vx**2 + vy**2)

    return {
        "t": t_uniform,
        "x": x_t,
        "y": y_t,
        "v": v_t,
        "duration": total_duration,
        "arc_length": total_arc / h_med,
        "dt": dt_actual
    }


def compute_5band_spatial_tremor_spectrum(
    strokes: List[np.ndarray],
    h_med: float = 25.0,
    v_estimated_norm: float = 2.0
) -> Dict[str, float]:
    """
    Computes 5-band spatial power spectrum normalized by H_med with dynamic frequency shifting:
      Band 1: Macro-structure (0.5 - 2 Hz equivalent)
      Band 2: Character-loop cadence (2 - 4 Hz equivalent)
      Band 3: Physiological tremor (4 - 8 Hz equivalent)
      Band 4: Rapid tremor (8 - 12 Hz equivalent)
      Band 5: Micro-noise (12 - 20 Hz equivalent)
    Dynamic velocity shifting ensures tremor bands stay aligned across fast and slow writers.
    """
    h_med = max(float(h_med), 1.0)
    v_est = max(float(v_estimated_norm), 0.5)  # H_med / s

    all_spatial_residuals = []
    for s in strokes:
        if len(s) < 8:
            continue
        diffs = np.diff(s, axis=0)
        dists = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        arc = np.concatenate([[0.0], np.cumsum(dists)])
        total_arc = arc[-1]
        if total_arc < 0.2 * h_med:
            continue

        # Uniform spatial resampling: delta_s = 0.05 * H_med
        ds_grid = 0.05 * h_med
        n_resamp = max(int(np.round(total_arc / ds_grid)), 10)
        u_space = np.linspace(0.0, total_arc, n_resamp)

        x_s = np.interp(u_space, arc, s[:, 0] / h_med)
        y_s = np.interp(u_space, arc, s[:, 1] / h_med)

        # High-pass spatial residual (subtract smooth trajectory)
        x_trend = gaussian_filter1d(x_s, sigma=max(int(0.3 * h_med / ds_grid), 2))
        y_trend = gaussian_filter1d(y_s, sigma=max(int(0.3 * h_med / ds_grid), 2))
        res = np.sqrt((x_s - x_trend)**2 + (y_s - y_trend)**2)
        all_spatial_residuals.append(res)

    if not all_spatial_residuals:
        return {"band1_05_2hz": 0.0, "band2_2_4hz": 0.0, "band3_4_8hz": 0.0, "band4_8_12hz": 0.0, "band5_12_20hz": 0.0}

    # Concatenate residuals with spatial sampling rate fs_space = 1.0 / ds_norm = 1.0 / 0.05 = 20.0 cycles/H_med
    fs_space = 20.0
    sig = np.concatenate(all_spatial_residuals)
    sig = sig - np.mean(sig)

    nperseg = min(len(sig), 128)
    freqs_space, psd = welch(sig, fs=fs_space, nperseg=nperseg)

    # Convert spatial frequency to equivalent temporal frequency via f_temp = k_space * v_est
    freqs_temp = freqs_space * v_est

    # Integrate power in each band
    b1_mask = (freqs_temp >= 0.5) & (freqs_temp < 2.0)
    b2_mask = (freqs_temp >= 2.0) & (freqs_temp < 4.0)
    b3_mask = (freqs_temp >= 4.0) & (freqs_temp < 8.0)   # Canonical 4-8 Hz tremor band
    b4_mask = (freqs_temp >= 8.0) & (freqs_temp < 12.0)
    b5_mask = (freqs_temp >= 12.0) & (freqs_temp <= 20.0)
    tot_mask = (freqs_temp >= 0.5) & (freqs_temp <= 20.0)

    tot_p = float(np.sum(psd[tot_mask])) + 1e-9
    return {
        "band1_05_2hz": float(np.sum(psd[b1_mask])) / tot_p,
        "band2_2_4hz": float(np.sum(psd[b2_mask])) / tot_p,
        "band3_4_8hz": float(np.sum(psd[b3_mask])) / tot_p,
        "band4_8_12hz": float(np.sum(psd[b4_mask])) / tot_p,
        "band5_12_20hz": float(np.sum(psd[b5_mask])) / tot_p,
    }


# =============================================================================
# 2c. Photometric Shading Division & Perpendicular Sub-pixel Cross-Sections
# =============================================================================

def extract_photometric_cross_sections(
    gray_image: np.ndarray,
    strokes: List[np.ndarray],
    h_med: float = 25.0
) -> Dict[str, float]:
    """
    Normalizes local background via shading division (divide grayscale by large Gaussian background),
    and measures perpendicular sub-pixel cross-sections along recovered strokes:
      - Width: Full-Width at Half-Maximum (FWHM)
      - Integrated Optical Density (absorbance: -log10(I / I_bg))
      - Downstroke vs. upstroke width and darkness ratios.
    """
    h_med = max(float(h_med), 1.0)
    h_img, w_img = gray_image.shape
    gray_flt = gray_image.astype(np.float64)

    # Local background estimation via large Gaussian filter (sigma = 2.0 * H_med)
    bg_est = gaussian_filter(gray_flt, sigma=max(2.0 * h_med, 20.0))
    bg_est = np.maximum(bg_est, 10.0)

    # Shading division
    norm_img = np.clip(gray_flt / bg_est, 0.0, 1.0)
    absorbance_map = -np.log10(np.clip(norm_img, 0.01, 1.0))

    all_widths = []
    all_absorbance = []
    downstroke_widths, upstroke_widths = [], []
    downstroke_darkness, upstroke_darkness = [], []

    profile_radius = max(int(0.35 * h_med), 4)

    for s in strokes:
        if len(s) < 3:
            continue
        diffs = np.diff(s, axis=0)
        # Determine downstroke vs upstroke
        dy_overall = s[-1, 1] - s[0, 1]
        is_downstroke = (dy_overall > 0.15 * h_med)
        is_upstroke = (dy_overall < -0.15 * h_med)

        for i in range(1, len(s) - 1):
            px, py = s[i]
            # Tangent and normal
            tx = s[i + 1, 0] - s[i - 1, 0]
            ty = s[i + 1, 1] - s[i - 1, 1]
            n_len = np.sqrt(tx**2 + ty**2)
            if n_len < 1e-4:
                continue
            # Normal: perpendicular to tangent
            nx_vec = -ty / n_len
            ny_vec = tx / n_len

            # Sample cross-section along normal
            r_vals = np.linspace(-profile_radius, profile_radius, 2 * profile_radius + 1)
            cx = px + r_vals * nx_vec
            cy = py + r_vals * ny_vec

            valid_sample = (cx >= 0) & (cx < w_img - 1) & (cy >= 0) & (cy < h_img - 1)
            if np.sum(valid_sample) < len(r_vals) * 0.7:
                continue

            # Bilinear interpolation
            ix = np.clip(cx, 0, w_img - 1).astype(int)
            iy = np.clip(cy, 0, h_img - 1).astype(int)
            profile_abs = absorbance_map[iy, ix]

            # FWHM of absorbance peak
            peak_abs = np.max(profile_abs)
            if peak_abs > 0.08:
                half_max = 0.5 * peak_abs
                above_half = profile_abs >= half_max
                fwhm = float(np.sum(above_half)) * (profile_radius * 2 / len(r_vals)) / h_med
                mean_abs_sample = float(np.mean(profile_abs[above_half]))

                all_widths.append(fwhm)
                all_absorbance.append(mean_abs_sample)

                if is_downstroke:
                    downstroke_widths.append(fwhm)
                    downstroke_darkness.append(mean_abs_sample)
                elif is_upstroke:
                    upstroke_widths.append(fwhm)
                    upstroke_darkness.append(mean_abs_sample)

    w_mean = float(np.mean(all_widths)) if all_widths else 0.12
    w_std = float(np.std(all_widths)) if len(all_widths) > 2 else 0.02
    abs_mean = float(np.mean(all_absorbance)) if all_absorbance else 0.35

    down_up_width_ratio = (
        float(np.mean(downstroke_widths)) / max(float(np.mean(upstroke_widths)), 1e-3)
        if downstroke_widths and upstroke_widths else 1.0
    )
    down_up_dark_ratio = (
        float(np.mean(downstroke_darkness)) / max(float(np.mean(upstroke_darkness)), 1e-3)
        if downstroke_darkness and upstroke_darkness else 1.0
    )

    return {
        "optical_width_mean": w_mean,
        "optical_width_std": w_std,
        "optical_absorbance_mean": abs_mean,
        "downstroke_upstroke_width_ratio": down_up_width_ratio,
        "downstroke_upstroke_darkness_ratio": down_up_dark_ratio,
    }

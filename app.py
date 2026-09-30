"""
Dysgraphia Feature Extraction & Kinematics Studio - Web Application Server
Run with: C:\\Users\\embar\\miniconda3\\envs\\ai_env\\python.exe app.py
Serves the modern glassmorphic web testing ground on http://127.0.0.1:7860
Provides objective 20D feature extraction, interactive point-by-point velocity inspection on handwriting,
toggable kinematic waveforms (v, a, jerk, curvature, NVI), and a 5-stage extraction methodology walkthrough.
"""

import os
import io
import json
import base64
import glob
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from src.pipeline import DysgraphiaFeaturePipeline
from src.branch_b.kinematics import estimate_stroke_velocity, estimate_stroke_pressure_proxy

# Global pipeline instance
pipeline = DysgraphiaFeaturePipeline(compute_kinematics=True)

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")


def render_overlay_b64(orig_img: Image.Image, results: Dict[str, Any]) -> str:
    """
    Renders the BHK explainability overlay:
    - Green bounding boxes around valid letters
    - Cyan centroids
    - Coral fitted baseline wander line
    """
    img_rgb = orig_img.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)
    text_lines = results.get("visual_artifacts", {}).get("text_lines", [])

    for line in text_lines:
        line_pts_x = []
        line_bottoms_y = []

        for c in line.components:
            draw.rectangle([c.x_min, c.y_min, c.x_max, c.y_max], outline="#10b981", width=2)
            cx, cy = int(round(c.x_center)), int(round(c.y_center))
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill="#38bdf8", outline="#0284c7")
            line_pts_x.append(cx)
            line_bottoms_y.append(c.y_bottom)

        if len(line_pts_x) >= 3 and (max(line_pts_x) - min(line_pts_x) > 20):
            try:
                poly = np.polyfit(line_pts_x, line_bottoms_y, 1)
                x_start = int(min(line_pts_x))
                x_end = int(max(line_pts_x))
                y_start = int(np.polyval(poly, x_start))
                y_end = int(np.polyval(poly, x_end))
                draw.line([(x_start, y_start), (x_end, y_end)], fill="#f43f5e", width=2)
            except Exception:
                pass

    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode("ascii")


def render_binary_mask_b64(mask: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.imshow(mask, cmap="gray")
    ax.axis("off")
    ax.set_title("Stage 1: Illumination-Normalized Binary Ink Mask", color="#f8fafc", fontsize=11, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def render_skeleton_b64(skeleton: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.imshow(skeleton, cmap="gray")
    ax.axis("off")
    ax.set_title("Stage 2: 1-Pixel Morphological Centerline Spine (Zhang-Suen)", color="#f8fafc", fontsize=11, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def render_strokes_graph_b64(strokes: List[np.ndarray], orig_w: int, orig_h: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0b0f19")
    ax.set_facecolor("#131b2e")
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for idx, s in enumerate(strokes):
        if len(s) < 2:
            continue
        c = colors[idx % len(colors)]
        ax.plot(s[:, 0], s[:, 1], color=c, lw=2.2)
        ax.plot(s[0, 0], s[0, 1], "o", color=c, markersize=3.5)
    ax.set_xlim(0, orig_w)
    ax.set_ylim(orig_h, 0)
    ax.axis("off")
    ax.set_title(f"Stage 4: Topological Recovered Strokes ({len(strokes)} Motor Programs)", color="#f8fafc", fontsize=11, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def render_velocity_heatmap_b64(strokes: List[np.ndarray], orig_w: int, orig_h: int, h_med: float = 25.0) -> str:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#0b0f19")
    ax.set_facecolor("#131b2e")
    all_v = []
    stroke_velocities = []

    for s in strokes:
        if len(s) >= 3:
            _, v, _ = estimate_stroke_velocity(s, h_med=h_med)
            all_v.extend(v)
            stroke_velocities.append(v)
        else:
            stroke_velocities.append(np.array([]))

    v_min, v_max = (np.percentile(all_v, 5), np.percentile(all_v, 95)) if all_v else (0.2, 4.0)
    if v_max <= v_min:
        v_max = v_min + 2.0

    norm = plt.Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.cm.turbo

    for idx, s in enumerate(strokes):
        if len(s) < 3 or idx >= len(stroke_velocities):
            continue
        v = stroke_velocities[idx]
        if len(v) != len(s):
            continue
        for j in range(len(s) - 1):
            seg_v = (v[j] + v[j+1]) / 2.0
            ax.plot([s[j, 0], s[j+1, 0]], [s[j, 1], s[j+1, 1]], color=cmap(norm(seg_v)), lw=2.4)

    ax.set_xlim(0, orig_w)
    ax.set_ylim(orig_h, 0)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.05)
    cbar.set_label("Reconstructed Instantaneous Velocity (H_med/s)", color="#f8fafc", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)

    ax.set_title("Stage 5: Centerline Kinematic Velocity Heatmap v(s)", color="#f8fafc", fontsize=11, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def prepare_point_kinematics(strokes: List[np.ndarray], dist_map: np.ndarray = None, h_med: float = 25.0, nominal_dt: float = 0.01) -> List[List[Any]]:
    """
    Extracts compact point-level kinematics for interactive hover inspection:
    Returns list of [x, y, v (H_med/s), a (H_med/s²), kappa_norm, stroke_id, is_nvi,
                     ink_width_px, pressure_proxy, stroke_len_px, stroke_len_hmed, arc_pos_px]
    """
    pts_data = []
    h_med = max(float(h_med), 1.0)
    h_dm, w_dm = (dist_map.shape if dist_map is not None else (0, 0))

    for s_idx, s in enumerate(strokes):
        if len(s) < 3:
            continue
        _, v, kappa = estimate_stroke_velocity(s, h_med=h_med)
        acc = np.gradient(v, nominal_dt)
        pks, _ = find_peaks(v) if len(v) >= 5 else ([], None)
        trgs, _ = find_peaks(-v) if len(v) >= 5 else ([], None)
        pks_set = set(pks)
        trgs_set = set(trgs)

        # Cumulative arc length along stroke
        diffs = np.diff(s, axis=0)
        seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        s_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len_px = float(s_arc[-1])
        total_len_hmed = float(total_len_px / h_med)

        for j in range(len(s)):
            is_nvi = int(j in pks_set or j in trgs_set)
            px_x = float(s[j, 0])
            px_y = float(s[j, 1])

            if dist_map is not None and h_dm > 0 and w_dm > 0:
                ix = int(np.clip(np.round(px_x), 0, w_dm - 1))
                iy = int(np.clip(np.round(px_y), 0, h_dm - 1))
                ink_w = float(2.0 * dist_map[iy, ix])
                press = float(ink_w / h_med)
            else:
                ink_w = 2.0
                press = float(2.0 / h_med)

            pts_data.append([
                round(px_x, 1),
                round(px_y, 1),
                round(float(v[j]), 2),
                round(float(acc[j]), 2),
                round(float(kappa[j]), 3),
                s_idx + 1,
                is_nvi,
                round(ink_w, 1),
                round(press, 3),
                round(total_len_px, 1),
                round(total_len_hmed, 2),
                round(float(s_arc[j]), 1)
            ])
    return pts_data


def prepare_waveform_series(results: Dict[str, Any], nominal_dt: float = 0.01) -> Dict[str, Any]:
    """
    Extracts time-series arrays for client-side togglable waveform chart:
    - time, velocity, pressure, acceleration, jerk, curvature, peaks, troughs
    """
    v = results.get("visual_artifacts", {}).get("velocity_profile", np.array([]))
    p = results.get("visual_artifacts", {}).get("pressure_profile", np.array([]))
    if len(v) == 0:
        v = np.zeros(50)
    if len(p) == 0:
        p = np.zeros(len(v))

    # Downsample if very long for fluid 60fps rendering in browser (max 1500 points)
    step = max(1, len(v) // 1500)
    v_sub = v[::step]
    if len(p) >= len(v):
        p_sub = p[::step]
    else:
        p_sub = np.resize(p, len(v_sub)) if len(p) > 0 else np.zeros(len(v_sub))

    dt = nominal_dt * step

    t = (np.arange(len(v_sub)) * dt).tolist()
    acc = np.gradient(v_sub, dt)
    jerk = np.gradient(acc, dt)

    # Estimate curvature approximation from velocity
    gamma = 1.2
    v_norm = np.maximum(v_sub, 1.0)
    kappa_approx = np.clip((np.maximum(100.0 / v_norm - 1.0, 0.0) / gamma)**3, 0.0, 5.0)

    pks, _ = find_peaks(v_sub) if len(v_sub) >= 5 else ([], None)
    trgs, _ = find_peaks(-v_sub) if len(v_sub) >= 5 else ([], None)

    return {
        "time": [round(float(val), 3) for val in t],
        "velocity": [round(float(val), 1) for val in v_sub],
        "pressure": [round(float(val), 3) for val in p_sub],
        "acceleration": [round(float(val), 1) for val in acc],
        "jerk": [round(float(val), 1) for val in jerk],
        "curvature": [round(float(val), 3) for val in kappa_approx],
        "peaks": [int(pk) for pk in pks],
        "troughs": [int(tr) for tr in trgs],
    }


def format_biomarkers_20d(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Builds comprehensive 20D feature items with formulas, units, estimated tags, and per-feature confidence."""
    bhk = results.get("bhk_metrics", {})
    kin = results.get("kinematic_metrics", {})
    pfc = results.get("per_feature_confidence", {})

    return [
        {
            "domain": "Spatial BHK",
            "feature_id": "bhk_size_covariance",
            "name": "BHK #1 & #8: Letter Size CoV",
            "val": bhk.get("size_covariance_score", 0.0),
            "unit": "CV",
            "confidence": pfc.get("bhk_size_covariance", 1.0),
            "formula": "0.6 * CV(Height) + 0.4 * CV(Area)",
            "meaning": "Quantifies inconsistency in letter dimensions and area across words and lines."
        },
        {
            "domain": "Spatial BHK",
            "feature_id": "bhk_height_ratio_consistency",
            "name": "BHK #9: Relative Height Ratio (IQR)",
            "val": bhk.get("height_iqr_ratio", 0.0),
            "unit": "IQR/median",
            "confidence": pfc.get("bhk_height_ratio_consistency", 1.0),
            "formula": "IQR(Height) / Median(Height)",
            "meaning": "Measures vertical dispersion between ascenders, descenders, and body x-height."
        },
        {
            "domain": "Spatial BHK",
            "feature_id": "bhk_baseline_drift",
            "name": "BHK #3: Baseline Drift & Wander",
            "val": bhk.get("baseline_drift_score", 0.0),
            "unit": "1/H_med",
            "confidence": pfc.get("bhk_baseline_drift", 1.0),
            "formula": "|Slope| + 2 * RMSE(Residuals)",
            "meaning": "Captures macro line tilt and micro baseline wobble along letter bottoms."
        },
        {
            "domain": "Spatial BHK",
            "feature_id": "bhk_spacing_entropy",
            "name": "BHK #4: Spacing Entropy",
            "val": bhk.get("spacing_entropy", 0.0),
            "unit": "nats",
            "confidence": pfc.get("bhk_spacing_entropy", 1.0),
            "formula": "-Σ p_i * ln(p_i) of gaps / H_med",
            "meaning": "Shannon entropy measuring irregularity and lack of rhythm in horizontal letter gaps."
        },
        {
            "domain": "Spatial BHK",
            "feature_id": "bhk_stroke_width_variance",
            "name": "BHK #6: Stroke Width CoV",
            "val": bhk.get("stroke_width_cv", 0.0),
            "unit": "CV",
            "confidence": pfc.get("bhk_stroke_width_variance", 1.0),
            "formula": "Std(Width) / Mean(Width) via EDT",
            "meaning": "Measures pen tremor and down-force variability. Note: sensitive to pen nib type."
        },
        {
            "domain": "Spatial BHK",
            "feature_id": "bhk_telescoping_overlap",
            "name": "BHK #7: Telescoping Overlap",
            "val": bhk.get("telescoping_score", 0.0),
            "unit": "ratio",
            "confidence": pfc.get("bhk_telescoping_overlap", 1.0),
            "formula": "Collision Count + Mean Overlap Depth",
            "meaning": "Frequency of horizontal letter intrusions where adjacent characters collide."
        },
        {
            "domain": "Spatial BHK",
            "feature_id": "bhk_acute_turns",
            "name": "BHK #5: Acute Directional Turns",
            "val": bhk.get("acute_turns_score", 0.0),
            "unit": "turns/H_med",
            "confidence": pfc.get("bhk_acute_turns", 1.0),
            "formula": "Points where |Δθ| ≥ 110° / H_med",
            "meaning": "Quantifies jagged high-curvature direction reversals reflecting fine motor stiffness."
        },
        {
            "domain": "Spatial BHK",
            "feature_id": "bhk_left_margin_drift",
            "name": "BHK #2: Left Margin Alignment Drift",
            "val": bhk.get("left_margin_score", 0.0),
            "unit": "1/H_med",
            "confidence": pfc.get("bhk_left_margin_drift", 1.0),
            "formula": "|Slope(x_start)| + Std(x_start) / H_med",
            "meaning": "Evaluates line-to-line alignment along the left vertical page boundary."
        },
        {
            "domain": "Spatial BHK",
            "feature_id": "bhk_line_collisions",
            "name": "BHK #13: Inter-Line Collisions",
            "val": bhk.get("line_collision_score", 0.0),
            "unit": "ratio",
            "confidence": pfc.get("bhk_line_collisions", 1.0),
            "formula": "Line Overlap Ratio + CV(Line Distances)",
            "meaning": "Frequency with which descenders crash into ascenders of preceding lines."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_nvi_per_stroke",
            "name": "NVI per Stroke (Fluency Proxy, ESTIMATED)",
            "val": kin.get("nvi_per_stroke", 0.0),
            "unit": "inv/stroke (EST)",
            "confidence": pfc.get("kin_nvi_per_stroke", 0.0),
            "formula": "Total Inversions / Recovered Strokes",
            "meaning": "Primary scale-invariant fluency estimate: velocity inversions per physical motor stroke."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_nvi_per_100px",
            "name": "NVI per 100px Arc Length (ESTIMATED)",
            "val": kin.get("nvi_per_100px", 0.0),
            "unit": "inv/100px (EST)",
            "confidence": pfc.get("kin_nvi_per_100px", 0.0),
            "formula": "Total Inversions / Total Arc Length * 100",
            "meaning": "Spatial density of velocity reversals along physical ink trajectories."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_nvi_rate",
            "name": "NVI Rate (Frequency of Hesitations, ESTIMATED)",
            "val": kin.get("nvi_rate", 0.0),
            "unit": "Hz (EST)",
            "confidence": pfc.get("kin_nvi_rate", 0.0),
            "formula": "Total Inversions / Total Duration",
            "meaning": "Temporal frequency estimate of speed reversals and motor planning interruptions."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_dimensionless_jerk",
            "name": "Flash & Hogan Dimensionless Jerk (ESTIMATED)",
            "val": kin.get("dimensionless_jerk", 0.0),
            "unit": "dimensionless (EST)",
            "confidence": pfc.get("kin_dimensionless_jerk", 0.0),
            "formula": "(T^5 / L^2) * ∫ (da/dt)^2 dt",
            "meaning": "Coordinate-free neuromotor smoothness proxy; elevated values indicate rough motor execution."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_tremor_index_4_8hz",
            "name": "4–8 Hz Tremor Power Ratio (ESTIMATED)",
            "val": round(kin.get("tremor_index_4_8hz", 0.0) * 100, 1),
            "unit": "% (EST)",
            "confidence": pfc.get("kin_tremor_index_4_8hz", 0.0),
            "formula": "PSD(4-8 Hz) / PSD(0.5-20 Hz) via Welch",
            "meaning": "Proportion of motor power consumed by involuntary 4–8 Hz physiological micro-tremor."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_velocity_skewness",
            "name": "Velocity Skewness (Asymmetry Tail, ESTIMATED)",
            "val": kin.get("velocity_skewness", 0.0),
            "unit": "dimensionless (EST)",
            "confidence": pfc.get("kin_velocity_skewness", 0.0),
            "formula": "E[(v - μ)^3] / σ^3",
            "meaning": "Quantifies asymmetry between acceleration burst phase and extended deceleration glide."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_mean_velocity",
            "name": "Estimated Mean Velocity (ESTIMATED)",
            "val": kin.get("mean_velocity", 0.0),
            "unit": "H_med/s (EST)",
            "confidence": pfc.get("kin_mean_velocity", 0.0),
            "formula": "Average v(s) via Two-Thirds Power Law",
            "meaning": "Relative execution speed proxy across continuous reconstructed trajectory strokes."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_peak_velocity",
            "name": "Estimated Peak Velocity (ESTIMATED)",
            "val": kin.get("peak_velocity", 0.0),
            "unit": "H_med/s (EST)",
            "confidence": pfc.get("kin_peak_velocity", 0.0),
            "formula": "Max(v) across ballistic straight segments",
            "meaning": "Peak ballistic impulse velocity estimate achieved during straight stroke segments."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_pen_lift_count",
            "name": "Pen Lift Count (Recovered Strokes)",
            "val": kin.get("pen_lift_count", 0),
            "unit": "strokes",
            "confidence": pfc.get("kin_pen_lift_count", 0.0),
            "formula": "Count of Continuous Recovered Paths",
            "meaning": "Reflects motor program segmentation and topological stroke connectivity."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_mean_stroke_length",
            "name": "Mean Stroke Arc Length (ESTIMATED)",
            "val": kin.get("mean_stroke_length", 0.0),
            "unit": "px (EST)",
            "confidence": pfc.get("kin_mean_stroke_length", 0.0),
            "formula": "Total Arc Length / Stroke Count",
            "meaning": "Average pixel length of unbroken continuous recovered motor trajectories."
        },
        {
            "domain": "Biophysical Kinematics (ESTIMATED)",
            "feature_id": "kin_jerk_metric",
            "name": "Total Jerk Metric (ESTIMATED)",
            "val": kin.get("jerk_metric", 0.0),
            "unit": "H_med/s³ (EST)",
            "confidence": pfc.get("kin_jerk_metric", 0.0),
            "formula": "∫ (da/dt)² dt",
            "meaning": "Integrated trajectory jerk estimate; sensitive to skeleton spurs and noise."
        },
        {
            "domain": "Biophysical Pressure & Ink",
            "feature_id": "kin_ink_width_ratio_mean",
            "name": "Optical Stylus Pressure Proxy (Mean)",
            "val": kin.get("ink_width_ratio_mean", 0.0),
            "unit": "W/H_med (EST)",
            "confidence": pfc.get("bhk_stroke_width_variance", 0.8),
            "formula": "Mean(2 * EDT / H_med) along recovered strokes",
            "meaning": "Scale-invariant optical proxy for pen down-force; thicker strokes correlate with heavier stylus pressure."
        },
        {
            "domain": "Biophysical Pressure & Ink",
            "feature_id": "kin_ink_width_ratio_std",
            "name": "Optical Pressure Variability (Std)",
            "val": kin.get("ink_width_ratio_std", 0.0),
            "unit": "W/H_med (EST)",
            "confidence": pfc.get("bhk_stroke_width_variance", 0.8),
            "formula": "Std(2 * EDT / H_med) along recovered strokes",
            "meaning": "Variability in pen down-force; elevated in children with dysgraphic grip instability."
        },
        {
            "domain": "Biophysical Pressure & Ink",
            "feature_id": "bhk_mean_stroke_width_px",
            "name": "Mean Ink Stroke Width (Physical)",
            "val": bhk.get("stroke_width_mean", 0.0),
            "unit": "px",
            "confidence": pfc.get("bhk_stroke_width_variance", 0.9),
            "formula": "Mean(2 * EDT) sampled along skeleton",
            "meaning": "Absolute physical stroke thickness in pixels across the handwriting image."
        },
    ]


class StudioHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Dysgraphia Feature Extraction Studio."""

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]} {args[1]}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(WEB_DIR, "index.html"), "text/html")
        elif path == "/style.css":
            self._serve_file(os.path.join(WEB_DIR, "style.css"), "text/css")
        elif path == "/app.js":
            self._serve_file(os.path.join(WEB_DIR, "app.js"), "application/javascript")
        elif path == "/api/demo":
            self._handle_demo(parsed.query)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/analyze":
            self._handle_analyze()
        else:
            self.send_error(404, "Not Found")

    def _serve_file(self, filepath: str, content_type: str):
        if not os.path.exists(filepath):
            self.send_error(404, "File Not Found")
            return
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_demo(self, query_str: str):
        params = urllib.parse.parse_qs(query_str)
        demo_type = params.get("type", ["malay_pd"])[0]

        file_map = {
            "malay_pd": "Datasets/DATASET DYSGRAPHIA HANDWRITING/Potential Dysgraphia/PD (10).jpg",
            "malay_lpd": "Datasets/DATASET DYSGRAPHIA HANDWRITING/Low Potential Dysgraphia/LPD (10).jpg",
            "drotar_dys": "Datasets/reconstructed_dataset/by_task/task_5_leto/dysgraphic/user_00006_task_5_leto.png",
            "drotar_ctrl": "Datasets/reconstructed_dataset/by_task/task_5_leto/control/user_00050_task_5_leto.png",
            "drotar_sentence": "Datasets/reconstructed_dataset/by_task/task_8_sentence/control/user_00050_task_8_sentence.png",
        }

        rel_path = file_map.get(demo_type)
        if not rel_path or not os.path.exists(rel_path):
            candidates = glob.glob("Datasets/**/*.jpg", recursive=True) or glob.glob("Datasets/**/*.png", recursive=True)
            if candidates:
                rel_path = candidates[0]
            else:
                self.send_error(404, "No demo file available")
                return

        with open(rel_path, "rb") as f:
            content = f.read()

        content_type = "image/png" if rel_path.endswith(".png") else "image/jpeg"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_analyze(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json({"error": "Empty request body"}, 400)
                return

            body = self.rfile.read(content_length)
            content_type = self.headers.get("Content-Type", "")

            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[1].encode()
                parts = body.split(b"--" + boundary)
                image_bytes = None
                for part in parts:
                    if b"Content-Type: image" in part or b"\r\n\r\n" in part:
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            chunk = part[header_end + 4:].rstrip(b"\r\n")
                            if len(chunk) > 100:
                                image_bytes = chunk
                                break
                if image_bytes is None:
                    image_bytes = body
            else:
                image_bytes = body

            # Load into PIL
            orig_pil = Image.open(io.BytesIO(image_bytes))
            orig_w, orig_h = orig_pil.size

            # Run 20D feature extraction pipeline
            results = pipeline.extract(orig_pil)

            strokes = results.get("visual_artifacts", {}).get("recovered_strokes", [])
            mask = results.get("visual_artifacts", {}).get("binary_mask", np.zeros((100, 100)))
            skeleton = results.get("visual_artifacts", {}).get("skeleton", np.zeros((100, 100)))

            h_med = float(results.get("image_scale_metadata", {}).get("h_med_px", 25.0))

            # Stage Visuals
            stage1_b64 = render_binary_mask_b64(mask)
            stage2_b64 = render_skeleton_b64(skeleton)
            stage3_b64 = render_overlay_b64(orig_pil, results)
            stage4_b64 = render_strokes_graph_b64(strokes, orig_w, orig_h)
            stage5_b64 = render_velocity_heatmap_b64(strokes, orig_w, orig_h, h_med=h_med)

            dist_map = results.get("visual_artifacts", {}).get("dist_map", None)
            if dist_map is None and np.any(mask):
                from scipy.ndimage import distance_transform_edt
                dist_map = distance_transform_edt(mask)

            # Interactive point kinematics & waveform series
            pts_data = prepare_point_kinematics(strokes, dist_map=dist_map, h_med=h_med)
            waveform_data = prepare_waveform_series(results)
            biomarkers_20d = format_biomarkers_20d(results)

            # Text line bounding boxes & baselines for client-side drawing
            text_lines = results.get("visual_artifacts", {}).get("text_lines", [])
            boxes_data = []
            baselines_data = []
            for line in text_lines:
                lx, ly = [], []
                for c in line.components:
                    boxes_data.append([c.x_min, c.y_min, c.x_max, c.y_max, round(c.x_center, 1), round(c.y_center, 1)])
                    lx.append(c.x_center)
                    ly.append(c.y_bottom)
                if len(lx) >= 3 and (max(lx) - min(lx) > 20):
                    try:
                        poly = np.polyfit(lx, ly, 1)
                        x0 = int(min(lx))
                        x1 = int(max(lx))
                        y0 = int(np.polyval(poly, x0))
                        y1 = int(np.polyval(poly, x1))
                        baselines_data.append([x0, y0, x1, y1])
                    except Exception:
                        pass

            kin = results.get("kinematic_metrics", {})
            bhk = results.get("bhk_metrics", {})

            strokes_data = results.get("strokes", [])
            stroke_lengths_px = [s.get("arc_length_px", 0.0) for s in strokes_data]
            stroke_lengths_hmed = [s.get("arc_length_h_med", 0.0) for s in strokes_data]
            mean_stk_len_px = float(np.mean(stroke_lengths_px)) if stroke_lengths_px else 0.0
            mean_stk_len_hmed = float(np.mean(stroke_lengths_hmed)) if stroke_lengths_hmed else 0.0
            total_arc_len_px = float(np.sum(stroke_lengths_px)) if stroke_lengths_px else 0.0

            mean_ink_w_px = float(bhk.get("stroke_width_mean", 0.0))
            mean_ink_w_hmed = float(mean_ink_w_px / max(h_med, 1.0))
            ink_w_cv = float(bhk.get("stroke_width_cv", 0.0))

            mean_press = float(kin.get("ink_width_ratio_mean", 0.0))
            std_press = float(kin.get("ink_width_ratio_std", 0.0))

            response_payload = {
                "status": "success",
                "dimensions": {"width": orig_w, "height": orig_h},
                "summary": {
                    "strokes_count": len(strokes),
                    "mean_stroke_length_px": round(mean_stk_len_px, 1),
                    "mean_stroke_length_hmed": round(mean_stk_len_hmed, 2),
                    "total_arc_length_px": round(total_arc_len_px, 1),
                    "mean_ink_width_px": round(mean_ink_w_px, 2),
                    "mean_ink_width_hmed": round(mean_ink_w_hmed, 3),
                    "ink_width_cv": round(ink_w_cv, 3),
                    "mean_ink_width_ratio": round(mean_press, 3),
                    "ink_width_ratio_std": round(std_press, 3),
                    "mean_velocity": round(float(kin.get("mean_velocity", 0.0)), 2),
                    "peak_velocity": round(float(kin.get("peak_velocity", 0.0)), 2),
                    "nvi_per_stroke": round(float(kin.get("nvi_per_stroke", 0.0)), 2),
                    "tremor_percent": round(float(kin.get("tremor_index_4_8hz", 0.0)) * 100, 1),
                    "size_covariance": round(float(bhk.get("size_covariance_score", 0.0)), 2),
                    "line_collisions": round(float(bhk.get("line_collision_score", 0.0)), 2),
                    "total_points": len(pts_data)
                },
                "strokes_data": strokes_data,
                "stages": {
                    "stage1_binarization": stage1_b64,
                    "stage2_skeleton": stage2_b64,
                    "stage3_bhk": stage3_b64,
                    "stage4_strokes": stage4_b64,
                    "stage5_heatmap": stage5_b64,
                },
                "point_kinematics": pts_data,
                "waveform": waveform_data,
                "spatial_boxes": boxes_data,
                "spatial_baselines": baselines_data,
                "biomarkers_20d": biomarkers_20d,
                "export_payload": results.get("export_payload", {}),
                "image_scale_metadata": results.get("image_scale_metadata", {}),
                "quality_flags": results.get("quality_flags", {}),
                "per_feature_confidence": results.get("per_feature_confidence", {}),
                "extraction_quality_flags": results.get("quality_flags", {}),
                "metadata": results.get("metadata", {})
            }

            self._send_json(response_payload, 200)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 7860):
    server = ThreadingHTTPServer(("127.0.0.1", port), StudioHandler)
    print("\n" + "=" * 70)
    print(">> Dysgraphia Feature Extraction & Kinematics Studio Active!")
    print(f">> Open in your web browser: http://127.0.0.1:{port}")
    print("=" * 70 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    run_server(7860)

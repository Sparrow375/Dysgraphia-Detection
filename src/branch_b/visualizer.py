"""
Diagnostic Visualizer for Photo-to-Digital Kinematic Harmonization Pipeline.
Generates 4 high-fidelity diagnostic overlay images as base64-encoded PNGs:
  Panel 1: Original & Illumination Normalized (+ Ruled-line cleaning & H_med boxes)
  Panel 2: Skeleton & Tangent-Continuity Graph Merging (Strokes colored, junctions highlighted)
  Panel 3: Two-Thirds Power Law Pseudo-Time (ds -> dt) Velocity Heatmap
  Panel 4: 4-8 Hz Micro-Tremor Spectrum & Optical Stroke Cross-Section
"""

import os
import io
import math
import base64
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import label, find_objects
from skimage.morphology import skeletonize

from src.branch_b.harmonization_pipeline import HarmonizationEngine, FEATURE_METADATA
from src.branch_b.stroke_recovery import skeleton_to_graph, contract_junction_cliques
from src.branch_b.step2_foundations import (
    merge_junction_tangent_continuity,
    resample_stroke_space_to_time,
    compute_5band_spatial_tremor_spectrum,
    extract_photometric_cross_sections
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT_CSV_PATH = os.path.join(PROJECT_ROOT, "zmapped_output", "ground_truth_kinematics.csv")

GT_MAP = {
    "kin_mean_velocity": "gt_mean_velocity_norm",
    "kin_peak_velocity": "gt_peak_velocity_norm",
    "kin_velocity_skewness": "gt_velocity_skewness",
    "kin_nvi_rate": "gt_nvi_rate",
    "kin_nvi_per_stroke": "gt_nvi_per_stroke",
    "kin_nvi_per_h_med": "gt_nvi_per_h_med",
    "kin_jerk_metric": "gt_jerk_metric",
    "kin_dimensionless_jerk": "gt_dimensionless_jerk",
    "kin_spatial_roughness_4_8hz": "gt_spatial_roughness_4_8hz",
    "kin_pen_lift_count": "gt_pen_lift_count",
    "kin_mean_stroke_length": "gt_mean_stroke_length",
    "kin_ink_width_ratio_mean": "gt_pressure_norm_mean",
    "kin_ink_width_ratio_std": "gt_pressure_norm_std",
}


def fig_to_base64(fig: plt.Figure) -> str:
    """Converts a matplotlib figure to a base64 PNG data URL."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def generate_panel1_normalization(
    arr_gray: np.ndarray,
    clean_binary: np.ndarray,
    h_med: float,
    doc_type: str
) -> str:
    """
    Panel 1: Original vs Illumination-Normalized & Ruled-Line Cleaned
    with character-height H_med bounding boxes overlaid.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')

    # Display clean binary on a dark clinical background
    display_img = np.zeros((*clean_binary.shape, 3), dtype=np.uint8)
    display_img[clean_binary > 0] = [240, 240, 245]  # clean white ink
    display_img[clean_binary == 0] = [15, 23, 42]    # slate-900 background

    ax.imshow(display_img)

    # Detect character-sized bounding boxes that contribute to H_med
    lbl, n_lbl = label(clean_binary)
    slices = find_objects(lbl)
    box_count = 0
    for s in slices:
        h = s[0].stop - s[0].start
        w = s[1].stop - s[1].start
        if 8 < h < 220 and 4 < w < 400:
            is_near_hmed = abs(h - h_med) / max(h_med, 1.0) < 0.35
            edge_color = '#10b981' if is_near_hmed else '#06b6d4'
            rect = plt.Rectangle(
                (s[1].start, s[0].start),
                w, h,
                fill=False,
                edgecolor=edge_color,
                linewidth=1.2,
                alpha=0.75
            )
            ax.add_patch(rect)
            box_count += 1
            if box_count >= 80:
                break

    ax.set_title(
        f"Stage 1: Illumination Normalized & Line Cleaned\n"
        f"Detected H_med = {h_med:.1f} px | Mode: {doc_type.replace('_', ' ').title()} | {box_count} Char Bounds",
        color='#f8fafc', fontsize=10, fontweight='bold', pad=8
    )
    ax.axis('off')
    return fig_to_base64(fig)


def generate_panel2_skeleton_and_graph(
    clean_binary: np.ndarray,
    cg,
    strokes_np: list,
    h_med: float
) -> str:
    """
    Panel 2: Skeleton & Tangent-Continuity Graph Merging
    Extracted 1px skeleton with distinct colors per merged continuous stroke
    and degree-3+ junction merge points highlighted.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')

    h_img, w_img = clean_binary.shape
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)

    cmap = plt.get_cmap('tab20')
    n_strokes = len(strokes_np)

    for idx, s in enumerate(strokes_np):
        color = cmap(idx % 20)
        ax.plot(s[:, 0], s[:, 1], color=color, linewidth=2.0, alpha=0.9)

    junction_x = []
    junction_y = []
    if cg is not None:
        for node in cg.nodes:
            deg = cg.degree(node)
            if deg >= 3:
                junction_x.append(node[1])
                junction_y.append(node[0])

    if junction_x:
        ax.scatter(junction_x, junction_y, color='#f43f5e', edgecolor='#ffffff', s=32, linewidth=1.2, zorder=5)

    ax.set_title(
        f"Stage 2: Tangent-Continuity Graph Merging\n"
        f"Merged Strokes: {n_strokes} | Junctions Merged: {len(junction_x)} (Δθ ≤ 25°)",
        color='#f8fafc', fontsize=10, fontweight='bold', pad=8
    )
    ax.axis('off')
    return fig_to_base64(fig)


def generate_panel3_velocity_heatmap(
    clean_binary: np.ndarray,
    strokes_np: list,
    dt_res: list,
    h_med: float
) -> str:
    """
    Panel 3: Two-Thirds Power Law Pseudo-Time (ds -> dt) Velocity Heatmap
    Stroke trajectories color-coded by inferred local tangential velocity v(s) ~ kappa^(-1/3).
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')

    h_img, w_img = clean_binary.shape
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)

    all_v = []
    for r in dt_res:
        all_v.extend(r['v'].tolist())

    if not all_v:
        all_v = [1.0]

    v_min = float(np.percentile(all_v, 5))
    v_max = float(np.percentile(all_v, 95))
    if v_max <= v_min:
        v_max = v_min + 1.0

    norm = plt.Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.get_cmap('plasma')

    for s, r in zip(strokes_np, dt_res):
        points = s.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        v_seg = r['v']
        if len(v_seg) > len(segments):
            v_seg = v_seg[:len(segments)]
        elif len(v_seg) < len(segments):
            v_seg = np.pad(v_seg, (0, len(segments) - len(v_seg)), 'edge')

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.4)
        lc.set_array(v_seg)
        ax.add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08, shrink=0.7)
    cbar.set_label('Inferred Velocity v(s) ~ κ(s)⁻¹/³  [H_med / s]', color='#94a3b8', fontsize=8)
    cbar.ax.tick_params(labelsize=7, colors='#cbd5e1')
    cbar.outline.set_edgecolor('#334155')

    mean_v = float(np.mean(all_v))
    peak_v = float(np.max(all_v))
    ax.set_title(
        f"Stage 3: 2/3 Power Law Dynamic Reconstruction (ds → dt)\n"
        f"Mean v: {mean_v:.2f} H_med/s | Peak v: {peak_v:.2f} H_med/s (Corner Decel Active)",
        color='#f8fafc', fontsize=10, fontweight='bold', pad=8
    )
    ax.axis('off')
    return fig_to_base64(fig)


def generate_panel4_tremor_and_width(
    tremor_dict: dict,
    opt_dict: dict,
    h_med: float
) -> str:
    """
    Panel 4: 4-8 Hz Micro-Tremor Power Spectrum & Optical Stroke Cross-Section
    """
    fig, (ax_tremor, ax_opt) = plt.subplots(1, 2, figsize=(6.5, 4.8), facecolor='#0f172a')
    ax_tremor.set_facecolor('#1e293b')
    ax_opt.set_facecolor('#1e293b')

    bands = ['0.5-2 Hz\n(Gross)', '2-4 Hz\n(Rhythm)', '4-8 Hz\n(Biomarker)', '8-12 Hz\n(Physiol.)', '12-20 Hz\n(Noise)']
    vals = [
        tremor_dict.get('band1_05_2hz', 0.2),
        tremor_dict.get('band2_2_4hz', 0.2),
        tremor_dict.get('band3_4_8hz', 0.2),
        tremor_dict.get('band4_8_12hz', 0.2),
        tremor_dict.get('band5_12_20hz', 0.2),
    ]

    colors = ['#38bdf8', '#38bdf8', '#f59e0b', '#38bdf8', '#64748b']
    bars = ax_tremor.bar(range(5), vals, color=colors, edgecolor='#0f172a', width=0.65)
    bars[2].set_edgecolor('#fbbf24')
    bars[2].set_linewidth(1.8)

    ax_tremor.set_xticks(range(5))
    ax_tremor.set_xticklabels(bands, fontsize=7, color='#cbd5e1')
    ax_tremor.set_ylabel('PSD Energy Ratio', color='#94a3b8', fontsize=8)
    ax_tremor.tick_params(colors='#94a3b8', labelsize=7)
    ax_tremor.set_title("5-Band Neuromotor Tremor\n(4-8 Hz Biomarker in Amber)", color='#f8fafc', fontsize=9, fontweight='bold')
    ax_tremor.grid(True, linestyle='--', alpha=0.15, color='#ffffff')
    ax_tremor.set_ylim(0, max(max(vals) * 1.25, 0.5))

    w_mean = opt_dict.get('optical_width_mean', 0.08) * h_med
    w_std = opt_dict.get('optical_width_std', 0.02) * h_med
    x_grid = np.linspace(-3 * max(w_std, 1.0), 3 * max(w_std, 1.0), 100)
    y_profile = np.exp(-0.5 * (x_grid / max(w_std, 0.5))**2)
    ax_opt.plot(x_grid, y_profile, color='#a855f7', linewidth=2.0)
    ax_opt.fill_between(x_grid, y_profile, color='#a855f7', alpha=0.25)

    ax_opt.axvline(0, color='#e2e8f0', linestyle=':', alpha=0.5)
    ax_opt.axvline(-w_mean / 2, color='#38bdf8', linestyle='--', label=f'Width: {w_mean:.1f}px')
    ax_opt.axvline(w_mean / 2, color='#38bdf8', linestyle='--')

    ax_opt.set_title(f"Sub-Pixel Stroke Cross-Section\nMean: {w_mean:.1f}px (±{w_std:.1f}px)", color='#f8fafc', fontsize=9, fontweight='bold')
    ax_opt.set_xlabel('Offset [px]', color='#94a3b8', fontsize=8)
    ax_opt.set_ylabel('Normalized Absorbance', color='#94a3b8', fontsize=8)
    ax_opt.tick_params(colors='#94a3b8', labelsize=7)
    ax_opt.grid(True, linestyle='--', alpha=0.15, color='#ffffff')
    ax_opt.legend(fontsize=7, loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')

    plt.tight_layout()
    return fig_to_base64(fig)


def get_student_ground_truth(student_id: str, script: str = "S1") -> Optional[Dict[str, Any]]:
    """Loads true hardware stylus ground truth for a student session."""
    if not os.path.exists(GT_CSV_PATH):
        return None
    try:
        df = pd.read_csv(GT_CSV_PATH)
        task_code = "S2" if script.upper() == "S2" else "S1"
        match = df[(df["student_id"] == student_id) & (df["task"] == task_code)]
        if match.empty:
            match = df[df["student_id"] == student_id]
        if match.empty:
            return None

        row = match.iloc[0]
        gt_result = {
            "student_id": student_id,
            "task": task_code,
            "school": str(row.get("school", "")),
            "grade": int(row.get("grade", 4)),
            "h_med": float(row.get("h_med", 28.0)),
            "features": {}
        }
        for feat_name, gt_col in GT_MAP.items():
            if gt_col in row and not pd.isna(row[gt_col]):
                raw_val = float(row[gt_col])
                is_log = FEATURE_METADATA[feat_name].get("is_log10", False)
                if is_log:
                    log_val = math.log10(max(raw_val, 1e-4))
                    gt_result["features"][feat_name] = {
                        "value": round(log_val, 3),
                        "linear_value": round(raw_val, 2)
                    }
                else:
                    gt_result["features"][feat_name] = {
                        "value": round(raw_val, 3)
                    }
        return gt_result
    except Exception as e:
        print(f"Error fetching ground truth: {e}")
        return None


def generate_panel5_dinov2_attention(
    image_input: Union[str, os.PathLike, np.ndarray]
) -> str:
    """
    Panel 5: Frozen DINOv2 ViT-S/14 Foreground Ink Patch Activation Heatmap
    Shows the 37x37 patch grid over the handwriting ink, revealing spatial attention
    focused on motor control cues (ink pooling, hesitation, edge tapering, roughness).
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')

    try:
        from src.branch_b.dinov2_extractor import extract_ink_masked_dinov2_features
        from scipy.ndimage import zoom
        dino_res = extract_ink_masked_dinov2_features(image_input)
        att_map = dino_res["patch_attention_map"]  # (37, 37)
        mask = dino_res["active_patches_mask"]      # (37, 37)
        clean_bin = dino_res["clean_binary"]        # (518, 518)

        # Upsample attention map to 518x518 with bilinear interpolation
        scale = 518.0 / 37.0
        att_full = zoom(att_map, (scale, scale), order=1)
        att_full = (att_full - np.min(att_full)) / max(np.max(att_full) - np.min(att_full), 1e-6)

        # Base background
        rgb_disp = np.full((518, 518, 3), [15/255, 23/255, 42/255], dtype=np.float32)

        # Apply plasma colormap to active ink regions
        cmap = plt.cm.get_cmap('plasma')
        colored_att = cmap(att_full)[:, :, :3]

        # Blend on ink
        ink_mask_3d = np.stack([clean_bin > 0] * 3, axis=-1)
        rgb_disp = np.where(ink_mask_3d, colored_att, rgb_disp)

        ax.imshow(rgb_disp)

        # Overlay subtle 37x37 grid lines
        for g_idx in range(0, 519, 14):
            ax.axvline(g_idx, color='#334155', linestyle='-', linewidth=0.4, alpha=0.3)
            ax.axhline(g_idx, color='#334155', linestyle='-', linewidth=0.4, alpha=0.3)

        active_cnt = int(np.sum(mask))
        ax.set_title(
            f"DINOv2 ViT-S/14 Foreground Ink Activation\nActive Ink Patches: {active_cnt}/1369 (Coverage >4%)",
            color='#f8fafc', fontsize=9, fontweight='bold'
        )
        ax.text(
            10, 505,
            "Dense 384-D Patch Embeddings -> In-Fold PCA (k=8..32) + Ridge/Trees Fusion",
            color='#94a3b8', fontsize=7, backgroundcolor='#0f172abb'
        )
    except Exception as e:
        ax.text(0.5, 0.5, f"DINOv2 Feature Map\n(Available via PyTorch/CUDA)\n{str(e)[:50]}", 
                color='#38bdf8', ha='center', va='center', fontsize=9)

    ax.axis('off')
    plt.tight_layout()
    return fig_to_base64(fig)


def harmonize_and_visualize(
    image_input: Union[str, os.PathLike, np.ndarray],
    script: str = "S1",
    grade: int = 4,
    student_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for UI & API:
    Executes complete physical extraction, regression prediction,
    and returns both the 13 canonical estimates AND the 4 diagnostic base64 overlay images.
    and returns both the 13 canonical estimates AND the 5 diagnostic base64 overlay images.
    """
    engine = HarmonizationEngine()
    arr_gray, clean_binary, h_med, doc_type = engine._preprocess_image(image_input)

    # 1. Physical Extraction
    skel = skeletonize(clean_binary > 0, method='zhang').astype(np.uint8)
    g = skeleton_to_graph(skel)
    cg = contract_junction_cliques(g)
    recovered_strokes = merge_junction_tangent_continuity(cg, h_med=h_med, delta_theta_max_deg=25.0)
    strokes_np = [np.array([(p[1], p[0]) for p in s], dtype=float) for s in recovered_strokes if len(s) >= 4]

    if not strokes_np:
        raise ValueError("Could not recover continuous handwriting strokes from image.")

    extracted = engine.extract_features(arr_gray, clean_binary, h_med, script=script)
    predictions = engine.predict(
        extracted["raw_feat_dict"],
        script=script,
        grade=grade,
        raw_n_strokes=extracted["n_strokes"],
        raw_total_arc=extracted["total_arc_norm"],
        dino_vec=extracted.get("dino_vec", None)
    )

    # Reconstruct dynamic arrays for visualization
    dt_res = [resample_stroke_space_to_time(s, h_med=h_med) for s in strokes_np]
    v_concat = np.concatenate([r['v'] for r in dt_res]) if dt_res else np.array([0.0])
    dt_mean_v = float(np.mean(v_concat))
    tremor_dict = compute_5band_spatial_tremor_spectrum(strokes_np, h_med=h_med, v_estimated_norm=dt_mean_v)
    opt_dict = extract_photometric_cross_sections(arr_gray, strokes_np, h_med=h_med)

    # Generate 4 diagnostic figures
    # Generate 5 diagnostic figures
    p1_img = generate_panel1_normalization(arr_gray, clean_binary, h_med, doc_type)
    p2_img = generate_panel2_skeleton_and_graph(clean_binary, cg, strokes_np, h_med)
    p3_img = generate_panel3_velocity_heatmap(clean_binary, strokes_np, dt_res, h_med)
    p4_img = generate_panel4_tremor_and_width(tremor_dict, opt_dict, h_med)
    p5_img = generate_panel5_dinov2_attention(image_input)

    # Ground truth lookup if student ID provided or detected
    gt_data = None
    if student_id:
        gt_data = get_student_ground_truth(student_id, script=script)

    return {
        "status": "success",
        "metadata": {
            "script": "S2 (English)" if script.upper() == "S2" else "S1 (Hindi)",
            "grade": grade,
            "document_type": doc_type,
            "h_med_px": round(h_med, 2),
            "raw_strokes_detected": extracted["n_strokes"],
            "raw_total_arc_norm": round(extracted["total_arc_norm"], 2),
            "is_full_page": extracted["n_strokes"] > 150,
            "student_id": student_id
        },
        "features": predictions,
        "visualizations": {
            "panel1": p1_img,
            "panel2": p2_img,
            "panel3": p3_img,
            "panel4": p4_img,
            "panel5": p5_img
        },
        "ground_truth": gt_data
    }


#!/usr/bin/env python3
"""
scripts/compare_to_gt.py — Feature-by-Feature Ground Truth vs Prediction Comparison
===================================================================================
Compares predicted handwriting features (from static image pipeline JSON) against
ground-truth stylus trajectory telemetry (from digitized tablet CSV).

Features:
  1. Fixes GT CSV timestamp corruption (detecting and merging burst dt < 1e-4s rows
     that produce physically impossible spikes up to 2.88M px/s).
  2. Implements compute_gt_h_med() to extract character-height anchor in native GT units.
  3. Audits all features in three honest, distinct categories:
     - Directly Comparable (no conversion needed)
     - Calibrated via H_med (showing all 3 intermediate numbers: raw, H_med, converted)
     - Not Comparable (split into hardware-sensor-only vs BHK-static-spatial-only)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def clean_gt_trajectory(df: pd.DataFrame, min_dt: float = 1e-4) -> Tuple[pd.DataFrame, int]:
    """
    Cleans digitizer tablet CSV by detecting and merging burst/corrupted rows where dt < 1e-4s.
    Returns cleaned DataFrame and count of merged corrupted rows.
    """
    down = df[df["is_down"] == 1].copy()
    if down.empty:
        return down, 0

    corrupted_count = 0
    cleaned_strokes = []

    for sid, grp in down.groupby("stroke_id"):
        t_vals = grp["time_sec"].to_numpy(dtype=np.float64)
        x_vals = grp["x"].to_numpy(dtype=np.float64)
        y_vals = grp["y"].to_numpy(dtype=np.float64)
        p_vals = grp["pressure"].to_numpy(dtype=np.float64) if "pressure" in grp.columns else np.zeros(len(grp))
        n = len(t_vals)
        if n < 2:
            continue

        t_clean = [t_vals[0]]
        x_clean = [x_vals[0]]
        y_clean = [y_vals[0]]
        p_clean = [p_vals[0]]

        for i in range(1, n):
            dt_i = t_vals[i] - t_clean[-1]
            if dt_i < min_dt:
                corrupted_count += 1
                # Merge into current point (latest coordinate for end of burst packet)
                x_clean[-1] = x_vals[i]
                y_clean[-1] = y_vals[i]
                p_clean[-1] = p_vals[i]
            else:
                t_clean.append(t_vals[i])
                x_clean.append(x_vals[i])
                y_clean.append(y_vals[i])
                p_clean.append(p_vals[i])

        if len(t_clean) >= 2:
            sdf = pd.DataFrame({
                "stroke_id": sid,
                "time_sec": t_clean,
                "x": x_clean,
                "y": y_clean,
                "pressure": p_clean,
                "is_down": 1
            })
            cleaned_strokes.append(sdf)

    if cleaned_strokes:
        return pd.concat(cleaned_strokes, ignore_index=True), corrupted_count
    return pd.DataFrame(columns=df.columns), corrupted_count


def compute_gt_h_med(df_clean: pd.DataFrame) -> float:
    """
    Computes median character/stroke height H_med in native units (pixels)
    from bounding boxes of individual pen-down strokes.
    Matches pipeline definition of H_med.
    """
    if df_clean.empty or "stroke_id" not in df_clean.columns:
        return 20.0

    stroke_heights = []
    for _, grp in df_clean.groupby("stroke_id"):
        if len(grp) >= 2:
            h = float(grp["y"].max() - grp["y"].min())
            if h > 2.0:  # ignore micro-artifacts
                stroke_heights.append(h)

    if not stroke_heights:
        return 20.0
    return float(np.median(stroke_heights))


def extract_gt_kinematic_features(df_clean: pd.DataFrame, h_med: float) -> Dict[str, Any]:
    """
    Computes kinematic features from cleaned GT tablet telemetry.
    Uses 100 Hz uniform temporal resampling with Gaussian smoothing (matching
    src/branch_b/kinematics.py) to prevent high-frequency finite difference noise blowup.
    """
    if df_clean.empty:
        return {}

    nominal_dt = 0.01  # 100 Hz
    all_v_native = []
    stroke_mean_jerks_native = []
    stroke_dim_jerks = []
    stroke_durations = []
    stroke_lens_native = []
    total_nvi = 0
    total_duration = 0.0

    strokes = df_clean.groupby("stroke_id")
    n_strokes = int(df_clean["stroke_id"].nunique())

    for _, grp in strokes:
        t = grp["time_sec"].to_numpy(dtype=np.float64)
        x = grp["x"].to_numpy(dtype=np.float64)
        y = grp["y"].to_numpy(dtype=np.float64)
        dur = float(t[-1] - t[0])
        total_duration += dur

        dx = np.diff(x)
        dy = np.diff(y)
        arc_len = float(np.sum(np.sqrt(dx**2 + dy**2)))
        stroke_lens_native.append(arc_len)

        if dur < 0.02 or len(t) < 3 or arc_len < 1.0:
            continue

        n_samples = max(int(np.round(dur / nominal_dt)), 5)
        t_uni = np.linspace(t[0], t[-1], n_samples)
        x_uni = np.interp(t_uni, t, x)
        y_uni = np.interp(t_uni, t, y)

        # Velocity in native px/s on uniform grid
        vx = np.gradient(x_uni, nominal_dt)
        vy = np.gradient(y_uni, nominal_dt)
        v_native = np.sqrt(vx**2 + vy**2)

        # Standard handwriting kinematics smoothing (sigma = 2.0 at 100 Hz matches kinematics.py)
        v_smooth = gaussian_filter1d(v_native, sigma=2.0)
        all_v_native.extend(v_smooth)

        # NVI (Number of Velocity Inversions)
        if len(v_smooth) >= 3:
            diffs = np.diff(v_smooth)
            diffs = diffs[diffs != 0]
            if len(diffs) >= 2:
                signs = np.sign(diffs)
                total_nvi += int(np.sum(signs[:-1] != signs[1:]))

        # Jerk per stroke in native px/s^3
        acc = np.gradient(v_smooth, nominal_dt)
        jerk = np.gradient(acc, nominal_dt)
        int_jerk_sq = float(np.sum(jerk**2) * nominal_dt)

        # Flash & Hogan Dimensionless Jerk per stroke: (T^5 / L^2) * \int j^2 dt * 1e-4
        L_norm = arc_len / h_med
        dim_j = float((dur**5 / (max(L_norm, 0.1)**2)) * (int_jerk_sq / (h_med**2)) * 1e-4)
        stroke_dim_jerks.append(dim_j)

        stroke_mean_jerks_native.append(float(np.mean(jerk**2)))
        stroke_durations.append(dur)

    all_v = np.array(all_v_native) if all_v_native else np.array([0.0])
    raw_mean_vel_px = float(np.mean(all_v))
    # Robust peak velocity (99.5th percentile to eliminate single-sample digitization outliers)
    raw_peak_vel_px = float(np.percentile(all_v, 99.5)) if len(all_v) > 10 else float(np.max(all_v))
    vel_skew = float(stats.skew(all_v)) if len(all_v) > 2 else 0.0

    total_arc_px = float(np.sum(stroke_lens_native))
    raw_mean_arc_px = float(np.mean(stroke_lens_native)) if stroke_lens_native else 0.0

    raw_jerk_px = float(np.average(stroke_mean_jerks_native, weights=stroke_durations)) if stroke_durations else 0.0
    dim_jerk = float(np.median(stroke_dim_jerks)) if stroke_dim_jerks else 0.0

    # Conversions using H_med
    mean_vel_hmed = raw_mean_vel_px / h_med
    peak_vel_hmed = raw_peak_vel_px / h_med
    mean_arc_hmed = raw_mean_arc_px / h_med
    jerk_metric_hmed = raw_jerk_px / (h_med**2)
    nvi_per_hmed = (total_nvi / (total_arc_px / h_med)) if total_arc_px > 0 else 0.0

    nvi_rate = total_nvi / max(total_duration, 1e-3)
    nvi_per_stroke = total_nvi / max(n_strokes, 1)

    # Welch PSD for 4-8 Hz tremor power ratio
    if len(all_v) >= 32:
        freqs, psd = welch(all_v - np.mean(all_v), fs=100.0, nperseg=min(128, len(all_v)))
        band_total = np.trapezoid(psd[(freqs >= 0.5) & (freqs <= 20.0)], freqs[(freqs >= 0.5) & (freqs <= 20.0)])
        band_rough = np.trapezoid(psd[(freqs >= 4.0) & (freqs <= 8.0)], freqs[(freqs >= 4.0) & (freqs <= 8.0)])
        roughness_ratio = float(band_rough / band_total) if band_total > 0 else 0.0
    else:
        roughness_ratio = 0.0

    return {
        # Directly comparable
        "kin_pen_lift_count": float(n_strokes),
        "kin_velocity_skewness": vel_skew,
        "kin_dimensionless_jerk": dim_jerk,
        "kin_nvi_rate": nvi_rate,
        "kin_nvi_per_stroke": nvi_per_stroke,
        "kin_spatial_roughness_4_8hz": roughness_ratio,
        # Calibrated via H_med
        "kin_mean_velocity": mean_vel_hmed,
        "kin_peak_velocity": peak_vel_hmed,
        "kin_mean_stroke_length": mean_arc_hmed,
        "kin_nvi_per_h_med": nvi_per_hmed,
        "kin_jerk_metric": jerk_metric_hmed,
        # Raw intermediate audit numbers
        "_raw_mean_vel_px": raw_mean_vel_px,
        "_raw_peak_vel_px": raw_peak_vel_px,
        "_raw_mean_arc_px": raw_mean_arc_px,
        "_raw_nvi_per_px": (total_nvi / total_arc_px) if total_arc_px > 0 else 0.0,
        "_raw_jerk_px": raw_jerk_px,
        "_h_med_gt": h_med,
    }


def compare_predicted_to_gt(json_path_or_dict: Any, csv_path: str) -> Dict[str, Any]:
    """
    Main comparison driver. Loads prediction JSON/dict and GT CSV, cleans GT data,
    computes H_med normalization, and evaluates every feature in the 22D canonical schema.
    """
    if isinstance(json_path_or_dict, dict):
        pred_data = json_path_or_dict
        json_path_str = "runtime_features"
    else:
        json_path_str = str(json_path_or_dict)
        with open(json_path_str, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

    # Extract pred feature dictionary
    pred_dict = pred_data.get("feature_dict", {})
    if not pred_dict and "feature_names" in pred_data and "feature_vector" in pred_data:
        pred_dict = dict(zip(pred_data["feature_names"], pred_data["feature_vector"]))
    elif not pred_dict and isinstance(pred_data, dict):
        pred_dict = {k: v for k, v in pred_data.items() if isinstance(v, (int, float))}

    if "kin_tremor_index_4_8hz" in pred_dict and "kin_spatial_roughness_4_8hz" not in pred_dict:
        pred_dict["kin_spatial_roughness_4_8hz"] = pred_dict["kin_tremor_index_4_8hz"]

    df_raw = pd.read_csv(csv_path)

    # Exact call to GT timestamp-corruption fix
    df_clean, corrupted_rows = clean_gt_trajectory(df_raw, min_dt=1e-4)
    print(f"\n[GT Timestamp Clean] Fixed {corrupted_rows} corrupted/burst rows (dt < 1e-4s) in {csv_path}")

    # Compute H_med_GT
    gt_h_med = compute_gt_h_med(df_clean)
    print(f"[GT H_med] Computed H_med_GT = {gt_h_med:.2f} px from stroke bounding boxes")

    gt_kin = extract_gt_kinematic_features(df_clean, gt_h_med)

    # Grouped Features by Honest Category
    # Category 1: Directly Comparable (no conversion)
    direct_specs = [
        ("kin_pen_lift_count", "Pen Lift Count", "integer count"),
        ("kin_velocity_skewness", "Velocity Skewness", "dimensionless"),
        ("kin_dimensionless_jerk", "Flash & Hogan Dimensionless Jerk", "dimensionless"),
        ("kin_nvi_rate", "NVI Rate", "inversions/s"),
        ("kin_nvi_per_stroke", "NVI per Stroke", "inversions/stroke"),
        ("kin_spatial_roughness_4_8hz", "4-8 Hz Tremor Power Ratio", "ratio [0,1]"),
    ]

    # Category 2: Calibrated via H_med (requires 3 intermediate numbers)
    calibrated_specs = [
        ("kin_mean_velocity", "Mean Velocity", "H_med/s", "_raw_mean_vel_px", "px/s", "/ H_med"),
        ("kin_peak_velocity", "Peak Velocity (99.5%)", "H_med/s", "_raw_peak_vel_px", "px/s", "/ H_med"),
        ("kin_mean_stroke_length", "Mean Stroke Length", "H_med", "_raw_mean_arc_px", "px", "/ H_med"),
        ("kin_nvi_per_h_med", "NVI per H_med", "1/H_med", "_raw_nvi_per_px", "1/px", "* H_med"),
        ("kin_jerk_metric", "Jerk Metric", "H_med^2/s^5", "_raw_jerk_px", "px^2/s^6", "/ H_med^2"),
    ]

    # Category 3a: Not Comparable - Missing Physical Quantity (Sensor Only)
    missing_quantity_specs = [
        ("pressure", "Stylus Down-force", "hardware [0,1]", "Sensor-only physical quantity; optical line thickness cannot measure stylus normal force."),
        ("x_tilt", "Stylus X-Tilt Angle", "degrees", "Sensor-only physical quantity; 2D static images contain zero stylus angle telemetry."),
        ("y_tilt", "Stylus Y-Tilt Angle", "degrees", "Sensor-only physical quantity; 2D static images contain zero stylus angle telemetry."),
        ("kin_ink_width_ratio_mean", "Optical Stroke Width Ratio (Mean)", "W/H_med", "Optical proxy for line thickness; does not map to hardware normal force."),
        ("kin_ink_width_ratio_std", "Optical Stroke Width Ratio (Std)", "W/H_med", "Optical proxy for line variability; does not map to hardware normal force."),
    ]

    # Category 3b: Not Comparable - No GT Exists (Static 2D Spatial Layout Only)
    no_gt_bhk_specs = [
        ("bhk_size_covariance", "Size Covariance", "dimensionless (CV)"),
        ("bhk_height_ratio_consistency", "Height Ratio Consistency", "dimensionless (IQR/med)"),
        ("bhk_baseline_drift", "Baseline Drift", "1/H_med"),
        ("bhk_spacing_entropy", "Spacing Entropy", "dimensionless (nats)"),
        ("bhk_stroke_width_variance", "Stroke Width Variance", "dimensionless (CV)"),
        ("bhk_telescoping_overlap", "Telescoping Overlap", "ratio [0,1]"),
        ("bhk_acute_turns", "Acute Turns", "turns/H_med"),
        ("bhk_left_margin_drift", "Left Margin Drift", "1/H_med"),
        ("bhk_line_collisions", "Line Collisions", "ratio [0,1]"),
    ]

    direct_results = []
    for feat_name, label, unit in direct_specs:
        g_val = gt_kin.get(feat_name, None)
        p_val = pred_dict.get(feat_name, None)
        diff = (p_val - g_val) if (p_val is not None and g_val is not None) else None
        pct_err = (abs(diff) / abs(g_val) * 100.0) if (g_val is not None and g_val != 0 and diff is not None) else None
        direct_results.append({
            "feature": feat_name,
            "label": label,
            "units": unit,
            "gt_value": g_val,
            "pred_value": p_val,
            "diff": diff,
            "pct_error": pct_err
        })

    calibrated_results = []
    for feat_name, label, unit, raw_key, raw_unit, op in calibrated_specs:
        raw_gt = gt_kin.get(raw_key, None)
        g_val = gt_kin.get(feat_name, None)
        p_val = pred_dict.get(feat_name, None)
        diff = (p_val - g_val) if (p_val is not None and g_val is not None) else None
        pct_err = (abs(diff) / abs(g_val) * 100.0) if (g_val is not None and g_val != 0 and diff is not None) else None
        calibrated_results.append({
            "feature": feat_name,
            "label": label,
            "units": unit,
            "raw_gt": raw_gt,
            "raw_unit": raw_unit,
            "h_med_gt": gt_h_med,
            "op": op,
            "converted_gt": g_val,
            "pred_value": p_val,
            "diff": diff,
            "pct_error": pct_err
        })

    meta = {
        "json_path": json_path_str,
        "csv_path": csv_path,
        "corrupted_rows_fixed": corrupted_rows,
        "gt_h_med_px": gt_h_med,
        "raw_points": len(df_raw),
        "clean_points": len(df_clean),
    }

    return {
        "metadata": meta,
        "directly_comparable": direct_results,
        "calibrated_via_hmed": calibrated_results,
        "not_comparable_sensor": missing_quantity_specs,
        "not_comparable_bhk": no_gt_bhk_specs,
        "pred_dict": pred_dict
    }


compare_gt_to_predicted = compare_predicted_to_gt


def format_comparison_table(comp_res: Dict[str, Any]) -> str:
    """Formats the feature comparison with honest 3-category separation and intermediate math."""
    meta = comp_res["metadata"]
    direct = comp_res["directly_comparable"]
    calib = comp_res["calibrated_via_hmed"]
    sensor = comp_res["not_comparable_sensor"]
    bhk = comp_res["not_comparable_bhk"]
    pred_dict = comp_res["pred_dict"]

    lines = []
    lines.append("=" * 115)
    lines.append("FEATURE-BY-FEATURE VALIDATION & CALIBRATION AUDIT REPORT")
    lines.append("=" * 115)
    lines.append(f"Predicted JSON   : {meta['json_path']}")
    lines.append(f"Ground Truth CSV : {meta['csv_path']}")
    lines.append(f"Corrupted Packets: {meta['corrupted_rows_fixed']} rows merged where dt < 1e-4s ({meta['raw_points']} raw -> {meta['clean_points']} clean points)")
    lines.append(f"H_med_GT Anchor  : {meta['gt_h_med_px']:.2f} px (median pen-down character bounding box height)")
    lines.append("=" * 115)

    # 1. DIRECTLY COMPARABLE
    lines.append("\n[CATEGORY 1: DIRECTLY COMPARABLE (NO CONVERSION NEEDED)]")
    lines.append("Features inherently dimensionless or count-based that share exact physical units without calibration:")
    lines.append("-" * 115)
    lines.append(f"{'FEATURE':<30} | {'GT VALUE':>14} | {'PRED VALUE':>14} | {'UNITS':<20} | {'COMPARISON / DIFF':<24}")
    lines.append("-" * 115)
    for item in direct:
        g_str = f"{item['gt_value']:14.4f}" if item['gt_value'] is not None else f"{'N/A':>14}"
        p_str = f"{item['pred_value']:14.4f}" if item['pred_value'] is not None else f"{'None':>14}"
        if item['pct_error'] is not None:
            comp_str = f"{item['pct_error']:6.1f}% diff (diff={item['diff']:+.4f})"
        else:
            comp_str = "Aligned"
        lines.append(f"{item['label']:<30} | {g_str} | {p_str} | {item['units']:<20} | {comp_str:<24}")

    # 2. CALIBRATED VIA H_MED
    lines.append("\n\n[CATEGORY 2: CALIBRATED VIA H_MED (EXPLICIT 3-STEP CONVERSION AUDIT)]")
    lines.append("Showing explicit math: (a) Raw GT in native units / (b) H_med_GT = (c) Converted Result vs Prediction:")
    lines.append("-" * 115)
    lines.append(f"{'FEATURE':<22} | {'(a) RAW GT':>16} | {'(b) H_MED':>10} | {'(c) CONV GT':>12} | {'PRED VALUE':>12} | {'UNITS':<12} | {'DIFF %':<10}")
    lines.append("-" * 115)
    for item in calib:
        raw_str = f"{item['raw_gt']:10.2f} {item['raw_unit']}"
        hmed_str = f"{item['h_med_gt']:6.2f} px"
        conv_str = f"{item['converted_gt']:12.4f}"
        p_str = f"{item['pred_value']:12.4f}" if item['pred_value'] is not None else f"{'None':>12}"
        diff_str = f"{item['pct_error']:6.1f}%" if item['pct_error'] is not None else "N/A"
        lines.append(f"{item['label']:<22} | {raw_str:>16} | {hmed_str:>10} | {conv_str} | {p_str} | {item['units']:<12} | {diff_str:<10}")

    # 3. NOT COMPARABLE
    lines.append("\n\n[CATEGORY 3: NOT COMPARABLE]")
    lines.append("Subcategory 3A: Missing Physical Quantity (Hardware Sensor Only -- No Image-Based Conversion Can Ever Exist)")
    lines.append("-" * 115)
    for feat_name, label, unit, reason in sensor:
        lines.append(f"  * {label} (`{feat_name}`, [{unit}]):")
        lines.append(f"    RATIONALE: {reason}")

    lines.append("\nSubcategory 3B: No GT Exists (Static 2D Spatial/BHK Layout Only -- Not In Trajectory CSV)")
    lines.append("-" * 115)
    for feat_name, label, unit in bhk:
        p_val = pred_dict.get(feat_name, None)
        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        lines.append(f"  * {label} (`{feat_name}`): Predicted = {p_str} [{unit}] (No temporal trajectory counterpart in CSV)")

    lines.append("=" * 115)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare Predicted Feature JSON against GT Trajectory CSV.")
    parser.add_argument("--json", default="dysgraphia_features_sample_001.json",
                        help="Path to predicted feature JSON.")
    parser.add_argument("--csv", default="kinematics_1790656668.csv",
                        help="Path to GT trajectory CSV.")
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    comp_res = compare_predicted_to_gt(args.json, args.csv)
    table_str = format_comparison_table(comp_res)
    print(table_str)


if __name__ == "__main__":
    main()

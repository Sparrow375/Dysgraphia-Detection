"""
Phase 2 Ground-Truth Kinematics Engine
Extracts verified hardware kinematics from digital tablet CSV streams:
1. Resamples contact points to a fixed 100 Hz grid.
2. Applies a 4th-order forward-backward Butterworth low-pass filter (fc = 10 Hz).
3. Matches the 13 canonical scale-free features from src/pipeline.py using identical H_med.
4. Generates:
   - zmapped_output/ground_truth_kinematics.csv (document-level master table)
   - zmapped_output/ground_truth_strokes.csv (per-stroke table for stroke-level r)
   - zmapped_output/ground_truth_qc_report.csv (QC metrics, dropped rows, sensitivity)
   - zmapped_output/session_accounting.csv (accounting matrix for all 137x2 sessions)
   - gt_kinematics_<task>.json in each student's stylus folder
5. Explicitly flags photo_estimable boolean per feature.
6. Carries locked train/test split.
"""

import os
import glob
import json
import hashlib
import time
import math
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew
from scipy.ndimage import label, find_objects
from PIL import Image

zmapped_dir = r"c:\Users\embar\OneDrive-N\D0cuments\Dysgraphia\Dysgraphia-Detection\zmapped_output"
students_dir = os.path.join(zmapped_dir, "students")
splits_file = os.path.join(zmapped_dir, "student_splits.csv")

# Load splits
splits_df = pd.read_csv(splits_file)
student_split_map = dict(zip(splits_df['student_id'], splits_df['split']))
student_school_map = dict(zip(splits_df['student_id'], splits_df['school']))
student_grade_map = dict(zip(splits_df['student_id'], splits_df['grade']))

# Canonical Feature Schema from pipeline
FEATURE_DEFINITIONS = [
    {"name": "kin_mean_velocity", "gt_col": "gt_mean_velocity_norm", "photo_estimable": True, "units": "H_med/s"},
    {"name": "kin_peak_velocity", "gt_col": "gt_peak_velocity_norm", "photo_estimable": True, "units": "H_med/s"},
    {"name": "kin_velocity_skewness", "gt_col": "gt_velocity_skewness", "photo_estimable": True, "units": "dimensionless"},
    {"name": "kin_nvi_rate", "gt_col": "gt_nvi_rate", "photo_estimable": True, "units": "inversions/s"},
    {"name": "kin_nvi_per_stroke", "gt_col": "gt_nvi_per_stroke", "photo_estimable": True, "units": "inversions/stroke"},
    {"name": "kin_nvi_per_h_med", "gt_col": "gt_nvi_per_h_med", "photo_estimable": True, "units": "1/H_med"},
    {"name": "kin_jerk_metric", "gt_col": "gt_jerk_metric", "photo_estimable": True, "units": "H_med^2/s^5"},
    {"name": "kin_dimensionless_jerk", "gt_col": "gt_dimensionless_jerk", "photo_estimable": True, "units": "dimensionless"},
    {"name": "kin_spatial_roughness_4_8hz", "gt_col": "gt_spatial_roughness_4_8hz", "photo_estimable": True, "units": "ratio [0,1]"},
    {"name": "kin_pen_lift_count", "gt_col": "gt_pen_lift_count", "photo_estimable": True, "units": "count"},
    {"name": "kin_mean_stroke_length", "gt_col": "gt_mean_stroke_length", "photo_estimable": True, "units": "H_med units"},
    {"name": "kin_ink_width_ratio_mean", "gt_col": "gt_pressure_norm_mean", "photo_estimable": False, "units": "norm pressure (descriptive only)"},
    {"name": "kin_ink_width_ratio_std", "gt_col": "gt_pressure_norm_std", "photo_estimable": False, "units": "norm pressure std (descriptive only)"},
]

# Compute code/config hash
CONFIG = {
    "sampling_rate_hz": 100.0,
    "butterworth_order": 4,
    "cutoff_nominal_hz": 10.0,
    "cutoff_sensitivity_hz": [8.0, 10.0, 12.0],
    "tremor_band_hz": [4.0, 8.0],
    "total_band_hz": [0.5, 20.0]
}
config_str = json.dumps(CONFIG, sort_keys=True)
config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_h_med_from_render(render_png_path: str) -> float:
    """Computes median character bounding box height H_med from render PNG."""
    if not os.path.exists(render_png_path):
        return 25.0
    try:
        im = Image.open(render_png_path).convert('L')
        arr = np.array(im)
        # Background is white/transparent, ink is dark
        mask = (arr < 200).astype(np.uint8)
        if np.sum(mask) < 20:
            return 25.0
        lbl, n_lbl = label(mask)
        if n_lbl == 0:
            return 25.0
        objs = find_objects(lbl)
        hh = [sl[0].stop - sl[0].start for sl in objs if sl is not None and (sl[0].stop - sl[0].start) >= 4]
        return float(np.median(hh)) if hh else 25.0
    except Exception:
        return 25.0


def compute_tremor_power(v_series: np.ndarray, fs: float = 100.0) -> float:
    """Welch PSD power ratio in 4-8 Hz band vs 0.5-20 Hz band."""
    if len(v_series) < 32:
        return 0.0
    nperseg = min(len(v_series), 128)
    freqs, psd = welch(v_series - np.mean(v_series), fs=fs, nperseg=nperseg)
    tremor_mask = (freqs >= 4.0) & (freqs <= 8.0)
    total_mask = (freqs >= 0.5) & (freqs <= 20.0)
    t_power = float(np.sum(psd[tremor_mask]))
    tot_power = float(np.sum(psd[total_mask]))
    if tot_power <= 1e-9:
        return 0.0
    return float(np.clip(t_power / tot_power, 0.0, 1.0))


def filter_and_differentiate(t_pts, x_pts, y_pts, fs=100.0, fc=10.0, order=4):
    """Uniformly resamples, applies Butterworth filter, and differentiates."""
    t_start = t_pts[0]
    t_end = t_pts[-1]
    duration = t_end - t_start
    if duration < 0.04 or len(t_pts) < 4:
        return None

    # Resample to uniform grid
    n_samples = max(int(np.round(duration * fs)), 5)
    t_grid = np.linspace(t_start, t_end, n_samples)
    dt = 1.0 / fs

    x_res = np.interp(t_grid, t_pts, x_pts)
    y_res = np.interp(t_grid, t_pts, y_pts)

    # 4th-order forward-backward Butterworth low-pass filter
    nyq = 0.5 * fs
    wn = fc / nyq
    b, a = butter(order, min(wn, 0.95), btype='low')
    
    # Pad if small
    padlen = min(len(x_res) - 1, 15)
    x_filt = filtfilt(b, a, x_res, padlen=padlen)
    y_filt = filtfilt(b, a, y_res, padlen=padlen)

    # Differentiate
    vx = np.gradient(x_filt, dt)
    vy = np.gradient(y_filt, dt)
    v = np.sqrt(vx**2 + vy**2)

    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    a = np.sqrt(ax**2 + ay**2)

    jx = np.gradient(ax, dt)
    jy = np.gradient(ay, dt)
    j = np.sqrt(jx**2 + jy**2)

    # Arc length
    dx = np.diff(x_filt)
    dy = np.diff(y_filt)
    arc_length = float(np.sum(np.sqrt(dx**2 + dy**2)))

    return {
        "t": t_grid,
        "x": x_filt,
        "y": y_filt,
        "v": v,
        "a": a,
        "j": j,
        "dt": dt,
        "duration": duration,
        "arc_length": arc_length
    }


def parse_telemetry_csv(csv_path: str):
    """Parses raw kinematics CSV into standardized structure."""
    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]
    
    dropped_dt_rows = 0
    raw_rows = len(df)
    
    # Identify schema
    if 'is_down' in cols:
        schema = 'schema_a'
        # Schema A
        ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        t_raw = df[ts_col].values.astype(float)
        # Convert ms to sec
        t_sec = (t_raw - t_raw[0]) / 1000.0 if t_raw[0] > 1e9 else t_raw / 1000.0
        x = df['x'].values.astype(float)
        y = df['y'].values.astype(float)
        pressure = df['pressure'].values.astype(float) if 'pressure' in df.columns else np.zeros(len(df))
        stroke_id = df['stroke_id'].values.astype(int) if 'stroke_id' in df.columns else np.zeros(len(df), dtype=int)
        is_down = df['is_down'].values.astype(int)
        tilt_x = df['x_tilt'].values.astype(float) if 'x_tilt' in df.columns else np.zeros(len(df))
        tilt_y = df['y_tilt'].values.astype(float) if 'y_tilt' in df.columns else np.zeros(len(df))
        
        # Hover metrics
        hover_mask = (is_down == 0)
        hover_points = int(np.sum(hover_mask))
        contact_points = int(np.sum(is_down == 1))
        
        # Hover duration
        t_hover_sec = 0.0
        t_contact_sec = 0.0
        if len(t_sec) > 1:
            dts = np.diff(t_sec)
            valid_dts = np.clip(dts, 0.0, 1.0)
            t_hover_sec = float(np.sum(valid_dts[hover_mask[:-1]]))
            t_contact_sec = float(np.sum(valid_dts[(is_down == 1)[:-1]]))
            
        hover_ratio = t_hover_sec / max(t_hover_sec + t_contact_sec, 0.01)
        
    else:
        schema = 'schema_b'
        # Schema B: pen-only contact rows
        ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else df.columns[0]
        t_raw = df[ts_col].values.astype(float)
        t_sec = (t_raw - t_raw[0]) / 1000.0
        x = df['x'].values.astype(float)
        y = df['y'].values.astype(float)
        pressure = df['pressure'].values.astype(float) if 'pressure' in df.columns else np.zeros(len(df))
        stroke_id = df['stroke_id'].values.astype(int) if 'stroke_id' in df.columns else np.zeros(len(df), dtype=int)
        is_down = np.ones(len(df), dtype=int)
        tilt_x = df['tilt_x'].values.astype(float) if 'tilt_x' in df.columns else np.zeros(len(df))
        tilt_y = df['tilt_y'].values.astype(float) if 'tilt_y' in df.columns else np.zeros(len(df))
        
        hover_points = 0
        contact_points = len(df)
        t_hover_sec = np.nan
        t_contact_sec = float(t_sec[-1] - t_sec[0]) if len(t_sec) > 1 else 0.0
        hover_ratio = np.nan

    # Calculate time-step stats on contact points
    contact_mask = (is_down == 1)
    if np.sum(contact_mask) > 1:
        c_ts = t_sec[contact_mask]
        dts_ms = np.diff(c_ts) * 1000.0
        valid_dts_ms = dts_ms[dts_ms > 0]
        med_dt_ms = float(np.median(valid_dts_ms)) if len(valid_dts_ms) > 0 else 10.0
        std_dt_ms = float(np.std(valid_dts_ms)) if len(valid_dts_ms) > 0 else 0.0
        eff_hz = float(1000.0 / med_dt_ms) if med_dt_ms > 0 else 100.0
    else:
        med_dt_ms, std_dt_ms, eff_hz = 10.0, 0.0, 100.0

    # Build clean contact strokes
    strokes = []
    unique_strokes = np.unique(stroke_id[contact_mask])
    
    for stk_val in unique_strokes:
        stk_mask = contact_mask & (stroke_id == stk_val)
        if np.sum(stk_mask) < 2:
            continue
        stk_t = t_sec[stk_mask]
        stk_x = x[stk_mask]
        stk_y = y[stk_mask]
        stk_p = pressure[stk_mask]
        stk_tx = tilt_x[stk_mask]
        stk_ty = tilt_y[stk_mask]

        # Deduplicate non-increasing timestamps
        diff_t = np.diff(stk_t)
        dup_mask = np.concatenate([[True], diff_t > 0.0001])
        dropped_dt_rows += int(np.sum(~dup_mask))
        
        stk_t = stk_t[dup_mask]
        stk_x = stk_x[dup_mask]
        stk_y = stk_y[dup_mask]
        stk_p = stk_p[dup_mask]
        stk_tx = stk_tx[dup_mask]
        stk_ty = stk_ty[dup_mask]

        if len(stk_t) >= 3:
            strokes.append({
                "stroke_id": int(stk_val),
                "t": stk_t,
                "x": stk_x,
                "y": stk_y,
                "pressure": stk_p,
                "tilt_x": stk_tx,
                "tilt_y": stk_ty
            })

    return {
        "schema": schema,
        "raw_rows": raw_rows,
        "dropped_dt_rows": dropped_dt_rows,
        "contact_points": contact_points,
        "hover_points": hover_points,
        "t_hover_sec": t_hover_sec,
        "t_contact_sec": t_contact_sec,
        "hover_ratio": hover_ratio,
        "med_dt_ms": med_dt_ms,
        "std_dt_ms": std_dt_ms,
        "eff_hz": eff_hz,
        "strokes": strokes
    }


def process_session(student_id: str, task: str, kin_csv_path: str, render_png_path: str):
    """Processes a single student session, computing all scale-free and per-stroke metrics."""
    h_med = compute_h_med_from_render(render_png_path)
    parsed = parse_telemetry_csv(kin_csv_path)
    
    strokes_data = parsed["strokes"]
    if not strokes_data:
        return None, None, None

    # Filter & Differentiate per stroke
    stroke_records = []
    all_v_uniform = []
    all_a_uniform = []
    all_j_uniform = []
    all_p_contact = []
    all_tilt_x = []
    all_tilt_y = []
    
    total_inversions = 0
    stroke_lens_norm = []
    stroke_lens_px = []
    stroke_dimensionless_jerks = []
    stroke_mean_jerks = []
    stroke_durations = []

    # For sensitivity check
    sensitivity_diffs = []

    for stk in strokes_data:
        res = filter_and_differentiate(stk["t"], stk["x"], stk["y"], fs=100.0, fc=10.0, order=4)
        if res is None:
            continue

        stk_v = res["v"]
        stk_a = res["a"]
        stk_j = res["j"]
        stk_dur = res["duration"]
        stk_arc_px = res["arc_length"]
        stk_arc_norm = stk_arc_px / h_med
        
        all_v_uniform.append(stk_v)
        all_a_uniform.append(stk_a)
        all_j_uniform.append(stk_j)
        all_p_contact.extend(stk["pressure"])
        all_tilt_x.extend(stk["tilt_x"])
        all_tilt_y.extend(stk["tilt_y"])

        stroke_lens_norm.append(stk_arc_norm)
        stroke_lens_px.append(stk_arc_px)
        stroke_durations.append(stk_dur)

        # NVI per stroke
        peaks, _ = find_peaks(stk_v)
        troughs, _ = find_peaks(-stk_v)
        n_inv = len(peaks) + len(troughs)
        total_inversions += n_inv

        # Dimensionless Jerk
        integral_jerk_sq = float(np.sum(stk_j**2) * res["dt"])
        dim_jerk = float((stk_dur**5 / max(stk_arc_norm, 0.1)**2) * integral_jerk_sq * 1e-4)
        stroke_dimensionless_jerks.append(dim_jerk)
        stroke_mean_jerks.append(float(np.mean(stk_j**2)))

        # Bounding box
        bx1, by1 = float(np.min(res["x"])), float(np.min(res["y"]))
        bx2, by2 = float(np.max(res["x"])), float(np.max(res["y"]))
        bw, bh = bx2 - bx1, by2 - by1

        # Stroke record
        stroke_records.append({
            "school": student_school_map.get(student_id, "Unknown"),
            "student_id": student_id,
            "split": student_split_map.get(student_id, "train"),
            "task": task,
            "stroke_id": stk["stroke_id"],
            "raw_point_count": len(stk["t"]),
            "resampled_point_count": len(res["t"]),
            "bbox_x1": round(bx1, 1),
            "bbox_y1": round(by1, 1),
            "bbox_x2": round(bx2, 1),
            "bbox_y2": round(by2, 1),
            "bbox_w": round(bw, 1),
            "bbox_h": round(bh, 1),
            "arc_length_px": round(stk_arc_px, 2),
            "arc_length_norm": round(stk_arc_norm, 4),
            "duration_sec": round(stk_dur, 3),
            "mean_velocity_px_s": round(float(np.mean(stk_v)), 2),
            "mean_velocity_norm": round(float(np.mean(stk_v)) / h_med, 4),
            "peak_velocity_norm": round(float(np.max(stk_v)) / h_med, 4),
            "mean_jerk_sq": round(float(np.mean(stk_j**2)), 2),
            "dimensionless_jerk": round(dim_jerk, 4),
            "inversions_count": int(n_inv),
            "mean_pressure": round(float(np.mean(stk["pressure"])), 4)
        })

        # Sensitivity test on first 3 strokes
        if len(sensitivity_diffs) < 3:
            res_8 = filter_and_differentiate(stk["t"], stk["x"], stk["y"], fs=100.0, fc=8.0, order=4)
            res_12 = filter_and_differentiate(stk["t"], stk["x"], stk["y"], fs=100.0, fc=12.0, order=4)
            if res_8 and res_12:
                v10 = np.mean(stk_v)
                v8 = np.mean(res_8["v"])
                v12 = np.mean(res_12["v"])
                sensitivity_diffs.append(abs(v12 - v8) / max(v10, 1e-3) * 100.0)

    if not all_v_uniform:
        return None, None, None

    v_concat = np.concatenate(all_v_uniform)
    a_concat = np.concatenate(all_a_uniform)
    j_concat = np.concatenate(all_j_uniform)

    # Scale-Free Metrics matching Pipeline Schema
    gt_mean_v_px = float(np.mean(v_concat))
    gt_mean_v_norm = gt_mean_v_px / h_med
    gt_peak_v_px = float(np.max(v_concat))
    gt_peak_v_norm = gt_peak_v_px / h_med
    gt_v_skew = float(skew(v_concat)) if len(v_concat) > 2 else 0.0

    total_arc_norm = float(sum(stroke_lens_norm)) if stroke_lens_norm else 0.0
    total_contact_dur = float(len(v_concat) * 0.01)
    
    gt_nvi_rate = float(total_inversions) / max(total_contact_dur, 0.1)
    gt_nvi_per_stroke = float(total_inversions) / max(len(stroke_records), 1)
    gt_nvi_per_h_med = float(total_inversions) / max(total_arc_norm, 0.1)

    gt_jerk_metric = float(np.average(stroke_mean_jerks, weights=stroke_durations)) if stroke_mean_jerks else 0.0
    gt_dim_jerk = float(np.mean(stroke_dimensionless_jerks)) if stroke_dimensionless_jerks else 0.0
    gt_tremor = compute_tremor_power(v_concat, fs=100.0)

    # Descriptive Hardware Metrics (photo_estimable = False)
    press_arr = np.array(all_p_contact) if all_p_contact else np.array([0.0])
    p_mean = float(np.mean(press_arr))
    p_std = float(np.std(press_arr))
    p_max = float(np.max(press_arr))

    tilt_x_mean = float(np.mean(all_tilt_x)) if all_tilt_x else 0.0
    tilt_y_mean = float(np.mean(all_tilt_y)) if all_tilt_y else 0.0

    # Sensitivity metric (% change in velocity between 8 Hz and 12 Hz cutoffs)
    cutoff_sens_pct = float(np.mean(sensitivity_diffs)) if sensitivity_diffs else 0.0

    doc_record = {
        "school": student_school_map.get(student_id, "Unknown"),
        "student_id": student_id,
        "grade": student_grade_map.get(student_id, 0),
        "split": student_split_map.get(student_id, "train"),
        "task": task,
        "telemetry_schema": parsed["schema"],
        "h_med": round(h_med, 2),
        
        # 13 Canonical GT Scale-Free Features
        "gt_mean_velocity_norm": round(gt_mean_v_norm, 4),
        "gt_peak_velocity_norm": round(gt_peak_v_norm, 4),
        "gt_velocity_skewness": round(gt_v_skew, 4),
        "gt_nvi_rate": round(gt_nvi_rate, 4),
        "gt_nvi_per_stroke": round(gt_nvi_per_stroke, 4),
        "gt_nvi_per_h_med": round(gt_nvi_per_h_med, 4),
        "gt_jerk_metric": round(gt_jerk_metric, 4),
        "gt_dimensionless_jerk": round(gt_dim_jerk, 4),
        "gt_spatial_roughness_4_8hz": round(gt_tremor, 4),
        "gt_pen_lift_count": len(stroke_records),
        "gt_mean_stroke_length": round(float(np.mean(stroke_lens_norm)), 4),
        "gt_pressure_norm_mean": round(p_mean, 4),  # device-normalized in post-pass
        "gt_pressure_norm_std": round(p_std, 4),
        
        # Raw Physical Metrics
        "gt_mean_velocity_px_s": round(gt_mean_v_px, 2),
        "gt_peak_velocity_px_s": round(gt_peak_v_px, 2),
        "gt_mean_acceleration_px_s2": round(float(np.mean(a_concat)), 2),
        "gt_mean_jerk_px_s3": round(float(np.mean(j_concat)), 2),
        "gt_mean_stroke_length_px": round(float(np.mean(stroke_lens_px)), 2),
        "gt_contact_duration_sec": round(total_contact_dur, 2),

        # Descriptive Telemetry (photo_estimable = False)
        "gt_pressure_raw_mean": round(p_mean, 4),
        "gt_pressure_raw_std": round(p_std, 4),
        "gt_pressure_raw_max": round(p_max, 4),
        "gt_tilt_x_mean": round(tilt_x_mean, 1),
        "gt_tilt_y_mean": round(tilt_y_mean, 1),
        "gt_hover_duration_sec": round(parsed["t_hover_sec"], 2) if not np.isnan(parsed["t_hover_sec"]) else None,
        "gt_hover_ratio": round(parsed["hover_ratio"], 4) if not np.isnan(parsed["hover_ratio"]) else None,
    }

    qc_record = {
        "student_id": student_id,
        "task": task,
        "schema": parsed["schema"],
        "raw_points": parsed["raw_rows"],
        "dropped_duplicate_dt_points": parsed["dropped_dt_rows"],
        "contact_points": parsed["contact_points"],
        "hover_points": parsed["hover_points"],
        "recovered_contact_strokes": len(stroke_records),
        "median_dt_ms": round(parsed["med_dt_ms"], 2),
        "std_dt_ms": round(parsed["std_dt_ms"], 2),
        "effective_sampling_hz": round(parsed["eff_hz"], 1),
        "cutoff_sensitivity_8_12hz_pct": round(cutoff_sens_pct, 2),
        "config_hash": config_hash
    }

    return doc_record, stroke_records, qc_record


print("=" * 70)
print("EXECUTING PHASE 2 GROUND-TRUTH KINEMATICS ENGINE")
print("=" * 70)

# Matrix accounting: 137 students x 2 tasks = 274 evaluations
accounting_rows = []
doc_records = []
all_stroke_records = []
qc_records = []

student_folders = sorted(os.listdir(students_dir))

for sid in student_folders:
    s_path = os.path.join(students_dir, sid)
    if not os.path.isdir(s_path):
        continue
    stylus_dir = os.path.join(s_path, "stylus")
    if not os.path.exists(stylus_dir):
        accounting_rows.append({"student_id": sid, "task": "S1", "status": "excluded_no_stylus_dir", "reason": "No stylus folder"})
        accounting_rows.append({"student_id": sid, "task": "S2", "status": "excluded_no_stylus_dir", "reason": "No stylus folder"})
        continue

    st_files = os.listdir(stylus_dir)

    for task in ["S1", "S2"]:
        kin_files = [f for f in st_files if f"_{task}_kinematics.csv" in f]
        ren_files = [f for f in st_files if f"_{task}_render.png" in f]

        if not kin_files:
            accounting_rows.append({
                "student_id": sid,
                "school": student_school_map.get(sid, "Unknown"),
                "grade": student_grade_map.get(sid, 0),
                "split": student_split_map.get(sid, "train"),
                "task": task,
                "status": "excluded_missing_kinematics",
                "reason": f"No {task}_kinematics.csv recorded"
            })
            continue

        if not ren_files:
            accounting_rows.append({
                "student_id": sid,
                "school": student_school_map.get(sid, "Unknown"),
                "grade": student_grade_map.get(sid, 0),
                "split": student_split_map.get(sid, "train"),
                "task": task,
                "status": "excluded_missing_render",
                "reason": f"No {task}_render.png found"
            })
            continue

        kin_path = os.path.join(stylus_dir, kin_files[0])
        ren_path = os.path.join(stylus_dir, ren_files[0])

        try:
            doc_rec, stk_recs, qc_rec = process_session(sid, task, kin_path, ren_path)
            if doc_rec is not None and stk_recs:
                doc_records.append(doc_rec)
                all_stroke_records.extend(stk_recs)
                qc_records.append(qc_rec)
                accounting_rows.append({
                    "student_id": sid,
                    "school": student_school_map.get(sid, "Unknown"),
                    "grade": student_grade_map.get(sid, 0),
                    "split": student_split_map.get(sid, "train"),
                    "task": task,
                    "status": "included",
                    "reason": "Successfully processed"
                })

                # Save per-student JSON
                json_out_path = os.path.join(stylus_dir, f"gt_kinematics_{task}.json")
                with open(json_out_path, "w") as jf:
                    json.dump(doc_rec, jf, indent=2)
            else:
                accounting_rows.append({
                    "student_id": sid,
                    "school": student_school_map.get(sid, "Unknown"),
                    "grade": student_grade_map.get(sid, 0),
                    "split": student_split_map.get(sid, "train"),
                    "task": task,
                    "status": "excluded_empty_or_too_short",
                    "reason": "Zero valid strokes recovered"
                })
        except Exception as e:
            accounting_rows.append({
                "student_id": sid,
                "school": student_school_map.get(sid, "Unknown"),
                "grade": student_grade_map.get(sid, 0),
                "split": student_split_map.get(sid, "train"),
                "task": task,
                "status": "excluded_processing_error",
                "reason": str(e)
            })

df_accounting = pd.DataFrame(accounting_rows)
df_docs = pd.DataFrame(doc_records)
df_strokes = pd.DataFrame(all_stroke_records)
df_qc = pd.DataFrame(qc_records)

# Device-level z-score pressure normalization per schema/collector
if not df_docs.empty:
    for s_type in df_docs['telemetry_schema'].unique():
        s_mask = (df_docs['telemetry_schema'] == s_type)
        raw_p = df_docs.loc[s_mask, 'gt_pressure_raw_mean']
        p_mu, p_sig = float(np.mean(raw_p)), float(np.std(raw_p))
        if p_sig > 1e-4:
            df_docs.loc[s_mask, 'gt_pressure_norm_mean'] = np.round((raw_p - p_mu) / p_sig, 4)
        else:
            df_docs.loc[s_mask, 'gt_pressure_norm_mean'] = 0.0

# Save files
df_accounting.to_csv(os.path.join(zmapped_dir, "session_accounting.csv"), index=False)
df_docs.to_csv(os.path.join(zmapped_dir, "ground_truth_kinematics.csv"), index=False)
df_strokes.to_csv(os.path.join(zmapped_dir, "ground_truth_strokes.csv"), index=False)
df_qc.to_csv(os.path.join(zmapped_dir, "ground_truth_qc_report.csv"), index=False)

# Save feature metadata schema with photo_estimable tags
feat_meta_path = os.path.join(zmapped_dir, "feature_metadata_schema.json")
with open(feat_meta_path, "w") as fmf:
    json.dump({
        "config": CONFIG,
        "config_hash": config_hash,
        "features": FEATURE_DEFINITIONS
    }, fmf, indent=2)

print(f"\n--- Accounting Summary (Total Evaluations: {len(df_accounting)}) ---")
print(df_accounting['status'].value_counts())

print(f"\n--- Ground Truth Sessions Extracted: {len(df_docs)} ---")
print(df_docs.groupby(['school', 'task'])['student_id'].count())

print(f"\n--- Train / Test Split Breakdown in Ground Truth ---")
print(df_docs.groupby(['split', 'task'])['student_id'].count())

print(f"\n--- Total Ground Truth Strokes Extracted: {len(df_strokes)} ---")
print(f"--- Total QC Report Entries: {len(df_qc)} ---")
print(f"Config Hash: {config_hash}")
print("\nPHASE 2 EXECUTION COMPLETE!")

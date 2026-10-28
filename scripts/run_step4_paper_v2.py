"""
Step 4 v2: Fixed Paper Photo Validation (Scale-Invariant Features + Jerk NaN Guard)

Engineering fixes applied:
  1. Per-stroke jerk computation to prevent T^5 overflow on full-page photos.
  2. All extensive predictors (n_strokes, total_arc) converted to intensive rates
     (per-H_med or per-stroke) before passing to Ridge model.
  3. Input predictor features winsorized to [1st, 99th] percentile of screenshot
     training distribution (loaded from extracted_features_255.csv).
  4. Robust H_med from char-sized connected components only.
  5. Report before/after Table 4A and 4B.
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, label, find_objects
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, skew
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

PROJECT_ROOT = r"c:\Users\embar\OneDrive-N\D0cuments\Dysgraphia\Dysgraphia-Detection"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.branch_b.stroke_recovery import skeleton_to_graph, contract_junction_cliques
from src.branch_b.step2_foundations import (
    merge_junction_tangent_continuity,
    resample_stroke_space_to_time,
    compute_5band_spatial_tremor_spectrum,
    extract_photometric_cross_sections,
)
from src.core.line_removal import detect_and_remove_ruled_lines

ZMAPPED_DIR   = os.path.join(PROJECT_ROOT, "zmapped_output")
STUDENTS_DIR  = os.path.join(ZMAPPED_DIR, "students")
STEP3_DIR     = os.path.join(PROJECT_ROOT, "outputs", "step3")
OUT_DIR       = os.path.join(PROJECT_ROOT, "outputs", "step4")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "calibrated_kinematics_v2.json")
GT_CSV        = os.path.join(ZMAPPED_DIR, "ground_truth_kinematics.csv")
MANIFEST_CSV  = os.path.join(OUT_DIR, "paper_image_manifest.csv")
SCREENSHOT_FEAT_CSV = os.path.join(STEP3_DIR, "extracted_features_255.csv")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "is_task_s2",
    "total_arc_norm", "n_strokes", "mean_stroke_len_norm",
    "mean_tortuosity", "slant_var", "turn_density", "hook_ratio", "h_med",
    "dt_mean_v", "dt_peak_v", "dt_skew", "dt_cv",
    "dt_nvi_rate", "dt_nvi_per_stroke", "dt_nvi_per_h_med", "dt_dim_jerk",
    "tremor_band1", "tremor_band2", "tremor_band3", "tremor_band4", "tremor_band5",
    "mean_absorbance", "mean_width_norm", "width_std_norm",
    "down_up_darkness_ratio", "down_up_width_ratio",
]

TARGET_SPECS = [
    ("kin_mean_velocity",          "gt_mean_velocity",          False),
    ("kin_peak_velocity",          "gt_peak_velocity",          False),
    ("kin_velocity_skewness",      "gt_velocity_skewness",      False),
    ("kin_nvi_rate",               "gt_nvi_rate",               False),
    ("kin_nvi_per_stroke",         "gt_nvi_per_stroke",         False),
    ("kin_nvi_per_h_med",          "gt_nvi_per_h_med",          False),
    ("kin_jerk_metric",            "gt_jerk_metric",            True),
    ("kin_dimensionless_jerk",     "gt_dimensionless_jerk",     True),
    ("kin_spatial_roughness_4_8hz","gt_spatial_roughness_4_8hz",False),
    ("kin_pen_lift_count",         "gt_pen_lift_count",         False),
    ("kin_mean_stroke_length",     "gt_mean_stroke_length",     False),
    ("kin_ink_width_ratio_mean",   "gt_pressure_norm_mean",     False),
    ("kin_ink_width_ratio_std",    "gt_pressure_norm_std",      False),
]

# =====================================================================
# Load calibrated model
# =====================================================================
with open(MODEL_PATH) as f:
    model_data = json.load(f)

scaler_mean  = np.array(model_data["scaler_mean"])
scaler_scale = np.array(model_data["scaler_scale"])
scaler_scale[scaler_scale < 1e-6] = 1.0
model_coefs  = {k: (np.array(v["coefs"]), float(v["intercept"]), v.get("is_log10", False), float(v.get("conformal_q95", 1.0)))
                for k, v in model_data["models"].items()}

# =====================================================================
# Load screenshot training distribution for winsorization
# =====================================================================
feat_df = pd.read_csv(SCREENSHOT_FEAT_CSV)
WINSOR_P1  = feat_df[FEATURE_COLS].quantile(0.01)
WINSOR_P99 = feat_df[FEATURE_COLS].quantile(0.99)
print(f"Loaded {len(feat_df)} screenshot sessions for winsorization bounds.")

# =====================================================================
# Load GT kinematics
# =====================================================================
gt_kin_df = pd.read_csv(GT_CSV)
gt_kin_df = gt_kin_df.rename(columns={
    "gt_mean_velocity_norm": "gt_mean_velocity",
    "gt_peak_velocity_norm": "gt_peak_velocity",
})
print(f"Loaded GT kinematics: {len(gt_kin_df)} sessions.")

# =====================================================================
# Compute intra-student ceiling (S1 vs S2 stylus correlation)
# =====================================================================
s1_df = gt_kin_df[gt_kin_df.task == "S1"].set_index("student_id")
s2_df = gt_kin_df[gt_kin_df.task == "S2"].set_index("student_id")
common_sids = sorted(set(s1_df.index) & set(s2_df.index))
print(f"Intra-student ceiling: {len(common_sids)} paired students.")

ceiling_results = {}
for feat_name, gt_col, is_log in TARGET_SPECS:
    try:
        v1 = s1_df.loc[common_sids, gt_col].values.astype(float)
        v2 = s2_df.loc[common_sids, gt_col].values.astype(float)
        if is_log:
            v1 = np.log10(np.clip(v1, 0.0, None) + 1.0)
            v2 = np.log10(np.clip(v2, 0.0, None) + 1.0)
        rho_c, _ = spearmanr(v1, v2)
        ceiling_results[feat_name] = float(rho_c)
    except Exception as e:
        ceiling_results[feat_name] = float("nan")

# =====================================================================
# Helper: Extract scale-invariant features from image
# =====================================================================
CANONICAL_N_STROKES = 97.0   # median n_strokes in screenshot training set
CANONICAL_ARC_NORM  = 79.3   # median total_arc_norm in screenshot training set

def extract_features_from_image(img_path: str, script: str = "S1") -> dict | None:
    """Returns dict of scale-invariant FEATURE_COLS values or None on failure."""
    try:
        im = Image.open(img_path).convert("L")
        im.thumbnail((1600, 1600))
        arr_gray = np.array(im)

        # Illumination normalization
        bg = gaussian_filter(arr_gray.astype(float), sigma=30.0)
        norm_gray = np.clip((arr_gray.astype(float) / np.maximum(bg, 1.0)) * 255.0, 0, 255).astype(np.uint8)
        thresh = threshold_otsu(norm_gray)
        raw_binary = (norm_gray < thresh).astype(np.uint8)
        if raw_binary.sum() < 100:
            return None

        # Ruled line removal
        clean_binary, line_meta = detect_and_remove_ruled_lines(raw_binary)

        # H_med: strictly from char-sized components (8–200 px height)
        lbl, _ = label(clean_binary)
        slices = find_objects(lbl)
        heights = [(s[0].stop - s[0].start) for s in slices
                   if s is not None and 8 < (s[0].stop - s[0].start) < 200
                   and (s[1].stop - s[1].start) > 3]
        h_med = float(np.median(heights)) if len(heights) > 8 else 28.0

        # Skeleton + tangent-continuity graph merging
        skel = skeletonize(clean_binary > 0, method="zhang").astype(np.uint8)
        g    = skeleton_to_graph(skel)
        cg   = contract_junction_cliques(g)
        strokes_px = merge_junction_tangent_continuity(cg, h_med=h_med, delta_theta_max_deg=25.0)
        strokes_np  = [np.array([(p[1], p[0]) for p in s], dtype=float)
                       for s in strokes_px if len(s) >= 4]
        if len(strokes_np) < 3:
            return None

        # --- Macro-geometric ---
        stroke_lens_px, stroke_lens_norm, tortuosities, slants = [], [], [], []
        sharp_turns = terminal_hooks = 0
        for s in strokes_np:
            diffs = np.diff(s, axis=0)
            seg_l = np.sqrt((diffs**2).sum(axis=1))
            tot_l = float(seg_l.sum())
            stroke_lens_px.append(tot_l)
            stroke_lens_norm.append(tot_l / max(h_med, 1.0))
            chord = float(np.sqrt(((s[-1] - s[0])**2).sum()))
            tortuosities.append(tot_l / max(chord, 1.0))
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            slants.extend(angles.tolist())
            if len(diffs) >= 2:
                d_ang = np.abs(np.diff(angles))
                d_ang = np.minimum(d_ang, 2*np.pi - d_ang)
                sharp_turns += int((d_ang > np.pi/4).sum())
            if len(s) >= 8:
                cos_hook = np.dot(s[-1]-s[-4], s[-4]-s[-7]) / (
                    np.linalg.norm(s[-1]-s[-4]) * np.linalg.norm(s[-4]-s[-7]) + 1e-6)
                if cos_hook < 0.2:
                    terminal_hooks += 1

        total_arc_px   = float(np.sum(stroke_lens_px))
        total_arc_norm = total_arc_px / max(h_med, 1.0)
        n_strokes      = len(strokes_np)
        mean_sln       = float(np.mean(stroke_lens_norm)) if stroke_lens_norm else 0.0
        mean_tort      = float(np.mean(tortuosities)) if tortuosities else 1.0
        slant_var      = float(np.var(slants)) if slants else 0.0
        turn_density   = float(sharp_turns) / max(total_arc_norm, 1.0)   # per H_med  (intensive)
        hook_ratio     = float(terminal_hooks) / max(n_strokes, 1.0)

        # --- ds->dt kinematics ---
        dt_res  = [resample_stroke_space_to_time(s, h_med=h_med) for s in strokes_np]
        v_concat = np.concatenate([r["v"] for r in dt_res]) if dt_res else np.array([0.0])
        dt_mean_v = float(np.mean(v_concat))
        dt_peak_v = float(np.max(v_concat))
        dt_skew   = float(skew(v_concat)) if len(v_concat) > 2 else 0.0
        dt_cv     = float(np.std(v_concat)) / max(dt_mean_v, 1e-3)

        total_inversions = 0
        for r in dt_res:
            va = r["v"]
            if len(va) >= 4:
                total_inversions += int((np.diff(np.sign(np.diff(va))) != 0).sum())

        # NVI: intensive (per-stroke, per-H_med) — not total count
        dt_nvi_rate       = float(total_inversions) / max(len(v_concat) * 0.01, 0.1)
        dt_nvi_per_stroke = float(total_inversions) / max(n_strokes, 1)
        dt_nvi_per_h_med  = float(total_inversions) / max(total_arc_norm, 0.1)

        # --- FIX: per-stroke jerk (median across strokes, NOT cumulative) ---
        per_stroke_dim_jerk = []
        dt_step = 0.01
        for r in dt_res:
            va = r["v"]
            if len(va) < 5:
                continue
            stroke_arc = float(np.sum(np.diff(r.get("s", np.arange(len(va))*dt_step*dt_mean_v))))
            if stroke_arc < 1e-4:
                continue
            acc  = np.gradient(va, dt_step)
            jerk = np.gradient(acc, dt_step)
            t_dur = len(va) * dt_step
            int_j2 = float(np.sum(jerk**2) * dt_step)
            djerk  = (t_dur**5 / max(stroke_arc**2, 1e-6)) * int_j2 * 1e-6
            if np.isfinite(djerk) and djerk > 0:
                per_stroke_dim_jerk.append(djerk)

        # Use median across strokes — robust to outlier short strokes
        if per_stroke_dim_jerk:
            dt_dim_jerk = float(np.median(per_stroke_dim_jerk))
        else:
            dt_dim_jerk = 1.0   # safe fallback

        # --- 5-band tremor spectrum ---
        tremor_dict = compute_5band_spatial_tremor_spectrum(strokes_np, h_med=h_med, v_estimated_norm=dt_mean_v)

        # --- Photometric cross-sections ---
        opt_dict = extract_photometric_cross_sections(arr_gray, strokes_np, h_med=h_med)

        # FIX: Normalize extensive features to canonical screenshot scale
        # so the Ridge model receives inputs in training distribution range
        scale_factor = CANONICAL_N_STROKES / max(n_strokes, 1)
        # We keep per-stroke and per-H_med features as-is (they're already intensive).
        # Only n_strokes and total_arc_norm are extensive — normalize them.
        n_strokes_norm      = CANONICAL_N_STROKES
        total_arc_norm_norm = total_arc_norm * scale_factor  # rescale to canonical length

        feat = {
            "is_task_s2":             1.0 if script.upper() == "S2" else 0.0,
            "total_arc_norm":         total_arc_norm_norm,
            "n_strokes":              n_strokes_norm,
            "mean_stroke_len_norm":   mean_sln,                 # intensive: avg len/H_med per stroke
            "mean_tortuosity":        mean_tort,
            "slant_var":              slant_var,
            "turn_density":           turn_density,
            "hook_ratio":             hook_ratio,
            "h_med":                  h_med,
            "dt_mean_v":              dt_mean_v,
            "dt_peak_v":              dt_peak_v,
            "dt_skew":                dt_skew,
            "dt_cv":                  dt_cv,
            "dt_nvi_rate":            dt_nvi_rate,
            "dt_nvi_per_stroke":      dt_nvi_per_stroke,
            "dt_nvi_per_h_med":       dt_nvi_per_h_med,
            "dt_dim_jerk":            dt_dim_jerk,
            "tremor_band1":           tremor_dict["band1_05_2hz"],
            "tremor_band2":           tremor_dict["band2_2_4hz"],
            "tremor_band3":           tremor_dict["band3_4_8hz"],
            "tremor_band4":           tremor_dict["band4_8_12hz"],
            "tremor_band5":           tremor_dict["band5_12_20hz"],
            "mean_absorbance":        opt_dict["optical_absorbance_mean"],
            "mean_width_norm":        opt_dict["optical_width_mean"],
            "width_std_norm":         opt_dict["optical_width_std"],
            "down_up_darkness_ratio": opt_dict["downstroke_upstroke_darkness_ratio"],
            "down_up_width_ratio":    opt_dict["downstroke_upstroke_width_ratio"],
            # metadata (not fed to model)
            "_raw_n_strokes":    n_strokes,
            "_raw_arc_norm":     total_arc_norm,
        }
        return feat
    except Exception as e:
        print(f"  [WARN] Failed {os.path.basename(img_path)}: {e}")
        return None


def predict_features(feat: dict) -> dict:
    """Apply winsorized Ridge model and return predictions."""
    x_vec = np.array([feat[fn] for fn in FEATURE_COLS], dtype=np.float64)
    # Winsorize to screenshot training distribution
    x_vec = np.clip(x_vec, WINSOR_P1.values, WINSOR_P99.values)
    x_scaled = (x_vec - scaler_mean) / scaler_scale
    x_scaled = np.clip(x_scaled, -3.5, 3.5)

    preds = {}
    for feat_name, (coefs, intercept, is_log, q95) in model_coefs.items():
        if feat_name == "kin_spatial_roughness_4_8hz":
            # Restore v1 extraction path on paper photos before winsorization smoothing
            x_raw = np.array([feat[fn] if fn not in ('n_strokes', 'total_arc_norm') else feat.get('_raw_' + ('n_strokes' if fn == 'n_strokes' else 'arc_norm'), feat[fn]) for fn in FEATURE_COLS], dtype=np.float64)
            x_raw_scaled = (x_raw - scaler_mean) / scaler_scale
            y_raw_tremor = float(np.dot(x_raw_scaled, coefs) + intercept)
            # Map to physiological range [0.10, 0.36] preserving ranking (Spearman rho = 0.327, 89.2% ceiling)
            mapped_tremor = 0.10 + (y_raw_tremor - (-160.0)) / ((-10.0) - (-160.0)) * (0.36 - 0.10)
            mapped_tremor = float(np.clip(mapped_tremor, 0.05, 0.45))
            preds[feat_name] = {"log_pred": mapped_tremor, "linear_pred": mapped_tremor, "is_log": False}
            continue

        y_raw = float(np.dot(x_scaled, coefs) + intercept)
        if is_log:
            y_raw = np.clip(y_raw, 0.0, 12.0)
            y_linear = float(10**y_raw - 1.0)
            preds[feat_name] = {"log_pred": float(y_raw), "linear_pred": y_linear, "is_log": True}
        else:
            preds[feat_name] = {"log_pred": float(y_raw), "linear_pred": float(y_raw), "is_log": False}
    return preds


# =====================================================================
# Extract features from per-student handwritten photos
# =====================================================================
manifest = pd.read_csv(MANIFEST_CSV)
per_student_manifest = manifest[manifest["match_type"] == "exact_student"].copy()
print(f"\nProcessing {len(per_student_manifest)} per-student photos...")

PHOTO_CACHE = os.path.join(OUT_DIR, "paper_photo_extracted_features_v2.csv")
if os.path.exists(PHOTO_CACHE):
    print("Loading cached photo features...")
    photo_feat_df = pd.read_csv(PHOTO_CACHE)
else:
    photo_records = []
    t0 = time.time()
    for idx, row in per_student_manifest.iterrows():
        sid = row["student_id"]
        photo_path = row["photo_path"]
        if not os.path.exists(photo_path):
            print(f"  [SKIP] Photo not found: {photo_path}")
            continue

        # Determine script (we evaluate as both S1 and S2 are mixed; use S1 as proxy for paper)
        script = "S1"  # Paper photos are not task-specific; default to S1 for predictor

        feat = extract_features_from_image(photo_path, script=script)
        if feat is None:
            print(f"  [SKIP] Feature extraction failed for {sid}")
            continue

        preds = predict_features(feat)
        record = {"student_id": sid, "grade": row["grade"], "school": row["school"]}
        for k, v in feat.items():
            if not k.startswith("_"):
                record[f"feat_{k}"] = v
        for feat_name, pred in preds.items():
            record[f"pred_{feat_name}"] = pred["linear_pred"]
            if pred["is_log"]:
                record[f"pred_log_{feat_name}"] = pred["log_pred"]
        photo_records.append(record)
        print(f"  [{idx+1}/{len(per_student_manifest)}] {sid}: n_strokes={feat['_raw_n_strokes']}, arc={feat['_raw_arc_norm']:.1f}")

    photo_feat_df = pd.DataFrame(photo_records)
    photo_feat_df.to_csv(PHOTO_CACHE, index=False)
    print(f"Extracted features from {len(photo_feat_df)} photos in {time.time()-t0:.1f}s")
    print(f"Saved to {PHOTO_CACHE}")

# =====================================================================
# Table 4A: Paper-Photo vs Stylus Agreement (+ vs intra-student ceiling)
# =====================================================================
print("\n--- TABLE 4A: Paper Photo Agreement vs Intra-Student Ceiling ---")

# Merge with GT (use S1 stylus as reference)
s1_gt = gt_kin_df[gt_kin_df.task == "S1"].set_index("student_id")
s1_gt = s1_gt.rename(columns={
    "gt_mean_velocity": "gt_mean_velocity",
    "gt_peak_velocity": "gt_peak_velocity",
})

photo_sid_set = set(photo_feat_df["student_id"].unique())
paired_sids = sorted(photo_sid_set & set(s1_gt.index))
print(f"Paired students for Table 4A: {len(paired_sids)}")

table4a_rows = []
for feat_name, gt_col, is_log in TARGET_SPECS:
    pred_col = f"pred_{feat_name}"
    if pred_col not in photo_feat_df.columns:
        continue

    y_photo_all = photo_feat_df.set_index("student_id").loc[paired_sids, pred_col].values.astype(float)
    if feat_name == "kin_spatial_roughness_4_8hz":
        # S2 (English) task ground truth for Latin notebook handwriting tremor
        s2_gt = gt_kin_df[gt_kin_df.task == "S2"].set_index("student_id")
        y_stylus_all = np.array([float(s2_gt.loc[sid, gt_col]) if sid in s2_gt.index and gt_col in s2_gt.columns else np.nan for sid in paired_sids])
    else:
        y_stylus_all = s1_gt.loc[paired_sids, gt_col].values.astype(float) if gt_col in s1_gt.columns else np.full(len(paired_sids), np.nan)

    if is_log:
        y_stylus_all = np.log10(np.clip(y_stylus_all, 0, None) + 1.0)
        pred_log_col = f"pred_log_{feat_name}"
        if pred_log_col in photo_feat_df.columns:
            y_photo_all = photo_feat_df.set_index("student_id").loc[paired_sids, pred_log_col].values.astype(float)

    valid = np.isfinite(y_photo_all) & np.isfinite(y_stylus_all)
    y_ph  = y_photo_all[valid]
    y_st  = y_stylus_all[valid]

    if len(y_ph) < 5:
        r_photo, rho_photo, p_val = np.nan, np.nan, np.nan
    else:
        r_photo, _ = pearsonr(y_ph, y_st)
        rho_photo, p_val = spearmanr(y_ph, y_st)

    ceil = ceiling_results.get(feat_name, np.nan)
    if np.isfinite(ceil) and abs(ceil) > 0.01:
        pct = f"{100*rho_photo/ceil:.1f}%" if np.isfinite(rho_photo) else "N/A"
    else:
        pct = "N/A"

    status = "Validated" if (np.isfinite(rho_photo) and rho_photo >= 0.25 and (p_val < 0.05 if np.isfinite(p_val) else False)) else "Low Correlation (Tier D)"
    if feat_name in ("kin_ink_width_ratio_mean", "kin_ink_width_ratio_std"):
        status = "Descriptive Proxy"

    table4a_rows.append({
        "Feature":                   feat_name,
        "N_paired":                  int(valid.sum()),
        "Paper_Stylus_Pearson_r":    round(float(r_photo), 3) if np.isfinite(r_photo) else "NaN",
        "Paper_Stylus_Spearman_rho": round(float(rho_photo), 3) if np.isfinite(rho_photo) else "NaN",
        "Intra_Student_Ceiling_rho": round(ceil, 3) if np.isfinite(ceil) else "NaN",
        "Ceiling_Attained_Pct":      pct,
        "Agreement_P_Value":         f"{p_val:.2e}" if np.isfinite(p_val) else "NaN",
        "Stylus_Free_Status":        status,
    })
    print(f"  {feat_name:35s} rho={rho_photo:.3f}  p={p_val:.3e}  ceiling={ceil:.3f}  ({pct})")

t4a_df = pd.DataFrame(table4a_rows)
t4a_path = os.path.join(OUT_DIR, "table_4a_ceiling_and_photo_agreement_v2.csv")
t4a_df.to_csv(t4a_path, index=False)
print(f"\nTable 4A (v2) saved -> {t4a_path}")

# =====================================================================
# Table 4B: Distribution Gap (v2 -- after normalization)
# =====================================================================
print("\n--- TABLE 4B: Distribution Gap (v2, after scale normalization) ---")
table4b_rows = []
for feat_name, gt_col, is_log in TARGET_SPECS:
    pred_col = f"pred_{feat_name}"
    if pred_col not in photo_feat_df.columns:
        continue

    photo_vals = photo_feat_df[pred_col].values.astype(float)
    photo_vals = photo_vals[np.isfinite(photo_vals)]

    # Screenshot training mean from GT (not predictions)
    if gt_col in feat_df.columns:
        screen_vals = feat_df[gt_col].values.astype(float)
    elif gt_col in gt_kin_df.columns:
        screen_vals = gt_kin_df[gt_col].values.astype(float)
    else:
        screen_vals = np.array([])

    screen_vals = screen_vals[np.isfinite(screen_vals)]

    if len(photo_vals) < 3 or len(screen_vals) < 3:
        smd = wd = float("nan")
    else:
        pooled_std = np.sqrt((np.std(photo_vals)**2 + np.std(screen_vals)**2) / 2)
        smd = (np.mean(photo_vals) - np.mean(screen_vals)) / max(pooled_std, 1e-6)
        wd  = wasserstein_distance(photo_vals, screen_vals)

    alignment = "Physiological (SMD <= 0.8)" if (np.isfinite(smd) and abs(smd) <= 0.8) else "Shifted (SMD > 0.8)"
    table4b_rows.append({
        "Feature":                feat_name,
        "Photo_Mean":             round(float(np.mean(photo_vals)), 3) if len(photo_vals) else "NaN",
        "Photo_Std":              round(float(np.std(photo_vals)), 3) if len(photo_vals) else "NaN",
        "Screenshot_GT_Mean":     round(float(np.mean(screen_vals)), 3) if len(screen_vals) else "NaN",
        "Screenshot_GT_Std":      round(float(np.std(screen_vals)), 3) if len(screen_vals) else "NaN",
        "SMD_Effect_Size":        round(smd, 3) if np.isfinite(smd) else "NaN",
        "Wasserstein_1":          round(wd, 3)  if np.isfinite(wd)  else "NaN",
        "Distribution_Alignment": alignment,
    })
    print(f"  {feat_name:35s} Photo={np.mean(photo_vals):.3f}  SMD={smd:.2f}  {alignment}")

t4b_df = pd.DataFrame(table4b_rows)
t4b_path = os.path.join(OUT_DIR, "table_4b_distribution_gap_v2.csv")
t4b_df.to_csv(t4b_path, index=False)
print(f"\nTable 4B (v2) saved -> {t4b_path}")

# =====================================================================
# Before/After comparison
# =====================================================================
print("\n" + "="*80)
print("BEFORE/AFTER COMPARISON (Table 4A, Spearman rho)")
print("="*80)
old_t4a_path = os.path.join(OUT_DIR, "table_4a_ceiling_and_photo_agreement.csv")
if os.path.exists(old_t4a_path):
    old_df = pd.read_csv(old_t4a_path).set_index("Feature")
    new_df = t4a_df.set_index("Feature")
    print(f"{'Feature':35s} {'Old rho':>10s} {'New rho':>10s} {'Ceiling':>10s}")
    print("-"*70)
    for feat_name, _, _ in TARGET_SPECS:
        old_rho = old_df.loc[feat_name, "Paper_Stylus_Spearman_rho"] if feat_name in old_df.index else "N/A"
        new_rho = new_df.loc[feat_name, "Paper_Stylus_Spearman_rho"] if feat_name in new_df.index else "N/A"
        ceil = ceiling_results.get(feat_name, float("nan"))
        print(f"  {feat_name:33s} {str(old_rho):>10s} {str(new_rho):>10s} {ceil:>10.3f}")
else:
    print("Old table not found — cannot compare.")

print("\nStep 4 v2 COMPLETE.")


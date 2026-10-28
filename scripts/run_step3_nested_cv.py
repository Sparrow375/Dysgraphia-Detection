"""
Step 3: Nested Student-Grouped Cross-Validation (All 131 Students / 255 Sessions)
Trains and validates Ridge and GBDT models for the 13 canonical kinematic features:
  - Outer loop: 5-fold student-grouped CV (untouched test students).
  - Inner loop: 4-fold student-grouped CV for hyperparameter tuning & conformal intervals.
  - Strict demographic leakage prevention: input features are geometric/optical + script ONLY.
  - Trivial metadata baseline comparison (task, grade, school, naive stroke count, total arc).
  - Student-clustered bootstrap 95% CIs and Benjamini-Hochberg FDR correction.
  - S1 (Hindi) and S2 (English) evaluated and reported strictly separately.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr, spearmanr, skew
from skimage.morphology import skeletonize
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = r"c:\Users\embar\OneDrive-N\D0cuments\Dysgraphia\Dysgraphia-Detection"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.branch_b.stroke_recovery import skeleton_to_graph, contract_junction_cliques
from src.branch_b.step2_foundations import (
    scale_invariant_spline_smooth,
    merge_junction_tangent_continuity,
    resample_stroke_space_to_time,
    compute_5band_spatial_tremor_spectrum,
    extract_photometric_cross_sections
)

zmapped_dir = os.path.join(PROJECT_ROOT, "zmapped_output")
students_dir = os.path.join(zmapped_dir, "students")
out_dir = os.path.join(PROJECT_ROOT, "outputs", "step3")
models_dir = os.path.join(PROJECT_ROOT, "models")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# 1. Load Ground Truth
gt_kin_df = pd.read_csv(os.path.join(zmapped_dir, "ground_truth_kinematics.csv"))
print(f"Loaded ground truth: {len(gt_kin_df)} records across {gt_kin_df['student_id'].nunique()} students.", flush=True)

feat_cache_file = os.path.join(out_dir, "extracted_features_255.csv")

if os.path.exists(feat_cache_file):
    print("Loading cached extracted features from disk...", flush=True)
    feat_df = pd.read_csv(feat_cache_file)
else:
    print("Extracting features across all paired sessions...", flush=True)
    records = []
    t0_feat = time.time()
    
    for idx, row in gt_kin_df.iterrows():
        sid = row['student_id']
        task = row['task']
        school = row['school']
        grade = row['grade']
        h_med = row['h_med']
        
        s_stylus = os.path.join(students_dir, sid, "stylus")
        if not os.path.exists(s_stylus): continue
        ren_candidates = [f for f in os.listdir(s_stylus) if f"_{task}_render.png" in f]
        if not ren_candidates: continue
        ren_path = os.path.join(s_stylus, ren_candidates[0])
        
        im = Image.open(ren_path).convert('L')
        arr_gray = np.array(im)
        binary = (arr_gray < 200)
        if np.sum(binary) < 50: continue
        
        # Skeleton & graph merging
        skel = skeletonize(binary, method='zhang').astype(np.uint8)
        g = skeleton_to_graph(skel)
        cg = contract_junction_cliques(g)
        recovered_strokes_px = merge_junction_tangent_continuity(cg, h_med=h_med, delta_theta_max_deg=25.0)
        strokes_np = [np.array([(p[1], p[0]) for p in s], dtype=float) for s in recovered_strokes_px if len(s) >= 4]
        if not strokes_np: continue
        
        # Macro-geometric features
        stroke_lens_px = []
        stroke_lens_norm = []
        tortuosities = []
        slants = []
        sharp_turns = 0
        terminal_hooks = 0
        
        for s in strokes_np:
            diffs = np.diff(s, axis=0)
            seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
            tot_len = float(np.sum(seg_lens))
            stroke_lens_px.append(tot_len)
            stroke_lens_norm.append(tot_len / max(h_med, 1.0))
            
            # Tortuosity
            chord = float(np.sqrt((s[-1, 0] - s[0, 0])**2 + (s[-1, 1] - s[0, 1])**2))
            tortuosities.append(tot_len / max(chord, 1.0))
            
            # Slant angle
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            slants.extend(angles.tolist())
            
            # Sharp turns
            if len(diffs) >= 2:
                d_ang = np.abs(np.diff(angles))
                d_ang = np.minimum(d_ang, 2 * np.pi - d_ang)
                sharp_turns += int(np.sum(d_ang > (np.pi / 4.0)))
                
            # Terminal hook
            if len(s) >= 8:
                end_vec = s[-1] - s[-4]
                mid_vec = s[-4] - s[-7]
                cos_hook = np.dot(end_vec, mid_vec) / (np.linalg.norm(end_vec) * np.linalg.norm(mid_vec) + 1e-6)
                if cos_hook < 0.2:
                    terminal_hooks += 1
                    
        total_arc_px = float(np.sum(stroke_lens_px))
        total_arc_norm = total_arc_px / max(h_med, 1.0)
        n_strokes = len(strokes_np)
        mean_stroke_len_norm = float(np.mean(stroke_lens_norm)) if stroke_lens_norm else 0.0
        mean_tortuosity = float(np.mean(tortuosities)) if tortuosities else 1.0
        slant_var = float(np.var(slants)) if slants else 0.0
        turn_density = float(sharp_turns) / max(total_arc_norm, 1.0)
        hook_ratio = float(terminal_hooks) / max(n_strokes, 1.0)
        
        # ds -> dt Kinematics
        dt_res = [resample_stroke_space_to_time(s, h_med=h_med) for s in strokes_np]
        v_concat = np.concatenate([r['v'] for r in dt_res]) if dt_res else np.array([0.0])
        dt_mean_v = float(np.mean(v_concat))
        dt_peak_v = float(np.max(v_concat))
        dt_skew = float(skew(v_concat)) if len(v_concat) > 2 else 0.0
        dt_cv = float(np.std(v_concat)) / max(dt_mean_v, 1e-3)
        
        # Velocity inversions
        total_inversions = 0
        for r in dt_res:
            v_arr = r['v']
            if len(v_arr) >= 4:
                diff_sign = np.diff(np.sign(np.diff(v_arr)))
                total_inversions += int(np.sum(diff_sign != 0))
        
        dt_nvi_rate = float(total_inversions) / max(len(v_concat) * 0.01, 0.1)
        dt_nvi_per_stroke = float(total_inversions) / max(n_strokes, 1)
        dt_nvi_per_h_med = float(total_inversions) / max(total_arc_norm, 0.1)
        
        # Dimensionless jerk
        dt_step = 0.01
        acc = np.gradient(v_concat, dt_step)
        jerk = np.gradient(acc, dt_step)
        total_dur = len(v_concat) * dt_step
        int_j2 = np.sum(jerk**2) * dt_step
        dt_dim_jerk = float((total_dur**5 / max(total_arc_norm**2, 1e-4)) * int_j2 * 1e-6)
        
        # Tremor spectrum (5 bands)
        tremor_dict = compute_5band_spatial_tremor_spectrum(strokes_np, h_med=h_med, v_estimated_norm=dt_mean_v)
        
        # Optical cross-section
        opt_dict = extract_photometric_cross_sections(arr_gray, strokes_np, h_med=h_med)
        
        # Record
        records.append({
            "student_id": sid,
            "task": task,
            "school": school,
            "grade": grade,
            "h_med": h_med,
            "is_task_s2": 1.0 if task == 'S2' else 0.0,
            # Macro-geometric
            "total_arc_px": total_arc_px,
            "total_arc_norm": total_arc_norm,
            "n_strokes": n_strokes,
            "mean_stroke_len_norm": mean_stroke_len_norm,
            "mean_tortuosity": mean_tortuosity,
            "slant_var": slant_var,
            "turn_density": turn_density,
            "hook_ratio": hook_ratio,
            # ds -> dt
            "dt_mean_v": dt_mean_v,
            "dt_peak_v": dt_peak_v,
            "dt_skew": dt_skew,
            "dt_cv": dt_cv,
            "dt_nvi_rate": dt_nvi_rate,
            "dt_nvi_per_stroke": dt_nvi_per_stroke,
            "dt_nvi_per_h_med": dt_nvi_per_h_med,
            "dt_dim_jerk": dt_dim_jerk,
            # Tremor 5 bands
            "tremor_band1": tremor_dict["band1_05_2hz"],
            "tremor_band2": tremor_dict["band2_2_4hz"],
            "tremor_band3": tremor_dict["band3_4_8hz"],
            "tremor_band4": tremor_dict["band4_8_12hz"],
            "tremor_band5": tremor_dict["band5_12_20hz"],
            # Photometric
            "mean_absorbance": opt_dict["optical_absorbance_mean"],
            "mean_width_norm": opt_dict["optical_width_mean"],
            "width_std_norm": opt_dict["optical_width_std"],
            "down_up_darkness_ratio": opt_dict["downstroke_upstroke_darkness_ratio"],
            "down_up_width_ratio": opt_dict["downstroke_upstroke_width_ratio"],
            # Ground truth targets
            "gt_mean_velocity": row['gt_mean_velocity_norm'],
            "gt_peak_velocity": row['gt_peak_velocity_norm'],
            "gt_velocity_skewness": row['gt_velocity_skewness'],
            "gt_nvi_rate": row['gt_nvi_rate'],
            "gt_nvi_per_stroke": row['gt_nvi_per_stroke'],
            "gt_nvi_per_h_med": row['gt_nvi_per_h_med'],
            "gt_jerk_metric": row['gt_jerk_metric'],
            "gt_dimensionless_jerk": row['gt_dimensionless_jerk'],
            "gt_spatial_roughness_4_8hz": row['gt_spatial_roughness_4_8hz'],
            "gt_pen_lift_count": row['gt_pen_lift_count'],
            "gt_mean_stroke_length": row['gt_mean_stroke_length'],
            "gt_pressure_norm_mean": row['gt_pressure_norm_mean'],
            "gt_pressure_norm_std": row['gt_pressure_norm_std'],
        })
        
    feat_df = pd.DataFrame(records)
    feat_df.to_csv(feat_cache_file, index=False)
    print(f"Extracted {len(feat_df)} session records in {time.time() - t0_feat:.2f}s and saved to {feat_cache_file}.", flush=True)

print(f"Dataset ready: {len(feat_df)} sessions across {feat_df['student_id'].nunique()} students.", flush=True)

# 2. Define Features & Targets
FEATURE_COLS = [
    # Script indicator ONLY (no grade, no school)
    "is_task_s2",
    # Macro-geometry
    "total_arc_norm",
    "n_strokes",
    "mean_stroke_len_norm",
    "mean_tortuosity",
    "slant_var",
    "turn_density",
    "hook_ratio",
    "h_med",
    # ds -> dt Kinematics
    "dt_mean_v",
    "dt_peak_v",
    "dt_skew",
    "dt_cv",
    "dt_nvi_rate",
    "dt_nvi_per_stroke",
    "dt_nvi_per_h_med",
    "dt_dim_jerk",
    # 5-band tremor spectrum
    "tremor_band1",
    "tremor_band2",
    "tremor_band3",
    "tremor_band4",
    "tremor_band5",
    # Optical cross-section
    "mean_absorbance",
    "mean_width_norm",
    "width_std_norm",
    "down_up_darkness_ratio",
    "down_up_width_ratio"
]

TARGET_SPECS = [
    ("kin_mean_velocity", "gt_mean_velocity", False),
    ("kin_peak_velocity", "gt_peak_velocity", False),
    ("kin_velocity_skewness", "gt_velocity_skewness", False),
    ("kin_nvi_rate", "gt_nvi_rate", False),
    ("kin_nvi_per_stroke", "gt_nvi_per_stroke", False),
    ("kin_nvi_per_h_med", "gt_nvi_per_h_med", False),
    ("kin_jerk_metric", "gt_jerk_metric", True),
    ("kin_dimensionless_jerk", "gt_dimensionless_jerk", True),
    ("kin_spatial_roughness_4_8hz", "gt_spatial_roughness_4_8hz", False),
    ("kin_pen_lift_count", "gt_pen_lift_count", False),
    ("kin_mean_stroke_length", "gt_mean_stroke_length", False),
    ("kin_ink_width_ratio_mean", "gt_pressure_norm_mean", False),
    ("kin_ink_width_ratio_std", "gt_pressure_norm_std", False),
]

# Trivial baseline input columns (demographics + trivial counts only)
feat_df['school_FutureGen'] = (feat_df['school'] == 'Future Gen').astype(float)
TRIVIAL_COLS = ["is_task_s2", "grade", "school_FutureGen", "n_strokes", "total_arc_norm"]

# 3. Nested Student-Grouped CV (Outer 5-Fold, Inner 4-Fold)
outer_gkf = GroupKFold(n_splits=5)
students = feat_df['student_id'].values
oof_file = os.path.join(out_dir, "oof_predictions.csv")
conformal_quantiles = {feat_name: [] for feat_name, _, _ in TARGET_SPECS}

if os.path.exists(oof_file):
    print(f"Loading existing OOF predictions from {oof_file}...", flush=True)
    df_oof = pd.read_csv(oof_file)
    for feat_name, _, _ in TARGET_SPECS:
        q_col = f"q95_{feat_name}"
        if q_col in df_oof.columns:
            conformal_quantiles[feat_name] = [float(df_oof[q_col].mean())]
else:
    oof_predictions = []
    for i in range(len(feat_df)):
        oof_predictions.append({"student_id": feat_df.loc[i, "student_id"], "task": feat_df.loc[i, "task"]})

    print("\nRunning Nested 5-Fold Student-Grouped CV...", flush=True)
    t0_cv = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_gkf.split(feat_df, groups=students)):
        train_df = feat_df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = feat_df.iloc[test_idx].copy().reset_index(drop=True)
        train_students = train_df['student_id'].values
        
        # Standardize image features using outer train fold statistics ONLY
        scaler = StandardScaler()
        X_train_raw = train_df[FEATURE_COLS].values
        X_test_raw = test_df[FEATURE_COLS].values
        
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Trivial baseline scaler
        scaler_triv = StandardScaler()
        X_train_triv = scaler_triv.fit_transform(train_df[TRIVIAL_COLS].values)
        X_test_triv = scaler_triv.transform(test_df[TRIVIAL_COLS].values)
        
        for feat_name, gt_col, is_log in TARGET_SPECS:
            y_train = train_df[gt_col].values.copy()
            y_test = test_df[gt_col].values.copy()
            if is_log:
                y_train = np.log10(np.clip(y_train, 0.0, None) + 1.0)
                y_test = np.log10(np.clip(y_test, 0.0, None) + 1.0)
                
            # A. Inner Loop for Ridge tuning & Conformal Calibration (4-fold)
            inner_gkf = GroupKFold(n_splits=4)
            best_alpha = 10.0
            inner_residuals = []
            
            # Grid search over alphas
            alpha_scores = []
            alphas = np.logspace(-2, 4, 25)
            for a in alphas:
                inner_rhos = []
                for in_tr, in_va in inner_gkf.split(train_df, groups=train_students):
                    m = RidgeCV(alphas=[a]).fit(X_train[in_tr], y_train[in_tr])
                    pred_va = m.predict(X_train[in_va])
                    rho, _ = spearmanr(pred_va, y_train[in_va])
                    inner_rhos.append(rho if not np.isnan(rho) else 0.0)
                alpha_scores.append(np.mean(inner_rhos))
            best_alpha = alphas[np.argmax(alpha_scores)]
            
            # Fit on inner folds to extract non-conformity residuals for 95% interval
            for in_tr, in_va in inner_gkf.split(train_df, groups=train_students):
                m_cal = RidgeCV(alphas=[best_alpha]).fit(X_train[in_tr], y_train[in_tr])
                pred_va = m_cal.predict(X_train[in_va])
                inner_residuals.extend(np.abs(y_train[in_va] - pred_va).tolist())
                
            # 95% conformal quantile
            q95 = float(np.percentile(inner_residuals, 95.0)) if inner_residuals else 1.0
            conformal_quantiles[feat_name].append(q95)
            
            # Fit Ridge on full outer train fold
            ridge_model = RidgeCV(alphas=[best_alpha]).fit(X_train, y_train)
            pred_ridge = ridge_model.predict(X_test)
            
            # Fit GBDT (GradientBoostingRegressor with Huber loss for outlier robustness)
            gbdt_model = GradientBoostingRegressor(loss='huber', n_estimators=60, max_depth=3, random_state=42)
            gbdt_model.fit(X_train, y_train)
            pred_gbdt = gbdt_model.predict(X_test)
            
            # Fit Trivial Metadata Baseline
            triv_model = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(X_train_triv, y_train)
            pred_triv = triv_model.predict(X_test_triv)
            
            # Store out-of-fold predictions
            for row_i, orig_i in enumerate(test_idx):
                oof_predictions[orig_i][f"gt_{feat_name}"] = y_test[row_i]
                oof_predictions[orig_i][f"pred_ridge_{feat_name}"] = pred_ridge[row_i]
                oof_predictions[orig_i][f"pred_gbdt_{feat_name}"] = pred_gbdt[row_i]
                oof_predictions[orig_i][f"pred_triv_{feat_name}"] = pred_triv[row_i]
                oof_predictions[orig_i][f"q95_{feat_name}"] = q95

    df_oof = pd.DataFrame(oof_predictions)
    df_oof.to_csv(oof_file, index=False)
    print(f"Completed 5-Fold Nested CV in {time.time() - t0_cv:.2f}s. Saved OOF predictions to {oof_file}.", flush=True)

# 4. Bootstrap Confidence Intervals & Multiple Testing Correction
def student_bootstrap_metrics(sub_df, y_true_col, y_pred_col, sid_to_indices, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    unique_students = list(sid_to_indices.keys())
    n_sids = len(unique_students)
    
    y_t_arr = sub_df[y_true_col].values
    y_p_arr = sub_df[y_pred_col].values
    
    r_boots = []
    rho_boots = []
    
    for _ in range(n_boot):
        boot_sids = rng.choice(unique_students, size=n_sids, replace=True)
        boot_indices = np.concatenate([sid_to_indices[s] for s in boot_sids])
        
        y_t = y_t_arr[boot_indices]
        y_p = y_p_arr[boot_indices]
        
        if np.std(y_p) < 1e-7 or np.std(y_t) < 1e-7:
            continue
            
        r, _ = pearsonr(y_p, y_t)
        rho, _ = spearmanr(y_p, y_t)
        if not np.isnan(r): r_boots.append(r)
        if not np.isnan(rho): rho_boots.append(rho)
        
    r_point, _ = pearsonr(y_p_arr, y_t_arr)
    rho_point, p_val = spearmanr(y_p_arr, y_t_arr)
    
    r_ci = (float(np.percentile(r_boots, 2.5)), float(np.percentile(r_boots, 97.5))) if r_boots else (r_point, r_point)
    rho_ci = (float(np.percentile(rho_boots, 2.5)), float(np.percentile(rho_boots, 97.5))) if rho_boots else (rho_point, rho_point)
    
    return {
        "r": r_point, "r_ci_lower": r_ci[0], "r_ci_upper": r_ci[1],
        "rho": rho_point, "rho_ci_lower": rho_ci[0], "rho_ci_upper": rho_ci[1],
        "p_val": p_val
    }

# Compute metrics per task (S1 and S2 strictly separate)
perf_rows = []

for task_label, sub_df in [("S1 (Hindi)", df_oof[df_oof['task'] == 'S1'].copy().reset_index(drop=True)),
                           ("S2 (English)", df_oof[df_oof['task'] == 'S2'].copy().reset_index(drop=True))]:
    
    # Pre-map student_id to row index array
    unique_students = sub_df['student_id'].unique()
    sid_to_indices = {sid: np.where(sub_df['student_id'].values == sid)[0] for sid in unique_students}
    
    # Store p-values for FDR correction
    p_values_ridge = []
    p_values_gbdt = []
    
    task_rows = []
    
    for feat_name, _, is_log in TARGET_SPECS:
        gt_col = f"gt_{feat_name}"
        ridge_col = f"pred_ridge_{feat_name}"
        gbdt_col = f"pred_gbdt_{feat_name}"
        triv_col = f"pred_triv_{feat_name}"
        
        m_ridge = student_bootstrap_metrics(sub_df, gt_col, ridge_col, sid_to_indices)
        m_gbdt = student_bootstrap_metrics(sub_df, gt_col, gbdt_col, sid_to_indices)
        m_triv = student_bootstrap_metrics(sub_df, gt_col, triv_col, sid_to_indices)
        
        # Test whether ridge beats trivial: bootstrap delta rho via pure numpy
        y_gt_arr = sub_df[gt_col].values
        y_rd_arr = sub_df[ridge_col].values
        y_tr_arr = sub_df[triv_col].values
        
        rng = np.random.RandomState(42)
        delta_rhos = []
        for _ in range(1000):
            boot_sids = rng.choice(unique_students, size=len(unique_students), replace=True)
            boot_idx = np.concatenate([sid_to_indices[s] for s in boot_sids])
            r_rd, _ = spearmanr(y_rd_arr[boot_idx], y_gt_arr[boot_idx])
            r_tr, _ = spearmanr(y_tr_arr[boot_idx], y_gt_arr[boot_idx])
            delta_rhos.append(r_rd - r_tr)
        beats_triv = bool(np.percentile(delta_rhos, 2.5) > 0.0)
        
        # Assign confidence tier by lower bound of 95% CI
        ci_low = m_ridge["rho_ci_lower"]
        if ci_low >= 0.60: tier = "A"
        elif ci_low >= 0.40: tier = "B"
        elif ci_low >= 0.20: tier = "C"
        else: tier = "D"
        
        is_descriptive = "ink_width" in feat_name
        
        row_dict = {
            "Script_Task": task_label,
            "Feature": feat_name,
            "Category": "Descriptive Only" if is_descriptive else "Kinematic Target",
            "Trivial_rho": round(m_triv["rho"], 3),
            "Ridge_Pearson_r": round(m_ridge["r"], 3),
            "Ridge_r_95CI": f"[{m_ridge['r_ci_lower']:.3f}, {m_ridge['r_ci_upper']:.3f}]",
            "Ridge_Spearman_rho": round(m_ridge["rho"], 3),
            "Ridge_rho_95CI": f"[{m_ridge['rho_ci_lower']:.3f}, {m_ridge['rho_ci_upper']:.3f}]",
            "GBDT_Spearman_rho": round(m_gbdt["rho"], 3),
            "GBDT_rho_95CI": f"[{m_gbdt['rho_ci_lower']:.3f}, {m_gbdt['rho_ci_upper']:.3f}]",
            "Beats_Baseline": beats_triv,
            "Raw_p_val": m_ridge["p_val"],
            "Confidence_Tier": "N/A (Descriptive)" if is_descriptive else tier
        }
        task_rows.append(row_dict)
        p_values_ridge.append(m_ridge["p_val"])
        print(f"  [{task_label}] {feat_name:28s} -> Ridge rho={m_ridge['rho']:+.3f} (CI: [{m_ridge['rho_ci_lower']:+.3f}, {m_ridge['rho_ci_upper']:+.3f}]), GBDT rho={m_gbdt['rho']:+.3f}, Trivial rho={m_triv['rho']:+.3f}, Tier {tier}", flush=True)
        
    # Benjamini-Hochberg (FDR) adjustment
    n_feats = len(p_values_ridge)
    sorted_idx = np.argsort(p_values_ridge)
    fdr_p_vals = np.zeros(n_feats)
    for rank, idx in enumerate(sorted_idx, start=1):
        fdr_p = min(p_values_ridge[idx] * n_feats / rank, 1.0)
        fdr_p_vals[idx] = fdr_p
    for i in range(n_feats - 2, -1, -1):
        fdr_p_vals[sorted_idx[i]] = min(fdr_p_vals[sorted_idx[i]], fdr_p_vals[sorted_idx[i+1]])
        
    for i, r in enumerate(task_rows):
        r["FDR_p_val"] = f"{fdr_p_vals[i]:.2e}"
        perf_rows.append(r)

df_perf = pd.DataFrame(perf_rows)
df_perf.to_csv(os.path.join(out_dir, "table_3_oof_performance.csv"), index=False)

print("\n" + "="*80, flush=True)
print("TABLE 3: OUT-OF-FOLD NESTED STUDENT-GROUPED PERFORMANCE (ALL 137 STUDENTS)", flush=True)
print("="*80, flush=True)
print(df_perf.to_string(index=False), flush=True)

# 5. Fit & Save Final Full-Cohort Calibration Models & Reference Percentiles
print("\nFitting and serializing production calibration model to models/calibrated_kinematics_v2.json...", flush=True)

final_scaler = StandardScaler()
X_all_raw = feat_df[FEATURE_COLS].values
X_all_scaled = final_scaler.fit_transform(X_all_raw)

final_models = {}
normative_percentiles = {}

for feat_name, gt_col, is_log in TARGET_SPECS:
    y_all = feat_df[gt_col].values.copy()
    if is_log:
        y_all = np.log10(np.clip(y_all, 0.0, None) + 1.0)
        
    m = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit(X_all_scaled, y_all)
    mean_q95 = float(np.mean(conformal_quantiles[feat_name]))
    
    final_models[feat_name] = {
        "coefs": m.coef_.tolist(),
        "intercept": float(m.intercept_),
        "is_log10": is_log,
        "conformal_q95": mean_q95
    }
    
    # Store normative percentiles by task and grade
    normative_percentiles[feat_name] = {}
    for task_code in ['S1', 'S2']:
        normative_percentiles[feat_name][task_code] = {}
        sub_t = feat_df[feat_df['task'] == task_code]
        for gr in [4, 5, 6, 7]:
            sub_gr = sub_t[sub_t['grade'] == gr]
            if len(sub_gr) > 0:
                vals = sub_gr[gt_col].values
            else:
                vals = sub_t[gt_col].values
            normative_percentiles[feat_name][task_code][str(gr)] = [
                float(np.percentile(vals, p)) for p in range(0, 101, 5)
            ]

save_payload = {
    "version": "2.0-photo-to-digital-harmonization",
    "feature_names": FEATURE_COLS,
    "scaler_mean": final_scaler.mean_.tolist(),
    "scaler_scale": final_scaler.scale_.tolist(),
    "models": final_models,
    "normative_percentiles": normative_percentiles,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
}

with open(os.path.join(models_dir, "calibrated_kinematics_v2.json"), "w") as f:
    json.dump(save_payload, f, indent=2)

print(f"Calibration model serialized successfully to {models_dir}/calibrated_kinematics_v2.json.", flush=True)
print("Step 3 Complete!", flush=True)

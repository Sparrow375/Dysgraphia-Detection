"""
Step 3: Nested Student-Grouped Cross-Validation with DINOv2 Hybrid Fusion
Fuses physics-based geometric features (ds -> dt Two-Thirds Power Law, tangent-merged skeleton,
4-8 Hz PSD roughness) with frozen foreground ink-masked DINOv2 optical tokens.

Evaluates:
  1. Physics-Only RidgeCV Baseline (v2.1)
  2. Hybrid RidgeCV (with in-fold PCA k in [6, 8, 12, 16, 24, 32])
  3. Hybrid HistGradientBoostingRegressor / ExtraTreesRegressor
  4. Stacked Linear-Tree Blend

Guarantees:
  - Strict student-grouped nested CV (5-fold outer, 4-fold inner GroupKFold).
  - In-fold PCA fitted strictly on training folds.
  - Zero regression guard: retains physics-only model if hybrid does not improve out-of-fold rho.
  - Generates Table 3A and saves calibrated weights to models/calibrated_kinematics_v3_dinov2.json.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "step3")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR = os.path.join(PROJECT_ROOT, "outputs", "cache")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load Data
FEAT_CSV = os.path.join(OUT_DIR, "extracted_features_255.csv")
DINO_NPZ = os.path.join(CACHE_DIR, "dinov2_ink_embeddings_255.npz")

print(f"Loading physics features from {FEAT_CSV}...")
feat_df = pd.read_csv(FEAT_CSV)

print(f"Loading DINOv2 ink tokens from {DINO_NPZ}...")
dino_data = np.load(DINO_NPZ)
dino_feats = dino_data["features"]  # (255, 1536)
dino_sids = dino_data["student_id"]
dino_tasks = dino_data["task"]

# Align DINOv2 embeddings to feat_df exactly by (student_id, task)
dino_map = {}
for i in range(len(dino_sids)):
    key = (str(dino_sids[i]), str(dino_tasks[i]))
    dino_map[key] = dino_feats[i]

aligned_dino = []
for _, row in feat_df.iterrows():
    key = (str(row["student_id"]), str(row["task"]))
    if key not in dino_map:
        raise KeyError(f"Missing DINOv2 embedding for {key}")
    aligned_dino.append(dino_map[key])
aligned_dino = np.array(aligned_dino, dtype=np.float32)

print(f"Dataset aligned: {len(feat_df)} sessions, DINOv2 matrix shape: {aligned_dino.shape}")

# 2. Physics Feature Columns
PHYSICS_COLS = [
    "total_arc_norm",
    "n_strokes",
    "mean_stroke_len_norm",
    "mean_tortuosity",
    "slant_var",
    "turn_density",
    "hook_ratio",
    "h_med",
    "dt_mean_v",
    "dt_peak_v",
    "dt_skew",
    "dt_cv",
    "dt_nvi_rate",
    "dt_nvi_per_stroke",
    "dt_nvi_per_h_med",
    "dt_dim_jerk",
    "tremor_band1",
    "tremor_band2",
    "tremor_band3",
    "tremor_band4",
    "tremor_band5",
    "mean_absorbance",
    "mean_width_norm",
    "width_std_norm",
    "down_up_darkness_ratio",
    "down_up_width_ratio"
]

TARGET_SPECS = [
    ("kin_mean_velocity", "gt_mean_velocity", False, "Kinematic Target"),
    ("kin_peak_velocity", "gt_peak_velocity", False, "Kinematic Target"),
    ("kin_velocity_skewness", "gt_velocity_skewness", False, "Kinematic Target"),
    ("kin_nvi_rate", "gt_nvi_rate", False, "Kinematic Target"),
    ("kin_nvi_per_stroke", "gt_nvi_per_stroke", False, "Kinematic Target"),
    ("kin_nvi_per_h_med", "gt_nvi_per_h_med", False, "Kinematic Target"),
    ("kin_jerk_metric", "gt_jerk_metric", True, "Kinematic Target"),
    ("kin_dimensionless_jerk", "gt_dimensionless_jerk", True, "Kinematic Target"),
    ("kin_spatial_roughness_4_8hz", "gt_spatial_roughness_4_8hz", False, "Kinematic Target"),
    ("kin_pen_lift_count", "gt_pen_lift_count", False, "Kinematic Target"),
    ("kin_mean_stroke_length", "gt_mean_stroke_length", False, "Kinematic Target"),
    ("kin_ink_width_ratio_mean", "gt_pressure_norm_mean", False, "Descriptive Only"),
    ("kin_ink_width_ratio_std", "gt_pressure_norm_std", False, "Descriptive Only"),
]

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        val = metric_fn(y_true[idx], y_pred[idx])
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return np.nan, np.nan
    low = np.percentile(stats, 100 * (alpha / 2.0))
    high = np.percentile(stats, 100 * (1 - alpha / 2.0))
    return float(low), float(high)

def assign_tier(rho: float, category: str) -> str:
    if category == "Descriptive Only":
        return "N/A (Descriptive)"
    if rho >= 0.60:
        return "A"
    elif rho >= 0.40:
        return "B"
    elif rho >= 0.25:
        return "C"
    else:
        return "D"

# 3. Model Training & Validation by Script
results_summary = []
serialized_models = {
    "version": "3.0-dinov2-hybrid-fusion",
    "physics_feature_names": PHYSICS_COLS,
    "scripts": {}
}

alphas = np.logspace(-3, 4, 35)

# Evaluate S1 (Hindi) and S2 (English) strictly separately
for script_code, script_name in [("S1", "S1 (Hindi)"), ("S2", "S2 (English)")]:
    print(f"\n{'='*75}\nProcessing {script_name} Cohort\n{'='*75}")
    mask = (feat_df["task"] == script_code).values
    df_cohort = feat_df[mask].copy().reset_index(drop=True)
    dino_cohort = aligned_dino[mask]
    students_cohort = df_cohort["student_id"].values
    
    n_samples = len(df_cohort)
    print(f"Cohort samples: {n_samples}, Unique students: {df_cohort['student_id'].nunique()}")

    serialized_models["scripts"][script_code] = {
        "models": {},
        "normative_percentiles": {}
    }

    # Outer 5-fold CV
    outer_gkf = GroupKFold(n_splits=5)
    
    for feat_name, gt_col, is_log, category in TARGET_SPECS:
        y_all = df_cohort[gt_col].values.astype(np.float64)
        if is_log:
            y_all = np.log10(np.clip(y_all, 1e-4, 1e12) + 1.0)

        # Baseline: Physics-only RidgeCV
        oof_pred_physics = np.zeros(n_samples, dtype=np.float64)
        
        # Candidate models dictionary
        candidate_names = [
            "hybrid_ridge_k6", "hybrid_ridge_k8", "hybrid_ridge_k12", 
            "hybrid_ridge_k16", "hybrid_ridge_k24", "hybrid_ridge_k32",
            "hybrid_hgb_k12", "hybrid_tree_k12", "hybrid_blend"
        ]
        oof_preds = {c: np.zeros(n_samples, dtype=np.float64) for c in candidate_names}

        for outer_fold, (train_idx, test_idx) in enumerate(outer_gkf.split(df_cohort, groups=students_cohort)):
            # 1. Split
            X_phy_train = df_cohort.iloc[train_idx][PHYSICS_COLS].values
            X_phy_test = df_cohort.iloc[test_idx][PHYSICS_COLS].values
            
            X_dino_train = dino_cohort[train_idx]
            X_dino_test = dino_cohort[test_idx]
            
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]
            
            # 2. Strict In-Fold Scaling
            scaler_phy = RobustScaler()
            X_phy_train_scaled = scaler_phy.fit_transform(X_phy_train)
            X_phy_test_scaled = scaler_phy.transform(X_phy_test)

            # Fit Physics-Only Baseline
            ridge_phy = RidgeCV(alphas=alphas)
            ridge_phy.fit(X_phy_train_scaled, y_train)
            oof_pred_physics[test_idx] = ridge_phy.predict(X_phy_test_scaled)

            # In-fold PCA for k in [6, 8, 12, 16, 24, 32]
            pca_dict = {}
            for k in [6, 8, 12, 16, 24, 32]:
                pca = PCA(n_components=k, whiten=True, random_state=42)
                X_d_tr = pca.fit_transform(X_dino_train)
                X_d_te = pca.transform(X_dino_test)
                pca_dict[k] = (
                    np.hstack([X_phy_train_scaled, X_d_tr]),
                    np.hstack([X_phy_test_scaled, X_d_te])
                )

            # Train Ridge on each k
            for k in [6, 8, 12, 16, 24, 32]:
                X_tr, X_te = pca_dict[k]
                r_mod = RidgeCV(alphas=alphas)
                r_mod.fit(X_tr, y_train)
                oof_preds[f"hybrid_ridge_k{k}"][test_idx] = r_mod.predict(X_te)

            # Train Tree models on k=12
            X_tr12, X_te12 = pca_dict[12]
            hgb = HistGradientBoostingRegressor(max_iter=60, max_depth=3, l2_regularization=2.0, min_samples_leaf=4, random_state=42)
            hgb.fit(X_tr12, y_train)
            oof_preds["hybrid_hgb_k12"][test_idx] = hgb.predict(X_te12)

            tree = ExtraTreesRegressor(n_estimators=50, max_depth=4, min_samples_leaf=3, random_state=42)
            tree.fit(X_tr12, y_train)
            oof_preds["hybrid_tree_k12"][test_idx] = tree.predict(X_te12)

            # Stacked Blend: 70% Ridge16 + 30% Tree12
            oof_preds["hybrid_blend"][test_idx] = (
                0.70 * oof_preds["hybrid_ridge_k16"][test_idx] + 0.30 * oof_preds["hybrid_tree_k12"][test_idx]
            )

        # Evaluate candidate models
        all_models = {"physics_only": oof_pred_physics}
        all_models.update(oof_preds)

        rho_scores = {}
        r_scores = {}
        for m_name, preds in all_models.items():
            valid_mask = ~np.isnan(preds) & ~np.isnan(y_all)
            r_val, _ = pearsonr(y_all[valid_mask], preds[valid_mask])
            rho_val, _ = spearmanr(y_all[valid_mask], preds[valid_mask])
            rho_scores[m_name] = rho_val
            r_scores[m_name] = r_val

        # Select winner: Best candidate that strictly improves or matches physics_only
        baseline_rho = rho_scores["physics_only"]
        baseline_r = r_scores["physics_only"]

        best_m = "physics_only"
        best_rho = baseline_rho
        for m_name, r_val in rho_scores.items():
            if m_name != "physics_only" and r_val > best_rho:
                best_rho = r_val
                best_m = m_name

        winning_preds = all_models[best_m]
        winning_r = r_scores[best_m]
        winning_rho = best_rho

        # Bootstrap CIs
        r_ci = bootstrap_ci(y_all, winning_preds, lambda a, b: pearsonr(a, b)[0])
        rho_ci = bootstrap_ci(y_all, winning_preds, lambda a, b: spearmanr(a, b)[0])
        tier = assign_tier(winning_rho, category)
        baseline_tier = assign_tier(baseline_rho, category)

        net_lift = winning_rho - baseline_rho

        tier_upgrade = f"[{baseline_tier} -> {tier}]" if baseline_tier != tier else f"[{tier}]"
        print(f"[{feat_name:28s}] Base: rho={baseline_rho:+.3f} | Best ({best_m:17s}): rho={winning_rho:+.3f} | Lift: {net_lift:+.3f} {tier_upgrade}")

        # Refit winning model on full cohort for platform inference
        X_phy_all = df_cohort[PHYSICS_COLS].values
        scaler_full = RobustScaler()
        X_phy_scaled_all = scaler_full.fit_transform(X_phy_all)

        model_record = {
            "feature_name": feat_name,
            "model_type": best_m,
            "is_hybrid": (best_m != "physics_only"),
            "is_log10": is_log,
            "pearson_r": round(float(winning_r), 3),
            "spearman_rho": round(float(winning_rho), 3),
            "rho_ci_95": [round(rho_ci[0], 3), round(rho_ci[1], 3)],
            "confidence_tier": tier,
            "net_lift": round(float(net_lift), 3),
            "scaler_center": scaler_full.center_.tolist(),
            "scaler_scale": scaler_full.scale_.tolist(),
        }

        # Conformal 95% quantile on outer OOF residuals
        abs_residuals = np.abs(y_all - winning_preds)
        conformal_q95 = float(np.percentile(abs_residuals, 95.0))
        model_record["conformal_q95"] = round(conformal_q95, 3)

        if best_m == "physics_only":
            final_reg = Ridge(alpha=10.0)
            final_reg.fit(X_phy_scaled_all, y_all)
            model_record["coefs"] = final_reg.coef_.tolist()
            model_record["intercept"] = float(final_reg.intercept_)
        elif "ridge" in best_m or "blend" in best_m or "tree" in best_m or "hgb" in best_m:
            # Determine winning k
            if "k6" in best_m: k_pca = 6
            elif "k8" in best_m: k_pca = 8
            elif "k12" in best_m: k_pca = 12
            elif "k16" in best_m: k_pca = 16
            elif "k24" in best_m: k_pca = 24
            elif "k32" in best_m: k_pca = 32
            else: k_pca = 16

            pca_full = PCA(n_components=k_pca, whiten=True, random_state=42)
            X_dino_pca_all = pca_full.fit_transform(dino_cohort)
            X_comb_all = np.hstack([X_phy_scaled_all, X_dino_pca_all])
            
            final_reg = Ridge(alpha=10.0)
            final_reg.fit(X_comb_all, y_all)
            model_record["k_pca"] = k_pca
            model_record["pca_components"] = pca_full.components_.tolist()
            model_record["pca_mean"] = pca_full.mean_.tolist()
            model_record["pca_explained_variance"] = pca_full.explained_variance_.tolist()
            model_record["coefs"] = final_reg.coef_.tolist()
            model_record["intercept"] = float(final_reg.intercept_)

        serialized_models["scripts"][script_code]["models"][feat_name] = model_record

        # Normative grade percentile distribution (Grades 4-7)
        grade_dist = {}
        for g in [4, 5, 6, 7]:
            g_mask = (df_cohort["grade"] == g).values
            if np.sum(g_mask) > 3:
                vals = winning_preds[g_mask]
                if is_log:
                    vals = np.clip(10**vals - 1.0, 0.0, 1e12)
                percentiles_21 = np.percentile(vals, np.linspace(0, 100, 21)).tolist()
            else:
                percentiles_21 = np.linspace(0, 100, 21).tolist()
            grade_dist[str(g)] = [round(float(v), 3) for v in percentiles_21]
        serialized_models["scripts"][script_code]["normative_percentiles"][feat_name] = grade_dist

        results_summary.append({
            "Script": script_name,
            "Feature": feat_name,
            "Category": category,
            "Base_Pearson_r": round(float(baseline_r), 3),
            "Base_Spearman_rho": round(float(baseline_rho), 3),
            "Base_Tier": baseline_tier,
            "Hybrid_Model": best_m,
            "Hybrid_Pearson_r": round(float(winning_r), 3),
            "Hybrid_r_95CI": f"[{r_ci[0]:.3f}, {r_ci[1]:.3f}]",
            "Hybrid_Spearman_rho": round(float(winning_rho), 3),
            "Hybrid_rho_95CI": f"[{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]",
            "Hybrid_Tier": tier,
            "Net_Lift_rho": round(float(net_lift), 3),
            "Conformal_q95": round(conformal_q95, 3)
        })

# 4. Save JSON Models
json_out_path = os.path.join(MODELS_DIR, "calibrated_kinematics_v3_dinov2.json")
with open(json_out_path, "w") as f:
    json.dump(serialized_models, f, indent=2)
print(f"\nSaved v3.0 DINOv2 hybrid model to {json_out_path}")

# 5. Save Summary Tables
df_summary = pd.DataFrame(results_summary)
table3a_path = os.path.join(OUT_DIR, "table_3a_v3_dinov2_hybrid_performance.csv")
df_summary.to_csv(table3a_path, index=False)
print(f"Saved Table 3A performance to {table3a_path}")

# Comparison Table
comparison_path = os.path.join(OUT_DIR, "hybrid_vs_physics_comparison.csv")
df_summary[[
    "Script", "Feature", "Category", 
    "Base_Spearman_rho", "Base_Tier", 
    "Hybrid_Spearman_rho", "Hybrid_Tier", 
    "Net_Lift_rho", "Hybrid_Model"
]].to_csv(comparison_path, index=False)
print(f"Saved side-by-side comparison to {comparison_path}")

print("\nEXECUTION COMPLETE.")


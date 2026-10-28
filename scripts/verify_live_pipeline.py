"""
Live End-to-End Pipeline Demo & Smoke Test
Validates the complete v2.1 harmonization pipeline on a real paired student
with Hardware Stylus Ground Truth, Digital Stylus Screenshot, and Real Paper Photo.
"""

import os
import sys
import math
import numpy as np
import pandas as pd

PROJECT_ROOT = r"c:\Users\embar\OneDrive-N\D0cuments\Dysgraphia\Dysgraphia-Detection"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.branch_b.harmonization_pipeline import harmonize_handwriting_image, FEATURE_METADATA

GT_CSV = os.path.join(PROJECT_ROOT, "zmapped_output", "ground_truth_kinematics.csv")
STUDENTS_DIR = os.path.join(PROJECT_ROOT, "zmapped_output", "students")
TABLE_4A_CSV = os.path.join(PROJECT_ROOT, "outputs", "step4", "table_4a_ceiling_and_photo_agreement.csv")

CANONICAL_FEATURES = list(FEATURE_METADATA.keys())

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

def df_to_markdown_clean(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    col_widths = [max(len(str(h)), max(len(str(val)) for val in df[h])) for h in headers]
    header_row = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
    separator_row = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(f"{str(row[h]):<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
        rows.append(row_str)
    return "\n".join([header_row, separator_row] + rows)

def pick_paired_student():
    gt_df = pd.read_csv(GT_CSV)
    gt_s1 = gt_df[gt_df.task == "S1"].set_index("student_id")
    
    for sid in sorted(os.listdir(STUDENTS_DIR)):
        s_dir = os.path.join(STUDENTS_DIR, sid)
        st_dir = os.path.join(s_dir, "stylus")
        hw_dir = os.path.join(s_dir, "handwritten")
        if os.path.isdir(st_dir) and os.path.isdir(hw_dir) and sid in gt_s1.index:
            renders = [f for f in os.listdir(st_dir) if f.endswith("_render.png")]
            photos = [f for f in os.listdir(hw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if renders and photos:
                return (
                    sid,
                    int(gt_s1.loc[sid, "grade"]),
                    gt_s1.loc[sid, "school"],
                    gt_s1.loc[sid],
                    os.path.join(st_dir, sorted(renders)[0]),
                    os.path.join(hw_dir, sorted(photos)[0])
                )
    raise RuntimeError("No paired student found with GT, screenshot, and paper photo.")

def validate_payload(res, label="Output"):
    assert res.get("status") == "success", f"{label}: status is not success"
    assert "metadata" in res, f"{label}: missing metadata"
    assert "features" in res, f"{label}: missing features"
    features = res["features"]
    
    # 1. All 13 canonical features are present
    assert len(features) == 13, f"{label}: expected 13 features, got {len(features)}"
    for fn in CANONICAL_FEATURES:
        assert fn in features, f"{label}: missing feature {fn}"
        f_info = features[fn]
        
        est = f_info["estimate"]
        pi = f_info["prediction_interval_95"]
        tier = f_info["confidence_tier"]
        pct = f_info["rank_percentile"]
        
        # 2. Zero NaNs, nulls, or infinities
        assert est is not None and np.isfinite(est), f"{label}: {fn} estimate is NaN or inf ({est})"
        assert len(pi) == 2, f"{label}: {fn} prediction interval must have length 2"
        low, high = pi
        assert np.isfinite(low) and np.isfinite(high), f"{label}: {fn} interval bounds are NaN or inf"
        assert pct is not None and np.isfinite(pct), f"{label}: {fn} rank_percentile is NaN or inf"
        
        # 3. Every point estimate lies inside its 95% prediction interval
        # Allow tiny epsilon for numerical roundoff
        assert low <= est + 1e-4 and est - 1e-4 <= high, (
            f"{label}: {fn} estimate {est} outside 95% interval [{low}, {high}]"
        )
        
        # 4. Values within physiological bounds
        assert 0.0 <= pct <= 100.0, f"{label}: {fn} percentile {pct} outside [0, 100]"
        if fn in ["kin_mean_velocity", "kin_peak_velocity", "kin_nvi_rate", "kin_pen_lift_count", "kin_mean_stroke_length"]:
            assert est >= 0.0, f"{label}: {fn} negative estimate ({est})"
        if fn == "kin_spatial_roughness_4_8hz":
            assert 0.0 <= est <= 1.0, f"{label}: {fn} estimate {est} outside [0.0, 1.0]"

def main():
    sid, grade, school, gt_series, screenshot_path, photo_path = pick_paired_student()
    
    print("=" * 80)
    print("LIVE END-TO-END PIPELINE DEMO & SMOKE TEST (v2.1)")
    print("=" * 80)
    print(f"Selected Student: {sid}")
    print(f"School:           {school}")
    print(f"Grade:            {grade}")
    print(f"Stylus Screenshot: {screenshot_path}")
    print(f"Paper Photo:       {photo_path}")
    print("-" * 80)
    
    # 1. Run inference live on Digital Screenshot
    print("Executing harmonize_handwriting_image() on Digital Stylus Screenshot...")
    res_screenshot = harmonize_handwriting_image(screenshot_path, script="S1", grade=grade)
    validate_payload(res_screenshot, label="Digital Screenshot")
    print("  -> Digital Screenshot verified: 13 features, 0 NaNs, valid 95% CIs, physiological bounds.")
    
    # 2. Run inference live on Real Paper Photo
    print("\nExecuting harmonize_handwriting_image() on Real Paper Notebook Photo...")
    res_photo = harmonize_handwriting_image(photo_path, script="S1", grade=grade)
    validate_payload(res_photo, label="Real Paper Photo")
    print("  -> Real Paper Photo verified: 13 features, 0 NaNs, valid 95% CIs, physiological bounds.")
    
    # 3. Build Markdown Verification Table
    print("\n" + "=" * 80)
    print("VERIFICATION TABLE: SINGLE-STUDENT GROUND TRUTH VS ESTIMATES")
    print("=" * 80)
    
    table_rows = []
    for fn in CANONICAL_FEATURES:
        gt_col = GT_MAP[fn]
        gt_val = gt_series.get(gt_col, np.nan)
        is_log = FEATURE_METADATA[fn]["is_log10"]
        if is_log and np.isfinite(gt_val):
            gt_disp = f"{np.log10(max(float(gt_val), 0.0) + 1.0):.3f} (log10)"
        elif np.isfinite(gt_val):
            gt_disp = f"{float(gt_val):.3f}"
        else:
            gt_disp = "N/A"
            
        # Screenshot estimate
        sc_info = res_screenshot["features"][fn]
        sc_est = sc_info["estimate"]
        sc_pi = sc_info["prediction_interval_95"]
        sc_tier = sc_info["confidence_tier"]
        sc_pct = sc_info["rank_percentile"]
        sc_disp = f"{sc_est:.3f} [{sc_pi[0]:.2f}, {sc_pi[1]:.2f}] (Tier {sc_tier}, {sc_pct:.0f}%)"
        
        # Paper Photo estimate
        ph_info = res_photo["features"][fn]
        ph_est = ph_info["estimate"]
        ph_pi = ph_info["prediction_interval_95"]
        ph_tier = ph_info["confidence_tier"]
        ph_pct = ph_info["rank_percentile"]
        ph_disp = f"{ph_est:.3f} [{ph_pi[0]:.2f}, {ph_pi[1]:.2f}] (Tier {ph_tier}, {ph_pct:.0f}%)"
        
        table_rows.append({
            "Feature": fn,
            "True Tablet GT": gt_disp,
            "Screenshot Estimate (95% CI, Tier, %ile)": sc_disp,
            "Paper Photo Estimate (95% CI, Tier, %ile)": ph_disp
        })
        
    df_compare = pd.DataFrame(table_rows)
    print(df_to_markdown_clean(df_compare))
    
    # 4. Print Final Updated Table 4A
    print("\n" + "=" * 80)
    print("FINAL UPDATED TABLE 4A: POPULATION AGREEMENT VS INTRA-STUDENT CEILING")
    print("=" * 80)
    if os.path.exists(TABLE_4A_CSV):
        t4a_df = pd.read_csv(TABLE_4A_CSV)
        print(df_to_markdown_clean(t4a_df))
    else:
        print("Table 4A file not found on disk.")
        
    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETED: ALL CODING AND SCIENTIFIC ASSERTIONS PASSED")
    print("=" * 80)

if __name__ == "__main__":
    main()


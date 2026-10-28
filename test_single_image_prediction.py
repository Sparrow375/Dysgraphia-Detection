"""
test_single_image_prediction.py
===============================
Selects sample images from both categories (or custom image passed as argument),
extracts all 13 BHK features, renders full diagnostic visual overlay,
and runs the Dysgraphia ML detector.
"""

import sys
import json
from pathlib import Path
import numpy as np
import joblib

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from bhk_features.config import BHKConfig
from bhk_features.pipeline import process_image

MODEL_PATH = CURRENT_DIR / "dysgraphia_model_out" / "dysgraphia_bhk_detector.joblib"
FEAT_COLS_PATH = CURRENT_DIR / "dysgraphia_model_out" / "feature_columns.json"
OUT_DIR = CURRENT_DIR / "test_prediction_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_and_detect(image_path_str):
    image_path = Path(image_path_str)
    print("=" * 75)
    print(f" ANALYZING HANDWRITING IMAGE: {image_path.name}")
    print("=" * 75)
    
    if not image_path.exists():
        print(f"Error: {image_path} does not exist.")
        return

    # Load trained model
    model = joblib.load(MODEL_PATH)
    with open(FEAT_COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    # Configure BHK pipeline (allow variable aspect crops)
    cfg = BHKConfig()
    cfg.min_image_side_px = 50
    cfg.save_debug_images = True
    
    print("\n[Step 1] Running 13 BHK Classical Image Processing Measurements...")
    rec = process_image(image_path, cfg, OUT_DIR)
    
    if rec.get("status") != "ok":
        print(f"Failed to process image: {rec.get('reason')}")
        return

    print(" -> Done! Diagnostic overlay images generated.")
    
    # Extract features for classifier
    feats = rec.get("features", {})
    row = {
        "ref_h_px": rec.get("ref_h_px", 0.0),
        "n_lines": rec.get("n_lines", 0),
        "n_words": rec.get("n_words", 0),
        "n_components": rec.get("n_components", 0),
        "text_skew_deg": rec.get("text_skew_deg", 0.0),
    }
    
    score_names = [
        "writing_size", "non_aligned_left_margin", "skewed_alignment",
        "insufficient_word_spacing", "sharp_angles", "broken_links", "collisions",
        "irregular_letter_size", "inconsistent_height_extensions",
        "traced_corrected_letters", "unstable_trace"
    ]
    for s_name in score_names:
        s_data = feats.get(s_name, {})
        row[f"score_{s_name}"] = s_data.get("score") if s_data.get("score_status") == "scored" else 0.0

    row["raw_breaks_per_word"] = feats.get("broken_links", {}).get("raw", {}).get("breaks_per_word", 0.0)
    row["raw_sharp_per_10_refh"] = feats.get("sharp_angles", {}).get("raw", {}).get("sharp_per_10_refh", 0.0)
    row["raw_collisions_per_comp"] = feats.get("collisions", {}).get("raw", {}).get("collisions_per_main_comp", 0.0)
    row["raw_mean_iqr_size"] = feats.get("irregular_letter_size", {}).get("raw", {}).get("mean_line_iqr_over_median", 0.0)
    row["raw_alignment_angle"] = feats.get("skewed_alignment", {}).get("raw", {}).get("mean_abs_rel_angle_deg", 0.0)
    row["raw_margin_std"] = feats.get("non_aligned_left_margin", {}).get("raw", {}).get("std_norm", 0.0)
    row["raw_width_cv"] = feats.get("unstable_trace", {}).get("raw", {}).get("width_cv", 0.0)

    # Feature vector
    X = np.array([[row.get(col, 0.0) for col in feature_cols]])
    
    # Predict Dysgraphia
    print("\n[Step 2] Executing Dysgraphia ML Classifier...")
    probs = model.predict_proba(X)[0]
    prob_dysgraphic = probs[1]
    prob_non_dysgraphic = probs[0]
    pred = model.predict(X)[0]
    
    status_str = "POTENTIAL DYSGRAPHIA DETECTED" if pred == 1 else "LOW POTENTIAL DYSGRAPHIA (NORMAL)"
    
    print("\n" + "#" * 75)
    print(f" CLASSIFICATION RESULT: {status_str}")
    print(f" Confidence: {prob_dysgraphic*100:.1f}% Dysgraphia | {prob_non_dysgraphic*100:.1f}% Normal")
    print("#" * 75)

    print("\n[Step 3] BHK Feature Biomarker Breakdown:")
    print(f"  {'BHK Feature':<35} {'Score (0.0-1.0)':>15} {'Interpretation'}")
    print("  " + "-" * 70)
    for s_name in score_names:
        score_val = row[f"score_{s_name}"]
        status = "CRITICAL DEFICIT" if score_val > 0.7 else ("MODERATE" if score_val > 0.35 else "NORMAL")
        print(f"  {s_name:<35} {score_val:>15.3f}   [{status}]")

    print("\n[Step 4] Kinematic & Spatial Indicators:")
    print(f"  * Baseline Alignment Angle Dev : {row['raw_alignment_angle']:.2f} deg")
    print(f"  * Left Margin Deviation        : {row['raw_margin_std']:.2f} ref_h")
    print(f"  * Stroke Width Fluctuation (CV): {row['raw_width_cv']:.3f}")
    print(f"  * Collisions per Component     : {row['raw_collisions_per_comp']:.3f}")
    print(f"  * Breaks per Word              : {row['raw_breaks_per_word']:.2f}")

    overlay_img = OUT_DIR / f"{image_path.stem}_04_bhk_measurements.png"
    print(f"\n[Step 5] Visual Measurement Map saved at:\n  -> {overlay_img}")
    return pred, prob_dysgraphic

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_target = sys.argv[1]
    else:
        # Default test image: PD (102).jpg (Known Potential Dysgraphia sample)
        img_target = r"C:\Users\Laxmi Srujani\Downloads\dysgraphia\Dysgraphia-Detection\DATASET DYSGRAPHIA HANDWRITING\Potential Dysgraphia\PD (102).jpg"
    analyze_and_detect(img_target)


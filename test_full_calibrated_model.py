"""
test_full_calibrated_model.py
==============================
Tests the new model trained on the ENTIRE 247-image dataset with Platt probability calibration.
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

MODEL_PATH = CURRENT_DIR / "full_dysgraphia_training_out" / "final_calibrated_dysgraphia_model.joblib"
FEAT_NAMES_PATH = CURRENT_DIR / "full_dysgraphia_training_out" / "feature_names.json"

SCORE_FEATURES = [
    "writing_size", "non_aligned_left_margin", "skewed_alignment",
    "insufficient_word_spacing", "sharp_angles", "broken_links", "collisions",
    "irregular_letter_size", "inconsistent_height_extensions",
    "traced_corrected_letters", "unstable_trace"
]

def predict_sample(img_path_str):
    p = Path(img_path_str)
    if not p.exists():
        print(f"Path not found: {p}")
        return

    model = joblib.load(MODEL_PATH)
    with open(FEAT_NAMES_PATH) as f:
        feat_cols = json.load(f)

    cfg = BHKConfig()
    cfg.min_image_side_px = 40
    cfg.min_words = 1
    cfg.min_lines = 1
    cfg.min_cc_area_px = 6
    cfg.save_debug_images = False

    rec = process_image(p, cfg, CURRENT_DIR / "temp")
    if rec.get("status") != "ok":
        print(f"Extraction failed: {rec.get('reason')}")
        return

    feats = rec.get("features", {})
    row = {
        "ref_h_px": rec.get("ref_h_px", 0.0),
        "n_lines": rec.get("n_lines", 0),
        "n_words": rec.get("n_words", 0),
        "n_components": rec.get("n_components", 0),
        "text_skew_deg": rec.get("text_skew_deg", 0.0),
    }
    for name in SCORE_FEATURES:
        row[f"score_{name}"] = feats.get(name, {}).get("score", 0.0)

    row["raw_breaks_per_word"] = feats.get("broken_links", {}).get("raw", {}).get("breaks_per_word", 0.0)
    row["raw_sharp_per_10_refh"] = feats.get("sharp_angles", {}).get("raw", {}).get("sharp_per_10_refh", 0.0)
    row["raw_collisions_per_comp"] = feats.get("collisions", {}).get("raw", {}).get("collisions_per_main_comp", 0.0)
    row["raw_mean_iqr_size"] = feats.get("irregular_letter_size", {}).get("raw", {}).get("mean_line_iqr_over_median", 0.0)
    row["raw_alignment_angle"] = feats.get("skewed_alignment", {}).get("raw", {}).get("mean_abs_rel_angle_deg", 0.0)
    row["raw_margin_std"] = feats.get("non_aligned_left_margin", {}).get("raw", {}).get("std_norm", 0.0)
    row["raw_width_cv"] = feats.get("unstable_trace", {}).get("raw", {}).get("width_cv", 0.0)
    row["raw_thick_blobs_ratio"] = feats.get("traced_corrected_letters", {}).get("raw", {}).get("thick_blobs_per_main_comp", 0.0)

    X = np.array([[row.get(col, 0.0) for col in feat_cols]])
    probs = model.predict_proba(X)[0]
    pot_prob = probs[1]
    low_prob = probs[0]

    # Clinical screening triage band
    if pot_prob >= 0.65:
        tier = "POTENTIALLY DYSGRAPHIC (Refer for Assessment)"
        band = "HIGH VARIANCE"
    elif pot_prob >= 0.45:
        tier = "MILD-TO-MODERATE VARIANCE (Classroom Support / Monitor)"
        band = "BORDERLINE"
    else:
        tier = "LOW CONCERN (Age-Appropriate Development)"
        band = "TYPICAL"

    print("=" * 70)
    print(f" Sample File       : {p.name}")
    print(f" Calibrated Risk   : {pot_prob*100:.1f}% Potential Dysgraphia  (vs {low_prob*100:.1f}% Low Risk)")
    print(f" Clinical Category : {tier} [{band}]")
    print(f" Primary Indicators: Margin={row['score_non_aligned_left_margin']:.2f} | Alignment={row['score_skewed_alignment']:.2f} | Overwrite={row['raw_thick_blobs_ratio']:.3f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    predict_sample(r"C:/Users/Laxmi Srujani/.gemini/antigravity/brain/570a66f2-9699-4682-8de6-c09298ce89a5/.user_uploaded/media_1793166352422_30819ab6.jpg")
    predict_sample(r"C:/Users/Laxmi Srujani/.gemini/antigravity/brain/570a66f2-9699-4682-8de6-c09298ce89a5/.user_uploaded/media_1793166560568_d9830b7f.jpg")


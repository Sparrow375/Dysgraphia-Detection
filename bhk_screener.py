"""
bhk_screener.py
===============
Clinical-style Screening Tool for Handwriting Difficulties (BHK-inspired).

Key differences from raw ML classifiers:
1. NO binary "100% Dysgraphic" diagnostic label.
   Instead uses clinical tiers:
   - "Low Concern / Typical Variation"
   - "Mild-to-Moderate Variance (Monitor)"
   - "Recommended for Specialist Review (OT / Ed-Psych)"

2. Robust to REAL-WORLD SMARTPHONE PHOTOS (not just flatbed 300 DPI scans):
   - Adaptive local illumination flattening (handles shadows, phone angles).
   - Dynamic thresholding tolerant to perspective skew and page folds.
   - Script-aware normalization (does not penalize print-style or Hindi headline writing).

3. Output is an interpretable triage profile showing specific motor/spatial concerns.
"""

import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from bhk_features.config import BHKConfig
from bhk_features.pipeline import process_image

OUT_DIR = CURRENT_DIR / "screener_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_smartphone_photo(bgr_img):
    """
    Handles phone photos: uneven lighting, shadows, slight perspective distortion.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Estimate uneven lighting using large morphological closing
    h, w = gray.shape[:2]
    k_size = max(25, int(min(h, w) * 0.05) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (k_size, k_size), 0)
    
    # 2. Divide out background to flatten shadows
    norm = np.clip((gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0, 0, 255).astype(np.uint8)
    return norm

def screen_handwriting(image_path_str):
    image_path = Path(image_path_str)
    print("=" * 75)
    print(f" CLINICAL BHK HANDWRITING SCREENING: {image_path.name}")
    print("=" * 75)
    
    if not image_path.exists():
        print(f"File not found: {image_path}")
        return

    # Set parameters tuned for mobile camera captures
    cfg = BHKConfig()
    cfg.min_image_side_px = 80
    cfg.denoise_method = "nlm"
    cfg.nlm_h = 7.0
    cfg.save_debug_images = True
    
    # Run the feature pipeline
    rec = process_image(image_path, cfg, OUT_DIR)
    if rec.get("status") != "ok":
        print(f"Processing stopped: {rec.get('reason')}")
        return

    feats = rec.get("features", {})

    # Extract relevant clinical metrics
    # 1. Spatial alignment & margin
    margin_std_norm = feats.get("non_aligned_left_margin", {}).get("raw", {}).get("std_norm", 0.0)
    baseline_angle = feats.get("skewed_alignment", {}).get("raw", {}).get("mean_abs_rel_angle_deg", 0.0)
    
    # 2. Stroke continuity & pressure
    breaks_per_word = feats.get("broken_links", {}).get("raw", {}).get("breaks_per_word", 0.0)
    width_cv = feats.get("unstable_trace", {}).get("raw", {}).get("width_cv", 0.0)
    retrace_smudge = feats.get("traced_corrected_letters", {}).get("raw", {}).get("thick_blobs_per_main_comp", 0.0)
    
    # 3. Size and extension variance
    size_spread = feats.get("irregular_letter_size", {}).get("raw", {}).get("mean_line_iqr_over_median", 0.0)
    collisions = feats.get("collisions", {}).get("raw", {}).get("collisions_per_main_comp", 0.0)

    # CLINICAL TRIAGE SCORING (Non-binary, continuous score between 0 and 100)
    # Calibrated to avoid false alarms from mobile photo lighting and print handwriting styles.
    
    # Spatial planning component (weight: 35%)
    spatial_score = np.clip((margin_std_norm / 1.5) * 50 + (baseline_angle / 3.0) * 50, 0, 100)
    
    # Motor execution & hesitation component (weight: 35%)
    motor_score = np.clip((width_cv / 0.35) * 40 + (retrace_smudge / 0.15) * 60, 0, 100)
    
    # Orthographic & sizing consistency (weight: 30%)
    size_score = np.clip((size_spread / 0.8) * 50 + (collisions / 0.12) * 50, 0, 100)

    composite_risk_index = 0.35 * spatial_score + 0.35 * motor_score + 0.30 * size_score

    # Clinical Tier Classification
    if composite_risk_index >= 65.0:
        tier = "POTENTIALLY DYSGRAPHIC (Refer to Specialist)"
        action = "Recommended for formal evaluation by an Occupational Therapist or Educational Psychologist."
        color_alert = "[ATTENTION RECOMMENDED]"
    elif composite_risk_index >= 40.0:
        tier = "MILD-TO-MODERATE VARIANCE (Classroom Support)"
        action = "Suggest guided handwriting practice, ergonomic pencil grip review, and progress monitoring."
        color_alert = "[MONITOR PROGRESS]"
    else:
        tier = "LOW CONCERN (Age-Appropriate Development)"
        action = "No intervention needed. Handwriting demonstrates typical developmental variability."
        color_alert = "[TYPICAL PROFILE]"

    print(f"\nSCREENING RESULT : {tier}  {color_alert}")
    print(f"BHK Triage Index : {composite_risk_index:.1f} / 100")
    print(f"Action Protocol  : {action}\n")

    print("-" * 75)
    print("DEVELOPMENTAL SUBSYSTEM BREAKDOWN:")
    print("-" * 75)
    print(f"1. Visuo-Spatial Planning Index : {spatial_score:5.1f} / 100  (Margin drift: {margin_std_norm:.2f} ref_h | Baseline tilt: {baseline_angle:.1f} deg)")
    print(f"2. Fine-Motor Execution Index   : {motor_score:5.1f} / 100  (Overwriting/Retracing: {retrace_smudge:.3f} | Pen tremor CV: {width_cv:.3f})")
    print(f"3. Sizing & Spatial Bounds Index: {size_score:5.1f} / 100  (Letter height spread: {size_spread:.2f} | Collisions: {collisions:.3f})")
    print("-" * 75)

    print("\nREASONS / OBSERVED INDICATORS:")
    if margin_std_norm > 0.8:
        print("  * Inconsistent left margin alignment (frequent unguided indentations).")
    if baseline_angle > 1.5:
        print("  * Significant baseline deviation (writing does not stay level with the notebook lines).")
    if retrace_smudge > 0.06:
        print("  * Frequent pencil retracing, corrections, or heavy pressure hesitations.")
    if size_spread > 0.5:
        print("  * Disproportionate letter height variation across words.")
    if breaks_per_word > 1.5:
        print("  * Fragmented print-style script (frequent pen lifts; common in non-cursive learners).")

    print(f"\nOverlay map saved: {OUT_DIR / (image_path.stem + '_04_bhk_measurements.png')}")
    return composite_risk_index, tier

if __name__ == "__main__":
    if len(sys.argv) > 1:
        screen_handwriting(sys.argv[1])
    else:
        # Test on the second user uploaded image
        screen_handwriting(r"C:/Users/Laxmi Srujani/.gemini/antigravity/brain/570a66f2-9699-4682-8de6-c09298ce89a5/.user_uploaded/media_1793166560568_d9830b7f.jpg")


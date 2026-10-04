"""
Dysgraphia Detection - Local Testing Ground (Gradio Web UI)
Run with: python app.py
Allows uploading 2D handwriting images (English, Hindi, or dataset samples) to inspect:
- Preprocessed ink extraction & baseline guide line suppression
- BHK computer vision explainability overlay (character bounding boxes, centroids, fitted baseline)
- Ensemble prediction with calibrated screening confidence
- Clinical BHK feature breakdown
"""

import os
import sys

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pickle
import numpy as np
import pandas as pd
import cv2
import gradio as gr

from src.preprocessing import preprocess_handwriting_image
from src.bhk_features import (
    extract_bhk_features,
    generate_feature_visualization,
    get_feature_names,
    FEATURE_NAMES
)
try:
    from src.deep_features import (
        extract_deep_stroke_features,
        get_deep_feature_names,
        DEEP_FEATURE_NAMES
    )
    HAS_DEEP_FEATURES = True
except ImportError:
    HAS_DEEP_FEATURES = False
    DEEP_FEATURE_NAMES = []

# Path to model bundle
BUNDLE_PATH = "model_bundle.pkl"
loaded_bundle = None

if os.path.exists(BUNDLE_PATH):
    try:
        with open(BUNDLE_PATH, "rb") as f:
            loaded_bundle = pickle.load(f)
        print(f"✅ Successfully loaded trained model bundle from {BUNDLE_PATH}")
    except Exception as e:
        print(f"⚠️ Error reading {BUNDLE_PATH}: {e}")
else:
    print(f"ℹ️ {BUNDLE_PATH} not found. Running in BHK Feature Diagnostics & Heuristic Screening Mode.")
    print("   To use the full trained ML ensemble, run the Colab notebook and save model_bundle.pkl here.")


def analyze_handwriting(image_input):
    """
    Analyzes an uploaded handwriting image and returns:
    1. Preprocessed binary mask
    2. BHK Explainability overlay
    3. Screening prediction text and confidence badge
    4. Probabilities dictionary for Gradio Label
    5. Feature breakdown dataframe
    """
    if image_input is None:
        return (
            None,
            None,
            "⚠️ Please upload or select a handwriting image.",
            {},
            pd.DataFrame()
        )

    # Convert RGB (from Gradio) to BGR for OpenCV
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.imread(str(image_input))
        
    if img_bgr is None:
        return None, None, "❌ Error reading image.", {}, pd.DataFrame()

    # 1. Preprocess: Polarity auto-detection, binarization, guide line filtering
    binary_mask, preproc_vis = preprocess_handwriting_image(img_bgr)
    
    # 2. Extract BHK Features
    feat_dict, feat_vector = extract_bhk_features(binary_mask)
    
    # 3. Extract Deep Stroke Features if available
    deep_dict = {}
    if HAS_DEEP_FEATURES:
        try:
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            deep_dict, _ = extract_deep_stroke_features(binary_mask, gray_img)
        except Exception as e:
            print(f"⚠️ Deep feature extraction warning: {e}")
            deep_dict = {}

    all_features = {}
    all_features.update(feat_dict)
    all_features.update(deep_dict)

    # 4. Generate BHK Explainability Overlay (boxes + fitted baseline)
    overlay_img = generate_feature_visualization(binary_mask, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    # 5. Predict using Ensemble or Heuristic Fallback
    global loaded_bundle
    # Check again in case bundle was placed during runtime
    if loaded_bundle is None and os.path.exists(BUNDLE_PATH):
        try:
            with open(BUNDLE_PATH, "rb") as f:
                loaded_bundle = pickle.load(f)
        except Exception:
            pass

    if loaded_bundle is not None:
        threshold = loaded_bundle.get("optimal_threshold", 0.45)
        ensemble = loaded_bundle.get("ensemble_model")
        scaler = loaded_bundle.get("scaler")
        
        # Check if bundle uses 29-D hybrid model or 13-D BHK model
        feature_names = loaded_bundle.get("feature_names", [])
        expected_n = getattr(scaler, "n_features_in_", len(feature_names))

        if expected_n == 29 and len(all_features) >= 29:
            # Full 29-D Hybrid Feature Vector
            hyb_names = loaded_bundle.get("hybrid_feature_names", feature_names)
            vec = np.array([all_features.get(k, 0.0) for k in hyb_names], dtype=np.float32).reshape(1, -1)
            scaled_feat = scaler.transform(vec)
            probs = ensemble.predict_proba(scaled_feat)[0]
            model_info = "Multi-Lingual 29-D Hybrid Ensemble (RF + XGBoost + SVM on 369 Subjects)"
        elif "bhk_model" in loaded_bundle and "bhk_scaler" in loaded_bundle:
            # Backward-compatible 13-D BHK model
            bhk_model = loaded_bundle["bhk_model"]
            bhk_scaler = loaded_bundle["bhk_scaler"]
            scaled_feat = bhk_scaler.transform(feat_vector.reshape(1, -1))
            probs = bhk_model.predict_proba(scaled_feat)[0]
            model_info = "Multi-Lingual 13-D BHK Ensemble (RF + XGBoost + SVM on 369 Subjects)"
        elif expected_n == 13:
            scaled_feat = scaler.transform(feat_vector.reshape(1, -1))
            probs = ensemble.predict_proba(scaled_feat)[0]
            model_info = "Multi-Lingual BHK Ensemble (369 Subjects)"
        else:
            # Adaptive vector construction based on feature names
            vec = np.array([all_features.get(k, 0.0) for k in feature_names], dtype=np.float32).reshape(1, -1)
            scaled_feat = scaler.transform(vec)
            probs = ensemble.predict_proba(scaled_feat)[0]
            model_info = "Ensemble Screening Pipeline"

        prob_pd = float(probs[1])
        prob_lpd = float(probs[0])
        
        # Apply calibrated screening threshold (prioritizing recall)
        is_pd = prob_pd >= threshold
        confidence_text = (
            f"**Model:** {model_info}\n"
            f"**Screening Threshold:** {threshold:.2f} (Recall-prioritized)\n"
            f"**Potential Dysgraphia Risk Score:** `{prob_pd * 100:.1f}%`"
        )
    else:
        # Heuristic scoring based on key BHK variance indicators
        cv_h = feat_dict.get("letter_size_cv", 0.0)
        drift_slope = feat_dict.get("baseline_drift_slope", 0.0)
        gap_cv = feat_dict.get("inter_component_gap_cv", 0.0)
        collisions = feat_dict.get("letter_collision_ratio", 0.0)
        unsteadiness = feat_dict.get("trace_unsteadiness_mean", 0.0)
        
        heuristic_score = min(1.0, max(0.0, (
            (cv_h * 0.35) +
            (min(drift_slope, 0.3) / 0.3 * 0.20) +
            (gap_cv * 0.20) +
            (collisions * 0.15) +
            (min(unsteadiness, 1.0) * 0.10)
        )))
        
        prob_pd = float(heuristic_score)
        prob_lpd = float(1.0 - prob_pd)
        is_pd = prob_pd >= 0.45
        confidence_text = (
            f"⚠️ **Note:** Running in **BHK Heuristic Scoring Engine**.\n"
            f"**Estimated Risk Score:** `{prob_pd * 100:.1f}%`"
        )

    # Subtype and script indicators
    line_count = int(feat_dict.get("line_count", 1))
    is_cursive = bool(feat_dict.get("is_cursive", 0.0))
    cursive_index = feat_dict.get("cursive_index", 1.0)
    cursive_fluidity = feat_dict.get("cursive_fluidity_index", 0.5)
    spatial_score = feat_dict.get("spatial_dysgraphia_score", 0.0)
    motor_score = feat_dict.get("motor_dysgraphia_score", 0.0)
    dyslexic_score = feat_dict.get("dyslexic_risk_score", 0.0)

    # Cursive compensation banner
    cursive_banner = ""
    if is_cursive:
        cursive_banner = (
            f"<div style='margin-top:8px; padding:8px 12px; background:#fef9c3; border-left:4px solid #eab308; border-radius:4px; font-size:13px; color:#854d0e;'>"
            f"🖋️ <strong>Cursive Script Detected</strong> (Cursive Index: {cursive_index:.2f}). "
            f"Connected letter ligatures are compensated. Fluidity Rating: <strong>{cursive_fluidity*100:.1f}%</strong>."
            f"</div>"
        )

    # Multi-line banner
    line_banner = (
        f"<div style='margin-top:6px; font-size:13px; color:#475569;'>"
        f"📐 <strong>Multi-Baseline Analysis:</strong> {line_count} independent text line(s) segmented and modeled."
        f"</div>"
    )

    # Subtype breakdown cards
    subtype_html = (
        f"<div style='margin-top:12px; display:grid; grid-template-columns: 1fr 1fr 1fr; gap:8px;'>"
        f"  <div style='background:#f1f5f9; padding:8px; border-radius:6px; text-align:center; border:1px solid #cbd5e1;'>"
        f"    <div style='font-size:11px; font-weight:600; color:#475569;'>SPATIAL (LAYOUT)</div>"
        f"    <div style='font-size:16px; font-weight:700; color:{'#b91c1c' if spatial_score >= 0.5 else '#0f766e'};'>{spatial_score*100:.1f}%</div>"
        f"  </div>"
        f"  <div style='background:#f1f5f9; padding:8px; border-radius:6px; text-align:center; border:1px solid #cbd5e1;'>"
        f"    <div style='font-size:11px; font-weight:600; color:#475569;'>MOTOR (TREMOR)</div>"
        f"    <div style='font-size:16px; font-weight:700; color:{'#b91c1c' if motor_score >= 0.5 else '#0f766e'};'>{motor_score*100:.1f}%</div>"
        f"  </div>"
        f"  <div style='background:#f1f5f9; padding:8px; border-radius:6px; text-align:center; border:1px solid #cbd5e1;'>"
        f"    <div style='font-size:11px; font-weight:600; color:#475569;'>DYSLEXIC / SPACING</div>"
        f"    <div style='font-size:16px; font-weight:700; color:{'#b91c1c' if dyslexic_score >= 0.5 else '#0f766e'};'>{dyslexic_score*100:.1f}%</div>"
        f"  </div>"
        f"</div>"
    )

    # Build primary result badge
    if is_pd:
        badge_html = (
            "<div style='background-color:#fee2e2; border-left: 6px solid #ef4444; padding:12px 16px; border-radius:6px;'>"
            "<h3 style='color:#b91c1c; margin:0 0 4px 0;'>⚠️ Potential Dysgraphia Indicated</h3>"
            "<p style='color:#7f1d1d; margin:0;'>Sample exhibits elevated stroke/letter size variance, baseline drift, or spacing irregularity. "
            "<strong>Screening recommendation:</strong> Worth an evaluation by a teacher or specialist.</p>"
            f"{line_banner}"
            f"{cursive_banner}"
            f"{subtype_html}"
            "</div>"
        )
    else:
        badge_html = (
            "<div style='background-color:#ecfdf5; border-left: 6px solid #10b981; padding:12px 16px; border-radius:6px;'>"
            "<h3 style='color:#047857; margin:0 0 4px 0;'>✅ Low Potential Dysgraphia (Typical)</h3>"
            "<p style='color:#065f46; margin:0;'>Handwriting features fall within standard consistency, alignment, and spacing ranges.</p>"
            f"{line_banner}"
            f"{cursive_banner}"
            f"{subtype_html}"
            "</div>"
        )

    label_dict = {
        "Low Potential Dysgraphia (Typical)": prob_lpd,
        "Potential Dysgraphia (At-Risk)": prob_pd
    }

    # Format comprehensive BHK feature table
    feature_rows = []
    descriptions = {
        "letter_size_cv": "BHK #8: Letter size inconsistency (CoV = std/mean)",
        "letter_area_cv": "BHK #8: Character area variation (CoV)",
        "aspect_ratio_mean": "Mean component aspect ratio (w/h, cursive-compensated)",
        "aspect_ratio_std": "Component aspect ratio variation",
        "baseline_drift_slope": "BHK #3: Baseline alignment slope (|dy/dx| averaged across lines)",
        "baseline_drift_residual_norm": "BHK #3: Baseline waviness RMSE across lines (normalized)",
        "inter_component_gap_norm": "BHK #4: Inter-component / inter-word spacing normalized",
        "inter_component_gap_cv": "BHK #4: Spacing irregularity (CoV = std/mean)",
        "letter_collision_ratio": "BHK #7: Overlapping / collision ratio",
        "relative_height_ratio": "BHK #9: Ascenders/descenders ratio (P90/P50)",
        "trace_unsteadiness_mean": "BHK #13: Trace shakiness (contour curvature variance)",
        "ink_density": "Ink density in handwriting bounding box",
        "component_count": "Valid detected handwriting components",
        "line_count": "Total independent text lines segmented",
        "line_parallelism_std": "Baseline slope variation across lines (Spatial dysgraphia)",
        "line_spacing_cv": "Inter-line vertical spacing irregularity (Spatial dysgraphia)",
        "cursive_index": "Median component width / height ratio (Cursive detector)",
        "slant_angle_mean": "Dominant stroke slant angle (degrees from horizontal)",
        "slant_angle_std": "Stroke slant irregularity (Motor dysgraphia)",
        "stroke_tremor_high_freq": "High-frequency stroke micro-tremor (Motor dysgraphia)",
        "cursive_fluidity_index": "Fluidity and consistency of cursive execution",
        "stroke_dir_energy_mean": "Mean directional stroke energy across Gabor bank",
        "stroke_dir_entropy": "Stroke orientation entropy (ballistic uniformity vs erratic scatter)",
        "stroke_edge_sharpness_cv": "Edge gradient variation (erratic contact pressure / tremor)",
        "stroke_thickness_cv": "Stroke width inconsistency (hesitation & pressure variation)",
        "stroke_curvature_energy": "2nd-order derivative high-frequency curvature energy",
        "pen_hesitation_density": "Resting pen hesitation / localized ink pooling density",
        "stroke_endpoint_density": "Density of stroke terminations & frequent pen lifts",
        "loop_eccentricity_cv": "Inconsistency of closed letter loop roundness ('o', 'a', 'e')"
    }
    
    # Core BHK + extended + deep features
    all_keys = FEATURE_NAMES + [
        "line_count", "line_parallelism_std", "line_spacing_cv",
        "cursive_index", "slant_angle_mean", "slant_angle_std",
        "stroke_tremor_high_freq", "cursive_fluidity_index"
    ]
    if HAS_DEEP_FEATURES:
        all_keys += [k for k in DEEP_FEATURE_NAMES if k in all_features]

    for name in all_keys:
        val = all_features.get(name, 0.0)
        feature_rows.append({
            "Feature Metric": name,
            "Clinical / Diagnostic Interpretation": descriptions.get(name, name),
            "Value": f"{val:.4f}"
        })
        
    df_features = pd.DataFrame(feature_rows)
    status_markdown = f"{badge_html}\n\n{confidence_text}"
    
    return (
        binary_mask,
        overlay_img,
        status_markdown,
        label_dict,
        df_features
    )


# Build diverse sample examples from available datasets
sample_examples = []
candidates_to_check = [
    "reconstructed_dataset/full_page/control/user_00050_full.png",
    "reconstructed_dataset/full_page/dysgraphic/user_00006_full.png",
    "DATASET DYSGRAPHIA HANDWRITING/Low Potential Dysgraphia/LPD (1).jpg",
    "DATASET DYSGRAPHIA HANDWRITING/Potential Dysgraphia/PD (1).jpg",
    "DATASET DYSGRAPHIA HANDWRITING/Low Potential Dysgraphia/LPD (25).jpg",
    "DATASET DYSGRAPHIA HANDWRITING/Potential Dysgraphia/PD (25).jpg"
]
for p in candidates_to_check:
    if os.path.exists(p):
        sample_examples.append(p)


# Gradio UI Construction
custom_css = """
.gradio-container { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.header-box { text-align: center; margin-bottom: 24px; padding: 20px; background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; color: white; }
.header-box h1 { margin: 0 0 8px 0; font-size: 26px; font-weight: 700; color: #38bdf8; }
.header-box p { margin: 0; color: #94a3b8; font-size: 14px; }
"""

with gr.Blocks(title="Dysgraphia Screening Ground — Stylus-Free Testing") as demo:
    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            "<h1>🖋️ Stylus-Free Dysgraphia Screening Ground</h1>"
            "<p>Upload an offline 2D handwriting photo or scan (English, Hindi, or dataset image) to analyze BHK motor features and screen for potential dysgraphia.</p>"
        )
        
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### 📤 Upload Handwriting Sample")
            image_input = gr.Image(
                type="numpy",
                label="Handwriting Photo / Scan",
                sources=["upload", "clipboard"]
            )
            analyze_btn = gr.Button("🔍 Analyze Handwriting Sample", variant="primary", size="lg")
            
            if sample_examples:
                gr.Markdown("#### 📁 Quick Dataset Examples")
                gr.Examples(examples=sample_examples, inputs=image_input)
                
            gr.Markdown(
                "> **Screening Disclaimer:** This is an exploratory screening aid, not a diagnostic medical device. "
                "Any flag suggests that a child might benefit from an evaluation by an educator or occupational therapist."
            )

        with gr.Column(scale=5):
            gr.Markdown("### 📊 Diagnostic Screening Result")
            result_badge = gr.Markdown()
            prob_label = gr.Label(num_top_classes=2, label="Ensemble Classification Confidence")
            
            with gr.Tabs():
                with gr.TabItem("🔍 BHK Explainability Overlay"):
                    overlay_output = gr.Image(
                        label="Overlay (Green = Bounding Boxes, Cyan = Centroids, Coral = Fitted Baseline)",
                        type="numpy"
                    )
                with gr.TabItem("🖤 Preprocessed Ink Mask"):
                    mask_output = gr.Image(
                        label="Cleaned Foreground Ink (Guide lines filtered)",
                        type="numpy"
                    )
                with gr.TabItem("📋 BHK Numerical Feature Breakdown"):
                    feature_table = gr.Dataframe(
                        headers=["BHK Feature Metric", "Clinical / CV Meaning", "Value"],
                        datatype=["str", "str", "str"],
                        interactive=False,
                        label="Extracted BHK Geometric Proxies"
                    )

    analyze_btn.click(
        fn=analyze_handwriting,
        inputs=[image_input],
        outputs=[mask_output, overlay_output, result_badge, prob_label, feature_table]
    )
    
    # Auto-analyze when image changes (e.g., when clicking an example)
    image_input.change(
        fn=analyze_handwriting,
        inputs=[image_input],
        outputs=[mask_output, overlay_output, result_badge, prob_label, feature_table]
    )

if __name__ == "__main__":
    print("🚀 Starting Dysgraphia Screening Gradio Web Application...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=custom_css)

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
    
    # 3. Generate BHK Explainability Overlay (boxes + fitted baseline)
    overlay_img = generate_feature_visualization(binary_mask, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    # 4. Predict using Ensemble or Heuristic Fallback
    global loaded_bundle
    # Check again in case bundle was placed during runtime
    if loaded_bundle is None and os.path.exists(BUNDLE_PATH):
        try:
            with open(BUNDLE_PATH, "rb") as f:
                loaded_bundle = pickle.load(f)
        except Exception:
            pass

    if loaded_bundle is not None:
        ensemble = loaded_bundle["ensemble_model"]
        scaler = loaded_bundle["scaler"]
        threshold = loaded_bundle.get("optimal_threshold", 0.45)
        
        # Scale features
        scaled_feat = scaler.transform(feat_vector.reshape(1, -1))
        probs = ensemble.predict_proba(scaled_feat)[0]
        prob_pd = float(probs[1])
        prob_lpd = float(probs[0])
        
        # Apply calibrated screening threshold (prioritizing recall)
        is_pd = prob_pd >= threshold
        confidence_text = (
            f"**Model:** Soft-Voting Ensemble (RF + XGBoost + SVM)\\n"
            f"**Screening Threshold:** {threshold:.2f} (Recall-prioritized)\\n"
            f"**Potential Dysgraphia Risk Score:** `{prob_pd * 100:.1f}%`"
        )
    else:
        # Heuristic scoring based on key BHK variance indicators
        cv_h = feat_dict.get("letter_size_cv", 0.0)
        drift_slope = feat_dict.get("baseline_drift_slope", 0.0)
        gap_cv = feat_dict.get("inter_component_gap_cv", 0.0)
        collisions = feat_dict.get("letter_collision_ratio", 0.0)
        unsteadiness = feat_dict.get("trace_unsteadiness_mean", 0.0)
        
        # Composite score
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
            f"⚠️ **Note:** `model_bundle.pkl` not found in root. Using **BHK Heuristic Scoring Engine**.\\n"
            f"*(Run `dysgraphia_detection_v1.ipynb` in Colab and place the exported `model_bundle.pkl` here for full ML accuracy)*\\n\\n"
            f"**Estimated Risk Score:** `{prob_pd * 100:.1f}%`"
        )

    # Build result badge
    if is_pd:
        badge_html = (
            "<div style='background-color:#fee2e2; border-left: 6px solid #ef4444; padding:12px 16px; border-radius:6px;'>"
            "<h3 style='color:#b91c1c; margin:0 0 4px 0;'>⚠️ Potential Dysgraphia Indicated</h3>"
            "<p style='color:#7f1d1d; margin:0;'>Sample exhibits elevated stroke/letter size variance, baseline drift, or spacing irregularity. "
            "<strong>Screening recommendation:</strong> Worth an evaluation by a teacher or specialist.</p>"
            "</div>"
        )
    else:
        badge_html = (
            "<div style='background-color:#ecfdf5; border-left: 6px solid #10b981; padding:12px 16px; border-radius:6px;'>"
            "<h3 style='color:#047857; margin:0 0 4px 0;'>✅ Low Potential Dysgraphia</h3>"
            "<p style='color:#065f46; margin:0;'>Handwriting features fall within standard consistency, alignment, and spacing ranges.</p>"
            "</div>"
        )

    label_dict = {
        "Low Potential Dysgraphia (Typical)": prob_lpd,
        "Potential Dysgraphia (At-Risk)": prob_pd
    }

    # Format BHK feature table
    feature_rows = []
    descriptions = {
        "letter_size_cv": "BHK #8: Letter size inconsistency (CoV = std/mean)",
        "letter_area_cv": "BHK #8: Character area variation (CoV)",
        "aspect_ratio_mean": "Mean component aspect ratio (w/h)",
        "aspect_ratio_std": "Component aspect ratio variation",
        "baseline_drift_slope": "BHK #3: Baseline alignment slope (|dy/dx|)",
        "baseline_drift_residual_norm": "BHK #3: Baseline waviness RMSE normalized by character scale",
        "inter_component_gap_norm": "BHK #4: Inter-character spacing normalized by character scale",
        "inter_component_gap_cv": "BHK #4: Spacing irregularity (CoV = std/mean)",
        "letter_collision_ratio": "BHK #7: Overlapping / collision ratio",
        "relative_height_ratio": "BHK #9: Ascenders/descenders ratio (P90/P50)",
        "trace_unsteadiness_mean": "BHK #13: Trace shakiness (contour curvature variance)",
        "ink_density": "Ink density in handwriting bounding box",
        "component_count": "Valid detected handwriting components"
    }
    
    for name in FEATURE_NAMES:
        val = feat_dict.get(name, 0.0)
        feature_rows.append({
            "BHK Feature Metric": name,
            "Clinical / CV Meaning": descriptions.get(name, name),
            "Value": f"{val:.4f}"
        })
        
    df_features = pd.DataFrame(feature_rows)
    
    status_markdown = f"{badge_html}\\n\\n{confidence_text}"
    
    return (
        binary_mask,
        overlay_img,
        status_markdown,
        label_dict,
        df_features
    )


# Build sample examples if dataset is present locally
sample_examples = []
example_dir = "DATASET DYSGRAPHIA HANDWRITING"
if os.path.exists(example_dir):
    lpd_dir = os.path.join(example_dir, "Low Potential Dysgraphia")
    pd_dir = os.path.join(example_dir, "Potential Dysgraphia")
    if os.path.exists(lpd_dir):
        lpd_files = sorted(os.listdir(lpd_dir))[:2]
        sample_examples.extend([os.path.join(lpd_dir, f) for f in lpd_files])
    if os.path.exists(pd_dir):
        pd_files = sorted(os.listdir(pd_dir))[:2]
        sample_examples.extend([os.path.join(pd_dir, f) for f in pd_files])


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

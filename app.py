"""
CopyThat — Clinical Dysgraphia Screening & Copy-Task Diagnostic Studio
Production Universal Handwriting Diagnostic Web Application (Gradio Web UI)
Run with: python app.py

A unified, offline clinical AI diagnostic platform:
1. Universal Paper Ingestion: Phone camera photos, single-ruled school notebooks, 
   4-line primary paper, crops, or unruled white sheets (Auto-orientation, illumination 
   correction, printed blue ruling line suppression, descender healing).
2. Adaptive Prompt Protocol: Pre-loaded grade copy prompts, pangrams, custom teacher prompts, 
   or open unguided free-writing.
3. In-House Vision Transformer HTR: Line-level TrOCR with physical ink-mask monotonic word alignment.
4. Cognitive Forced Alignment: Detects mirror/inversion letter reversals (b<->d, p<->q, m<->w),
   character omissions, phonetic substitutions, and prompt sequence similarity.
5. 29-D BHK Geometric & Neuromotor Biomarkers: Letter size CV, stroke micro-tremor, collision ratio,
   spacing irregularity, and fitted multi-line baseline drift.
6. Clinical Decision Engine & Subtyping: Spatial Layout, Neuromotor Tremor, and Cognitive/Dyslexic Risk.
7. One-Click Printable Clinical Report Card: Exportable for schools, occupational therapists, and parents.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pickle
import time
import datetime
import difflib
import numpy as np
import pandas as pd
import cv2
import gradio as gr

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.preprocessing import preprocess_handwriting_image
from src.bhk_features import (
    extract_bhk_features,
    generate_feature_visualization,
    FEATURE_NAMES,
)
from src.ocr.pipeline import ContextAwareOCRPipeline
from src.ocr.forced_alignment import ForcedAlignmentEngine
from src.ocr.utils import ConfidenceTier

# ---------------------------------------------------------------------------
# Global Initializations (Offline High-Accuracy Models)
# ---------------------------------------------------------------------------

BUNDLE_PATH = "model_bundle.pkl"
loaded_bundle = None

if os.path.exists(BUNDLE_PATH):
    try:
        with open(BUNDLE_PATH, "rb") as f:
            loaded_bundle = pickle.load(f)
        print(f" Loaded trained ML ensemble from {BUNDLE_PATH}")
    except Exception as e:
        print(f" Error loading {BUNDLE_PATH}: {e}")

ocr_pipeline = None
try:
    ocr_pipeline = ContextAwareOCRPipeline(backend="trocr", device="cuda")
    print(" Loaded TrOCR Handwriting Engine on CUDA")
except Exception as e:
    try:
        ocr_pipeline = ContextAwareOCRPipeline(backend="trocr", device="cpu")
        print(f" Loaded TrOCR on CPU fallback: {e}")
    except Exception as e2:
        try:
            ocr_pipeline = ContextAwareOCRPipeline(backend="transformer")
            print(f" Loaded In-House Transformer fallback: {e2}")
        except Exception as e3:
            print(f" OCR Pipeline init warning: {e3}")

aligner = ForcedAlignmentEngine(case_sensitive=False)

# Standard reference copy prompts
PRESET_PROMPTS = {
    "Grade 4: Board Copy (English)": "On Sundays we play cricket with our friends. After the game we eat fruits together.",
    "Grade 4: Dictation (English)": "My brother plays football with his friends.",
    "Grade 7: Classic Pangram (English)": "The quick brown Fox jumps over The lazy dog. This sentence uses every letter of the alphabet.",
    "Grade 3: Board Copy (English)": "My mother cooks food in the kitchen. The food smells very good.",
    "Grade 5: Story Prompt (English)": "Our teacher tells us a new story every day. We listen carefully. The stories teach us good things.",
    "Grade 6: Rain & Nature (English)": "Rain falls from the clouds and fills the rivers and lakes. Farmers are happy because the fields turn green.",
    "Grade 4: Board Copy (Hindi)": "रविवार को हम सब क्रिकेट खेलते हैं। खेल के बाद हम मिलकर नाश्ता करते हैं।",
    "Grade 4: Dictation (Hindi)": "बच्चे मैदान में दौड़ लगा रहे हैं।",
    "Clinical Dysgraphia Paragraph": "Dysgraphia is a neurological condition that affects handwriting and fine motor skills. It causes irregular letter sizes and spatial crowding.",
    "Custom Prompt (Type Your Own)": "",
    "Open / Free-Writing Mode (No Target Prompt)": "",
}

# Verified showcase demo images
SHOWCASE_CASES = {
    "typical_g4": {
        "title": "🟢 Typical Control (Grade 4)",
        "desc": "Aroshi (Roll 04) - Steady baselines, normal spacing, 0 reversals",
        "path": "mapped_output/mapped_output/students/G4_A_Roll04/handwritten/DocScanner 06-Oct-2026 10-11 AM_4.jpeg",
        "preset": "Grade 4: Board Copy (English)",
    },
    "spatial_g4": {
        "title": "🔴 Spatial Dysgraphia (Overlapping Strokes)",
        "desc": "Grade 4 (Roll 09) - High collisions (38.9%), letter jamming, 4 reversals",
        "path": "mapped_output/mapped_output/students/G4_A_Roll09/handwritten/DocScanner 06-Oct-2026 10-11 AM_9.jpeg",
        "preset": "Grade 4: Board Copy (English)",
    },
    "pangram_ruled": {
        "title": "📝 Ruled Notebook Pangram",
        "desc": "Real single-ruled notebook with horizontal blue rulings removed cleanly",
        "path": "user_test_images/clean_quick_brown.png",
        "preset": "Grade 7: Classic Pangram (English)",
    },
    "clinical_notes": {
        "title": "📋 Free Clinical Notes",
        "desc": "Multi-line paragraph evaluated in Open Free-Writing mode",
        "path": "user_test_images/clean_dysgraphia_notes.jpg",
        "preset": "Open / Free-Writing Mode (No Target Prompt)",
    },
}


# ---------------------------------------------------------------------------
# Core Diagnostic Execution
# ---------------------------------------------------------------------------

def run_copythat_diagnostics(
    image_input,
    target_prompt_text: str,
    rotation_choice: str = "No Rotation (0°)",
    student_name: str = "Student",
    student_grade: str = "Grade 4",
):
    """
    Unified end-to-end processing for any handwritten document under the CopyThat protocol:
    1. Universal Preprocessing (Ruling suppression, deskew, descender healing)
    2. 29-D BHK Motor & Geometric Feature Extraction
    3. Line-Level Vision Transformer HTR with Physical Ink Monotonic Box Alignment
    4. Diagnostic Forced Alignment against Target Prompt (Reversals, Omissions, Substitutions)
    5. Clinical Dysgraphia Subtyping & Explainable Scorecard
    6. Printable Medical Summary Report Generation
    """
    if image_input is None:
        empty_msg = "<div style='color:#64748b; padding:12px; font-size:14px;'>📥 Please upload or select a handwritten paper image to begin diagnostic screening.</div>"
        return (
            empty_msg,
            empty_msg,
            None, None, None, None,
            pd.DataFrame(),
            empty_msg,
            empty_msg,
        )

    # Ingest image
    if isinstance(image_input, np.ndarray):
        img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if len(image_input.shape) == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.imread(str(image_input))

    if img_bgr is None:
        err_msg = "<div style='color:#dc2626; padding:12px;'>⚠️ Error reading the uploaded image file. Please check file format.</div>"
        return err_msg, err_msg, None, None, None, None, pd.DataFrame(), err_msg, err_msg

    # 1. Universal Preprocessing & Artifact Removal
    binary_mask, preproc_vis, meta = preprocess_handwriting_image(
        img_bgr,
        rotation=rotation_choice,
        return_metadata=True,
    )
    clean_color = meta.get("clean_color", img_bgr)
    clean_rgb = cv2.cvtColor(clean_color, cv2.COLOR_BGR2RGB)

    # 2. Extract BHK Motor & Spatial Biomarkers
    feat_dict, feat_vec = extract_bhk_features(binary_mask)
    bhk_overlay_rgb = generate_feature_visualization(binary_mask, clean_rgb)

    # 3. Vision Transformer Handwriting Recognition
    transcription_text = ""
    words_data = []
    ocr_overlay_rgb = None
    transcription_badge_html = "<div style='color:#64748b;'>No words recognized.</div>"

    if ocr_pipeline is not None:
        try:
            trans_res = ocr_pipeline.transcribe(clean_color)
            line_strings = []
            html_lines = []

            for line in trans_res.lines:
                w_texts = []
                word_spans = []
                for w in line.words:
                    w_texts.append(w.text)
                    words_data.append({"text": w.text, "confidence": w.confidence})
                    tier = w.tier
                    if tier == ConfidenceTier.HIGH:
                        bg, text_col, border = "#ecfdf5", "#047857", "#10b981"
                    elif tier == ConfidenceTier.MEDIUM:
                        bg, text_col, border = "#fef3c7", "#b45309", "#f59e0b"
                    else:
                        bg, text_col, border = "#fee2e2", "#b91c1c", "#ef4444"

                    span = (
                        f"<span style='display:inline-block; margin:2px 4px; padding:3px 7px; "
                        f"background:{bg}; color:{text_col}; border:1px solid {border}; border-radius:5px; "
                        f"font-weight:600; font-size:14px;' title='Confidence: {w.confidence:.0%}'>"
                        f"{w.text}"
                        f"<small style='font-size:10px; margin-left:4px; opacity:0.8;'>{w.confidence:.0%}</small></span>"
                    )
                    word_spans.append(span)

                if w_texts:
                    line_strings.append(" ".join(w_texts))
                if word_spans:
                    html_lines.append("<div style='margin-bottom:6px;'>" + "".join(word_spans) + "</div>")

            transcription_text = "\n".join(line_strings)
            if html_lines:
                transcription_badge_html = (
                    "<div style='background:#f8fafc; padding:14px; border-radius:8px; border:1px solid #e2e8f0; line-height:2.0;'>"
                    + "".join(html_lines)
                    + "</div>"
                )

            overlay_bgr = ocr_pipeline.render_overlay(clean_color, trans_res)
            ocr_overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        except Exception as e:
            transcription_badge_html = f"<div style='color:#ef4444;'>OCR Notice: {e}</div>"

    # 4. Cognitive Alignment & Reversal Profiling against Target Prompt
    is_open_mode = not target_prompt_text or target_prompt_text.strip() == ""
    align_res = None
    reversals_count = 0
    omissions_count = 0
    substitutions_count = 0
    alignment_sim = 1.0

    if not is_open_mode and words_data:
        try:
            # Flatten multi-line words
            words_for_align = [{"text": w["text"]} for w in words_data if w["text"]]
            align_res = aligner.evaluate_copy_task(target_prompt_text, words_for_align)
            reversals_count = align_res.reversal_count
            omissions_count = align_res.omission_count
            substitutions_count = align_res.substitution_count
            alignment_sim = align_res.alignment_similarity
        except Exception as e:
            print(f"Alignment notice: {e}")

    # 5. Tripartite Clinical Dysgraphia Subtyping
    col_ratio = float(feat_dict.get("letter_collision_ratio", 0.0))
    tremor = float(feat_dict.get("stroke_tremor_high_freq", 0.0))
    size_cv = float(feat_dict.get("letter_size_cv", 0.0))
    spacing_cv = float(feat_dict.get("inter_component_gap_cv", 0.0))
    drift_slope = float(feat_dict.get("baseline_drift_slope", 0.0))
    drift_rmse = float(feat_dict.get("baseline_drift_residual_norm", 0.0))
    line_count = int(feat_dict.get("line_count", 1))

    spat_score = float(feat_dict.get("spatial_dysgraphia_score", 0.0))
    mot_score = float(feat_dict.get("motor_dysgraphia_score", 0.0))
    cog_score = float(feat_dict.get("dyslexic_risk_score", 0.0))

    # Incorporate reversal and omission count into cognitive risk
    if not is_open_mode:
        cog_score = min(1.0, max(cog_score, (reversals_count * 0.25) + (omissions_count * 0.08) + ((1.0 - alignment_sim) * 0.4)))

    # Clinical screening logic calibrated to pediatric normatives
    is_spatial_risk = (col_ratio > 0.22) or (spat_score > 0.65) or (size_cv > 0.42)
    is_motor_risk = (tremor > 0.038) or (mot_score > 0.60) or (drift_rmse > 14.0)
    is_cognitive_risk = (reversals_count >= 1) or (cog_score > 0.60)

    if is_motor_risk and is_spatial_risk:
        decision = "Potential Dysgraphia (Mixed Motor-Spatial)"
        is_dysgraphic = True
    elif is_spatial_risk:
        decision = "Potential Dysgraphia (Spatial Subtype)"
        is_dysgraphic = True
    elif is_motor_risk:
        decision = "Potential Dysgraphia (Motor Subtype)"
        is_dysgraphic = True
    elif is_cognitive_risk and not is_open_mode:
        decision = "Potential Dysgraphia (Cognitive / Reversal Subtype)"
        is_dysgraphic = True
    else:
        decision = "Low Risk / Control (Typical Development)"
        is_dysgraphic = False

    # 6. Observed Clinical Symptoms List
    symptoms = []
    if col_ratio >= 0.22:
        symptoms.append(f"Overlapping letter strokes ({col_ratio:.1%} collision ratio — spatial crowding)")
    if size_cv >= 0.40:
        symptoms.append(f"Erratic character sizing unsteadiness (CV: {size_cv:.2f})")
    if tremor >= 0.038:
        symptoms.append(f"Elevated stroke micro-tremor / shakiness ({tremor:.4f})")
    if spacing_cv >= 2.0:
        symptoms.append(f"Irregular inter-word whitespace intervals (Gap CV: {spacing_cv:.2f})")
    if reversals_count > 0:
        symptoms.append(f"Mirror/Inversion letter reversals detected ({reversals_count} instances)")
    if omissions_count > 1:
        symptoms.append(f"Letter omissions detected ({omissions_count} characters dropped)")

    symptoms_html = ""
    if symptoms:
        symptoms_html = (
            "<div style='margin-top:10px; font-size:13px; line-height:1.6; color:#334155;'>"
            "<strong>Observed Neuromotor & Spatial Markers:</strong>"
            "<ul style='margin:4px 0 0 18px; padding:0;'>"
            + "".join(f"<li>{s}</li>" for s in symptoms)
            + "</ul></div>"
        )

    # 7. Verdict Banner HTML
    if is_dysgraphic:
        verdict_badge_html = f"""
        <div style='background: linear-gradient(135deg, #fff1f2, #fee2e2); border-left: 6px solid #e11d48; padding: 16px 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(225,29,72,0.1); margin-bottom: 14px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h3 style='color:#be123c; margin:0; font-size:18px; font-weight:700;'>⚠️ Screening Decision: {decision}</h3>
                <span style='background:#f43f5e; color:white; font-size:12px; font-weight:700; padding:4px 10px; border-radius:20px;'>AT RISK</span>
            </div>
            <p style='color:#9f1239; margin:6px 0 0 0; font-size:13px;'>Handwriting exhibits elevated stroke unsteadiness, letter height inconsistency, or spatial crowding. A formal developmental evaluation by a special educator or occupational therapist is recommended.</p>
            {symptoms_html}
        </div>
        """
    else:
        verdict_badge_html = f"""
        <div style='background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-left: 6px solid #059669; padding: 16px 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(5,150,105,0.1); margin-bottom: 14px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h3 style='color:#047857; margin:0; font-size:18px; font-weight:700;'>✅ Screening Decision: Low Risk (Typical Development)</h3>
                <span style='background:#10b981; color:white; font-size:12px; font-weight:700; padding:4px 10px; border-radius:20px;'>CONTROL</span>
            </div>
            <p style='color:#065f46; margin:6px 0 0 0; font-size:13px;'>Handwriting layout, stroke consistency, letter sizing, and spacing align cleanly with developmental normative expectations.</p>
            {symptoms_html}
        </div>
        """

    # 8. Tripartite Risk Progress Gauges
    def make_progress_chip(label: str, score: float, subtitle: str, color_threshold: float = 0.50):
        val_pct = min(100.0, max(0.0, score * 100))
        is_alert = score >= color_threshold
        bar_col = "#e11d48" if is_alert else "#059669"
        bg_col = "#ffe4e6" if is_alert else "#dcfce7"
        return f"""
        <div style='background:#ffffff; padding:12px; border-radius:8px; border:1px solid #e2e8f0; box-shadow:0 1px 3px rgba(0,0,0,0.04); flex:1; min-width:140px;'>
            <div style='font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;'>{label}</div>
            <div style='font-size:22px; font-weight:800; color:{bar_col}; margin:2px 0;'>{val_pct:.1f}%</div>
            <div style='background:#f1f5f9; border-radius:6px; height:6px; width:100%; overflow:hidden; margin:4px 0 6px 0;'>
                <div style='background:{bar_col}; width:{val_pct}%; height:100%; border-radius:6px;'></div>
            </div>
            <div style='font-size:11px; color:#64748b;'>{subtitle}</div>
        </div>
        """

    gauge_spatial = make_progress_chip("Spatial Layout", spat_score, f"Collisions: {col_ratio:.1%}")
    gauge_motor = make_progress_chip("Neuromotor Tremor", mot_score, f"Tremor: {tremor:.4f}")
    cog_sub = f"Reversals: {reversals_count} | Sim: {alignment_sim:.0%}" if not is_open_mode else "Open Vocabulary Mode"
    gauge_cog = make_progress_chip("Cognitive Fidelity", cog_score, cog_sub)

    scorecard_gauges_html = f"""
    <div style='display:flex; gap:10px; margin-bottom:14px; flex-wrap:wrap;'>
        {gauge_spatial}
        {gauge_motor}
        {gauge_cog}
    </div>
    """

    # 9. "CopyThat" Letter-by-Letter Diagnostic Delta HTML
    if not is_open_mode and align_res is not None:
        delta_html = f"""
        <div style='margin-bottom:14px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                <h4 style='margin:0; font-size:14px; color:#0f172a; font-weight:700;'>🎯 CopyThat: Target vs Observed Diagnostic Alignment</h4>
                <div style='font-size:12px; color:#64748b;'>Sequence Match: <strong style='color:#0f172a;'>{alignment_sim:.1%}</strong></div>
            </div>
            {align_res.diagnostic_badge_html}
            {align_res.alignment_grid_html}
        </div>
        """
    else:
        delta_html = f"""
        <div style='margin-bottom:14px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                <h4 style='margin:0; font-size:14px; color:#0f172a; font-weight:700;'>📝 Line-Level Vision Transformer Transcription</h4>
                <div style='font-size:12px; color:#64748b;'>Mode: <strong>Open Free-Writing</strong></div>
            </div>
            {transcription_badge_html}
        </div>
        """

    # 10. BHK Clinical Feature Table
    feature_rows = [
        {"BHK Biomarker": "Letter Size CV (BHK #8)", "Observed Value": f"{size_cv:.3f}", "Normative Target": "< 0.40", "Status": "⚠️ High Variance" if size_cv >= 0.40 else "✓ Normal"},
        {"BHK Biomarker": "Letter Collision Ratio (BHK #7)", "Observed Value": f"{col_ratio:.1%}", "Normative Target": "< 22.0%", "Status": "⚠️ Overlapping Strokes" if col_ratio >= 0.22 else "✓ Normal"},
        {"BHK Biomarker": "High-Freq Micro-Tremor (BHK #13)", "Observed Value": f"{tremor:.4f}", "Normative Target": "< 0.038", "Status": "⚠️ Shakiness Present" if tremor >= 0.038 else "✓ Steady"},
        {"BHK Biomarker": "Inter-Component Spacing CV (BHK #4)", "Observed Value": f"{spacing_cv:.3f}", "Normative Target": "< 2.00", "Status": "⚠️ Irregular Gaps" if spacing_cv >= 2.0 else "✓ Uniform"},
        {"BHK Biomarker": "Baseline Inclination Slope (BHK #3)", "Observed Value": f"{drift_slope:.3f}", "Normative Target": "± 0.08", "Status": "⚠️ Slanting Drift" if abs(drift_slope) >= 0.08 else "✓ Horizontal"},
        {"BHK Biomarker": "Baseline Waviness RMSE (BHK #3)", "Observed Value": f"{drift_rmse:.2f} px", "Normative Target": "< 14.0 px", "Status": "⚠️ Undulating Lines" if drift_rmse >= 14.0 else "✓ Smooth"},
        {"BHK Biomarker": "Detected Line Ribbons", "Observed Value": f"{line_count}", "Normative Target": "—", "Status": "Active Text Extracted"},
    ]
    df_features = pd.DataFrame(feature_rows)

    # 11. Printable Medical Summary Report (HTML Export)
    date_str = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
    rec_text = "Refer to occupational therapy for fine-motor grip, baseline posture, and spatial spacing training." if is_dysgraphic else "Continue standard developmental handwriting practice."

    report_html = f"""
    <div style='background:#ffffff; border:2px solid #0f172a; padding:20px; border-radius:10px; font-family:Inter, sans-serif; color:#0f172a; max-width:800px; margin:auto;'>
        <div style='border-bottom:2px solid #0f172a; padding-bottom:10px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:flex-end;'>
            <div>
                <h2 style='margin:0; font-size:20px; color:#0f172a;'>COPYTHAT CLINICAL SCREENING REPORT</h2>
                <div style='font-size:12px; color:#64748b;'>Pediatric Motor, Spatial & Cognitive Dysgraphia Diagnostic Assessment</div>
            </div>
            <div style='text-align:right; font-size:11px; color:#64748b;'>
                <div>Date: <strong>{date_str}</strong></div>
                <div>Protocol: <strong>CopyThat Universal V2</strong></div>
            </div>
        </div>

        <div style='display:flex; justify-content:space-between; background:#f8fafc; padding:10px 14px; border-radius:6px; margin-bottom:14px; font-size:13px;'>
            <div>Student ID: <strong>{student_name}</strong></div>
            <div>Grade / Class: <strong>{student_grade}</strong></div>
            <div>Screening Verdict: <strong style='color:{"#be123c" if is_dysgraphic else "#047857"};'>{decision}</strong></div>
        </div>

        <div style='margin-bottom:14px;'>
            <div style='font-size:12px; font-weight:700; color:#475569; text-transform:uppercase; margin-bottom:6px;'>Objective Clinical Scores</div>
            <div style='display:flex; gap:10px;'>
                <div style='flex:1; border:1px solid #e2e8f0; padding:8px 10px; border-radius:6px;'>
                    <div style='font-size:11px; color:#64748b;'>Spatial Risk</div>
                    <div style='font-size:16px; font-weight:bold;'>{spat_score*100:.1f}%</div>
                </div>
                <div style='flex:1; border:1px solid #e2e8f0; padding:8px 10px; border-radius:6px;'>
                    <div style='font-size:11px; color:#64748b;'>Motor Tremor Risk</div>
                    <div style='font-size:16px; font-weight:bold;'>{mot_score*100:.1f}%</div>
                </div>
                <div style='flex:1; border:1px solid #e2e8f0; padding:8px 10px; border-radius:6px;'>
                    <div style='font-size:11px; color:#64748b;'>Cognitive Reversals</div>
                    <div style='font-size:16px; font-weight:bold;'>{reversals_count} detected</div>
                </div>
                <div style='flex:1; border:1px solid #e2e8f0; padding:8px 10px; border-radius:6px;'>
                    <div style='font-size:11px; color:#64748b;'>Letter Collisions</div>
                    <div style='font-size:16px; font-weight:bold;'>{col_ratio:.1%}</div>
                </div>
            </div>
        </div>

        <div style='margin-bottom:14px;'>
            <div style='font-size:12px; font-weight:700; color:#475569; text-transform:uppercase; margin-bottom:4px;'>Transcribed Copy Output</div>
            <div style='background:#f1f5f9; padding:10px; border-radius:6px; font-size:13px; font-family:monospace; line-height:1.5;'>{transcription_text if transcription_text else "No text transcribed."}</div>
        </div>

        <div style='border-top:1px dashed #cbd5e1; padding-top:10px; font-size:12px;'>
            <strong>Special Educator / Therapist Clinical Guidance:</strong> {rec_text}
        </div>
    </div>
    """

    return (
        verdict_badge_html,
        scorecard_gauges_html,
        bhk_overlay_rgb,
        preproc_vis,
        ocr_overlay_rgb,
        clean_rgb,
        df_features,
        delta_html,
        report_html,
    )


# ---------------------------------------------------------------------------
# Gradio UI Construction (CopyThat Studio)
# ---------------------------------------------------------------------------

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

.gradio-container {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f8fafc !important;
}

.copythat-header {
    background: linear-gradient(135deg, #090d16 0%, #0f172a 60%, #1e293b 100%);
    border-radius: 14px;
    padding: 22px 28px;
    color: #ffffff;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.15);
    margin-bottom: 20px;
    border: 1px solid #1e293b;
}

.copythat-header h1 {
    font-size: 26px;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
    color: #38bdf8;
}

.copythat-header p {
    font-size: 13.5px;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
}

.showcase-bar {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 16px;
}

.card-container {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    padding: 16px;
}
"""

with gr.Blocks(title="CopyThat — Clinical Dysgraphia Diagnostic Studio") as demo:
    # 1. Header Box
    with gr.Column(elem_classes=["copythat-header"]):
        gr.Markdown(
            "<h1>🖋️ CopyThat — Clinical Copy-Task & Dysgraphia Diagnostic Studio</h1>"
            "<p>Offline Healthcare AI: Evaluates Fine-Motor Tremor, Spatial Crowding & Cognitive Letter Reversals directly from Paper Handwriting (Ruled School Notebooks & Plain Sheets alike). No stylus required.</p>"
        )

    # 2. Protocol Control Bar (Target Reference Prompt)
    with gr.Column(elem_classes=["showcase-bar"]):
        with gr.Row():
            with gr.Column(scale=4):
                prompt_dropdown = gr.Dropdown(
                    choices=list(PRESET_PROMPTS.keys()),
                    value="Grade 4: Board Copy (English)",
                    label="📋 Reference Copy-Task Protocol",
                    info="Select target grade syllabus prompt, classic pangram, or open unguided writing.",
                )
            with gr.Column(scale=8):
                target_prompt_input = gr.Textbox(
                    value=PRESET_PROMPTS["Grade 4: Board Copy (English)"],
                    label="🎯 Target Reference Prompt (Expected Text to Copy)",
                    placeholder="Enter the prompt sentence given to the student...",
                    lines=2,
                    interactive=True,
                )

    # 3. Two-Column Clinical Workspace
    with gr.Row():
        # LEFT COLUMN: Document Input & Visual Layers
        with gr.Column(scale=5):
            gr.Markdown("### 📤 Student Document Upload")

            paper_input = gr.Image(
                type="numpy",
                label="Handwritten Document Photo or Scan (Phone camera, school notebook, crop)",
                sources=["upload", "clipboard"],
            )

            with gr.Row():
                rotation_choice = gr.Radio(
                    choices=[
                        "No Rotation (0°)",
                        "Auto-Detect Lines",
                        "Rotate 90° Clockwise",
                        "Rotate 180°",
                        "Rotate 270°",
                    ],
                    value="No Rotation (0°)",
                    label="📐 Document Orientation",
                    info="Select rotation angle if camera took image sideways.",
                )

            with gr.Row():
                student_name_input = gr.Textbox(
                    value="Student-Aroshi",
                    label="Student Name / ID",
                    placeholder="e.g. Roll 04",
                    scale=2,
                )
                student_grade_input = gr.Dropdown(
                    choices=["Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7"],
                    value="Grade 4",
                    label="Grade Level",
                    scale=1,
                )

            analyze_btn = gr.Button("🔍 Run CopyThat Diagnostic Screening", variant="primary", size="lg")

            # Quick Load Showcase Presets
            gr.Markdown("#### 📁 Quick-Load Real-World Student Showcases")
            with gr.Row():
                btn_case_ctrl = gr.Button("🟢 Grade 4 Control (Typical)", size="sm")
                btn_case_dys = gr.Button("🔴 Grade 4 Dysgraphia (Mixed)", size="sm")
            with gr.Row():
                btn_case_pangram = gr.Button("📝 Ruled Notebook Pangram", size="sm")
                btn_case_notes = gr.Button("📋 Free Clinical Notes", size="sm")

            # Multi-Layer Visual Canvas
            gr.Markdown("### 🔬 Multi-Layer Visual Inspection Canvas")
            with gr.Tabs():
                with gr.TabItem("🔬 BHK Explainability Overlay"):
                    bhk_overlay_output = gr.Image(label="Green=Letters, Cyan=Centroids, Coral=Fitted Baselines", type="numpy")
                with gr.TabItem("✨ Cleaned Ink Mask"):
                    cleaned_mask_output = gr.Image(label="Pure Ink Mask (Notebook rulings stripped cleanly)", type="numpy")
                with gr.TabItem("🎯 Physical Word Boxes"):
                    ocr_boxes_output = gr.Image(label="Word Bounding Boxes Monotonically Mapped", type="numpy")
                with gr.TabItem("📸 Raw Cleaned Color"):
                    raw_cleaned_output = gr.Image(label="Deskewed Contrast-Normalized Document", type="numpy")

        # RIGHT COLUMN: Clinical Diagnostics & Decision Scorecard
        with gr.Column(scale=7):
            gr.Markdown("### 📊 Diagnostic Screening Verdict & Scorecard")

            verdict_badge = gr.HTML()
            scorecard_gauges = gr.HTML()

            with gr.Tabs():
                with gr.TabItem("🎯 CopyThat: Target vs Observed Alignment"):
                    delta_output = gr.HTML()
                with gr.TabItem("📋 BHK Objective Motor Biomarkers"):
                    bhk_table = gr.Dataframe(
                        headers=["BHK Biomarker", "Observed Value", "Normative Target", "Status"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                        label="Geometric & Neuromotor Quantification (BHK Scale)",
                    )
                with gr.TabItem("📥 Printable Clinical Report"):
                    with gr.Column():
                        gr.Markdown("> **Official Screening Documentation:** Printable clinical assessment card ready for school records, special educators, and therapists.")
                        report_export_html = gr.HTML()

    # 4. Interactive Wiring & Callbacks

    # Switch prompt dropdown -> update target prompt textbox
    def on_prompt_change(preset_key):
        return PRESET_PROMPTS.get(preset_key, "")

    prompt_dropdown.change(
        fn=on_prompt_change,
        inputs=[prompt_dropdown],
        outputs=[target_prompt_input],
    )

    # Showcase Case 1: Typical Control
    def load_case_ctrl():
        info = SHOWCASE_CASES["typical_g4"]
        img = cv2.imread(info["path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        return img_rgb, info["preset"], PRESET_PROMPTS[info["preset"]], "Aroshi (Roll 04)", "Grade 4"

    btn_case_ctrl.click(
        fn=load_case_ctrl,
        outputs=[paper_input, prompt_dropdown, target_prompt_input, student_name_input, student_grade_input],
    )

    # Showcase Case 2: Dysgraphic Mixed Case
    def load_case_dys():
        info = SHOWCASE_CASES["spatial_g4"]
        img = cv2.imread(info["path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        return img_rgb, info["preset"], PRESET_PROMPTS[info["preset"]], "Roll 09 (At-Risk)", "Grade 4"

    btn_case_dys.click(
        fn=load_case_dys,
        outputs=[paper_input, prompt_dropdown, target_prompt_input, student_name_input, student_grade_input],
    )

    # Showcase Case 3: Ruled Notebook Pangram
    def load_case_pangram():
        info = SHOWCASE_CASES["pangram_ruled"]
        img = cv2.imread(info["path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        return img_rgb, info["preset"], PRESET_PROMPTS[info["preset"]], "Student-Pangram", "Grade 7"

    btn_case_pangram.click(
        fn=load_case_pangram,
        outputs=[paper_input, prompt_dropdown, target_prompt_input, student_name_input, student_grade_input],
    )

    # Showcase Case 4: Clinical Free Notes
    def load_case_notes():
        info = SHOWCASE_CASES["clinical_notes"]
        img = cv2.imread(info["path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        return img_rgb, info["preset"], PRESET_PROMPTS[info["preset"]], "Clinical-Observation", "Grade 6"

    btn_case_notes.click(
        fn=load_case_notes,
        outputs=[paper_input, prompt_dropdown, target_prompt_input, student_name_input, student_grade_input],
    )

    # Main Analysis Trigger
    analyze_outputs = [
        verdict_badge,
        scorecard_gauges,
        bhk_overlay_output,
        cleaned_mask_output,
        ocr_boxes_output,
        raw_cleaned_output,
        bhk_table,
        delta_output,
        report_export_html,
    ]

    analyze_btn.click(
        fn=run_copythat_diagnostics,
        inputs=[paper_input, target_prompt_input, rotation_choice, student_name_input, student_grade_input],
        outputs=analyze_outputs,
    )


if __name__ == "__main__":
    print("=" * 80)
    print(" STARTING COPYTHAT — CLINICAL COPY-TASK & DYSGRAPHIA DIAGNOSTIC STUDIO")
    print(" Universal Paper HTR Engine Active (TrOCR + Physical Ink Box Valley Alignment)")
    print(" Serving locally on http://127.0.0.1:7860")
    print("=" * 80)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=custom_css)

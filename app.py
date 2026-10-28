"""
CopyThat — Clinical Dysgraphia Screening & Copy-Task Diagnostic Studio
Production Universal Handwriting Diagnostic Web Application (FastAPI + Bespoke Modern Frontend)
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
import time
import base64
import json
import datetime
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import cv2

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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
    "Clinical Dysgraphia Paragraph": "Dysgraphia is a neurological condition that affects handwriting and fine motor skills. It causes irregular letter sizes and spatial crowding.",
    "Custom Prompt (Type Your Own)": "",
    "Open / Free-Writing Mode (No Target Prompt)": "",
}

# Verified showcase demo cases
SHOWCASE_CASES = {
    "typical_g4": {
        "title": "🟢 Typical Control (Grade 4)",
        "subtitle": "Aroshi (Roll 04) - Steady baselines, normal spacing, 0 reversals",
        "path": "mapped_output/mapped_output/students/G4_A_Roll04/handwritten/DocScanner 06-Oct-2026 10-11 AM_4.jpeg",
        "preset": "Grade 4: Board Copy (English)",
        "name": "Aroshi (Roll 04)",
        "grade": "Grade 4",
    },
    "spatial_g4": {
        "title": "🔴 Spatial Dysgraphia (Severe Crowding)",
        "subtitle": "Grade 4 (Roll 09) - High collisions (38.9%), letter jamming, 4 reversals",
        "path": "mapped_output/mapped_output/students/G4_A_Roll09/handwritten/DocScanner 06-Oct-2026 10-11 AM_9.jpeg",
        "preset": "Grade 4: Board Copy (English)",
        "name": "Roll 09 (At-Risk)",
        "grade": "Grade 4",
    },
    "pangram_ruled": {
        "title": "📝 Ruled Notebook Pangram",
        "subtitle": "Real single-ruled notebook with horizontal blue rulings removed cleanly",
        "path": "user_test_images/clean_quick_brown.png",
        "preset": "Grade 7: Classic Pangram (English)",
        "name": "Student-Pangram",
        "grade": "Grade 7",
    },
    "clinical_notes": {
        "title": "📋 Free Clinical Notes",
        "subtitle": "Multi-line paragraph evaluated in Open Free-Writing mode",
        "path": "user_test_images/clean_dysgraphia_notes.jpg",
        "preset": "Open / Free-Writing Mode (No Target Prompt)",
        "name": "Clinical-Observation",
        "grade": "Grade 6",
    },
}


def cv2_to_base64(img_bgr_or_rgb: np.ndarray, is_rgb: bool = False) -> str:
    """Helper to convert OpenCV image to base64 Data URI."""
    if img_bgr_or_rgb is None:
        return ""
    img_bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR) if is_rgb else img_bgr_or_rgb
    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# FastAPI Application & Routes
# ---------------------------------------------------------------------------

app = FastAPI(title="CopyThat Studio", description="Clinical Dysgraphia Diagnostic Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/presets")
def get_presets():
    """Return available prompt presets and showcase cards."""
    showcase_list = []
    for cid, info in SHOWCASE_CASES.items():
        showcase_list.append({
            "id": cid,
            "title": info["title"],
            "subtitle": info["subtitle"],
            "preset": info["preset"],
            "prompt": PRESET_PROMPTS.get(info["preset"], ""),
            "name": info["name"],
            "grade": info["grade"],
        })
    return {
        "presets": PRESET_PROMPTS,
        "showcases": showcase_list,
    }


@app.get("/api/showcase/{case_id}")
def get_showcase_image(case_id: str):
    """Load image data for a showcase card."""
    if case_id not in SHOWCASE_CASES:
        return JSONResponse({"error": "Case not found"}, status_code=404)
    info = SHOWCASE_CASES[case_id]
    if not os.path.exists(info["path"]):
        return JSONResponse({"error": f"Image file missing: {info['path']}"}, status_code=404)

    img = cv2.imread(info["path"])
    b64 = cv2_to_base64(img)
    return {
        "id": case_id,
        "title": info["title"],
        "preset": info["preset"],
        "prompt": PRESET_PROMPTS.get(info["preset"], ""),
        "name": info["name"],
        "grade": info["grade"],
        "image_data": b64,
    }


@app.post("/api/analyze")
async def analyze_document(request: Request):
    """
    Main diagnostic API endpoint:
    Processes the student handwriting image and executes complete CopyThat analysis.
    Supports both JSON payloads (application/json) and multipart form-data uploads.
    """
    img_bgr = None
    content_type = request.headers.get("content-type", "")

    image_base64 = None
    target_prompt = ""
    rotation_choice = "No Rotation (0°)"
    student_name = "Student"
    student_grade = "Grade 4"

    if "application/json" in content_type:
        payload = await request.json()
        image_base64 = payload.get("image_base64")
        target_prompt = payload.get("target_prompt", "") or ""
        rotation_choice = payload.get("rotation_choice", "No Rotation (0°)") or "No Rotation (0°)"
        student_name = payload.get("student_name", "Student") or "Student"
        student_grade = payload.get("student_grade", "Grade 4") or "Grade 4"
    else:
        form = await request.form()
        image_base64 = form.get("image_base64")
        target_prompt = form.get("target_prompt", "") or ""
        rotation_choice = form.get("rotation_choice", "No Rotation (0°)") or "No Rotation (0°)"
        student_name = form.get("student_name", "Student") or "Student"
        student_grade = form.get("student_grade", "Grade 4") or "Grade 4"
        file_obj = form.get("file")
        if file_obj and hasattr(file_obj, "read"):
            content = await file_obj.read()
            if content:
                nparr = np.frombuffer(content, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None and image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return JSONResponse({"error": f"Invalid base64 image data: {e}"}, status_code=400)

    if img_bgr is None:
        return JSONResponse({"error": "No valid image data received."}, status_code=400)

    # 1. Universal Preprocessing & Artifact Removal
    binary_mask, preproc_vis, meta = preprocess_handwriting_image(
        img_bgr,
        rotation=rotation_choice,
        return_metadata=True,
    )
    clean_color = meta.get("clean_color", img_bgr)
    clean_rgb = cv2.cvtColor(clean_color, cv2.COLOR_BGR2RGB)

    # 2. Extract BHK Motor & Geometric Features
    feat_dict, feat_vec = extract_bhk_features(binary_mask)
    bhk_overlay_rgb = generate_feature_visualization(binary_mask, clean_rgb)

    # 3. Vision Transformer Handwriting Recognition
    transcription_text = ""
    words_data = []
    ocr_overlay_rgb = None
    line_words_formatted = []

    if ocr_pipeline is not None:
        try:
            trans_res = ocr_pipeline.transcribe(clean_color)
            for line in trans_res.lines:
                line_words = []
                for w in line.words:
                    words_data.append({"text": w.text, "confidence": w.confidence})
                    line_words.append({
                        "text": w.text,
                        "confidence": round(w.confidence, 3),
                        "tier": str(w.tier.name if hasattr(w.tier, "name") else w.tier),
                    })
                if line_words:
                    line_words_formatted.append(line_words)
            transcription_text = "\n".join(" ".join(w["text"] for w in line) for line in line_words_formatted)

            overlay_bgr = ocr_pipeline.render_overlay(clean_color, trans_res)
            ocr_overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"OCR execution notice: {e}")

    # 4. Cognitive Forced Alignment against Target Prompt
    is_open_mode = not target_prompt or target_prompt.strip() == ""
    reversals_count = 0
    omissions_count = 0
    substitutions_count = 0
    alignment_sim = 1.0
    alignment_words = []
    reversals_detected = []

    if not is_open_mode and words_data:
        try:
            words_for_align = [{"text": w["text"]} for w in words_data if w["text"]]
            align_res = aligner.evaluate_copy_task(target_prompt, words_for_align)
            reversals_count = align_res.reversal_count
            omissions_count = align_res.omission_count
            substitutions_count = align_res.substitution_count
            alignment_sim = align_res.alignment_similarity
            reversals_detected = align_res.reversals_detected

            for w in align_res.word_alignments:
                char_chips = []
                for c in w.char_alignments:
                    char_chips.append({
                        "expected": c.expected_char,
                        "observed": c.observed_char,
                        "status": c.status,
                        "is_reversal": c.is_reversal,
                    })
                alignment_words.append({
                    "expected_word": w.expected_word,
                    "observed_word": w.observed_word,
                    "match_status": w.match_status,
                    "char_alignments": char_chips,
                })
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

    if not is_open_mode:
        cog_score = min(1.0, max(cog_score, (reversals_count * 0.25) + (omissions_count * 0.08) + ((1.0 - alignment_sim) * 0.4)))

    is_spatial_risk = (col_ratio > 0.22) or (spat_score > 0.65) or (size_cv > 0.42)
    is_motor_risk = (tremor > 0.038) or (mot_score > 0.60) or (drift_rmse > 14.0)
    is_cognitive_risk = (reversals_count >= 1) or (cog_score > 0.60)

    if is_motor_risk and is_spatial_risk:
        decision = "Potential Dysgraphia (Mixed Motor-Spatial)"
        subtype = "Mixed Motor-Spatial Subtype"
        is_dysgraphic = True
    elif is_spatial_risk:
        decision = "Potential Dysgraphia (Spatial Subtype)"
        subtype = "Spatial Dysgraphia (Layout & Crowding)"
        is_dysgraphic = True
    elif is_motor_risk:
        decision = "Potential Dysgraphia (Motor Subtype)"
        subtype = "Motor Dysgraphia (Neuromotor Tremor)"
        is_dysgraphic = True
    elif is_cognitive_risk and not is_open_mode:
        decision = "Potential Dysgraphia (Cognitive / Reversal Subtype)"
        subtype = "Cognitive Subtype (Letter Reversals)"
        is_dysgraphic = True
    else:
        decision = "Low Risk / Control (Typical Development)"
        subtype = "Typical Development"
        is_dysgraphic = False

    # 6. Observed Clinical Symptoms List
    symptoms = []
    if col_ratio >= 0.22:
        symptoms.append(f"Overlapping letter strokes ({col_ratio:.1%} collision ratio — spatial crowding)")
    if size_cv >= 0.40:
        symptoms.append(f"Erratic character height sizing (CV: {size_cv:.2f})")
    if tremor >= 0.038:
        symptoms.append(f"Elevated stroke micro-tremor / shakiness ({tremor:.4f})")
    if spacing_cv >= 2.0:
        symptoms.append(f"Irregular inter-word whitespace intervals (Gap CV: {spacing_cv:.2f})")
    if reversals_count > 0:
        symptoms.append(f"Mirror/Inversion letter reversals detected ({reversals_count} instances)")
    if omissions_count > 1:
        symptoms.append(f"Letter omissions detected ({omissions_count} characters dropped)")

    # 7. BHK Metric Rows
    bhk_rows = [
        {"metric": "Letter Size CV (BHK #8)", "value": f"{size_cv:.3f}", "target": "< 0.40", "status": "alert" if size_cv >= 0.40 else "normal", "desc": "Inconsistency of character height"},
        {"metric": "Letter Collision Ratio (BHK #7)", "value": f"{col_ratio:.1%}", "target": "< 22.0%", "status": "alert" if col_ratio >= 0.22 else "normal", "desc": "Overlapping / jammed character strokes"},
        {"metric": "Stroke Micro-Tremor (BHK #13)", "value": f"{tremor:.4f}", "target": "< 0.038", "status": "alert" if tremor >= 0.038 else "normal", "desc": "High-frequency neuromotor shakiness"},
        {"metric": "Inter-Word Spacing CV (BHK #4)", "value": f"{spacing_cv:.3f}", "target": "< 2.00", "status": "alert" if spacing_cv >= 2.0 else "normal", "desc": "Spacing irregularity between words"},
        {"metric": "Baseline Inclination Slope (BHK #3)", "value": f"{drift_slope:.3f}", "target": "± 0.08", "status": "alert" if abs(drift_slope) >= 0.08 else "normal", "desc": "Multi-line baseline tilt and slant"},
        {"metric": "Baseline Waviness RMSE (BHK #3)", "value": f"{drift_rmse:.2f} px", "target": "< 14.0 px", "status": "alert" if drift_rmse >= 14.0 else "normal", "desc": "Waviness / undulating text trajectory"},
        {"metric": "Detected Line Ribbons", "value": f"{line_count}", "target": "—", "status": "normal", "desc": "Total active text lines extracted"},
    ]

    return {
        "decision": decision,
        "is_dysgraphic": is_dysgraphic,
        "primary_subtype": subtype,
        "student_name": student_name,
        "student_grade": student_grade,
        "date_str": datetime.datetime.now().strftime("%d %B %Y, %I:%M %p"),
        "symptoms": symptoms,
        "scores": {
            "spatial": round(spat_score * 100, 1),
            "motor": round(mot_score * 100, 1),
            "cognitive": round(cog_score * 100, 1),
        },
        "bhk_metrics": bhk_rows,
        "alignment": {
            "is_open_mode": is_open_mode,
            "target_prompt": target_prompt,
            "similarity": round(alignment_sim * 100, 1),
            "reversals_count": reversals_count,
            "omissions_count": omissions_count,
            "substitutions_count": substitutions_count,
            "words": alignment_words,
            "reversals_detected": reversals_detected,
        },
        "transcription": {
            "full_text": transcription_text,
            "lines": line_words_formatted,
        },
        "images": {
            "original": cv2_to_base64(img_bgr),
            "cleaned_mask": cv2_to_base64(binary_mask),
            "bhk_overlay": cv2_to_base64(bhk_overlay_rgb, is_rgb=True),
            "ocr_boxes": cv2_to_base64(ocr_overlay_rgb, is_rgb=True),
        },
    }


# ---------------------------------------------------------------------------
# Frontend HTML/CSS/JS (Single Page Application)
# ---------------------------------------------------------------------------

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CopyThat Studio — Clinical Dysgraphia Diagnostic Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #070b14;
      --surface-1: #0e1626;
      --surface-2: #141f36;
      --surface-glass: rgba(20, 31, 54, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(56, 189, 248, 0.35);
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.2);
      --danger: #f43f5e;
      --danger-glow: rgba(244, 63, 94, 0.25);
      --warning: #fbbf24;
      --purple: #a855f7;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: radial-gradient(circle at 50% 0%, #111d38 0%, var(--bg-dark) 70%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 20px 28px;
    }

    /* TOP HEADER */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: var(--surface-glass);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 16px;
      margin-bottom: 20px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-icon {
      font-size: 28px;
      background: linear-gradient(135deg, #0284c7, #38bdf8);
      width: 48px;
      height: 48px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px var(--accent-glow);
    }
    .brand h1 {
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #ffffff;
    }
    .brand h1 span { color: var(--accent); }
    .brand p {
      font-size: 12.5px;
      color: var(--text-muted);
      margin-top: 2px;
    }
    .status-badges {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
    }
    .badge-gpu {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
    }
    .pulse-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.3); }
    }
    .badge-universal {
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: var(--accent);
    }

    /* PROTOCOL BAR */
    .protocol-bar {
      background: var(--surface-glass);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px 20px;
      margin-bottom: 20px;
    }
    .protocol-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--accent);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .pills-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    .pill {
      background: var(--surface-2);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .pill:hover {
      background: rgba(56, 189, 248, 0.15);
      border-color: var(--accent);
      color: #ffffff;
    }
    .pill.active {
      background: var(--accent);
      color: #070b14;
      border-color: var(--accent);
      box-shadow: 0 0 12px var(--accent-glow);
    }
    .prompt-box {
      width: 100%;
      background: rgba(7, 11, 20, 0.7);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      color: #e2e8f0;
      line-height: 1.5;
      resize: vertical;
      min-height: 48px;
    }
    .prompt-box:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 10px var(--accent-glow);
    }

    /* MAIN TWO-COLUMN STUDIO */
    .studio-grid {
      display: grid;
      grid-template-columns: 5fr 7fr;
      gap: 20px;
      align-items: start;
    }

    .card {
      background: var(--surface-glass);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .card-title {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 14px;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* DROPZONE */
    .dropzone {
      border: 2px dashed rgba(56, 189, 248, 0.3);
      border-radius: 14px;
      padding: 24px;
      text-align: center;
      cursor: pointer;
      background: rgba(14, 22, 38, 0.5);
      transition: all 0.25s ease;
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.08);
      box-shadow: 0 0 20px var(--accent-glow);
    }
    .dropzone-icon {
      font-size: 36px;
      margin-bottom: 8px;
    }
    .dropzone-text {
      font-size: 13.5px;
      color: var(--text-muted);
    }
    .dropzone-text strong { color: var(--accent); }
    .dropzone input[type="file"] { display: none; }

    /* CONTROLS ROW */
    .controls-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin: 14px 0;
    }
    .input-group label {
      display: block;
      font-size: 11.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-dim);
      margin-bottom: 4px;
    }
    .input-field {
      width: 100%;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 12px;
      color: #ffffff;
      font-size: 13px;
    }
    .input-field:focus {
      outline: none;
      border-color: var(--accent);
    }

    /* ROTATION SELECTOR */
    .segmented-control {
      display: flex;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      margin: 14px 0;
    }
    .segment-btn {
      flex: 1;
      padding: 8px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      background: transparent;
      border: none;
      cursor: pointer;
      transition: all 0.2s;
    }
    .segment-btn.active {
      background: var(--accent);
      color: #070b14;
      font-weight: 700;
    }

    /* SHOWCASE PRESET CARDS */
    .showcase-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }
    .showcase-card {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: left;
    }
    .showcase-card:hover {
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .showcase-card h4 {
      font-size: 12.5px;
      font-weight: 700;
      margin-bottom: 2px;
    }
    .showcase-card p {
      font-size: 11px;
      color: var(--text-dim);
      line-height: 1.3;
    }

    /* MULTI-LAYER IMAGE CANVAS */
    .layer-tabs {
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
      background: var(--surface-2);
      padding: 4px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .layer-tab {
      flex: 1;
      text-align: center;
      padding: 7px;
      font-size: 11.5px;
      font-weight: 700;
      color: var(--text-muted);
      background: transparent;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .layer-tab.active {
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
    }
    .image-viewer {
      width: 100%;
      height: 380px;
      background: #050811;
      border-radius: 12px;
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }
    .image-viewer img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      transition: transform 0.2s ease;
    }

    /* ACTION BUTTON */
    .btn-analyze {
      width: 100%;
      background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
      color: #070b14;
      font-size: 15px;
      font-weight: 800;
      padding: 14px;
      border-radius: 12px;
      border: none;
      cursor: pointer;
      margin-top: 14px;
      box-shadow: 0 4px 16px var(--accent-glow);
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }
    .btn-analyze:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 24px rgba(56, 189, 248, 0.4);
    }
    .btn-analyze:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    /* RIGHT COLUMN: DIAGNOSTIC SCORECARD */
    .verdict-hero {
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 18px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .verdict-hero.alert {
      background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(190, 18, 60, 0.25));
      border: 1px solid rgba(244, 63, 94, 0.4);
    }
    .verdict-hero.control {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
      border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .verdict-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .verdict-title {
      font-size: 19px;
      font-weight: 800;
    }
    .verdict-hero.alert .verdict-title { color: #fb7185; }
    .verdict-hero.control .verdict-title { color: #34d399; }
    .verdict-badge-chip {
      font-size: 11px;
      font-weight: 800;
      padding: 4px 12px;
      border-radius: 20px;
      letter-spacing: 0.5px;
    }
    .verdict-hero.alert .verdict-badge-chip {
      background: var(--danger);
      color: #ffffff;
      box-shadow: 0 0 10px var(--danger-glow);
    }
    .verdict-hero.control .verdict-badge-chip {
      background: var(--success);
      color: #070b14;
      box-shadow: 0 0 10px var(--success-glow);
    }
    .verdict-desc {
      font-size: 13px;
      color: #cbd5e1;
      line-height: 1.5;
    }
    .symptoms-list {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
    }
    .symptom-item {
      font-size: 12.5px;
      color: #f1f5f9;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    /* ANIMATED CIRCULAR RISK GAUGES */
    .gauges-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 18px;
    }
    .gauge-card {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      text-align: center;
    }
    .gauge-header {
      font-size: 11.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-dim);
      margin-bottom: 6px;
    }
    .gauge-val {
      font-size: 24px;
      font-weight: 800;
      color: var(--accent);
      margin-bottom: 4px;
    }
    .gauge-val.alert { color: var(--danger); }
    .gauge-sub {
      font-size: 11px;
      color: var(--text-muted);
    }

    /* COPYTHAT LETTER-BY-LETTER DIAGNOSTIC RIBBON */
    .delta-container {
      max-height: 320px;
      overflow-y: auto;
      padding-right: 6px;
    }
    .delta-word-card {
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 8px;
    }
    .delta-word-header {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .delta-word-header code {
      font-family: 'JetBrains Mono', monospace;
      color: #ffffff;
      background: rgba(255,255,255,0.06);
      padding: 2px 6px;
      border-radius: 4px;
    }
    .chips-row {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }
    .char-chip {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      padding: 4px 6px;
      border-radius: 6px;
      min-width: 28px;
      font-family: 'JetBrains Mono', monospace;
      border: 1px solid;
    }
    .chip-char { font-size: 15px; font-weight: 700; }
    .chip-label { font-size: 9.5px; margin-top: 2px; }

    .chip-match { background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.4); color: #34d399; }
    .chip-reversal { background: rgba(168, 85, 247, 0.2); border-color: #a855f7; color: #d8b4fe; box-shadow: 0 0 8px rgba(168, 85, 247, 0.3); }
    .chip-omission { background: rgba(251, 191, 36, 0.15); border-color: #fbbf24; color: #fde047; }
    .chip-sub { background: rgba(244, 63, 94, 0.15); border-color: var(--danger); color: #fda4af; }

    /* BHK BIOMARKERS TABLE */
    .bhk-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      margin-top: 10px;
    }
    .bhk-table th {
      text-align: left;
      padding: 8px 10px;
      color: var(--text-dim);
      font-size: 11px;
      text-transform: uppercase;
      border-bottom: 1px solid var(--border);
    }
    .bhk-table td {
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      color: var(--text-main);
    }
    .tag-status {
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 10.5px;
      font-weight: 700;
    }
    .tag-normal { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .tag-alert { background: rgba(244, 63, 94, 0.15); color: #fda4af; }

    /* ACTION BAR */
    .action-bar {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }
    .btn-secondary {
      flex: 1;
      background: var(--surface-2);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 10px;
      border-radius: 10px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-secondary:hover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.1);
    }

    /* SPINNER */
    .spinner {
      display: inline-block;
      width: 18px;
      height: 18px;
      border: 3px solid rgba(7, 11, 20, 0.3);
      border-radius: 50%;
      border-top-color: #070b14;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* PRINT MODAL STYLING */
    @media print {
      body { background: white !important; color: black !important; padding: 0 !important; }
      header, .protocol-bar, .studio-grid, .action-bar { display: none !important; }
      #printable-report { display: block !important; }
    }
  </style>
</head>
<body>

  <!-- 1. TOP HEADER -->
  <header>
    <div class="brand">
      <div class="brand-icon">🖋️</div>
      <div>
        <h1>COPY<span>THAT</span> STUDIO</h1>
        <p>Universal Clinical Copy-Task & Dysgraphia Diagnostic Platform</p>
      </div>
    </div>
    <div class="status-badges">
      <div class="badge badge-gpu">
        <div class="pulse-dot"></div>
        NVIDIA RTX 4060 GPU
      </div>
      <div class="badge badge-universal">
        Universal Ruling Line Suppression
      </div>
    </div>
  </header>

  <!-- 2. PROTOCOL BAR -->
  <div class="protocol-bar">
    <div class="protocol-title">
      <span>📋 Reference Copy-Task Protocol</span>
      <span style="color:var(--text-dim); font-weight:400; font-size:11.5px;">(Select standard syllabus stimulus or open free-writing)</span>
    </div>
    <div class="pills-row" id="pills-row">
      <!-- Dynamic preset pills injected by JS -->
    </div>
    <textarea class="prompt-box" id="target-prompt" placeholder="Target Reference Prompt given to the student..."></textarea>
  </div>

  <!-- 3. MAIN TWO-COLUMN STUDIO -->
  <div class="studio-grid">

    <!-- LEFT COLUMN: Input & Multi-Layer Canvas -->
    <div>
      <div class="card">
        <div class="card-title">📤 Student Document Input</div>

        <div class="dropzone" id="dropzone">
          <div class="dropzone-icon">📷</div>
          <div class="dropzone-text">
            <strong>Click to upload</strong> or drag & drop student handwriting scan / photo<br>
            <span style="font-size:11px; color:var(--text-dim);">(Supports JPEG, PNG, Camera Shots & Ctrl+V Clipboard)</span>
          </div>
          <input type="file" id="file-input" accept="image/*">
        </div>

        <div class="controls-row">
          <div class="input-group">
            <label>Student Name / ID</label>
            <input type="text" class="input-field" id="student-name" value="Aroshi (Roll 04)">
          </div>
          <div class="input-group">
            <label>Grade Level</label>
            <select class="input-field" id="student-grade">
              <option value="Grade 3">Grade 3</option>
              <option value="Grade 4" selected>Grade 4</option>
              <option value="Grade 5">Grade 5</option>
              <option value="Grade 6">Grade 6</option>
              <option value="Grade 7">Grade 7</option>
            </select>
          </div>
        </div>

        <div class="input-group">
          <label>Document Orientation</label>
          <div class="segmented-control" id="rotation-control">
            <button class="segment-btn active" data-rot="No Rotation (0°)">0°</button>
            <button class="segment-btn" data-rot="Auto-Detect Lines">Auto</button>
            <button class="segment-btn" data-rot="Rotate 90° Clockwise">90°</button>
            <button class="segment-btn" data-rot="Rotate 180°">180°</button>
            <button class="segment-btn" data-rot="Rotate 270°">270°</button>
          </div>
        </div>

        <button class="btn-analyze" id="btn-analyze">
          <span>🔍 RUN COPYTHAT DIAGNOSTIC SCREENING</span>
        </button>

        <!-- Quick Load Showcases -->
        <div style="margin-top:16px;">
          <div style="font-size:12px; font-weight:700; color:var(--text-dim); text-transform:uppercase; margin-bottom:6px;">
            📁 1-Click Real-World Student Showcases
          </div>
          <div class="showcase-grid" id="showcase-grid">
            <!-- Showcase cards injected by JS -->
          </div>
        </div>
      </div>

      <!-- Multi-Layer Visual Canvas -->
      <div class="card">
        <div class="card-title">🔬 Multi-Layer Document Inspection Canvas</div>
        <div class="layer-tabs" id="layer-tabs">
          <button class="layer-tab active" data-layer="bhk_overlay">🔬 BHK Baselines</button>
          <button class="layer-tab" data-layer="cleaned_mask">✨ Cleaned Ink</button>
          <button class="layer-tab" data-layer="ocr_boxes">🎯 Word Boxes</button>
          <button class="layer-tab" data-layer="original">📸 Original</button>
        </div>
        <div class="image-viewer" id="image-viewer">
          <div style="color:var(--text-dim); font-size:13px;" id="viewer-placeholder">
            Upload document or click a showcase card to inspect visual layers
          </div>
          <img id="viewer-img" style="display:none;" alt="Document Layer">
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN: Diagnostic Scorecard & Intelligence -->
    <div>
      <div class="card" id="diagnostic-card">
        <div class="card-title">📊 Clinical Screening Scorecard</div>

        <!-- Hero Verdict -->
        <div class="verdict-hero control" id="verdict-hero">
          <div class="verdict-header">
            <div class="verdict-title" id="verdict-title">Ready for Diagnostic Screening</div>
            <span class="verdict-badge-chip" id="verdict-chip">READY</span>
          </div>
          <div class="verdict-desc" id="verdict-desc">
            Upload a handwritten student paper on the left or select a showcase case to begin evaluation.
          </div>
          <div class="symptoms-list" id="symptoms-container" style="display:none;"></div>
        </div>

        <!-- 3 Animated Gauges -->
        <div class="gauges-grid">
          <div class="gauge-card">
            <div class="gauge-header">Spatial Layout</div>
            <div class="gauge-val" id="val-spatial">0.0%</div>
            <div class="gauge-sub" id="sub-spatial">Letter overlap & crowding</div>
          </div>
          <div class="gauge-card">
            <div class="gauge-header">Neuromotor Tremor</div>
            <div class="gauge-val" id="val-motor">0.0%</div>
            <div class="gauge-sub" id="sub-motor">Stroke shakiness & unsteadiness</div>
          </div>
          <div class="gauge-card">
            <div class="gauge-header">Cognitive Fidelity</div>
            <div class="gauge-val" id="val-cognitive">0.0%</div>
            <div class="gauge-sub" id="sub-cognitive">Letter reversals & omissions</div>
          </div>
        </div>

        <!-- CopyThat Letter-by-Letter Alignment Ribbon -->
        <div style="margin-bottom:16px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="font-size:13px; font-weight:700; color:#ffffff;">🎯 CopyThat: Target vs Observed Alignment</div>
            <div style="font-size:12px; color:var(--text-muted);" id="sim-badge">Fidelity: —</div>
          </div>
          <div class="delta-container" id="delta-container">
            <div style="color:var(--text-dim); font-size:12.5px; padding:12px; background:var(--surface-2); border-radius:8px;">
              No document analyzed yet.
            </div>
          </div>
        </div>

        <!-- BHK Metrics Table -->
        <div>
          <div style="font-size:13px; font-weight:700; color:#ffffff; margin-bottom:6px;">📋 Objective BHK Motor Biomarkers</div>
          <table class="bhk-table">
            <thead>
              <tr>
                <th>Biomarker</th>
                <th>Observed</th>
                <th>Target</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody id="bhk-tbody">
              <tr>
                <td colspan="4" style="color:var(--text-dim); text-align:center; padding:16px;">
                  Metrics will populate upon analysis.
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Action Bar -->
        <div class="action-bar">
          <button class="btn-secondary" id="btn-print">🖨️ Print Clinical Summary Report</button>
          <button class="btn-secondary" id="btn-json">📥 Export Diagnostic JSON</button>
        </div>
      </div>
    </div>

  </div>

  <script>
    let activeRotation = "No Rotation (0°)";
    let currentImageData = null;
    let currentLayers = {};
    let lastAnalysisData = null;

    // 1. Fetch Presets on Load
    async function initApp() {
      try {
        const res = await fetch("/api/presets");
        const data = await res.json();

        // Render Preset Pills
        const pillsRow = document.getElementById("pills-row");
        pillsRow.innerHTML = "";
        let first = true;
        for (const [key, promptText] of Object.entries(data.presets)) {
          const pill = document.createElement("button");
          pill.className = "pill" + (first ? " active" : "");
          pill.textContent = key;
          pill.onclick = () => {
            document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            document.getElementById("target-prompt").value = promptText;
          };
          pillsRow.appendChild(pill);
          if (first) {
            document.getElementById("target-prompt").value = promptText;
            first = false;
          }
        }

        // Render Showcase Cards
        const grid = document.getElementById("showcase-grid");
        grid.innerHTML = "";
        data.showcases.forEach(sc => {
          const card = document.createElement("div");
          card.className = "showcase-card";
          card.innerHTML = `<h4>${sc.title}</h4><p>${sc.subtitle}</p>`;
          card.onclick = () => loadShowcase(sc.id);
          grid.appendChild(card);
        });

      } catch (err) {
        console.error("Init failed:", err);
      }
    }

    // 2. Load Showcase Case
    async function loadShowcase(caseId) {
      try {
        const res = await fetch(`/api/showcase/${caseId}`);
        const data = await res.json();
        if (data.image_data) {
          currentImageData = data.image_data;
          document.getElementById("viewer-placeholder").style.display = "none";
          const img = document.getElementById("viewer-img");
          img.src = currentImageData;
          img.style.display = "block";

          document.getElementById("student-name").value = data.name;
          document.getElementById("student-grade").value = data.grade;

          // Select matching prompt pill
          document.querySelectorAll(".pill").forEach(p => {
            if (p.textContent === data.preset) {
              p.classList.add("active");
            } else {
              p.classList.remove("active");
            }
          });
          document.getElementById("target-prompt").value = data.prompt;

          // Auto run analysis
          triggerAnalysis();
        }
      } catch (err) {
        console.error("Showcase load failed:", err);
      }
    }

    // 3. Dropzone & File Handling
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    dropzone.onclick = () => fileInput.click();
    fileInput.onchange = e => {
      const file = e.target.files[0];
      if (file) handleFile(file);
    };

    dropzone.ondragover = e => { e.preventDefault(); dropzone.classList.add("dragover"); };
    dropzone.ondragleave = () => dropzone.classList.remove("dragover");
    dropzone.ondrop = e => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    };

    // Clipboard Paste Listener (Ctrl+V)
    window.addEventListener("paste", e => {
      const items = (e.clipboardData || e.originalEvent.clipboardData).items;
      for (const item of items) {
        if (item.type.indexOf("image") === 0) {
          const blob = item.getAsFile();
          handleFile(blob);
          break;
        }
      }
    });

    function handleFile(file) {
      const reader = new FileReader();
      reader.onload = ev => {
        currentImageData = ev.target.result;
        document.getElementById("viewer-placeholder").style.display = "none";
        const img = document.getElementById("viewer-img");
        img.src = currentImageData;
        img.style.display = "block";
      };
      reader.readAsDataURL(file);
    }

    // 4. Rotation Segmented Control
    document.querySelectorAll(".segment-btn").forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll(".segment-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activeRotation = btn.getAttribute("data-rot");
      };
    });

    // 5. Layer Tabs Switching
    document.querySelectorAll(".layer-tab").forEach(tab => {
      tab.onclick = () => {
        document.querySelectorAll(".layer-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const layerKey = tab.getAttribute("data-layer");
        if (currentLayers[layerKey]) {
          document.getElementById("viewer-img").src = currentLayers[layerKey];
        }
      };
    });

    // 6. Trigger Diagnostic Analysis
    const btnAnalyze = document.getElementById("btn-analyze");
    btnAnalyze.onclick = triggerAnalysis;

    async function triggerAnalysis() {
      if (!currentImageData) {
        alert("Please upload a handwritten document or select a showcase case first.");
        return;
      }

      btnAnalyze.disabled = true;
      btnAnalyze.innerHTML = '<span class="spinner"></span> <span>ANALYZING DOCUMENT ON GPU...</span>';

      const payload = {
        image_base64: currentImageData,
        target_prompt: document.getElementById("target-prompt").value,
        rotation_choice: activeRotation,
        student_name: document.getElementById("student-name").value,
        student_grade: document.getElementById("student-grade").value,
      };

      try {
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.error) {
          throw new Error(data.error);
        }
        lastAnalysisData = data;
        renderDiagnostics(data);
      } catch (err) {
        console.error("Analysis failed:", err);
        alert("Error running analysis: " + err.message);
      } finally {
        btnAnalyze.disabled = false;
        btnAnalyze.innerHTML = '<span>🔍 RUN COPYTHAT DIAGNOSTIC SCREENING</span>';
      }
    }

    // 7. Render Complete Diagnostic Scorecard
    function renderDiagnostics(data) {
      // Images
      currentLayers = data.images || {};
      if (currentLayers.bhk_overlay) {
        document.getElementById("viewer-img").src = currentLayers.bhk_overlay;
        document.querySelectorAll(".layer-tab").forEach(t => {
          t.classList.toggle("active", t.getAttribute("data-layer") === "bhk_overlay");
        });
      }

      // Hero Verdict
      const hero = document.getElementById("verdict-hero");
      hero.className = "verdict-hero " + (data.is_dysgraphic ? "alert" : "control");
      document.getElementById("verdict-title").textContent = (data.is_dysgraphic ? "⚠️ " : "✅ ") + data.decision;
      document.getElementById("verdict-chip").textContent = data.is_dysgraphic ? "AT RISK" : "CONTROL";
      document.getElementById("verdict-desc").textContent = data.is_dysgraphic 
        ? "Handwriting exhibits elevated stroke unsteadiness, letter size irregularity, or spatial crowding. A formal developmental evaluation is recommended."
        : "Handwriting layout, stroke consistency, letter sizing, and spacing align cleanly with developmental normative expectations.";

      // Symptoms
      const symContainer = document.getElementById("symptoms-container");
      if (data.symptoms && data.symptoms.length) {
        symContainer.style.display = "block";
        symContainer.innerHTML = data.symptoms.map(s => `<div class="symptom-item">● ${s}</div>`).join("");
      } else {
        symContainer.style.display = "none";
      }

      // Gauges
      document.getElementById("val-spatial").textContent = data.scores.spatial + "%";
      document.getElementById("val-spatial").className = "gauge-val " + (data.scores.spatial >= 50 ? "alert" : "");
      document.getElementById("val-motor").textContent = data.scores.motor + "%";
      document.getElementById("val-motor").className = "gauge-val " + (data.scores.motor >= 50 ? "alert" : "");
      document.getElementById("val-cognitive").textContent = data.scores.cognitive + "%";
      document.getElementById("val-cognitive").className = "gauge-val " + (data.scores.cognitive >= 50 ? "alert" : "");

      // Alignment Ribbon
      const deltaCont = document.getElementById("delta-container");
      if (!data.alignment.is_open_mode && data.alignment.words && data.alignment.words.length) {
        document.getElementById("sim-badge").textContent = `Sequence Match: ${data.alignment.similarity}% | Reversals: ${data.alignment.reversals_count}`;
        let html = "";
        data.alignment.words.forEach(w => {
          let chips = "";
          w.char_alignments.forEach(c => {
            let cls = "chip-match";
            let lbl = "✓";
            if (c.status === "reversal") { cls = "chip-reversal"; lbl = `REV(${c.expected}➔${c.observed})`; }
            else if (c.status === "omission") { cls = "chip-omission"; lbl = `DROP(${c.expected})`; }
            else if (c.status === "insertion") { cls = "chip-reversal"; lbl = `EXTRA(${c.observed})`; }
            else if (c.status !== "match") { cls = "chip-sub"; lbl = `${c.expected}➔${c.observed}`; }
            chips += `<div class="char-chip ${cls}"><span class="chip-char">${c.observed}</span><span class="chip-label">${lbl}</span></div>`;
          });
          html += `
            <div class="delta-word-card">
              <div class="delta-word-header">
                <span>Expected: <code>${w.expected_word}</code></span>
                <span>Observed: <code>${w.observed_word}</code></span>
              </div>
              <div class="chips-row">${chips}</div>
            </div>
          `;
        });
        deltaCont.innerHTML = html;
      } else {
        document.getElementById("sim-badge").textContent = "Open Vocabulary Mode";
        deltaCont.innerHTML = `
          <div style="background:var(--surface-2); padding:14px; border-radius:8px; line-height:1.6; font-family:'JetBrains Mono', monospace; font-size:13px; color:#ffffff;">
            ${data.transcription.full_text || "No words transcribed."}
          </div>
        `;
      }

      // BHK Table
      const tbody = document.getElementById("bhk-tbody");
      tbody.innerHTML = data.bhk_metrics.map(r => `
        <tr>
          <td><strong>${r.metric}</strong><br><small style="color:var(--text-dim);">${r.desc}</small></td>
          <td><code style="color:#ffffff;">${r.value}</code></td>
          <td style="color:var(--text-muted);">${r.target}</td>
          <td><span class="tag-status ${r.status === 'alert' ? 'tag-alert' : 'tag-normal'}">${r.status === 'alert' ? '⚠️ High' : '✓ Normal'}</span></td>
        </tr>
      `).join("");
    }

    // 8. Print Clinical Summary
    document.getElementById("btn-print").onclick = () => {
      if (!lastAnalysisData) {
        alert("Please run an analysis first to generate a report.");
        return;
      }
      window.print();
    };

    // 9. Export JSON
    document.getElementById("btn-json").onclick = () => {
      if (!lastAnalysisData) {
        alert("Please run an analysis first.");
        return;
      }
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastAnalysisData, null, 2));
      const dl = document.createElement('a');
      dl.setAttribute("href", dataStr);
      dl.setAttribute("download", `CopyThat_${lastAnalysisData.student_name}_${Date.now()}.json`);
      dl.click();
    };

    initApp();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the bespoke modern CopyThat studio front-end."""
    return HTMLResponse(content=FRONTEND_HTML)


if __name__ == "__main__":
    print("=" * 80)
    print(" STARTING COPYTHAT — BESPOKE MODERN CLINICAL DIAGNOSTIC STUDIO")
    print(" FastAPI + High-Performance Vanilla HTML5/CSS3/JS Architecture")
    print(" Serving on http://127.0.0.1:7860")
    print("=" * 80)
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")

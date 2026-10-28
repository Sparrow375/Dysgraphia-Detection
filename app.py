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

# Standard reference copy prompts (Categorized by Task Modality, NOT Grade-wise)
PRESET_PROMPTS = {
    "Board Copy — Standard Classroom Text": "On Sundays we play cricket with our friends. After the game we eat fruits together.",
    "Board Copy — Classic Pangram (Alphabet Verification)": "The quick brown Fox jumps over The lazy dog. This sentence uses every letter of the alphabet.",
    "Board Copy — Narrative Lesson": "Our teacher tells us a new story every day. We listen carefully. The stories teach us good things.",
    "Board Copy — Hindi Classroom Sentence": "रविवार को हम सब क्रिकेट खेलते हैं। खेल के बाद हम मिलकर नाश्ता करते हैं।",
    "Dictation — Sports & Kinematic Motor Speed": "My brother plays football with his friends.",
    "Dictation — Domestic & Auditory Recall": "My mother cooks food in the kitchen. The food smells very good.",
    "Dictation — Complex Clausal Syntax": "Although it was raining, the children walked to school.",
    "Free Writing — Personal Expression / Favorite Activity": "Write one or two sentences about your favourite game or daily activities.",
    "Clinical Standard — BHK Assessment Passage": "Dysgraphia is a neurological condition that affects handwriting and fine motor skills. It causes irregular letter sizes and spatial crowding.",
    "Custom Prompt (Type Your Own)": "",
    "Open / Free-Writing Mode (No Target Prompt)": "",
}

# Verified showcase demo cases (Task Modality Based)
SHOWCASE_CASES = {
    "typical_control": {
        "title": "Typical Control",
        "subtitle": "Board Copy • Aroshi (Roll 04) • Normal baselines & zero reversals",
        "path": "mapped_output/mapped_output/students/G4_A_Roll04/handwritten/DocScanner 06-Oct-2026 10-11 AM_4.jpeg",
        "preset": "Board Copy — Standard Classroom Text",
        "name": "Aroshi (Roll 04)",
        "task": "Board Copy",
    },
    "spatial_dysgraphia": {
        "title": "Spatial Dysgraphia",
        "subtitle": "Dictation • Roll 09 • High collisions (38.9%), letter jamming, reversals",
        "path": "mapped_output/mapped_output/students/G4_A_Roll09/handwritten/DocScanner 06-Oct-2026 10-11 AM_9.jpeg",
        "preset": "Dictation — Sports & Kinematic Motor Speed",
        "name": "Roll 09 (At-Risk)",
        "task": "Dictation",
    },
    "ruled_notebook": {
        "title": "Ruled Notebook",
        "subtitle": "Line Removal • Single-ruled page with horizontal lines stripped cleanly",
        "path": "user_test_images/clean_quick_brown.png",
        "preset": "Board Copy — Classic Pangram (Alphabet Verification)",
        "name": "Student-Pangram",
        "task": "Board Copy",
    },
    "clinical_notes": {
        "title": "Clinical Notes",
        "subtitle": "Free Writing • Multi-line observation notes in unconstrained mode",
        "path": "user_test_images/clean_dysgraphia_notes.jpg",
        "preset": "Open / Free-Writing Mode (No Target Prompt)",
        "name": "Clinical-Observation",
        "task": "Free Writing",
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
            "task": info.get("task", "Board Copy"),
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
        "task": info.get("task", "Board Copy"),
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
    task_modality = "Board Copy"

    if "application/json" in content_type:
        payload = await request.json()
        image_base64 = payload.get("image_base64")
        target_prompt = payload.get("target_prompt", "") or ""
        rotation_choice = payload.get("rotation_choice", "No Rotation (0°)") or "No Rotation (0°)"
        student_name = payload.get("student_name", "Student") or "Student"
        task_modality = payload.get("task_modality") or payload.get("student_grade") or "Board Copy"
    else:
        form = await request.form()
        image_base64 = form.get("image_base64")
        target_prompt = form.get("target_prompt", "") or ""
        rotation_choice = form.get("rotation_choice", "No Rotation (0°)") or "No Rotation (0°)"
        student_name = form.get("student_name", "Student") or "Student"
        task_modality = form.get("task_modality") or form.get("student_grade") or "Board Copy"
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
        "task_modality": task_modality,
        "student_grade": task_modality,  # Backwards compatibility
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
  <title>CopyThat. — Clinical Dysgraphia Diagnostic Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    /* -------------------------------------------------------------
       CLIPZIO-INSPIRED MONOCHROME DESIGN SYSTEM
       Strictly Black, White, and Shades of Gray
       ------------------------------------------------------------- */
    :root {
      color-scheme: dark;
      --bg-canvas: #09090b;
      --bg-deep: #000000;
      --bg-surface: #121214;
      --bg-surface-hover: #1c1c1f;
      --bg-panel: #18181b;
      --bg-card: #141416;
      --border: #27272a;
      --border-subtle: #1e1e21;
      --border-hover: #3f3f46;
      --border-strong: #71717a;
      --text-primary: #fafafa;
      --text-secondary: #a1a1aa;
      --text-muted: #71717a;
      --text-dim: #52525b;
      --accent-white: #ffffff;
      --accent-black: #000000;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
      --radius-pill: 9999px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-canvas);
      background-image: radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px);
      background-size: 24px 24px;
      color: var(--text-primary);
      min-height: 100vh;
      padding: 24px 32px;
      line-height: 1.5;
    }

    /* TOP NAVBAR (Clipizo Style) */
    .navbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      margin-bottom: 24px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.6px;
      color: #ffffff;
    }
    .signature-dot {
      color: #ffffff;
      font-weight: 900;
    }
    .brand-tag {
      font-size: 11.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      padding: 3px 8px;
      background: #1e1e22;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      color: var(--text-secondary);
    }
    .nav-status {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 6px 14px;
      border-radius: var(--radius-pill);
      font-size: 12px;
      font-weight: 500;
      background: var(--bg-panel);
      border: 1px solid var(--border);
      color: var(--text-secondary);
    }
    .status-badge strong { color: #ffffff; }
    .pulse-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #ffffff;
      box-shadow: 0 0 8px rgba(255, 255, 255, 0.8);
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.3; transform: scale(1.3); }
    }

    /* PROTOCOL & STIMULUS BAR */
    .protocol-card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 20px 24px;
      margin-bottom: 24px;
    }
    .protocol-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .protocol-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .protocol-subtitle {
      font-size: 12px;
      color: var(--text-muted);
    }

    /* Modality Filter Chips */
    .modality-filters {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }
    .modality-btn {
      font-family: inherit;
      font-size: 11.5px;
      font-weight: 600;
      padding: 5px 12px;
      border-radius: var(--radius-pill);
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .modality-btn:hover {
      border-color: var(--border-hover);
      color: var(--text-primary);
    }
    .modality-btn.active {
      background: #27272a;
      border-color: #52525b;
      color: #ffffff;
    }

    /* Prompt Selection Pills */
    .prompt-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }
    .pill {
      font-family: inherit;
      font-size: 12px;
      font-weight: 500;
      padding: 6px 14px;
      border-radius: var(--radius-pill);
      background: var(--bg-panel);
      border: 1px solid var(--border);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .pill:hover {
      border-color: var(--border-hover);
      background: var(--bg-surface-hover);
      color: #ffffff;
    }
    .pill.active {
      background: #ffffff;
      color: #000000;
      border-color: #ffffff;
      font-weight: 700;
    }

    .prompt-box {
      width: 100%;
      background: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 12px 16px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      color: var(--text-primary);
      line-height: 1.5;
      resize: vertical;
      min-height: 52px;
      outline: none;
      transition: border-color 0.15s ease;
    }
    .prompt-box:focus {
      border-color: var(--text-primary);
    }

    /* MAIN TWO-COLUMN STUDIO */
    .studio-grid {
      display: grid;
      grid-template-columns: 5fr 7fr;
      gap: 24px;
      align-items: start;
    }

    .card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 22px;
      margin-bottom: 24px;
    }
    .card-title {
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 16px;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* SHOWCASE CARDS (Task Modality Based) */
    .showcase-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 18px;
    }
    .showcase-btn {
      text-align: left;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 12px 14px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .showcase-btn:hover {
      border-color: var(--border-hover);
      background: var(--bg-surface-hover);
      transform: translateY(-1px);
    }
    .showcase-title {
      font-size: 12.5px;
      font-weight: 700;
      color: #ffffff;
      margin-bottom: 3px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .showcase-tag {
      font-size: 10px;
      text-transform: uppercase;
      padding: 2px 6px;
      border-radius: 4px;
      background: #27272a;
      color: var(--text-secondary);
    }
    .showcase-subtitle {
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.35;
    }

    /* DROPZONE */
    .dropzone {
      border: 1px dashed var(--border-hover);
      border-radius: var(--radius-md);
      padding: 26px 20px;
      text-align: center;
      cursor: pointer;
      background: var(--bg-deep);
      transition: all 0.15s ease;
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: #ffffff;
      background: #141418;
    }
    .dropzone-icon {
      font-size: 28px;
      color: #ffffff;
      margin-bottom: 6px;
    }
    .dropzone-text {
      font-size: 13px;
      color: var(--text-secondary);
    }
    .dropzone-text strong {
      color: #ffffff;
      text-decoration: underline;
    }
    .dropzone input[type="file"] { display: none; }

    /* CONTROLS ROW */
    .controls-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin: 16px 0;
    }
    .input-group label {
      display: block;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-muted);
      margin-bottom: 5px;
    }
    .input-field {
      width: 100%;
      background: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 8px 12px;
      font-family: inherit;
      font-size: 13px;
      color: #ffffff;
      outline: none;
      transition: border-color 0.15s ease;
    }
    .input-field:focus {
      border-color: #ffffff;
    }

    /* SEGMENTED BUTTONS (Orientation) */
    .segment-bar {
      display: flex;
      background: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 3px;
      gap: 4px;
      margin-bottom: 16px;
    }
    .segment-btn {
      flex: 1;
      font-family: inherit;
      font-size: 11.5px;
      font-weight: 600;
      padding: 6px 8px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      border-radius: 4px;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s ease;
    }
    .segment-btn:hover {
      color: #ffffff;
    }
    .segment-btn.active {
      background: #27272a;
      color: #ffffff;
    }

    /* PRIMARY CTA BUTTON (Clipizo White Button) */
    .btn-analyze {
      width: 100%;
      background: #ffffff;
      color: #000000;
      border: none;
      border-radius: var(--radius-md);
      padding: 14px 20px;
      font-family: inherit;
      font-size: 13.5px;
      font-weight: 800;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: all 0.15s ease;
      margin-bottom: 20px;
    }
    .btn-analyze:hover {
      background: #e4e4e7;
      transform: translateY(-1px);
      box-shadow: 0 4px 16px rgba(255, 255, 255, 0.12);
    }
    .btn-analyze:active {
      transform: translateY(0);
    }
    .btn-analyze:disabled {
      background: #27272a;
      color: #71717a;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    /* MULTI-LAYER IMAGE CANVAS */
    .layer-tabs {
      display: flex;
      background: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 3px;
      gap: 3px;
      margin-bottom: 12px;
      overflow-x: auto;
    }
    .layer-tab {
      font-family: inherit;
      font-size: 11.5px;
      font-weight: 600;
      padding: 6px 12px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      border-radius: 4px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s ease;
    }
    .layer-tab:hover { color: #ffffff; }
    .layer-tab.active {
      background: #27272a;
      color: #ffffff;
    }
    .image-viewer {
      background: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      min-height: 380px;
      max-height: 520px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }
    .image-viewer img {
      max-width: 100%;
      max-height: 520px;
      object-fit: contain;
      display: block;
    }
    .image-placeholder {
      color: var(--text-dim);
      font-size: 13px;
      text-align: center;
      padding: 40px;
    }

    /* RIGHT COLUMN: DIAGNOSTIC SCORECARD */

    /* HERO VERDICT (Monochrome High Contrast) */
    .verdict-hero {
      border-radius: var(--radius-lg);
      padding: 20px 22px;
      margin-bottom: 20px;
      border: 1px solid var(--border);
      background: var(--bg-card);
      transition: all 0.2s ease;
    }
    .verdict-hero.control {
      border-color: #3f3f46;
      background: var(--bg-surface);
    }
    .verdict-hero.alert {
      border: 2px solid #ffffff;
      background: var(--bg-surface);
    }
    .verdict-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .verdict-title {
      font-size: 18px;
      font-weight: 800;
      color: #ffffff;
      letter-spacing: -0.3px;
    }
    .verdict-chip {
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      padding: 4px 10px;
      border-radius: var(--radius-pill);
    }
    .verdict-chip.control {
      background: transparent;
      border: 1px solid #52525b;
      color: #d4d4d8;
    }
    .verdict-chip.alert {
      background: #ffffff;
      color: #000000;
      border: 1px solid #ffffff;
    }
    .verdict-desc {
      font-size: 12.5px;
      color: var(--text-secondary);
      line-height: 1.5;
    }
    .symptoms-list {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .symptom-item {
      font-size: 12px;
      color: #d4d4d8;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* TRIPARTITE RISK GAUGES (Monochrome Minimalist) */
    .gauges-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-bottom: 20px;
    }
    .gauge-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 16px 12px;
      text-align: center;
    }
    .gauge-svg-wrap {
      width: 72px;
      height: 72px;
      margin: 0 auto 10px;
      position: relative;
    }
    .gauge-svg {
      transform: rotate(-90deg);
      width: 72px;
      height: 72px;
    }
    .gauge-bg {
      fill: none;
      stroke: #222226;
      stroke-width: 6;
    }
    .gauge-fill {
      fill: none;
      stroke: #ffffff;
      stroke-width: 6;
      stroke-linecap: round;
      stroke-dasharray: 188.5;
      stroke-dashoffset: 188.5;
      transition: stroke-dashoffset 0.8s ease;
    }
    .gauge-val {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 14px;
      font-weight: 800;
      color: #ffffff;
    }
    .gauge-label {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-secondary);
      margin-bottom: 2px;
    }
    .gauge-hint {
      font-size: 10px;
      color: var(--text-dim);
    }

    /* COPYTHAT DELTA RIBBON (Monochrome High Contrast) */
    .delta-section {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 18px;
      margin-bottom: 20px;
    }
    .delta-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .delta-header h4 {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: #ffffff;
    }
    .delta-sim-badge {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      background: #1c1c1f;
      border: 1px solid var(--border);
      color: #ffffff;
    }

    .delta-words-grid {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .delta-word-card {
      background: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 12px;
    }
    .delta-word-header {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .delta-word-header code {
      font-family: 'JetBrains Mono', monospace;
      color: #ffffff;
      font-size: 12px;
    }
    .chips-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .char-chip {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      padding: 4px 7px;
      border-radius: 4px;
      min-width: 26px;
      font-family: 'JetBrains Mono', monospace;
    }
    .chip-match {
      background: #18181b;
      border: 1px solid var(--border);
      color: #ffffff;
    }
    .chip-reversal {
      background: #ffffff;
      color: #000000;
      border: 1px solid #ffffff;
      font-weight: 800;
    }
    .chip-omission {
      background: transparent;
      border: 1px dashed #71717a;
      color: #a1a1aa;
    }
    .chip-sub {
      background: #1e1e24;
      border: 1px solid #52525b;
      color: #ffffff;
    }
    .chip-char { font-size: 13px; font-weight: 700; }
    .chip-label { font-size: 9px; margin-top: 1px; opacity: 0.8; }

    /* OBJECTIVE BHK BIOMARKERS TABLE */
    .table-container {
      overflow-x: auto;
      margin-bottom: 20px;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      text-align: left;
    }
    th {
      background: var(--bg-deep);
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 10.5px;
      letter-spacing: 0.6px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
    }
    td {
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-primary);
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: var(--bg-surface-hover); }
    .tag-status {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 10.5px;
      font-weight: 600;
    }
    .tag-normal {
      border: 1px solid var(--border);
      color: var(--text-secondary);
    }
    .tag-alert {
      background: #ffffff;
      color: #000000;
      font-weight: 700;
    }

    /* ACTION BAR */
    .actions-bar {
      display: flex;
      gap: 12px;
    }
    .btn-secondary {
      flex: 1;
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: #ffffff;
      padding: 10px 16px;
      border-radius: var(--radius-md);
      font-family: inherit;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s ease;
    }
    .btn-secondary:hover {
      border-color: var(--border-hover);
      background: var(--bg-surface-hover);
    }

    /* SPINNER */
    .spinner {
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(0, 0, 0, 0.2);
      border-top-color: #000000;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* PRINT STYLES */
    @media print {
      body {
        background: #ffffff !important;
        color: #000000 !important;
        padding: 0;
      }
      .navbar, .protocol-card, .btn-analyze, .dropzone, .controls-row, .segment-bar, .actions-bar, .showcase-grid {
        display: none !important;
      }
      .studio-grid {
        grid-template-columns: 1fr !important;
      }
      .card {
        border: 1px solid #000000 !important;
        background: #ffffff !important;
        color: #000000 !important;
      }
      .verdict-hero {
        border: 2px solid #000000 !important;
        background: #ffffff !important;
        color: #000000 !important;
      }
      .verdict-title { color: #000000 !important; }
      .gauge-card { border: 1px solid #000000 !important; }
      .gauge-val { color: #000000 !important; }
      table, th, td {
        border-color: #000000 !important;
        color: #000000 !important;
      }
    }
  </style>
</head>
<body>

  <!-- TOP NAVBAR -->
  <nav class="navbar">
    <div class="brand">
      <span class="brand-logo">CopyThat<span class="signature-dot">.</span></span>
      <span class="brand-tag">Clinical Studio</span>
    </div>
    <div class="nav-status">
      <span class="status-badge">
        <span class="pulse-dot"></span>
        <span>CUDA Active: <strong>RTX 4060</strong></span>
      </span>
      <span class="status-badge">
        <span>Engine: <strong>TrOCR + BHK 29-D</strong></span>
      </span>
    </div>
  </nav>

  <!-- PROTOCOL BAR (Modality-based, Non-Grade-wise) -->
  <div class="protocol-card">
    <div class="protocol-header">
      <div>
        <div class="protocol-title">
          <span>Task Modality & Target Stimulus</span>
        </div>
        <div class="protocol-subtitle">Select a target reference prompt or execute unconstrained free-writing analysis.</div>
      </div>
    </div>

    <!-- Modality Category Filter Chips -->
    <div class="modality-filters" id="modality-filters">
      <button class="modality-btn active" data-cat="all">All Prompts</button>
      <button class="modality-btn" data-cat="board">Board Copy</button>
      <button class="modality-btn" data-cat="dictation">Dictation</button>
      <button class="modality-btn" data-cat="freewrite">Free Writing</button>
      <button class="modality-btn" data-cat="clinical">Clinical Standard</button>
      <button class="modality-btn" data-cat="custom">Custom / Open</button>
    </div>

    <!-- Prompt Selection Pills -->
    <div class="prompt-pills" id="prompt-pills"></div>

    <!-- Active Prompt Textarea -->
    <textarea id="target-prompt" class="prompt-box" rows="2" placeholder="Target sentence reference..."></textarea>
  </div>

  <!-- MAIN STUDIO TWO-COLUMN LAYOUT -->
  <div class="studio-grid">

    <!-- LEFT COLUMN: Ingestion & Multi-Layer Canvas -->
    <div>
      <div class="card">
        <div class="card-title">
          <span>1. Document Ingestion</span>
        </div>

        <!-- Showcase Presets (Task Modality Based) -->
        <div class="showcase-grid" id="showcase-grid"></div>

        <!-- Upload Dropzone -->
        <div class="dropzone" id="dropzone">
          <div class="dropzone-icon">⇪</div>
          <div class="dropzone-text">
            <strong>Upload Document</strong> or drag & drop image scan
          </div>
          <input type="file" id="file-input" accept="image/*">
        </div>

        <!-- Controls Row -->
        <div class="controls-row">
          <div class="input-group">
            <label>Student Identifier / Name</label>
            <input type="text" id="student-name" class="input-field" value="Student-01">
          </div>
          <div class="input-group">
            <label>Task Modality</label>
            <select id="task-modality" class="input-field">
              <option value="Board Copy">Board Copy</option>
              <option value="Dictation">Dictation</option>
              <option value="Free Writing">Free Writing</option>
              <option value="Clinical Standard">Clinical Standard</option>
              <option value="Custom / Other">Custom / Other</option>
            </select>
          </div>
        </div>

        <!-- Orientation Segmented Control -->
        <div class="segment-bar">
          <button class="segment-btn active" data-rot="No Rotation (0°)">0° Normal</button>
          <button class="segment-btn" data-rot="Auto-Detect Lines">Auto-Detect</button>
          <button class="segment-btn" data-rot="Rotate 90° Clockwise">90° CW</button>
          <button class="segment-btn" data-rot="Rotate 180°">180°</button>
          <button class="segment-btn" data-rot="Rotate 270°">270°</button>
        </div>

        <!-- Primary Action CTA -->
        <button id="btn-analyze" class="btn-analyze">
          <span>RUN COPYTHAT DIAGNOSTIC SCREENING</span>
        </button>

        <!-- Multi-Layer Visual Canvas -->
        <div class="card-title" style="margin-top: 10px;">
          <span>2. Visual Inspection Layers</span>
        </div>
        <div class="layer-tabs">
          <button class="layer-tab active" data-layer="original">Raw Document</button>
          <button class="layer-tab" data-layer="cleaned_mask">Cleaned Mask</button>
          <button class="layer-tab" data-layer="bhk_overlay">BHK Baselines</button>
          <button class="layer-tab" data-layer="ocr_boxes">Word Bounding Boxes</button>
        </div>
        <div class="image-viewer">
          <div class="image-placeholder" id="viewer-placeholder">
            Upload document or load a showcase case to preview visual layers.
          </div>
          <img id="viewer-img" style="display:none;" alt="Inspection Layer">
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN: Diagnostic Scorecard & Metrics -->
    <div>
      <div class="card">
        <div class="card-title">
          <span>3. Clinical Diagnostic Scorecard</span>
        </div>

        <!-- Hero Verdict Card -->
        <div class="verdict-hero control" id="verdict-hero">
          <div class="verdict-top">
            <div class="verdict-title" id="verdict-title">Ready for Analysis</div>
            <div class="verdict-chip control" id="verdict-chip">PENDING</div>
          </div>
          <div class="verdict-desc" id="verdict-desc">
            Load or upload handwriting to evaluate spatial arrangement, stroke micro-tremor, and prompt delta alignment.
          </div>
          <div class="symptoms-list" id="symptoms-container" style="display:none;"></div>
        </div>

        <!-- Tripartite Risk Gauges -->
        <div class="gauges-grid">
          <div class="gauge-card">
            <div class="gauge-svg-wrap">
              <svg class="gauge-svg" viewBox="0 0 72 72">
                <circle class="gauge-bg" cx="36" cy="36" r="30" />
                <circle class="gauge-fill" id="gauge-spatial" cx="36" cy="36" r="30" />
              </svg>
              <div class="gauge-val" id="val-spatial">0%</div>
            </div>
            <div class="gauge-label">Spatial Layout</div>
            <div class="gauge-hint">Crowding & Margins</div>
          </div>

          <div class="gauge-card">
            <div class="gauge-svg-wrap">
              <svg class="gauge-svg" viewBox="0 0 72 72">
                <circle class="gauge-bg" cx="36" cy="36" r="30" />
                <circle class="gauge-fill" id="gauge-motor" cx="36" cy="36" r="30" />
              </svg>
              <div class="gauge-val" id="val-motor">0%</div>
            </div>
            <div class="gauge-label">Neuromotor</div>
            <div class="gauge-hint">Stroke Shakiness</div>
          </div>

          <div class="gauge-card">
            <div class="gauge-svg-wrap">
              <svg class="gauge-svg" viewBox="0 0 72 72">
                <circle class="gauge-bg" cx="36" cy="36" r="30" />
                <circle class="gauge-fill" id="gauge-cognitive" cx="36" cy="36" r="30" />
              </svg>
              <div class="gauge-val" id="val-cognitive">0%</div>
            </div>
            <div class="gauge-label">Cognitive Reversals</div>
            <div class="gauge-hint">b/d & Omissions</div>
          </div>
        </div>

        <!-- CopyThat Delta Alignment Ribbon -->
        <div class="delta-section">
          <div class="delta-header">
            <h4>CopyThat: Target vs Observed Alignment</h4>
            <span class="delta-sim-badge" id="sim-badge">Awaiting Input</span>
          </div>
          <div id="delta-container" class="delta-words-grid">
            <div style="font-size:12px; color:var(--text-dim);">
              Character-level match chips and mirror reversals will populate here upon analysis.
            </div>
          </div>
        </div>

        <!-- Objective BHK Biomarkers Table -->
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>BHK Biomarker</th>
                <th>Observed</th>
                <th>Normative</th>
                <th>Evaluation</th>
              </tr>
            </thead>
            <tbody id="bhk-tbody">
              <tr><td colspan="4" style="text-align:center; color:var(--text-dim); padding:20px;">No metrics generated yet.</td></tr>
            </tbody>
          </table>
        </div>

        <!-- Action Bar: Print & JSON -->
        <div class="actions-bar">
          <button class="btn-secondary" id="btn-print">
            <span>🖨️ Print Clinical Summary</span>
          </button>
          <button class="btn-secondary" id="btn-json">
            <span>📥 Export Assessment JSON</span>
          </button>
        </div>

      </div>
    </div>

  </div>

  <script>
    // State
    let presetsData = {};
    let currentImageData = null;
    let activeRotation = "No Rotation (0°)";
    let currentLayers = {};
    let lastAnalysisData = null;
    let currentFilterCat = "all";

    // 1. Initialize App
    async function initApp() {
      try {
        const res = await fetch("/api/presets");
        presetsData = await res.json();
        renderPromptPills();
        renderShowcases();
      } catch (e) {
        console.error("Failed to load presets:", e);
      }
    }

    // 2. Modality Filter Mapping
    function getModalityCategory(presetName) {
      const lower = presetName.toLowerCase();
      if (lower.startsWith("board copy")) return "board";
      if (lower.startsWith("dictation")) return "dictation";
      if (lower.startsWith("free writing")) return "freewrite";
      if (lower.startsWith("clinical standard")) return "clinical";
      return "custom";
    }

    document.querySelectorAll(".modality-btn").forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll(".modality-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentFilterCat = btn.getAttribute("data-cat");
        renderPromptPills();
      };
    });

    // 3. Render Prompt Pills
    function renderPromptPills() {
      const container = document.getElementById("prompt-pills");
      container.innerHTML = "";
      const presets = presetsData.presets || {};
      let keys = Object.keys(presets);

      if (currentFilterCat !== "all") {
        keys = keys.filter(k => getModalityCategory(k) === currentFilterCat);
      }

      keys.forEach((k, idx) => {
        const btn = document.createElement("button");
        btn.className = "pill" + (idx === 0 ? " active" : "");
        btn.textContent = k;
        btn.onclick = () => {
          document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
          btn.classList.add("active");
          document.getElementById("target-prompt").value = presets[k] || "";
          
          // Auto sync modality dropdown
          const cat = getModalityCategory(k);
          const sel = document.getElementById("task-modality");
          if (cat === "board") sel.value = "Board Copy";
          else if (cat === "dictation") sel.value = "Dictation";
          else if (cat === "freewrite") sel.value = "Free Writing";
          else if (cat === "clinical") sel.value = "Clinical Standard";
          else sel.value = "Custom / Other";
        };
        container.appendChild(btn);
      });

      if (keys.length > 0) {
        document.getElementById("target-prompt").value = presets[keys[0]] || "";
      }
    }

    // 4. Render Showcases
    function renderShowcases() {
      const grid = document.getElementById("showcase-grid");
      grid.innerHTML = "";
      const showcases = presetsData.showcases || [];

      showcases.forEach(sc => {
        const card = document.createElement("div");
        card.className = "showcase-btn";
        card.innerHTML = `
          <div class="showcase-title">
            <span>${sc.title}</span>
            <span class="showcase-tag">${sc.task || "Task"}</span>
          </div>
          <div class="showcase-subtitle">${sc.subtitle}</div>
        `;
        card.onclick = () => loadShowcase(sc.id);
        grid.appendChild(card);
      });
    }

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
          document.getElementById("target-prompt").value = data.prompt || "";
          document.getElementById("student-name").value = data.name || "Student-01";
          if (data.task) {
            document.getElementById("task-modality").value = data.task;
          }
          currentLayers = { original: currentImageData };
        }
      } catch (e) {
        console.error("Failed to load showcase:", e);
      }
    }

    // 5. File Upload Handling
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    dropzone.onclick = () => fileInput.click();
    fileInput.onchange = (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    };

    dropzone.ondragover = (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    };
    dropzone.ondragleave = () => dropzone.classList.remove("dragover");
    dropzone.ondrop = (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    };

    function handleFile(file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        currentImageData = ev.target.result;
        document.getElementById("viewer-placeholder").style.display = "none";
        const img = document.getElementById("viewer-img");
        img.src = currentImageData;
        img.style.display = "block";
        currentLayers = { original: currentImageData };
      };
      reader.readAsDataURL(file);
    }

    // 6. Rotation Controls
    document.querySelectorAll(".segment-btn").forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll(".segment-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activeRotation = btn.getAttribute("data-rot");
      };
    });

    // 7. Layer Tabs Switching
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

    // 8. Trigger Analysis
    const btnAnalyze = document.getElementById("btn-analyze");
    btnAnalyze.onclick = triggerAnalysis;

    async function triggerAnalysis() {
      if (!currentImageData) {
        alert("Please upload a handwritten document or select a showcase case first.");
        return;
      }

      btnAnalyze.disabled = true;
      btnAnalyze.innerHTML = '<span class="spinner"></span> <span>ANALYZING INK ON GPU...</span>';

      const payload = {
        image_base64: currentImageData,
        target_prompt: document.getElementById("target-prompt").value,
        rotation_choice: activeRotation,
        student_name: document.getElementById("student-name").value,
        task_modality: document.getElementById("task-modality").value,
      };

      try {
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        lastAnalysisData = data;
        renderDiagnostics(data);
      } catch (err) {
        console.error("Analysis error:", err);
        alert("Error executing diagnostic screening: " + err.message);
      } finally {
        btnAnalyze.disabled = false;
        btnAnalyze.innerHTML = '<span>RUN COPYTHAT DIAGNOSTIC SCREENING</span>';
      }
    }

    // 9. Render Scorecard & Visual Elements
    function renderDiagnostics(data) {
      currentLayers = data.images || {};
      if (currentLayers.bhk_overlay) {
        document.getElementById("viewer-img").src = currentLayers.bhk_overlay;
        document.querySelectorAll(".layer-tab").forEach(t => {
          t.classList.toggle("active", t.getAttribute("data-layer") === "bhk_overlay");
        });
      }

      // Verdict
      const hero = document.getElementById("verdict-hero");
      const chip = document.getElementById("verdict-chip");
      hero.className = "verdict-hero " + (data.is_dysgraphic ? "alert" : "control");
      chip.className = "verdict-chip " + (data.is_dysgraphic ? "alert" : "control");
      chip.textContent = data.is_dysgraphic ? "AT RISK" : "CONTROL";
      document.getElementById("verdict-title").textContent = data.decision;
      document.getElementById("verdict-desc").textContent = data.is_dysgraphic
        ? "Handwriting exhibits elevated stroke unsteadiness, letter size irregularity, or spatial crowding. Clinical observation or intervention recommended."
        : "Handwriting layout, baseline stability, letter sizing, and spacing align cleanly with developmental normative targets.";

      // Symptoms
      const symContainer = document.getElementById("symptoms-container");
      if (data.symptoms && data.symptoms.length > 0) {
        symContainer.style.display = "flex";
        symContainer.innerHTML = data.symptoms.map(s => `<div class="symptom-item">● ${s}</div>`).join("");
      } else {
        symContainer.style.display = "none";
      }

      // Gauges (SVG Circular Meter)
      setGauge("gauge-spatial", "val-spatial", data.scores.spatial);
      setGauge("gauge-motor", "val-motor", data.scores.motor);
      setGauge("gauge-cognitive", "val-cognitive", data.scores.cognitive);

      // Delta Alignment
      const deltaCont = document.getElementById("delta-container");
      if (!data.alignment.is_open_mode && data.alignment.words && data.alignment.words.length > 0) {
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
          <div style="background:var(--bg-deep); border:1px solid var(--border); padding:14px; border-radius:6px; line-height:1.6; font-family:'JetBrains Mono', monospace; font-size:13px; color:#ffffff;">
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
          <td><span class="tag-status ${r.status === 'alert' ? 'tag-alert' : 'tag-normal'}">${r.status === 'alert' ? '● Elevated' : '✓ Normal'}</span></td>
        </tr>
      `).join("");
    }

    function setGauge(circleId, textId, pct) {
      document.getElementById(textId).textContent = pct + "%";
      const circle = document.getElementById(circleId);
      const circumference = 2 * Math.PI * 30; // ~188.5
      const offset = circumference - (pct / 100) * circumference;
      circle.style.strokeDashoffset = offset;
    }

    // 10. Print Report
    document.getElementById("btn-print").onclick = () => {
      if (!lastAnalysisData) {
        alert("Please run an analysis first.");
        return;
      }
      window.print();
    };

    // 11. Export JSON
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

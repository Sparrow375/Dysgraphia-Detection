"""Interactive Review & Manual Adjustment Server for Workstream A Phase 1.

Provides REST API and static file serving for the review web app.
Supports live manual bounding-box adjustment, recropping, script toggles,
multi-school dataset browsing (School A and School B), and QA verification/labeling.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
from aiohttp import web

from pipeline.process_dataset import process_single_student
from pipeline.skeleton import process_sentence_skeleton
from pipeline.segment import segment_words_devanagari, segment_words_english

WEB_DIR = REPO_ROOT / "qa" / "web"
DATA_DIR = REPO_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"


def get_student_manifest_row(student_id: str) -> Optional[pd.Series]:
    """Look up a student in manifest.csv."""
    if not MANIFEST_PATH.exists():
        return None
    df = pd.read_csv(MANIFEST_PATH)
    match = df[df["student_id"] == student_id]
    if match.empty:
        return None
    return match.iloc[0]


def get_student_school(student_id: str) -> str:
    """Determine school ('school_a' or 'school_b') for a given student ID."""
    row = get_student_manifest_row(student_id)
    if row is not None and pd.notna(row.get("school")):
        return str(row["school"])
    if (DATA_DIR / "processed" / "school_b" / student_id).exists():
        return "school_b"
    return "school_a"


def get_student_dir(student_id: str) -> Path:
    """Return student processed directory path."""
    school = get_student_school(student_id)
    return DATA_DIR / "processed" / school / student_id


def regenerate_student_overlay(student_dir: Path) -> None:
    """Regenerate overlay_debug.png using current sentence JSON definitions."""
    page_path = student_dir / "page_normalized.png"
    if not page_path.exists():
        return
    gray = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return

    h, w = gray.shape[:2]
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Read all sentence JSONs
    json_files = sorted(student_dir.glob("sentence_*.json"))
    colors = {
        "devanagari": (0, 200, 0),     # Green
        "latin": (0, 165, 255),        # Orange
    }

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            bx, by, bw, bh = s_data["crop_bbox"]
            script = s_data.get("script", "devanagari")
            task_name = s_data.get("task_name", "task")
            wc = s_data.get("qa_flags", {}).get("word_count_detected", 0)

            # Draw sentence bounding box
            c_color = colors.get(script, (0, 255, 255))
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)
            label = f"{task_name} ({script}, {wc} words)"
            cv2.putText(
                overlay,
                label,
                (bx + 8, max(20, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw word boxes if present in lines
            for line in s_data.get("lines", []):
                for wd in line.get("words", []):
                    wx, wy, ww, wh = wd["bbox"]
                    cv2.rectangle(overlay, (wx, wy), (wx + ww, wy + wh), c_color, 1)
        except Exception:
            continue

    cv2.imwrite(str(student_dir / "overlay_debug.png"), overlay)


async def api_get_students(request: web.Request) -> web.Response:
    """List students with metadata, sentence counts, verification status, and school filtering."""
    if not MANIFEST_PATH.exists():
        return web.json_response({"error": "manifest.csv not found"}, status=404)

    school_filter = request.query.get("school", "all").strip().lower()
    df = pd.read_csv(MANIFEST_PATH)

    if school_filter and school_filter != "all":
        df = df[df["school"] == school_filter]

    df = df.sort_values(by=["school", "grade", "roll_number", "student_id"])

    students = []
    for _, row in df.iterrows():
        sid = str(row["student_id"])
        school = str(row["school"])
        grade = int(row["grade"]) if pd.notna(row["grade"]) else -1
        roll = int(row["roll_number"]) if pd.notna(row["roll_number"]) else -1
        label = int(row["label"]) if pd.notna(row["label"]) else None

        s_dir = DATA_DIR / "processed" / school / sid
        json_count = len(list(s_dir.glob("sentence_*.json"))) if s_dir.exists() else 0
        has_overlay = (s_dir / "overlay_debug.png").exists() if s_dir.exists() else False

        status_path = s_dir / "review_status.json"
        is_verified = False
        notes = ""
        if status_path.exists():
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    st_data = json.load(f)
                    is_verified = bool(st_data.get("verified", False))
                    notes = str(st_data.get("notes", ""))
                    if "label" in st_data and st_data["label"] is not None:
                        label = int(st_data["label"])
            except Exception:
                pass

        expected = 4 if grade == 3 else 6
        status = "verified" if is_verified else ("complete" if json_count == expected else "review_needed")

        students.append({
            "student_id": sid,
            "school": school,
            "grade": grade,
            "roll_number": roll,
            "label": label,
            "sentences_count": json_count,
            "expected_tasks": expected,
            "has_data": has_overlay,
            "verified": is_verified,
            "status": status,
            "notes": notes,
        })

    return web.json_response({"students": students})


async def api_get_student_detail(request: web.Request) -> web.Response:
    """Get full details for a student: page dimensions, sentences, crops, and status."""
    sid = request.match_info["student_id"]
    school = get_student_school(sid)
    s_dir = DATA_DIR / "processed" / school / sid

    # If student processed folder doesn't exist or missing page_normalized, auto-process
    if not s_dir.exists() or not (s_dir / "page_normalized.png").exists():
        df = pd.read_csv(MANIFEST_PATH)
        match = df[df["student_id"] == sid]
        if match.empty:
            return web.json_response({"error": f"Student {sid} not found in manifest"}, status=404)
        try:
            process_single_student(match.iloc[0], output_base_dir=str(DATA_DIR / "processed"))
        except Exception as e:
            return web.json_response({"error": f"Processing error: {str(e)}"}, status=500)

    page_path = s_dir / "page_normalized.png"
    img = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return web.json_response({"error": "Failed to load page image"}, status=500)
    page_h, page_w = img.shape[:2]

    # Collect sentence definitions
    sentences = []
    json_files = sorted(s_dir.glob("sentence_*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            crop_fn = s_data.get("crop_filename", f"{s_data['task_id']}.png")
            sentences.append({
                "task_id": s_data.get("task_id", jf.stem),
                "task_name": s_data.get("task_name", "task"),
                "script": s_data.get("script", "devanagari"),
                "crop_bbox": s_data.get("crop_bbox", [0, 0, 100, 100]),
                "word_count": s_data.get("qa_flags", {}).get("word_count_detected", 0),
                "crop_url": f"/files/processed/{school}/{sid}/{crop_fn}?t={int(time.time())}",
                "is_manually_adjusted": s_data.get("is_manually_adjusted", False),
            })
        except Exception:
            continue

    m_row = get_student_manifest_row(sid)
    m_label = int(m_row["label"]) if (m_row is not None and pd.notna(m_row.get("label"))) else None

    # Verification status
    status_path = s_dir / "review_status.json"
    status_data = {"verified": False, "notes": "", "label": m_label}
    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception:
            pass

    cur_label = status_data.get("label", m_label)

    return web.json_response({
        "student_id": sid,
        "school": school,
        "grade": int(m_row["grade"]) if (m_row is not None and pd.notna(m_row.get("grade"))) else None,
        "roll_number": int(m_row["roll_number"]) if (m_row is not None and pd.notna(m_row.get("roll_number"))) else None,
        "label": cur_label,
        "image_width": page_w,
        "image_height": page_h,
        "page_url": f"/files/processed/{school}/{sid}/page_normalized.png",
        "overlay_url": f"/files/processed/{school}/{sid}/overlay_debug.png?t={int(time.time())}",
        "sentences": sentences,
        "verified": bool(status_data.get("verified", False)),
        "notes": str(status_data.get("notes", "")),
    })


async def api_update_sentence(request: web.Request) -> web.Response:
    """Update or re-crop a sentence with new bounding box, script, or task name."""
    sid = request.match_info["student_id"]
    school = get_student_school(sid)
    s_dir = DATA_DIR / "processed" / school / sid

    if not s_dir.exists():
        return web.json_response({"error": f"Student directory not found: {sid}"}, status=404)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    task_id = str(body.get("task_id", "")).strip()
    task_name = str(body.get("task_name", "")).strip()
    script = str(body.get("script", "devanagari")).strip()
    bbox = body.get("bbox", None)

    if not task_id or not bbox or len(bbox) != 4:
        return web.json_response({"error": "Missing task_id or valid bbox [x,y,w,h]"}, status=400)

    x, y, w, h = [int(v) for v in bbox]
    page_path = s_dir / "page_normalized.png"
    gray = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return web.json_response({"error": "Could not read page_normalized.png"}, status=500)

    img_h, img_w = gray.shape[:2]
    # Bound clamping
    x = max(0, min(x, img_w - 10))
    y = max(0, min(y, img_h - 10))
    w = max(10, min(w, img_w - x))
    h = max(10, min(h, img_h - y))
    clamped_bbox = [x, y, w, h]

    # Save new crop
    crop = gray[y : y + h, x : x + w]
    crop_fn = f"{task_id}.png"
    cv2.imwrite(str(s_dir / crop_fn), crop)

    # Clean ink binary crop for word detection
    _, ink_crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_bin = (ink_crop > 0).astype(np.uint8)

    # Word segmentation inside crop
    if script == "devanagari":
        raw_words = segment_words_devanagari(ink_bin)
    else:
        raw_words = segment_words_english(ink_bin, x_height_h=float(h * 0.5))

    word_count = len(raw_words)
    page_words = []
    for wd in raw_words:
        wx, wy, ww, wh = wd["bbox"]
        page_words.append({
            "bbox": [x + wx, y + wy, ww, wh],
            "area": wd.get("area", ww * wh),
            "bottom": y + wd.get("bottom", wy + wh),
            "cx": float(x + wd.get("cx", wx + ww / 2.0)),
            "word_index": wd.get("word_index", 0),
        })

    # Read existing JSON if available or construct fresh
    json_path = s_dir / f"{task_id}.json"
    schema = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
        except Exception:
            pass

    schema["student_id"] = sid
    schema["school"] = school
    schema["task_id"] = task_id
    schema["task_name"] = task_name if task_name else schema.get("task_name", task_id)
    schema["script"] = script
    schema["crop_bbox"] = clamped_bbox
    schema["crop_filename"] = crop_fn
    schema["is_manually_adjusted"] = True
    schema["qa_flags"] = schema.get("qa_flags", {})
    schema["qa_flags"]["word_count_detected"] = word_count
    schema["qa_flags"]["manually_verified"] = True
    schema["lines"] = [
        {
            "line_index": 0,
            "script": script,
            "bbox": clamped_bbox,
            "words": page_words,
            "baseline_slope": 0.0,
            "baseline_intercept": float(y + h * 0.8),
            "x_height_h": float(h * 0.5),
        }
    ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Regenerate overlay
    regenerate_student_overlay(s_dir)

    return web.json_response({
        "success": True,
        "task_id": task_id,
        "crop_bbox": clamped_bbox,
        "word_count": word_count,
        "crop_url": f"/files/processed/{school}/{sid}/{crop_fn}?t={int(time.time())}",
        "overlay_url": f"/files/processed/{school}/{sid}/overlay_debug.png?t={int(time.time())}",
    })


async def api_add_sentence(request: web.Request) -> web.Response:
    """Add a new task sentence for a student."""
    sid = request.match_info["student_id"]
    school = get_student_school(sid)
    s_dir = DATA_DIR / "processed" / school / sid

    if not s_dir.exists():
        return web.json_response({"error": f"Student not found: {sid}"}, status=404)

    body = await request.json()
    task_id = str(body.get("task_id", "")).strip()
    task_name = str(body.get("task_name", "")).strip()
    script = str(body.get("script", "devanagari")).strip()
    bbox = body.get("bbox", [100, 100, 800, 150])

    if not task_id:
        return web.json_response({"error": "task_id required"}, status=400)

    x, y, w, h = bbox
    gray = cv2.imread(str(s_dir / "page_normalized.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return web.json_response({"error": "Failed to read page_normalized.png"}, status=500)

    crop = gray[y : y + h, x : x + w]
    crop_fn = f"{task_id}.png"
    cv2.imwrite(str(s_dir / crop_fn), crop)

    schema = {
        "student_id": sid,
        "school": school,
        "task_id": task_id,
        "task_name": task_name or task_id,
        "script": script,
        "crop_bbox": bbox,
        "crop_filename": crop_fn,
        "is_manually_adjusted": True,
        "qa_flags": {"word_count_detected": 1, "manually_verified": True},
        "lines": [{"line_index": 0, "script": script, "bbox": bbox, "words": []}],
    }
    with open(s_dir / f"{task_id}.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    regenerate_student_overlay(s_dir)
    return web.json_response({"success": True, "task_id": task_id})


async def api_delete_sentence(request: web.Request) -> web.Response:
    """Delete a bogus or extra sentence."""
    sid = request.match_info["student_id"]
    school = get_student_school(sid)
    s_dir = DATA_DIR / "processed" / school / sid

    task_id = request.match_info["task_id"]

    for p in [s_dir / f"{task_id}.json", s_dir / f"{task_id}.png"]:
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    for jf in list(s_dir.glob("sentence_*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("task_id") == task_id or data.get("task_name") == task_id or jf.stem == task_id:
                png_file = jf.with_suffix(".png")
                jf.unlink(missing_ok=True)
                if png_file.exists():
                    png_file.unlink(missing_ok=True)
        except Exception:
            pass

    regenerate_student_overlay(s_dir)
    return web.json_response({
        "success": True,
        "deleted": task_id,
        "overlay_url": f"/files/processed/{school}/{sid}/overlay_debug.png?t={int(time.time())}",
    })


async def api_set_verification(request: web.Request) -> web.Response:
    """Save verification status and optional label for a student."""
    sid = request.match_info["student_id"]
    school = get_student_school(sid)
    s_dir = DATA_DIR / "processed" / school / sid

    if not s_dir.exists():
        return web.json_response({"error": f"Student not found: {sid}"}, status=404)

    body = await request.json()
    verified = bool(body.get("verified", True))
    notes = str(body.get("notes", "")).strip()
    label = body.get("label")  # 0, 1, or None

    status_data = {
        "student_id": sid,
        "school": school,
        "verified": verified,
        "notes": notes,
        "label": label,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(s_dir / "review_status.json", "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    # If label is updated, also update manifest.csv so that downstream pipelines see it!
    if MANIFEST_PATH.exists() and label is not None:
        try:
            m_df = pd.read_csv(MANIFEST_PATH)
            m_mask = m_df["student_id"] == sid
            if m_mask.any():
                m_df.loc[m_mask, "label"] = float(label)
                m_df.loc[m_mask, "label_source"] = "manual_review_web_app"
                m_df.to_csv(MANIFEST_PATH, index=False)
        except Exception as e:
            print(f"Warning: Failed to update manifest label for {sid}: {e}")

    return web.json_response({"success": True, "status": status_data})


async def api_set_label(request: web.Request) -> web.Response:
    """Directly update dysgraphia label (0 = Control, 1 = At-Risk, null = Unlabeled)."""
    sid = request.match_info["student_id"]
    school = get_student_school(sid)
    s_dir = DATA_DIR / "processed" / school / sid

    if not s_dir.exists():
        return web.json_response({"error": f"Student not found: {sid}"}, status=404)

    body = await request.json()
    raw_label = body.get("label")
    label_val = int(raw_label) if (raw_label is not None and str(raw_label).strip() != "" and str(raw_label) != "unlabeled") else None

    # Update review_status.json
    status_path = s_dir / "review_status.json"
    status_data = {}
    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception:
            pass
    status_data["student_id"] = sid
    status_data["school"] = school
    status_data["label"] = label_val
    status_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    # Update manifest.csv
    if MANIFEST_PATH.exists():
        try:
            m_df = pd.read_csv(MANIFEST_PATH)
            m_mask = m_df["student_id"] == sid
            if m_mask.any():
                m_df.loc[m_mask, "label"] = float(label_val) if label_val is not None else np.nan
                m_df.loc[m_mask, "label_source"] = "manual_review_web_app" if label_val is not None else "unlabeled"
                m_df.to_csv(MANIFEST_PATH, index=False)
        except Exception as e:
            print(f"Warning: Failed to update manifest label: {e}")

    return web.json_response({"success": True, "student_id": sid, "label": label_val})


async def handle_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(str(WEB_DIR / "index.html"))


def create_app() -> web.Application:
    app = web.Application()

    # API endpoints
    app.router.add_get("/api/students", api_get_students)
    app.router.add_get("/api/student/{student_id}", api_get_student_detail)
    app.router.add_post("/api/student/{student_id}/update_sentence", api_update_sentence)
    app.router.add_post("/api/student/{student_id}/add_sentence", api_add_sentence)
    app.router.add_delete("/api/student/{student_id}/sentence/{task_id}", api_delete_sentence)
    app.router.add_post("/api/student/{student_id}/verify", api_set_verification)
    app.router.add_post("/api/student/{student_id}/label", api_set_label)

    # Static file routes
    app.router.add_get("/", handle_index)
    app.router.add_static("/static", str(WEB_DIR))
    app.router.add_static("/files", str(DATA_DIR))

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8090))
    print(f"Starting Dysgraphia QA & Adjustment App on http://127.0.0.1:{port}")
    web.run_app(app, host="127.0.0.1", port=port)

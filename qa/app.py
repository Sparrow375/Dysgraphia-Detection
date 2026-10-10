"""Interactive Review & Manual Adjustment Server for Workstream A Phase 1.

Provides REST API and static file serving for the review web app.
Supports live manual bounding-box adjustment, recropping, script toggles, and QA verification.
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
PROCESSED_DIR = DATA_DIR / "processed" / "school_a"
MANIFEST_PATH = DATA_DIR / "manifest.csv"


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
    """List all School A students with metadata, sentence counts, and verification status."""
    if not MANIFEST_PATH.exists():
        return web.json_response({"error": "manifest.csv not found"}, status=404)

    df = pd.read_csv(MANIFEST_PATH)
    df_a = df[df["school"] == "school_a"].sort_values(by=["grade", "roll_number"])

    students = []
    for _, row in df_a.iterrows():
        sid = str(row["student_id"])
        grade = int(row["grade"])
        roll = int(row["roll_number"]) if pd.notna(row["roll_number"]) else -1
        label = int(row["label"]) if pd.notna(row["label"]) else 0

        s_dir = PROCESSED_DIR / sid
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
            except Exception:
                pass

        expected = 4 if grade == 3 else 6
        status = "verified" if is_verified else ("complete" if json_count == expected else "review_needed")

        students.append({
            "student_id": sid,
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
    s_dir = PROCESSED_DIR / sid

    # If student processed folder doesn't exist or missing page_normalized, auto-process
    if not s_dir.exists() or not (s_dir / "page_normalized.png").exists():
        df = pd.read_csv(MANIFEST_PATH)
        match = df[(df["school"] == "school_a") & (df["student_id"] == sid)]
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
                "crop_url": f"/files/processed/school_a/{sid}/{crop_fn}?t={int(time.time())}",
                "is_manually_adjusted": s_data.get("is_manually_adjusted", False),
            })
        except Exception:
            continue

    # Verification status
    status_path = s_dir / "review_status.json"
    status_data = {"verified": False, "notes": ""}
    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception:
            pass

    return web.json_response({
        "student_id": sid,
        "image_width": page_w,
        "image_height": page_h,
        "page_url": f"/files/processed/school_a/{sid}/page_normalized.png",
        "overlay_url": f"/files/processed/school_a/{sid}/overlay_debug.png?t={int(time.time())}",
        "sentences": sentences,
        "verified": bool(status_data.get("verified", False)),
        "notes": str(status_data.get("notes", "")),
    })


async def api_update_sentence(request: web.Request) -> web.Response:
    """Update or re-crop a sentence with new bounding box, script, or task name."""
    sid = request.match_info["student_id"]
    s_dir = PROCESSED_DIR / sid
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
    # Adjust word bboxes to page coordinates
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
    schema["school"] = "school_a"
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
        "crop_url": f"/files/processed/school_a/{sid}/{crop_fn}?t={int(time.time())}",
        "overlay_url": f"/files/processed/school_a/{sid}/overlay_debug.png?t={int(time.time())}",
    })


async def api_add_sentence(request: web.Request) -> web.Response:
    """Add a new task sentence for a student."""
    sid = request.match_info["student_id"]
    s_dir = PROCESSED_DIR / sid
    if not s_dir.exists():
        return web.json_response({"error": f"Student not found: {sid}"}, status=404)

    body = await request.json()
    task_id = str(body.get("task_id", "")).strip()
    task_name = str(body.get("task_name", "")).strip()
    script = str(body.get("script", "devanagari")).strip()
    bbox = body.get("bbox", [100, 100, 800, 150])

    if not task_id:
        return web.json_response({"error": "task_id required"}, status=400)

    # Delegate to update_sentence logic
    request_data = {"task_id": task_id, "task_name": task_name, "script": script, "bbox": bbox}
    # Create fake request for update_sentence
    dummy_req = request.clone(method="POST")
    # Call logic directly
    x, y, w, h = bbox
    gray = cv2.imread(str(s_dir / "page_normalized.png"), cv2.IMREAD_GRAYSCALE)
    crop = gray[y : y + h, x : x + w]
    crop_fn = f"{task_id}.png"
    cv2.imwrite(str(s_dir / crop_fn), crop)

    schema = {
        "student_id": sid,
        "school": "school_a",
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
    task_id = request.match_info["task_id"]
    s_dir = PROCESSED_DIR / sid

    for p in [s_dir / f"{task_id}.json", s_dir / f"{task_id}.png"]:
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    regenerate_student_overlay(s_dir)
    return web.json_response({"success": True, "deleted": task_id})


async def api_set_verification(request: web.Request) -> web.Response:
    """Save verification status for a student."""
    sid = request.match_info["student_id"]
    s_dir = PROCESSED_DIR / sid
    if not s_dir.exists():
        return web.json_response({"error": f"Student not found: {sid}"}, status=404)

    body = await request.json()
    verified = bool(body.get("verified", True))
    notes = str(body.get("notes", "")).strip()

    status_data = {
        "student_id": sid,
        "verified": verified,
        "notes": notes,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(s_dir / "review_status.json", "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    return web.json_response({"success": True, "status": status_data})


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

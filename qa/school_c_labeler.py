"""Lightweight, zero-friction Dysgraphia Labeling Web App for School C (Dawre).

Serves images directly from data/school_c/raw/ and allows 1-click / keyboard
labeling (1 = Potential Dysgraphic, 0 = Control / Normal).
Automatically updates:
  - data/school_c/labels.json
  - data/school_c/labels.csv
  - data/manifest.csv (synced)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from aiohttp import web

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SCHOOL_C_RAW = DATA_DIR / "school_c" / "raw"
LABELS_JSON = DATA_DIR / "school_c" / "labels.json"
LABELS_CSV = DATA_DIR / "school_c" / "labels.csv"
MANIFEST_CSV = DATA_DIR / "manifest.csv"

# Fallback download dir if local raw missing
DOWNLOAD_DIR = Path(r"F:\Avaneesh\download\Dawre Handwritten -20261027T140353Z-1-001\Dawre Handwritten")


def get_base_dir() -> Path:
    if SCHOOL_C_RAW.exists() and any(SCHOOL_C_RAW.glob("*/*.jpg")):
        return SCHOOL_C_RAW
    return DOWNLOAD_DIR


def load_labels() -> Dict[str, Any]:
    if LABELS_JSON.exists():
        try:
            with open(LABELS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_labels(labels: Dict[str, Any]) -> None:
    LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    # Sync to labels.csv
    rows = []
    base_dir = get_base_dir()
    for img_path in sorted(base_dir.glob("*/*.jpg")):
        rel = img_path.relative_to(base_dir).as_posix()
        grade_match = re.search(r"\d+", img_path.parent.name)
        grade = int(grade_match.group(0)) if grade_match else -1
        student_id = f"SCHC_G{grade}_{img_path.stem}"
        lbl_info = labels.get(student_id, {})
        val = lbl_info.get("label")
        rows.append({
            "student_id": student_id,
            "school": "school_c",
            "school_name": "Dawre",
            "grade": grade,
            "filename": img_path.name,
            "relative_image_path": f"data/school_c/raw/{rel}",
            "label": val if val is not None else "",
            "label_name": "Dysgraphic" if val == 1 else ("Control" if val == 0 else "Unlabeled"),
            "updated_at": lbl_info.get("updated_at", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(LABELS_CSV, index=False)

    # Sync to manifest.csv if manifest exists
    sync_to_manifest(rows)


def sync_to_manifest(school_c_rows: List[Dict[str, Any]]) -> None:
    if not MANIFEST_CSV.exists():
        return
    try:
        manifest_df = pd.read_csv(MANIFEST_CSV)
        # Drop existing school_c entries
        manifest_df = manifest_df[manifest_df["school"] != "school_c"]

        new_entries = []
        for r in school_c_rows:
            label_val = float(r["label"]) if r["label"] != "" else float("nan")
            label_source = "manual_review_web_app" if label_val == 1.0 else (
                "manual_review_web_app_control" if label_val == 0.0 else "unlabeled"
            )
            new_entries.append({
                "student_id": r["student_id"],
                "school": "school_c",
                "school_name": "Dawre",
                "grade": r["grade"],
                "section": "unassigned",
                "roll_number": -1,
                "label": label_val,
                "label_source": label_source,
                "match_status": "HELD_OUT",
                "image_filename": r["filename"],
                "relative_image_path": r["relative_image_path"],
                "is_held_out": True,
            })

        combined = pd.concat([manifest_df, pd.DataFrame(new_entries)], ignore_index=True)
        combined.to_csv(MANIFEST_CSV, index=False)
    except Exception as e:
        print(f"Warning: Manifest sync error: {e}")


def scan_images() -> List[Dict[str, Any]]:
    base_dir = get_base_dir()
    labels = load_labels()
    items = []
    for img_path in sorted(base_dir.glob("*/*.jpg")):
        rel = img_path.relative_to(base_dir).as_posix()
        grade_match = re.search(r"\d+", img_path.parent.name)
        grade = int(grade_match.group(0)) if grade_match else -1
        student_id = f"SCHC_G{grade}_{img_path.stem}"
        lbl_info = labels.get(student_id, {})
        items.append({
            "student_id": student_id,
            "grade": grade,
            "folder": img_path.parent.name,
            "filename": img_path.name,
            "relative_path": rel,
            "image_url": f"/api/image/{rel}",
            "label": lbl_info.get("label", None),
            "updated_at": lbl_info.get("updated_at", None),
        })
    return items


async def handle_get_images(request: web.Request) -> web.Response:
    items = scan_images()
    return web.json_response({"images": items, "total": len(items)})


async def handle_set_label(request: web.Request) -> web.Response:
    data = await request.json()
    student_id = data.get("student_id")
    raw_label = data.get("label")
    if student_id is None:
        return web.json_response({"error": "student_id required"}, status=400)

    label_val: Optional[int] = None
    if raw_label in (0, 1, "0", "1"):
        label_val = int(raw_label)

    labels = load_labels()
    if label_val is not None:
        labels[student_id] = {
            "label": label_val,
            "updated_at": datetime.now().isoformat(),
        }
    else:
        labels.pop(student_id, None)

    save_labels(labels)
    return web.json_response({"success": True, "student_id": student_id, "label": label_val})


async def handle_serve_image(request: web.Request) -> web.Response:
    rel_path = request.match_info["rel_path"]
    base_dir = get_base_dir()
    full_path = (base_dir / rel_path).resolve()
    if not full_path.exists():
        return web.Response(text="Image not found", status=404)
    return web.FileResponse(full_path)


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>School C (Dawre) — Dysgraphia Labeler</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-gray: #64748b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        /* Top Navigation Header */
        header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
        }
        .header-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-title h1 {
            font-size: 1.15rem;
            font-weight: 700;
            color: #38bdf8;
            letter-spacing: -0.02em;
        }
        .badge-school {
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .stats-bar {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .stat-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .stat-num { font-weight: 700; }
        .stat-green { color: var(--accent-green); }
        .stat-red { color: var(--accent-red); }
        .stat-gray { color: var(--accent-gray); }

        /* Main Workspace */
        .workspace {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        /* Sidebar List */
        .sidebar {
            width: 320px;
            background: #111827;
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }
        .filter-section {
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .filter-row {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }
        .btn-filter {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn-filter:hover { background: #334155; color: var(--text-main); }
        .btn-filter.active { background: var(--accent-blue); border-color: var(--accent-blue); color: white; font-weight: 600; }

        .items-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }
        .item-card {
            padding: 10px 12px;
            border-radius: 8px;
            border: 1px solid transparent;
            margin-bottom: 6px;
            cursor: pointer;
            background: rgba(30, 41, 59, 0.4);
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.15s;
        }
        .item-card:hover { background: var(--bg-card); }
        .item-card.active {
            background: #1e293b;
            border-color: #38bdf8;
            box-shadow: 0 0 0 1px #38bdf8;
        }
        .item-card-left {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .item-id { font-size: 0.85rem; font-weight: 600; color: #f1f5f9; }
        .item-filename { font-size: 0.72rem; color: var(--text-muted); }
        .label-pill {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.72rem;
            font-weight: 600;
        }
        .pill-dysgraphic { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .pill-control { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
        .pill-unlabeled { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.4); }

        /* Viewer Section */
        .viewer-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #020617;
            position: relative;
            overflow: hidden;
        }
        .viewer-toolbar {
            background: rgba(15, 23, 42, 0.9);
            border-bottom: 1px solid var(--border);
            padding: 8px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
            z-index: 10;
        }
        .student-header {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .student-title { font-size: 1rem; font-weight: 700; color: #f8fafc; }
        .student-grade {
            background: #334155;
            color: #cbd5e1;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .zoom-controls {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .btn-tool {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .btn-tool:hover { background: #334155; }

        /* Main Image View */
        .image-wrapper {
            flex: 1;
            overflow: auto;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .image-wrapper img {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
            transition: transform 0.15s ease-out;
        }

        /* Action / Label Floating Bar at Bottom */
        .action-dock {
            background: rgba(30, 41, 59, 0.95);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            gap: 16px;
            box-shadow: 0 20px 35px -10px rgba(0,0,0,0.7);
            z-index: 20;
        }
        .btn-action {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            font-size: 0.95rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.12s ease;
        }
        .btn-control {
            background: #15803d;
            color: white;
        }
        .btn-control:hover, .btn-control.selected {
            background: #22c55e;
            box-shadow: 0 0 15px rgba(34, 197, 94, 0.5);
            transform: translateY(-2px);
        }
        .btn-dysgraphic {
            background: #b91c1c;
            color: white;
        }
        .btn-dysgraphic:hover, .btn-dysgraphic.selected {
            background: #ef4444;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.5);
            transform: translateY(-2px);
        }
        .btn-clear {
            background: #475569;
            color: white;
        }
        .btn-clear:hover { background: #64748b; }
        .kbd-hint {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .nav-buttons {
            display: flex;
            gap: 6px;
            margin-left: 8px;
            border-left: 1px solid var(--border);
            padding-left: 14px;
        }
        .btn-nav {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 8px 14px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-nav:hover { background: #334155; }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>School C (Dawre) Dysgraphia Labeler</h1>
            <span class="badge-school">HELD-OUT COHORT</span>
        </div>
        <div class="stats-bar">
            <div class="stat-item">Total: <span class="stat-num" id="stat-total">47</span></div>
            <div class="stat-item stat-green">Control (0): <span class="stat-num" id="stat-control">0</span></div>
            <div class="stat-item stat-red">Dysgraphic (1): <span class="stat-num" id="stat-dysgraphic">0</span></div>
            <div class="stat-item stat-gray">Remaining: <span class="stat-num" id="stat-remaining">47</span></div>
        </div>
    </header>

    <div class="workspace">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="filter-section">
                <div class="filter-row" id="grade-filters">
                    <button class="btn-filter active" data-grade="all">All Grades</button>
                    <button class="btn-filter" data-grade="3">G3</button>
                    <button class="btn-filter" data-grade="4">G4</button>
                    <button class="btn-filter" data-grade="5">G5</button>
                    <button class="btn-filter" data-grade="6">G6</button>
                    <button class="btn-filter" data-grade="7">G7</button>
                </div>
                <div class="filter-row" id="status-filters">
                    <button class="btn-filter active" data-status="all">All</button>
                    <button class="btn-filter" data-status="unlabeled">Unlabeled</button>
                    <button class="btn-filter" data-status="labeled">Labeled</button>
                </div>
            </div>
            <div class="items-list" id="items-list">
                <!-- Dynamically populated -->
            </div>
        </div>

        <!-- Viewer -->
        <div class="viewer-container">
            <div class="viewer-toolbar">
                <div class="student-header">
                    <span class="student-title" id="current-title">—</span>
                    <span class="student-grade" id="current-grade">—</span>
                    <span style="font-size:0.75rem; color:#94a3b8;" id="current-index">—</span>
                </div>
                <div class="zoom-controls">
                    <button class="btn-tool" onclick="adjustZoom(-0.15)">−</button>
                    <button class="btn-tool" onclick="resetZoom()">Fit</button>
                    <button class="btn-tool" onclick="adjustZoom(0.15)">+</button>
                    <button class="btn-tool" onclick="rotateImage()">Rotate ↻</button>
                </div>
            </div>

            <div class="image-wrapper" id="image-wrapper">
                <img id="main-image" src="" alt="Handwriting sheet" />
            </div>

            <!-- Action Dock -->
            <div class="action-dock">
                <button class="btn-action btn-control" id="btn-label-0" onclick="setLabel(0)">
                    <span>✓ Control / Normal</span>
                    <span class="kbd-hint">0</span>
                </button>
                <button class="btn-action btn-dysgraphic" id="btn-label-1" onclick="setLabel(1)">
                    <span>⚠ Potential Dysgraphic</span>
                    <span class="kbd-hint">1</span>
                </button>
                <button class="btn-action btn-clear" onclick="setLabel(null)">
                    <span>Clear</span>
                    <span class="kbd-hint">U</span>
                </button>
                <div class="nav-buttons">
                    <button class="btn-nav" onclick="prevImage()">← Prev</button>
                    <button class="btn-nav" onclick="nextImage()">Next →</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let allImages = [];
        let filteredImages = [];
        let currentIndex = 0;
        let activeGrade = 'all';
        let activeStatus = 'all';
        let currentZoom = 1.0;
        let currentRotation = 0;

        async function init() {
            const res = await fetch('/api/images');
            const data = await res.json();
            allImages = data.images;
            applyFilters();
            setupKeyboard();
        }

        function applyFilters() {
            filteredImages = allImages.filter(item => {
                if (activeGrade !== 'all' && item.grade != activeGrade) return false;
                if (activeStatus === 'unlabeled' && item.label !== null && item.label !== undefined) return false;
                if (activeStatus === 'labeled' && (item.label === null || item.label === undefined)) return false;
                return true;
            });

            updateStats();
            renderList();
            if (filteredImages.length > 0) {
                if (currentIndex >= filteredImages.length) currentIndex = 0;
                selectItem(currentIndex);
            } else {
                document.getElementById('current-title').textContent = 'No matching images';
                document.getElementById('main-image').src = '';
            }
        }

        function updateStats() {
            const total = allImages.length;
            const controls = allImages.filter(i => i.label === 0).length;
            const dysgraphics = allImages.filter(i => i.label === 1).length;
            const remaining = total - controls - dysgraphics;

            document.getElementById('stat-total').textContent = total;
            document.getElementById('stat-control').textContent = controls;
            document.getElementById('stat-dysgraphic').textContent = dysgraphics;
            document.getElementById('stat-remaining').textContent = remaining;
        }

        function renderList() {
            const container = document.getElementById('items-list');
            container.innerHTML = '';
            filteredImages.forEach((item, idx) => {
                const el = document.createElement('div');
                el.className = `item-card ${idx === currentIndex ? 'active' : ''}`;
                el.onclick = () => selectItem(idx);

                let pillClass = 'pill-unlabeled';
                let pillText = 'Unlabeled';
                if (item.label === 0) {
                    pillClass = 'pill-control';
                    pillText = 'Control (0)';
                } else if (item.label === 1) {
                    pillClass = 'pill-dysgraphic';
                    pillText = 'Dysgraphic (1)';
                }

                el.innerHTML = `
                    <div class="item-card-left">
                        <span class="item-id">${item.student_id}</span>
                        <span class="item-filename">Grade ${item.grade} • ${item.filename}</span>
                    </div>
                    <span class="label-pill ${pillClass}">${pillText}</span>
                `;
                container.appendChild(el);
            });
        }

        function selectItem(idx) {
            currentIndex = idx;
            const item = filteredImages[idx];
            if (!item) return;

            document.getElementById('current-title').textContent = item.student_id;
            document.getElementById('current-grade').textContent = `Grade ${item.grade}`;
            document.getElementById('current-index').textContent = `(${idx + 1} of ${filteredImages.length})`;

            const img = document.getElementById('main-image');
            img.src = item.image_url;
            resetZoom();

            // Highlight buttons
            const b0 = document.getElementById('btn-label-0');
            const b1 = document.getElementById('btn-label-1');
            b0.classList.toggle('selected', item.label === 0);
            b1.classList.toggle('selected', item.label === 1);

            // Highlight list item
            const cards = document.querySelectorAll('.item-card');
            cards.forEach((c, i) => c.classList.toggle('active', i === idx));
            cards[idx]?.scrollIntoView({ block: 'nearest' });
        }

        async function setLabel(labelValue) {
            const item = filteredImages[currentIndex];
            if (!item) return;

            item.label = labelValue;
            // Also update in allImages
            const global = allImages.find(i => i.student_id === item.student_id);
            if (global) global.label = labelValue;

            updateStats();
            renderList();

            // Post to backend
            await fetch('/api/label', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ student_id: item.student_id, label: labelValue })
            });

            // Automatically advance to next image
            if (currentIndex < filteredImages.length - 1) {
                selectItem(currentIndex + 1);
            }
        }

        function prevImage() {
            if (currentIndex > 0) selectItem(currentIndex - 1);
        }

        function nextImage() {
            if (currentIndex < filteredImages.length - 1) selectItem(currentIndex + 1);
        }

        function adjustZoom(delta) {
            currentZoom = Math.max(0.3, Math.min(3.0, currentZoom + delta));
            applyTransform();
        }

        function resetZoom() {
            currentZoom = 1.0;
            currentRotation = 0;
            applyTransform();
        }

        function rotateImage() {
            currentRotation = (currentRotation + 90) % 360;
            applyTransform();
        }

        function applyTransform() {
            const img = document.getElementById('main-image');
            img.style.transform = `scale(${currentZoom}) rotate(${currentRotation}deg)`;
        }

        function setupKeyboard() {
            window.addEventListener('keydown', (e) => {
                if (e.target.tagName === 'INPUT') return;
                if (e.key === '0') {
                    setLabel(0);
                } else if (e.key === '1') {
                    setLabel(1);
                } else if (e.key === 'u' || e.key === 'U') {
                    setLabel(null);
                } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
                    nextImage();
                } else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
                    prevImage();
                } else if (e.key === 'r' || e.key === 'R') {
                    rotateImage();
                }
            });

            // Filter button listeners
            document.querySelectorAll('#grade-filters button').forEach(btn => {
                btn.onclick = () => {
                    document.querySelectorAll('#grade-filters button').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    activeGrade = btn.dataset.grade;
                    applyFilters();
                };
            });

            document.querySelectorAll('#status-filters button').forEach(btn => {
                btn.onclick = () => {
                    document.querySelectorAll('#status-filters button').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    activeStatus = btn.dataset.status;
                    applyFilters();
                };
            });
        }

        window.onload = init;
    </script>
</body>
</html>
"""


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=HTML_CONTENT, content_type="text/html")


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/images", handle_get_images)
    app.router.add_post("/api/label", handle_set_label)
    app.router.add_get("/api/image/{rel_path:.*}", handle_serve_image)
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run School C Dysgraphia Labeler")
    parser.add_argument("--port", type=int, default=8090, help="Port to run server on")
    args = parser.parse_args()

    # Pre-sync labels csv initially
    save_labels(load_labels())
    print(f"School C Dysgraphia Labeler running at http://localhost:{args.port}")
    web.run_app(create_app(), host="127.0.0.1", port=args.port)

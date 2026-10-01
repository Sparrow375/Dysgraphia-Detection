import os
import json
import time
import base64
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
INDEX_CSV = DATA_DIR / "index.csv"
CONFIG_FILE = BASE_DIR / "config.json"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Universal Stylus Data Collection Server", version="1.0.0")

# Enable CORS for local Wi-Fi hotspot access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def init_index_csv():
    if not INDEX_CSV.exists():
        header = [
            "timestamp_iso",
            "epoch_ms",
            "student_id",
            "grade",
            "section",
            "roll_number",
            "dominant_hand",
            "device_id",
            "operator",
            "tasks_completed",
            "total_samples",
            "total_strokes",
            "total_duration_sec",
            "json_filename",
            "notes"
        ]
        with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


init_index_csv()


@app.get("/api/config")
def get_config():
    return load_config()


@app.get("/api/status")
def get_status():
    config = load_config()
    total_sessions = 0
    by_grade = {}
    recent_sessions = []

    if INDEX_CSV.exists():
        try:
            df = pd.read_csv(INDEX_CSV).fillna("")
            total_sessions = len(df)
            if not df.empty and "grade" in df.columns:
                by_grade = df["grade"].value_counts().to_dict()
                # Cast keys to string
                by_grade = {str(k): int(v) for k, v in by_grade.items()}
                # Get last 10 sessions
                recent = df.tail(10).to_dict(orient="records")
                recent_sessions = recent[::-1]
        except Exception as e:
            print(f"Error reading index.csv: {e}")

    return {
        "status": "online",
        "app_name": config.get("name", "Stylus Collector"),
        "version": config.get("app_version", "1.0.0"),
        "total_sessions": total_sessions,
        "by_grade": by_grade,
        "recent_sessions": recent_sessions,
        "sessions_dir": str(SESSIONS_DIR),
    }


@app.get("/api/roster")
def get_roster():
    if not INDEX_CSV.exists():
        return {"records": []}
    try:
        df = pd.read_csv(INDEX_CSV).fillna("")
        # Sort by grade, section, roll_number
        if "roll_number" in df.columns and "grade" in df.columns:
            df = df.sort_values(by=["grade", "section", "roll_number"])
        return {"records": df.to_dict(orient="records")}
    except Exception as e:
        return {"records": [], "error": str(e)}


@app.post("/api/check_student")
async def check_student(data: Dict[str, Any]):
    grade = data.get("grade")
    section = str(data.get("section", "")).strip().upper()
    roll_number = data.get("roll_number")

    if not INDEX_CSV.exists():
        return {"exists": False}

    try:
        df = pd.read_csv(INDEX_CSV)
        if df.empty:
            return {"exists": False}
        
        matches = df[
            (df["grade"] == int(grade)) & 
            (df["section"].astype(str).str.upper() == section) & 
            (df["roll_number"] == int(roll_number))
        ]
        if not matches.empty:
            return {
                "exists": True,
                "count": len(matches),
                "last_timestamp": matches.iloc[-1].get("timestamp_iso", "")
            }
    except Exception as e:
        print(f"Error checking student: {e}")

    return {"exists": False}


@app.post("/api/session")
async def save_session(session: Dict[str, Any]):
    try:
        grade = session.get("grade", 3)
        section = str(session.get("section", "A")).strip().upper()
        roll_number = session.get("roll_number", 1)
        student_id = f"G{grade}_{section}_Roll{int(roll_number):02d}"
        
        epoch_ms = int(time.time() * 1000)
        timestamp_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        file_prefix = f"{student_id}__{epoch_ms}"
        
        tasks = session.get("tasks", [])
        total_samples = 0
        total_strokes = 0
        total_duration_ms = 0
        completed_task_ids = []

        # Save individual task CSVs and PNGs
        for task in tasks:
            task_id = task.get("task_id", "task")
            completed_task_ids.append(task_id)
            records = task.get("records", [])
            strokes = task.get("strokes", [])
            total_samples += len(records)
            total_strokes += len(strokes)
            total_duration_ms += task.get("duration_ms", 0)

            # 1. Save Task CSV
            if records:
                csv_filename = f"{file_prefix}_{task_id}_kinematics.csv"
                csv_path = SESSIONS_DIR / csv_filename
                df_task = pd.DataFrame(records)
                df_task.to_csv(csv_path, index=False)
                task["csv_file"] = csv_filename

            # 2. Save Task PNG render if provided
            png_b64 = task.get("png_base64")
            if png_b64:
                try:
                    if "," in png_b64:
                        png_b64 = png_b64.split(",", 1)[1]
                    png_bytes = base64.b64decode(png_b64)
                    png_filename = f"{file_prefix}_{task_id}_render.png"
                    png_path = SESSIONS_DIR / png_filename
                    with open(png_path, "wb") as f_png:
                        f_png.write(png_bytes)
                    task["png_file"] = png_filename
                except Exception as pe:
                    print(f"Failed to decode task image: {pe}")

        # Save Master Session JSON
        json_filename = f"{file_prefix}_session.json"
        json_path = SESSIONS_DIR / json_filename
        session["server_received_iso"] = timestamp_iso
        session["server_received_epoch"] = epoch_ms
        session["student_id"] = student_id

        # Atomic JSON write via temp file
        temp_path = SESSIONS_DIR / f"temp_{json_filename}"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        temp_path.replace(json_path)

        # Append to Master index.csv
        total_duration_sec = round(total_duration_ms / 1000.0, 1)
        row = [
            timestamp_iso,
            epoch_ms,
            student_id,
            grade,
            section,
            roll_number,
            session.get("dominant_hand", "R"),
            session.get("device_id", "Universal-Stylus"),
            session.get("operator", "Default"),
            ";".join(completed_task_ids),
            total_samples,
            total_strokes,
            total_duration_sec,
            json_filename,
            session.get("notes", "")
        ]

        with open(INDEX_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        next_roll = int(roll_number) + 1
        return {
            "status": "success",
            "message": f"Saved session for {student_id}",
            "student_id": student_id,
            "next_roll": next_roll,
            "json_file": json_filename,
            "samples_recorded": total_samples,
            "tasks_count": len(completed_task_ids)
        }

    except Exception as e:
        print(f"Error in save_session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Universal Stylus Data Collector</h1><p>index.html not found</p>")


@app.get("/dash")
def serve_dashboard():
    dash_path = STATIC_DIR / "dash.html"
    if dash_path.exists():
        return FileResponse(dash_path)
    return HTMLResponse("<h1>Stylus Dashboard</h1><p>dash.html not found</p>")


# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    print("=" * 65)
    print("🚀 Universal Stylus Data Collection Server Starting...")
    print("📱 Client App:     http://localhost:8000")
    print("📊 Dashboard:      http://localhost:8000/dash")
    print("📁 Data Directory: " + str(DATA_DIR))
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=8000)

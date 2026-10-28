"""
Dysgraphia Kinematics Harmonization Studio
Interactive Flask Web Application and API.
Estimates the 13 canonical kinematic features on the DIGITAL (stylus tablet) scale
from any handwriting image (digital screenshot or real paper notebook photo)
with 95% conformal prediction intervals, confidence tiers (A-D), and normative rank percentiles.
"""

import os
import sys
import json
import uuid
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.branch_b.visualizer import harmonize_and_visualize, get_student_ground_truth
from src.branch_b.harmonization_pipeline import FEATURE_METADATA

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preset sample definitions for FG_G4_A_Roll01
SAMPLE_STUDENTS = {
    "screenshot_fg_g4": {
        "id": "screenshot_fg_g4",
        "name": "FG_G4_A_Roll01 (Digital Stylus Screenshot)",
        "student_id": "FG_G4_A_Roll01",
        "script": "S1",
        "grade": 4,
        "type": "Digital Screenshot",
        "file_path": os.path.join(PROJECT_ROOT, "zmapped_output", "students", "FG_G4_A_Roll01", "stylus", "G4_A_Roll01__1791186296178_S1_render.png"),
        "description": "Native tablet stylus rendering (Task S1 Hindi prompt) with known hardware ground truth."
    },
    "paper_fg_g4": {
        "id": "paper_fg_g4",
        "name": "FG_G4_A_Roll01 (Real Paper Notebook Photo)",
        "student_id": "FG_G4_A_Roll01",
        "script": "S1",
        "grade": 4,
        "type": "Paper Notebook Photo",
        "file_path": os.path.join(PROJECT_ROOT, "zmapped_output", "students", "FG_G4_A_Roll01", "handwritten", "DocScanner 06-Oct-2026 10-11 AM_1.jpeg"),
        "description": "High-res smartphone photo of ruled paper notebook writing from the matched child."
    }
}


@app.route("/")
def index():
    """Serves the main studio dashboard."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy", "service": "Kinematics Harmonization Studio"})


@app.route("/api/samples")
def list_samples():
    """Returns list of available sample benchmarks."""
    samples_list = []
    for s_id, s_info in SAMPLE_STUDENTS.items():
        exists = os.path.isfile(s_info["file_path"])
        samples_list.append({
            "id": s_id,
            "name": s_info["name"],
            "student_id": s_info["student_id"],
            "script": s_info["script"],
            "grade": s_info["grade"],
            "type": s_info["type"],
            "description": s_info["description"],
            "available": exists
        })
    return jsonify({"samples": samples_list})


@app.route("/api/sample_image/<sample_id>")
def get_sample_image(sample_id):
    """Serves the raw image for a preset sample."""
    sample = SAMPLE_STUDENTS.get(sample_id)
    if not sample or not os.path.isfile(sample["file_path"]):
        return jsonify({"error": "Sample image not found"}), 404
    return send_file(sample["file_path"])


@app.route("/api/ground_truth/<student_id>")
def ground_truth(student_id):
    """Fetches true hardware stylus ground truth for a student."""
    script = request.args.get("script", "S1")
    gt = get_student_ground_truth(student_id, script=script)
    if gt is None:
        return jsonify({"error": f"No ground truth found for student {student_id}"}), 404
    return jsonify(gt)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main API endpoint:
    Accepts an uploaded image file or a sample_id, runs physical extraction and
    calibrated kinematic harmonization, and returns feature estimates and 4 visual panels.
    """
    try:
        temp_path = None
        script = "S1"
        grade = 4
        student_id = None

        if request.is_json:
            data = request.get_json()
            sample_id = data.get("sample_id")
            script = data.get("script", "S1")
            grade = int(data.get("grade", 4))
            student_id = data.get("student_id")

            if sample_id in SAMPLE_STUDENTS:
                sample = SAMPLE_STUDENTS[sample_id]
                image_source = sample["file_path"]
                if not student_id:
                    student_id = sample["student_id"]
            else:
                return jsonify({"status": "error", "message": f"Invalid sample_id: {sample_id}"}), 400
        else:
            # Multipart form upload
            script = request.form.get("script", "S1")
            grade = int(request.form.get("grade", 4))
            student_id = request.form.get("student_id") or None
            sample_id = request.form.get("sample_id")

            if sample_id and sample_id in SAMPLE_STUDENTS:
                sample = SAMPLE_STUDENTS[sample_id]
                image_source = sample["file_path"]
                if not student_id:
                    student_id = sample["student_id"]
            elif "image" in request.files:
                file = request.files["image"]
                if file.filename == "":
                    return jsonify({"status": "error", "message": "No selected file"}), 400

                ext = os.path.splitext(secure_filename(file.filename))[1]
                if not ext:
                    ext = ".png"
                temp_filename = f"upload_{uuid.uuid4().hex[:12]}{ext}"
                temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
                file.save(temp_path)
                image_source = temp_path
            else:
                return jsonify({"status": "error", "message": "No image uploaded or sample specified"}), 400

        # Execute complete physical extraction and calibrated prediction
        result = harmonize_and_visualize(
            image_input=image_source,
            script=script,
            grade=grade,
            student_id=student_id
        )

        # Clean up temporary upload file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Dysgraphia Kinematics Harmonization Studio on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


"""
Automated Integration and Smoke Test for Dysgraphia Kinematics Web Studio.
Tests Flask routes, preset samples, live file upload, and Ground Truth matching.
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app_kinematics_demo import app, SAMPLE_STUDENTS
from src.branch_b.harmonization_pipeline import FEATURE_METADATA

CANONICAL_FEATURES = list(FEATURE_METADATA.keys())


def test_flask_app():
    print("=" * 70)
    print("STARTING FLASK WEB APP INTEGRATION & SMOKE TEST")
    print("=" * 70)

    client = app.test_client()

    # 1. Test GET /
    res_index = client.get("/")
    assert res_index.status_code == 200, f"Expected 200, got {res_index.status_code}"
    assert b"Dysgraphia Kinematics Harmonization Studio" in res_index.data
    print("[PASS] GET / -> 200 OK (Studio Dashboard HTML served)")

    # 2. Test GET /api/health
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json.get("status") == "healthy"
    print("[PASS] GET /api/health -> 200 OK (Service healthy)")

    # 3. Test GET /api/samples
    res_samples = client.get("/api/samples")
    assert res_samples.status_code == 200
    samples = res_samples.json.get("samples", [])
    assert len(samples) >= 2
    sample_ids = [s["id"] for s in samples]
    assert "screenshot_fg_g4" in sample_ids and "paper_fg_g4" in sample_ids
    print(f"[PASS] GET /api/samples -> 200 OK ({len(samples)} preset samples found)")

    # 4. Test POST /api/analyze with preset digital screenshot
    print("\n--- Testing POST /api/analyze (Digital Stylus Screenshot: FG_G4_A_Roll01) ---")
    res_scr = client.post("/api/analyze", json={
        "sample_id": "screenshot_fg_g4",
        "script": "S1",
        "grade": 4,
        "student_id": "FG_G4_A_Roll01"
    })
    assert res_scr.status_code == 200, f"Error: {res_scr.data}"
    data_scr = res_scr.json
    assert data_scr["status"] == "success"
    assert data_scr["metadata"]["document_type"] == "screenshot"
    assert len(data_scr["features"]) == 13
    assert all(fn in data_scr["features"] for fn in CANONICAL_FEATURES)
    assert all(k in data_scr["visualizations"] for k in ["panel1", "panel2", "panel3", "panel4"])
    assert data_scr["ground_truth"] is not None
    assert len(data_scr["ground_truth"]["features"]) == 13
    print(f"[PASS] Digital Screenshot analyzed successfully:")
    print(f"       Detected H_med: {data_scr['metadata']['h_med_px']} px")
    print(f"       Merged strokes: {data_scr['metadata']['raw_strokes_detected']}")
    print(f"       GT Mean Velocity: {data_scr['ground_truth']['features']['kin_mean_velocity']['value']} vs Est: {data_scr['features']['kin_mean_velocity']['estimate']}")

    # 5. Test POST /api/analyze with preset real paper notebook photo
    print("\n--- Testing POST /api/analyze (Real Paper Notebook Photo: FG_G4_A_Roll01) ---")
    res_paper = client.post("/api/analyze", json={
        "sample_id": "paper_fg_g4",
        "script": "S1",
        "grade": 4,
        "student_id": "FG_G4_A_Roll01"
    })
    assert res_paper.status_code == 200, f"Error: {res_paper.data}"
    data_paper = res_paper.json
    assert data_paper["status"] == "success"
    assert data_paper["metadata"]["document_type"] == "paper_photo"
    assert len(data_paper["features"]) == 13
    assert all(fn in data_paper["features"] for fn in CANONICAL_FEATURES)
    assert all(k in data_paper["visualizations"] for k in ["panel1", "panel2", "panel3", "panel4"])

    # Verify zero NaNs and valid conformal intervals
    for fn, f_entry in data_paper["features"].items():
        est = f_entry["estimate"]
        low, high = f_entry["prediction_interval_95"]
        assert np.isfinite(est), f"{fn} estimate is not finite: {est}"
        assert np.isfinite(low) and np.isfinite(high), f"{fn} interval not finite: [{low}, {high}]"
        assert low <= est + 1e-4 and est - 1e-4 <= high, f"{fn} interval violation: {est} not in [{low}, {high}]"

    print(f"[PASS] Real Paper Photo analyzed successfully:")
    print(f"       Detected H_med: {data_paper['metadata']['h_med_px']} px")
    print(f"       Full page strokes: {data_paper['metadata']['raw_strokes_detected']}")
    print(f"       Zero NaNs, all 95% conformal intervals validated.")

    # 6. Test multipart file upload simulation
    print("\n--- Testing POST /api/analyze (Multipart Form File Upload) ---")
    img_path = SAMPLE_STUDENTS["screenshot_fg_g4"]["file_path"]
    with open(img_path, "rb") as f_img:
        res_upload = client.post(
            "/api/analyze",
            data={
                "image": (f_img, "custom_handwriting.png"),
                "script": "S1",
                "grade": "4",
                "student_id": "FG_G4_A_Roll01"
            },
            content_type="multipart/form-data"
        )
    assert res_upload.status_code == 200, f"Upload error: {res_upload.data}"
    data_upload = res_upload.json
    assert data_upload["status"] == "success"
    assert len(data_upload["features"]) == 13
    print("[PASS] Multipart file upload analyzed successfully.")

    print("\n" + "=" * 70)
    print("ALL WEB APP API & PIPELINE INTEGRATION TESTS PASSED (100% SUCCESS)")
    print("=" * 70)


if __name__ == "__main__":
    test_flask_app()


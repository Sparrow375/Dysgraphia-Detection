"""
Unit and Integration Test for Step 5 Harmonization Pipeline
Tests harmonize_handwriting_image on both a tablet screenshot and a real handwritten paper photo.
"""

import os
import sys
import json
import numpy as np

# Ensure root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.branch_b.harmonization_pipeline import harmonize_handwriting_image, FEATURE_METADATA

def run_tests():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zmapped_output", "students")
    student_ids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Loaded student data directory. Found {len(student_ids)} student directories.")
    
    # 1. Locate a screenshot
    screenshot_path = None
    paper_photo_path = None
    
    for sid in student_ids:
        s_dir = os.path.join(data_dir, sid)
        # Check stylus render
        stylus_dir = os.path.join(s_dir, "stylus")
        if screenshot_path is None and os.path.exists(stylus_dir):
            renders = [f for f in os.listdir(stylus_dir) if f.endswith("_render.png")]
            if renders:
                screenshot_path = os.path.join(stylus_dir, renders[0])
                
        # Check paper photo
        hw_dir = os.path.join(s_dir, "handwritten")
        if paper_photo_path is None and os.path.exists(hw_dir):
            jpegs = [f for f in os.listdir(hw_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            if jpegs:
                paper_photo_path = os.path.join(hw_dir, jpegs[0])
                
        if screenshot_path and paper_photo_path:
            break
            
    print(f"\n[Test 1] Testing Tablet Screenshot Harmonization:")
    print(f"Target file: {screenshot_path}")
    assert screenshot_path is not None, "Could not find a sample screenshot."
    
    res_screenshot = harmonize_handwriting_image(screenshot_path, script="S1", grade=4)
    assert res_screenshot["status"] == "success"
    assert "metadata" in res_screenshot
    assert "features" in res_screenshot
    assert len(res_screenshot["features"]) == 13
    
    for feat_name, meta in FEATURE_METADATA.items():
        assert feat_name in res_screenshot["features"], f"Missing {feat_name}"
        f_dict = res_screenshot["features"][feat_name]
        assert "estimate" in f_dict
        assert "prediction_interval_95" in f_dict
        assert len(f_dict["prediction_interval_95"]) == 2
        assert f_dict["prediction_interval_95"][0] <= f_dict["prediction_interval_95"][1]
        assert "confidence_tier" in f_dict
        assert "rank_percentile" in f_dict
        assert 0.0 <= f_dict["rank_percentile"] <= 100.0
        assert "units" in f_dict
        assert "description" in f_dict
        
    print("Screenshot harmonization verified successfully!")
    print("\nSample Output for kin_mean_velocity (Screenshot):")
    print(json.dumps(res_screenshot["features"]["kin_mean_velocity"], indent=2))
    
    print(f"\n[Test 2] Testing Real Handwritten Paper Photo Harmonization:")
    print(f"Target file: {paper_photo_path}")
    assert paper_photo_path is not None, "Could not find a sample paper photo."
    
    res_paper = harmonize_handwriting_image(paper_photo_path, script="S1", grade=5)
    assert res_paper["status"] == "success"
    assert "metadata" in res_paper
    assert "features" in res_paper
    assert len(res_paper["features"]) == 13
    
    for feat_name, meta in FEATURE_METADATA.items():
        assert feat_name in res_paper["features"], f"Missing {feat_name}"
        f_dict = res_paper["features"][feat_name]
        assert f_dict["prediction_interval_95"][0] <= f_dict["prediction_interval_95"][1]
        assert 0.0 <= f_dict["rank_percentile"] <= 100.0
        
    print("Paper photo harmonization verified successfully!")
    print("\nSample Output for kin_spatial_roughness_4_8hz (Paper Photo):")
    print(json.dumps(res_paper["features"]["kin_spatial_roughness_4_8hz"], indent=2))
    
    # 3. Test In-Memory Numpy Array Input
    print("\n[Test 3] Testing In-Memory Numpy Array Input:")
    fake_img = np.full((300, 400), 255, dtype=np.uint8)
    # Draw simple ink strokes
    fake_img[100:104, 50:250] = 30
    fake_img[100:200, 150:154] = 30
    fake_img[196:200, 50:250] = 30
    res_arr = harmonize_handwriting_image(fake_img, script="S2", grade=6)
    assert res_arr["status"] == "success"
    assert len(res_arr["features"]) == 13
    print("In-memory array input verified successfully!")
    
    # Verify complete JSON serializability
    json_str_1 = json.dumps(res_screenshot, indent=2)
    json_str_2 = json.dumps(res_paper, indent=2)
    assert len(json_str_1) > 0 and len(json_str_2) > 0
    print("\nAll JSON payloads verified 100% valid and serializable!")
    print("\nALL STEP 5 TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_tests()

import os
import sys
import pickle
import cv2
import numpy as np

PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_handwriting_image
from src.bhk_features import extract_bhk_features

with open("model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

img = cv2.imread("scraped_candidates/images/ENG_CAND_001.jpg")
print("Image shape:", img.shape)

try:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    binary_mask, preproc_vis = preprocess_handwriting_image(rgb)
    print("Preprocess done. Mask shape:", binary_mask.shape)
    feat_dict, feat_vector = extract_bhk_features(binary_mask)
    print("Features extracted:", len(feat_dict))
    
    scaler = bundle["scaler"]
    ensemble = bundle["ensemble_model"]
    X_scaled = scaler.transform(feat_vector.reshape(1, -1))
    proba = ensemble.predict_proba(X_scaled)[0, 1]
    print(f"🎉 PREDICTION SUCCESS! Potential Dysgraphia Risk: {proba * 100:.1f}%")
    print(f"Letter size CoV: {feat_dict.get('letter_size_cv', 0):.3f}")
    print(f"Baseline drift slope: {feat_dict.get('baseline_drift_slope', 0):.3f}")
except Exception as e:
    import traceback
    traceback.print_exc()

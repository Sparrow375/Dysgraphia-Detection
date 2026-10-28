"""
train_full_dysgraphia_dataset.py
================================
Trains on the ENTIRE dataset (all 249 images: 135 Low Potential + 114 Potential Dysgraphia)
using BHK computational features adapted for:
1. Real-world photo crops (variable aspect ratios, heights 50-600px).
2. Illumination compensation for camera lighting.
3. Rigorous cross-validation with probability calibration (Platt scaling / Isotonic regression)
   so predictions are realistic, calibrated risk scores (e.g. 68% or 42%), NEVER arbitrary 100%.
"""

import sys
import os
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from bhk_features.config import BHKConfig
from bhk_features.pipeline import process_image

DATASET_ROOT = Path(r"C:\Users\Laxmi Srujani\Downloads\dysgraphia\Dysgraphia-Detection\DATASET DYSGRAPHIA HANDWRITING")
LOW_DIR = DATASET_ROOT / "Low Potential Dysgraphia"
POT_DIR = DATASET_ROOT / "Potential Dysgraphia"
OUT_DIR = CURRENT_DIR / "full_dysgraphia_training_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_FEATURES = [
    "writing_size", "non_aligned_left_margin", "skewed_alignment",
    "insufficient_word_spacing", "sharp_angles", "broken_links", "collisions",
    "irregular_letter_size", "inconsistent_height_extensions",
    "traced_corrected_letters", "unstable_trace"
]

def extract_features_for_all_images():
    csv_cache = OUT_DIR / "full_dataset_bhk_features.csv"
    if csv_cache.exists():
        print(f"Loading cached full dataset features from {csv_cache}...")
        df = pd.read_csv(csv_cache)
        print(f"Loaded {len(df)} samples ({sum(df['label']==1)} Potential, {sum(df['label']==0)} Low Potential)")
        return df

    # Configuration tuned for real-world handwritten snippets & photos
    cfg = BHKConfig()
    cfg.min_image_side_px = 40        # Allow 40px+ lines/crops
    cfg.min_words = 1                 # Allow single line snippets
    cfg.min_lines = 1
    cfg.min_cc_area_px = 6
    cfg.save_debug_images = False

    low_images = sorted([p for p in LOW_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    pot_images = sorted([p for p in POT_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    all_targets = [(p, 0, "Low_Potential") for p in low_images] + [(p, 1, "Potential_Dysgraphia") for p in pot_images]
    print(f"Extracting BHK features across all {len(all_targets)} images in dataset...")

    dataset = []
    t_start = time.time()

    for idx, (img_path, label_val, label_str) in enumerate(all_targets, 1):
        if idx % 25 == 0 or idx == 1:
            print(f"  [{idx:3d}/{len(all_targets)}] Processing ({time.time()-t_start:.1f}s elapsed)...", flush=True)

        rec = process_image(img_path, cfg, OUT_DIR / "temp")
        if rec.get("status") != "ok":
            continue

        feats = rec.get("features", {})
        row = {
            "image": img_path.name,
            "label": label_val,
            "class_name": label_str,
            "ref_h_px": rec.get("ref_h_px", np.nan),
            "n_lines": rec.get("n_lines", 0),
            "n_words": rec.get("n_words", 0),
            "n_components": rec.get("n_components", 0),
            "text_skew_deg": rec.get("text_skew_deg", np.nan),
        }

        # 11 BHK scores
        for name in SCORE_FEATURES:
            f_data = feats.get(name, {})
            row[f"score_{name}"] = f_data.get("score") if f_data.get("score_status") == "scored" else np.nan

        # Raw physical measurements
        row["raw_breaks_per_word"] = feats.get("broken_links", {}).get("raw", {}).get("breaks_per_word", np.nan)
        row["raw_sharp_per_10_refh"] = feats.get("sharp_angles", {}).get("raw", {}).get("sharp_per_10_refh", np.nan)
        row["raw_collisions_per_comp"] = feats.get("collisions", {}).get("raw", {}).get("collisions_per_main_comp", np.nan)
        row["raw_mean_iqr_size"] = feats.get("irregular_letter_size", {}).get("raw", {}).get("mean_line_iqr_over_median", np.nan)
        row["raw_alignment_angle"] = feats.get("skewed_alignment", {}).get("raw", {}).get("mean_abs_rel_angle_deg", np.nan)
        row["raw_margin_std"] = feats.get("non_aligned_left_margin", {}).get("raw", {}).get("std_norm", np.nan)
        row["raw_width_cv"] = feats.get("unstable_trace", {}).get("raw", {}).get("width_cv", np.nan)
        row["raw_thick_blobs_ratio"] = feats.get("traced_corrected_letters", {}).get("raw", {}).get("thick_blobs_per_main_comp", np.nan)

        dataset.append(row)

    df = pd.DataFrame(dataset)
    df.to_csv(csv_cache, index=False)
    print(f"\nExtraction complete! Successfully processed {len(df)}/{len(all_targets)} images ({time.time()-t_start:.1f}s).")
    print(f"Saved dataset matrix to: {csv_cache}")
    return df

def train_and_evaluate_calibrated_models(df):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
    import joblib

    score_cols = [c for c in df.columns if c.startswith("score_")]
    raw_cols = [c for c in df.columns if c.startswith("raw_")]
    meta_cols = ["ref_h_px", "n_lines", "n_words", "n_components", "text_skew_deg"]
    feature_cols = score_cols + raw_cols + meta_cols

    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    print("\n" + "="*80)
    print("  TRAINING CALIBRATED DYSGRAPHIA CLASSIFIERS (FULL DATASET)")
    print("="*80)
    print(f"Total Usable Dataset : {len(df)} images")
    print(f"Potential Dysgraphia : {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    print(f"Low Potential (Norm) : {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
    print(f"Features in Vector   : {len(feature_cols)} BHK biomarkers & physical proxies")

    # Base pipelines
    pipelines = {
        "Random Forest (Calibrated)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42))
        ]),
        "Gradient Boosting (Calibrated)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42))
        ]),
        "Logistic Regression (L2)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", random_state=42))
        ]),
        "Support Vector Machine": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("\n5-Fold Stratified Cross-Validation Evaluation:")
    print(f"{'Classifier Model':<32} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Brier Score':>12}")
    print("-" * 100)

    best_f1 = 0
    best_name = None
    best_pipe = None

    for name, pipe in pipelines.items():
        # Calibrate probabilities using Platt Scaling (sigmoid) to eliminate false 100% certainty
        calibrated_pipe = CalibratedClassifierCV(pipe, cv=cv, method="sigmoid")
        
        # Out-of-fold predictions
        y_pred_proba = cross_val_predict(calibrated_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        brier = brier_score_loss(y, y_pred_proba) # Lower is better calibrated

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "ROC_AUC": auc, "Brier_Score": brier
        })

        print(f"{name:<32} {acc:>10.3f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {auc:>10.3f} {brier:>12.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipe = pipe

    # Save results table
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "full_dataset_model_comparison.csv", index=False)

    # Train and fit the final calibrated model on all data
    final_model = CalibratedClassifierCV(best_pipe, cv=5, method="sigmoid")
    final_model.fit(X, y)
    joblib.dump(final_model, OUT_DIR / "final_calibrated_dysgraphia_model.joblib")

    with open(OUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\n" + "="*80)
    print(f" Best Calibrated Model: {best_name}")
    print(f" Final F1-Score: {best_f1:.3f} | Accuracy: {res_df.loc[res_df['Model']==best_name, 'Accuracy'].values[0]:.3f}")
    print(f" Saved Production Artifact to: {OUT_DIR / 'final_calibrated_dysgraphia_model.joblib'}")
    print("="*80)

    # Feature Importance ranking
    rf = pipelines["Random Forest (Calibrated)"].named_steps["clf"]
    imp_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", rf)
    ])
    imp_pipe.fit(X, y)
    importances = imp_pipe.named_steps["clf"].feature_importances_
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    feat_df.to_csv(OUT_DIR / "feature_importance_full_dataset.csv", index=False)

    print("\nTop 10 Most Predictive BHK Features Across the Entire Dataset:")
    for rank, (_, row) in enumerate(feat_df.head(10).iterrows(), 1):
        print(f"  {rank:2d}. {row['feature']:<35} : {row['importance']:.4f}")

    return res_df, feat_df

if __name__ == "__main__":
    df = extract_features_for_all_images()
    train_and_evaluate_calibrated_models(df)


"""
dysgraphia_detector.py
======================
Detects Dysgraphia from handwriting images using BHK features.
Two modalities:
1. Supervised Learning on ground-truth labeled dataset:
   - "Potential Dysgraphia" (Class 1)
   - "Low Potential Dysgraphia" (Class 0)
2. Clinical Threshold Screening based on BHK Cutoff / Composite Index.
"""

import sys
import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure bhk_features is importable
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from bhk_features.config import BHKConfig
from bhk_features.pipeline import process_image

DATASET_ROOT = Path(r"C:\Users\Laxmi Srujani\Downloads\dysgraphia\Dysgraphia-Detection\DATASET DYSGRAPHIA HANDWRITING")
LOW_DIR = DATASET_ROOT / "Low Potential Dysgraphia"
POT_DIR = DATASET_ROOT / "Potential Dysgraphia"
OUT_DIR = CURRENT_DIR / "dysgraphia_model_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_FEATURES = [
    "writing_size", "non_aligned_left_margin", "skewed_alignment",
    "insufficient_word_spacing", "sharp_angles", "broken_links", "collisions",
    "irregular_letter_size", "inconsistent_height_extensions",
    "traced_corrected_letters", "unstable_trace"
]

def extract_features_for_dataset(sample_limit=None):
    """Extract BHK features for both classes."""
    cache_file = OUT_DIR / "bhk_dysgraphia_dataset.csv"
    if cache_file.exists():
        print(f"Loading cached features from {cache_file}...")
        return pd.read_csv(cache_file)

    cfg = BHKConfig()
    cfg.save_debug_images = False

    low_images = sorted([p for p in LOW_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    pot_images = sorted([p for p in POT_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    if sample_limit:
        low_images = low_images[:sample_limit]
        pot_images = pot_images[:sample_limit]

    print(f"Processing {len(low_images)} Low Potential (0) and {len(pot_images)} Potential Dysgraphia (1) images...")

    dataset = []
    
    def process_list(imgs, label_val, label_str):
        for idx, img_path in enumerate(imgs, 1):
            print(f"  [{label_str} {idx}/{len(imgs)}] {img_path.name} ... ", end="", flush=True)
            t0 = time.time()
            rec = process_image(img_path, cfg, OUT_DIR / "temp")
            if rec.get("status") != "ok":
                print(f"FAIL ({rec.get('reason')})")
                continue
            
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
            feats = rec.get("features", {})
            for name in SCORE_FEATURES:
                f_data = feats.get(name, {})
                row[f"score_{name}"] = f_data.get("score") if f_data.get("score_status") == "scored" else np.nan

            # Extra raw features
            row["raw_breaks_per_word"] = feats.get("broken_links", {}).get("raw", {}).get("breaks_per_word", np.nan)
            row["raw_sharp_per_10_refh"] = feats.get("sharp_angles", {}).get("raw", {}).get("sharp_per_10_refh", np.nan)
            row["raw_collisions_per_comp"] = feats.get("collisions", {}).get("raw", {}).get("collisions_per_main_comp", np.nan)
            row["raw_mean_iqr_size"] = feats.get("irregular_letter_size", {}).get("raw", {}).get("mean_line_iqr_over_median", np.nan)
            row["raw_alignment_angle"] = feats.get("skewed_alignment", {}).get("raw", {}).get("mean_abs_rel_angle_deg", np.nan)
            row["raw_margin_std"] = feats.get("non_aligned_left_margin", {}).get("raw", {}).get("std_norm", np.nan)
            row["raw_width_cv"] = feats.get("unstable_trace", {}).get("raw", {}).get("width_cv", np.nan)

            dataset.append(row)
            print(f"OK ({time.time()-t0:.1f}s)")

    process_list(low_images, 0, "Low_Potential")
    process_list(pot_images, 1, "Potential_Dysgraphia")

    df = pd.DataFrame(dataset)
    df.to_csv(cache_file, index=False)
    print(f"\nSaved extracted features to {cache_file} (Total: {len(df)} images)")
    return df

def train_dysgraphia_classifiers(df):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    score_cols = [c for c in df.columns if c.startswith("score_")]
    raw_cols = [c for c in df.columns if c.startswith("raw_")]
    meta_cols = ["ref_h_px", "n_lines", "n_words", "n_components", "text_skew_deg"]
    
    feature_cols = score_cols + raw_cols + meta_cols
    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    print("\n" + "="*70)
    print("  TRAINING DYSGRAPHIA CLASSIFICATION MODELS (BHK FEATURES)")
    print("="*70)
    print(f"Dataset Size: {len(df)} samples | Positive (Dysgraphic): {sum(y==1)} | Negative: {sum(y==0)}")
    print(f"Features Used: {len(feature_cols)} (11 BHK scores + 7 kinematic/geometric proxies + 5 structural)")

    classifiers = {
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42))
        ]),
        "Support Vector Machine (RBF)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.5, probability=True, random_state=42))
        ]),
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = []
    print("\n5-Fold Stratified Cross-Validation Results:")
    print(f"{'Classifier':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 85)

    best_f1 = 0
    best_model_name = None
    best_pipeline = None

    for name, pipeline in classifiers.items():
        cv_res = cross_validate(pipeline, X, y, cv=cv, scoring=metrics)
        acc = cv_res["test_accuracy"].mean()
        prec = cv_res["test_precision"].mean()
        rec = cv_res["test_recall"].mean()
        f1 = cv_res["test_f1"].mean()
        auc = cv_res["test_roc_auc"].mean()

        results.append({
            "Classifier": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": auc
        })

        print(f"{name:<30} {acc:>10.3f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {auc:>10.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "dysgraphia_classification_results.csv", index=False)

    # Train final best model on all data
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, OUT_DIR / "dysgraphia_bhk_detector.joblib")
    
    # Save feature metadata
    with open(OUT_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nBest Model: {best_model_name} (F1 = {best_f1:.3f})")
    print(f"Saved model artifact to: {OUT_DIR / 'dysgraphia_bhk_detector.joblib'}")

    # Feature Importance analysis (using Random Forest)
    rf_pipe = classifiers["Random Forest"]
    rf_pipe.fit(X, y)
    importances = rf_pipe.named_steps["clf"].feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    feat_imp_df.to_csv(OUT_DIR / "dysgraphia_feature_importances.csv", index=False)

    print("\nTop 8 Discriminative BHK Features for Dysgraphia Detection:")
    for _, r in feat_imp_df.head(8).iterrows():
        print(f"  * {r['Feature']:<35} : {r['Importance']:.4f}")

    return res_df, feat_imp_df

if __name__ == "__main__":
    # If not previously extracted, let's run on a representative subset of 40 images (20 Low, 20 Potential)
    # or the full set if needed.
    df = extract_features_for_dataset(sample_limit=25)
    train_dysgraphia_classifiers(df)


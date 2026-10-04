"""
Dysgraphia Detection - Multi-Dataset Training & Benchmarking Suite (v2.2 Hybrid)
Evaluates and benchmarks models across:
1. DATASET DYSGRAPHIA HANDWRITING (Malay, 249 samples)
2. reconstructed_dataset (Slovak, Drotar & Dobes 2020 / Kunhoth et al. base paper, 120 subjects)
3. scraped_candidates (English in-the-wild, 113 samples)

Performs:
- Extraction of both Cursive-Aware Multi-Baseline BHK Geometric Features (13-D / 26-D)
  and Deep Visual Stroke & Texture Features (16-D)
- 3-Way Comparative Benchmark:
    Model A: Deep Visual Stroke Features alone (16-D, Base Paper CNN philosophy)
    Model B: Pure BHK Geometric Features (13-D, v2.1)
    Model C: Hybrid Feature Fusion (29-D Deep + Geometric, Proposed)
- Within-dataset 5-Fold Stratified Cross-Validation (with SMOTE isolation)
- Cross-dataset zero-shot transfer evaluation (Train Malay -> Test Slovak Full Page)
- Combined multi-dataset training & calibration (369 clinical subjects)
- Out-of-distribution evaluation on English in-the-wild candidates (113 samples)
- Production model bundle export (model_bundle.pkl v2.2)
"""

import os
import sys
import glob
import pickle
import warnings
from typing import Dict, Any, Tuple, List

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.preprocessing import preprocess_handwriting_image
from src.bhk_features import extract_bhk_features, FEATURE_NAMES, EXTENDED_FEATURE_NAMES
from src.deep_features import extract_deep_stroke_features, DEEP_FEATURE_NAMES

MALAY_DATASET_DIR = "DATASET DYSGRAPHIA HANDWRITING"
SLOVAK_FULLPAGE_DIR = "reconstructed_dataset/full_page"
SLOVAK_TASK8_DIR = "reconstructed_dataset/by_task/task_8_sentence"
SCRAPED_DIR = "scraped_candidates"
BUNDLE_OUTPUT_PATH = "model_bundle.pkl"

# Feature definitions
BHK_CORE_FEATURES = FEATURE_NAMES.copy()  # 13-D
DEEP_STROKE_FEATURES = DEEP_FEATURE_NAMES.copy()  # 16-D
HYBRID_FEATURES = BHK_CORE_FEATURES + DEEP_STROKE_FEATURES  # 29-D


def extract_hybrid_features_from_directory(
    control_dir: str,
    dysgraphic_dir: str,
    dataset_name: str,
    cache_csv: str
) -> pd.DataFrame:
    """Extracts both BHK geometric and deep stroke texture features with caching."""
    if os.path.exists(cache_csv):
        print(f"📦 Loading cached hybrid features for [{dataset_name}] from {cache_csv}...")
        df = pd.read_csv(cache_csv)
        # Check if all hybrid features are present in cache
        if all(f in df.columns for f in HYBRID_FEATURES):
            return df
        print(f"   Cache missing some hybrid columns, regenerating...")

    print(f"🔍 Extracting hybrid (geometric + deep stroke) features for [{dataset_name}]...")
    ctrl_files = sorted(glob.glob(os.path.join(control_dir, "*.jpg")) +
                        glob.glob(os.path.join(control_dir, "*.png")))
    dys_files = sorted(glob.glob(os.path.join(dysgraphic_dir, "*.jpg")) +
                       glob.glob(os.path.join(dysgraphic_dir, "*.png")))

    rows = []
    print(f"   Processing {len(ctrl_files)} Control / LPD samples...")
    for p in ctrl_files:
        img = cv2.imread(p)
        if img is None:
            continue
        try:
            mask, gray = preprocess_handwriting_image(img)
            bhk_feats, _ = extract_bhk_features(mask)
            deep_feats, _ = extract_deep_stroke_features(mask, gray)
            
            combined_feats = {}
            combined_feats.update(bhk_feats)
            combined_feats.update(deep_feats)
            combined_feats['label'] = 0
            combined_feats['dataset'] = dataset_name
            combined_feats['filepath'] = os.path.basename(p)
            rows.append(combined_feats)
        except Exception as e:
            print(f"   ⚠️ Error on {p}: {e}")

    print(f"   Processing {len(dys_files)} Dysgraphic / PD samples...")
    for p in dys_files:
        img = cv2.imread(p)
        if img is None:
            continue
        try:
            mask, gray = preprocess_handwriting_image(img)
            bhk_feats, _ = extract_bhk_features(mask)
            deep_feats, _ = extract_deep_stroke_features(mask, gray)
            
            combined_feats = {}
            combined_feats.update(bhk_feats)
            combined_feats.update(deep_feats)
            combined_feats['label'] = 1
            combined_feats['dataset'] = dataset_name
            combined_feats['filepath'] = os.path.basename(p)
            rows.append(combined_feats)
        except Exception as e:
            print(f"   ⚠️ Error on {p}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    print(f"💾 Cached {len(df)} samples to {cache_csv}")
    return df


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """Computes clinical screening metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / max(precision + sensitivity, 1e-4)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "auc": auc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }


def evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    dataset_title: str,
    feature_set_title: str = "Features",
    threshold: float = 0.45
) -> Dict[str, Any]:
    """Runs 5-fold cross validation with SMOTE and soft-voting ensemble."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = {"rf": [], "xgb": [], "svm": [], "ensemble": []}
    oof_probs = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Apply SMOTE only to training fold
        smote = SMOTE(random_state=42 + fold)
        X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

        rf = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=2, random_state=42)
        xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42)
        svm = SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=42)

        rf.fit(X_res, y_res)
        xgb.fit(X_res, y_res)
        svm.fit(X_res, y_res)

        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb), ("svm", svm)],
            voting="soft"
        )
        ensemble.fit(X_res, y_res)

        p_rf = rf.predict_proba(X_val_scaled)[:, 1]
        p_xgb = xgb.predict_proba(X_val_scaled)[:, 1]
        p_svm = svm.predict_proba(X_val_scaled)[:, 1]
        p_ens = ensemble.predict_proba(X_val_scaled)[:, 1]

        oof_probs[val_idx] = p_ens

        fold_metrics["rf"].append(calculate_metrics(y_val, p_rf, threshold))
        fold_metrics["xgb"].append(calculate_metrics(y_val, p_xgb, threshold))
        fold_metrics["svm"].append(calculate_metrics(y_val, p_svm, threshold))
        fold_metrics["ensemble"].append(calculate_metrics(y_val, p_ens, threshold))

    ens_acc = np.mean([m["accuracy"] for m in fold_metrics["ensemble"]]) * 100
    ens_rec = np.mean([m["sensitivity"] for m in fold_metrics["ensemble"]]) * 100
    ens_spec = np.mean([m["specificity"] for m in fold_metrics["ensemble"]]) * 100
    ens_f1 = np.mean([m["f1"] for m in fold_metrics["ensemble"]]) * 100
    ens_auc = np.mean([m["auc"] for m in fold_metrics["ensemble"]])

    print(f"  [{feature_set_title:<28}] Acc: {ens_acc:5.1f}% | Rec: {ens_rec:5.1f}% | Spec: {ens_spec:5.1f}% | F1: {ens_f1:5.1f}% | AUC: {ens_auc:.3f}")
    return {
        "oof_probs": oof_probs,
        "accuracy": ens_acc,
        "recall": ens_rec,
        "specificity": ens_spec,
        "f1": ens_f1,
        "auc": ens_auc,
        "fold_metrics": fold_metrics
    }


def main():
    print("=" * 88)
    print("🚀 DYSGRAPHIA DETECTION: HYBRID DEEP & GEOMETRIC MULTI-DATASET BENCHMARK")
    print("=" * 88)

    # 1. Extract Malay Dataset
    df_malay = extract_hybrid_features_from_directory(
        control_dir=os.path.join(MALAY_DATASET_DIR, "Low Potential Dysgraphia"),
        dysgraphic_dir=os.path.join(MALAY_DATASET_DIR, "Potential Dysgraphia"),
        dataset_name="Malay_Sentences",
        cache_csv="cache_malay_hybrid_features.csv"
    )

    # 2. Extract Slovak Full Page Dataset (Base paper replication)
    df_slovak_fp = extract_hybrid_features_from_directory(
        control_dir=os.path.join(SLOVAK_FULLPAGE_DIR, "control"),
        dysgraphic_dir=os.path.join(SLOVAK_FULLPAGE_DIR, "dysgraphic"),
        dataset_name="Slovak_FullPage",
        cache_csv="cache_slovak_fullpage_hybrid_features.csv"
    )

    # 3. Extract Slovak Sentence Dataset (Task 8)
    df_slovak_task8 = extract_hybrid_features_from_directory(
        control_dir=os.path.join(SLOVAK_TASK8_DIR, "control"),
        dysgraphic_dir=os.path.join(SLOVAK_TASK8_DIR, "dysgraphic"),
        dataset_name="Slovak_Task8_Sentence",
        cache_csv="cache_slovak_task8_hybrid_features.csv"
    )

    y_malay = df_malay['label'].values.astype(int)
    y_slovak = df_slovak_fp['label'].values.astype(int)

    # --- 3-WAY COMPARATIVE BENCHMARK: DEEP vs GEOMETRIC vs HYBRID ---
    print("\n" + "=" * 88)
    print("📊 3-WAY COMPARATIVE BENCHMARK: MALAY DATASET (249 Samples, 5-Fold CV)")
    print("=" * 88)
    evaluate_cv(df_malay[DEEP_STROKE_FEATURES].values, y_malay, "Malay", "Model A: Deep Stroke (16-D)")
    evaluate_cv(df_malay[BHK_CORE_FEATURES].values, y_malay, "Malay", "Model B: Pure BHK Geom (13-D)")
    evaluate_cv(df_malay[HYBRID_FEATURES].values, y_malay, "Malay", "Model C: HYBRID FUSION (29-D)")

    print("\n" + "=" * 88)
    print("📊 3-WAY COMPARATIVE BENCHMARK: SLOVAK FULL-PAGE (120 Subjects, 5-Fold CV)")
    print("=" * 88)
    evaluate_cv(df_slovak_fp[DEEP_STROKE_FEATURES].values, y_slovak, "Slovak FP", "Model A: Deep Stroke (16-D)")
    evaluate_cv(df_slovak_fp[BHK_CORE_FEATURES].values, y_slovak, "Slovak FP", "Model B: Pure BHK Geom (13-D)")
    evaluate_cv(df_slovak_fp[HYBRID_FEATURES].values, y_slovak, "Slovak FP", "Model C: HYBRID FUSION (29-D)")

    # 4. Cross-Dataset Transfer Testing
    print("\n" + "=" * 88)
    print("🌐 CROSS-DATASET ZERO-SHOT TRANSFER EVALUATION (Train Malay -> Test Slovak FP)")
    print("=" * 88)

    for feat_name, feat_cols in [
        ("Model A: Deep Stroke (16-D)", DEEP_STROKE_FEATURES),
        ("Model B: Pure BHK Geom (13-D)", BHK_CORE_FEATURES),
        ("Model C: HYBRID FUSION (29-D)", HYBRID_FEATURES)
    ]:
        X_m = df_malay[feat_cols].values
        X_s = df_slovak_fp[feat_cols].values

        scaler_tr = StandardScaler()
        X_m_sc = scaler_tr.fit_transform(X_m)
        smote_tr = SMOTE(random_state=42)
        X_m_res, y_m_res = smote_tr.fit_resample(X_m_sc, y_malay)

        ens_tr = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)),
                ("xgb", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)),
                ("svm", SVC(kernel="rbf", C=1.5, probability=True, random_state=42))
            ],
            voting="soft"
        )
        ens_tr.fit(X_m_res, y_m_res)

        X_s_sc = scaler_tr.transform(X_s)
        p_s = ens_tr.predict_proba(X_s_sc)[:, 1]
        m = calculate_metrics(y_slovak, p_s, 0.45)
        print(f"  [{feat_name:<28}] Acc: {m['accuracy']*100:5.1f}% | Rec (PD): {m['sensitivity']*100:5.1f}% | Spec: {m['specificity']*100:5.1f}% | AUC: {m['auc']:.3f}")

    # 5. Combined Multi-Lingual Training (Malay + Slovak = 369 clinical subjects)
    print("\n" + "=" * 88)
    print("🔗 COMBINED MULTI-LINGUAL BENCHMARK (369 Clinical Subjects: Malay + Slovak)")
    print("=" * 88)
    df_combined = pd.concat([df_malay, df_slovak_fp], ignore_index=True)
    y_comb = df_combined['label'].values.astype(int)

    evaluate_cv(df_combined[DEEP_STROKE_FEATURES].values, y_comb, "Combined", "Model A: Deep Stroke (16-D)")
    evaluate_cv(df_combined[BHK_CORE_FEATURES].values, y_comb, "Combined", "Model B: Pure BHK Geom (13-D)")
    eval_comb_hybrid = evaluate_cv(df_combined[HYBRID_FEATURES].values, y_comb, "Combined", "Model C: HYBRID FUSION (29-D)")

    # 6. Fit & Export Production Multi-Lingual Bundles (Both Hybrid and Core BHK for compatibility)
    print("\n📦 Fitting Final Production Ensemble on Combined 369 Samples...")
    
    # Hybrid Model (29-D)
    scaler_hybrid = StandardScaler()
    X_comb_hyb_sc = scaler_hybrid.fit_transform(df_combined[HYBRID_FEATURES].values)
    smote_hyb = SMOTE(random_state=42)
    X_res_hyb, y_res_hyb = smote_hyb.fit_resample(X_comb_hyb_sc, y_comb)

    rf_hyb = RandomForestClassifier(n_estimators=160, max_depth=6, min_samples_leaf=2, random_state=42)
    xgb_hyb = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.07, eval_metric="logloss", random_state=42)
    svm_hyb = SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=42)

    hybrid_ensemble = VotingClassifier(
        estimators=[("rf", rf_hyb), ("xgb", xgb_hyb), ("svm", svm_hyb)],
        voting="soft"
    )
    hybrid_ensemble.fit(X_res_hyb, y_res_hyb)

    # Core BHK Model (13-D, for backward compatibility)
    scaler_bhk = StandardScaler()
    X_comb_bhk_sc = scaler_bhk.fit_transform(df_combined[BHK_CORE_FEATURES].values)
    smote_bhk = SMOTE(random_state=42)
    X_res_bhk, y_res_bhk = smote_bhk.fit_resample(X_comb_bhk_sc, y_comb)

    rf_bhk = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=2, random_state=42)
    xgb_bhk = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.07, eval_metric="logloss", random_state=42)
    svm_bhk = SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=42)

    bhk_ensemble = VotingClassifier(
        estimators=[("rf", rf_bhk), ("xgb", xgb_bhk), ("svm", svm_bhk)],
        voting="soft"
    )
    bhk_ensemble.fit(X_res_bhk, y_res_bhk)

    # Compute deployed performance of Hybrid model
    p_prod = hybrid_ensemble.predict_proba(scaler_hybrid.transform(df_combined[HYBRID_FEATURES].values))[:, 1]
    m_prod = calculate_metrics(y_comb, p_prod, 0.45)

    print(f"  Hybrid Ensemble Deployed Accuracy: {m_prod['accuracy']*100:.2f}% | Sensitivity (Recall): {m_prod['sensitivity']*100:.2f}% | AUC: {m_prod['auc']:.4f}")

    bundle = {
        "ensemble_model": hybrid_ensemble,           # Default deployed ensemble (Hybrid)
        "scaler": scaler_hybrid,                     # Hybrid scaler
        "feature_names": HYBRID_FEATURES,            # 29 hybrid features
        "hybrid_model": hybrid_ensemble,
        "hybrid_scaler": scaler_hybrid,
        "hybrid_feature_names": HYBRID_FEATURES,
        "bhk_model": bhk_ensemble,                   # Backward-compatible BHK ensemble
        "bhk_scaler": scaler_bhk,                    # BHK scaler
        "bhk_feature_names": BHK_CORE_FEATURES,      # 13 BHK features
        "extended_feature_names": EXTENDED_FEATURE_NAMES,
        "deep_feature_names": DEEP_STROKE_FEATURES,
        "optimal_threshold": 0.45,
        "metadata": {
            "version": "2.2-hybrid-deep-geometric-multilingual",
            "train_dataset": "Combined Malay (249) + Slovak Drotar Reconstructed FullPage (120)",
            "total_samples": len(y_comb),
            "features_count": len(HYBRID_FEATURES),
            "deployed_accuracy": m_prod['accuracy'],
            "deployed_recall": m_prod['sensitivity'],
            "deployed_auc": m_prod['auc']
        }
    }

    with open(BUNDLE_OUTPUT_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"✅ Successfully exported updated v2.2 hybrid production bundle to {BUNDLE_OUTPUT_PATH}!")

    # 7. Benchmark on English in-the-wild Scraped Candidates (if available)
    if os.path.exists(SCRAPED_DIR):
        print("\n" + "=" * 88)
        print("🌍 EVALUATING OUT-OF-DISTRIBUTION ON ENGLISH CANDIDATES (scraped_candidates)")
        print("=" * 88)
        scraped_images = sorted(glob.glob(os.path.join(SCRAPED_DIR, "images", "*.jpg")) +
                                glob.glob(os.path.join(SCRAPED_DIR, "images", "*.png")))
        print(f"Found {len(scraped_images)} scraped candidate images.")
        
        scraped_results = []
        for p in scraped_images:
            img = cv2.imread(p)
            if img is None:
                continue
            try:
                mask, gray = preprocess_handwriting_image(img)
                f_bhk, _ = extract_bhk_features(mask)
                f_deep, _ = extract_deep_stroke_features(mask, gray)
                
                all_feats = {}
                all_feats.update(f_bhk)
                all_feats.update(f_deep)
                
                hyb_vec = np.array([all_feats[name] for name in HYBRID_FEATURES], dtype=np.float32)
                scaled_vec = scaler_hybrid.transform(hyb_vec.reshape(1, -1))
                prob_pd = float(hybrid_ensemble.predict_proba(scaled_vec)[0, 1])
                
                scraped_results.append({
                    "candidate": os.path.basename(p),
                    "prob_pd": prob_pd,
                    "is_pd_screened": prob_pd >= 0.45,
                    "is_cursive": bool(f_bhk["is_cursive"]),
                    "cursive_fluidity": f_bhk["cursive_fluidity_index"],
                    "spatial_score": f_bhk["spatial_dysgraphia_score"],
                    "motor_score": f_bhk["motor_dysgraphia_score"],
                    "dyslexic_score": f_bhk["dyslexic_risk_score"]
                })
            except Exception as e:
                continue

        if scraped_results:
            df_scr = pd.DataFrame(scraped_results)
            print(f"• Total Evaluated Candidates: {len(df_scr)}")
            print(f"• Mean AI Screening Risk: {df_scr['prob_pd'].mean() * 100:.1f}%")
            print(f"• Detected Cursive Writing Samples: {df_scr['is_cursive'].sum()} of {len(df_scr)}")
            print(f"• Mean Cursive Fluidity: {df_scr['cursive_fluidity'].mean() * 100:.1f}%")
            print(f"• Screened Potential Dysgraphia: {(df_scr['is_pd_screened']).sum()} ({df_scr['is_pd_screened'].mean()*100:.1f}%)")
    else:
        print("\nℹ️ Scraped English candidates directory not found (available on 'data-harvesting' branch). Skipping OOD evaluation.")

    print("\n🎯 All benchmarks and training completed successfully.")


if __name__ == "__main__":
    main()

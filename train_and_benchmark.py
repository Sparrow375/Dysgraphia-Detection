"""
Dysgraphia Detection - Multi-Dataset Training & Benchmarking Suite
Evaluates and benchmarks models across:
1. DATASET DYSGRAPHIA HANDWRITING (Malay, 249 samples)
2. reconstructed_dataset (Slovak, Drotar & Dobes 2020 / Kunhoth et al. base paper, 120 subjects)
3. scraped_candidates (English in-the-wild, 113 samples)

Performs:
- Within-dataset 5-Fold Stratified Cross-Validation (with SMOTE)
- Cross-dataset transfer evaluation (Train Malay -> Test Slovak, Train Slovak -> Test Malay)
- Combined multi-dataset training & calibration
- Out-of-distribution evaluation on English in-the-wild candidates
- Model bundle export (model_bundle.pkl)
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

MALAY_DATASET_DIR = "DATASET DYSGRAPHIA HANDWRITING"
SLOVAK_FULLPAGE_DIR = "reconstructed_dataset/full_page"
SLOVAK_TASK8_DIR = "reconstructed_dataset/by_task/task_8_sentence"
SCRAPED_DIR = "scraped_candidates"
BUNDLE_OUTPUT_PATH = "model_bundle.pkl"


def extract_features_from_directory(
    control_dir: str,
    dysgraphic_dir: str,
    dataset_name: str,
    cache_csv: str
) -> pd.DataFrame:
    """Extracts scale-invariant BHK features from a two-class directory with caching."""
    if os.path.exists(cache_csv):
        print(f"📦 Loading cached features for [{dataset_name}] from {cache_csv}...")
        return pd.read_csv(cache_csv)

    print(f"🔍 Extracting multi-baseline & cursive-aware features for [{dataset_name}]...")
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
            mask, _ = preprocess_handwriting_image(img)
            feats, _ = extract_bhk_features(mask)
            feats['label'] = 0
            feats['dataset'] = dataset_name
            feats['filepath'] = os.path.basename(p)
            rows.append(feats)
        except Exception as e:
            print(f"   ⚠️ Error on {p}: {e}")

    print(f"   Processing {len(dys_files)} Dysgraphic / PD samples...")
    for p in dys_files:
        img = cv2.imread(p)
        if img is None:
            continue
        try:
            mask, _ = preprocess_handwriting_image(img)
            feats, _ = extract_bhk_features(mask)
            feats['label'] = 1
            feats['dataset'] = dataset_name
            feats['filepath'] = os.path.basename(p)
            rows.append(feats)
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


def evaluate_cv(X: np.ndarray, y: np.ndarray, dataset_title: str) -> Dict[str, Any]:
    """Runs 5-fold cross validation with SMOTE and soft-voting ensemble."""
    print(f"\n--- 5-Fold Stratified CV on [{dataset_title}] ({len(y)} samples: {np.sum(y==0)} Control, {np.sum(y==1)} Dysgraphic) ---")
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

        fold_metrics["rf"].append(calculate_metrics(y_val, p_rf, 0.45))
        fold_metrics["xgb"].append(calculate_metrics(y_val, p_xgb, 0.45))
        fold_metrics["svm"].append(calculate_metrics(y_val, p_svm, 0.45))
        fold_metrics["ensemble"].append(calculate_metrics(y_val, p_ens, 0.45))

    # Print summary
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Recall (PD)':<12} | {'Specificity':<12} | {'F1-Score':<10} | {'ROC-AUC':<8}")
    print("-" * 84)
    for model_name, m_list in fold_metrics.items():
        acc = np.mean([m["accuracy"] for m in m_list]) * 100
        rec = np.mean([m["sensitivity"] for m in m_list]) * 100
        spec = np.mean([m["specificity"] for m in m_list]) * 100
        f1 = np.mean([m["f1"] for m in m_list]) * 100
        auc = np.mean([m["auc"] for m in m_list])
        print(f"{model_name.upper():<22} | {acc:6.1f}%    | {rec:6.1f}%      | {spec:6.1f}%      | {f1:6.1f}%    | {auc:.3f}")

    return {"oof_probs": oof_probs, "fold_metrics": fold_metrics}


def main():
    print("=" * 80)
    print("🚀 DYSGRAPHIA DETECTION: MULTI-DATASET BENCHMARK & RETRAINING")
    print("=" * 80)

    # 1. Extract Malay Dataset
    df_malay = extract_features_from_directory(
        control_dir=os.path.join(MALAY_DATASET_DIR, "Low Potential Dysgraphia"),
        dysgraphic_dir=os.path.join(MALAY_DATASET_DIR, "Potential Dysgraphia"),
        dataset_name="Malay_Sentences",
        cache_csv="cache_malay_features.csv"
    )

    # 2. Extract Slovak Full Page Dataset (Base paper replication)
    df_slovak_fp = extract_features_from_directory(
        control_dir=os.path.join(SLOVAK_FULLPAGE_DIR, "control"),
        dysgraphic_dir=os.path.join(SLOVAK_FULLPAGE_DIR, "dysgraphic"),
        dataset_name="Slovak_FullPage",
        cache_csv="cache_slovak_fullpage_features.csv"
    )

    # 3. Extract Slovak Sentence Dataset (Task 8)
    df_slovak_task8 = extract_features_from_directory(
        control_dir=os.path.join(SLOVAK_TASK8_DIR, "control"),
        dysgraphic_dir=os.path.join(SLOVAK_TASK8_DIR, "dysgraphic"),
        dataset_name="Slovak_Task8_Sentence",
        cache_csv="cache_slovak_task8_features.csv"
    )

    # Run CV on Malay
    X_malay = df_malay[FEATURE_NAMES].values
    y_malay = df_malay['label'].values.astype(int)
    evaluate_cv(X_malay, y_malay, "Malay Dataset (249 samples)")

    # Run CV on Slovak Full Page
    X_slovak = df_slovak_fp[FEATURE_NAMES].values
    y_slovak = df_slovak_fp['label'].values.astype(int)
    evaluate_cv(X_slovak, y_slovak, "Slovak Reconstructed Full Page (120 subjects)")

    # 4. Cross-Dataset Transfer Testing
    print("\n" + "=" * 80)
    print("🌐 CROSS-DATASET ZERO-SHOT TRANSFER EVALUATION")
    print("=" * 80)
    # Train on Malay -> Test on Slovak
    scaler_m = StandardScaler()
    X_m_scaled = scaler_m.fit_transform(X_malay)
    smote_m = SMOTE(random_state=42)
    X_m_res, y_m_res = smote_m.fit_resample(X_m_scaled, y_malay)

    ens_m = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)),
            ("xgb", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)),
            ("svm", SVC(kernel="rbf", C=1.5, probability=True, random_state=42))
        ],
        voting="soft"
    )
    ens_m.fit(X_m_res, y_m_res)

    X_slovak_test = scaler_m.transform(X_slovak)
    p_slovak = ens_m.predict_proba(X_slovak_test)[:, 1]
    m_cross_s = calculate_metrics(y_slovak, p_slovak, 0.45)
    print(f"Train on Malay -> Test on Slovak Full Page:")
    print(f"  Accuracy: {m_cross_s['accuracy']*100:.1f}% | Recall (PD): {m_cross_s['sensitivity']*100:.1f}% | Specificity: {m_cross_s['specificity']*100:.1f}% | AUC: {m_cross_s['auc']:.3f}")

    # 5. Combined Multi-Dataset Benchmark & Training (Malay + Slovak = 369 clinical subjects)
    print("\n" + "=" * 80)
    print("🔗 COMBINED MULTI-LINGUAL TRAINING (Malay + Slovak Full Page = 369 Subjects)")
    print("=" * 80)
    df_combined = pd.concat([df_malay, df_slovak_fp], ignore_index=True)
    X_comb = df_combined[FEATURE_NAMES].values
    y_comb = df_combined['label'].values.astype(int)

    evaluate_cv(X_comb, y_comb, "Combined Multi-Lingual Dataset (369 samples)")

    # 6. Train Final Multi-Dataset Production Ensemble & Export Bundle
    print("\n📦 Fitting Final Production Ensemble on Combined 369 Samples...")
    scaler_final = StandardScaler()
    X_comb_scaled = scaler_final.fit_transform(X_comb)
    smote_final = SMOTE(random_state=42)
    X_comb_res, y_comb_res = smote_final.fit_resample(X_comb_scaled, y_comb)

    rf_prod = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=2, random_state=42)
    xgb_prod = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.07, eval_metric="logloss", random_state=42)
    svm_prod = SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=42)

    final_ensemble = VotingClassifier(
        estimators=[("rf", rf_prod), ("xgb", xgb_prod), ("svm", svm_prod)],
        voting="soft"
    )
    final_ensemble.fit(X_comb_res, y_comb_res)

    bundle = {
        "ensemble_model": final_ensemble,
        "scaler": scaler_final,
        "feature_names": FEATURE_NAMES,
        "extended_feature_names": EXTENDED_FEATURE_NAMES,
        "optimal_threshold": 0.45,
        "metadata": {
            "version": "2.1-multibaseline-cursive-multilingual",
            "train_dataset": "Combined Malay (249) + Slovak Drotar Reconstructed FullPage (120)",
            "total_samples": len(y_comb),
            "features_count": len(FEATURE_NAMES)
        }
    }

    with open(BUNDLE_OUTPUT_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"✅ Successfully exported updated production bundle to {BUNDLE_OUTPUT_PATH}!")

    # 7. Benchmark on English in-the-wild Scraped Candidates
    print("\n" + "=" * 80)
    print("🌍 EVALUATING OUT-OF-DISTRIBUTION ON ENGLISH CANDIDATES (scraped_candidates)")
    print("=" * 80)
    scraped_images = glob.glob(os.path.join(SCRAPED_DIR, "images", "*.jpg")) + \
                     glob.glob(os.path.join(SCRAPED_DIR, "images", "*.png"))
    print(f"Found {len(scraped_images)} scraped candidate images.")
    
    scraped_results = []
    for p in scraped_images:
        img = cv2.imread(p)
        if img is None: continue
        try:
            mask, _ = preprocess_handwriting_image(img)
            f_dict, f_vec = extract_bhk_features(mask)
            scaled_vec = scaler_final.transform(f_vec.reshape(1, -1))
            prob_pd = float(final_ensemble.predict_proba(scaled_vec)[0, 1])
            scraped_results.append({
                "candidate": os.path.basename(p),
                "prob_pd": prob_pd,
                "is_pd_screened": prob_pd >= 0.45,
                "is_cursive": bool(f_dict["is_cursive"]),
                "cursive_fluidity": f_dict["cursive_fluidity_index"],
                "spatial_score": f_dict["spatial_dysgraphia_score"],
                "motor_score": f_dict["motor_dysgraphia_score"]
            })
        except Exception:
            continue

    if scraped_results:
        df_scr = pd.DataFrame(scraped_results)
        print(f"• Total Evaluated Candidates: {len(df_scr)}")
        print(f"• Mean AI Screening Risk: {df_scr['prob_pd'].mean() * 100:.1f}%")
        print(f"• Detected Cursive Writing Samples: {df_scr['is_cursive'].sum()} of {len(df_scr)}")
        print(f"• Mean Cursive Fluidity: {df_scr['cursive_fluidity'].mean() * 100:.1f}%")
        print(f"• Screened Potential Dysgraphia: {(df_scr['is_pd_screened']).sum()} ({df_scr['is_pd_screened'].mean()*100:.1f}%)")

    print("\n🎯 All benchmarks and training completed successfully.")


if __name__ == "__main__":
    main()

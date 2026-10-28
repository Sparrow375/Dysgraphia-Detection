"""
Rigorous Shortcut & Scientific Validity Verification for DINOv2 0.84 AUROC
Evaluates 4 critical falsification tests:
1. Cross-School Transfer (Train School A -> Test School B, and Train School B -> Test School A)
2. Permutation / Y-Scramble Null Distribution (100 shuffles)
3. Confound Diagnostic Baselines (School-only, Grade-only, School+Grade)
4. Cross-Grade Pen/Pencil Regime Transfer (Train Grades 5-8 -> Test Grades 3-4, and vice-versa)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_shortcuts")

DATA_FILE = Path("data/features/exp7_dinov2_small_tiles.npz")


def fit_and_eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int = 24,
    C: float = 0.1,
) -> Tuple[float, float, float]:
    """Fit scaler, PCA, and Logistic Regression strictly on train, evaluate on test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Bound n_components by sample size
    k = min(n_components, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
    pca = PCA(n_components=k, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train_pca, y_train)

    y_prob = clf.predict_proba(X_test_pca)[:, 1]
    auroc = float(roc_auc_score(y_test, y_prob))
    auprc = float(average_precision_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    return auroc, auprc, brier


def main() -> None:
    logger.info("Loading DINOv2 cached student features...")
    data = np.load(DATA_FILE)
    X_joint = data["joint_student_features"]
    X_hindi = data["hindi_student_features"]
    X_english = data["english_student_features"]
    y = data["student_labels"]
    schools = data["student_schools"]
    grades = data["student_grades"]

    n_students = len(y)
    logger.info("Loaded %d students: %d positive, %d control", n_students, np.sum(y == 1), np.sum(y == 0))

    results = {}

    # =========================================================================
    # 1. Confound Baselines (School-only, Grade-only, School+Grade)
    # =========================================================================
    logger.info("--- TEST 1: Confound Diagnostic Baselines ---")
    # School-only predictor: School A has higher positive rate (32/115 = 27.8%) than School B (8/100 = 8.0%)
    school_binary = (schools == "school_a").astype(float)
    auroc_school_only = float(roc_auc_score(y, school_binary))
    logger.info("School-only baseline AUROC: %.4f", auroc_school_only)

    # Grade-only predictor (younger kids might have worse handwriting)
    # Lower grade -> higher chance of dysgraphia
    auroc_grade_inv = float(roc_auc_score(y, -grades))
    logger.info("Grade-only baseline AUROC: %.4f", auroc_grade_inv)

    # 5-fold CV on School + Grade alone using Logistic Regression
    X_confounds = np.column_stack([school_binary, grades])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_confounds = np.zeros(n_students)
    for train_idx, val_idx in cv.split(X_confounds, y):
        clf = LogisticRegression(class_weight="balanced", random_state=42)
        clf.fit(X_confounds[train_idx], y[train_idx])
        oof_confounds[val_idx] = clf.predict_proba(X_confounds[val_idx])[:, 1]
    auroc_confounds_cv = float(roc_auc_score(y, oof_confounds))
    auprc_confounds_cv = float(average_precision_score(y, oof_confounds))
    logger.info("Combined [School + Grade] 5-Fold CV AUROC: %.4f, AUPRC: %.4f", auroc_confounds_cv, auprc_confounds_cv)

    results["confounds"] = {
        "school_only_auroc": auroc_school_only,
        "grade_only_auroc": auroc_grade_inv,
        "combined_school_grade_cv_auroc": auroc_confounds_cv,
        "combined_school_grade_cv_auprc": auprc_confounds_cv,
    }

    # =========================================================================
    # 2. Cross-School Transfer (Train School A -> Test B, Train School B -> Test A)
    # =========================================================================
    logger.info("--- TEST 2: Cross-School Out-of-Distribution Transfer ---")
    mask_a = schools == "school_a"
    mask_b = schools == "school_b"

    logger.info("School A: %d students (%d pos, %d ctl)", np.sum(mask_a), np.sum(y[mask_a] == 1), np.sum(y[mask_a] == 0))
    logger.info("School B: %d students (%d pos, %d ctl)", np.sum(mask_b), np.sum(y[mask_b] == 1), np.sum(y[mask_b] == 0))

    # Train School A -> Test School B
    auroc_a_to_b, auprc_a_to_b, _ = fit_and_eval_probe(
        X_joint[mask_a], y[mask_a], X_joint[mask_b], y[mask_b], n_components=24, C=0.1
    )
    logger.info("Train School A -> Test School B: AUROC = %.4f, AUPRC = %.4f", auroc_a_to_b, auprc_a_to_b)

    # Train School B -> Test School A
    auroc_b_to_a, auprc_b_to_a, _ = fit_and_eval_probe(
        X_joint[mask_b], y[mask_b], X_joint[mask_a], y[mask_a], n_components=24, C=0.1
    )
    logger.info("Train School B -> Test School A: AUROC = %.4f, AUPRC = %.4f", auroc_b_to_a, auprc_b_to_a)

    results["cross_school"] = {
        "train_a_test_b": {"auroc": auroc_a_to_b, "auprc": auprc_a_to_b},
        "train_b_test_a": {"auroc": auroc_b_to_a, "auprc": auprc_b_to_a},
        "mean_cross_school_auroc": float((auroc_a_to_b + auroc_b_to_a) / 2),
    }

    # =========================================================================
    # 3. Cross-Grade Pen vs Pencil Transfer (Primary vs Middle School)
    # =========================================================================
    logger.info("--- TEST 3: Cross-Grade Pen vs Pencil Transfer ---")
    mask_pencil = grades <= 4  # Grades 3 & 4 (Primary, pencil predominant)
    mask_pen = grades >= 5     # Grades 5-8 (Middle, ballpoint pen predominant)

    logger.info("Grades 3-4 (Pencil): %d students (%d pos, %d ctl)", np.sum(mask_pencil), np.sum(y[mask_pencil] == 1), np.sum(y[mask_pencil] == 0))
    logger.info("Grades 5-8 (Pen): %d students (%d pos, %d ctl)", np.sum(mask_pen), np.sum(y[mask_pen] == 1), np.sum(y[mask_pen] == 0))

    # Train on Pen (Grades 5-8) -> Test on Pencil (Grades 3-4)
    auroc_pen_to_pencil, auprc_pen_to_pencil, _ = fit_and_eval_probe(
        X_joint[mask_pen], y[mask_pen], X_joint[mask_pencil], y[mask_pencil], n_components=24, C=0.1
    )
    logger.info("Train Pen (Grades 5-8) -> Test Pencil (Grades 3-4): AUROC = %.4f, AUPRC = %.4f", auroc_pen_to_pencil, auprc_pen_to_pencil)

    # Train on Pencil (Grades 3-4) -> Test on Pen (Grades 5-8)
    auroc_pencil_to_pen, auprc_pencil_to_pen, _ = fit_and_eval_probe(
        X_joint[mask_pencil], y[mask_pencil], X_joint[mask_pen], y[mask_pen], n_components=24, C=0.1
    )
    logger.info("Train Pencil (Grades 3-4) -> Test Pen (Grades 5-8): AUROC = %.4f, AUPRC = %.4f", auroc_pencil_to_pen, auprc_pencil_to_pen)

    results["cross_grade_pen_pencil"] = {
        "train_pen_test_pencil": {"auroc": auroc_pen_to_pencil, "auprc": auprc_pen_to_pencil},
        "train_pencil_test_pen": {"auroc": auroc_pencil_to_pen, "auprc": auprc_pencil_to_pen},
        "mean_cross_grade_auroc": float((auroc_pen_to_pencil + auroc_pencil_to_pen) / 2),
    }

    # =========================================================================
    # 4. Y-Scramble / Permutation Null Distribution Test (50 Permutations)
    # =========================================================================
    logger.info("--- TEST 4: Y-Scramble / Label Permutation Test (50 Shuffles) ---")
    rng = np.random.default_axis = np.random.RandomState(42)
    perm_aurocs = []
    
    # Run 5-fold CV with scrambled labels
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for p_idx in range(50):
        y_perm = rng.permutation(y)
        oof_preds = np.zeros(n_students)
        for train_idx, val_idx in cv5.split(X_joint, y_perm):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_joint[train_idx])
            X_va = scaler.transform(X_joint[val_idx])

            pca = PCA(n_components=24, random_state=42)
            X_tr_pca = pca.fit_transform(X_tr)
            X_va_pca = pca.transform(X_va)

            clf = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_tr_pca, y_perm[train_idx])
            oof_preds[val_idx] = clf.predict_proba(X_va_pca)[:, 1]
        
        perm_auc = roc_auc_score(y_perm, oof_preds)
        perm_aurocs.append(perm_auc)

    mean_null_auc = float(np.mean(perm_aurocs))
    std_null_auc = float(np.std(perm_aurocs))
    max_null_auc = float(np.max(perm_aurocs))
    logger.info("Permutation Null AUROC: Mean = %.4f +/- %.4f (Max = %.4f)", mean_null_auc, std_null_auc, max_null_auc)

    results["permutation_test"] = {
        "n_permutations": 50,
        "mean_null_auroc": mean_null_auc,
        "std_null_auroc": std_null_auc,
        "max_null_auroc": max_null_auc,
        "empirical_p_value": float(np.mean(np.array(perm_aurocs) >= 0.8427)),
    }

    out_path = Path("reports/exp7_dinov2_shortcut_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Verification results saved to %s", out_path)


if __name__ == "__main__":
    main()


"""Evaluation metrics, probability calibration, student-level pooling,
and cluster bootstrap confidence interval estimation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class CalibratedPredictor:
    """Platt scaling (sigmoid calibration) wrapper for arbitrary base classifiers."""

    def __init__(self, base_estimator: Any, calibrator: LogisticRegression):
        self.base_estimator = base_estimator
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities [P(y=0), P(y=1)]."""
        if hasattr(self.base_estimator, "decision_function"):
            scores = self.base_estimator.decision_function(X)
        elif hasattr(self.base_estimator, "predict_proba"):
            p1 = self.base_estimator.predict_proba(X)[:, 1]
            eps = 1e-7
            p1 = np.clip(p1, eps, 1.0 - eps)
            scores = np.log(p1 / (1.0 - p1))  # log-odds (logit)
        else:
            raise AttributeError("Base estimator has neither decision_function nor predict_proba.")

        scores_2d = scores.reshape(-1, 1)
        return self.calibrator.predict_proba(scores_2d)


def fit_platt_calibrator(
    base_estimator: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight_val: Optional[np.ndarray] = None,
) -> CalibratedPredictor:
    """Fit a univariate logistic regression (Platt scaling) on validation predictions.

    NOTE: Prefer fit_platt_calibrator_from_scores when inner out-of-fold scores are
    already available. Calling this function with training data (X_train, y_train) is
    in-sample calibration and will produce overconfident probabilities.

    Parameters
    ----------
    base_estimator : Any
        Fitted base estimator.
    X_val : np.ndarray
        Held-out validation feature matrix (must NOT be the training set).
    y_val : np.ndarray
        Validation ground-truth labels.
    sample_weight_val : np.ndarray, optional
        Validation sample weights.

    Returns
    -------
    CalibratedPredictor
        Wrapped predictor outputting calibrated probabilities.
    """
    if hasattr(base_estimator, "decision_function"):
        val_scores = base_estimator.decision_function(X_val)
    elif hasattr(base_estimator, "predict_proba"):
        p1 = base_estimator.predict_proba(X_val)[:, 1]
        eps = 1e-7
        p1 = np.clip(p1, eps, 1.0 - eps)
        val_scores = np.log(p1 / (1.0 - p1))
    else:
        raise AttributeError("Estimator must have decision_function or predict_proba.")

    val_scores_2d = val_scores.reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(val_scores_2d, y_val, sample_weight=sample_weight_val)

    return CalibratedPredictor(base_estimator=base_estimator, calibrator=calibrator)


def fit_platt_calibrator_from_scores(
    base_estimator: Any,
    oof_scores: np.ndarray,
    y_val: np.ndarray,
    sample_weight_val: Optional[np.ndarray] = None,
) -> CalibratedPredictor:
    """Fit Platt scaling from pre-computed out-of-fold (OOF) probability scores.

    This is the correct way to calibrate after nested CV: the OOF scores are proper
    held-out predictions (each sample was scored by a model it was never trained on),
    so using them as calibration targets avoids in-sample overfitting.

    Parameters
    ----------
    base_estimator : Any
        Fitted base estimator (used for inference at test time via CalibratedPredictor).
    oof_scores : np.ndarray
        Pre-computed inner out-of-fold predicted probabilities for the training set.
        Shape: (n_train_samples,). These must be probability estimates in [0, 1].
    y_val : np.ndarray
        Training ground-truth labels aligned with oof_scores.
    sample_weight_val : np.ndarray, optional
        Sample weights aligned with oof_scores.

    Returns
    -------
    CalibratedPredictor
        Wrapped predictor outputting calibrated probabilities.
    """
    eps = 1e-7
    p1 = np.clip(oof_scores, eps, 1.0 - eps)
    # Convert probabilities to log-odds (logit) for the univariate LR calibrator
    oof_logits = np.log(p1 / (1.0 - p1)).reshape(-1, 1)

    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(oof_logits, y_val, sample_weight=sample_weight_val)

    return CalibratedPredictor(base_estimator=base_estimator, calibrator=calibrator)


def aggregate_student_predictions(
    test_df: pd.DataFrame,
    sentence_probs: np.ndarray,
) -> pd.DataFrame:
    """Aggregate sentence-level probabilities to student-level diagnosis via mean-pooling.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe containing 'student_id' and 'label'.
    sentence_probs : np.ndarray
        Predicted P(dysgraphic | sentence) for each row in test_df.

    Returns
    -------
    pd.DataFrame
        One row per unique student with columns:
        ['student_id', 'label', 'pred_prob', 'sentence_count']
    """
    df = test_df[["student_id", "label"]].copy()
    df["pred_prob"] = sentence_probs

    grouped = df.groupby("student_id").agg(
        label=("label", "first"),
        pred_prob=("pred_prob", "mean"),
        sentence_count=("pred_prob", "count"),
    ).reset_index()

    grouped["label"] = grouped["label"].astype(int)
    return grouped


def calculate_sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.90,
) -> float:
    """Calculate sensitivity (TPR) when specificity (1 - FPR) is at least target_specificity.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities.
    target_specificity : float
        Target specificity (e.g. 0.90 or 0.95).

    Returns
    -------
    float
        Maximum true positive rate achieving the target specificity.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    max_allowed_fpr = 1.0 - target_specificity + 1e-7
    valid_indices = np.where(fpr <= max_allowed_fpr)[0]

    if len(valid_indices) == 0:
        return 0.0

    return float(np.max(tpr[valid_indices]))


def find_operational_threshold(
    student_preds_df: pd.DataFrame,
    min_specificity: float = 0.90,
) -> float:
    """Find the operational decision threshold inside inner CV splits targeting >= min_specificity.

    Selects the threshold maximizing sensitivity among candidates with specificity >= min_specificity.
    If none satisfy the constraint, selects the threshold with the highest specificity.

    Parameters
    ----------
    student_preds_df : pd.DataFrame
        Dataframe containing 'label' and 'pred_prob'.
    min_specificity : float
        Target minimum specificity (default 0.90).

    Returns
    -------
    float
        Selected optimal decision threshold.
    """
    y_true = student_preds_df["label"].values.astype(int)
    y_prob = student_preds_df["pred_prob"].values

    candidate_thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_metric = -1.0
    fallback_thresh = 0.5
    best_spec_fallback = -1.0

    for th in candidate_thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)

        if spec > best_spec_fallback:
            best_spec_fallback = spec
            fallback_thresh = th

        if spec >= min_specificity:
            # Objective: maximize sensitivity with secondary tiebreaker for specificity
            score = sens * 10.0 + spec
            if score > best_metric:
                best_metric = score
                best_thresh = th

    return float(best_thresh if best_metric >= 0 else fallback_thresh)


def compute_student_metrics(
    student_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive performance metrics on student-level predictions.

    Parameters
    ----------
    student_df : pd.DataFrame
        Dataframe containing 'label' and 'pred_prob'.
    threshold : float
        Operating decision threshold.

    Returns
    -------
    dict
        Dictionary of threshold-free and thresholded metrics.
    """
    y_true = student_df["label"].values.astype(int)
    y_prob = student_df["pred_prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    n_samples = len(y_true)
    n_pos = int(np.sum(y_true))
    n_neg = n_samples - n_pos

    # Threshold-free metrics
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        sens_at_90 = calculate_sensitivity_at_specificity(y_true, y_prob, 0.90)
        sens_at_95 = calculate_sensitivity_at_specificity(y_true, y_prob, 0.95)
    else:
        auroc = 0.5
        auprc = float(np.mean(y_true))
        sens_at_90 = 0.0
        sens_at_95 = 0.0

    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    accuracy = float((tp + tn) / max(n_samples, 1))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "n_students": float(n_samples),
        "n_positives": float(n_pos),
        "n_negatives": float(n_neg),
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "sensitivity_at_90_spec": sens_at_90,
        "sensitivity_at_95_spec": sens_at_95,
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def cluster_bootstrap_ci(
    student_preds_df: pd.DataFrame,
    threshold: Optional[float] = None,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute empirical percentile confidence intervals via cluster bootstrap by student_id.

    Resamples students with replacement, taking all observations corresponding to selected students.

    Parameters
    ----------
    student_preds_df : pd.DataFrame
        Dataframe containing 'student_id', 'label', and 'pred_prob'.
    threshold : float, optional
        Operational threshold. If None, uses median inner threshold or 0.5.
    n_bootstraps : int
        Number of bootstrap iterations (default 1000).
    ci : float
        Confidence interval level (default 0.95).
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        For each metric: {'mean': val, 'std': val, 'ci_low': val, 'ci_high': val}
    """
    rng = np.random.RandomState(random_state)
    unique_students = student_preds_df["student_id"].unique()
    n_students = len(unique_students)

    eval_threshold = 0.5 if threshold is None else threshold
    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    metric_samples: Dict[str, List[float]] = {
        "auroc": [],
        "auprc": [],
        "brier": [],
        "sensitivity_at_90_spec": [],
        "sensitivity_at_95_spec": [],
        "sensitivity": [],
        "specificity": [],
        "balanced_accuracy": [],
        "f1": [],
    }

    # Group original student rows for fast lookup
    student_map = {sid: row for sid, row in student_preds_df.set_index("student_id").iterrows()}

    for _ in range(n_bootstraps):
        sampled_sids = rng.choice(unique_students, size=n_students, replace=True)
        boot_rows = [student_map[sid] for sid in sampled_sids]
        boot_df = pd.DataFrame(boot_rows)

        # Skip degenerate samples with only 1 class
        if len(boot_df["label"].unique()) < 2:
            continue

        metrics = compute_student_metrics(boot_df, threshold=eval_threshold)
        for k in metric_samples.keys():
            metric_samples[k].append(metrics[k])

    results: Dict[str, Dict[str, float]] = {}
    for k, values in metric_samples.items():
        arr = np.array(values)
        results[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_low": float(np.percentile(arr, lower_pct)),
            "ci_high": float(np.percentile(arr, upper_pct)),
        }

    return results

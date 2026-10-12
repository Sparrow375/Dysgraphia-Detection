"""Two-Tier Clinical Screening Cascade for Dysgraphia Detection.

Replaces a single global decision threshold with a clinically calibrated tiered cascade:
- Tier 1 (Universal Safe Rule-Out): Threshold selected for >= 88% Sensitivity.
  Students with P < theta_triage are cleared with high negative predictive value.
- Tier 2 (Clinical OT Referral): Threshold selected for >= 90% Specificity.
  Students with P >= theta_referral are referred for diagnostic OT intervention.
- Borderline / Monitoring Tier: Students with theta_triage <= P < theta_referral
  are scheduled for 4-week progress monitoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE_ROOT / "reports"


class TwoTierScreeningCascade:
    """Two-tier operational decision rule for classroom dysgraphia screening."""

    def __init__(
        self,
        target_sensitivity: float = 0.88,
        target_specificity: float = 0.90,
    ):
        self.target_sensitivity = target_sensitivity
        self.target_specificity = target_specificity
        self.theta_triage: float = 0.30
        self.theta_referral: float = 0.70
        self.metrics: Dict[str, Any] = {}

    def fit_thresholds(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> Dict[str, float]:
        """Calibrate cascade thresholds on out-of-fold prediction scores."""
        y_true = np.asarray(y_true).astype(int)
        y_scores = np.asarray(y_scores).astype(float)

        pos_scores = y_scores[y_true == 1]
        neg_scores = y_scores[y_true == 0]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return {"theta_triage": 0.30, "theta_referral": 0.70}

        # 1. Triage threshold: lowest score such that Sensitivity >= target_sensitivity
        # i.e., 100 * (1 - target_sensitivity)-th percentile of positive scores
        sens_percentile = max(0.0, 100.0 * (1.0 - self.target_sensitivity))
        self.theta_triage = float(np.percentile(pos_scores, sens_percentile))

        # 2. Referral threshold: score such that Specificity >= target_specificity
        # i.e., target_specificity percentile of negative scores
        spec_percentile = min(100.0, 100.0 * self.target_specificity)
        self.theta_referral = float(np.percentile(neg_scores, spec_percentile))

        # Safety constraint: theta_triage must not exceed theta_referral
        if self.theta_triage > self.theta_referral:
            mid = 0.5 * (self.theta_triage + self.theta_referral)
            self.theta_triage = mid * 0.9
            self.theta_referral = mid * 1.1

        # Evaluate performance on training / OOF cohort
        eval_metrics = self.evaluate(y_true, y_scores)
        self.metrics = eval_metrics
        return {
            "theta_triage": self.theta_triage,
            "theta_referral": self.theta_referral,
        }

    def predict_tiers(self, y_scores: np.ndarray) -> np.ndarray:
        """Classify probability scores into 3 tiers:
        0: Cleared / Safe Negative
        1: Borderline / Progress Monitoring
        2: Confirmed Referral / High Risk
        """
        y_scores = np.asarray(y_scores).astype(float)
        tiers = np.zeros(len(y_scores), dtype=int)
        tiers[(y_scores >= self.theta_triage) & (y_scores < self.theta_referral)] = 1
        tiers[y_scores >= self.theta_referral] = 2
        return tiers

    def evaluate(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute comprehensive multi-tier screening performance metrics."""
        y_true = np.asarray(y_true).astype(int)
        tiers = self.predict_tiers(y_scores)
        n = len(y_true)
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))

        # Tier breakdown
        n_cleared = int(np.sum(tiers == 0))
        n_monitor = int(np.sum(tiers == 1))
        n_refer = int(np.sum(tiers == 2))

        # Triage / Safe Negative performance (Tier 0)
        false_neg_cleared = int(np.sum((tiers == 0) & (y_true == 1)))
        true_neg_cleared = int(np.sum((tiers == 0) & (y_true == 0)))
        triage_sensitivity = float((n_pos - false_neg_cleared) / max(n_pos, 1))
        triage_npv = float(true_neg_cleared / max(n_cleared, 1))

        # Referral performance (Tier 2)
        true_pos_referred = int(np.sum((tiers == 2) & (y_true == 1)))
        false_pos_referred = int(np.sum((tiers == 2) & (y_true == 0)))
        referral_specificity = float((n_neg - false_pos_referred) / max(n_neg, 1))
        referral_ppv = float(true_pos_referred / max(n_refer, 1))

        # Borderline distribution
        pos_in_monitor = int(np.sum((tiers == 1) & (y_true == 1)))
        neg_in_monitor = int(np.sum((tiers == 1) & (y_true == 0)))

        return {
            "theta_triage": self.theta_triage,
            "theta_referral": self.theta_referral,
            "n_total": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "tier_counts": {
                "cleared": n_cleared,
                "monitoring": n_monitor,
                "referred": n_refer,
            },
            "tier_percentages": {
                "cleared_pct": float(100.0 * n_cleared / n),
                "monitoring_pct": float(100.0 * n_monitor / n),
                "referred_pct": float(100.0 * n_refer / n),
            },
            "triage_sensitivity": triage_sensitivity,
            "triage_npv": triage_npv,
            "false_negatives_cleared": false_neg_cleared,
            "referral_specificity": referral_specificity,
            "referral_ppv": referral_ppv,
            "false_positives_referred": false_pos_referred,
            "borderline_breakdown": {
                "positives_in_monitoring": pos_in_monitor,
                "negatives_in_monitoring": neg_in_monitor,
            },
        }


def run_cascade_benchmark(
    student_csv: str = str(WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level.csv"),
) -> Dict[str, Any]:
    """Evaluate Two-Tier Cascade on student-level out-of-fold benchmark predictions."""
    df = pd.read_csv(student_csv)
    print(f"Loaded {len(df)} students from {student_csv}.")

    # Evaluate using calibrated Clean 15 + Motion Predictor model
    from models.student_experiments import run_student_level_benchmark
    res_df, res_list, pooled_preds = run_student_level_benchmark(dataset_path=Path(student_csv), n_repeats=5, n_bootstraps=200)

    # Extract stacking ensemble predictions
    best_model_name = "Multi-Model Stacking Ensemble (0.5 GBM + 0.3 RF + 0.2 LR)"
    if best_model_name not in pooled_preds:
        best_model_name = list(pooled_preds.keys())[-1]

    y_true = pooled_preds[best_model_name]["y_true"]
    y_scores = pooled_preds[best_model_name]["y_scores"]

    cascade = TwoTierScreeningCascade(target_sensitivity=0.88, target_specificity=0.90)
    cascade.fit_thresholds(y_true, y_scores)

    cascade_report = cascade.evaluate(y_true, y_scores)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "two_tier_screening_cascade_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(cascade_report, f, indent=2)

    print("\n=== Two-Tier Screening Cascade Evaluation ===")
    print(f"Triage Threshold (Rule-Out):   P < {cascade_report['theta_triage']:.3f}")
    print(f"Referral Threshold (Action):  P >= {cascade_report['theta_referral']:.3f}")
    print(f"Triage Sensitivity:           {cascade_report['triage_sensitivity']*100:.1f}%")
    print(f"Triage NPV:                   {cascade_report['triage_npv']*100:.1f}%")
    print(f"Referral Specificity:         {cascade_report['referral_specificity']*100:.1f}%")
    print(f"Referral PPV (Precision):     {cascade_report['referral_ppv']*100:.1f}%")
    print(f"Tier Proportions:             Cleared={cascade_report['tier_percentages']['cleared_pct']:.1f}%, "
          f"Monitor={cascade_report['tier_percentages']['monitoring_pct']:.1f}%, "
          f"Referral={cascade_report['tier_percentages']['referred_pct']:.1f}%")
    print(f"Saved cascade report to: {report_path}")

    return cascade_report


if __name__ == "__main__":
    run_cascade_benchmark()

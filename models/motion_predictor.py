"""Stylus-Anchored Motion Predictor.

Calibrates static paper handwriting metrics against 200 Hz digital stylus
ground-truth kinematics from 94 matched students:
- In-air hesitation ratio (air_ratio)
- Total stroke fragmentation count (stroke_count)
- Mean in-air pause duration (mean_pause_sec)
- Velocity dysfluency / coefficient of variation (cv_vel)

Infers calibrated continuous kinematic surrogates for the full 215-student cohort
without requiring specialized hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
STYLUS_CSV_PATH = WORKSPACE_ROOT / "data" / "school_a" / "stylus_student_features.csv"
STUDENT_LEVEL_CSV = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level.csv"
MODEL_SAVE_PATH = WORKSPACE_ROOT / "data" / "models" / "motion_predictor_weights.json"

# Predictor features available on static paper scans
PREDICTOR_PAPER_FEATURES: List[str] = [
    "endpoints_per_unit_width_mean",
    "components_per_unit_width_mean",
    "junctions_per_unit_width_mean",
    "jerk_proxy_mean",
    "tortuosity_median_mean",
    "words_written_ratio_mean",
    "word_width_per_char_mean",
    "gap_cv_mean",
    "gap_fraction_above_2h_mean",
    "baseline_rmse_norm_mean",
    "components_per_unit_width_dict_mean",
    "components_per_unit_width_max",
]

# Key target kinematic metrics from 200 Hz stylus
TARGET_KINEMATIC_METRICS: List[str] = [
    "stroke_count",
    "air_ratio",
    "mean_pause_sec",
    "cv_vel",
    "mean_acc",
]


class StylusMotionPredictor:
    """Supervised regression ensemble mapping paper handwriting to true kinematics."""

    def __init__(self, alphas: Optional[np.ndarray] = None):
        self.alphas = alphas if alphas is not None else np.logspace(-2, 3, 25)
        self.models: Dict[str, Pipeline] = {}
        self.feature_names: List[str] = PREDICTOR_PAPER_FEATURES
        self.cv_metrics: Dict[str, Dict[str, float]] = {}

    def fit_cv(
        self,
        merged_df: pd.DataFrame,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """Fit RidgeCV pipelines with outer K-Fold CV validation to measure generalization."""
        valid_feats = [f for f in self.feature_names if f in merged_df.columns]
        self.feature_names = valid_feats

        cv_results: Dict[str, Dict[str, float]] = {}

        for tgt in TARGET_KINEMATIC_METRICS:
            if tgt not in merged_df.columns:
                continue

            sub = merged_df[valid_feats + [tgt]].dropna()
            if len(sub) < 25:
                continue

            X = sub[valid_feats].values
            y = sub[tgt].values

            # Outer K-Fold CV for unbiased out-of-fold performance estimation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            oof_preds = np.zeros(len(y))

            for train_idx, val_idx in kf.split(X):
                fold_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=self.alphas)),
                ])
                fold_pipe.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx] = fold_pipe.predict(X[val_idx])

            r_val, p_val = pearsonr(y, oof_preds)
            rho_val, _ = spearmanr(y, oof_preds)
            mae = float(np.mean(np.abs(y - oof_preds)))
            rmse = float(np.sqrt(np.mean((y - oof_preds) ** 2)))

            cv_results[tgt] = {
                "n_samples": len(sub),
                "pearson_r": float(r_val),
                "pearson_p": float(p_val),
                "spearman_rho": float(rho_val),
                "mae": mae,
                "rmse": rmse,
            }

            # Fit final production model on all matched data
            final_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=self.alphas)),
            ])
            final_pipe.fit(X, y)
            self.models[tgt] = final_pipe

        self.cv_metrics = cv_results
        return cv_results

    def predict(self, student_df: pd.DataFrame) -> pd.DataFrame:
        """Infer kinematic metrics for any student dataframe."""
        df_out = student_df.copy()
        valid_feats = [f for f in self.feature_names if f in df_out.columns]

        X_df = df_out[valid_feats].copy()
        # Impute missing feature columns with column medians
        for col in valid_feats:
            if X_df[col].isna().any():
                X_df[col] = X_df[col].fillna(X_df[col].median())

        X_mat = X_df.values

        for tgt, pipe in self.models.items():
            pred_col = f"motion_pred_{tgt}"
            preds = pipe.predict(X_mat)
            df_out[pred_col] = preds

        return df_out

    def save(self, filepath: Optional[Path] = None) -> None:
        """Save model coefficients and CV metrics to JSON."""
        save_p = filepath or MODEL_SAVE_PATH
        save_p.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "feature_names": self.feature_names,
            "cv_metrics": self.cv_metrics,
            "models": {},
        }

        for tgt, pipe in self.models.items():
            scaler = pipe.named_steps["scaler"]
            ridge = pipe.named_steps["ridge"]
            payload["models"][tgt] = {
                "alpha": float(ridge.alpha_),
                "intercept": float(ridge.intercept_),
                "coef": [float(c) for c in ridge.coef_],
                "scaler_mean": [float(m) for m in scaler.mean_],
                "scaler_scale": [float(s) for s in scaler.scale_],
            }

        with open(save_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Motion predictor weights saved to: {save_p}")


def train_motion_predictor(
    paper_csv: str = str(STUDENT_LEVEL_CSV),
    stylus_csv: str = str(STYLUS_CSV_PATH),
    save_weights: bool = True,
) -> Tuple[StylusMotionPredictor, pd.DataFrame]:
    """Train motion predictor on matched cohort and return predictor instance + augmented dataset."""
    paper_df = pd.read_csv(paper_csv)
    stylus_df = pd.read_csv(stylus_csv)

    paper_df["student_id"] = paper_df["student_id"].astype(str)
    stylus_df["student_id"] = stylus_df["student_id"].astype(str)

    merged = pd.merge(paper_df, stylus_df, on="student_id", suffixes=("_paper", "_stylus"))
    print(f"Matched {len(merged)} students with paired paper scans and 200 Hz stylus metrics.")

    predictor = StylusMotionPredictor()
    cv_metrics = predictor.fit_cv(merged)

    print("\n=== Stylus Motion Predictor: 5-Fold Cross-Validation Metrics ===")
    for tgt, met in cv_metrics.items():
        print(
            f"  Target: {tgt:16s} | Pearson r = {met['pearson_r']:+.3f} (p={met['pearson_p']:.2e}) | "
            f"Spearman rho = {met['spearman_rho']:+.3f} | RMSE = {met['rmse']:.3f}"
        )

    if save_weights:
        predictor.save()

    # Apply predictions across entire student cohort
    augmented_df = predictor.predict(paper_df)
    return predictor, augmented_df


if __name__ == "__main__":
    train_motion_predictor()

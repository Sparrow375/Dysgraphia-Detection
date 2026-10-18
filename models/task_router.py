"""Task-Aware Dynamic Routing Architecture for Dysgraphia Detection.

Solves the task-discrepancy and missing-dictation failure mode:
1. In real school notebooks and non-standard protocols (e.g. School B Grade 3 and Grade 8),
   dictation tasks are frequently absent.
2. The core Workstream A model relies heavily on dictation features (e.g.,
   components_per_unit_width_dict_mean, words_written_ratio_dict_minus_copy).
   Imputing missing dictation features with population medians dilutes the signal,
   causing dysgraphic students without dictation to appear as normal controls.
3. The Task-Aware Router detects which tasks are present on the notebook sheet,
   and routes the student to a dedicated sub-model:
   - FullProtocolSubModel: For sheets with Copy + Dictation (all 6 tasks or standard protocol).
   - NoDictationSubModel: For sheets with Copy + Free-Writing (no dictation).
   - CopyOnlySubModel: For sheets with only Copy tasks.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


class TaskRoute(str, Enum):
    """Categorical routing decision based on available tasks."""
    FULL_PROTOCOL = "full_protocol"
    NO_DICTATION = "no_dictation"
    COPY_ONLY = "copy_only"


# 1. Full Protocol Features: Exploits auditory-motor dictation load collapse
FULL_PROTOCOL_FEATURES: List[str] = [
    "components_per_unit_width_dict_mean",
    "components_per_unit_width_max",
    "components_per_unit_width_std",
    "components_per_unit_width_eng_minus_hindi",
    "endpoints_per_unit_width_mean",
    "junctions_per_unit_width_mean",
    "junctions_per_unit_width_own_mean",
    "words_written_ratio_eng_minus_hindi",
    "words_written_ratio_dict_minus_copy",
    "matra_ratio_std",
    "matra_ratio_max",
    "shirorekha_rms_deviation_norm_max",
    "tortuosity_median_cv",
    "word_width_per_char_std",
    "gap_fraction_above_2h_mean",
]

# 2. No-Dictation Features: Focuses on spatial alignment, rule offset, copy size jitter & script constraints
NO_DICTATION_FEATURES: List[str] = [
    "rule_offset_std_mean",
    "baseline_rmse_norm_mean",
    "word_height_cv_copy_mean",
    "components_per_unit_width_copy_mean",
    "components_per_unit_width_max",
    "components_per_unit_width_std",
    "components_per_unit_width_eng_minus_hindi",
    "endpoints_per_unit_width_mean",
    "junctions_per_unit_width_mean",
    "junctions_per_unit_width_own_mean",
    "words_written_ratio_eng_minus_hindi",
    "matra_ratio_std",
    "matra_ratio_max",
    "shirorekha_rms_deviation_norm_max",
    "tortuosity_median_cv",
    "word_width_per_char_std",
    "gap_fraction_above_2h_mean",
]

# 3. Copy-Only Features: Strictly uses Copy tasks + Spatial-Geometric properties (zero free-write or dictation)
COPY_ONLY_FEATURES: List[str] = [
    "rule_offset_std_mean",
    "baseline_rmse_norm_mean",
    "word_height_cv_copy_mean",
    "components_per_unit_width_copy_mean",
    "components_per_unit_width_max",
    "components_per_unit_width_std",
    "components_per_unit_width_eng_minus_hindi",
    "endpoints_per_unit_width_mean",
    "junctions_per_unit_width_mean",
    "words_written_ratio_eng_minus_hindi",
    "matra_ratio_std",
    "matra_ratio_max",
    "shirorekha_rms_deviation_norm_max",
    "tortuosity_median_cv",
    "word_width_per_char_std",
    "gap_fraction_above_2h_mean",
]


def detect_route_from_tasks(task_names: List[str]) -> TaskRoute:
    """Determine the optimal sub-model route from a list of present task strings."""
    t_lower = [t.lower() for t in task_names]
    has_dict = any("dict" in t for t in t_lower)
    has_own = any("own" in t or "free" in t for t in t_lower)

    if has_dict:
        return TaskRoute.FULL_PROTOCOL
    elif has_own:
        return TaskRoute.NO_DICTATION
    else:
        return TaskRoute.COPY_ONLY


def detect_route_from_features(record: Union[pd.Series, Dict[str, Any]]) -> TaskRoute:
    """Determine the sub-model route by inspecting available non-null features."""
    dict_val = record.get("components_per_unit_width_dict_mean", np.nan)
    has_dict = pd.notna(dict_val)

    if has_dict:
        return TaskRoute.FULL_PROTOCOL

    own_val = record.get("junctions_per_unit_width_own_mean", np.nan)
    has_own = pd.notna(own_val)

    if has_own:
        return TaskRoute.NO_DICTATION

    return TaskRoute.COPY_ONLY


class TaskSubModel:
    """Encapsulates a specialized sub-model for a specific task availability regime."""

    def __init__(
        self,
        name: str,
        feature_names: List[str],
        model_type: str = "gbm",
        random_state: int = 42,
    ):
        self.name = name
        self.feature_names = feature_names
        self.model_type = model_type
        self.random_state = random_state
        self.medians: Dict[str, float] = {}
        self.clf: Any = None
        self._init_classifier()

    def _init_classifier(self) -> None:
        if self.model_type == "gbm":
            self.clf = HistGradientBoostingClassifier(
                max_depth=2,
                learning_rate=0.03,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=self.random_state,
            )
        elif self.model_type == "rf":
            self.clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(self, X_df: pd.DataFrame, y: np.ndarray) -> TaskSubModel:
        """Fit sub-model and compute training-set feature medians without test leakage."""
        valid_cols = [c for c in self.feature_names if c in X_df.columns]
        self.feature_names = valid_cols

        # Compute medians strictly on training fold
        self.medians = {}
        for col in valid_cols:
            med_val = float(X_df[col].dropna().median()) if len(X_df[col].dropna()) > 0 else 0.0
            self.medians[col] = med_val

        X_imputed = X_df[valid_cols].copy()
        for col in valid_cols:
            X_imputed[col] = X_imputed[col].fillna(self.medians[col])

        self.clf.fit(X_imputed.values, y)
        return self

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """Predict positive class probabilities for input dataframe."""
        X_imputed = pd.DataFrame(index=X_df.index)
        for col in self.feature_names:
            if col in X_df.columns:
                X_imputed[col] = X_df[col].fillna(self.medians.get(col, 0.0))
            else:
                X_imputed[col] = self.medians.get(col, 0.0)

        probs = self.clf.predict_proba(X_imputed.values)[:, 1]
        return probs


class TaskAwareRouter:
    """Dynamic routing engine dispatching handwriting sheets to protocol-matched sub-models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.sub_models: Dict[TaskRoute, TaskSubModel] = {
            TaskRoute.FULL_PROTOCOL: TaskSubModel(
                name="full_protocol",
                feature_names=FULL_PROTOCOL_FEATURES,
                model_type="gbm",
                random_state=random_state,
            ),
            TaskRoute.NO_DICTATION: TaskSubModel(
                name="no_dictation",
                feature_names=NO_DICTATION_FEATURES,
                model_type="rf",
                random_state=random_state,
            ),
            TaskRoute.COPY_ONLY: TaskSubModel(
                name="copy_only",
                feature_names=COPY_ONLY_FEATURES,
                model_type="rf",
                random_state=random_state,
            ),
        }
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, label_col: str = "label") -> TaskAwareRouter:
        """Fit sub-models on appropriate student subsets of the training data."""
        y_all = train_df[label_col].values.astype(int)

        # 1. Full Protocol: Students who have dictation tasks
        has_dict_mask = ~train_df["components_per_unit_width_dict_mean"].isna().values
        if np.sum(has_dict_mask) >= 15:
            df_full = train_df[has_dict_mask]
            y_full = y_all[has_dict_mask]
            self.sub_models[TaskRoute.FULL_PROTOCOL].fit(df_full, y_full)
        else:
            # Fallback to all data if dictation subset is small
            self.sub_models[TaskRoute.FULL_PROTOCOL].fit(train_df, y_all)

        # 2. No-Dictation Model: Trained on all training students using only non-dictation features
        # This teaches the model to classify without dictation crutches
        self.sub_models[TaskRoute.NO_DICTATION].fit(train_df, y_all)

        # 3. Copy-Only Model: Trained on all training students using strictly copy + spatial features
        self.sub_models[TaskRoute.COPY_ONLY].fit(train_df, y_all)

        self.is_fitted = True
        return self

    def route_students(self, df: pd.DataFrame) -> List[TaskRoute]:
        """Assign each student row to their appropriate sub-model route."""
        routes: List[TaskRoute] = []
        for _, row in df.iterrows():
            r = detect_route_from_features(row)
            routes.append(r)
        return routes

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Compute routed probability predictions for all students."""
        if not self.is_fitted:
            raise RuntimeError("TaskAwareRouter must be fitted before predict_proba.")

        routes = self.route_students(df)
        preds = np.zeros(len(df), dtype=float)

        for route_type, sub_model in self.sub_models.items():
            mask = np.array([r == route_type for r in routes])
            if np.any(mask):
                sub_df = df[mask]
                preds[mask] = sub_model.predict_proba(sub_df)

        return preds

    def predict_student(
        self,
        student_record: Union[pd.Series, Dict[str, Any]],
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify a single student with route explanation and audit metadata."""
        if not self.is_fitted:
            raise RuntimeError("TaskAwareRouter must be fitted before predict_student.")

        rec_series = pd.Series(student_record) if isinstance(student_record, dict) else student_record

        if task_names is not None:
            route = detect_route_from_tasks(task_names)
        else:
            route = detect_route_from_features(rec_series)

        sub_model = self.sub_models[route]
        df_single = pd.DataFrame([rec_series])
        prob = float(sub_model.predict_proba(df_single)[0])
        pred_label = int(prob >= 0.5)

        # Tier classification
        if prob < 0.276:
            tier = "Tier 1: Safe Rule-Out"
        elif prob >= 0.515:
            tier = "Tier 2: OT Referral"
        else:
            tier = "Borderline / Monitoring"

        return {
            "risk_score": prob,
            "predicted_label": pred_label,
            "route_assigned": route.value,
            "sub_model_used": sub_model.name,
            "tier": tier,
            "active_features_count": len(sub_model.feature_names),
            "rationale": f"Routed to '{route.value}' based on task presence; evaluated with {len(sub_model.feature_names)} features.",
        }

    def evaluate(self, df: pd.DataFrame, label_col: str = "label") -> Dict[str, Any]:
        """Compute performance metrics with breakdown by assigned route and grade."""
        y_true = df[label_col].values.astype(int)
        y_scores = self.predict_proba(df)
        routes = self.route_students(df)

        auroc = float(roc_auc_score(y_true, y_scores))
        auprc = float(average_precision_score(y_true, y_scores))
        brier = float(brier_score_loss(y_true, y_scores))

        neg = y_scores[y_true == 0]
        pos = y_scores[y_true == 1]
        thresh_90 = np.percentile(neg, 90) if len(neg) > 0 else 0.5
        thresh_80 = np.percentile(neg, 80) if len(neg) > 0 else 0.5
        sens_at_90 = float(np.mean(pos >= thresh_90)) if len(pos) > 0 else 0.0
        sens_at_80 = float(np.mean(pos >= thresh_80)) if len(pos) > 0 else 0.0

        # Operational metrics at threshold 0.5
        preds_bin = (y_scores >= 0.5).astype(int)
        cm = confusion_matrix(y_true, preds_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / max(tp + fn, 1))
        spec = float(tn / max(tn + fp, 1))
        bacc = float(0.5 * (sens + spec))

        # Route breakdown
        route_breakdown: Dict[str, Dict[str, Any]] = {}
        for r_type in TaskRoute:
            r_mask = np.array([r == r_type for r in routes])
            count = int(np.sum(r_mask))
            if count > 0:
                y_sub = y_true[r_mask]
                scores_sub = y_scores[r_mask]
                n_pos = int(np.sum(y_sub == 1))
                n_neg = int(np.sum(y_sub == 0))
                sub_auc = float(roc_auc_score(y_sub, scores_sub)) if n_pos > 0 and n_neg > 0 else None
                route_breakdown[r_type.value] = {
                    "count": count,
                    "positives": n_pos,
                    "controls": n_neg,
                    "auroc": sub_auc,
                }

        # Grade breakdown
        grade_breakdown: Dict[str, Dict[str, Any]] = {}
        if "grade" in df.columns:
            for g, gdf in df.groupby("grade"):
                g_idx = gdf.index.values
                # Map gdf indices into df indices
                g_pos = (df["grade"] == g).values
                y_g = y_true[g_pos]
                scores_g = y_scores[g_pos]
                n_pos = int(np.sum(y_g == 1))
                n_neg = int(np.sum(y_g == 0))
                g_auc = float(roc_auc_score(y_g, scores_g)) if n_pos > 0 and n_neg > 0 else None
                grade_breakdown[f"Grade_{g}"] = {
                    "count": len(gdf),
                    "positives": n_pos,
                    "controls": n_neg,
                    "auroc": g_auc,
                }

        return {
            "n_students": len(df),
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier,
            "balanced_acc": bacc,
            "sensitivity": sens,
            "specificity": spec,
            "sens_at_80_spec": sens_at_80,
            "sens_at_90_spec": sens_at_90,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "route_breakdown": route_breakdown,
            "grade_breakdown": grade_breakdown,
        }

"""Unified Main Screening Pipeline: Multilingual Dysgraphia Detection System.

Stage 25 System Overhaul:
This module unifies all active project workstreams into a single production-grade,
task-agnostic screening engine:
- Stream 1: Task-Agnostic Spatial Geometry & Intra-Page Volatility
- Stream 2: Writing-Tool-Calibrated Stroke-Width Dynamics & Dynamic Pressure Texture
- Stream 3: Domain-Adapted Prompt-Free TrOCR Legibility & Ink-to-Character Density Ratio
- Meta-Learner: Unified Logistic Regression meta-model fit on cross-validated stream predictions
- Clinical Decision: Two-Tier Clinical Screening Cascade (Safe Rule-Out vs Referral)

Zero manual per-grade weight hacks. Zero task-routing dependencies.
Unified across all 215 students (115 School A + 100 School B).

Usage:
    from pipeline.main_pipeline import DysgraphiaScreeningPipeline

    pipeline = DysgraphiaScreeningPipeline()
    pipeline.fit()
    result = pipeline.screen_student("SCHB_G4_IMG_20261007_100013")
    print(result.summary())
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.cascade import TwoTierScreeningCascade
from models.multi_stream_experiments import (
    STREAM1_FEATURES,
    STREAM2_FEATURES,
    STREAM3_FEATURES,
    load_multistream_cohort_dataset,
)

logger = logging.getLogger("pipeline.main_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ScreeningTier(str, Enum):
    """Clinical decision tier according to Two-Tier Screening Cascade."""
    TIER_1_RULE_OUT = "TIER_1_RULE_OUT"
    BORDERLINE_MONITORING = "BORDERLINE_MONITORING"
    TIER_2_REFERRAL = "TIER_2_REFERRAL"


@dataclass
class ScreeningResult:
    """Structured clinical report for an individual student screening."""
    student_id: str
    grade: int
    school: Optional[str] = None
    label_ground_truth: Optional[int] = None
    p_dysgraphia: float = 0.0
    stream_probabilities: Dict[str, float] = field(default_factory=dict)
    stream_weights_used: Dict[str, float] = field(default_factory=dict)
    route_taken: str = "task_agnostic_unified"
    screening_tier: ScreeningTier = ScreeningTier.TIER_1_RULE_OUT
    tier_label: str = "Cleared / Safe Rule-Out"
    clinical_action: str = "No clinical follow-up required"
    is_dysgraphia_suspected: bool = False
    is_pencil: Optional[bool] = None
    key_risk_factors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        res = asdict(self)
        res["screening_tier"] = self.screening_tier.value
        return res

    def summary(self) -> str:
        """Formatted clinical summary string."""
        status_icon = (
            "🔴 URGENT REFERRAL"
            if self.screening_tier == ScreeningTier.TIER_2_REFERRAL
            else (
                "🟡 MONITORING"
                if self.screening_tier == ScreeningTier.BORDERLINE_MONITORING
                else "🟢 CLEARED"
            )
        )
        s1 = self.stream_probabilities.get("stream_1_geometry", 0.0)
        s2 = self.stream_probabilities.get("stream_2_texture", 0.0)
        s3 = self.stream_probabilities.get("stream_3_ocr", 0.0)
        tool_desc = "Pencil" if self.is_pencil else "Pen (Ballpoint/Gel)"

        lines = [
            f"=== DYSGRAPHIA SCREENING REPORT: {self.student_id} (Grade {self.grade}) ===",
            f"Overall Status: {status_icon} [{self.tier_label}]",
            f"Calibrated Dysgraphia Risk: {self.p_dysgraphia:.1%}",
            f"Writing Instrument Detected: {tool_desc}",
            f"Architecture: Unified Task-Agnostic Multi-Stream Meta-Learner",
            f"Multi-Stream Breakdown:",
            f"  • Stream 1 (Geometry & Intra-Page Volatility): {s1:.1%} (Weight: {self.stream_weights_used.get('stream_1_geometry', 0.0):.2f})",
            f"  • Stream 2 (Tool-Calibrated Width/Pressure):   {s2:.1%} (Weight: {self.stream_weights_used.get('stream_2_texture', 0.0):.2f})",
            f"  • Stream 3 (Prompt-Free TrOCR Legibility):     {s3:.1%} (Weight: {self.stream_weights_used.get('stream_3_ocr', 0.0):.2f})",
            f"Clinical Recommendation: {self.clinical_action}",
        ]
        if self.key_risk_factors:
            lines.append("Key Biomarkers & Risk Indicators:")
            for rf in self.key_risk_factors[:5]:
                lines.append(f"  - {rf['domain']}: {rf['feature']} = {rf['value']:.3f} ({rf['note']})")
        return "\n".join(lines)


class DysgraphiaScreeningPipeline:
    """Unified task-agnostic multi-stream dysgraphia screening engine."""

    # Default Clinical Screening Thresholds calibrated on 215-student cohort
    TIER_1_THRESHOLD: float = 0.250  # Safe Rule-Out (High Sensitivity triage)
    TIER_2_THRESHOLD: float = 0.450  # Specialist Referral (High Specificity)

    def __init__(
        self,
        tier1_threshold: float = TIER_1_THRESHOLD,
        tier2_threshold: float = TIER_2_THRESHOLD,
        random_state: int = 42,
    ):
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.random_state = random_state

        # Stream 1: Spatial Geometry & Intra-Page Volatility
        self.pipe_s1: Optional[Pipeline] = None
        self.s1_features: List[str] = STREAM1_FEATURES.copy()
        self.s1_medians: Dict[str, float] = {}

        # Stream 2: Tool-Calibrated Micro-Texture & Pressure Model
        self.pipe_s2: Optional[Pipeline] = None
        self.s2_features: List[str] = STREAM2_FEATURES.copy()
        self.s2_medians: Dict[str, float] = {}

        # Stream 3: Prompt-Free TrOCR Legibility Model
        self.pipe_s3: Optional[Pipeline] = None
        self.s3_features: List[str] = STREAM3_FEATURES.copy()
        self.s3_medians: Dict[str, float] = {}

        # Meta-Learner: Learned Logistic Combiner (Unified across all grades)
        self.meta_learner: Optional[LogisticRegression] = None

        # Cascade Decision Engine
        self.cascade_engine = TwoTierScreeningCascade(
            target_sensitivity=0.88,
            target_specificity=0.90,
        )
        self.cascade_engine.theta_triage = tier1_threshold
        self.cascade_engine.theta_referral = tier2_threshold

        self.is_fitted: bool = False
        self._df_all_students: Optional[pd.DataFrame] = None

    def fit(self, df_train: Optional[pd.DataFrame] = None) -> DysgraphiaScreeningPipeline:
        """Fit all 3 stream models and train the unified meta-learner."""
        logger.info("Initializing unified Dysgraphia Screening Pipeline (Stage 25 Overhaul)...")

        if df_train is None:
            logger.info("Loading unified multi-stream cohort dataset (215 students)...")
            self._df_all_students = load_multistream_cohort_dataset()
            df_train = self._df_all_students.copy()
        else:
            self._df_all_students = df_train.copy()

        y_train = df_train["label"].values.astype(int)

        # 1. Fit Stream 1: Geometry & Intra-Page Volatility
        logger.info(f"Fitting Stream 1 ({len(self.s1_features)} task-agnostic geometric features)...")
        s1_df = df_train[[c for c in self.s1_features if c in df_train.columns]].copy()
        self.s1_medians = s1_df.median().fillna(0.0).to_dict()
        X_train_s1 = s1_df.fillna(self.s1_medians).fillna(0.0).values

        self.pipe_s1 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=self.random_state)),
        ])
        self.pipe_s1.fit(X_train_s1, y_train)

        # 2. Fit Stream 2: Tool-Calibrated Width & Dynamic Pressure
        logger.info(f"Fitting Stream 2 ({len(self.s2_features)} tool-calibrated micro-texture features)...")
        s2_df = df_train[[c for c in self.s2_features if c in df_train.columns]].copy()
        self.s2_medians = s2_df.median().fillna(0.0).to_dict()
        X_train_s2 = s2_df.fillna(self.s2_medians).fillna(0.0).values

        self.pipe_s2 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=self.random_state)),
        ])
        self.pipe_s2.fit(X_train_s2, y_train)

        # 3. Fit Stream 3: Prompt-Free TrOCR Legibility
        logger.info(f"Fitting Stream 3 ({len(self.s3_features)} prompt-free TrOCR features)...")
        s3_df = df_train[[c for c in self.s3_features if c in df_train.columns]].copy()
        self.s3_medians = s3_df.median().fillna(0.0).to_dict()
        X_train_s3 = s3_df.fillna(self.s3_medians).fillna(0.0).values

        self.pipe_s3 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=self.random_state)),
        ])
        self.pipe_s3.fit(X_train_s3, y_train)

        # 4. Generate Out-of-Fold probability vectors via StratifiedKFold to fit Meta-Learner
        logger.info("Generating out-of-fold probability vectors for meta-learner training...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_s1 = np.zeros(len(df_train))
        oof_s2 = np.zeros(len(df_train))
        oof_s3 = np.zeros(len(df_train))

        for tr_idx, val_idx in skf.split(df_train, y_train):
            # Inner fold s1
            p1 = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=self.random_state))])
            p1.fit(X_train_s1[tr_idx], y_train[tr_idx])
            oof_s1[val_idx] = p1.predict_proba(X_train_s1[val_idx])[:, 1]

            # Inner fold s2
            p2 = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=self.random_state))])
            p2.fit(X_train_s2[tr_idx], y_train[tr_idx])
            oof_s2[val_idx] = p2.predict_proba(X_train_s2[val_idx])[:, 1]

            # Inner fold s3
            p3 = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=self.random_state))])
            p3.fit(X_train_s3[tr_idx], y_train[tr_idx])
            oof_s3[val_idx] = p3.predict_proba(X_train_s3[val_idx])[:, 1]

        # 5. Fit Unified Meta-Learner
        logger.info("Fitting unified Multi-Stream Meta-Learner (C=0.5, balanced)...")
        is_pencil_vec = df_train["is_pencil"].values.astype(float) if "is_pencil" in df_train.columns else np.zeros(len(df_train))
        X_meta_tr = np.column_stack([oof_s1, oof_s2, oof_s3, is_pencil_vec])

        self.meta_learner = LogisticRegression(C=0.5, class_weight="balanced", random_state=self.random_state)
        self.meta_learner.fit(X_meta_tr, y_train)

        # Calibrate screening cascade thresholds if feasible
        meta_train_probs = self.meta_learner.predict_proba(X_meta_tr)[:, 1]
        try:
            self.cascade_engine.calibrate(y_train, meta_train_probs)
            self.tier1_threshold = float(self.cascade_engine.theta_triage)
            self.tier2_threshold = float(self.cascade_engine.theta_referral)
            logger.info(f"Screening cascade calibrated: Tier 1={self.tier1_threshold:.3f}, Tier 2={self.tier2_threshold:.3f}")
        except Exception as e:
            logger.info(f"Using default cascade thresholds: {e}")

        self.is_fitted = True
        logger.info("Unified Dysgraphia Screening Pipeline fitted successfully.")
        return self

    def screen_features(
        self,
        features_dict: Union[Dict[str, Any], pd.Series],
        grade: int = 4,
        student_id: str = "UNKNOWN_STUDENT",
        school: Optional[str] = None,
        label: Optional[int] = None,
    ) -> ScreeningResult:
        """Screen a student from their extracted multi-stream feature record."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted via `fit()` before calling `screen_features()`.")

        df_row = pd.DataFrame([features_dict])
        if "grade" not in df_row.columns:
            df_row["grade"] = grade

        # 1. Stream 1: Task-Agnostic Geometry
        X_s1 = np.zeros((1, len(self.s1_features)))
        for j, c in enumerate(self.s1_features):
            val = df_row[c].values[0] if c in df_row.columns and pd.notna(df_row[c].values[0]) else self.s1_medians.get(c, 0.0)
            X_s1[0, j] = float(val) if pd.notna(val) else self.s1_medians.get(c, 0.0)
        s1_prob = float(self.pipe_s1.predict_proba(X_s1)[0, 1])

        # 2. Stream 2: Tool-Calibrated Micro-Texture & Stroke Dynamics
        X_s2 = np.zeros((1, len(self.s2_features)))
        for j, c in enumerate(self.s2_features):
            val = df_row[c].values[0] if c in df_row.columns and pd.notna(df_row[c].values[0]) else self.s2_medians.get(c, 0.0)
            X_s2[0, j] = float(val) if pd.notna(val) else self.s2_medians.get(c, 0.0)
        s2_prob = float(self.pipe_s2.predict_proba(X_s2)[0, 1])

        # 3. Stream 3: Prompt-Free TrOCR Legibility
        X_s3 = np.zeros((1, len(self.s3_features)))
        for j, c in enumerate(self.s3_features):
            val = df_row[c].values[0] if c in df_row.columns and pd.notna(df_row[c].values[0]) else self.s3_medians.get(c, 0.0)
            X_s3[0, j] = float(val) if pd.notna(val) else self.s3_medians.get(c, 0.0)
        s3_prob = float(self.pipe_s3.predict_proba(X_s3)[0, 1])

        # Tool classification indicator
        if "is_pencil" in df_row.columns and pd.notna(df_row["is_pencil"].values[0]):
            is_pencil = bool(float(df_row["is_pencil"].values[0]) > 0.5)
        elif grade <= 3:
            is_pencil = True
        else:
            is_pencil = False

        # 4. Unified Meta-Learner Prediction
        X_meta = np.array([[s1_prob, s2_prob, s3_prob, float(is_pencil)]])
        p_final = float(self.meta_learner.predict_proba(X_meta)[0, 1])
        p_final = float(np.clip(p_final, 0.001, 0.999))

        # Stream weight attribution from Meta-Learner coefficients
        coefs = self.meta_learner.coef_[0]
        pos_coefs = np.maximum(coefs[:3], 0.0)
        sum_c = np.sum(pos_coefs)
        if sum_c > 0:
            weights = {
                "stream_1_geometry": float(pos_coefs[0] / sum_c),
                "stream_2_texture": float(pos_coefs[1] / sum_c),
                "stream_3_ocr": float(pos_coefs[2] / sum_c),
            }
        else:
            weights = {"stream_1_geometry": 0.333, "stream_2_texture": 0.333, "stream_3_ocr": 0.334}

        # Two-Tier Screening Cascade Decision
        if p_final >= self.tier2_threshold:
            tier = ScreeningTier.TIER_2_REFERRAL
            tier_label = "Tier 2 Specialist Referral"
            clinical_action = (
                "Formal occupational therapy (OT) and neurodevelopmental evaluation indicated. "
                "Significant, multi-domain graphomotor and transcription breakdown detected."
            )
            is_suspected = True
        elif p_final >= self.tier1_threshold:
            tier = ScreeningTier.BORDERLINE_MONITORING
            tier_label = "Borderline / Progress Monitoring"
            clinical_action = (
                "Tier 1 classroom accommodations and 4-week structured handwriting progress monitoring. "
                "Re-screen at 4 weeks before specialist referral."
            )
            is_suspected = True
        else:
            tier = ScreeningTier.TIER_1_RULE_OUT
            tier_label = "Cleared / Safe Rule-Out"
            clinical_action = "No dysgraphia signs detected. Neurotypical handwriting motor development."
            is_suspected = False

        # Identify Key Clinical Biomarkers
        risk_factors = self._extract_risk_factors(features_dict)

        return ScreeningResult(
            student_id=student_id,
            grade=grade,
            school=school,
            label_ground_truth=label,
            p_dysgraphia=p_final,
            stream_probabilities={
                "stream_1_geometry": s1_prob,
                "stream_2_texture": s2_prob,
                "stream_3_ocr": s3_prob,
            },
            stream_weights_used=weights,
            route_taken="task_agnostic_unified",
            screening_tier=tier,
            tier_label=tier_label,
            clinical_action=clinical_action,
            is_dysgraphia_suspected=is_suspected,
            is_pencil=is_pencil,
            key_risk_factors=risk_factors,
        )

    def screen_student(self, student_id: str) -> ScreeningResult:
        """Screen a specific student by student_id from cohort datasets."""
        if not self.is_fitted:
            self.fit()

        if self._df_all_students is None or student_id not in self._df_all_students["student_id"].values:
            self._df_all_students = load_multistream_cohort_dataset()

        matches = self._df_all_students[self._df_all_students["student_id"] == student_id]
        if len(matches) == 0:
            raise KeyError(f"Student ID '{student_id}' not found in cohort datasets.")

        row = matches.iloc[0]
        grade = int(row.get("grade", 4))
        school = str(row.get("school", "unknown"))
        label = int(row["label"]) if "label" in row and pd.notna(row["label"]) else None

        return self.screen_features(
            features_dict=row,
            grade=grade,
            student_id=student_id,
            school=school,
            label=label,
        )

    def screen_cohort(
        self,
        df_cohort: Optional[pd.DataFrame] = None,
        school: Optional[str] = None,
    ) -> pd.DataFrame:
        """Screen a batch cohort and return structured results DataFrame."""
        if not self.is_fitted:
            self.fit()

        if df_cohort is None:
            if self._df_all_students is None:
                self._df_all_students = load_multistream_cohort_dataset()
            df_target = self._df_all_students.copy()
        else:
            df_target = df_cohort.copy()

        if school is not None:
            df_target = df_target[df_target["school"].str.lower() == school.lower()].copy()

        logger.info(f"Screening cohort of {len(df_target)} students...")
        records = []
        for _, row in df_target.iterrows():
            sid = str(row.get("student_id", "UNKNOWN"))
            grade = int(row.get("grade", 4))
            sch = str(row.get("school", "unknown"))
            lbl = int(row["label"]) if "label" in row and pd.notna(row["label"]) else None

            res = self.screen_features(
                features_dict=row,
                grade=grade,
                student_id=sid,
                school=sch,
                label=lbl,
            )
            records.append({
                "student_id": res.student_id,
                "school": res.school,
                "grade": res.grade,
                "label": res.label_ground_truth,
                "p_dysgraphia": round(res.p_dysgraphia, 4),
                "screening_tier": res.screening_tier.value,
                "tier_label": res.tier_label,
                "route_taken": res.route_taken,
                "p_s1_geometry": round(res.stream_probabilities.get("stream_1_geometry", 0.0), 4),
                "p_s2_texture": round(res.stream_probabilities.get("stream_2_texture", 0.0), 4),
                "p_s3_ocr": round(res.stream_probabilities.get("stream_3_ocr", 0.0), 4),
                "is_pencil": res.is_pencil,
                "is_dysgraphia_suspected": res.is_dysgraphia_suspected,
                "clinical_action": res.clinical_action,
            })

        return pd.DataFrame(records)

    def _extract_risk_factors(self, feat: Union[Dict[str, Any], pd.Series]) -> List[Dict[str, Any]]:
        """Identify anomalous biomarkers explaining the screening decision."""
        factors = []
        # 1. Peak intra-page stroke fragmentation
        val = feat.get("components_per_unit_width_p90", None)
        if val is not None and pd.notna(val) and float(val) > 2.5:
            factors.append({
                "domain": "Fine Motor Execution",
                "feature": "components_per_unit_width_p90",
                "value": float(val),
                "note": "Severe stroke fragmentation and pen-stutter across high-stress lines",
            })

        # 2. Graphomotor fatigue decay
        val = feat.get("components_per_unit_width_late_minus_early", None)
        if val is not None and pd.notna(val) and float(val) > 0.4:
            factors.append({
                "domain": "Neuromotor Stamina",
                "feature": "components_per_unit_width_late_minus_early",
                "value": float(val),
                "note": "Pronounced handwriting degradation and motor fatigue from page top to bottom",
            })

        # 3. Tool-normalized stroke width jitter
        val = feat.get("stroke_width_jitter_tool_norm", None)
        if val is not None and pd.notna(val) and float(val) > 0.40:
            factors.append({
                "domain": "Fine Motor Stability",
                "feature": "stroke_width_jitter_tool_norm",
                "value": float(val),
                "note": "Irregular nib pressure and high stroke-width volatility relative to tool baseline",
            })

        # 4. Tool-normalized pressure intensity CV
        val = feat.get("pressure_intensity_cv_tool_norm", None)
        if val is not None and pd.notna(val) and float(val) > 0.35:
            factors.append({
                "domain": "Dynamic Pressure Control",
                "feature": "pressure_intensity_cv_tool_norm",
                "value": float(val),
                "note": "High pressure coefficient of variation indicating unstable grip or tremor",
            })

        # 5. Prompt-free TrOCR character entropy
        val = feat.get("trocr_char_entropy_mean", None)
        if val is not None and pd.notna(val) and float(val) > 0.45:
            factors.append({
                "domain": "Transcription Legibility",
                "feature": "trocr_char_entropy_mean",
                "value": float(val),
                "note": "Ambiguous, highly diffuse character formations confusing optical recognition",
            })

        # 6. Physical ink to character density mismatch
        val = feat.get("ink_to_char_density_ratio", None)
        if val is not None and pd.notna(val) and float(val) > 3.0:
            factors.append({
                "domain": "Graphemic Coherence",
                "feature": "ink_to_char_density_ratio",
                "value": float(val),
                "note": "Excessive ink fragments relative to recognized characters (scribbles/overwriting)",
            })

        return factors


def main():
    """Command-line entrypoint for the unified main screening pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified Multilingual Dysgraphia Screening Pipeline"
    )
    parser.add_argument(
        "--student",
        type=str,
        default=None,
        help="Screen a specific student ID (e.g. SCHB_G4_IMG_20261007_100013)",
    )
    parser.add_argument(
        "--school",
        type=str,
        default=None,
        choices=["school_a", "school_b"],
        help="Screen all students from a specific school",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Filepath to export screening CSV or JSON report",
    )
    args = parser.parse_args()

    pipeline = DysgraphiaScreeningPipeline()
    pipeline.fit()

    if args.student:
        result = pipeline.screen_student(args.student)
        print("\n" + result.summary() + "\n")
        if args.export:
            out_p = Path(args.export)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Exported student report to {out_p}")
    else:
        df_results = pipeline.screen_cohort(school=args.school)
        print("\n=== SCREENING SUMMARY ===")
        print(f"Total Students Screened: {len(df_results)}")
        print(f"Tier 1 (Safe Rule-Out / Cleared): {sum(df_results['screening_tier'] == 'TIER_1_RULE_OUT')} ({sum(df_results['screening_tier'] == 'TIER_1_RULE_OUT')/len(df_results):.1%})")
        print(f"Monitoring (Borderline):          {sum(df_results['screening_tier'] == 'BORDERLINE_MONITORING')} ({sum(df_results['screening_tier'] == 'BORDERLINE_MONITORING')/len(df_results):.1%})")
        print(f"Tier 2 (Specialist Referral):     {sum(df_results['screening_tier'] == 'TIER_2_REFERRAL')} ({sum(df_results['screening_tier'] == 'TIER_2_REFERRAL')/len(df_results):.1%})")

        if args.export:
            out_p = Path(args.export)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() == ".json":
                df_results.to_json(out_p, orient="records", indent=2)
            else:
                df_results.to_csv(out_p, index=False)
            print(f"Exported cohort screening report to {out_p}")


if __name__ == "__main__":
    main()

"""Unified Main Screening Pipeline: Multilingual Dysgraphia Detection System.

This module unifies all active project workstreams into a single production-grade
screening engine:
- Stream 1: Spatial Geometry, Rule Adherence & Graph Kinematics (Task-Aware Router)
- Stream 2: Stroke-Width Dynamics & Dynamic Pencil Pressure Micro-Texture
- Stream 3: Modern TrOCR Legibility, Omission Ratios & Letter-Level Analytics
- Stream 4: Deep Visual Patch Representation (DenseNet201)
- Ensemble: Grade-Adaptive Multi-Stream Stacking Router
- Clinical Decision: Two-Tier Clinical Screening Cascade (Safe Rule-Out vs Referral)

Usage:
    from pipeline.main_pipeline import DysgraphiaScreeningPipeline

    pipeline = DysgraphiaScreeningPipeline()
    pipeline.fit()
    result = pipeline.screen_student("SCHB_G4_Roll09")
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.cascade import TwoTierScreeningCascade
from models.multi_stream_experiments import (
    STREAM2_CORE_FEATURES,
    STREAM3_CORE_FEATURES,
    assemble_multistream_student_dataset,
)
from models.task_router import FULL_PROTOCOL_FEATURES, NO_DICTATION_FEATURES, TaskAwareRouter

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
    route_taken: str = "full_protocol"
    screening_tier: ScreeningTier = ScreeningTier.TIER_1_RULE_OUT
    tier_label: str = "Cleared / Safe Rule-Out"
    clinical_action: str = "No clinical follow-up required"
    is_dysgraphia_suspected: bool = False
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
        s4 = self.stream_probabilities.get("stream_4_vision", 0.0)

        lines = [
            f"=== DYSGRAPHIA SCREENING REPORT: {self.student_id} (Grade {self.grade}) ===",
            f"Overall Status: {status_icon} [{self.tier_label}]",
            f"Calibrated Dysgraphia Risk: {self.p_dysgraphia:.1%}",
            f"Task Routing Strategy: {self.route_taken}",
            f"Multi-Stream Breakdown:",
            f"  • Stream 1 (Geometry & Task Kinematics): {s1:.1%} (Weight: {self.stream_weights_used.get('stream_1_geometry', 0.0):.2f})",
            f"  • Stream 2 (Stroke Width & Pressure):   {s2:.1%} (Weight: {self.stream_weights_used.get('stream_2_texture', 0.0):.2f})",
            f"  • Stream 3 (TrOCR Legibility & Errors): {s3:.1%} (Weight: {self.stream_weights_used.get('stream_3_ocr', 0.0):.2f})",
            f"  • Stream 4 (Deep Vision Patch Probe):   {s4:.1%} (Weight: {self.stream_weights_used.get('stream_4_vision', 0.0):.2f})",
            f"Clinical Recommendation: {self.clinical_action}",
        ]
        if self.key_risk_factors:
            lines.append("Key Biomarkers & Risk Indicators:")
            for rf in self.key_risk_factors[:5]:
                lines.append(f"  - {rf['domain']}: {rf['feature']} = {rf['value']:.3f} ({rf['note']})")
        return "\n".join(lines)


class DysgraphiaScreeningPipeline:
    """Unified multi-stream dysgraphia screening engine."""

    # Default Grade-Adaptive Stacking Weights based on developmental maturity
    GRADE_ADAPTIVE_WEIGHTS: Dict[int, Dict[str, float]] = {
        3: {"stream_1_geometry": 0.60, "stream_2_texture": 0.40, "stream_3_ocr": 0.00, "stream_4_vision": 0.00},
        4: {"stream_1_geometry": 0.15, "stream_2_texture": 0.15, "stream_3_ocr": 0.70, "stream_4_vision": 0.00},
        5: {"stream_1_geometry": 0.15, "stream_2_texture": 0.15, "stream_3_ocr": 0.60, "stream_4_vision": 0.10},
        6: {"stream_1_geometry": 0.40, "stream_2_texture": 0.00, "stream_3_ocr": 0.35, "stream_4_vision": 0.25},
        7: {"stream_1_geometry": 0.20, "stream_2_texture": 0.40, "stream_3_ocr": 0.30, "stream_4_vision": 0.10},
        8: {"stream_1_geometry": 0.20, "stream_2_texture": 0.00, "stream_3_ocr": 0.70, "stream_4_vision": 0.10},
    }
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "stream_1_geometry": 0.40,
        "stream_2_texture": 0.35,
        "stream_3_ocr": 0.20,
        "stream_4_vision": 0.05,
    }

    # Clinical screening thresholds
    TIER_1_THRESHOLD: float = 0.276  # High Sensitivity safe triage / rule-out
    TIER_2_THRESHOLD: float = 0.515  # High Specificity formal referral

    def __init__(
        self,
        tier1_threshold: float = TIER_1_THRESHOLD,
        tier2_threshold: float = TIER_2_THRESHOLD,
        random_state: int = 42,
    ):
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.random_state = random_state

        # Stream 1: Task-Aware Router
        self.s1_router: Optional[TaskAwareRouter] = None

        # Stream 2: Micro-Texture / Pressure Model
        self.pipe_s2: Optional[Pipeline] = None
        self.s2_features: List[str] = []
        self.s2_medians: Dict[str, float] = {}

        # Stream 3: TrOCR Legibility & Linguistic Model
        self.pipe_s3: Optional[Pipeline] = None
        self.s3_features: List[str] = []
        self.s3_medians: Dict[str, float] = {}

        # Stream 4: Vision predictions dictionary or probe
        self.vision_predictions: Dict[str, float] = {}

        # Cascade Decision Engine
        self.cascade_engine = TwoTierScreeningCascade(
            target_sensitivity=0.88,
            target_specificity=0.90,
        )
        self.cascade_engine.theta_triage = tier1_threshold
        self.cascade_engine.theta_referral = tier2_threshold

        self.is_fitted: bool = False
        self._df_train: Optional[pd.DataFrame] = None
        self._df_all_students: Optional[pd.DataFrame] = None

    def fit(self, df_train: Optional[pd.DataFrame] = None) -> DysgraphiaScreeningPipeline:
        """Fit all stream models on the training cohort (School A)."""
        logger.info("Initializing unified Dysgraphia Screening Pipeline...")

        if df_train is None:
            logger.info("Assembling multi-stream training dataset from School A...")
            df_train_raw, df_test_raw = assemble_multistream_student_dataset(use_cached_ocr_only=True)
            self._df_train = df_train_raw
            self._df_all_students = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
        else:
            self._df_train = df_train.copy()

        y_train = self._df_train["label"].values.astype(int)

        # 1. Fit Stream 1: Task-Aware Router (Geometry & Kinematics)
        logger.info("Fitting Stream 1 (Geometric Task-Aware Router)...")
        self.s1_router = TaskAwareRouter(random_state=self.random_state)
        self.s1_router.fit(self._df_train)

        # 2. Fit Stream 2: Stroke-Width & Pencil Pressure Micro-Texture
        logger.info("Fitting Stream 2 (Stroke-Width & Pressure Texture)...")
        self.s2_features = [c for c in STREAM2_CORE_FEATURES if c in self._df_train.columns]
        s2_train_df = self._df_train[self.s2_features].copy()
        self.s2_medians = s2_train_df.median().fillna(0.0).to_dict()
        X_train_s2 = s2_train_df.fillna(self.s2_medians).fillna(0.0).values

        self.pipe_s2 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=self.random_state)),
        ])
        self.pipe_s2.fit(X_train_s2, y_train)

        # 3. Fit Stream 3: TrOCR Legibility & Letter Error Alignment
        logger.info("Fitting Stream 3 (TrOCR Legibility & Error Analytics)...")
        self.s3_features = [c for c in STREAM3_CORE_FEATURES if c in self._df_train.columns]
        s3_train_df = self._df_train[self.s3_features].copy()
        self.s3_medians = s3_train_df.median().fillna(0.0).to_dict()
        X_train_s3 = s3_train_df.fillna(self.s3_medians).fillna(0.0).values

        self.pipe_s3 = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", random_state=self.random_state)),
        ])
        self.pipe_s3.fit(X_train_s3, y_train)

        # 4. Load Stream 4: Vision Predictions / Probe
        logger.info("Loading Stream 4 (Deep DenseNet201 Visual Patch Embeddings)...")
        pv_json = WORKSPACE_ROOT / "data" / "features" / "p_vision_predictions.json"
        if pv_json.exists():
            try:
                with open(pv_json, "r", encoding="utf-8") as f:
                    self.vision_predictions = json.load(f)
                logger.info(f"Loaded {len(self.vision_predictions)} visual patch predictions.")
            except Exception as e:
                logger.warning(f"Failed to load visual predictions from {pv_json}: {e}")
                self.vision_predictions = {}
        else:
            logger.info("p_vision_predictions.json not found; running with 3-stream late stacking.")
            self.vision_predictions = {}

        self.is_fitted = True
        logger.info("Unified Dysgraphia Screening Pipeline fitted successfully.")
        return self

    def get_grade_weights(
        self,
        grade: int,
        available_streams: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute normalized grade-adaptive weights across available streams."""
        raw_weights = self.GRADE_ADAPTIVE_WEIGHTS.get(grade, self.DEFAULT_WEIGHTS).copy()

        if available_streams is not None:
            # Zero out any stream not present
            for stream in list(raw_weights.keys()):
                if stream not in available_streams:
                    raw_weights[stream] = 0.0

        total = sum(raw_weights.values())
        if total > 0.0:
            return {k: v / total for k, v in raw_weights.items()}
        # If all weights 0, fall back to equal weighting among available
        if available_streams:
            eq = 1.0 / len(available_streams)
            return {s: eq for s in available_streams}
        return self.DEFAULT_WEIGHTS.copy()

    def screen_features(
        self,
        features_dict: Union[Dict[str, Any], pd.Series],
        grade: int,
        student_id: str = "UNKNOWN_STUDENT",
        school: Optional[str] = None,
        label: Optional[int] = None,
        p_vision: Optional[float] = None,
    ) -> ScreeningResult:
        """Screen a student from their extracted multi-stream feature record."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted via `fit()` before calling `screen_features()`.")

        df_row = pd.DataFrame([features_dict])
        if "grade" not in df_row.columns:
            df_row["grade"] = grade

        # 1. Stream 1: Geometry & Kinematics (Task-Aware Router)
        s1_prob = float(self.s1_router.predict_proba(df_row)[0])
        # Detect route taken
        try:
            route = self.s1_router.detect_route_from_features(df_row.iloc[0]).value
        except Exception:
            route = "full_protocol"

        # 2. Stream 2: Stroke-Width & Pencil Pressure Micro-Texture
        s2_row = df_row[[c for c in self.s2_features if c in df_row.columns]].copy()
        X_s2 = s2_row.fillna(self.s2_medians).fillna(0.0).values
        # Pad if missing columns
        if X_s2.shape[1] < len(self.s2_features):
            full_s2 = np.zeros((1, len(self.s2_features)))
            for j, c in enumerate(self.s2_features):
                full_s2[0, j] = df_row[c].values[0] if c in df_row.columns and pd.notna(df_row[c].values[0]) else self.s2_medians.get(c, 0.0)
            X_s2 = full_s2
        s2_prob = float(self.pipe_s2.predict_proba(X_s2)[0, 1])

        # 3. Stream 3: TrOCR Legibility & Letter Error Alignment
        s3_row = df_row[[c for c in self.s3_features if c in df_row.columns]].copy()
        X_s3 = s3_row.fillna(self.s3_medians).fillna(0.0).values
        if X_s3.shape[1] < len(self.s3_features):
            full_s3 = np.zeros((1, len(self.s3_features)))
            for j, c in enumerate(self.s3_features):
                full_s3[0, j] = df_row[c].values[0] if c in df_row.columns and pd.notna(df_row[c].values[0]) else self.s3_medians.get(c, 0.0)
            X_s3 = full_s3
        s3_prob = float(self.pipe_s3.predict_proba(X_s3)[0, 1])

        # 4. Stream 4: Deep Visual Patch Probe
        if p_vision is not None:
            s4_prob = float(p_vision)
            has_s4 = True
        elif student_id in self.vision_predictions:
            s4_prob = float(self.vision_predictions[student_id])
            has_s4 = True
        else:
            s4_prob = 0.50  # Uninformative prior
            has_s4 = False

        available_streams = ["stream_1_geometry", "stream_2_texture", "stream_3_ocr"]
        if has_s4:
            available_streams.append("stream_4_vision")

        # Grade-Adaptive Late Stacking Weights
        weights = self.get_grade_weights(grade, available_streams)

        p_final = (
            weights.get("stream_1_geometry", 0.0) * s1_prob
            + weights.get("stream_2_texture", 0.0) * s2_prob
            + weights.get("stream_3_ocr", 0.0) * s3_prob
            + weights.get("stream_4_vision", 0.0) * s4_prob
        )
        p_final = float(np.clip(p_final, 0.001, 0.999))

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
                "stream_4_vision": s4_prob,
            },
            stream_weights_used=weights,
            route_taken=route,
            screening_tier=tier,
            tier_label=tier_label,
            clinical_action=clinical_action,
            is_dysgraphia_suspected=is_suspected,
            key_risk_factors=risk_factors,
        )

    def screen_student(self, student_id: str) -> ScreeningResult:
        """Screen a specific student by student_id from cohort datasets."""
        if not self.is_fitted:
            self.fit()

        if self._df_all_students is None or student_id not in self._df_all_students["student_id"].values:
            # Try reloading multi-stream student dataset
            df_train_raw, df_test_raw = assemble_multistream_student_dataset(use_cached_ocr_only=True)
            self._df_all_students = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

        matches = self._df_all_students[self._df_all_students["student_id"] == student_id]
        if len(matches) == 0:
            raise KeyError(f"Student ID '{student_id}' not found in cohort datasets.")

        row = matches.iloc[0]
        grade = int(row.get("grade", 4))
        school = str(row.get("school", "unknown"))
        label = int(row["label"]) if "label" in row and pd.notna(row["label"]) else None
        p_vision = self.vision_predictions.get(student_id, None)

        return self.screen_features(
            features_dict=row,
            grade=grade,
            student_id=student_id,
            school=school,
            label=label,
            p_vision=p_vision,
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
                df_train_raw, df_test_raw = assemble_multistream_student_dataset(use_cached_ocr_only=True)
                self._df_all_students = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
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
            pv = self.vision_predictions.get(sid, None)

            res = self.screen_features(
                features_dict=row,
                grade=grade,
                student_id=sid,
                school=sch,
                label=lbl,
                p_vision=pv,
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
                "p_s4_vision": round(res.stream_probabilities.get("stream_4_vision", 0.0), 4),
                "is_dysgraphia_suspected": res.is_dysgraphia_suspected,
                "clinical_action": res.clinical_action,
            })

        df_out = pd.DataFrame(records)
        return df_out

    def _extract_risk_factors(self, feat: Union[Dict[str, Any], pd.Series]) -> List[Dict[str, Any]]:
        """Identify anomalous biomarkers explaining the screening decision."""
        factors = []
        # 1. Stroke fragmentation in dictation
        val = feat.get("components_per_unit_width_dict_mean", None)
        if val is not None and pd.notna(val) and float(val) > 2.5:
            factors.append({
                "domain": "Auditory-Motor Integration",
                "feature": "components_per_unit_width_dict_mean",
                "value": float(val),
                "note": "Severe stroke fragmentation and pen-stutter under dictation pacing",
            })

        # 2. Peak fragmentation episode
        val = feat.get("components_per_unit_width_max", None)
        if val is not None and pd.notna(val) and float(val) > 3.0:
            factors.append({
                "domain": "Episodic Motor Control",
                "feature": "components_per_unit_width_max",
                "value": float(val),
                "note": "Severe episodic motor breakdown with broken letter formations",
            })

        # 3. Stroke width jitter
        val = feat.get("stroke_width_jitter_mean", None)
        if val is not None and pd.notna(val) and float(val) > 0.45:
            factors.append({
                "domain": "Fine Motor Stability",
                "feature": "stroke_width_jitter_mean",
                "value": float(val),
                "note": "Irregular nib pressure and high stroke-width variance",
            })

        # 4. TrOCR character omission rate
        val = feat.get("char_omission_rate_mean", None)
        if val is not None and pd.notna(val) and float(val) > 0.15:
            factors.append({
                "domain": "Linguistic Transcription",
                "feature": "char_omission_rate_mean",
                "value": float(val),
                "note": "High rate of dropped letters and truncated words",
            })

        # 5. Spatial Confusable Reversals
        val = feat.get("confusable_rate_max", None)
        if val is not None and pd.notna(val) and float(val) > 0.05:
            factors.append({
                "domain": "Visuomotor Alignment",
                "feature": "confusable_rate_max",
                "value": float(val),
                "note": "Spatial letter reversal patterns (e.g. b/d, p/q, or Hindi virama)",
            })

        # 6. Headline Wobble
        val = feat.get("shirorekha_rms_deviation_norm_max", None)
        if val is not None and pd.notna(val) and float(val) > 0.20:
            factors.append({
                "domain": "Devanagari Mechanics",
                "feature": "shirorekha_rms_deviation_norm_max",
                "value": float(val),
                "note": "Severe Devanagari Shirorekha headline tremor and vertical tilt",
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
        help="Screen a specific student ID (e.g. SCHB_G4_Roll09)",
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

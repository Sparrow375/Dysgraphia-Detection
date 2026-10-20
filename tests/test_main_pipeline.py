"""Unit and Integration Tests for Unified Main Screening Pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
import pytest

from pipeline.main_pipeline import DysgraphiaScreeningPipeline, ScreeningResult, ScreeningTier


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Shared pipeline fixture fitted once for module tests."""
    pipe = DysgraphiaScreeningPipeline(tier1_threshold=0.276, tier2_threshold=0.515, random_state=42)
    pipe.fit()
    return pipe


def test_pipeline_initialization():
    """Verify clean pipeline state before fitting."""
    pipe = DysgraphiaScreeningPipeline()
    assert not pipe.is_fitted
    assert pipe.tier1_threshold == 0.276
    assert pipe.tier2_threshold == 0.515
    assert len(pipe.GRADE_ADAPTIVE_WEIGHTS) >= 6


def test_grade_adaptive_weights(fitted_pipeline):
    """Verify grade-adaptive weight logic across developmental stages."""
    # Grade 3: zero OCR, zero Vision
    w_g3 = fitted_pipeline.get_grade_weights(3, ["stream_1_geometry", "stream_2_texture", "stream_3_ocr", "stream_4_vision"])
    assert w_g3["stream_1_geometry"] == pytest.approx(0.60, abs=1e-3)
    assert w_g3["stream_2_texture"] == pytest.approx(0.40, abs=1e-3)
    assert w_g3["stream_3_ocr"] == pytest.approx(0.00, abs=1e-3)
    assert sum(w_g3.values()) == pytest.approx(1.0, abs=1e-5)

    # Grade 4: primary OCR (0.70)
    w_g4 = fitted_pipeline.get_grade_weights(4, ["stream_1_geometry", "stream_2_texture", "stream_3_ocr", "stream_4_vision"])
    assert w_g4["stream_3_ocr"] == pytest.approx(0.70, abs=1e-3)
    assert sum(w_g4.values()) == pytest.approx(1.0, abs=1e-5)

    # Renormalization when Stream 4 is unavailable
    w_g5_no_s4 = fitted_pipeline.get_grade_weights(5, ["stream_1_geometry", "stream_2_texture", "stream_3_ocr"])
    assert "stream_4_vision" not in w_g5_no_s4 or w_g5_no_s4.get("stream_4_vision", 0.0) == 0.0
    assert sum(w_g5_no_s4.values()) == pytest.approx(1.0, abs=1e-5)


def test_screen_real_student(fitted_pipeline):
    """Screen an existing verified student and verify clinical report output."""
    res = fitted_pipeline.screen_student("SCHB_G4_IMG_20261007_100013")
    assert isinstance(res, ScreeningResult)
    assert res.student_id == "SCHB_G4_IMG_20261007_100013"
    assert res.grade == 4
    assert 0.0 <= res.p_dysgraphia <= 1.0
    assert "stream_1_geometry" in res.stream_probabilities
    assert "stream_2_texture" in res.stream_probabilities
    assert "stream_3_ocr" in res.stream_probabilities
    assert res.screening_tier in [
        ScreeningTier.TIER_1_RULE_OUT,
        ScreeningTier.BORDERLINE_MONITORING,
        ScreeningTier.TIER_2_REFERRAL,
    ]
    summary_text = res.summary()
    assert "DYSGRAPHIA SCREENING REPORT" in summary_text
    assert "Multi-Stream Breakdown" in summary_text


def test_confirmed_control_vs_dysgraphic_screening(fitted_pipeline):
    """Verify tier assignment and risk probability on confirmed control vs dysgraphic cases."""
    # Real confirmed control student (Grade 3)
    res_control = fitted_pipeline.screen_student("SCHB_G3_IMG_20261007_101611")
    assert res_control.p_dysgraphia < fitted_pipeline.tier1_threshold
    assert res_control.screening_tier == ScreeningTier.TIER_1_RULE_OUT
    assert not res_control.is_dysgraphia_suspected

    # Real confirmed dysgraphic student (Grade 4)
    res_dys = fitted_pipeline.screen_student("SCHB_G4_IMG_20261007_100148")
    assert res_dys.p_dysgraphia >= fitted_pipeline.tier2_threshold
    assert res_dys.screening_tier == ScreeningTier.TIER_2_REFERRAL
    assert res_dys.is_dysgraphia_suspected
    assert res_dys.p_dysgraphia > res_control.p_dysgraphia
    assert len(res_dys.key_risk_factors) > 0


def test_synthetic_feature_screening(fitted_pipeline):
    """Verify that screen_features processes arbitrary feature dictionaries without crashing."""
    dummy_features = {
        "grade": 5,
        "components_per_unit_width_dict_mean": 2.2,
        "stroke_width_jitter_mean": 0.35,
        "ocr_confidence_mean_mean": 0.72,
    }
    res = fitted_pipeline.screen_features(dummy_features, grade=5, student_id="SYNTH_TEST")
    assert isinstance(res, ScreeningResult)
    assert 0.0 <= res.p_dysgraphia <= 1.0
    assert res.grade == 5


def test_cohort_screening_batch(fitted_pipeline):
    """Test batch cohort screening across School B subset."""
    df_cohort = fitted_pipeline.screen_cohort(school="school_b")
    assert isinstance(df_cohort, pd.DataFrame)
    assert len(df_cohort) == 100
    assert "p_dysgraphia" in df_cohort.columns
    assert "screening_tier" in df_cohort.columns
    assert "p_s1_geometry" in df_cohort.columns
    assert "p_s2_texture" in df_cohort.columns
    assert "p_s3_ocr" in df_cohort.columns
    assert "p_s4_vision" in df_cohort.columns
    assert set(df_cohort["screening_tier"].unique()).issubset({
        "TIER_1_RULE_OUT",
        "BORDERLINE_MONITORING",
        "TIER_2_REFERRAL",
    })

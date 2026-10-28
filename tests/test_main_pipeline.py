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
    pipe = DysgraphiaScreeningPipeline(tier1_threshold=0.250, tier2_threshold=0.450, random_state=42)
    pipe.fit()
    return pipe


def test_pipeline_initialization():
    """Verify clean pipeline state before fitting."""
    pipe = DysgraphiaScreeningPipeline()
    assert not pipe.is_fitted
    assert pipe.tier1_threshold == 0.250
    assert pipe.tier2_threshold == 0.450
    assert len(pipe.s1_features) == 14
    assert len(pipe.s2_features) == 7
    assert len(pipe.s3_features) == 5
    # Strict rule adherence: No per-grade stacking dictionaries
    assert not hasattr(pipe, "GRADE_ADAPTIVE_WEIGHTS")


def test_pipeline_meta_learner_unified(fitted_pipeline):
    """Verify single unified meta-learner model fitted across all 215 students."""
    assert fitted_pipeline.is_fitted
    assert fitted_pipeline.meta_learner is not None
    # Meta-learner has 4 inputs: [s1, s2, s3, is_pencil]
    assert fitted_pipeline.meta_learner.coef_.shape == (1, 4)
    # Stream weights are dynamically extracted from meta-learner coefficients
    assert fitted_pipeline.pipe_s1 is not None
    assert fitted_pipeline.pipe_s2 is not None
    assert fitted_pipeline.pipe_s3 is not None


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
    assert "Writing Instrument Detected" in summary_text


def test_confirmed_control_vs_dysgraphic_screening(fitted_pipeline):
    """Verify risk probability and ordering on confirmed control vs dysgraphic cases."""
    # Real confirmed control student (Grade 3)
    res_control = fitted_pipeline.screen_student("SCHB_G3_IMG_20261007_101611")
    assert 0.0 <= res_control.p_dysgraphia <= 1.0

    # Real confirmed dysgraphic student (Grade 4)
    res_dys = fitted_pipeline.screen_student("SCHB_G4_IMG_20261007_100148")
    assert 0.0 <= res_dys.p_dysgraphia <= 1.0
    assert res_dys.p_dysgraphia > res_control.p_dysgraphia


def test_synthetic_feature_screening(fitted_pipeline):
    """Verify that screen_features processes arbitrary task-agnostic feature dictionaries without crashing."""
    dummy_features = {
        "grade": 5,
        "is_pencil": 0,
        "components_per_unit_width_median": 1.2,
        "components_per_unit_width_p90": 2.8,
        "stroke_width_jitter_tool_norm": 0.42,
        "pressure_intensity_cv_tool_norm": 0.38,
        "trocr_char_entropy_mean": 0.51,
        "ink_to_char_density_ratio": 3.4,
    }
    res = fitted_pipeline.screen_features(dummy_features, grade=5, student_id="SYNTH_TEST")
    assert isinstance(res, ScreeningResult)
    assert 0.0 <= res.p_dysgraphia <= 1.0
    assert res.grade == 5
    assert res.is_pencil is False
    assert len(res.key_risk_factors) >= 3


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
    assert "is_pencil" in df_cohort.columns
    assert set(df_cohort["screening_tier"].unique()).issubset({
        "TIER_1_RULE_OUT",
        "BORDERLINE_MONITORING",
        "TIER_2_REFERRAL",
    })


def test_champion_dual_vit_screening(fitted_pipeline):
    """Test champion Dual-ViT screening and tile attribution integration."""
    res = fitted_pipeline.screen_student_champion("SCHB_G4_IMG_20261007_100013")
    assert isinstance(res, ScreeningResult)
    assert res.route_taken == "champion_dual_vit_consensus"
    assert res.p_dual_vit is not None
    assert 0.0 <= res.p_dual_vit <= 1.0
    assert res.p_dino is not None
    assert res.p_clip is not None
    summary_txt = res.summary()
    assert "Dual-ViT" in summary_txt
    assert "Rank Consensus" in summary_txt

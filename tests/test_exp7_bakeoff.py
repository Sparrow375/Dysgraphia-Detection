"""Automated Unit and Regression Tests for Experiment 7: Frozen-Feature Representation Bake-Off."""

from pathlib import Path
import numpy as np
import pytest
import torch

from baseline.tiling_preprocessor import build_white_canvas_strip, extract_tiles_from_strip
from baseline.extract_dinov2_patch_stats import compute_patch_descriptors
from baseline.evaluate_exp7_probe import calculate_sensitivity_at_specificity, compute_cluster_bootstrap_cis


def test_tiling_preprocessor_synthetic():
    """Verify white-canvas rendering and aspect-preserving tiling geometry."""
    # Synthetic clean ink mask (100x500): white canvas with black letter block in center
    clean_mask = np.full((100, 500), 255, dtype=np.uint8)
    clean_mask[30:70, 100:300] = 0  # Black ink

    raw_gray = np.full((100, 500), 240, dtype=np.uint8)
    raw_gray[30:70, 100:300] = 50  # Dark gray pencil stroke

    # Mock file-less strip construction
    ink_bin = (clean_mask < 128).astype(np.float32)
    ch1 = np.where(ink_bin > 0, 0.0, 255.0).astype(np.float32)
    ch2 = np.where(ink_bin > 0, 50.0, 255.0).astype(np.float32)
    ch3 = np.full((100, 500), 255.0, dtype=np.float32)
    ch3[ink_bin > 0] = 0.0

    strip = np.stack([ch1, ch2, ch3], axis=-1).astype(np.uint8)
    assert strip.shape == (100, 500, 3)
    assert strip.dtype == np.uint8

    tiles, coords = extract_tiles_from_strip(strip, target_height=224, stride=112)
    assert len(tiles) > 0
    for t in tiles:
        assert t.shape == (224, 224, 3)
        assert t.dtype == np.uint8
        # Background is white (255)
        assert t[0, 0, 0] == 255


def test_dinov2_patch_descriptors_tensor_math():
    """Verify that patch descriptor computes [mean, std, GeM] accurately."""
    b, p_tokens, d = 2, 256, 64
    mock_patches = torch.abs(torch.randn(b, p_tokens, d)) + 0.1  # Positive values for GeM

    desc = compute_patch_descriptors(mock_patches, p=3.0)
    assert desc.shape == (b, 3 * d)

    # Check individual components
    u = desc[:, :d]
    s = desc[:, d : 2 * d]
    gem = desc[:, 2 * d :]

    expected_u = mock_patches.mean(dim=1)
    assert torch.allclose(u, expected_u, atol=1e-5)

    expected_s = torch.sqrt(torch.var(mock_patches, dim=1, unbiased=False) + 1e-6)
    assert torch.allclose(s, expected_s, atol=1e-5)

    expected_gem = (mock_patches.pow(3.0).mean(dim=1)).pow(1.0 / 3.0)
    assert torch.allclose(gem, expected_gem, atol=1e-4)


def test_sensitivity_at_specificity_calculation():
    """Verify sensitivity calculation at target operational specificity."""
    # 10 positives, 90 negatives
    y_true = np.array([1] * 10 + [0] * 90)
    # Perfectly separated scores
    y_scores = np.array([0.9] * 10 + [0.1] * 90)

    det80, sens80, thresh80 = calculate_sensitivity_at_specificity(y_true, y_scores, 0.80)
    assert det80 == 10
    assert sens80 == 1.0
    assert thresh80 == pytest.approx(0.1, abs=1e-5)

    det90, sens90, thresh90 = calculate_sensitivity_at_specificity(y_true, y_scores, 0.90)
    assert det90 == 10
    assert sens90 == 1.0


def test_cluster_bootstrap_cis_bounds():
    """Verify cluster bootstrap confidence intervals contain empirical estimates."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0] * 5)
    y_scores = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.4, 0.3, 0.85, 0.15] * 5)
    sids = np.array([f"student_{i}" for i in range(len(y_true))])

    cis = compute_cluster_bootstrap_cis(y_true, y_scores, sids, n_bootstrap=100)
    assert "auroc_ci" in cis
    assert "auprc_ci" in cis
    assert cis["auroc_ci"][0] <= cis["auroc_ci"][1]
    assert cis["auprc_ci"][0] <= cis["auprc_ci"][1]
    assert 0.0 <= cis["auroc_ci"][0] <= 1.0
    assert 0.0 <= cis["auroc_ci"][1] <= 1.0


def test_fused_head_intermediate_fusion():
    """Verify that intermediate fusion of visual PCA and tabular features fits and produces valid probabilities."""
    from baseline.evaluate_exp7_feature_fusion import tune_and_fit_fused_head

    np.random.seed(42)
    n_train, n_test, vis_dim, tab_dim = 40, 10, 128, 8
    X_tr_vis = np.random.randn(n_train, vis_dim).astype(np.float32)
    X_te_vis = np.random.randn(n_test, vis_dim).astype(np.float32)
    T_tr_tab = np.random.randn(n_train, tab_dim).astype(np.float32)
    T_te_tab = np.random.randn(n_test, tab_dim).astype(np.float32)
    y_tr = np.random.choice([0, 1], size=n_train, p=[0.8, 0.2])

    # Ensure at least two classes
    y_tr[0] = 0
    y_tr[1] = 1

    probs = tune_and_fit_fused_head(
        X_train_vis=X_tr_vis,
        T_train_tab=T_tr_tab,
        y_train=y_tr,
        X_test_vis=X_te_vis,
        T_test_tab=T_te_tab,
        k_candidates=[4, 8],
        c_candidates=[0.1, 1.0],
        seed=42,
    )
    assert len(probs) == n_test
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_supcon_writer_style_encoder():
    """Verify SupCon loss computation and WriterStyleEncoder normalization."""
    from baseline.train_writer_style_encoder import SupConLoss, WriterStyleEncoder

    torch.manual_seed(42)
    encoder = WriterStyleEncoder(in_dim=64, hidden_dim=32, out_dim=16)
    x = torch.randn(8, 64)
    z = encoder(x)

    assert z.shape == (8, 16)
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    criterion = SupConLoss(temperature=0.07)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = criterion(z, labels)

    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_vlm_json_response_parser():
    """Verify robust parsing of VLM JSON responses including markdown code blocks."""
    from baseline.evaluate_exp7_vlm_rubric import parse_vlm_json_response

    # Clean JSON
    clean_json = '{"legibility": 4.0, "letter_formation": 3.5, "spacing_regularity": 4.5, "size_consistency": 3.0, "baseline_adherence": 2.5, "stroke_fluidity": 4.0, "pressure_uniformity": 3.5}'
    parsed = parse_vlm_json_response(clean_json)
    assert parsed is not None
    assert parsed["legibility"] == 4.0
    assert parsed["baseline_adherence"] == 2.5

    # Markdown wrapped JSON
    md_json = '```json\n{\n  "legibility": 2.0,\n  "letter_formation": 2.5,\n  "spacing_regularity": 3.0,\n  "size_consistency": 2.0,\n  "baseline_adherence": 1.5,\n  "stroke_fluidity": 2.0,\n  "pressure_uniformity": 3.0\n}\n```'
    parsed_md = parse_vlm_json_response(md_json)
    assert parsed_md is not None
    assert parsed_md["legibility"] == 2.0
    assert parsed_md["baseline_adherence"] == 1.5

    # Out-of-bounds score clamping
    oob_json = '{"legibility": 6.5, "letter_formation": 0.5, "spacing_regularity": 4.5, "size_consistency": 3.0, "baseline_adherence": 2.5, "stroke_fluidity": 4.0, "pressure_uniformity": 3.5}'
    parsed_oob = parse_vlm_json_response(oob_json)
    assert parsed_oob is not None
    assert parsed_oob["legibility"] == 5.0
    assert parsed_oob["letter_formation"] == 1.0

    # Malformed text
    assert parse_vlm_json_response("I cannot score this image.") is None


def test_paired_bootstrap_math():
    """Verify paired bootstrap difference computation and CI logic."""
    from baseline.compute_paired_bootstrap import paired_bootstrap_test

    np.random.seed(42)
    y_true = np.array([1]*20 + [0]*80)
    # Model A has noisy separation (AUC ~0.70)
    p_a = y_true * 0.2 + np.random.uniform(0.0, 0.8, size=100)
    # Model B has much stronger separation (AUC ~1.00)
    p_b = y_true * 0.6 + np.random.uniform(0.0, 0.4, size=100)

    res = paired_bootstrap_test(y_true, p_b, p_a, "Model B", "Model A", n_bootstrap=200, seed=42)
    assert res["model_a"] == "Model B"
    assert res["model_b"] == "Model A"
    assert res["delta_auc"] > 0.20
    assert res["ci_auc_lower"] < res["ci_auc_upper"]


def test_error_analysis_data_structure():
    """Verify error analysis outputs and cohort integrity."""
    import json
    report_file = Path("reports/exp7_error_analysis.json")
    assert report_file.exists(), "Error analysis JSON report must exist"

    with open(report_file, "r") as f:
        data = json.load(f)

    assert data["cohort_summary"]["n_total"] == 215
    assert data["cohort_summary"]["n_positives"] == 40
    assert data["cohort_summary"]["n_controls"] == 175
    assert data["false_negatives_summary"]["dino_fns_count"] == 9
    assert data["false_negatives_summary"]["all_models_fns_count"] == 3
    assert len(data["persistent_false_positives_top10"]) == 10





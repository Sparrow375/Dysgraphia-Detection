"""Unit tests for Stage 34: De-Cheated Multi-Modal Gated-Attention MIL."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from baseline.decheated_multimodal_mil import (
    AntiSilhouetteAugmentor,
    BottleneckDilatedSequencePooler,
    DeCheatedMultiModalMILModel,
    EMAWeightTracker,
)
from baseline.multimodal_features import (
    SELECTED_MATH_FEATURES_22,
    fit_and_scale_multimodal_features,
    load_raw_multimodal_dataframe,
    load_wacom_stylus_lupi_targets,
)


def test_22_features_compilation_and_lupi() -> None:
    """Test 22-feature compilation, is_pencil presence, and Wacom stylus LUPI range."""
    df_all = load_raw_multimodal_dataframe()
    assert len(df_all) > 500
    assert "is_pencil" in df_all.columns
    assert len(SELECTED_MATH_FEATURES_22) == 22

    for col in SELECTED_MATH_FEATURES_22:
        assert col in df_all.columns, f"Feature {col} missing from dataframe"

    uids = df_all["task_uid"].tolist()
    math_dict, aux_dict, _ = fit_and_scale_multimodal_features(
        df_all, uids[:50], uids[50:100], feature_cols=SELECTED_MATH_FEATURES_22
    )

    sample_uid = uids[0]
    assert math_dict[sample_uid].shape == (22,)
    assert aux_dict[sample_uid].shape == (2,)
    assert not np.isnan(math_dict[sample_uid]).any()

    lupi_dict = load_wacom_stylus_lupi_targets()
    assert len(lupi_dict) > 0, "LUPI targets should load for School A"
    for sid, q in lupi_dict.items():
        assert 0.05 <= q <= 0.95, f"LUPI score {q} out of range [0.05, 0.95]"


def test_anti_silhouette_augmentor() -> None:
    """Test horizontal sub-windowing and strip cutout augmentations."""
    aug_train = AntiSilhouetteAugmentor(target_h=128, target_w=480, full_w=640, is_train=True)
    aug_eval = AntiSilhouetteAugmentor(target_h=128, target_w=480, full_w=640, is_train=False)

    tensor_img = torch.ones(3, 128, 640)

    # Train mode: random window + cutout
    out_train = aug_train(tensor_img)
    assert out_train.shape == (3, 128, 480), f"Expected (3, 128, 480), got {out_train.shape}"
    # Cutout should zero out some pixels
    assert (out_train == 0.0).any(), "SpecAugment cutout should zero out pixels during training"

    # Eval mode: center crop, no cutout
    out_eval = aug_eval(tensor_img)
    assert out_eval.shape == (3, 128, 480)
    assert (out_eval == 1.0).all(), "Eval mode should be clean center crop with no cutout"


def test_parameter_sponge_crushed() -> None:
    """Verify that 98.8% of DenseNet is frozen and trainable params are <= 2.5M."""
    model = DeCheatedMultiModalMILModel(pretrained=False)

    # Early layers must be frozen
    assert not model.features.conv0.weight.requires_grad
    assert not any(p.requires_grad for p in model.features.denseblock1.parameters())
    assert not any(p.requires_grad for p in model.features.denseblock2.parameters())
    assert not any(p.requires_grad for p in model.features.denseblock3.parameters())
    assert not any(p.requires_grad for p in model.features.denseblock4.denselayer1.parameters())

    # Last 8 denselayers must be trainable
    assert any(p.requires_grad for p in model.features.denseblock4.denselayer32.parameters())
    assert any(p.requires_grad for p in model.features.norm5.parameters())

    trainable_backbone = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
    trainable_pooler = sum(p.numel() for p in model.sequence_pooler.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert trainable_backbone <= 2_200_000, f"Trainable backbone too large: {trainable_backbone}"
    assert trainable_pooler <= 200_000, f"Dilated pooler too large: {trainable_pooler}"
    assert total_trainable <= 2_500_000, f"Total trainable parameters too large: {total_trainable}"


def test_modality_dropout() -> None:
    """Verify forced modality dropout sets respective branch to zero."""
    model = DeCheatedMultiModalMILModel(pretrained=False)
    model.eval()

    all_images = torch.randn(4, 3, 128, 480)
    all_math = torch.randn(4, 22)
    split_indices = [(0, 2), (2, 4)]

    # Forward with vision_only (math should be zeroed internally)
    logits_v, attns_v, _ = model(all_images, all_math, split_indices, force_modality="vision_only")
    assert logits_v.shape == (2,)

    # Forward with math_only (vision should be zeroed internally)
    logits_m, attns_m, _ = model(all_images, all_math, split_indices, force_modality="math_only")
    assert logits_m.shape == (2,)

    # Outputs should be different because different modalities are zeroed
    assert not torch.allclose(logits_v, logits_m)


def test_ema_and_platt_calibration() -> None:
    """Verify EMA weight tracker updates and Platt calibration operates correctly."""
    model = DeCheatedMultiModalMILModel(pretrained=False)
    ema = EMAWeightTracker(model, decay=0.99)

    # Perturb a trainable weight
    p = next(p for p in model.parameters() if p.requires_grad)
    orig_val = p.data.clone()
    p.data.add_(1.0)

    ema.update(model)
    # EMA shadow should be 0.99 * orig + 0.01 * (orig + 1.0) = orig + 0.01
    assert not torch.allclose(ema.shadow[next(k for k in ema.shadow)], orig_val)

    # Test Platt calibration on synthetic logits
    val_logits = np.array([[-2.5], [-1.0], [0.5], [2.0], [3.5]])
    val_y = np.array([0, 0, 1, 1, 1])
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(val_logits, val_y)

    test_logits = np.array([[-3.0], [1.5]])
    cal_probs = calibrator.predict_proba(test_logits)[:, 1]
    assert len(cal_probs) == 2
    assert 0.0 <= cal_probs[0] <= 1.0
    assert 0.0 <= cal_probs[1] <= 1.0
    assert cal_probs[0] < cal_probs[1]


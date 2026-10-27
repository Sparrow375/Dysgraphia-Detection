"""Unit tests for Stage 33: Multi-Modal Gated-Attention MIL."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from baseline.multimodal_features import (
    SELECTED_MATH_FEATURES,
    fit_and_scale_multimodal_features,
    load_raw_multimodal_dataframe,
)
from baseline.train_multimodal_mil import (
    HorizontalSequencePooler,
    MultiModalWholePageMILModel,
)


def test_multimodal_features_loading_and_scaling() -> None:
    """Test loading dataframe and fitting leak-free feature dictionaries."""
    df_all = load_raw_multimodal_dataframe()
    assert len(df_all) > 500, f"Expected >500 rows, got {len(df_all)}"
    assert "trocr_mean_confidence" in df_all.columns
    assert "trocr_low_conf_ratio" in df_all.columns

    for col in SELECTED_MATH_FEATURES:
        assert col in df_all.columns, f"Missing feature {col}"

    uids = df_all["task_uid"].tolist()
    train_uids = uids[:50]
    test_uids = uids[50:70]

    math_dict, aux_dict, stats = fit_and_scale_multimodal_features(df_all, train_uids, test_uids)

    # Check sample keys
    sample_uid = uids[0]
    assert sample_uid in math_dict
    assert sample_uid in aux_dict

    m_vec = math_dict[sample_uid]
    a_vec = aux_dict[sample_uid]

    assert m_vec.shape == (16,), f"Expected shape (16,), got {m_vec.shape}"
    assert a_vec.shape == (2,), f"Expected shape (2,), got {a_vec.shape}"
    assert not np.isnan(m_vec).any(), "NaN found in math vector"
    assert not np.isnan(a_vec).any(), "NaN found in aux vector"
    assert not np.isinf(m_vec).any(), "Inf found in math vector"
    assert not np.isinf(a_vec).any(), "Inf found in aux vector"


def test_horizontal_sequence_pooler() -> None:
    """Test 1D sequence pooler across horizontal bins."""
    pooler = HorizontalSequencePooler(in_channels=1920, seq_channels=64, out_dim=32)
    pooler.eval()

    # Synthetic DenseNet201 feature map on (128, 640) crop: (B, 1920, 4, 20)
    feat_map = torch.randn(4, 1920, 4, 20, requires_grad=True)
    out = pooler(feat_map)

    assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"

    loss = out.sum()
    loss.backward()
    assert feat_map.grad is not None
    assert not torch.isnan(feat_map.grad).any()


def test_partial_backbone_freezing() -> None:
    """Verify that conv0 through denseblock2 are frozen, while denseblock3-4 remain trainable."""
    model = MultiModalWholePageMILModel(pretrained=False)

    # conv0 should be frozen
    assert not model.features.conv0.weight.requires_grad

    # denseblock1 and denseblock2 should be frozen
    for p in model.features.denseblock1.parameters():
        assert not p.requires_grad
    for p in model.features.denseblock2.parameters():
        assert not p.requires_grad

    # denseblock3 and denseblock4 should be trainable
    db3_trainable = any(p.requires_grad for p in model.features.denseblock3.parameters())
    db4_trainable = any(p.requires_grad for p in model.features.denseblock4.parameters())
    assert db3_trainable, "denseblock3 must be trainable"
    assert db4_trainable, "denseblock4 must be trainable"


def test_multimodal_whole_page_mil_forward_backward() -> None:
    """Verify forward and multi-task backward pass through joint vision + math MIL model."""
    model = MultiModalWholePageMILModel(pretrained=False)
    model.train()

    # 2 students: student 0 has 3 crops, student 1 has 2 crops (total = 5 crops)
    all_images = torch.randn(5, 3, 128, 640)
    all_math = torch.randn(5, 16)
    split_indices = [(0, 3), (3, 5)]
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    aux_targets = torch.tensor([
        [0.85, 0.12],
        [0.72, 0.25],
        [0.60, 0.40],
        [0.91, 0.08],
        [0.88, 0.10],
    ], dtype=torch.float32)

    student_logits, attentions, aux_preds = model(all_images, all_math, split_indices)

    # Verify forward output dimensions
    assert student_logits.shape == (2,), f"Expected student_logits (2,), got {student_logits.shape}"
    assert len(attentions) == 2, f"Expected 2 attention sets, got {len(attentions)}"
    assert attentions[0].shape == (3,), f"Expected attention (3,), got {attentions[0].shape}"
    assert attentions[1].shape == (2,), f"Expected attention (2,), got {attentions[1].shape}"
    assert torch.allclose(attentions[0].sum(), torch.tensor(1.0), atol=1e-5), "Attn weights must sum to 1.0"
    assert torch.allclose(attentions[1].sum(), torch.tensor(1.0), atol=1e-5), "Attn weights must sum to 1.0"
    assert aux_preds.shape == (5, 2), f"Expected aux_preds (5, 2), got {aux_preds.shape}"

    # Verify multi-task loss computation and backward pass
    loss_student = F.binary_cross_entropy_with_logits(student_logits, labels)
    loss_aux = F.mse_loss(aux_preds, aux_targets)
    loss_total = loss_student + 0.3 * loss_aux

    loss_total.backward()

    # Verify gradient updates
    assert model.sequence_pooler.conv1d[0].weight.grad is not None
    assert model.aux_head[0].weight.grad is not None
    assert model.math_proj[0].weight.grad is not None
    assert model.attention_V.weight.grad is not None
    assert model.attention_U.weight.grad is not None
    assert model.attention_w.weight.grad is not None
    assert model.classifier[1].weight.grad is not None

    # Check frozen vs trainable backbone gradients
    assert model.features.conv0.weight.grad is None
    assert model.features.denseblock4.denselayer1.conv1.weight.grad is not None

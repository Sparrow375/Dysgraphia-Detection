"""Unit tests for Gated Attention MIL Model and Dataset Collator."""

import pytest
import torch
import torch.nn as nn
from models.gated_attention_mil import (
    GatedAttention,
    GatedAttentionMIL,
    StudentBagDataset,
    collate_student_bags,
)


def test_gated_attention_forward_shape():
    """Verify output shapes and probability simplex properties of Gated Attention."""
    batch_size = 4
    num_instances = 25
    in_features = 64
    attention_dim = 4

    attention = GatedAttention(in_features=in_features, attention_dim=attention_dim)
    h = torch.randn(batch_size, num_instances, in_features)

    z, weights = attention(h)

    # Check shapes
    assert z.shape == (batch_size, in_features), f"Expected {(batch_size, in_features)}, got {z.shape}"
    assert weights.shape == (batch_size, num_instances), f"Expected {(batch_size, num_instances)}, got {weights.shape}"

    # Attention weights must form a valid probability distribution: non-negative and sum to 1.0
    assert torch.all(weights >= 0.0), "Attention weights must be non-negative"
    sums = torch.sum(weights, dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), "Attention weights must sum to 1.0"


def test_gated_attention_masking():
    """Verify masked (padded) word positions receive zero attention weight."""
    batch_size = 2
    num_instances = 10
    in_features = 32

    attention = GatedAttention(in_features=in_features, attention_dim=4)
    h = torch.randn(batch_size, num_instances, in_features)

    # Student 0 has 6 words; Student 1 has 3 words
    mask = torch.zeros(batch_size, num_instances, dtype=torch.bool)
    mask[0, :6] = True
    mask[1, :3] = True

    z, weights = attention(h, mask=mask)

    # Padded positions must have ~0 attention
    assert torch.all(weights[0, 6:] < 1e-5), "Masked positions in student 0 must be zero"
    assert torch.all(weights[1, 3:] < 1e-5), "Masked positions in student 1 must be zero"

    # Valid positions must sum to 1.0
    assert torch.allclose(torch.sum(weights[0, :6]), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(torch.sum(weights[1, :3]), torch.tensor(1.0), atol=1e-5)


def test_parameter_count_limit():
    """Strictly enforce that trainable parameters do not exceed 1,000 to prevent overfitting."""
    # When projected features d=64, attention_dim=4
    model = GatedAttentionMIL(
        encoder_dim=64,
        projected_dim=64,
        attention_dim=4,
        dropout=0.1,
    )
    n_params = model.count_trainable_parameters()
    # V: 4*64=256, U: 4*64=256, w: 4*1=4, cls: 64*1+1=65 -> total = 581
    assert n_params <= 1000, f"Trainable parameter count ({n_params}) exceeded 1,000 limit"
    assert n_params > 0, "Model must have trainable parameters"


def test_student_bag_collation():
    """Verify collator handles variable-sized student bags cleanly."""
    feature_dim = 64
    bag1 = torch.randn(15, feature_dim)  # Student 1: 15 words
    bag2 = torch.randn(72, feature_dim)  # Student 2: 72 words
    bag3 = torch.randn(30, feature_dim)  # Student 3: 30 words

    dataset = StudentBagDataset(
        student_bags=[bag1, bag2, bag3],
        labels=[0, 1, 0],
        student_ids=["S1", "S2", "S3"],
    )

    batch = [dataset[0], dataset[1], dataset[2]]
    collated = collate_student_bags(batch)

    assert collated["features"].shape == (3, 72, feature_dim)
    assert collated["mask"].shape == (3, 72)
    assert collated["labels"].shape == (3,)
    assert collated["student_ids"] == ["S1", "S2", "S3"]

    # Verify mask counts match bag sizes
    assert collated["mask"][0].sum().item() == 15
    assert collated["mask"][1].sum().item() == 72
    assert collated["mask"][2].sum().item() == 30


def test_mil_gradient_flow():
    """Verify backward gradients propagate cleanly without NaN."""
    model = GatedAttentionMIL(encoder_dim=64, projected_dim=64, attention_dim=4)
    h = torch.randn(2, 20, 64)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 15:] = False

    logits, attn = model(h, mask=mask)
    labels = torch.tensor([1.0, 0.0])

    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    # Check all parameter gradients are finite
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient detected in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient detected in {name}"

"""Gated-Attention Multiple Instance Learning (MIL) for Dysgraphia Detection.

Implements Ilse et al. (ICML 2018) Gated-Attention pooling over handwriting
word instances from frozen Vision Transformer (ViT) encoders:
- Low-parameter attention layer (~250-600 trainable parameters total)
- Zooms in on the most dysgraphic/malformed words on a student's submission
- Eliminates page averaging without visual overfitting
- Outputs out-of-fold student risk probabilities P_student in [0, 1]
- Outputs interpretable attention weights a_k for every word crop on the page.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class GatedAttention(nn.Module):
    """Ilse et al. (ICML 2018) Gated Attention Pooling Layer.
    
    Computes dynamic attention weights across a bag of K instances:
    a_k = softmax(w^T (tanh(V h_k) ⊙ sigmoid(U h_k)))
    """

    def __init__(self, in_features: int = 64, attention_dim: int = 4):
        super().__init__()
        self.in_features = in_features
        self.attention_dim = attention_dim

        # V: non-linear feature transformation
        self.attention_v = nn.Linear(in_features, attention_dim, bias=False)
        # U: gating mechanism (controls gradient flow and instance filtering)
        self.attention_u = nn.Linear(in_features, attention_dim, bias=False)
        # w: project to scalar score
        self.attention_w = nn.Linear(attention_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.attention_v.weight)
        nn.init.xavier_uniform_(self.attention_u.weight)
        nn.init.xavier_uniform_(self.attention_w.weight)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-pooled bag representation and attention distribution.
        
        Args:
            h: Instance feature tensor of shape (batch_size, num_instances, in_features)
            mask: Optional boolean mask of shape (batch_size, num_instances),
                  where True indicates a valid word instance and False indicates padding.
                  
        Returns:
            z: Pooled bag representation of shape (batch_size, in_features)
            weights: Attention weights of shape (batch_size, num_instances), sum(weights, dim=-1) == 1.0
        """
        # (batch_size, num_instances, attention_dim)
        v = torch.tanh(self.attention_v(h))
        u = torch.sigmoid(self.attention_u(h))

        # Gated attention scores: (batch_size, num_instances, 1)
        scores = self.attention_w(v * u)

        if mask is not None:
            # Mask padded positions with large negative value prior to softmax
            scores = scores.squeeze(-1) # (batch_size, num_instances)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=-1) # (batch_size, num_instances)
            weights = weights.unsqueeze(-1) # (batch_size, num_instances, 1)
        else:
            weights = torch.softmax(scores, dim=1)

        # Weighted pooling: z = sum_{k=1}^K a_k * h_k
        z = torch.sum(weights * h, dim=1) # (batch_size, in_features)

        return z, weights.squeeze(-1)


class GatedAttentionMIL(nn.Module):
    """Multiple Instance Learning Network with Gated Attention Pooling.
    
    Takes variable-sized bags of frozen ViT word embeddings and predicts
    student-level dysgraphia risk with strictly constrained parameter capacity.
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        projected_dim: int = 64,
        attention_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.projected_dim = projected_dim
        self.attention_dim = attention_dim

        # Optional linear projection if input feature dim != projected_dim
        if encoder_dim != projected_dim:
            self.projection = nn.Sequential(
                nn.Linear(encoder_dim, projected_dim),
                nn.LayerNorm(projected_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.projection = nn.Identity()

        # Gated Attention module (~250-500 parameters)
        self.attention = GatedAttention(
            in_features=projected_dim,
            attention_dim=attention_dim,
        )

        # Linear classification head (projected_dim -> 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(projected_dim, 1),
        )

        self._init_classifier()

    def _init_classifier(self) -> None:
        if isinstance(self.classifier[-1], nn.Linear):
            nn.init.xavier_uniform_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Gated Attention MIL.
        
        Args:
            h: Tensor of shape (batch_size, num_instances, encoder_dim)
            mask: Optional boolean mask of shape (batch_size, num_instances)
            
        Returns:
            logits: Logit tensor of shape (batch_size,)
            attention_weights: Tensor of shape (batch_size, num_instances)
        """
        # Map to low-rank representation
        proj = self.projection(h)

        # Attention pooling
        z, attn_weights = self.attention(proj, mask=mask)

        # Classification logits
        logits = self.classifier(z).squeeze(-1)

        return logits, attn_weights

    def predict_proba(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute calibrated probabilities in [0, 1] and attention weights."""
        logits, attn_weights = self.forward(h, mask=mask)
        probs = torch.sigmoid(logits)
        return probs, attn_weights

    def count_trainable_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StudentBagDataset(Dataset):
    """Dataset storing precomputed word embeddings grouped by student."""

    def __init__(
        self,
        student_bags: List[torch.Tensor],
        labels: List[int],
        student_ids: List[str],
        word_metadata: Optional[List[List[Dict[str, Any]]]] = None,
    ):
        """
        Args:
            student_bags: List of tensors, each of shape (K_i, feature_dim)
            labels: List of binary student labels (0 or 1)
            student_ids: List of student identifier strings
            word_metadata: Optional metadata dict for each word instance
        """
        assert len(student_bags) == len(labels) == len(student_ids)
        self.student_bags = student_bags
        self.labels = labels
        self.student_ids = student_ids
        self.word_metadata = word_metadata

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        bag = self.student_bags[idx]
        label = self.labels[idx]
        sid = self.student_ids[idx]
        item: Dict[str, Any] = {
            "features": bag,
            "label": torch.tensor(label, dtype=torch.float32),
            "student_id": sid,
            "num_words": bag.shape[0],
        }
        if self.word_metadata is not None:
            item["word_metadata"] = self.word_metadata[idx]
        return item


def collate_student_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length bags into a padded tensor with a boolean mask.
    
    Args:
        batch: List of items from StudentBagDataset
        
    Returns:
        padded_features: (batch_size, max_words, feature_dim)
        mask: (batch_size, max_words) boolean mask, True for valid words
        labels: (batch_size,) float tensor
        student_ids: List of student IDs
    """
    batch_size = len(batch)
    max_words = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]

    padded_features = torch.zeros((batch_size, max_words, feature_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.float32)
    student_ids: List[str] = []

    for i, item in enumerate(batch):
        bag = item["features"]
        k = bag.shape[0]
        padded_features[i, :k, :] = bag
        mask[i, :k] = True
        labels[i] = item["label"]
        student_ids.append(item["student_id"])

    return {
        "features": padded_features,
        "mask": mask,
        "labels": labels,
        "student_ids": student_ids,
    }

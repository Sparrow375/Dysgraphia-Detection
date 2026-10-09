"""
HandwritingTransformerOCR — In-House Vision Transformer for Handwriting Recognition.
Designed for IAM Word/Line Datasets & NIST SD19 Character-Level Pretraining.

Architecture:
  1. Patch/Convolutional Feature Stem: Converts (B, 1, H=64, W=256) -> (B, D=256, H'=1, W'=64)
  2. Sequence Projection & 1D Learnable Positional Embeddings
  3. Bidirectional Multi-Head Self-Attention (MHSA) Transformer Encoder (4-6 Layers)
  4. Linear CTC Projection Head (maps to Vocabulary + CTC Blank)
  5. CTC Greedy & Beam-Search Decoders with Character Attention Extraction
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Default Character Alphabet (IAM + NIST compatible)
# ---------------------------------------------------------------------------
DEFAULT_ALPHABET = (
    " abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?'\"-()/"
)


class ConvFeatureStem(nn.Module):
    """
    Downsamples 2D image height to 1 while preserving horizontal sequence length.
    Transforms (B, 1, 64, W) -> (B, embed_dim, 1, W // 4).
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # Stage 1: (B, 1, 64, W) -> (B, 64, 32, W // 2)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: (B, 64, 32, W // 2) -> (B, 128, 16, W // 4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: (B, 128, 16, W // 4) -> (B, 256, 4, W // 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),

            # Stage 4: Height compression (B, 256, 4, W // 4) -> (B, embed_dim, 1, W // 4)
            nn.Conv2d(256, embed_dim, kernel_size=(4, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1, 64, W) -> Output: (B, embed_dim, 1, W // 4)
        return self.net(x)


class TransformerOCRBlock(nn.Module):
    """
    Standard Pre-LN Transformer Encoder Block for sequence processing.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(
            norm_x, norm_x, norm_x,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class HandwritingTransformerOCR(nn.Module):
    """
    Vision Transformer OCR Model for Word & Line-level Handwriting Recognition.
    
    Trained via Connectionist Temporal Classification (CTC) loss.
    Compatible with IAM Words, IAM Lines, and NIST synthetic words.
    """

    def __init__(
        self,
        alphabet: str = DEFAULT_ALPHABET,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.alphabet = alphabet
        self.blank_idx = 0  # CTC blank token is index 0
        self.num_classes = len(alphabet) + 1  # alphabet + blank
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 1. Convolutional Tokenizer Stem
        self.stem = ConvFeatureStem(in_channels=1, embed_dim=embed_dim)

        # 2. Learnable 1D Horizontal Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder Stack
        self.blocks = nn.ModuleList([
            TransformerOCRBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4. CTC Classification Head
        self.head = nn.Linear(embed_dim, self.num_classes)

        # 5. Dual-Task Dysgraphia Hesitation / Tremor Head
        # Quantifies token-level entropy/uncertainty to help flag motor dysgraphia
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            x: (B, 1, 64, W) handwriting images normalized to [0, 1].
            return_attention: If True, returns attention maps for character alignment inspection.

        Returns:
            Dict containing:
              - 'log_probs': (T, B, num_classes) for PyTorch CTC loss.
              - 'logits': (B, T, num_classes)
              - 'uncertainty': (B, T, 1) token-level hesitation metrics
              - 'attention_maps': Optional list of attention tensors
        """
        B, C, H, W = x.shape

        # 1. Convolutional patch feature extraction
        feat = self.stem(x)  # (B, embed_dim, 1, W // 4)
        feat = feat.squeeze(2).permute(0, 2, 1)  # (B, T, embed_dim) where T = W // 4
        T = feat.shape[1]

        # 2. Add Positional Embedding (interpolated if T > max_seq_len)
        if T <= self.max_seq_len:
            pos = self.pos_embed[:, :T, :]
        else:
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=T,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        x_seq = self.pos_dropout(feat + pos)

        # 3. Transformer Encoder Layers
        attn_maps = []
        for block in self.blocks:
            x_seq, attn = block(x_seq, return_attention=return_attention)
            if return_attention and attn is not None:
                attn_maps.append(attn)

        x_seq = self.norm(x_seq)  # (B, T, embed_dim)

        # 4. CTC Logits
        logits = self.head(x_seq)  # (B, T, num_classes)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, num_classes) for CTC

        # 5. Token-level uncertainty/hesitation
        uncertainty = self.uncertainty_head(x_seq)  # (B, T, 1)

        res = {
            "log_probs": log_probs,
            "logits": logits,
            "uncertainty": uncertainty,
        }
        if return_attention:
            res["attention_maps"] = attn_maps

        return res

    def decode_greedy(self, logits: torch.Tensor) -> List[str]:
        """
        Greedy CTC decoding: argmax per frame -> collapse repeats -> remove blank.
        Args:
            logits: (B, T, num_classes)
        Returns:
            List of decoded strings.
        """
        preds = torch.argmax(logits, dim=-1).cpu().numpy()  # (B, T)
        results = []
        for seq in preds:
            chars = []
            prev = None
            for idx in seq:
                if idx != prev:
                    if idx != self.blank_idx:
                        # Character index mapping: 1 -> alphabet[0], 2 -> alphabet[1], ...
                        char_pos = idx - 1
                        if 0 <= char_pos < len(self.alphabet):
                            chars.append(self.alphabet[char_pos])
                    prev = idx
            results.append("".join(chars))
        return results


def build_handwriting_transformer(
    embed_dim: int = 256,
    depth: int = 6,
    num_heads: int = 8,
) -> HandwritingTransformerOCR:
    """Factory builder for standard HandwritingTransformerOCR."""
    return HandwritingTransformerOCR(
        alphabet=DEFAULT_ALPHABET,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )


if __name__ == "__main__":
    model = build_handwriting_transformer()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HandwritingTransformerOCR initialized with {num_params:,} trainable parameters.")
    
    # Test forward pass with synthetic word image batch (B=4, C=1, H=64, W=256)
    dummy_img = torch.randn(4, 1, 64, 256)
    out = model(dummy_img, return_attention=True)
    print("Forward pass successful:")
    print(f"  Log Probs shape (T, B, Classes): {out['log_probs'].shape}")
    print(f"  Logits shape (B, T, Classes):    {out['logits'].shape}")
    print(f"  Uncertainty shape (B, T, 1):     {out['uncertainty'].shape}")
    print(f"  Attention layers extracted:      {len(out['attention_maps'])}")
    
    decoded = model.decode_greedy(out['logits'])
    print(f"  Decoded dummy sample:            {decoded[0]!r}")

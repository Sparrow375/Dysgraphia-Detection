"""Stage 34: Experiment 5 — De-Cheated Multi-Modal Gated-Attention MIL Architecture.

Anti-Memorization & De-Cheating Blueprint:
------------------------------------------
1. Crushed Parameter Sponge:
   - DenseNet201 frozen through conv0..denseblock3 and denselayer1..denselayer24.
   - Only last 8 layers of denseblock4 + norm5 trainable.
   - 1D Bottleneck Dilated Sequence Conv (1920 -> 32 channels) with GroupNorm and Spatial Dropout (p=0.5).
2. Anti-Silhouette Augmentations:
   - Random horizontal sub-windowing (128x480).
   - Vertical strip cutout (SpecAugment for handwriting).
   - Bag line-dropout.
3. Modality Dropout:
   - Training: 30% drop vision, 20% drop math, 50% pass both.
   - Evaluation: 100% pass both.
4. Exponential Moving Average (EMA, decay=0.99) tracking.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision.models import DenseNet201_Weights
import torchvision.transforms as transforms


class AntiSilhouetteAugmentor:
    """Applies random horizontal sub-windowing, vertical strip cutout, and line-dropout."""

    def __init__(
        self,
        target_h: int = 128,
        target_w: int = 480,
        full_w: int = 640,
        n_strips: int = 2,
        min_strip_w: int = 25,
        max_strip_w: int = 40,
        is_train: bool = True,
    ) -> None:
        self.target_h = target_h
        self.target_w = target_w
        self.full_w = full_w
        self.n_strips = n_strips
        self.min_strip_w = min_strip_w
        self.max_strip_w = max_strip_w
        self.is_train = is_train

    def __call__(self, tensor_img: torch.Tensor) -> torch.Tensor:
        """Args: tensor_img of shape (3, H, full_w) in [0, 1]. Returns: (3, H, target_w)."""
        _, h, w = tensor_img.shape

        # 1. Horizontal Sub-Windowing
        if self.is_train and w > self.target_w:
            start_x = random.randint(0, w - self.target_w)
            cropped = tensor_img[:, :, start_x : start_x + self.target_w].clone()
        elif w > self.target_w:
            # Center crop for evaluation
            start_x = (w - self.target_w) // 2
            cropped = tensor_img[:, :, start_x : start_x + self.target_w].clone()
        else:
            cropped = tensor_img.clone()

        # 2. Vertical Strip Cutout (SpecAugment for Handwriting)
        if self.is_train:
            _, cur_h, cur_w = cropped.shape
            for _ in range(self.n_strips):
                strip_w = random.randint(self.min_strip_w, self.max_strip_w)
                if cur_w > strip_w:
                    x0 = random.randint(0, cur_w - strip_w)
                    cropped[:, :, x0 : x0 + strip_w] = 0.0

        return cropped


class DeCheatedStudentBagDataset(Dataset):
    """Dataset delivering student bags with anti-silhouette augmentations and line-dropout."""

    def __init__(
        self,
        student_records: List[Dict[str, Any]],
        tensors_dict: Dict[str, np.ndarray],
        math_dict: Dict[str, np.ndarray],
        aux_dict: Dict[str, np.ndarray],
        lupi_dict: Optional[Dict[str, float]] = None,
        is_train: bool = True,
        sub_w: int = 480,
    ) -> None:
        self.student_records = student_records
        self.tensors_dict = tensors_dict
        self.math_dict = math_dict
        self.aux_dict = aux_dict
        self.lupi_dict = lupi_dict or {}
        self.is_train = is_train

        self.augmentor = AntiSilhouetteAugmentor(
            target_h=128,
            target_w=sub_w,
            full_w=640,
            n_strips=2,
            min_strip_w=25,
            max_strip_w=40,
            is_train=is_train,
        )

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.is_train:
            self.aug_pil = transforms.Compose([
                transforms.RandomRotation(degrees=(-4, 4), fill=0),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=0),
            ])
        else:
            self.aug_pil = None

    def __len__(self) -> int:
        return len(self.student_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.student_records[idx]
        sid = rec["student_id"]
        task_uids = list(rec["task_uids"])
        task_names = list(rec["task_names"])

        # 3. Bag Line-Dropout (Drop 1 random line if bag has K > 3)
        if self.is_train and len(task_uids) > 3 and random.random() < 0.50:
            drop_idx = random.randint(0, len(task_uids) - 1)
            task_uids.pop(drop_idx)
            task_names.pop(drop_idx)

        bag_images: List[torch.Tensor] = []
        bag_math: List[torch.Tensor] = []
        bag_aux: List[torch.Tensor] = []

        for uid in task_uids:
            uint8_arr = self.tensors_dict[uid]
            pil_img = Image.fromarray(uint8_arr)

            if self.is_train and self.aug_pil is not None:
                pil_img = self.aug_pil(pil_img)

            t_img = transforms.functional.to_tensor(pil_img)

            # Apply horizontal sub-windowing and strip cutout
            t_img = self.augmentor(t_img)
            t_img = self.norm(t_img)
            bag_images.append(t_img)

            bag_math.append(torch.tensor(self.math_dict[uid], dtype=torch.float32))
            bag_aux.append(torch.tensor(self.aux_dict[uid], dtype=torch.float32))

        # Soft LUPI Target Calculation
        raw_label = float(rec["label"])
        if self.is_train:
            if sid in self.lupi_dict:
                # Wacom stylus continuous impairment distillation
                q_stylus = self.lupi_dict[sid]
                soft_label = 0.75 * raw_label + 0.25 * q_stylus
            else:
                # Label smoothing: [0.05, 0.95]
                soft_label = 0.05 if raw_label == 0.0 else 0.95
        else:
            soft_label = raw_label

        return {
            "student_id": sid,
            "school": rec["school"],
            "label": torch.tensor(raw_label, dtype=torch.float32),
            "soft_label": torch.tensor(soft_label, dtype=torch.float32),
            "grade": rec["grade"],
            "task_names": task_names,
            "num_crops": len(task_uids),
            "images": torch.stack(bag_images, dim=0),  # (K, 3, 128, 480)
            "math": torch.stack(bag_math, dim=0),      # (K, 22)
            "aux": torch.stack(bag_aux, dim=0),        # (K, 2)
        }


def collate_decheated_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-sized student bags."""
    student_ids = [b["student_id"] for b in batch]
    schools = [b["school"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    soft_labels = torch.stack([b["soft_label"] for b in batch], dim=0)
    grades = [b["grade"] for b in batch]
    task_names_list = [b["task_names"] for b in batch]
    num_crops_list = [b["num_crops"] for b in batch]

    all_images = torch.cat([b["images"] for b in batch], dim=0)
    all_math = torch.cat([b["math"] for b in batch], dim=0)
    all_aux = torch.cat([b["aux"] for b in batch], dim=0)

    split_indices: List[Tuple[int, int]] = []
    curr = 0
    for k in num_crops_list:
        split_indices.append((curr, curr + k))
        curr += k

    return {
        "student_ids": student_ids,
        "schools": schools,
        "labels": labels,
        "soft_labels": soft_labels,
        "grades": grades,
        "task_names_list": task_names_list,
        "num_crops_list": num_crops_list,
        "all_images": all_images,
        "all_math": all_math,
        "all_aux": all_aux,
        "split_indices": split_indices,
    }


class BottleneckDilatedSequencePooler(nn.Module):
    """Bottleneck Dilated 1D Sequence Pooler with Spatial Dropout and GroupNorm.

    Total trainable parameters: ~184,000 (Crushes the 15M parameter sponge).
    """

    def __init__(
        self,
        in_channels: int = 1920,
        bottleneck_channels: int = 32,
        out_dim: int = 32,
        dropout: float = 0.50,
    ) -> None:
        super().__init__()
        self.conv1d_dilated = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=4, num_channels=bottleneck_channels),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=bottleneck_channels),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(bottleneck_channels * 2, out_dim),
            nn.ReLU(),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Args: feat_map (N, 1920, 4, W_feat). Returns: (N, out_dim)."""
        seq = feat_map.mean(dim=2)           # (N, 1920, W_feat)
        h_seq = self.conv1d_dilated(seq)     # (N, 32, W_feat)

        h_max = h_seq.max(dim=2)[0]          # (N, 32)
        h_mean = h_seq.mean(dim=2)           # (N, 32)
        h_pooled = torch.cat([h_max, h_mean], dim=1)  # (N, 64)

        return self.proj(h_pooled)           # (N, out_dim)


class DeCheatedMultiModalMILModel(nn.Module):
    """Experiment 5 Model: 98.8% Frozen DenseNet201 + Modality Dropout + Gated Attention."""

    def __init__(
        self,
        math_dim: int = 22,
        visual_dim: int = 32,
        math_emb_dim: int = 32,
        attn_dim: int = 32,
        pretrained: bool = True,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        # 1. Backbone: DenseNet201
        weights = DenseNet201_Weights.DEFAULT if pretrained else None
        densenet = models.densenet201(weights=weights)
        self.features = densenet.features

        # 2. Freeze 98.8% of backbone parameters
        # Freeze conv0 through denseblock3 + transition3
        # In denseblock4: freeze denselayer1 through denselayer24; keep denselayer25..32 + norm5 trainable
        for param in self.features.parameters():
            param.requires_grad = False

        # Unfreeze last 8 denselayers of denseblock4
        for i in range(25, 33):
            layer_name = f"denselayer{i}"
            if hasattr(self.features.denseblock4, layer_name):
                layer = getattr(self.features.denseblock4, layer_name)
                for p in layer.parameters():
                    p.requires_grad = True

        # Unfreeze norm5
        for p in self.features.norm5.parameters():
            p.requires_grad = True

        # 3. Bottleneck Dilated 1D Sequence Pooler
        self.sequence_pooler = BottleneckDilatedSequencePooler(
            in_channels=1920,
            bottleneck_channels=32,
            out_dim=visual_dim,
            dropout=0.50,
        )

        # 4. Multi-Task Auxiliary Legibility Head
        self.aux_head = nn.Sequential(
            nn.Linear(visual_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Predicts [trocr_mean_confidence, trocr_low_conf_ratio]
        )

        # 5. Math Feature Projection
        self.math_proj = nn.Sequential(
            nn.Linear(math_dim, math_emb_dim),
            nn.ReLU(),
            nn.Dropout(p=0.20),
        )

        # 6. Combined Line Vector & Gated Attention
        fused_dim = visual_dim + math_emb_dim  # 32 + 32 = 64
        self.attention_V = nn.Linear(fused_dim, attn_dim)
        self.attention_U = nn.Linear(fused_dim, attn_dim)
        self.attention_w = nn.Linear(attn_dim, 1)

        # 7. Student Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, 1),
        )

    def forward(
        self,
        all_images: torch.Tensor,
        all_math: torch.Tensor,
        split_indices: List[Tuple[int, int]],
        force_modality: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Forward pass with Modality Dropout.

        Args:
            all_images: (sum(K), 3, 128, W)
            all_math: (sum(K), math_dim)
            split_indices: List of (start, end) index pairs per student
            force_modality: None (random during training), 'both', 'vision_only', or 'math_only'

        Returns:
            student_logits: (B,)
            all_attentions: List of attention weight tensors per student
            aux_preds: (sum(K), 2)
        """
        # Visual feature extraction
        feat_map = self.features(all_images)           # (sum(K), 1920, 4, W_feat)
        v_emb = self.sequence_pooler(feat_map)         # (sum(K), visual_dim)
        aux_preds = self.aux_head(v_emb)               # (sum(K), 2)

        # Math feature projection
        m_emb = self.math_proj(all_math)               # (sum(K), math_emb_dim)

        # Modality Dropout (Forced Independence)
        if self.training and force_modality is None:
            r = random.random()
            if r < 0.30:
                # 30% drop vision -> Forces classifier to rely on math features
                v_emb = torch.zeros_like(v_emb)
            elif r < 0.50:
                # 20% drop math -> Forces classifier to rely on vision features
                m_emb = torch.zeros_like(m_emb)
            # 50% pass both
        elif force_modality == "vision_only":
            m_emb = torch.zeros_like(m_emb)
        elif force_modality == "math_only":
            v_emb = torch.zeros_like(v_emb)

        # Line-level fusion
        x_line = torch.cat([v_emb, m_emb], dim=-1)     # (sum(K), 64)

        student_logits_list: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        for start_idx, end_idx in split_indices:
            student_lines = x_line[start_idx:end_idx]  # (K_i, 64)
            v = torch.tanh(self.attention_V(student_lines))
            u = torch.sigmoid(self.attention_U(student_lines))
            gated = v * u
            attn_logits = self.attention_w(gated).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=0)

            z_student = torch.sum(student_lines * attn_weights.unsqueeze(-1), dim=0, keepdim=True)
            logit = self.classifier(z_student).squeeze(-1)

            all_attentions.append(attn_weights)
            student_logits_list.append(logit)

        student_logits = torch.cat(student_logits_list, dim=0)
        return student_logits, all_attentions, aux_preds


class EMAWeightTracker:
    """Maintains Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.99) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with exponential decay."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        """Apply shadow weights to model for evaluation."""
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore_backup(self, model: nn.Module) -> None:
        """Restore original model parameters after evaluation."""
        if hasattr(self, "backup"):
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name])


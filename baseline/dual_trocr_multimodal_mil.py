"""Dual-Script TrOCR Vision + Multi-Modal Gated-Attention MIL Model Architecture.

Components:
1. DualScriptSequenceBottleneck:
   - Hindi (Devanagari): Conv1d(768 -> 32, k=3, p=1) + GroupNorm(4, 32) + GELU + Dropout(0.3)
   - English (Latin): Conv1d(384 -> 32, k=3, p=1) + GroupNorm(4, 32) + GELU + Dropout(0.3)
   - Both pool horizontally across 24 positions to form unified visual vector v_k in R^32.
2. Math & TrOCR Feature Projection:
   - Linear(16 -> 32) + ReLU + Dropout(0.2) -> m_k in R^32.
3. Multi-Modal Line-Level Gated Attention MIL:
   - Fused line representation: x_k = [v_k || m_k] in R^64.
   - Attention gate: alpha_k = softmax(w^T (tanh(V x_k) * sigmoid(U x_k))).
   - Bag embedding: z_student = sum(alpha_k * x_k) in R^64.
   - Classifier: Dropout(0.4) + Linear(64 -> 1).
4. Multi-Task Auxiliary Legibility Head:
   - Linear(32 -> 16) + ReLU + Linear(16 -> 2) predicting TrOCR confidence metrics.
5. SWAModelTracker:
   - Accumulates and averages model weights across peak generalization window (Epochs 7-12).
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DualScriptSequenceBottleneck(nn.Module):
    """Script-specific 1D sequence convolutions projecting Hindi & English TrOCR tokens to R^32."""

    def __init__(self, out_dim: int = 32, dropout: float = 0.30) -> None:
        super().__init__()
        # Hindi branch: 768 -> out_dim
        self.conv_hi = nn.Sequential(
            nn.Conv1d(768, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_dim),
            nn.GELU(),
            nn.Dropout1d(p=dropout),
        )
        # English branch: 384 -> out_dim
        self.conv_en = nn.Sequential(
            nn.Conv1d(384, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_dim),
            nn.GELU(),
            nn.Dropout1d(p=dropout),
        )

    def forward(self, seq_tokens_list: List[torch.Tensor], script_list: List[str]) -> torch.Tensor:
        """Args:

        seq_tokens_list: List of length N, each tensor shape (24, D_script)
        script_list: List of length N, strings ('devanagari' or 'latin')

        Returns:
            v_embs: (N, out_dim)
        """
        v_list: List[torch.Tensor] = []

        for tokens, script in zip(seq_tokens_list, script_list):
            # tokens: (24, D) -> transpose to (1, D, 24)
            x = tokens.unsqueeze(0).permute(0, 2, 1)

            if script == "devanagari":
                feat = self.conv_hi(x)  # (1, 32, 24)
            else:
                feat = self.conv_en(x)  # (1, 32, 24)

            # Dual spatial pooling: max + mean over horizontal 24 steps
            v_max = feat.max(dim=2)[0]  # (1, 32)
            v_mean = feat.mean(dim=2)   # (1, 32)
            v_k = 0.5 * (v_max + v_mean)  # (1, 32)
            v_list.append(v_k)

        return torch.cat(v_list, dim=0)  # (N, out_dim)


class DualScriptMultiModalMILModel(nn.Module):
    """Complete Dual-Script Multi-Modal MIL Architecture with Gated Attention & Aux Head."""

    def __init__(
        self,
        math_dim: int = 16,
        visual_dim: int = 32,
        math_emb_dim: int = 32,
        attn_dim: int = 32,
        dropout: float = 0.40,
        pure_vision: bool = False,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.pure_vision = pure_vision
        self.math_emb_dim = 0 if pure_vision else math_emb_dim

        # 1. Dual-Script Sequence Bottleneck
        self.visual_adapter = DualScriptSequenceBottleneck(out_dim=visual_dim, dropout=0.30)

        # 2. Math & OCR Feature Projection (skipped if pure_vision)
        if not pure_vision:
            self.math_proj = nn.Sequential(
                nn.Linear(math_dim, math_emb_dim),
                nn.ReLU(),
                nn.Dropout(p=0.20),
            )
        else:
            self.math_proj = None

        # 3. Auxiliary Legibility Head (Predicts [mean_conf, low_conf_ratio])
        self.aux_head = nn.Sequential(
            nn.Linear(visual_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

        # 4. Gated Attention Pooling
        fused_dim = visual_dim if pure_vision else (visual_dim + math_emb_dim)
        self.attention_V = nn.Linear(fused_dim, attn_dim)
        self.attention_U = nn.Linear(fused_dim, attn_dim)
        self.attention_w = nn.Linear(attn_dim, 1)

        # 5. Student Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, 1),
        )

    def forward(
        self,
        seq_tokens_list: List[torch.Tensor],
        script_list: List[str],
        all_math: torch.Tensor,
        split_indices: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Args:

        seq_tokens_list: List of length sum(K), each tensor shape (24, D)
        script_list: List of length sum(K) ('devanagari' or 'latin')
        all_math: (sum(K), math_dim)
        split_indices: List of (start_idx, end_idx) bag bounds

        Returns:
            student_logits: (B,)
            all_attentions: List of attention weights per student
            aux_preds: (sum(K), 2)
        """
        # Visual representation
        v_emb = self.visual_adapter(seq_tokens_list, script_list)  # (sum(K), 32)
        aux_preds = self.aux_head(v_emb)                           # (sum(K), 2)

        # Line representation
        if self.pure_vision:
            x_line = v_emb                                         # (sum(K), 32) Pure TrOCR Vision
        else:
            m_emb = self.math_proj(all_math)                       # (sum(K), 32)
            x_line = torch.cat([v_emb, m_emb], dim=-1)             # (sum(K), 64) Multi-Modal Fusion

        student_logits_list: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        for start_idx, end_idx in split_indices:
            student_lines = x_line[start_idx:end_idx]              # (K_i, 64)
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


class SWAModelTracker:
    """Stochastic Weight Averaging (SWA) tracker across specified epoch window."""

    def __init__(self, model: nn.Module) -> None:
        self.swa_state: Optional[Dict[str, torch.Tensor]] = None
        self.n_models = 0
        self.device = next(model.parameters()).device

    def update(self, model: nn.Module) -> None:
        """Accumulate model weights."""
        current_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        if self.swa_state is None:
            self.swa_state = current_state
            self.n_models = 1
        else:
            self.n_models += 1
            alpha = 1.0 / self.n_models
            for k in self.swa_state:
                self.swa_state[k] += alpha * (current_state[k] - self.swa_state[k])

    def apply_to(self, model: nn.Module) -> None:
        """Load averaged weights into target model."""
        if self.swa_state is not None:
            gpu_state = {k: v.to(self.device) for k, v in self.swa_state.items()}
            model.load_state_dict(gpu_state)


class DualScriptStudentBagDataset(Dataset):
    """Delivers student bags with cached TrOCR tokens and duplicate-only augmentations."""

    def __init__(
        self,
        student_records: List[Dict[str, Any]],
        tokens_cache: Dict[str, np.ndarray],
        math_dict: Dict[str, np.ndarray],
        aux_dict: Dict[str, np.ndarray],
        scripts_dict: Dict[str, str],
        bag_type: str = "clean",  # 'clean', 'cutout', 'shear'
    ) -> None:
        self.student_records = student_records
        self.tokens_cache = tokens_cache
        self.math_dict = math_dict
        self.aux_dict = aux_dict
        self.scripts_dict = scripts_dict
        self.bag_type = bag_type

    def __len__(self) -> int:
        return len(self.student_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.student_records[idx]
        task_uids = list(rec["task_uids"])

        # Bag 3 (Sub-bag duplicate): randomly drop 1 line if K > 2
        if self.bag_type == "shear" and len(task_uids) > 2:
            drop_idx = random.randint(0, len(task_uids) - 1)
            task_uids = [uid for i, uid in enumerate(task_uids) if i != drop_idx]

        bag_tokens: List[torch.Tensor] = []
        bag_scripts: List[str] = []
        bag_math: List[torch.Tensor] = []
        bag_aux: List[torch.Tensor] = []

        for uid in task_uids:
            key = f"{uid}_{self.bag_type}"
            if key not in self.tokens_cache:
                key = f"{uid}_clean"

            token_arr = self.tokens_cache[key]
            bag_tokens.append(torch.from_numpy(token_arr).float())
            bag_scripts.append(self.scripts_dict.get(uid, "latin"))

            math_vec = torch.tensor(self.math_dict[uid], dtype=torch.float32)
            bag_math.append(math_vec)

            aux_vec = torch.tensor(self.aux_dict[uid], dtype=torch.float32)
            bag_aux.append(aux_vec)

        return {
            "student_id": rec["student_id"],
            "school": rec["school"],
            "label": torch.tensor(rec["label"], dtype=torch.float32),
            "grade": rec["grade"],
            "task_uids": task_uids,
            "num_crops": len(task_uids),
            "tokens": bag_tokens,
            "scripts": bag_scripts,
            "math": torch.stack(bag_math, dim=0),
            "aux": torch.stack(bag_aux, dim=0),
        }


def collate_dual_script_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-sized dual-script student bags."""
    student_ids = [b["student_id"] for b in batch]
    schools = [b["school"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    grades = [b["grade"] for b in batch]
    num_crops_list = [b["num_crops"] for b in batch]

    all_tokens: List[torch.Tensor] = []
    all_scripts: List[str] = []
    for b in batch:
        all_tokens.extend(b["tokens"])
        all_scripts.extend(b["scripts"])

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
        "grades": grades,
        "num_crops_list": num_crops_list,
        "all_tokens": all_tokens,
        "all_scripts": all_scripts,
        "all_math": all_math,
        "all_aux": all_aux,
        "split_indices": split_indices,
    }


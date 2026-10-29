"""
Experiment 7: Multi-Backbone Feature Extractor for Representation Bake-Off
Extracts spatial patch-token summary statistics [mean, std, GeM] from 224x224
native stroke-scale tiles using various vision backbones:
1. 'trocr': Microsoft TrOCR-Base Handwritten ViT encoder (handwriting-specific pretraining)
2. 'densenet201': ImageNet DenseNet201 CNN spatial tokens (7x7=49 tokens, D=1920)
3. 'clip': OpenAI CLIP-ViT-B/32 (multimodal contrastive semantic ViT)

Enables fair empirical comparison against DINOv2 self-supervised patch tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models as tv_models
from transformers import CLIPVisionModelWithProjection, VisionEncoderDecoderModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.tiling_preprocessor import build_white_canvas_strip, extract_tiles_from_strip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("exp7_backbones")

DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"


def compute_patch_descriptors(
    patch_tokens: torch.Tensor,
    p: float = 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute mean, std, and GeM pooling across spatial patch tokens."""
    u = patch_tokens.mean(dim=1)
    var = torch.mean((patch_tokens - u.unsqueeze(1)) ** 2, dim=1)
    s = torch.sqrt(var + eps)
    gem = (torch.clamp(patch_tokens, min=eps).pow(p).mean(dim=1)).pow(1.0 / p)
    return torch.cat([u, s, gem], dim=-1)


class BackboneWrapper(nn.Module):
    """Unified wrapper returning spatial patch tokens (B, P, D) for any backbone."""

    def __init__(self, backbone_name: str, device: torch.device):
        super().__init__()
        self.backbone_name = backbone_name
        self.device = device

        if backbone_name == "trocr":
            logger.info("Loading TrOCR handwritten vision encoder...")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            self.encoder = model.get_encoder()
            self.feature_dim = self.encoder.config.hidden_size  # 768
        elif backbone_name == "densenet201":
            logger.info("Loading torchvision DenseNet201...")
            densenet = tv_models.densenet201(weights=tv_models.DenseNet201_Weights.DEFAULT)
            self.encoder = densenet.features
            self.feature_dim = 1920
        elif backbone_name == "clip":
            logger.info("Loading OpenAI CLIP-ViT-B/32...")
            clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            self.encoder = clip_model.vision_model
            self.feature_dim = self.encoder.config.hidden_size  # 768
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch tokens (B, num_spatial_tokens, D)."""
        if self.backbone_name == "trocr":
            # TrOCR ViT encoder expects 384x384 input or interpolated pos encodings
            x_384 = nn.functional.interpolate(x, size=(384, 384), mode="bicubic", align_corners=False)
            outputs = self.encoder(x_384)
            # ViT tokens: index 1: are patch tokens (skipping [CLS])
            return outputs.last_hidden_state[:, 1:, :]
        elif self.backbone_name == "densenet201":
            # Feature map: (B, 1920, 7, 7)
            feat_maps = self.encoder(x)
            # Flatten spatial 7x7 to 49 tokens: (B, 49, 1920)
            return feat_maps.flatten(2).permute(0, 2, 1)
        elif self.backbone_name == "clip":
            outputs = self.encoder(x)
            return outputs.last_hidden_state[:, 1:, :]
        raise NotImplementedError


def extract_backbone_features(
    backbone_name: str = "trocr",
    batch_size: int = 64,
    device_str: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Extract and cache patch statistics for specified backbone across all cohort tiles."""
    if output_path is None:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FEATURES_DIR / f"exp7_{backbone_name}_tiles.npz"

    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s for backbone: %s", device, backbone_name)

    model = BackboneWrapper(backbone_name, device=device)

    # Load dataset index (reference cohort: School A + School B)
    df_index = pd.read_csv(INDEX_CSV)
    df_index = df_index[df_index["school"].isin(["school_a", "school_b"])].copy()
    logger.info("Loading tiles across %d sentence tasks...", len(df_index))

    # Preprocessing ImageNet normalization constants
    mean_rgb = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_rgb = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    all_tiles: List[np.ndarray] = []
    tile_student_ids: List[str] = []
    tile_task_uids: List[str] = []
    tile_scripts: List[str] = []
    tile_grades: List[int] = []
    tile_labels: List[int] = []
    tile_schools: List[str] = []
    tile_task_ids: List[str] = []

    student_meta_map: Dict[str, Dict[str, Any]] = {}
    start_time = time.time()

    for idx, row in df_index.iterrows():
        sid = str(row["student_id"])
        sch = str(row["school"])
        tid = str(row["task_id"])
        tuid = str(row["task_uid"])
        script = str(row["script"])
        grade = int(row["grade"])
        label = int(row["label"])

        if sid not in student_meta_map:
            student_meta_map[sid] = {
                "student_id": sid,
                "school": sch,
                "grade": grade,
                "label": label,
                "repeat_0_fold": int(row.get("repeat_0_fold", -1)),
                "repeat_1_fold": int(row.get("repeat_1_fold", -1)),
                "repeat_2_fold": int(row.get("repeat_2_fold", -1)),
                "repeat_3_fold": int(row.get("repeat_3_fold", -1)),
                "repeat_4_fold": int(row.get("repeat_4_fold", -1)),
            }

        proc_dir = PROCESSED_DIR / sch / sid
        raw_p = proc_dir / f"{tid}.png"
        clean_p = proc_dir / f"{tid}_clean_ink.png"
        json_p = proc_dir / f"{tid}.json"

        if not raw_p.exists() or not clean_p.exists():
            continue

        try:
            strip = build_white_canvas_strip(clean_p, raw_p, json_p)
            tiles, _ = extract_tiles_from_strip(strip, target_height=224, stride=112)
        except Exception as e:
            logger.warning("Failed tiling for %s: %s", tuid, e)
            continue

        for tile in tiles:
            all_tiles.append(tile)
            tile_student_ids.append(sid)
            tile_task_uids.append(tuid)
            tile_scripts.append(script)
            tile_grades.append(grade)
            tile_labels.append(label)
            tile_schools.append(sch)
            tile_task_ids.append(tid)

    total_tiles = len(all_tiles)
    logger.info("Total tiles to process: %d across %d tasks in %.1fs", total_tiles, len(df_index), time.time() - start_time)

    # Batch GPU feature extraction
    tile_descriptors: List[np.ndarray] = []
    inference_start = time.time()

    for i in range(0, total_tiles, batch_size):
        batch_tiles = all_tiles[i : i + batch_size]
        batch_np = np.stack(batch_tiles, axis=0).astype(np.float32) / 255.0
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device)
        norm_tensor = (batch_tensor - mean_rgb) / std_rgb

        with torch.no_grad():
            spatial_tokens = model(norm_tensor)
            desc_tensor = compute_patch_descriptors(spatial_tokens)
            tile_descriptors.append(desc_tensor.cpu().numpy().astype(np.float32))

        if (i // batch_size) % 50 == 0:
            logger.info("Processed %d / %d tiles...", i + len(batch_tiles), total_tiles)

    all_descriptors = np.concatenate(tile_descriptors, axis=0)
    logger.info("Extraction complete: shape=%s in %.1fs", all_descriptors.shape, time.time() - inference_start)

    # Aggregate student-level features
    unique_students = list(student_meta_map.keys())
    unique_students.sort()
    n_students = len(unique_students)
    desc_dim = all_descriptors.shape[1]

    hindi_student_features = np.zeros((n_students, 2 * desc_dim), dtype=np.float32)
    english_student_features = np.zeros((n_students, 2 * desc_dim), dtype=np.float32)
    joint_student_features = np.zeros((n_students, 4 * desc_dim), dtype=np.float32)

    student_labels = np.array([student_meta_map[s]["label"] for s in unique_students], dtype=np.int64)
    student_grades = np.array([student_meta_map[s]["grade"] for s in unique_students], dtype=np.int64)
    student_schools = np.array([student_meta_map[s]["school"] for s in unique_students])
    repeat_folds = {
        f"repeat_{r}_fold": np.array([student_meta_map[s][f"repeat_{r}_fold"] for s in unique_students], dtype=np.int64)
        for r in range(5)
    }

    tile_sids_arr = np.array(tile_student_ids)
    tile_scripts_arr = np.array(tile_scripts)

    for i, sid in enumerate(unique_students):
        s_mask = (tile_sids_arr == sid)
        s_desc = all_descriptors[s_mask]
        s_scripts = tile_scripts_arr[s_mask]

        h_mask = (s_scripts == "devanagari")
        e_mask = (s_scripts == "latin")

        h_desc = s_desc[h_mask] if np.any(h_mask) else s_desc
        e_desc = s_desc[e_mask] if np.any(e_mask) else s_desc

        h_mean = np.mean(h_desc, axis=0) if len(h_desc) > 0 else np.zeros(desc_dim, dtype=np.float32)
        h_std = np.std(h_desc, axis=0) if len(h_desc) > 0 else np.zeros(desc_dim, dtype=np.float32)
        hindi_student_features[i] = np.concatenate([h_mean, h_std])

        e_mean = np.mean(e_desc, axis=0) if len(e_desc) > 0 else np.zeros(desc_dim, dtype=np.float32)
        e_std = np.std(e_desc, axis=0) if len(e_desc) > 0 else np.zeros(desc_dim, dtype=np.float32)
        english_student_features[i] = np.concatenate([e_mean, e_std])

        joint_student_features[i] = np.concatenate([hindi_student_features[i], english_student_features[i]])

    save_dict = {
        "tile_descriptors": all_descriptors,
        "tile_student_ids": np.array(tile_student_ids),
        "tile_task_uids": np.array(tile_task_uids),
        "tile_scripts": np.array(tile_scripts),
        "tile_grades": np.array(tile_grades),
        "tile_labels": np.array(tile_labels),
        "tile_schools": np.array(tile_schools),
        "tile_task_ids": np.array(tile_task_ids),
        "student_ids": np.array(unique_students),
        "student_labels": student_labels,
        "student_grades": student_grades,
        "student_schools": student_schools,
        "hindi_student_features": hindi_student_features,
        "english_student_features": english_student_features,
        "joint_student_features": joint_student_features,
        **repeat_folds,
    }

    np.savez_compressed(output_path, **save_dict)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %s (%.2f MB)", output_path, file_size_mb)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract features for representation bake-off")
    parser.add_argument(
        "--backbone",
        type=str,
        default="trocr",
        choices=["trocr", "densenet201", "clip"],
        help="Backbone model",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    extract_backbone_features(backbone_name=args.backbone, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

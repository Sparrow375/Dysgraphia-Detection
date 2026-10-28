"""Extract Frozen DINOv2 Patch-Token Statistics for Experiment 7.

Extracts local patch-token statistics from 224x224 native stroke-scale tiles:
1. Mean across 256 patches (mu_patch in R^D)
2. Standard deviation across 256 patches (sigma_patch in R^D)
3. Generalized Mean (GeM) across 256 patches (g_patch in R^D, p=3.0)

Descriptor per tile: e_tile = [mu_patch, sigma_patch, g_patch] in R^(3D)
(384 * 3 = 1,152-d for DINOv2-Small; 768 * 3 = 2,304-d for DINOv2-Base)

Persists tile-level descriptors and aggregated student vectors to .npz cache.
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
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.tiling_preprocessor import build_white_canvas_strip, extract_tiles_from_strip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"


MODEL_MAPPING = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
}


def compute_patch_descriptors(
    patch_tokens: torch.Tensor,
    p: float = 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute mean, std, and GeM pooling across spatial patch tokens.

    Args:
        patch_tokens: Tensor of shape (B, num_patches, D).
        p: GeM pooling power exponent.
        eps: Small constant for numerical stability.

    Returns:
        Tensor of shape (B, 3 * D).
    """
    # 1. Spatial Mean
    u = patch_tokens.mean(dim=1)  # (B, D)

    # 2. Spatial Standard Deviation
    var = torch.mean((patch_tokens - u.unsqueeze(1)) ** 2, dim=1)
    s = torch.sqrt(var + eps)  # (B, D)

    # 3. Generalized Mean (GeM)
    gem = (torch.clamp(patch_tokens, min=eps).pow(p).mean(dim=1)).pow(1.0 / p)  # (B, D)

    return torch.cat([u, s, gem], dim=-1)


def extract_features_for_cohort(
    model_name_or_key: str = "dinov2-small",
    batch_size: int = 64,
    device_str: Optional[str] = None,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Path:
    """Extract and cache DINOv2 patch statistics across all cohort tiles."""
    hf_model_id = MODEL_MAPPING.get(model_name_or_key, model_name_or_key)
    tag = "small" if "small" in model_name_or_key else "base"
    
    if output_path is None:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FEATURES_DIR / f"exp7_dinov2_{tag}_tiles.npz"

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info("Loading DINOv2 model: %s on device: %s", hf_model_id, device)
    model = AutoModel.from_pretrained(hf_model_id)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    # Load dataset index
    df_index = pd.read_csv(INDEX_CSV)
    if limit is not None and limit > 0:
        unique_students = df_index["student_id"].unique()[:limit]
        df_index = df_index[df_index["student_id"].isin(unique_students)].copy()
        logger.info("Limited extraction to %d students (%d tasks)", limit, len(df_index))

    logger.info("Extracting tiles across %d sentence tasks...", len(df_index))

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

    # Student metadata map
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
    logger.info("Total tiles to process: %d (across %d tasks in %.1fs)", total_tiles, len(df_index), time.time() - start_time)

    # Batch GPU feature extraction
    tile_descriptors: List[np.ndarray] = []
    inference_start = time.time()

    for i in range(0, total_tiles, batch_size):
        batch_tiles = all_tiles[i : i + batch_size]
        # Stack to (B, H, W, 3) -> (B, 3, H, W)
        batch_np = np.stack(batch_tiles, axis=0).astype(np.float32) / 255.0
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device)
        norm_tensor = (batch_tensor - mean_rgb) / std_rgb

        with torch.no_grad():
            outputs = model(norm_tensor)
            # Patch tokens start at index 1 (skipping [CLS] at 0)
            patches = outputs.last_hidden_state[:, 1:, :]  # (B, 256, D)
            desc = compute_patch_descriptors(patches, p=3.0)  # (B, 3 * D)

        tile_descriptors.append(desc.cpu().numpy().astype(np.float32))

        if (i // batch_size) % 25 == 0 or i + batch_size >= total_tiles:
            elapsed = time.time() - inference_start
            logger.info("Processed %d / %d tiles (%.1f%%) in %.1fs (%.1f tiles/s)",
                        min(i + batch_size, total_tiles), total_tiles,
                        (min(i + batch_size, total_tiles) / total_tiles) * 100.0,
                        elapsed, min(i + batch_size, total_tiles) / max(elapsed, 0.01))

    all_desc = np.concatenate(tile_descriptors, axis=0)  # Shape: (N_tiles, 3 * D)
    desc_dim = all_desc.shape[1]
    logger.info("All tile descriptors computed. Shape: %s", all_desc.shape)

    # Group tiles by student and script
    unique_students = list(student_meta_map.keys())
    hindi_vecs: List[np.ndarray] = []
    english_vecs: List[np.ndarray] = []
    joint_vecs: List[np.ndarray] = []
    student_labels: List[int] = []
    student_grades: List[int] = []
    student_schools: List[str] = []
    repeat_folds: Dict[str, List[int]] = {f"repeat_{r}_fold": [] for r in range(5)}

    tile_stu_arr = np.array(tile_student_ids)
    tile_scr_arr = np.array(tile_scripts)

    for sid in unique_students:
        meta = student_meta_map[sid]
        student_labels.append(meta["label"])
        student_grades.append(meta["grade"])
        student_schools.append(meta["school"])
        for r in range(5):
            repeat_folds[f"repeat_{r}_fold"].append(meta[f"repeat_{r}_fold"])

        stu_mask = tile_stu_arr == sid
        stu_descs = all_desc[stu_mask]
        stu_scripts = tile_scr_arr[stu_mask]

        hin_mask = stu_scripts == "devanagari"
        eng_mask = stu_scripts == "latin"

        # Hindi student vector: mean + std
        if np.count_nonzero(hin_mask) > 0:
            h_descs = stu_descs[hin_mask]
            h_mean = np.mean(h_descs, axis=0)
            h_std = np.std(h_descs, axis=0)
            h_vec = np.concatenate([h_mean, h_std], axis=-1)  # (6 * D)
        else:
            h_vec = np.zeros(2 * desc_dim, dtype=np.float32)

        # English student vector: mean + std
        if np.count_nonzero(eng_mask) > 0:
            e_descs = stu_descs[eng_mask]
            e_mean = np.mean(e_descs, axis=0)
            e_std = np.std(e_descs, axis=0)
            e_vec = np.concatenate([e_mean, e_std], axis=-1)  # (6 * D)
        else:
            e_vec = np.zeros(2 * desc_dim, dtype=np.float32)

        joint_vec = np.concatenate([h_vec, e_vec], axis=-1)  # (12 * D)

        hindi_vecs.append(h_vec)
        english_vecs.append(e_vec)
        joint_vecs.append(joint_vec)

    hindi_mat = np.array(hindi_vecs, dtype=np.float32)
    english_mat = np.array(english_vecs, dtype=np.float32)
    joint_mat = np.array(joint_vecs, dtype=np.float32)

    logger.info("Aggregated Student Vectors:")
    logger.info("  Hindi Matrix:   %s", hindi_mat.shape)
    logger.info("  English Matrix: %s", english_mat.shape)
    logger.info("  Joint Matrix:   %s", joint_mat.shape)

    # Save everything to compressed npz
    save_dict: Dict[str, Any] = {
        # Tile level
        "tile_descriptors": all_desc,
        "tile_student_ids": np.array(tile_student_ids),
        "tile_task_uids": np.array(tile_task_uids),
        "tile_scripts": np.array(tile_scripts),
        "tile_grades": np.array(tile_grades),
        "tile_labels": np.array(tile_labels),
        "tile_schools": np.array(tile_schools),
        "tile_task_ids": np.array(tile_task_ids),
        # Student level
        "student_ids": np.array(unique_students),
        "student_labels": np.array(student_labels),
        "student_grades": np.array(student_grades),
        "student_schools": np.array(student_schools),
        "hindi_student_features": hindi_mat,
        "english_student_features": english_mat,
        "joint_student_features": joint_mat,
    }
    for r in range(5):
        save_dict[f"repeat_{r}_fold"] = np.array(repeat_folds[f"repeat_{r}_fold"])

    np.savez_compressed(output_path, **save_dict)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved extracted features to %s (%.2f MB)", output_path, size_mb)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 Patch-Token Statistics for Experiment 7.")
    parser.add_argument("--model", choices=["dinov2-small", "dinov2-base"], default="dinov2-small",
                        help="DINOv2 model architecture scale.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of students for quick dry run.")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    extract_features_for_cohort(
        model_name_or_key=args.model,
        batch_size=args.batch_size,
        device_str=args.device,
        output_path=out_p,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()


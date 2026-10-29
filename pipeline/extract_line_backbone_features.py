"""
Extract DINOv2 and CLIP tile descriptors on the decomposed single-line dataset.
Saves to data/features/exp7_line_dinov2_small_tiles.npz and exp7_line_clip_tiles.npz.
"""
import sys
import logging
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModelWithProjection
from baseline.tiling_preprocessor import extract_tiles_from_strip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("line_backbones")

DATA_DIR = REPO_ROOT / "data"
LINES_DIR = DATA_DIR / "line_crops_dataset"
LINES_INDEX = LINES_DIR / "lines_index.csv"
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

def extract_for_backbone(
    backbone_name: str,
    batch_size: int = 64,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("Extracting %s on %s...", backbone_name, device)

    # 1. Load model
    if backbone_name == "dinov2-small":
        model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
        model.eval()
        mean_rgb = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std_rgb = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        out_name = "exp7_line_dinov2_small_tiles.npz"
    elif backbone_name == "clip":
        clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_model.eval()
        mean_rgb = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std_rgb = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        out_name = "exp7_line_clip_tiles.npz"
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    df_lines = pd.read_csv(LINES_INDEX)
    logger.info("Loaded %d single lines from %s", len(df_lines), LINES_INDEX)

    # 2. Extract tiles
    t0 = time.time()
    all_tiles = []
    tile_student_ids = []
    tile_line_uids = []
    tile_scripts = []
    tile_grades = []
    tile_labels = []
    tile_schools = []

    student_metadata = {}

    for _, row in df_lines.iterrows():
        sid = str(row["student_id"])
        sch = str(row["school"])
        grade = int(row["grade"]) if pd.notna(row["grade"]) else -1
        label = int(row["label"]) if pd.notna(row["label"]) else 0
        luid = str(row["line_uid"])
        script = str(row.get("script", "devanagari"))
        rel = str(row["line_rel_path"])
        img_p = LINES_DIR / rel

        if sid not in student_metadata:
            student_metadata[sid] = {
                "school": sch,
                "grade": grade,
                "label": label,
                "repeat_0_fold": int(row.get("repeat_0_fold", -1)),
                "repeat_1_fold": int(row.get("repeat_1_fold", -1)),
                "repeat_2_fold": int(row.get("repeat_2_fold", -1)),
                "repeat_3_fold": int(row.get("repeat_3_fold", -1)),
                "repeat_4_fold": int(row.get("repeat_4_fold", -1)),
            }

        if not img_p.exists():
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (gray < 200).astype(np.float32)
        ch1 = np.where(ink > 0, 0, 255).astype(np.uint8)
        ch2 = gray
        ch3 = ch1
        strip = np.stack([ch1, ch2, ch3], axis=-1)

        tiles, _ = extract_tiles_from_strip(strip, target_height=224, stride=112)
        for tile in tiles:
            all_tiles.append(tile)
            tile_student_ids.append(sid)
            tile_line_uids.append(luid)
            tile_scripts.append(script)
            tile_grades.append(grade)
            tile_labels.append(label)
            tile_schools.append(sch)

    total_tiles = len(all_tiles)
    logger.info("Total line tiles: %d (in %.1fs)", total_tiles, time.time() - t0)

    # 3. GPU Batch Inference
    tile_desc_list = []
    infer_start = time.time()

    for i in range(0, total_tiles, batch_size):
        batch = all_tiles[i : i + batch_size]
        batch_np = np.stack(batch, axis=0).astype(np.float32) / 255.0
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device)
        norm_tensor = (batch_tensor - mean_rgb) / std_rgb

        with torch.no_grad():
            if backbone_name == "dinov2-small":
                outputs = model(norm_tensor)
                patches = outputs.last_hidden_state[:, 1:, :]
                desc = compute_patch_descriptors(patches, p=3.0)
            elif backbone_name == "clip":
                outputs = clip_model(norm_tensor)
                patches = outputs.last_hidden_state[:, 1:, :]
                desc = compute_patch_descriptors(patches, p=3.0)

        tile_desc_list.append(desc.cpu().numpy().astype(np.float32))

        if (i // batch_size) % 50 == 0 or i + batch_size >= total_tiles:
            elapsed = time.time() - infer_start
            logger.info("Processed %d / %d tiles (%.1f%%) in %.1fs (%.1f tiles/s)",
                        min(i + batch_size, total_tiles), total_tiles,
                        (min(i + batch_size, total_tiles) / total_tiles) * 100,
                        elapsed, min(i + batch_size, total_tiles) / max(elapsed, 0.01))

    all_desc = np.concatenate(tile_desc_list, axis=0)

    # 4. Student-Level Aggregation
    unique_students = sorted(list(student_metadata.keys()))
    tile_sids_arr = np.array(tile_student_ids)
    tile_scripts_arr = np.array(tile_scripts)

    hindi_feats = []
    english_feats = []
    joint_feats = []

    student_labels = []
    student_grades = []
    student_schools = []
    repeat_folds = {f"repeat_{r}_fold": [] for r in range(5)}

    desc_dim = all_desc.shape[1]

    for sid in unique_students:
        meta = student_metadata[sid]
        student_labels.append(meta["label"])
        student_grades.append(meta["grade"])
        student_schools.append(meta["school"])
        for r in range(5):
            repeat_folds[f"repeat_{r}_fold"].append(meta[f"repeat_{r}_fold"])

        s_mask = (tile_sids_arr == sid)
        s_desc = all_desc[s_mask]
        s_scripts = tile_scripts_arr[s_mask]

        # Hindi
        h_desc = s_desc[s_scripts == "devanagari"]
        if len(h_desc) > 0:
            h_vec = np.concatenate([h_desc.mean(axis=0), h_desc.std(axis=0), np.percentile(h_desc, 90, axis=0)])
        else:
            h_vec = np.zeros(desc_dim * 3, dtype=np.float32)
        hindi_feats.append(h_vec)

        # English
        e_desc = s_desc[s_scripts == "latin"]
        if len(e_desc) > 0:
            e_vec = np.concatenate([e_desc.mean(axis=0), e_desc.std(axis=0), np.percentile(e_desc, 90, axis=0)])
        else:
            e_vec = np.zeros(desc_dim * 3, dtype=np.float32)
        english_feats.append(e_vec)

        # Joint
        if len(s_desc) > 0:
            j_vec = np.concatenate([s_desc.mean(axis=0), s_desc.std(axis=0), np.percentile(s_desc, 90, axis=0)])
        else:
            j_vec = np.zeros(desc_dim * 3, dtype=np.float32)
        joint_feats.append(j_vec)

    out_file = FEATURES_DIR / out_name
    save_dict = {
        "tile_descriptors": all_desc,
        "tile_student_ids": tile_sids_arr,
        "tile_line_uids": np.array(tile_line_uids),
        "tile_scripts": tile_scripts_arr,
        "student_ids": np.array(unique_students),
        "student_labels": np.array(student_labels),
        "student_grades": np.array(student_grades),
        "student_schools": np.array(student_schools),
        "hindi_student_features": np.stack(hindi_feats, axis=0),
        "english_student_features": np.stack(english_feats, axis=0),
        "joint_student_features": np.stack(joint_feats, axis=0),
    }
    for r in range(5):
        save_dict[f"repeat_{r}_fold"] = np.array(repeat_folds[f"repeat_{r}_fold"])

    np.savez_compressed(out_file, **save_dict)
    logger.info("Saved %s (%.2f MB)", out_file, out_file.stat().st_size / (1024 * 1024))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["dinov2-small", "clip"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    extract_for_backbone(args.backbone, batch_size=args.batch_size)

if __name__ == "__main__":
    main()

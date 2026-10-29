"""
Caches Foreground Ink-Masked DINOv2 Embeddings for:
  1. The 255 training sessions (digital stylus renders) -> outputs/cache/dinov2_ink_embeddings_255.npz
  2. The 60 paired real paper photos -> outputs/cache/dinov2_ink_embeddings_paper60.npz
Uses GPU (NVIDIA RTX 3050) acceleration for fast batch extraction.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.branch_b.dinov2_extractor import extract_ink_masked_dinov2_features

CACHE_DIR = os.path.join(PROJECT_ROOT, "outputs", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

STUDENTS_DIR = os.path.join(PROJECT_ROOT, "zmapped_output", "students")
FEAT_255_CSV = os.path.join(PROJECT_ROOT, "outputs", "step3", "extracted_features_255.csv")
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "outputs", "step4", "paper_image_manifest.csv")


def cache_255_sessions():
    out_file = os.path.join(CACHE_DIR, "dinov2_ink_embeddings_255.npz")
    if os.path.exists(out_file):
        print(f"[Cache 255] Found existing cache at {out_file}, skipping extraction.")
        return

    print("[Cache 255] Starting DINOv2 ink-masked extraction on 255 sessions...")
    feat_df = pd.read_csv(FEAT_255_CSV)
    
    features_list = []
    sids_list = []
    tasks_list = []
    t0 = time.time()

    for idx, row in feat_df.iterrows():
        sid = row['student_id']
        task = row['task']
        s_stylus = os.path.join(STUDENTS_DIR, sid, "stylus")
        ren_candidates = [f for f in os.listdir(s_stylus) if f"_{task}_render.png" in f] if os.path.exists(s_stylus) else []
        if not ren_candidates:
            raise FileNotFoundError(f"Missing render image for {sid} {task}")
        img_path = os.path.join(s_stylus, ren_candidates[0])

        dino_res = extract_ink_masked_dinov2_features(img_path)
        features_list.append(dino_res["combined_features"])
        sids_list.append(sid)
        tasks_list.append(task)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(feat_df):
            print(f"  Processed {idx + 1}/{len(feat_df)} sessions ({time.time() - t0:.1f}s)")

    feat_arr = np.array(features_list, dtype=np.float32)
    np.savez_compressed(
        out_file,
        features=feat_arr,
        student_id=np.array(sids_list),
        task=np.array(tasks_list)
    )
    print(f"[Cache 255] Saved {feat_arr.shape} embeddings to {out_file} ({time.time() - t0:.1f}s total)")


def cache_60_paper_photos():
    out_file = os.path.join(CACHE_DIR, "dinov2_ink_embeddings_paper60.npz")
    if os.path.exists(out_file):
        print(f"[Cache Paper 60] Found existing cache at {out_file}, skipping extraction.")
        return

    print("[Cache Paper 60] Starting DINOv2 ink-masked extraction on 60 paper photos...")
    manifest = pd.read_csv(MANIFEST_CSV)
    usable_60 = manifest[manifest["usable_for_correlation"] == True].copy()
    
    features_list = []
    sids_list = []
    paths_list = []
    t0 = time.time()

    for idx, (_, row) in enumerate(usable_60.iterrows()):
        sid = row['student_id']
        photo_path = row['photo_path']
        if not os.path.isabs(photo_path):
            photo_path = os.path.join(PROJECT_ROOT, photo_path)

        dino_res = extract_ink_masked_dinov2_features(photo_path)
        features_list.append(dino_res["combined_features"])
        sids_list.append(sid)
        paths_list.append(photo_path)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(usable_60):
            print(f"  Processed {idx + 1}/{len(usable_60)} paper photos ({time.time() - t0:.1f}s)")

    feat_arr = np.array(features_list, dtype=np.float32)
    np.savez_compressed(
        out_file,
        features=feat_arr,
        student_id=np.array(sids_list),
        photo_path=np.array(paths_list)
    )
    print(f"[Cache Paper 60] Saved {feat_arr.shape} embeddings to {out_file} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    cache_255_sessions()
    cache_60_paper_photos()
    print("ALL DINOv2 EMBEDDINGS CACHED SUCCESSFULLY!")

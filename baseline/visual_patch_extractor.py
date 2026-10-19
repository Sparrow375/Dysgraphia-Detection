"""Pretrained Vision Backbone & Patch Extraction Engine (Stream 4).

Extracts aspect-preserving square handwriting patches from sentence crops,
computes deep visual representations via pretrained DenseNet201 (1920-d),
and pools embeddings at sentence and student levels.

Saves persistent cache to data/features/densenet_embeddings.npz.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Standard ImageNet normalization for pretrained vision backbones
IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def extract_aspect_preserving_patches(
    image_gray: np.ndarray,
    target_size: int = 224,
    max_patches: int = 3,
) -> List[np.ndarray]:
    """Extract square aspect-preserving patches from a handwriting crop.

    Avoids horizontal squishing by slicing natural H x H squares along the line.

    Args:
        image_gray: 2D uint8 grayscale image crop.
        target_size: Output patch dimension (e.g., 224).
        max_patches: Maximum number of patches to extract (e.g. 2 or 3).

    Returns:
        List of (target_size, target_size, 3) float32 arrays in [0, 1].
    """
    if len(image_gray.shape) == 3:
        gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_gray.copy()

    # Standardize polarity: paper should be white/bright (~240), ink dark (~40)
    if np.mean(gray) < 128:
        gray = 255 - gray

    H, W = gray.shape[:2]
    if H < 10 or W < 10:
        patch = cv2.resize(gray, (target_size, target_size))
        rgb = np.repeat(patch[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
        return [rgb]

    raw_patches = []
    if W <= H:
        # Pad width symmetrically with white background to form a square
        pad_left = (H - W) // 2
        pad_right = H - W - pad_left
        bg_val = int(np.percentile(gray, 90))
        sq = cv2.copyMakeBorder(gray, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=bg_val)
        raw_patches.append(sq)
    else:
        # Line is wider than tall (standard horizontal text)
        # Extract square patches of size H x H along the horizontal strip
        if max_patches == 1 or W < 1.4 * H:
            # Single center square
            start_x = max(0, (W - H) // 2)
            raw_patches.append(gray[:, start_x : start_x + H])
        elif max_patches == 2 or W < 2.2 * H:
            # Left and Right squares
            raw_patches.append(gray[:, 0:H])
            raw_patches.append(gray[:, W - H : W])
        else:
            # 3 squares: Left, Center, Right
            mid_x = (W - H) // 2
            raw_patches.append(gray[:, 0:H])
            raw_patches.append(gray[:, mid_x : mid_x + H])
            raw_patches.append(gray[:, W - H : W])

    # Resize to target_size and format as 3-channel float32
    rgb_patches = []
    for p in raw_patches:
        p_resized = cv2.resize(p, (target_size, target_size), interpolation=cv2.INTER_AREA)
        rgb = np.repeat(p_resized[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
        rgb_patches.append(rgb)

    return rgb_patches


class DenseNetFeatureExtractor:
    """Frozen DenseNet201 feature extractor extracting 1920-d visual descriptors."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def extract_batch(self, patches_rgb: List[np.ndarray]) -> np.ndarray:
        """Extract 1920-d features for a batch of patches.

        Args:
            patches_rgb: List of (224, 224, 3) float32 arrays in [0, 1].

        Returns:
            (B, 1920) numpy array of pooled deep features.
        """
        if len(patches_rgb) == 0:
            return np.empty((0, 1920), dtype=np.float32)

        # Convert to torch tensor: (B, 3, 224, 224)
        t_list = []
        for p in patches_rgb:
            t = torch.from_numpy(p).permute(2, 0, 1)  # (3, H, W)
            t = IMAGENET_NORMALIZE(t)
            t_list.append(t)

        batch_tensor = torch.stack(t_list, dim=0).to(self.device)
        feats = self.model.features(batch_tensor)
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        return pooled.cpu().numpy()


def extract_and_cache_cohort_embeddings(
    processed_dir: str = "data/processed",
    manifest_csv: str = "data/manifest.csv",
    output_cache: str = "data/features/densenet_embeddings.npz",
    max_patches_per_sentence: int = 2,
    batch_size: int = 32,
    device: str = "cpu",
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """Extract and cache DenseNet201 visual features across the entire cohort.

    Returns dict mapping:
      - student_ids -> (N_students,)
      - student_embeddings -> (N_students, 1920)
      - student_std_embeddings -> (N_students, 1920)
    """
    out_path = Path(output_cache)
    if out_path.exists() and not force_recompute:
        print(f"Loading cached visual embeddings from: {out_path}")
        data = np.load(str(out_path), allow_pickle=True)
        return {k: data[k] for k in data.files}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    p_base = Path(processed_dir)
    json_paths = sorted(glob.glob(str(p_base / "*" / "*" / "sentence_*.json")))
    print(f"Found {len(json_paths)} sentence crops across cohort. Initializing DenseNet201 extractor on {device}...")

    extractor = DenseNetFeatureExtractor(device=device)

    # Group tasks by student
    student_sentences: Dict[str, List[Path]] = {}
    for jp in json_paths:
        p = Path(jp)
        sid = p.parent.name
        student_sentences.setdefault(sid, []).append(p)

    print(f"Aggregating visual patch embeddings for {len(student_sentences)} students...")

    student_means: Dict[str, np.ndarray] = {}
    student_stds: Dict[str, np.ndarray] = {}

    # Extract all patches in batches to maximize CPU vectorization
    all_student_ids = sorted(student_sentences.keys())

    for sid in tqdm(all_student_ids, desc="DenseNet201 cohort feature extraction"):
        s_paths = student_sentences[sid]
        patches = []
        for jp in s_paths:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    s_json = json.load(f)
                crop_fn = s_json.get("crop_filename", f"{jp.stem}.png")
                crop_path = jp.parent / crop_fn
                if crop_path.exists():
                    img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        s_patches = extract_aspect_preserving_patches(
                            img, target_size=224, max_patches=max_patches_per_sentence
                        )
                        patches.extend(s_patches)
            except Exception:
                continue

        if len(patches) > 0:
            # Process in sub-batches if needed
            feat_chunks = []
            for b in range(0, len(patches), batch_size):
                chunk = patches[b : b + batch_size]
                f_chunk = extractor.extract_batch(chunk)
                feat_chunks.append(f_chunk)
            all_feats = np.vstack(feat_chunks)  # (N_patches, 1920)

            # Student-level pooling
            student_means[sid] = np.mean(all_feats, axis=0)  # (1920,)
            student_stds[sid] = np.std(all_feats, axis=0) if len(all_feats) > 1 else np.zeros(1920, dtype=np.float32)
        else:
            student_means[sid] = np.zeros(1920, dtype=np.float32)
            student_stds[sid] = np.zeros(1920, dtype=np.float32)

    s_ids = np.array(all_student_ids)
    mean_matrix = np.array([student_means[s] for s in all_student_ids], dtype=np.float32)
    std_matrix = np.array([student_stds[s] for s in all_student_ids], dtype=np.float32)

    np.savez_compressed(
        str(out_path),
        student_ids=s_ids,
        student_embeddings=mean_matrix,
        student_std_embeddings=std_matrix,
    )
    print(f"Saved cohort visual embeddings to: {out_path} ({len(s_ids)} students, 1920 dims)")

    return {
        "student_ids": s_ids,
        "student_embeddings": mean_matrix,
        "student_std_embeddings": std_matrix,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and cache DenseNet201 visual features.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--output-cache", default="data/features/densenet_embeddings.npz")
    parser.add_argument("--max-patches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    extract_and_cache_cohort_embeddings(
        processed_dir=args.processed_dir,
        manifest_csv=args.manifest,
        output_cache=args.output_cache,
        max_patches_per_sentence=args.max_patches,
        batch_size=args.batch_size,
        force_recompute=args.force,
    )

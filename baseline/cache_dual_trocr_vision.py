"""Phase 2 & 3: Dual-Script TrOCR Vision Pre-Extraction & Caching with White-Canvas Physics.

Performs:
1. White-Canvas Rendering (384x384x3):
   - Ch1: Clean Inverted Binary Ink (255 background, 0 ink)
   - Ch2: Natural Grayscale Shading strictly inside ink (255 background, optical density inside ink)
   - Ch3: Clean Ink + 0.7 Gray Guideline Reference (255 background, 178 guideline, 0 ink)
2. Duplicate Bag Generation (3 bags per student):
   - Bag 1 (Clean Original): 100% untouched white-canvas crop
   - Bag 2 (Cutout Duplicate): 2 vertical whiteout strips (20px width)
   - Bag 3 (Micro-Shear Duplicate): Subtle affine shear (+/- 1.5 deg)
3. Dual-Script Routing:
   - Devanagari (Hindi) -> paudelanil/trocr-devanagari-2 (ViT, 768-d)
   - Latin (English) -> microsoft/trocr-small-handwritten (DeiT, 384-d)
4. Spatial Sequence Reshaping:
   - Patch tokens (24x24) -> Mean vertical pooling -> (24, 768) Hindi / (24, 384) English
5. Disk Caching:
   - Saves all sequence representations to data/features/dual_trocr_cached_tokens.npz (~180MB)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel

from baseline.train_whole_page_mil import FlushingStreamHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[FlushingStreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"
FEATURES_DIR = DATA_DIR / "features"
CACHE_NPZ_PATH = FEATURES_DIR / "dual_trocr_cached_tokens.npz"


def build_white_canvas_tensor(
    clean_mask_path: Path,
    raw_gray_path: Path,
    json_path: Path,
    target_size: int = 384,
) -> np.ndarray:
    """Construct custom 3-channel white-canvas tensor for TrOCR pretraining distribution.

    Returns:
        np.ndarray of shape (target_size, target_size, 3) with dtype uint8 (values 0..255).
        Background = 255 (pure white canvas).
        Ink = dark strokes (0..128).
    """
    # 1. Load clean ink mask
    mask_gray = cv2.imread(str(clean_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"Missing clean ink mask: {clean_mask_path}")
    ink_bin = (mask_gray < 128).astype(np.float32)

    # 2. Load raw grayscale crop
    raw_gray = cv2.imread(str(raw_gray_path), cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        raise FileNotFoundError(f"Missing raw gray crop: {raw_gray_path}")

    if ink_bin.shape != raw_gray.shape:
        ink_bin = cv2.resize(
            ink_bin, (raw_gray.shape[1], raw_gray.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # Standardize polarity (paper bright, ink dark)
    if np.mean(raw_gray) < 128:
        raw_gray = 255 - raw_gray

    h, w = raw_gray.shape

    # Channel 1: Clean Inverted Binary Ink (255 background, 0 ink)
    ch1 = np.where(ink_bin > 0, 0.0, 255.0).astype(np.float32)

    # Channel 2: Natural Normalized Grayscale Shading inside ink
    non_ink = raw_gray[ink_bin == 0]
    bg_level = float(np.percentile(non_ink, 90)) if len(non_ink) > 50 else 255.0
    bg_level = max(128.0, bg_level)
    optical_density = np.clip((bg_level - raw_gray.astype(np.float32)) / bg_level, 0.0, 1.0)
    # Background is 255; ink intensity scales inversely with optical density
    ink_intensity = 255.0 * (1.0 - optical_density)
    ch2 = np.where(ink_bin > 0, ink_intensity, 255.0).astype(np.float32)

    # Channel 3: Clean Ink + 0.7 Gray Ruled Guideline Reference
    rule_map = np.zeros((h, w), dtype=np.float32)
    has_rule = False

    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
            bbox = meta.get("crop_bbox", [0, 0, w, h])
            bx, by = bbox[0], bbox[1]
            lines = meta.get("physical_lines", [])
            for pl in lines:
                slope = pl.get("baseline_slope")
                intercept = pl.get("baseline_intercept")
                if slope is not None and intercept is not None:
                    xs = np.arange(w)
                    ys = np.round(slope * (xs + bx) + intercept - by).astype(int)
                    valid = (ys >= 0) & (ys < h)
                    for x_val, y_val in zip(xs[valid], ys[valid]):
                        y_lo = max(0, y_val - 1)
                        y_hi = min(h, y_val + 2)
                        rule_map[y_lo:y_hi, x_val] = 1.0
                    has_rule = True
        except Exception:
            pass

    if not has_rule or np.count_nonzero(rule_map) < 50:
        no_ink = raw_gray.copy()
        no_ink[ink_bin > 0] = 255
        inverted = 255 - no_ink
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1))
        opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        _, bin_rule = cv2.threshold(opened, 25, 1.0, cv2.THRESH_BINARY)
        rule_map = bin_rule.astype(np.float32)

    # Background = 255, guideline = 178 (0.7 intensity), ink = 0 over rule
    ch3 = np.full((h, w), 255.0, dtype=np.float32)
    ch3[rule_map > 0] = 178.0
    ch3[ink_bin > 0] = 0.0

    # Stack into (H, W, 3)
    t3 = np.stack([ch1, ch2, ch3], axis=-1)

    # Proportional aspect-preserving resize to fit within target_size x target_size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(t3, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pure white canvas of 255s
    canvas = np.full((target_size, target_size, 3), 255.0, dtype=np.float32)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized

    return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


def apply_vertical_cutout(img_uint8: np.ndarray, num_strips: int = 2, strip_w: int = 20) -> np.ndarray:
    """Bag 2 Augmentation: Apply vertical whiteout strips to force local loop inspection."""
    h, w, c = img_uint8.shape
    out = img_uint8.copy()
    min_x = int(0.15 * w)
    max_x = int(0.85 * w) - strip_w

    if max_x <= min_x:
        return out

    x_positions = np.random.choice(np.arange(min_x, max_x), size=min(num_strips, max_x - min_x), replace=False)
    for x in x_positions:
        out[:, x : x + strip_w, :] = 255  # Whiteout strip

    return out


def apply_micro_shear(img_uint8: np.ndarray, max_shear_deg: float = 1.5) -> np.ndarray:
    """Bag 3 Augmentation: Apply subtle affine shear jitter."""
    h, w, c = img_uint8.shape
    shear_angle = np.random.uniform(-max_shear_deg, max_shear_deg)
    shear_rad = np.deg2rad(shear_angle)
    m_shear = np.array([
        [1.0, float(np.tan(shear_rad)), 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)

    # Shift center so content doesn't drift off edge
    cx, cy = w / 2.0, h / 2.0
    m_shear[0, 2] = -m_shear[0, 1] * cy

    out = cv2.warpAffine(img_uint8, m_shear, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out


class DualScriptTrOCRExtractor:
    """Pre-extracts horizontal patch sequences using pretrained Hindi and English TrOCR encoders."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        logger.info("Loading English TrOCR encoder: microsoft/trocr-small-handwritten...")
        en_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
        self.en_encoder = en_model.encoder.to(device).eval()

        logger.info("Loading Hindi TrOCR encoder: paudelanil/trocr-devanagari-2...")
        hi_model = VisionEncoderDecoderModel.from_pretrained("paudelanil/trocr-devanagari-2")
        self.hi_encoder = hi_model.encoder.to(device).eval()

        # Free all encoder parameters
        for p in self.en_encoder.parameters():
            p.requires_grad = False
        for p in self.hi_encoder.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def extract_english_sequence(self, imgs_uint8: List[np.ndarray]) -> np.ndarray:
        """Args: List of (384, 384, 3) uint8 arrays. Returns: (B, 24, 384) float32 array."""
        t_batch = torch.from_numpy(np.stack(imgs_uint8, axis=0)).float().permute(0, 3, 1, 2)  # (B, 3, 384, 384)
        pixel_values = ((t_batch / 255.0) - 0.5) / 0.5
        pixel_values = pixel_values.to(self.device)

        out = self.en_encoder(pixel_values)
        tokens = out.last_hidden_state  # (B, 578, 384)

        # DeiT has [CLS] at 0 and [DIST] at 1; 576 patch tokens follow
        patches = tokens[:, 2:, :].view(-1, 24, 24, 384)  # (B, 24 rows, 24 cols, 384)
        seq = patches.mean(dim=1)  # Mean pool over height rows -> (B, 24 cols, 384)
        return seq.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def extract_hindi_sequence(self, imgs_uint8: List[np.ndarray]) -> np.ndarray:
        """Args: List of (384, 384, 3) uint8 arrays. Returns: (B, 24, 768) float32 array."""
        t_batch = torch.from_numpy(np.stack(imgs_uint8, axis=0)).float().permute(0, 3, 1, 2)  # (B, 3, 384, 384)
        pixel_values = ((t_batch / 255.0) - 0.5) / 0.5
        pixel_values = pixel_values.to(self.device)

        out = self.hi_encoder(pixel_values, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state  # (B, 577, 768)

        # ViT has [CLS] at 0; 576 patch tokens follow
        patches = tokens[:, 1:, :].view(-1, 24, 24, 768)  # (B, 24 rows, 24 cols, 768)
        seq = patches.mean(dim=1)  # Mean pool over height rows -> (B, 24 cols, 768)
        return seq.cpu().numpy().astype(np.float32)


def precompute_and_cache_dual_trocr_tokens(force: bool = False) -> Dict[str, np.ndarray]:
    """Precompute and cache dual-script TrOCR patch sequences for Clean, Cutout, and Shear bags."""
    if CACHE_NPZ_PATH.exists() and not force:
        logger.info("Loading cached dual-script TrOCR tokens from %s...", CACHE_NPZ_PATH)
        cached = np.load(CACHE_NPZ_PATH)
        logger.info("Loaded %d arrays from TrOCR cache successfully.", len(cached.files))
        return {k: cached[k] for k in cached.files}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DualScriptTrOCRExtractor(device)

    df_index = pd.read_csv(INDEX_CSV)
    logger.info("Building white-canvas crops and extracting TrOCR sequences for %d sentences...", len(df_index))

    t0 = time.time()
    cache_dict: Dict[str, np.ndarray] = {}

    batch_size = 32

    # Group tasks by script to batch efficiently
    for script in ["devanagari", "latin"]:
        sub_df = df_index[df_index["script"] == script].copy()
        logger.info("Processing script: %s (%d sentences)...", script, len(sub_df))

        uids = list(sub_df["task_uid"])
        rel_paths = list(sub_df["rel_path"])

        for i in range(0, len(sub_df), batch_size):
            batch_uids = uids[i : i + batch_size]
            batch_rel = rel_paths[i : i + batch_size]

            clean_imgs: List[np.ndarray] = []
            cutout_imgs: List[np.ndarray] = []
            shear_imgs: List[np.ndarray] = []

            for uid, rel in zip(batch_uids, batch_rel):
                raw_gray_path = SENTENCE_DATASET_DIR / rel
                clean_mask_path = SENTENCE_DATASET_DIR / "clean_ink_masks" / f"{uid}_clean_ink.png"
                json_path = SENTENCE_DATASET_DIR / "metadata" / f"{uid}.json"

                # Fallback paths
                if not clean_mask_path.exists():
                    clean_mask_path = raw_gray_path
                if not json_path.exists():
                    json_path = Path("non_existent_dummy_file.json")

                # 1. Clean Original
                img_clean = build_white_canvas_tensor(clean_mask_path, raw_gray_path, json_path, target_size=384)
                # 2. Cutout Duplicate
                img_cutout = apply_vertical_cutout(img_clean, num_strips=2, strip_w=20)
                # 3. Shear Duplicate
                img_shear = apply_micro_shear(img_clean, max_shear_deg=1.5)

                clean_imgs.append(img_clean)
                cutout_imgs.append(img_cutout)
                shear_imgs.append(img_shear)

            # Extract features in batches
            if script == "devanagari":
                seqs_clean = extractor.extract_hindi_sequence(clean_imgs)
                seqs_cutout = extractor.extract_hindi_sequence(cutout_imgs)
                seqs_shear = extractor.extract_hindi_sequence(shear_imgs)
            else:
                seqs_clean = extractor.extract_english_sequence(clean_imgs)
                seqs_cutout = extractor.extract_english_sequence(cutout_imgs)
                seqs_shear = extractor.extract_english_sequence(shear_imgs)

            for j, uid in enumerate(batch_uids):
                cache_dict[f"{uid}_clean"] = seqs_clean[j]
                cache_dict[f"{uid}_cutout"] = seqs_cutout[j]
                cache_dict[f"{uid}_shear"] = seqs_shear[j]

            if (i // batch_size + 1) % 5 == 0 or (i + batch_size) >= len(sub_df):
                logger.info("  Processed %d/%d %s crops...", min(i + batch_size, len(sub_df)), len(sub_df), script)

    elapsed = time.time() - t0
    logger.info("Extracted all TrOCR sequence tokens in %.1f seconds.", elapsed)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Saving cache to %s...", CACHE_NPZ_PATH)
    np.savez_compressed(CACHE_NPZ_PATH, **cache_dict)
    logger.info("Cached %d token arrays to %s (File size: %.2f MB)", len(cache_dict), CACHE_NPZ_PATH, CACHE_NPZ_PATH.stat().st_size / (1024 * 1024))

    return cache_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute and cache Dual-Script TrOCR vision features.")
    parser.add_argument("--force", action="store_true", help="Force recomputation of tokens.")
    args = parser.parse_args()

    precompute_and_cache_dual_trocr_tokens(force=args.force)


if __name__ == "__main__":
    main()


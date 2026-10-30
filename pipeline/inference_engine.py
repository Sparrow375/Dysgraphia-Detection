"""
Production Single-Image Inference Engine for Multilingual Dysgraphia Detection.

Extracts line-level handwriting tiles using true character height scaling (h -> 48px),
evaluates them through pretrained Vision Transformers (CLIP ViT-B/32, DINOv2-Small),
and infers calibrated dysgraphia risk using cross-validated probe head ensembles.

Supports:
- Configurations: 'no_guideline' (Champion), 'baseline', 'raw_ch1', 'raw_no_guideline'
- Models: 'clip', 'dinov2', 'dual_vit'
- Live manual crops from full scanned pages or direct line image uploads.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, CLIPVisionModelWithProjection

from baseline.tiling_preprocessor import (
    build_white_canvas_strip,
    estimate_strip_x_height,
    extract_tiles_from_strip,
)

logger = logging.getLogger("inference_engine")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
OOF_NPZ = FEATURES_DIR / "exp7_all_models_oof_probs.npz"


def compute_patch_descriptors(patch_tokens: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Mean, std, and GeM pooling across spatial patch tokens."""
    u = patch_tokens.mean(dim=1)
    var = torch.mean((patch_tokens - u.unsqueeze(1)) ** 2, dim=1)
    s = torch.sqrt(var + eps)
    gem = (torch.clamp(patch_tokens, min=eps).pow(p).mean(dim=1)).pow(1.0 / p)
    return torch.cat([u, s, gem], dim=-1)


def encode_image_base64(img: np.ndarray, format: str = "PNG") -> str:
    """Convert numpy image (RGB or BGR) to base64 data URI string."""
    if img.ndim == 3 and img.shape[2] == 3:
        # Assume RGB if format is RGB
        pil_img = Image.fromarray(img)
    elif img.ndim == 2:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


class InferenceEngine:
    """Singleton-style inference engine with cached backbones and probe ensembles."""

    _instance: Optional[InferenceEngine] = None

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Initializing InferenceEngine on device: %s", self.device)

        # Backbones (lazy loaded)
        self.clip_model: Optional[CLIPVisionModelWithProjection] = None
        self.dinov2_model: Optional[AutoModel] = None

        # Normalization constants
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Cache for fitted probe heads per (backbone, tag)
        self.probe_heads: Dict[str, List[Tuple[StandardScaler, PCA, LogisticRegression]]] = {}
        self.cohort_scores: Dict[str, np.ndarray] = {}

    @classmethod
    def get_instance(cls) -> InferenceEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_clip(self):
        if self.clip_model is None:
            logger.info("Loading CLIP Vision backbone (openai/clip-vit-base-patch32)...")
            self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            self.clip_model.eval()

    def _load_dinov2(self):
        if self.dinov2_model is None:
            logger.info("Loading DINOv2-Small backbone (facebook/dinov2-small)...")
            self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-small").to(self.device)
            self.dinov2_model.eval()

    def _get_probe_heads(self, backbone: str, tag: str) -> List[Tuple[StandardScaler, PCA, LogisticRegression]]:
        """Fit and cache 5x5 fold probe heads on reference cohort features."""
        key = f"{backbone}_{tag}"
        if key in self.probe_heads:
            return self.probe_heads[key]

        logger.info("Fitting probe head ensemble for %s...", key)
        feat_name = f"exp7_line_{backbone}_tiles_{tag}.npz"
        feat_path = FEATURES_DIR / feat_name
        if not feat_path.exists():
            # Fallback for default tag
            feat_name_default = f"exp7_line_{backbone}_tiles.npz"
            feat_path = FEATURES_DIR / feat_name_default

        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feat_path}")

        oof_data = np.load(OOF_NPZ)
        sids_benchmark = list(oof_data["student_ids"])
        y_true = oof_data["y_true"]

        raw = np.load(feat_path)
        sids_map = {sid: i for i, sid in enumerate(raw["student_ids"])}
        align_idx = [sids_map[s] for s in sids_benchmark]

        X_cohort = raw["joint_student_features"][align_idx]

        # 5x5 Repeated Outer Fold Heads
        heads = []
        cohort_preds = np.zeros(len(y_true), dtype=np.float32)

        # Optimal k candidates
        k_pca = 24 if backbone == "clip" else 32
        c_lr = 0.1

        for r in range(5):
            fold_col = raw[f"repeat_{r}_fold"][align_idx]
            for f in range(5):
                te_mask = (fold_col == f)
                tr_mask = (fold_col != f)
                seed = 42 + r * 10 + f

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_cohort[tr_mask])
                pca = PCA(n_components=min(k_pca, X_tr_s.shape[0], X_tr_s.shape[1]), random_state=seed)
                X_tr_pca = pca.fit_transform(X_tr_s)

                lr = LogisticRegression(C=c_lr, class_weight="balanced", max_iter=1000, random_state=seed)
                lr.fit(X_tr_pca, y_true[tr_mask])

                # OOF prediction
                X_te_s = scaler.transform(X_cohort[te_mask])
                X_te_pca = pca.transform(X_te_s)
                cohort_preds[te_mask] += lr.predict_proba(X_te_pca)[:, 1] / 5.0

                heads.append((scaler, pca, lr))

        self.probe_heads[key] = heads
        self.cohort_scores[key] = cohort_preds
        logger.info("Cached %d probe heads for %s (cohort AUROC: %.4f)", len(heads), key, float(np.mean(cohort_preds)))
        return heads

    def extract_crop_tiles(
        self,
        image_crop: np.ndarray,
        ch1_mode: str = "binary",
        include_guideline: bool = False,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Preprocess single image crop, extract tiles with h->48px scaling and rejection gate."""
        if image_crop.ndim == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop.copy()

        # Clean ink binarization (gentle Otsu with polarity check)
        if np.mean(gray) < 128:
            gray = 255 - gray

        # Otsu thresholding for clean ink mask
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bin_ink = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_mask = bin_ink.astype(np.uint8)

        # Estimate character x-height h
        h_char = estimate_strip_x_height(ink_mask, default=40.0)

        # Build 3-channel strip
        strip = build_white_canvas_strip(
            clean_mask_or_path=ink_mask,
            raw_gray_or_path=gray,
            ch1_mode=ch1_mode,
            include_guideline=include_guideline,
        )

        # Slicing with rejection gate
        tiles, coords = extract_tiles_from_strip(
            strip,
            target_height=224,
            stride=112,
            x_height=h_char,
            target_x_height=48.0,
            min_ink_pixels=1250,
            min_vertical_span=28,
            min_vertical_std=8.0,
            ink_mask=ink_mask,
            allow_empty=True,
        )

        meta = {
            "x_height": float(h_char),
            "original_shape": list(gray.shape),
            "tile_count": len(tiles),
            "coords": coords,
            "ch1_mode": ch1_mode,
            "include_guideline": include_guideline,
        }
        return tiles, meta

    def infer_crop(
        self,
        image_crop: np.ndarray,
        config: str = "no_guideline",
        model_name: str = "all",
    ) -> Dict[str, Any]:
        """Run end-to-end inference on a single handwriting crop.

        Args:
            image_crop: Grayscale or BGR image of the handwriting crop.
            config: 'no_guideline' (Champion), 'baseline', 'raw_ch1', 'raw_no_guideline'
            model_name: 'clip', 'dinov2', 'dual_vit', or 'all'

        Returns:
            Dict containing scores, risk percentile, tier, and preview tile thumbnails.
        """
        # Map config to preprocessor parameters
        if config == "no_guideline":
            ch1_mode = "binary"
            include_guideline = False
            tag = "no_guideline"
        elif config == "raw_ch1":
            ch1_mode = "raw"
            include_guideline = True
            tag = "raw_ch1"
        elif config == "raw_no_guideline":
            ch1_mode = "raw"
            include_guideline = False
            tag = "raw_no_guideline"
        else:  # baseline
            ch1_mode = "binary"
            include_guideline = True
            tag = "baseline"

        tiles, meta = self.extract_crop_tiles(image_crop, ch1_mode=ch1_mode, include_guideline=include_guideline)

        if len(tiles) == 0:
            return {
                "success": False,
                "error": "No valid handwriting strokes detected in selection (check crop margins or contrast).",
                "meta": meta,
            }

        # Convert tiles to tensor batch (N, 3, 224, 224)
        tiles_np = np.stack(tiles, axis=0)  # (N, 224, 224, 3)
        tiles_t = torch.from_numpy(tiles_np).permute(0, 3, 1, 2).float().to(self.device) / 255.0

        scores: Dict[str, float] = {}

        # 1. CLIP Inference
        if model_name in ("clip", "dual_vit", "all"):
            self._load_clip()
            heads = self._get_probe_heads("clip", tag)
            with torch.no_grad():
                norm_tiles = (tiles_t - self.clip_mean) / self.clip_std
                out = self.clip_model.vision_model(norm_tiles)
                patch_tokens = out.last_hidden_state[:, 1:, :]
                tile_desc = compute_patch_descriptors(patch_tokens).cpu().numpy()

            # Aggregate across tiles: [mean, std, p90]
            desc_mean = tile_desc.mean(axis=0)
            desc_std = tile_desc.std(axis=0) if len(tile_desc) > 1 else np.zeros_like(desc_mean)
            desc_p90 = np.percentile(tile_desc, 90, axis=0)
            x_crop = np.concatenate([desc_mean, desc_std, desc_p90]).reshape(1, -1)

            # Predict across 25 fold heads
            clip_probs = []
            for scaler, pca, lr in heads:
                x_s = scaler.transform(x_crop)
                x_p = pca.transform(x_s)
                clip_probs.append(lr.predict_proba(x_p)[0, 1])
            scores["clip_score"] = float(np.mean(clip_probs))

        # 2. DINOv2 Inference
        if model_name in ("dinov2", "dual_vit", "all"):
            self._load_dinov2()
            heads = self._get_probe_heads("dinov2_small", tag)
            with torch.no_grad():
                norm_tiles = (tiles_t - self.dino_mean) / self.dino_std
                out = self.dinov2_model(norm_tiles)
                patch_tokens = out.last_hidden_state[:, 1:, :]
                tile_desc = compute_patch_descriptors(patch_tokens).cpu().numpy()

            # Aggregate across tiles
            desc_mean = tile_desc.mean(axis=0)
            desc_std = tile_desc.std(axis=0) if len(tile_desc) > 1 else np.zeros_like(desc_mean)
            desc_p90 = np.percentile(tile_desc, 90, axis=0)
            x_crop = np.concatenate([desc_mean, desc_std, desc_p90]).reshape(1, -1)

            dino_probs = []
            for scaler, pca, lr in heads:
                x_s = scaler.transform(x_crop)
                x_p = pca.transform(x_s)
                dino_probs.append(lr.predict_proba(x_p)[0, 1])
            scores["dinov2_score"] = float(np.mean(dino_probs))

        # 3. Dual-ViT Consensus
        if "clip_score" in scores and "dinov2_score" in scores:
            # Dual-ViT 50/50 consensus
            # Compute cohort rank percentile for each
            clip_cohort = self.cohort_scores[f"clip_{tag}"]
            dino_cohort = self.cohort_scores[f"dinov2_small_{tag}"]
            r_clip = float(np.mean(clip_cohort <= scores["clip_score"]))
            r_dino = float(np.mean(dino_cohort <= scores["dinov2_score"]))
            dual_rank = 0.50 * r_clip + 0.50 * r_dino
            scores["dual_vit_rank"] = float(dual_rank)
            scores["primary_score"] = float(scores["clip_score"])  # CLIP is champion backbone
            overall_percentile = float(dual_rank * 100.0)
        elif "clip_score" in scores:
            scores["primary_score"] = float(scores["clip_score"])
            clip_cohort = self.cohort_scores[f"clip_{tag}"]
            overall_percentile = float(np.mean(clip_cohort <= scores["clip_score"]) * 100.0)
        else:
            scores["primary_score"] = float(scores["dinov2_score"])
            dino_cohort = self.cohort_scores[f"dinov2_small_{tag}"]
            overall_percentile = float(np.mean(dino_cohort <= scores["dinov2_score"]) * 100.0)

        # Clinical Tiers
        if overall_percentile >= 90.0:
            tier = "TIER_2_REFERRAL"
            tier_label = "Tier 2: Specialist Referral (High Risk)"
            tier_badge = "red"
        elif overall_percentile >= 80.0:
            tier = "TIER_1_ALERT"
            tier_label = "Tier 1: Community Triage Alert (Moderate Risk)"
            tier_badge = "yellow"
        else:
            tier = "TIER_1_RULE_OUT"
            tier_label = "Tier 1: Safe Rule-Out (Low Risk / Clean)"
            tier_badge = "green"

        # Generate base64 previews for top 4 tiles
        preview_tiles = []
        for i, tile in enumerate(tiles[:4]):
            tile_rgb = tile[:, :, :3]
            ch1_gray = tile[:, :, 0]
            ch2_gray = tile[:, :, 1]
            ch3_gray = tile[:, :, 2]
            preview_tiles.append({
                "tile_index": i,
                "rgb_b64": encode_image_base64(tile_rgb),
                "ch1_b64": encode_image_base64(ch1_gray),
                "ch2_b64": encode_image_base64(ch2_gray),
                "ch3_b64": encode_image_base64(ch3_gray),
            })

        crop_b64 = encode_image_base64(image_crop)

        return {
            "success": True,
            "config": config,
            "scores": scores,
            "overall_percentile": round(overall_percentile, 1),
            "tier": tier,
            "tier_label": tier_label,
            "tier_badge": tier_badge,
            "meta": meta,
            "crop_b64": crop_b64,
            "preview_tiles": preview_tiles,
        }

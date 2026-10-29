"""
Foreground Ink-Masked DINOv2 Feature Extractor (Step 1)
Integrates pre-trained Vision Transformer (dinov2_vits14) to extract optical
neuromotor signatures (ink pooling, hesitation teardrops, edge tapering, and micro-roughness)
restricted strictly to foreground ink patches.
"""

import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Union, Tuple, Dict, Any, Optional
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from src.core.line_removal import detect_and_remove_ruled_lines

_DINOV2_MODEL = None
_DEVICE = None


def get_dinov2_model(device_str: Optional[str] = None):
    """Loads and caches frozen dinov2_vits14 model on GPU/CPU."""
    global _DINOV2_MODEL, _DEVICE
    if _DINOV2_MODEL is None:
        if device_str is not None:
            _DEVICE = torch.device(device_str)
        else:
            _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load from torch.hub with cached weights
        _DINOV2_MODEL = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        _DINOV2_MODEL.to(_DEVICE).eval()
        for p in _DINOV2_MODEL.parameters():
            p.requires_grad = False
            
    return _DINOV2_MODEL, _DEVICE


# ImageNet normalization standard for DINOv2
DINO_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image_for_dinov2(
    image_input: Union[str, os.PathLike, np.ndarray],
    target_size: int = 518
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Stage 1 illumination normalization and ruled-line removal,
    then formats the image and binary ink mask into target_size x target_size.
    Returns: (rgb_normalized, clean_binary_resized, raw_gray)
    """
    if isinstance(image_input, (str, os.PathLike)):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at {image_input}")
        im_pil = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            im_pil = Image.fromarray(image_input).convert("L")
        else:
            im_pil = Image.fromarray(image_input.astype(np.uint8))
    else:
        raise TypeError("image_input must be a file path or numpy array")

    # Standardize scale
    im_pil.thumbnail((1200, 1200))
    arr_gray = np.array(im_pil)

    # Shading normalization
    bg = gaussian_filter(arr_gray.astype(float), sigma=25.0)
    norm_gray = np.clip((arr_gray.astype(float) / np.maximum(bg, 1.0)) * 255.0, 0, 255).astype(np.uint8)

    # Adaptive Otsu + ruled line clean
    thresh = threshold_otsu(norm_gray)
    raw_binary = (norm_gray < thresh).astype(np.uint8)
    clean_binary, _ = detect_and_remove_ruled_lines(raw_binary)

    # Resize to exact target_size (518 x 518)
    pil_gray = Image.fromarray(norm_gray).resize((target_size, target_size), Image.Resampling.BILINEAR)
    pil_bin = Image.fromarray((clean_binary * 255).astype(np.uint8)).resize((target_size, target_size), Image.Resampling.NEAREST)

    arr_clean_bin = (np.array(pil_bin) > 128).astype(np.float32)
    # Convert grayscale into 3-channel RGB for DINOv2
    arr_rgb = np.stack([np.array(pil_gray)] * 3, axis=-1)

    return arr_rgb, arr_clean_bin, arr_gray


def extract_ink_masked_dinov2_features(
    image_input: Union[str, os.PathLike, np.ndarray],
    target_size: int = 518,
    patch_size: int = 14,
    min_ink_coverage: float = 0.04
) -> Dict[str, Any]:
    """
    Extracts foreground ink-masked DINOv2 visual features from an image.
    
    Returns a dict with:
      - 'mean_ink_token': (384,)
      - 'std_ink_token': (384,)
      - 'idr_ink_token': (384,) [q90 - q10]
      - 'cls_token': (384,)
      - 'combined_features': (1536,)
      - 'patch_activation_map': (37, 37) 2D heatmap for visualization
      - 'active_patch_count': int
    """
    model, device = get_dinov2_model()

    arr_rgb, clean_binary, arr_gray = preprocess_image_for_dinov2(image_input, target_size=target_size)

    # Downsample clean_binary mask to patch grid
    grid_dim = target_size // patch_size  # e.g., 37
    t_mask = torch.from_numpy(clean_binary)
    # (37, 14, 37, 14) -> mean over patch pixels
    patch_coverage = t_mask.view(grid_dim, patch_size, grid_dim, patch_size).mean(dim=(1, 3)).numpy()

    # Active ink patches
    active_mask = patch_coverage > min_ink_coverage
    if active_mask.sum() < 8:
        # Fallback if writing is sparse: take top 5% patches with highest ink density
        flat_cov = patch_coverage.flatten()
        top_indices = np.argsort(flat_cov)[-max(15, int(0.05 * len(flat_cov))):]
        active_mask = np.zeros_like(flat_cov, dtype=bool)
        active_mask[top_indices] = True
        active_mask = active_mask.reshape((grid_dim, grid_dim))

    active_mask_tensor = torch.from_numpy(active_mask.flatten()).to(device)

    # Transform image for PyTorch
    pil_rgb = Image.fromarray(arr_rgb)
    img_tensor = DINO_TRANSFORM(pil_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.forward_features(img_tensor)
        patch_tokens = out['x_norm_patchtokens'][0]  # (N_patches, 384)
        cls_token = out['x_norm_clstoken'][0]        # (384,)

    ink_tokens = patch_tokens[active_mask_tensor]  # (N_active, 384)

    mean_tok = ink_tokens.mean(dim=0).cpu().numpy()
    std_tok = ink_tokens.std(dim=0).cpu().numpy()

    q90 = torch.quantile(ink_tokens, 0.90, dim=0)
    q10 = torch.quantile(ink_tokens, 0.10, dim=0)
    idr_tok = (q90 - q10).cpu().numpy()
    cls_tok = cls_token.cpu().numpy()

    # Compute patch activation map for diagnostic inspection (norm of patch tokens modulated by ink)
    token_norms = torch.norm(patch_tokens, dim=-1).cpu().numpy().reshape((grid_dim, grid_dim))
    # Emphasize active ink regions
    activation_map = token_norms * (0.3 + 0.7 * patch_coverage)

    combined = np.concatenate([mean_tok, std_tok, idr_tok, cls_tok])

    return {
        "mean_ink_token": mean_tok,
        "std_ink_token": std_tok,
        "idr_ink_token": idr_tok,
        "cls_token": cls_tok,
        "combined_features": combined,
        "patch_activation_map": activation_map,
        "active_patch_count": int(active_mask.sum()),
        "grid_dim": grid_dim
    }


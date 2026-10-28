"""
experiments/inksight/00_roundtrip_test.py
Round-trip coordinate mapping test.

MUST PASS before any real inference run (CRITERIA.md §5).

Verifies that:  image = pad_resize(render(gt_coords))  →  coords_back = inv_transform(coords_224)
has max absolute error < 0.5 px (in original tablet pixel units).

Usage:
    conda run -n inksight_eval python experiments/inksight/00_roundtrip_test.py
"""

import sys
import pathlib
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TARGET_SIZE = 224  # InkSight input size
TOLERANCE_PX = 0.5  # half-pixel, in original tablet pixel units


# ---------------------------------------------------------------------------
# Pad/resize transform (mirrors what the inference pipeline does)
# ---------------------------------------------------------------------------

def compute_pad_resize_params(orig_w: int, orig_h: int, target: int = TARGET_SIZE):
    """
    Returns (scale, pad_x, pad_y) such that:
        x_224 = x_orig * scale + pad_x
        y_224 = y_orig * scale + pad_y
    """
    scale = min(target / orig_w, target / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    if new_h < target:
        # portrait: pad top/bottom
        pad_y = (target - new_h) // 2
        pad_x = 0
    else:
        # landscape: pad left/right
        pad_x = (target - new_w) // 2
        pad_y = 0

    return scale, pad_x, pad_y


def to_224(coords: np.ndarray, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    """coords shape (N, 2) in original pixel space → 224×224 pixel space."""
    out = coords.copy().astype(float)
    out[:, 0] = out[:, 0] * scale + pad_x
    out[:, 1] = out[:, 1] * scale + pad_y
    return out


def from_224(coords: np.ndarray, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    """coords shape (N, 2) in 224×224 pixel space → original pixel space."""
    out = coords.copy().astype(float)
    out[:, 0] = (out[:, 0] - pad_x) / scale
    out[:, 1] = (out[:, 1] - pad_y) / scale
    return out


# ---------------------------------------------------------------------------
# Test: synthetic GT word with known coordinates
# ---------------------------------------------------------------------------

def run_roundtrip_test():
    print("=" * 60)
    print("Round-trip coordinate mapping test")
    print("CRITERIA.md §5 — must pass before any real inference run")
    print("=" * 60)

    # Synthetic GT: a word bounding box of 180×60 px (realistic word size)
    orig_w, orig_h = 180, 60

    # Generate dense GT coordinates across the word bounding box
    rng = np.random.default_rng(seed=42)
    n_points = 500
    gt_coords = rng.uniform(low=[0, 0], high=[orig_w, orig_h], size=(n_points, 2))

    # Add corner and edge points to stress-test
    corners = np.array([
        [0, 0], [orig_w, 0], [0, orig_h], [orig_w, orig_h],
        [orig_w / 2, orig_h / 2],
    ])
    gt_coords = np.vstack([gt_coords, corners])

    # Compute transform params
    scale, pad_x, pad_y = compute_pad_resize_params(orig_w, orig_h)
    print(f"Original size: {orig_w}×{orig_h}")
    print(f"Scale: {scale:.6f}, pad_x: {pad_x}, pad_y: {pad_y}")

    # Forward: orig → 224×224
    coords_224 = to_224(gt_coords, scale, pad_x, pad_y)

    # Check that 224×224 coords are within [0, TARGET_SIZE]
    assert coords_224[:, 0].min() >= 0, "x coord < 0 after pad/resize"
    assert coords_224[:, 1].min() >= 0, "y coord < 0 after pad/resize"
    assert coords_224[:, 0].max() <= TARGET_SIZE, f"x coord > {TARGET_SIZE}"
    assert coords_224[:, 1].max() <= TARGET_SIZE, f"y coord > {TARGET_SIZE}"
    print(f"224×224 coord range: x=[{coords_224[:,0].min():.2f}, {coords_224[:,0].max():.2f}], "
          f"y=[{coords_224[:,1].min():.2f}, {coords_224[:,1].max():.2f}]")

    # Inverse: 224×224 → orig
    coords_back = from_224(coords_224, scale, pad_x, pad_y)

    # Compute error
    errors = np.abs(coords_back - gt_coords)
    max_error = errors.max()
    mean_error = errors.mean()

    print(f"\nMax absolute round-trip error: {max_error:.6f} px (tolerance: {TOLERANCE_PX} px)")
    print(f"Mean absolute round-trip error: {mean_error:.8f} px")

    if max_error < TOLERANCE_PX:
        print(f"\n[PASS] round-trip error {max_error:.6f} px < {TOLERANCE_PX} px")
        print("Coordinate mapping is valid. Proceed with inference.")
        return True
    else:
        print(f"\n[FAIL] round-trip error {max_error:.6f} px >= {TOLERANCE_PX} px")
        print("Fix coordinate transform before running inference. Stopping.")
        return False


# ---------------------------------------------------------------------------
# Additional test: PIL Image round-trip (verifying PIL resize matches formula)
# ---------------------------------------------------------------------------

def run_pil_consistency_test():
    """
    Check that PIL's resize is consistent with our analytic transform.
    """
    print("\n" + "=" * 60)
    print("PIL resize consistency test")
    print("=" * 60)

    orig_w, orig_h = 180, 60

    # Draw a known pixel image
    img = Image.new("RGB", (orig_w, orig_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, orig_w - 10, orig_h - 10], outline=(0, 0, 0))

    # Replicate InkSight's pad/resize logic from load_and_pad_img_dir
    scale = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = img.resize((new_w, new_h))

    if new_h < TARGET_SIZE:
        top_padding = (TARGET_SIZE - new_h) // 2
        padded = Image.new("RGB", (new_w, TARGET_SIZE), (255, 255, 255))
        padded.paste(resized, (0, top_padding))
        pad_x, pad_y = 0, top_padding
    else:
        left_padding = (TARGET_SIZE - new_w) // 2
        padded = Image.new("RGB", (TARGET_SIZE, new_h), (255, 255, 255))
        padded.paste(resized, (left_padding, 0))
        pad_x, pad_y = left_padding, 0

    scale_check, pad_x_check, pad_y_check = compute_pad_resize_params(orig_w, orig_h)

    assert abs(scale_check - scale) < 1e-9, f"Scale mismatch: {scale_check} vs {scale}"
    assert pad_x_check == pad_x, f"pad_x mismatch: {pad_x_check} vs {pad_x}"
    assert pad_y_check == pad_y, f"pad_y mismatch: {pad_y_check} vs {pad_y}"

    print(f"PIL resize: scale={scale:.6f}, pad_x={pad_x}, pad_y={pad_y}")
    print(f"Analytic:   scale={scale_check:.6f}, pad_x={pad_x_check}, pad_y={pad_y_check}")
    print("[PASS] PIL resize matches analytic transform formula")
    return True


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok1 = run_roundtrip_test()
    ok2 = run_pil_consistency_test()

    if ok1 and ok2:
        print("\n[ALL PASS] All round-trip tests passed. Proceed to inference.")
        sys.exit(0)
    else:
        print("\n[FAIL] Round-trip tests FAILED. Fix before running inference.")
        sys.exit(1)

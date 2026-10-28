"""
experiments/inksight/02_run_inference.py
Run InkSight and Skeleton inference on prepared word images.

Reads manifest from 01_prepare_data.py output.
For each sample:
  - Runs InkSight Small-p (prompt: "Derender the ink.")
  - Runs our skeleton-based recovery on the same 224×224 image
  - Saves predicted strokes as NPZ files

CRITERIA.md rules enforced:
  - Both methods receive identical 224×224 padded images (§3 fairness)
  - InkSight: "Derender the ink." prompt (spatial-only, no OCR)
  - Empty output (0 strokes) recorded as inf/empty, not dropped (§13)
  - NO velocity/jerk/NVI computed here (§12)

Usage:
    # Quick test-set clean run (~5-7 mins):
    wsl -d Ubuntu-22.04 -- bash experiments/inksight/run_in_wsl.sh experiments/inksight/02_run_inference.py --split test --variant clean

    # All test-set variants:
    wsl -d Ubuntu-22.04 -- bash experiments/inksight/run_in_wsl.sh experiments/inksight/02_run_inference.py --split test

    # Full benchmark (all 650 images):
    wsl -d Ubuntu-22.04 -- bash experiments/inksight/run_in_wsl.sh experiments/inksight/02_run_inference.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import re
import pathlib
import csv
import time
import argparse
import subprocess
import logging
import numpy as np
from PIL import Image

logging.getLogger("tensorflow").setLevel(logging.ERROR)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.core.preprocessing import load_and_binarize, zhang_suen_skeletonize
from src.core.line_removal import detect_and_remove_ruled_lines
from src.branch_b.stroke_recovery import recover_handwriting_trajectory

MANIFEST_PATH = REPO_ROOT / "experiments" / "inksight" / "data" / "manifest.csv"
OUT_DIR = REPO_ROOT / "experiments" / "inksight" / "predictions"
OUT_DIR_IS = OUT_DIR / "inksight"
OUT_DIR_SK = OUT_DIR / "skeleton"
LOG_PATH = OUT_DIR / "inference_log.csv"

TARGET_SIZE = 224
INKSIGHT_PROMPT = "Derender the ink."


# ---------------------------------------------------------------------------
# Official Google Research InkSight Token Decoder
# ---------------------------------------------------------------------------

def text_to_tokens(text: str) -> list[int]:
    pattern = r"<ink_token_(\d+)>"
    matches = re.findall(pattern, text)
    return [int(tok) for tok in matches]


def detokenize(tokens: list[int]) -> list[np.ndarray]:
    coordinate_length = 224
    num_token_per_dimension = coordinate_length + 1  # 225
    vocabulary_size = num_token_per_dimension * 2 + 1  # 451
    start_token = num_token_per_dimension * 2  # 450

    if any([t < 0 or t >= vocabulary_size for t in tokens]):
        return []

    idx = 0
    res = []
    current_stroke = []

    while idx < len(tokens):
        token = tokens[idx]
        if token == start_token:
            if current_stroke:
                res.append(current_stroke)
            current_stroke = []
            idx += 1
        else:
            if idx + 1 < len(tokens) and (tokens[idx + 1] != start_token):
                x = tokens[idx]
                y = tokens[idx + 1] - num_token_per_dimension
                if (0 <= x <= coordinate_length) and (0 <= y <= coordinate_length):
                    current_stroke.append((float(x), float(y)))
                idx += 2
            else:
                idx += 1
    if current_stroke:
        res.append(current_stroke)

    strokes = [np.array(s, dtype=np.float32) for s in res if len(s) >= 2]
    return strokes


# ---------------------------------------------------------------------------
# Model Loader & Inference
# ---------------------------------------------------------------------------

def load_inksight_model():
    """Load InkSight Small-p SavedModel."""
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    import tensorflow as tf
    import tensorflow_text  # noqa: F401
    from huggingface_hub import snapshot_download

    print("Loading InkSight Small-p SavedModel on GPU...", flush=True)
    model_path = snapshot_download("Derendering/InkSight-Small-p")
    model = tf.saved_model.load(model_path)
    cf = model.signatures["serving_default"]
    print("InkSight model loaded successfully.\n", flush=True)
    return cf, tf


def run_inksight(image_path: pathlib.Path, cf, tf_module) -> tuple[list[np.ndarray], int]:
    """
    Run InkSight on a 224×224 image.
    Returns:
        (strokes, token_count)
    """
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)[:, :, :3]
    image_encoded = tf_module.reshape(tf_module.io.encode_jpeg(img_arr), (1, 1))
    input_text = tf_module.constant([INKSIGHT_PROMPT], dtype=tf_module.string)

    try:
        raw_out = cf(**{"input_text": input_text, "image/encoded": image_encoded})
        val = raw_out["output_0"].numpy()
        if isinstance(val, np.ndarray):
            raw_bytes = val.flat[0]
        else:
            raw_bytes = val
        if isinstance(raw_bytes, bytes):
            output_text = raw_bytes.decode("utf-8", errors="ignore")
        else:
            output_text = str(raw_bytes)
        tokens = text_to_tokens(output_text)
        strokes = detokenize(tokens)
        return strokes, len(tokens)
    except Exception as e:
        print(f"\n    InkSight error on {image_path.name}: {e}", flush=True)
        return [], 0


def run_skeleton(image_path: pathlib.Path, h_med: float = 25.0) -> list[np.ndarray]:
    """
    Run Skeleton-based trajectory recovery pipeline on a 224×224 image.
    Returns:
        strokes: list of np.ndarray shape (N, 2)
    """
    try:
        binary = load_and_binarize(str(image_path))
        cleaned_mask, _ = detect_and_remove_ruled_lines(binary)
        skeleton = zhang_suen_skeletonize(cleaned_mask)
        strokes = recover_handwriting_trajectory(skeleton, h_med=h_med)
        return strokes
    except Exception as e:
        print(f"\n    Skeleton error on {image_path.name}: {e}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run InkSight & Skeleton trajectory recovery inference.")
    parser.add_argument("--split", choices=["all", "test", "train"], default="all",
                        help="Filter by split (default: all)")
    parser.add_argument("--variant", choices=["all", "clean", "ruled", "grid", "blur", "camera"], default="all",
                        help="Filter by variant (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total number of samples (default: all)")
    args = parser.parse_args()

    # 1. Coordinate Round-Trip Verification
    print("Checking coordinate round-trip test...", flush=True)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "inksight" / "00_roundtrip_test.py")],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR: Round-trip test failed. Coordinate transform must pass first.")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print("Round-trip test verified (max error < 0.5px).\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR_IS.mkdir(exist_ok=True)
    OUT_DIR_SK.mkdir(exist_ok=True)

    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, newline="") as f:
        all_manifest = list(csv.DictReader(f))

    # Apply filters
    filtered_manifest = all_manifest
    if args.split != "all":
        filtered_manifest = [r for r in filtered_manifest if r["split"] == args.split]
    if args.variant != "all":
        filtered_manifest = [r for r in filtered_manifest if r["variant"] == args.variant]
    if args.max_samples is not None and args.max_samples > 0:
        filtered_manifest = filtered_manifest[:args.max_samples]

    total_samples = len(filtered_manifest)
    print("=" * 70)
    print(f"INFERENCE CONFIGURATION:")
    print(f"  Split filter:   {args.split}")
    print(f"  Variant filter: {args.variant}")
    print(f"  Total samples:  {total_samples} (out of {len(all_manifest)} in manifest)")
    print("=" * 70, flush=True)

    if total_samples == 0:
        print("No samples match the selected filter. Exiting.")
        return

    # 2. Load InkSight
    cf, tf_module = load_inksight_model()

    # Load existing log rows if present to allow incremental updates
    existing_log = {}
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    existing_log[row["sample_key"]] = row
        except Exception:
            pass

    start_time = time.time()
    for i, row in enumerate(filtered_manifest):
        sample_start = time.time()
        img_path = REPO_ROOT / row["image_path"]
        subj = row["subject_id"]
        word_id = row["word_id"]
        variant = row["variant"]
        sample_key = f"{subj}_{word_id}_{variant}"

        # Target save paths
        is_path = OUT_DIR_IS / f"{sample_key}.npz"
        sk_path = OUT_DIR_SK / f"{sample_key}.npz"

        # InkSight inference (with caching)
        if not is_path.exists():
            is_strokes, is_token_cnt = run_inksight(img_path, cf, tf_module)
            np.savez_compressed(
                str(is_path),
                n_strokes=len(is_strokes),
                tokens=is_token_cnt,
                strokes=np.array(is_strokes, dtype=object)
            )
            is_cached = False
        else:
            data = np.load(str(is_path), allow_pickle=True)
            is_strokes = list(data["strokes"])
            is_token_cnt = int(data["tokens"]) if "tokens" in data else 0
            is_cached = True

        # Skeleton inference (with caching)
        try:
            h_med = float(row.get("h_med_subj", 25.0)) * float(row.get("scale", 1.0))
            if h_med <= 0 or np.isnan(h_med):
                h_med = 25.0
        except Exception:
            h_med = 25.0

        if not sk_path.exists():
            sk_strokes = run_skeleton(img_path, h_med=h_med)
            np.savez_compressed(
                str(sk_path),
                n_strokes=len(sk_strokes),
                strokes=np.array(sk_strokes, dtype=object)
            )
        else:
            data = np.load(str(sk_path), allow_pickle=True)
            sk_strokes = list(data["strokes"])

        is_empty = int(len(is_strokes) == 0)
        sk_empty = int(len(sk_strokes) == 0)

        elapsed = time.time() - start_time
        speed = (i + 1) / max(elapsed, 0.001)
        remaining = (total_samples - (i + 1)) / max(speed, 0.001)
        sample_dur = time.time() - sample_start

        cache_str = "(cached)" if is_cached else f"{sample_dur:.1f}s"
        print(
            f"[{i+1:3d}/{total_samples}] {sample_key:32s} | "
            f"IS: {len(is_strokes):2d} strokes | "
            f"SK: {len(sk_strokes):2d} strokes | "
            f"{cache_str:8s} | "
            f"ETA: {remaining/60:.1f}m",
            flush=True
        )

        existing_log[sample_key] = {
            "sample_key": sample_key,
            "subject_id": subj,
            "dataset": row.get("dataset", ""),
            "word_id": word_id,
            "variant": variant,
            "label": row["label"],
            "split": row["split"],
            "inksight_n_strokes": len(is_strokes),
            "inksight_empty": is_empty,
            "inksight_tokens": is_token_cnt,
            "skeleton_n_strokes": len(sk_strokes),
            "skeleton_empty": sk_empty,
            "gt_n_strokes": row["n_gt_strokes"],
            "inksight_pred_path": str(is_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "skeleton_pred_path": str(sk_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "gt_strokes_path": row["gt_strokes_path"].replace("\\", "/"),
            "word_w_px": row["word_w_px"],
            "word_h_px": row["word_h_px"],
            "h_med_subj": row["h_med_subj"],
            "scale": row["scale"],
            "pad_x": row["pad_x"],
            "pad_y": row["pad_y"],
            "bbox_json": row["bbox_json"],
        }

        # Flush log file every 10 samples
        if (i + 1) % 10 == 0 or (i + 1) == total_samples:
            fieldnames = list(next(iter(existing_log.values())).keys())
            with open(LOG_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_log.values())

    print("\n" + "=" * 70)
    print(f"Inference complete! Log updated at: {LOG_PATH}")
    processed_sub = [v for k, v in existing_log.items() if any(k == f"{r['subject_id']}_{r['word_id']}_{r['variant']}" for r in filtered_manifest)]
    n_is_empty = sum(int(r["inksight_empty"]) for r in processed_sub)
    n_sk_empty = sum(int(r["skeleton_empty"]) for r in processed_sub)
    print(f"Processed subset: {len(processed_sub)} samples.")
    print(f"  InkSight Empty: {n_is_empty}/{len(processed_sub)}")
    print(f"  Skeleton Empty: {n_sk_empty}/{len(processed_sub)}")
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()

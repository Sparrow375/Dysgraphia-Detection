"""Batch Precompute Teacher-Forced Target Scoring for TrOCR Cache.

Iterates through all cached sentence tasks, passes each child's handwriting crop
alongside the known target prompt through the VisionEncoderDecoderModel, and evaluates
the exact conditional token log-probabilities:
    log P(y_t^{target} | image, y_{<t}^{target})

Populates `forced_metrics` in `data/features/trocr_cache.json`:
- `forced_mean_prob`: Mean posterior probability assigned to target tokens.
- `forced_min_prob`: Minimum token probability across the prompt (worst-formed letter/word).
- `forced_low_prob_ratio`: Proportion of target tokens with probability < 0.10.
- `forced_mean_logprob`: Normalized sequence negative log-likelihood.
- `forced_min_logprob`: Peak local motor deformity score.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from features.content import CANDIDATE_PROMPTS, clean_text, levenshtein_distance
from features.ocr_engine import EnglishTrOCREngine, get_trocr_cache, save_trocr_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_crop_path(student_id: str, task_name: str) -> Optional[Path]:
    for school in ["school_a", "school_b"]:
        p = WORKSPACE_ROOT / "data" / "processed" / school / student_id / f"{task_name}.png"
        if p.exists():
            return p
    return None


def get_target_prompt(task_name: str, decoded_text: str) -> str:
    """Select the closest candidate prompt for the given task."""
    tn = task_name.lower()
    cat = None
    if "copy" in tn and "english" in tn:
        cat = "copy_english"
    elif "dict" in tn and "english" in tn:
        cat = "dictated_english"
    elif "copy" in tn and "hindi" in tn:
        cat = "copy_hindi"
    elif "dict" in tn and "hindi" in tn:
        cat = "dictated_hindi"

    if not cat:
        return ""
    candidates = CANDIDATE_PROMPTS.get(cat, [])
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    cl = clean_text(decoded_text)
    return min(candidates, key=lambda p: levenshtein_distance(clean_text(p), cl))


def run_batch_forced_scoring(
    batch_size: int = 8,
    save_interval: int = 20,
    force_recompute: bool = False,
) -> None:
    """Compute and persist teacher-forced target scoring for all cached sentence tasks."""
    trocr_cache = get_trocr_cache()
    logger.info("Loaded TrOCR cache with %d entries.", len(trocr_cache))

    # Identify tasks that require forced scoring
    tasks_to_process: List[Tuple[str, str, str, Path, str]] = []
    for k, c in trocr_cache.items():
        if not force_recompute and "forced_metrics" in c and len(c["forced_metrics"]) > 0:
            continue

        parts = k.split("_sentence_")
        if len(parts) != 2:
            continue
        sid = parts[0]
        task_name = "sentence_" + parts[1]

        # Only process English tasks (EnglishTrOCR engine)
        if "english" not in task_name.lower():
            continue

        decoded = c.get("text", "")
        target_prompt = get_target_prompt(task_name, decoded)
        if not target_prompt:
            continue

        crop_path = find_crop_path(sid, task_name)
        if crop_path is None:
            continue

        tasks_to_process.append((k, sid, task_name, crop_path, target_prompt))

    logger.info("Found %d tasks needing teacher-forced scoring.", len(tasks_to_process))
    if not tasks_to_process:
        logger.info("All tasks already have forced scoring computed!")
        return

    # Initialize English TrOCR engine
    logger.info("Initializing EnglishTrOCREngine...")
    engine = EnglishTrOCREngine()

    # Process in batches grouped by prompt for maximum CPU vectorized throughput
    processed_count = 0
    start_time = time.time()

    # Sort by target prompt to maximize batch homogeneity
    tasks_to_process.sort(key=lambda x: x[4])

    for i in range(0, len(tasks_to_process), batch_size):
        batch = tasks_to_process[i : i + batch_size]
        pil_images = []
        valid_items = []

        for k, sid, task_name, crop_path, prompt in batch:
            img_bgr = cv2.imread(str(crop_path))
            if img_bgr is None or img_bgr.size == 0 or img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
                continue
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))
            valid_items.append((k, prompt))

        if not pil_images:
            continue

        prompts = [item[1] for item in valid_items]

        try:
            inputs = engine.processor(pil_images, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(engine.device)
            labels = engine.processor.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(engine.device)

            with torch.no_grad():
                outputs = engine.model(pixel_values=pixel_values, labels=labels)
                logits = outputs.logits[:, :-1, :]  # [B, seq_len - 1, vocab_size]
                probs = torch.softmax(logits, dim=-1)
                target_ids = labels[:, 1:]  # [B, seq_len - 1]

                # Gather probabilities of the target tokens
                token_probs = torch.gather(probs, -1, target_ids.unsqueeze(-1)).squeeze(-1).cpu().numpy()

            pad_id = getattr(engine.model.config, "pad_token_id", 1)
            target_ids_np = target_ids.cpu().numpy()

            for b_idx, (k, _) in enumerate(valid_items):
                # Mask out pad tokens
                non_pad_mask = target_ids_np[b_idx] != pad_id
                t_probs = token_probs[b_idx][non_pad_mask]

                if len(t_probs) == 0:
                    forced_res = {
                        "forced_mean_prob": 0.0,
                        "forced_min_prob": 0.0,
                        "forced_low_prob_ratio": 1.0,
                        "forced_mean_logprob": -20.0,
                        "forced_min_logprob": -20.0,
                    }
                else:
                    log_probs = np.log(np.maximum(t_probs, 1e-12))
                    forced_res = {
                        "forced_mean_prob": float(np.mean(t_probs)),
                        "forced_min_prob": float(np.min(t_probs)),
                        "forced_low_prob_ratio": float(np.mean(t_probs < 0.10)),
                        "forced_mean_logprob": float(np.mean(log_probs)),
                        "forced_min_logprob": float(np.min(log_probs)),
                    }

                trocr_cache[k]["forced_metrics"] = forced_res
                processed_count += 1

        except Exception as e:
            logger.warning("Error processing batch starting at %d: %s. Falling back to sequential.", i, e)
            for k, _, _, crop_path, prompt in batch:
                try:
                    img = cv2.imread(str(crop_path))
                    res = engine.score_target_text(img, prompt)
                    trocr_cache[k]["forced_metrics"] = res
                    processed_count += 1
                except Exception as ex:
                    logger.warning("Failed on task %s: %s", k, ex)

        if (i // batch_size) % (save_interval // batch_size) == 0:
            save_trocr_cache(trocr_cache)
            elapsed = time.time() - start_time
            rate = processed_count / max(1.0, elapsed)
            remaining = (len(tasks_to_process) - processed_count) / max(0.01, rate)
            logger.info(
                "Progress: %d / %d (%.1f%%) | %.2f crops/sec | ETA: %.1f sec",
                processed_count,
                len(tasks_to_process),
                100.0 * processed_count / len(tasks_to_process),
                rate,
                remaining,
            )

    save_trocr_cache(trocr_cache)
    logger.info("Successfully finished forced scoring for %d crops in %.1f seconds!", processed_count, time.time() - start_time)


if __name__ == "__main__":
    run_batch_forced_scoring(batch_size=8, save_interval=20)

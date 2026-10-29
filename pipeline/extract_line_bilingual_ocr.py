"""
End-to-End Bilingual OCR (HTR) Decoding on the Decomposed Single-Line Dataset.
Decodes single-line strips and groups lines per task, eliminating multi-line hallucinations.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.content import levenshtein_distance, word_levenshtein_distance
from features.letter_analytics import analyze_letter_errors
from features.ocr_engine import DevanagariTrOCREngine, EnglishTrOCREngine, OCRTaskResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("line_bilingual_ocr")

DATA_DIR = REPO_ROOT / "data"
LINES_DIR = DATA_DIR / "line_crops_dataset"
LINES_INDEX = LINES_DIR / "lines_index.csv"
FEATURES_DIR = DATA_DIR / "features"
OOF_NPZ = FEATURES_DIR / "exp7_all_models_oof_probs.npz"
OUTPUT_NPZ = FEATURES_DIR / "exp7_line_bilingual_ocr_student_features.npz"


def compute_task_features(
    text: str,
    token_confs: List[float],
    target_prompt: str,
    script: str,
) -> Dict[str, float]:
    clean_decoded = " ".join(text.strip().split())
    clean_target = " ".join(target_prompt.strip().split()) if target_prompt else ""

    if not token_confs:
        token_confs = [1.0] if clean_decoded else [0.0]

    mean_conf = float(np.mean(token_confs))
    min_conf = float(np.min(token_confs))
    std_conf = float(np.std(token_confs))
    low_conf_ratio = float(np.mean([1.0 if c < 0.40 else 0.0 for c in token_confs]))

    char_count = len(clean_decoded.replace(" ", ""))
    word_count = len(clean_decoded.split())
    target_chars = len(clean_target.replace(" ", ""))
    target_words = len(clean_target.split()) if clean_target else 1

    char_completion = float(char_count / max(1, target_chars))
    word_completion = float(word_count / max(1, target_words))
    char_deficit = float(max(0, target_chars - char_count) / max(1, target_chars))

    cer = float(levenshtein_distance(clean_target, clean_decoded) / max(1, len(clean_target))) if clean_target else 0.0
    wer = float(word_levenshtein_distance(clean_target.split(), clean_decoded.split()) / max(1, target_words)) if clean_target else 0.0

    err_dict = analyze_letter_errors(clean_target, clean_decoded, token_confs, script=script)

    feats = {
        "ocr_mean_confidence": mean_conf,
        "ocr_min_confidence": min_conf,
        "ocr_std_confidence": std_conf,
        "ocr_low_conf_ratio": low_conf_ratio,
        "ocr_char_count": float(char_count),
        "ocr_word_count": float(word_count),
        "ocr_char_completion_ratio": char_completion,
        "ocr_word_completion_ratio": word_completion,
        "ocr_char_deficit": char_deficit,
        "ocr_cer": cer,
        "ocr_wer": wer,
    }
    feats.update(err_dict)
    return feats


def run_extraction(device: str = "cuda"):
    logger.info("Loading lines index and cohort...")
    df_lines = pd.read_csv(LINES_INDEX)

    oof_data = np.load(OOF_NPZ)
    cohort_sids = list(oof_data["student_ids"])
    cohort_set = set(cohort_sids)

    df_lines = df_lines[df_lines["student_id"].isin(cohort_set)].copy()
    logger.info("Filtered to %d lines across %d cohort students", len(df_lines), len(cohort_sids))

    en_engine = EnglishTrOCREngine(device=device)
    hi_engine = DevanagariTrOCREngine(device=device)

    # Group lines by task_uid
    grouped = df_lines.groupby("task_uid", sort=False)
    logger.info("Total tasks to transcribe: %d", len(grouped))

    task_results = {}
    t0 = time.time()
    n_done = 0

    for task_uid, group in grouped:
        group_sorted = group.sort_values("line_index")
        script = group_sorted["script"].iloc[0]
        target_prompt = group_sorted["target_prompt"].iloc[0] if pd.notna(group_sorted["target_prompt"].iloc[0]) else ""
        sid = group_sorted["student_id"].iloc[0]

        line_texts = []
        line_confs = []

        for _, row in group_sorted.iterrows():
            img_p = LINES_DIR / row["line_rel_path"]
            if not img_p.exists():
                continue
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            if script == "latin":
                res = en_engine.transcribe_line(img)
                line_texts.append(res.text)
                line_confs.extend(res.token_confs)
            else:
                # For Hindi, transcribe line / words
                res = hi_engine.transcribe_task(img)
                line_texts.append(res.text)
                confs = res.token_confidences if hasattr(res, "token_confidences") else []
                line_confs.extend(confs)

        full_task_text = " ".join(line_texts)
        feats = compute_task_features(full_task_text, line_confs, target_prompt, script)
        task_results[task_uid] = {
            "student_id": sid,
            "script": script,
            "text": full_task_text,
            "features": feats,
        }

        n_done += 1
        if n_done % 100 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            logger.info("Processed %d / %d tasks (%.1f%%) in %.1fs (%.1f tasks/s)",
                        n_done, len(grouped), 100.0 * n_done / len(grouped), elapsed, rate)

    CACHE_JSON = FEATURES_DIR / "line_trocr_cache.json"
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(task_results, f)
    logger.info("Saved task cache to %s", CACHE_JSON)

    # Aggregate to student vectors
    logger.info("Aggregating student feature matrices...")
    feature_names = list(next(iter(task_results.values()))["features"].keys())

    X_en_list = []
    X_hi_list = []
    X_joint_list = []

    for sid in cohort_sids:
        s_tasks = [t for t in task_results.values() if t["student_id"] == sid]
        en_tasks = [t for t in s_tasks if t["script"] == "latin"]
        hi_tasks = [t for t in s_tasks if t["script"] == "devanagari"]

        def agg_feats(t_list):
            if not t_list:
                return np.zeros(len(feature_names), dtype=np.float32)
            mat = np.array([[t["features"][fn] for fn in feature_names] for t in t_list], dtype=np.float32)
            return np.mean(mat, axis=0)

        vec_en = agg_feats(en_tasks)
        vec_hi = agg_feats(hi_tasks)
        vec_joint = np.concatenate([vec_hi, vec_en], axis=-1)

        X_en_list.append(vec_en)
        X_hi_list.append(vec_hi)
        X_joint_list.append(vec_joint)

    X_en = np.array(X_en_list, dtype=np.float32)
    X_hi = np.array(X_hi_list, dtype=np.float32)
    X_joint = np.array(X_joint_list, dtype=np.float32)

    fold_source = np.load(FEATURES_DIR / "exp7_bilingual_ocr_student_features.npz")
    save_dict = {
        "student_ids": np.array(cohort_sids),
        "y_true": oof_data["y_true"],
        "feature_names": np.array(feature_names),
        "X_english_ocr": X_en,
        "X_hindi_ocr": X_hi,
        "X_joint_ocr": X_joint,
    }
    for r in range(5):
        save_dict[f"repeat_{r}_fold"] = fold_source[f"repeat_{r}_fold"]

    np.savez_compressed(OUTPUT_NPZ, **save_dict)
    logger.info("Saved line bilingual OCR features to %s (shape: %s)", OUTPUT_NPZ, X_joint.shape)


if __name__ == "__main__":
    run_extraction()

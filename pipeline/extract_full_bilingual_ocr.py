"""End-to-End Bilingual OCR (HTR) Decoding & Clinical Linguistic Error Extraction.

Decodes all recropped handwriting crops across the cohort on GPU:
1. English Tasks (Latin): Microsoft TrOCR Small Handwritten (`microsoft/trocr-small-handwritten`)
2. Hindi Tasks (Devanagari): Devanagari TrOCR (`paudelanil/trocr-devanagari-2`)

For every task:
- Transcribes text, computes token posterior confidences and confidence dynamics.
- Computes teacher-forced log-probabilities for the ground-truth target prompt.
- Extracts 38 granular letter-level and linguistic error features (omissions, transpositions,
  reversals b/d & p/q, Devanagari matras/conjuncts, mid-word caps, fatigue ratio).
- Computes Character Error Rate (CER) and Word Error Rate (WER) edit distances.

Persists:
- Full task transcripts & tokens: `data/features/trocr_cache.json`
- Aggregated student-level feature matrix: `data/features/exp7_bilingual_ocr_student_features.npz`
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bilingual_ocr")

DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"
FEATURES_DIR = DATA_DIR / "features"
TROCR_CACHE_PATH = FEATURES_DIR / "trocr_cache.json"
OUTPUT_NPZ_PATH = FEATURES_DIR / "exp7_bilingual_ocr_student_features.npz"


def extract_task_features(
    res: OCRTaskResult,
    target_prompt: str,
    forced_res: Dict[str, float],
    script: str,
) -> Dict[str, float]:
    """Compute 53 rich diagnostic OCR and linguistic error features for a single task."""
    text = res.text
    token_confs = res.token_confs if hasattr(res, "token_confs") else res.token_confidences
    if not token_confs:
        token_confs = [1.0] if text else [0.0]

    # 1. OCR Confidence & Volume Metrics
    mean_conf = float(res.mean_confidence)
    min_conf = float(res.min_confidence)
    std_conf = float(res.std_confidence)
    low_conf_ratio = float(res.low_conf_ratio)
    char_count = int(res.char_count)
    word_count = int(res.word_count)

    clean_target = " ".join(target_prompt.strip().split()) if target_prompt else ""
    target_chars = len(clean_target.replace(" ", ""))
    target_words = len(clean_target.split()) if clean_target else 1

    char_completion = float(char_count / max(1, target_chars))
    word_completion = float(word_count / max(1, target_words))
    char_deficit = float(max(0, target_chars - char_count) / max(1, target_chars))

    # 2. Levenshtein Edit Distances (CER & WER)
    clean_decoded = " ".join(text.strip().split())
    cer = float(levenshtein_distance(clean_target, clean_decoded) / max(1, len(clean_target))) if clean_target else 0.0
    wer = float(word_levenshtein_distance(clean_target.split(), clean_decoded.split()) / max(1, target_words)) if clean_target else 0.0

    # 3. Teacher-Forced Probabilities
    f_mean_p = float(forced_res.get("forced_mean_prob", 0.0))
    f_min_p = float(forced_res.get("forced_min_prob", 0.0))
    f_low_p_ratio = float(forced_res.get("forced_low_prob_ratio", 1.0))
    f_mean_logp = float(forced_res.get("forced_mean_logprob", -20.0))
    f_min_logp = float(forced_res.get("forced_min_logprob", -20.0))

    # 4. Granular Letter & Linguistic Error Analytics (38 features)
    err_dict = analyze_letter_errors(clean_target, clean_decoded, token_confs, script=script)

    feats: Dict[str, float] = {
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
        "forced_mean_prob": f_mean_p,
        "forced_min_prob": f_min_p,
        "forced_low_prob_ratio": f_low_p_ratio,
        "forced_mean_logprob": f_mean_logp,
        "forced_min_logprob": f_min_logp,
    }

    # Add all 38 letter analytics features
    feats.update(err_dict)
    return feats


def run_full_bilingual_ocr_pipeline(
    limit: Optional[int] = None,
    device: Optional[str] = None,
    save_cache_every: int = 100,
) -> None:
    """Execute end-to-end bilingual OCR decoding across the entire dataset on GPU."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Bilingual OCR Pipeline on device: %s", dev)

    # 1. Load Dataset Index
    df_index = pd.read_csv(INDEX_CSV)
    logger.info("Loaded %d total sentence crop records from %s", len(df_index), INDEX_CSV)

    if limit is not None and limit > 0:
        unique_students = df_index["student_id"].unique()[:limit]
        df_index = df_index[df_index["student_id"].isin(unique_students)].copy()
        logger.info("Limited extraction to %d students (%d tasks)", limit, len(df_index))

    # 2. Initialize Models
    logger.info("Loading English TrOCR Engine (microsoft/trocr-small-handwritten)...")
    t0_en = time.time()
    en_engine = EnglishTrOCREngine(device=dev)
    logger.info("English TrOCR loaded in %.2fs", time.time() - t0_en)

    logger.info("Loading Devanagari TrOCR Engine (paudelanil/trocr-devanagari-2)...")
    t0_hi = time.time()
    hi_engine = DevanagariTrOCREngine(device=dev)
    logger.info("Devanagari TrOCR loaded in %.2fs", time.time() - t0_hi)

    # 3. Process All Sentence Crops
    trocr_cache: Dict[str, Any] = {}
    task_feature_records: Dict[str, Dict[str, float]] = {}

    total_tasks = len(df_index)
    t_start = time.time()

    logger.info("Starting GPU decoding loop across %d sentence tasks...", total_tasks)

    for i, (_, row) in enumerate(df_index.iterrows()):
        task_uid = str(row["task_uid"])
        student_id = str(row["student_id"])
        script = str(row["script"]).lower()
        target_prompt = str(row["target_prompt"]) if pd.notna(row["target_prompt"]) else ""
        rel_path = str(row["rel_path"])
        img_path = SENTENCE_DATASET_DIR / rel_path

        if not img_path.exists():
            logger.warning("Image path does not exist: %s", img_path)
            continue

        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            logger.warning("Failed to decode image: %s", img_path)
            continue

        # Transcribe & score according to script
        try:
            if script == "latin":
                res = en_engine.transcribe_task(img)
                forced = en_engine.score_target_text(img, target_prompt)
            else:
                res = hi_engine.transcribe_task(img)
                forced = hi_engine.score_target_text(img, target_prompt)
        except Exception as e:
            logger.error("Error processing %s (%s): %s", task_uid, script, e)
            res = OCRTaskResult(text="", tokens=[], token_confidences=[], mean_confidence=0.0, min_confidence=0.0, std_confidence=0.0, low_conf_ratio=1.0, char_count=0, word_count=0, line_count=1)
            forced = {}

        # Extract rich diagnostic feature vector
        feats = extract_task_features(res, target_prompt, forced, script=script)
        task_feature_records[task_uid] = feats

        # Save to trocr_cache
        trocr_cache[task_uid] = {
            "task_uid": task_uid,
            "student_id": student_id,
            "script": script,
            "target_prompt": target_prompt,
            "text": res.text,
            "tokens": res.tokens,
            "token_confidences": res.token_confidences,
            "mean_confidence": res.mean_confidence,
            "min_confidence": res.min_confidence,
            "std_confidence": res.std_confidence,
            "low_conf_ratio": res.low_conf_ratio,
            "char_count": res.char_count,
            "word_count": res.word_count,
            "line_count": res.line_count,
            "forced_metrics": forced,
        }

        # Periodic logging & saving
        if (i + 1) % 50 == 0 or (i + 1) == total_tasks:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (total_tasks - (i + 1)) / rate if rate > 0 else 0
            logger.info(
                "[%d/%d] Processed in %.1fs (%.1f tasks/s, ETA: %.1fs) | Last: %s -> '%s'",
                i + 1,
                total_tasks,
                elapsed,
                rate,
                eta,
                task_uid,
                res.text[:30],
            )

        if (i + 1) % save_cache_every == 0:
            with open(TROCR_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(trocr_cache, f, indent=2, ensure_ascii=False)

    # Save final cache
    with open(TROCR_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(trocr_cache, f, indent=2, ensure_ascii=False)
    logger.info("Saved complete TrOCR cache with %d entries to %s", len(trocr_cache), TROCR_CACHE_PATH)

    # 4. Aggregate Student-Level Feature Matrices
    logger.info("Aggregating student-level bilingual error matrices...")
    aggregate_student_level_features(df_index, task_feature_records)


def aggregate_student_level_features(
    df_index: pd.DataFrame,
    task_feature_records: Dict[str, Dict[str, float]],
) -> None:
    """Aggregate per-task OCR error features into unified student-level vectors."""
    # Reference cohort: School A + School B
    ref_mask = df_index["school"].isin(["school_a", "school_b"])
    df_ref = df_index[ref_mask].copy()

    # Get unique students in reference cohort (order preserved)
    student_meta = df_ref.drop_duplicates(subset=["student_id"]).set_index("student_id")
    student_ids = list(student_meta.index)
    student_labels = student_meta["label"].values.astype(np.int64)

    # All unique feature keys from task records
    sample_key = next(iter(task_feature_records.keys()))
    feat_names = sorted(list(task_feature_records[sample_key].keys()))
    n_feats = len(feat_names)

    hindi_student_vecs: List[np.ndarray] = []
    english_student_vecs: List[np.ndarray] = []
    joint_student_vecs: List[np.ndarray] = []

    # Aggregations per feature: [mean, max, std] -> 3 stats per feature
    stat_suffixes = ["mean", "max", "std"]
    hindi_feat_names = [f"hi_{name}_{stat}" for name in feat_names for stat in stat_suffixes]
    english_feat_names = [f"en_{name}_{stat}" for name in feat_names for stat in stat_suffixes]
    joint_feat_names = hindi_feat_names + english_feat_names

    for sid in student_ids:
        s_tasks = df_ref[df_ref["student_id"] == sid]

        hi_tasks = s_tasks[s_tasks["script"] == "devanagari"]["task_uid"].tolist()
        en_tasks = s_tasks[s_tasks["script"] == "latin"]["task_uid"].tolist()

        def compute_student_stats(task_uids: List[str]) -> np.ndarray:
            if not task_uids:
                return np.zeros(n_feats * 3, dtype=np.float32)

            task_mat = []
            for tuid in task_uids:
                rec = task_feature_records.get(tuid, {})
                vec = [rec.get(fn, 0.0) for fn in feat_names]
                task_mat.append(vec)

            arr = np.array(task_mat, dtype=np.float32)
            # Impute NaN/inf with column medians or 0
            nan_mask = np.isnan(arr) | np.isinf(arr)
            arr[nan_mask] = 0.0

            m = np.mean(arr, axis=0)
            mx = np.max(arr, axis=0)
            s = np.std(arr, axis=0) if len(arr) > 1 else np.zeros_like(m)

            # Interleave: [mean_0, max_0, std_0, mean_1, max_1, std_1, ...]
            stats = np.column_stack([m, mx, s]).flatten()
            return stats.astype(np.float32)

        hi_vec = compute_student_stats(hi_tasks)
        en_vec = compute_student_stats(en_tasks)
        joint_vec = np.concatenate([hi_vec, en_vec], axis=-1)

        hindi_student_vecs.append(hi_vec)
        english_student_vecs.append(en_vec)
        joint_student_vecs.append(joint_vec)

    X_hi = np.array(hindi_student_vecs, dtype=np.float32)
    X_en = np.array(english_student_vecs, dtype=np.float32)
    X_joint = np.array(joint_student_vecs, dtype=np.float32)

    logger.info("Constructed Student Matrices: Hindi %s, English %s, Joint %s", X_hi.shape, X_en.shape, X_joint.shape)

    # Save to NPZ
    save_dict: Dict[str, Any] = {
        "student_ids": np.array(student_ids),
        "student_labels": student_labels,
        "X_hindi_ocr": X_hi,
        "X_english_ocr": X_en,
        "X_joint_ocr": X_joint,
        "feature_names_hindi": np.array(hindi_feat_names),
        "feature_names_english": np.array(english_feat_names),
        "feature_names_joint": np.array(joint_feat_names),
    }

    # Include CV folds
    for r in range(5):
        fold_col = f"repeat_{r}_fold"
        save_dict[fold_col] = student_meta[fold_col].values.astype(np.int64)

    np.savez_compressed(OUTPUT_NPZ_PATH, **save_dict)
    logger.info("Successfully persisted bilingual student features to %s (%.2f MB)", OUTPUT_NPZ_PATH, OUTPUT_NPZ_PATH.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Bilingual OCR GPU Extraction Pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of students for debugging.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    run_full_bilingual_ocr_pipeline(limit=args.limit, device=args.device)


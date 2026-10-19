"""Content and OCR-derived linguistic and legibility features.

Extracts objective legibility, transcription accuracy, and task omission metrics
using modern Handwritten Text Recognition (Microsoft TrOCR + Devanagari TrOCR):
- ocr_confidence_mean: Mean confidence score of tokens decoded by OCR (legibility index).
- ocr_confidence_std: Standard deviation of word recognition confidences.
- ocr_confidence_min: Minimum token confidence (worst local legibility).
- ocr_low_conf_ratio: Fraction of recognized tokens with confidence < 0.40 (illegibility fraction).
- ocr_words_recognized: Total count of words recognized by OCR.
- ocr_word_completion_ratio: Recognized words divided by expected prompt word count.
- ocr_char_count: Total non-space characters decoded by OCR.
- ocr_char_completion_ratio: Decoded characters divided by expected prompt character count.
- ocr_omission_ratio: Character deficit fraction max(0, (expected - decoded) / expected).
- ocr_cer: Minimum Character Error Rate (normalized Levenshtein) against prompt candidates.
- ocr_wer: Minimum Word Error Rate against prompt candidates.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from features.ocr_engine import (
    CACHE_FILE_PATH,
    DualEngineOCRPipeline,
    OCRTaskResult,
    get_trocr_cache,
    save_trocr_cache,
)
from features.letter_analytics import analyze_letter_errors, _empty_analytics_result

logger = logging.getLogger(__name__)

# Global dual engine pipeline instance
_DUAL_PIPELINE: Optional[DualEngineOCRPipeline] = None


def get_ocr_pipeline() -> DualEngineOCRPipeline:
    """Lazily initialize global DualEngineOCRPipeline."""
    global _DUAL_PIPELINE
    if _DUAL_PIPELINE is None:
        _DUAL_PIPELINE = DualEngineOCRPipeline()
    return _DUAL_PIPELINE


# Standardized prompt word counts
EXPECTED_WORD_COUNTS: Dict[str, int] = {
    "copy_hindi": 7,
    "copy_english": 14,
    "dictated_hindi": 6,
    "dictated_english": 7,
}

# Standardized prompt character counts
EXPECTED_CHAR_COUNTS: Dict[str, int] = {
    "copy_hindi": 34,
    "copy_english": 74,
    "dictated_hindi": 28,
    "dictated_english": 38,
}

# Candidate ground truth texts for normalized error calculations
CANDIDATE_PROMPTS: Dict[str, List[str]] = {
    "copy_hindi": [
        "हमारे स्कूल मे एक सूंदर बगीचा है",
        "हमारे स्कूल में एक सुंदर बगीचा है",
        "हमारे स्कूल मे एक सुंदर बगीचा है",
        "हमारे स्कूल में एक बगीचा है",
    ],
    "copy_english": [
        "on sundays we play cricket in the park with four friends after game we eat snacks together",
        "on sundays we play cricket in the park with four friends",
        "my mother cooks food in the kitchen",
        "my brother plays football with his friends",
    ],
    "dictated_hindi": [
        "मेरी माँ रसोई में खाना बनाती है",
        "मेरी माँ रसोई मे खाना बनाती है",
        "मेरे विद्यालय में एक बगीचा है",
    ],
    "dictated_english": [
        "my name is deepti",
        "my brother plays football with his friends",
        "my name is",
        "we play cricket in the park",
    ],
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute classic Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def word_levenshtein_distance(tokens1: List[str], tokens2: List[str]) -> int:
    """Compute word-level Levenshtein edit distance between token sequences."""
    if len(tokens1) < len(tokens2):
        tokens1, tokens2 = tokens2, tokens1
    if len(tokens2) == 0:
        return len(tokens1)

    previous_row = list(range(len(tokens2) + 1))
    for i, t1 in enumerate(tokens1):
        current_row = [i + 1]
        for j, t2 in enumerate(tokens2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (t1 != t2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def clean_text(text: str) -> str:
    """Normalize text: lowercase, remove non-alphanumeric punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    return " ".join(text.split())


def compute_content_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
    use_cache: bool = True,
) -> Dict[str, float]:
    """Extract content, OCR legibility, omission, and error rate features using TrOCR.

    Returns:
        dict with keys:
            - ocr_confidence_mean: float | NaN
            - ocr_confidence_std: float | NaN
            - ocr_confidence_min: float | NaN
            - ocr_low_conf_ratio: float | NaN
            - ocr_words_recognized: float
            - ocr_word_completion_ratio: float | NaN
            - ocr_char_count: float
            - ocr_char_completion_ratio: float | NaN
            - ocr_omission_ratio: float | NaN
            - ocr_cer: float | NaN
            - ocr_wer: float | NaN
    """
    letter_empty = _empty_analytics_result()
    forced_empty = {
        "forced_mean_prob": np.nan,
        "forced_min_prob": np.nan,
        "forced_low_prob_ratio": np.nan,
        "forced_mean_logprob": np.nan,
        "forced_min_logprob": np.nan,
    }

    empty_result = {
        "ocr_confidence_mean": np.nan,
        "ocr_confidence_std": np.nan,
        "ocr_confidence_min": np.nan,
        "ocr_low_conf_ratio": np.nan,
        "ocr_words_recognized": 0.0,
        "ocr_word_completion_ratio": np.nan,
        "ocr_char_count": 0.0,
        "ocr_char_completion_ratio": np.nan,
        "ocr_omission_ratio": np.nan,
        "ocr_cer": np.nan,
        "ocr_wer": np.nan,
        **letter_empty,
        **forced_empty,
    }

    if image_crop is None or image_crop.size == 0:
        return empty_result

    pipeline = get_ocr_pipeline()
    task_res: OCRTaskResult = pipeline.transcribe_task(
        image_crop=image_crop,
        sentence_json=sentence_json,
        use_cache=use_cache,
    )

    task_name = str(sentence_json.get("task_name", "")).lower()

    if not task_res.text.strip():
        # Zero words recognized by OCR (severe illegibility or blank crop)
        exp_words = EXPECTED_WORD_COUNTS.get(task_name)
        exp_chars = EXPECTED_CHAR_COUNTS.get(task_name)
        res = dict(empty_result)
        res.update({
            "ocr_confidence_mean": 0.0,
            "ocr_confidence_std": 0.0,
            "ocr_confidence_min": 0.0,
            "ocr_low_conf_ratio": 1.0,
            "ocr_words_recognized": 0.0,
            "ocr_word_completion_ratio": 0.0 if exp_words else np.nan,
            "ocr_char_count": 0.0,
            "ocr_char_completion_ratio": 0.0 if exp_chars else np.nan,
            "ocr_omission_ratio": 1.0 if exp_chars else np.nan,
            "ocr_cer": 1.0 if task_name in CANDIDATE_PROMPTS else np.nan,
            "ocr_wer": 1.0 if task_name in CANDIDATE_PROMPTS else np.nan,
            "char_omission_rate": 1.0 if exp_chars else np.nan,
            "word_omission_rate": 1.0 if exp_words else np.nan,
        })
        return res

    cleaned_ocr_text = clean_text(task_res.text)
    ocr_tokens = cleaned_ocr_text.split()
    total_chars = float(len(cleaned_ocr_text.replace(" ", "")))
    recognized_words_count = float(len(ocr_tokens))

    exp_words = EXPECTED_WORD_COUNTS.get(task_name)
    word_comp_ratio = (recognized_words_count / exp_words) if exp_words else np.nan

    exp_chars = EXPECTED_CHAR_COUNTS.get(task_name)
    char_comp_ratio = (total_chars / exp_chars) if exp_chars else np.nan
    omission_ratio = max(0.0, (exp_chars - total_chars) / exp_chars) if exp_chars else np.nan

    # Calculate minimum CER and WER against candidate prompts
    cer = np.nan
    wer = np.nan
    best_cand = ""

    candidates = CANDIDATE_PROMPTS.get(task_name)
    if candidates:
        cer_scores = []
        wer_scores = []
        for cand in candidates:
            cand_clean = clean_text(cand)
            cand_tokens = cand_clean.split()
            cand_len = max(len(cand_clean), 1)
            cand_wlen = max(len(cand_tokens), 1)

            d_char = levenshtein_distance(cand_clean, cleaned_ocr_text)
            d_word = word_levenshtein_distance(cand_tokens, ocr_tokens)

            cer_scores.append((d_char / cand_len, cand))
            wer_scores.append(d_word / cand_wlen)

        best_score, best_cand = min(cer_scores, key=lambda x: x[0])
        cer = float(best_score)
        wer = float(min(wer_scores))

    script = str(sentence_json.get("script", "latin")).lower()
    letter_feats = analyze_letter_errors(
        target_text=best_cand,
        decoded_text=task_res.text,
        token_confidences=task_res.token_confidences,
        script=script,
    )

    forced_metrics = getattr(task_res, "forced_metrics", {})
    if not forced_metrics and best_cand and not use_cache and image_crop is not None:
        try:
            forced_metrics = pipeline.score_target_text(image_crop, best_cand, script=script)
        except Exception:
            forced_metrics = {}

    for k in ["forced_mean_prob", "forced_min_prob", "forced_low_prob_ratio", "forced_mean_logprob", "forced_min_logprob"]:
        letter_feats[k] = float(forced_metrics.get(k, np.nan))

    out_dict = {
        "ocr_confidence_mean": task_res.mean_confidence,
        "ocr_confidence_std": task_res.std_confidence,
        "ocr_confidence_min": task_res.min_confidence,
        "ocr_low_conf_ratio": task_res.low_conf_ratio,
        "ocr_words_recognized": recognized_words_count,
        "ocr_word_completion_ratio": word_comp_ratio,
        "ocr_char_count": total_chars,
        "ocr_char_completion_ratio": char_comp_ratio,
        "ocr_omission_ratio": omission_ratio,
        "ocr_cer": cer,
        "ocr_wer": wer,
    }
    out_dict.update(letter_feats)
    return out_dict


def precompute_ocr_cache(
    processed_dir: str = "data/processed",
    school_filter: Optional[str] = None,
    task_filter: Optional[str] = None,
    limit: Optional[int] = None,
    batch_save: int = 5,
) -> None:
    """Scan all processed sentence crops and populate data/features/trocr_cache.json using TrOCR."""
    import glob
    from tqdm import tqdm

    school_glob = "*" if (not school_filter or school_filter.lower() == "all") else school_filter
    search_glob = os.path.join(processed_dir, school_glob, "*", "sentence_*.json")
    json_paths = sorted(glob.glob(search_glob))
    print(f"Found {len(json_paths)} sentence JSON files to inspect for TrOCR caching (school={school_filter}).")

    cache = get_trocr_cache()
    pipeline = get_ocr_pipeline()

    allowed_tasks = [t.strip().lower() for t in task_filter.split(",")] if task_filter else None

    unprocessed: List[Tuple[str, str, Path]] = []
    for jp in json_paths:
        p = Path(jp)
        try:
            with open(p, "r", encoding="utf-8") as f:
                sj = json.load(f)
            sid = str(sj.get("student_id", p.parent.name))
            tid = str(sj.get("task_id", p.stem))
            tname = str(sj.get("task_name", "")).lower()

            if allowed_tasks:
                if not any(at in tname for at in allowed_tasks):
                    continue

            key = f"{sid}_{tid}"
            if key not in cache:
                crop_p = p.parent / sj.get("crop_filename", f"{tid}.png")
                if crop_p.exists():
                    unprocessed.append((key, str(crop_p), p))
        except Exception:
            continue

    if limit is not None and limit > 0:
        unprocessed = unprocessed[:limit]

    print(f"Already cached in trocr_cache.json: {len(cache)}. Remaining tasks to OCR: {len(unprocessed)}")
    if not unprocessed:
        print("All target crops already cached in trocr_cache.json!")
        return

    count = 0
    for key, crop_path, json_path in tqdm(unprocessed, desc="TrOCR Feature Extraction"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                sj = json.load(f)
            img = cv2.imread(crop_path)
            if img is None:
                continue

            res = pipeline.transcribe_task(img, sj, use_cache=False)
            cache[key] = res.to_dict()
            count += 1

            if count % batch_save == 0:
                pipeline.persist_cache()
        except Exception as e:
            logger.error("Error processing %s: %s", key, e)
            continue

    pipeline.persist_cache()
    print(f"Successfully finished caching {len(cache)} entries to {CACHE_FILE_PATH}!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Precompute TrOCR results cache.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--school", default=None, help="Filter by school (e.g. school_a, school_b, or all)")
    parser.add_argument("--tasks", default=None, help="Comma-separated task filter (e.g. copy,dictated)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of crops to process")
    parser.add_argument("--batch-save", type=int, default=5)
    args = parser.parse_args()

    precompute_ocr_cache(
        processed_dir=args.processed_dir,
        school_filter=args.school,
        task_filter=args.tasks,
        limit=args.limit,
        batch_save=args.batch_save,
    )

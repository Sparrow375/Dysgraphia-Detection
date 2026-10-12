"""Content and OCR-derived linguistic and legibility features.

Extracts objective legibility, transcription accuracy, and task omission metrics
using cached deep-learning OCR (EasyOCR):
- ocr_confidence_mean: Mean confidence score of words decoded by OCR (legibility index).
- ocr_confidence_std: Standard deviation of word recognition confidences.
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

import re
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Cache global OCR readers and OCR result disk cache
_OCR_READER = None
_OCR_CACHE = None
CACHE_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "features" / "ocr_cache.json"


def get_ocr_cache() -> Dict[str, List[Tuple[str, float]]]:
    """Load or return cached OCR results from data/features/ocr_cache.json."""
    global _OCR_CACHE
    if _OCR_CACHE is None:
        if CACHE_FILE_PATH.exists():
            try:
                with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                    _OCR_CACHE = json.load(f)
            except Exception:
                _OCR_CACHE = {}
        else:
            _OCR_CACHE = {}
    return _OCR_CACHE


def save_ocr_cache(cache_dict: Dict[str, Any]) -> None:
    """Save in-memory OCR cache to disk."""
    CACHE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_dict, f, indent=2)


def get_ocr_reader():
    """Lazily initialize and cache EasyOCR reader."""
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            # Load bilingual reader (Hindi + English)
            _OCR_READER = easyocr.Reader(["hi", "en"], gpu=False, verbose=False)
        except Exception:
            _OCR_READER = False
    return _OCR_READER if _OCR_READER is not False else None


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
    """Normalize text: lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    return " ".join(text.split())


def compute_content_features(
    sentence_json: Dict[str, Any],
    image_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Extract content, OCR legibility, omission, and error rate features.

    Returns:
        dict with keys:
            - ocr_confidence_mean: float | NaN
            - ocr_confidence_std: float | NaN
            - ocr_low_conf_ratio: float | NaN
            - ocr_words_recognized: float
            - ocr_word_completion_ratio: float | NaN
            - ocr_char_count: float
            - ocr_char_completion_ratio: float | NaN
            - ocr_omission_ratio: float | NaN
            - ocr_cer: float | NaN
            - ocr_wer: float | NaN
    """
    empty_result = {
        "ocr_confidence_mean": np.nan,
        "ocr_confidence_std": np.nan,
        "ocr_low_conf_ratio": np.nan,
        "ocr_words_recognized": 0.0,
        "ocr_word_completion_ratio": np.nan,
        "ocr_char_count": 0.0,
        "ocr_char_completion_ratio": np.nan,
        "ocr_omission_ratio": np.nan,
        "ocr_cer": np.nan,
        "ocr_wer": np.nan,
    }

    if image_crop is None or image_crop.size == 0:
        return empty_result

    # Look up in persistent disk cache first
    sid = str(sentence_json.get("student_id", ""))
    tid = str(sentence_json.get("task_id", ""))
    cache_key = f"{sid}_{tid}" if (sid and tid) else ""

    cache = get_ocr_cache()
    if cache_key and cache_key in cache:
        cached_entries = cache[cache_key]
        results = [([], entry[0], float(entry[1])) for entry in cached_entries]
    else:
        reader = get_ocr_reader()
        if reader is None:
            return empty_result

        # Format image for EasyOCR
        if len(image_crop.shape) == 2:
            img_ocr = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2RGB)
        elif len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
            img_ocr = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        else:
            return empty_result

        # Downsample large crops to max height 96px for high-throughput CPU inference
        if img_ocr.shape[0] > 96:
            scale = 96.0 / img_ocr.shape[0]
            new_w = max(16, int(img_ocr.shape[1] * scale))
            img_ocr = cv2.resize(img_ocr, (new_w, 96), interpolation=cv2.INTER_AREA)

        try:
            results = reader.readtext(img_ocr)
            if cache_key:
                cache[cache_key] = [[str(r[1]), float(r[2])] for r in results]
        except Exception:
            return empty_result

    task_name = str(sentence_json.get("task_name", ""))
    script = str(sentence_json.get("script", "latin"))

    if not results:
        # Zero words recognized by OCR (severe illegibility or empty crop)
        exp_words = EXPECTED_WORD_COUNTS.get(task_name)
        exp_chars = EXPECTED_CHAR_COUNTS.get(task_name)
        return {
            "ocr_confidence_mean": 0.0,
            "ocr_confidence_std": 0.0,
            "ocr_low_conf_ratio": 1.0,
            "ocr_words_recognized": 0.0,
            "ocr_word_completion_ratio": 0.0 if exp_words else np.nan,
            "ocr_char_count": 0.0,
            "ocr_char_completion_ratio": 0.0 if exp_chars else np.nan,
            "ocr_omission_ratio": 1.0 if exp_chars else np.nan,
            "ocr_cer": 1.0 if task_name in CANDIDATE_PROMPTS else np.nan,
            "ocr_wer": 1.0 if task_name in CANDIDATE_PROMPTS else np.nan,
        }

    raw_tokens = [str(r[1]) for r in results]
    confs = [float(r[2]) for r in results]

    conf_mean = float(np.mean(confs))
    conf_std = float(np.std(confs)) if len(confs) > 1 else 0.0
    low_conf_ratio = float(np.mean([1.0 if c < 0.40 else 0.0 for c in confs]))

    recognized_words_count = float(len(raw_tokens))
    combined_raw_text = " ".join(raw_tokens)
    cleaned_ocr_text = clean_text(combined_raw_text)
    ocr_tokens = cleaned_ocr_text.split()
    total_chars = float(len(cleaned_ocr_text.replace(" ", "")))

    exp_words = EXPECTED_WORD_COUNTS.get(task_name)
    word_comp_ratio = (recognized_words_count / exp_words) if exp_words else np.nan

    exp_chars = EXPECTED_CHAR_COUNTS.get(task_name)
    char_comp_ratio = (total_chars / exp_chars) if exp_chars else np.nan
    omission_ratio = max(0.0, (exp_chars - total_chars) / exp_chars) if exp_chars else np.nan

    # Calculate minimum CER and WER against candidate prompts
    cer = np.nan
    wer = np.nan

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

            cer_scores.append(d_char / cand_len)
            wer_scores.append(d_word / cand_wlen)

        cer = float(min(cer_scores))
        wer = float(min(wer_scores))

    return {
        "ocr_confidence_mean": conf_mean,
        "ocr_confidence_std": conf_std,
        "ocr_low_conf_ratio": low_conf_ratio,
        "ocr_words_recognized": recognized_words_count,
        "ocr_word_completion_ratio": word_comp_ratio,
        "ocr_char_count": total_chars,
        "ocr_char_completion_ratio": char_comp_ratio,
        "ocr_omission_ratio": omission_ratio,
        "ocr_cer": cer,
        "ocr_wer": wer,
    }


def _worker_ocr_task(task_tuple: Tuple[str, str, str]) -> Tuple[str, List[List[Any]]]:
    """Worker function to perform OCR on a single image crop."""
    cache_key, crop_path_str, _ = task_tuple
    reader = get_ocr_reader()
    if reader is None or not os.path.exists(crop_path_str):
        return cache_key, []
    try:
        img = cv2.imread(crop_path_str)
        if img is None:
            return cache_key, []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[0] > 96:
            scale = 96.0 / img_rgb.shape[0]
            new_w = max(16, int(img_rgb.shape[1] * scale))
            img_rgb = cv2.resize(img_rgb, (new_w, 96), interpolation=cv2.INTER_AREA)
        res = reader.readtext(img_rgb)
        return cache_key, [[str(r[1]), float(r[2])] for r in res]
    except Exception:
        return cache_key, []


def precompute_ocr_cache(processed_dir: str = "data/processed", workers: int = 4) -> None:
    """Scan all processed sentence crops and populate data/features/ocr_cache.json."""
    import glob
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    search_glob = os.path.join(processed_dir, "*", "*", "sentence_*.json")
    json_paths = sorted(glob.glob(search_glob))
    print(f"Found {len(json_paths)} sentence JSON files to inspect for OCR caching.")

    cache = get_ocr_cache()
    tasks_to_run = []

    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                sj = json.load(f)
            sid = str(sj.get("student_id", Path(jp).parent.name))
            tid = str(sj.get("task_id", Path(jp).stem))
            key = f"{sid}_{tid}"
            if key not in cache:
                crop_p = Path(jp).parent / sj.get("crop_filename", f"{tid}.png")
                if crop_p.exists():
                    tasks_to_run.append((key, str(crop_p), str(jp)))
        except Exception:
            continue

    print(f"Already cached: {len(cache)}. Remaining tasks to OCR: {len(tasks_to_run)}")
    if not tasks_to_run:
        print("All crops already cached in data/features/ocr_cache.json!")
        return

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(_worker_ocr_task, tasks_to_run), total=len(tasks_to_run), desc=f"OCR Caching ({workers} workers)"))
    else:
        results = [
            _worker_ocr_task(t)
            for t in tqdm(tasks_to_run, desc="OCR Caching (1 worker)")
        ]

    for k, entries in results:
        cache[k] = entries

    save_ocr_cache(cache)
    print(f"Successfully saved {len(cache)} entries to {CACHE_FILE_PATH}!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Precompute EasyOCR results cache.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    precompute_ocr_cache(processed_dir=args.processed_dir, workers=args.workers)

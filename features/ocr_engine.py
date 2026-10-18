"""Rigid Dual-Engine Transformer OCR (HTR) Engine for Dysgraphia Detection.

Replaces legacy heuristic / scene-text OCR (EasyOCR) with deep Handwritten Text
Recognition (HTR) transformer models:
1. English Tasks: Microsoft TrOCR Small Handwritten (`microsoft/trocr-small-handwritten`).
   - Line-level Vision Transformer (ViT) encoder + RoBERTa/TrOCR decoder.
   - Evaluates child handwriting line crops sliced via physical line segmentations.
   - Derives exact posterior token probabilities from autoregressive transition scores.
2. Hindi Tasks: Devanagari TrOCR (`paudelanil/trocr-devanagari-2`).
   - Word-level ViT + Devanagari language decoder trained on handwritten Indic words.
   - Slices word tokens via physical line word bounding boxes.
   - Derives word-level posterior probabilities and character legibility metrics.

Features computed:
- Text transcription and token sequence
- Mean, min, and standard deviation of token confidence scores
- Low-confidence token fraction (ratio of tokens < 0.40 confidence)
- Total recognized words and character counts
- Full disk caching in data/features/trocr_cache.json to avoid redundant CPU inference.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

CACHE_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "features" / "trocr_cache.json"


@dataclass
class OCRLineResult:
    """Transcription and confidence metrics for an individual line or word crop."""
    text: str
    confidence: float
    token_confs: List[float] = field(default_factory=list)
    bbox: Optional[List[int]] = None


@dataclass
class OCRTaskResult:
    """Aggregated OCR transcript and confidence metrics for a complete sentence task."""
    text: str
    tokens: List[str]
    token_confidences: List[float]
    mean_confidence: float
    min_confidence: float
    std_confidence: float
    low_conf_ratio: float
    char_count: int
    word_count: int
    line_count: int
    lines: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_GLOBAL_CACHE: Optional[Dict[str, Any]] = None


def get_trocr_cache() -> Dict[str, Any]:
    """Retrieve or lazily load the global TrOCR disk cache."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        if CACHE_FILE_PATH.exists():
            try:
                with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                    _GLOBAL_CACHE = json.load(f)
            except Exception as e:
                logger.warning("Failed to load TrOCR cache from %s: %s", CACHE_FILE_PATH, e)
                _GLOBAL_CACHE = {}
        else:
            _GLOBAL_CACHE = {}
    return _GLOBAL_CACHE


def save_trocr_cache(cache_dict: Optional[Dict[str, Any]] = None) -> None:
    """Save in-memory TrOCR results to disk cache."""
    global _GLOBAL_CACHE
    target = cache_dict if cache_dict is not None else _GLOBAL_CACHE
    if target is None:
        return
    CACHE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(target, f, indent=2, ensure_ascii=False)


def segment_lines_projection(image_gray: np.ndarray, min_line_height: int = 24) -> List[Tuple[int, int]]:
    """Segment line slices using horizontal projection profile valleys.

    Fallback when physical_lines are not available or invalid.
    """
    if len(image_gray.shape) == 3:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    # Invert so ink is white
    _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(thresh > 0, axis=1)

    # Threshold for ink presence on a line
    has_ink = proj > (0.01 * np.max(proj) if np.max(proj) > 0 else 0)

    lines: List[Tuple[int, int]] = []
    in_line = False
    start_y = 0

    for y, active in enumerate(has_ink):
        if active and not in_line:
            in_line = True
            start_y = y
        elif not active and in_line:
            in_line = False
            if (y - start_y) >= min_line_height:
                lines.append((start_y, y))

    if in_line and (len(has_ink) - start_y) >= min_line_height:
        lines.append((start_y, len(has_ink)))

    if not lines:
        lines.append((0, image_gray.shape[0]))

    return lines


def _extract_token_confidences(outputs: Any, clean_str: str) -> List[float]:
    """Safely extract token posterior probabilities from model generation outputs."""
    token_confs: List[float] = []
    if outputs.scores and len(outputs.scores) > 0:
        try:
            for step_idx, step_logits in enumerate(outputs.scores):
                if step_idx + 1 < outputs.sequences.shape[1]:
                    token_id = outputs.sequences[0, step_idx + 1]
                    prob = float(torch.softmax(step_logits[0], dim=-1)[token_id].item())
                    token_confs.append(prob)
        except Exception:
            pass

    if not token_confs:
        token_confs = [1.0] if clean_str else [0.0]
    return token_confs


class EnglishTrOCREngine:
    """Microsoft TrOCR Small Handwritten line-level HTR engine."""

    def __init__(self, model_name: str = "microsoft/trocr-small-handwritten", device: Optional[str] = None):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def transcribe_line(self, line_img: np.ndarray, max_new_tokens: int = 40) -> OCRLineResult:
        """Transcribe a single horizontal line crop with posterior token probabilities."""
        if line_img is None or line_img.size == 0 or line_img.shape[0] < 8 or line_img.shape[1] < 8:
            return OCRLineResult(text="", confidence=0.0, token_confs=[])

        if len(line_img.shape) == 2:
            rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
        elif line_img.shape[2] == 3:
            rgb = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        else:
            return OCRLineResult(text="", confidence=0.0, token_confs=[])

        pil_img = Image.fromarray(rgb)
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
            raw_text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            clean_str = " ".join(raw_text.strip().split())

            token_confs = _extract_token_confidences(outputs, clean_str)
            mean_conf = float(np.mean(token_confs)) if token_confs else 0.0

        return OCRLineResult(text=clean_str, confidence=mean_conf, token_confs=token_confs)

    def transcribe_task(
        self,
        image_crop: np.ndarray,
        physical_lines: Optional[List[Dict[str, Any]]] = None,
        crop_bbox: Optional[List[int]] = None,
    ) -> OCRTaskResult:
        """Transcribe an English sentence crop by line-level slicing and scoring."""
        if image_crop is None or image_crop.size == 0:
            return self._empty_result()

        line_results: List[OCRLineResult] = []
        h, w = image_crop.shape[:2]

        if physical_lines and crop_bbox and len(crop_bbox) >= 4:
            # Map full-page line bounding boxes to local sentence crop coordinates
            for pline in physical_lines:
                bbox = pline.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                bx, by, bw, bh = bbox
                lx = max(0, bx - crop_bbox[0])
                ly = max(0, by - crop_bbox[1])
                lw = min(w - lx, bw)
                lh = min(h - ly, bh)

                if lw <= 10 or lh <= 10:
                    continue

                line_slice = image_crop[ly : ly + lh, lx : lx + lw]
                res = self.transcribe_line(line_slice)
                res.bbox = [int(lx), int(ly), int(lw), int(lh)]
                line_results.append(res)

        # Fallback to projection slicing if no valid physical lines
        if not line_results:
            y_intervals = segment_lines_projection(image_crop)
            for y1, y2 in y_intervals:
                line_slice = image_crop[y1:y2, :]
                res = self.transcribe_line(line_slice)
                res.bbox = [0, int(y1), int(w), int(y2 - y1)]
                line_results.append(res)

        return self._aggregate_results(line_results)

    def _aggregate_results(self, line_results: List[OCRLineResult]) -> OCRTaskResult:
        transcripts = [lr.text for lr in line_results if lr.text]
        full_text = " ".join(transcripts)
        tokens = full_text.split()

        all_confs: List[float] = []
        for lr in line_results:
            all_confs.extend(lr.token_confs)

        if not all_confs:
            return self._empty_result(line_count=len(line_results))

        mean_c = float(np.mean(all_confs))
        min_c = float(np.min(all_confs))
        std_c = float(np.std(all_confs)) if len(all_confs) > 1 else 0.0
        low_c = float(np.mean([1.0 if c < 0.40 else 0.0 for c in all_confs]))

        char_count = len(full_text.replace(" ", ""))
        word_count = len(tokens)

        return OCRTaskResult(
            text=full_text,
            tokens=tokens,
            token_confidences=all_confs,
            mean_confidence=mean_c,
            min_confidence=min_c,
            std_confidence=std_c,
            low_conf_ratio=low_c,
            char_count=char_count,
            word_count=word_count,
            line_count=len(line_results),
            lines=[{"text": lr.text, "confidence": lr.confidence, "bbox": lr.bbox} for lr in line_results],
        )

    def _empty_result(self, line_count: int = 0) -> OCRTaskResult:
        return OCRTaskResult(
            text="",
            tokens=[],
            token_confidences=[],
            mean_confidence=0.0,
            min_confidence=0.0,
            std_confidence=0.0,
            low_conf_ratio=1.0,
            char_count=0,
            word_count=0,
            line_count=line_count,
            lines=[],
        )


class DevanagariTrOCREngine:
    """Devanagari TrOCR word-level HTR engine."""

    def __init__(self, model_name: str = "paudelanil/trocr-devanagari-2", device: Optional[str] = None):
        from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = ViTImageProcessor(
            do_normalize=True,
            do_rescale=True,
            do_resize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            size={"height": 224, "width": 224},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def transcribe_word(self, word_img: np.ndarray, max_new_tokens: int = 20) -> OCRLineResult:
        """Transcribe an individual Devanagari word crop."""
        if word_img is None or word_img.size == 0 or word_img.shape[0] < 8 or word_img.shape[1] < 8:
            return OCRLineResult(text="", confidence=0.0, token_confs=[])

        if len(word_img.shape) == 2:
            rgb = cv2.cvtColor(word_img, cv2.COLOR_GRAY2RGB)
        elif word_img.shape[2] == 3:
            rgb = cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB)
        else:
            return OCRLineResult(text="", confidence=0.0, token_confs=[])

        pil_img = Image.fromarray(rgb)
        pixel_values = self.image_processor(pil_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
            raw_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            clean_str = " ".join(raw_text.strip().split())

            token_confs = _extract_token_confidences(outputs, clean_str)
            mean_conf = float(np.mean(token_confs)) if token_confs else 0.0

        return OCRLineResult(text=clean_str, confidence=mean_conf, token_confs=token_confs)

    def transcribe_task(
        self,
        image_crop: np.ndarray,
        physical_lines: Optional[List[Dict[str, Any]]] = None,
        crop_bbox: Optional[List[int]] = None,
    ) -> OCRTaskResult:
        """Transcribe a Devanagari task crop by extracting word crops from line data."""
        if image_crop is None or image_crop.size == 0:
            return self._empty_result()

        word_results: List[OCRLineResult] = []
        h, w = image_crop.shape[:2]

        if physical_lines and crop_bbox and len(crop_bbox) >= 4:
            for pline in physical_lines:
                words = pline.get("words", [])
                for word in words:
                    bbox = word.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    bx, by, bw, bh = bbox
                    lx = max(0, bx - crop_bbox[0])
                    ly = max(0, by - crop_bbox[1])
                    lw = min(w - lx, bw)
                    lh = min(h - ly, bh)

                    if lw <= 8 or lh <= 8:
                        continue

                    w_crop = image_crop[ly : ly + lh, lx : lx + lw]
                    res = self.transcribe_word(w_crop)
                    res.bbox = [int(lx), int(ly), int(lw), int(lh)]
                    word_results.append(res)

        # Fallback if no word boxes: segment by connected components / contours
        if not word_results:
            contours = self._find_word_contours(image_crop)
            for c_bbox in contours:
                cx, cy, cw, ch = c_bbox
                w_crop = image_crop[cy : cy + ch, cx : cx + cw]
                res = self.transcribe_word(w_crop)
                res.bbox = [int(cx), int(cy), int(cw), int(ch)]
                word_results.append(res)

        return self._aggregate_results(word_results)

    def _find_word_contours(self, image_crop: np.ndarray) -> List[List[int]]:
        """Fallback word box finder using morphological dilation."""
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 15 and h >= 15:
                bboxes.append([x, y, w, h])

        # Sort left-to-right, top-to-bottom
        bboxes.sort(key=lambda b: (b[1] // 40, b[0]))
        return bboxes

    def _aggregate_results(self, word_results: List[OCRLineResult]) -> OCRTaskResult:
        words_text = [wr.text for wr in word_results if wr.text]
        full_text = " ".join(words_text)
        tokens = full_text.split()

        all_confs: List[float] = []
        for wr in word_results:
            all_confs.extend(wr.token_confs)

        if not all_confs:
            return self._empty_result(line_count=len(word_results))

        mean_c = float(np.mean(all_confs))
        min_c = float(np.min(all_confs))
        std_c = float(np.std(all_confs)) if len(all_confs) > 1 else 0.0
        low_c = float(np.mean([1.0 if c < 0.40 else 0.0 for c in all_confs]))

        char_count = len(full_text.replace(" ", ""))
        word_count = len(tokens)

        return OCRTaskResult(
            text=full_text,
            tokens=tokens,
            token_confidences=all_confs,
            mean_confidence=mean_c,
            min_confidence=min_c,
            std_confidence=std_c,
            low_conf_ratio=low_c,
            char_count=char_count,
            word_count=word_count,
            line_count=len(word_results),
            lines=[{"text": wr.text, "confidence": wr.confidence, "bbox": wr.bbox} for wr in word_results],
        )

    def _empty_result(self, line_count: int = 0) -> OCRTaskResult:
        return OCRTaskResult(
            text="",
            tokens=[],
            token_confidences=[],
            mean_confidence=0.0,
            min_confidence=0.0,
            std_confidence=0.0,
            low_conf_ratio=1.0,
            char_count=0,
            word_count=0,
            line_count=line_count,
            lines=[],
        )


# Global singleton engines
_ENGLISH_ENGINE: Optional[EnglishTrOCREngine] = None
_DEVANAGARI_ENGINE: Optional[DevanagariTrOCREngine] = None


def get_english_engine() -> EnglishTrOCREngine:
    global _ENGLISH_ENGINE
    if _ENGLISH_ENGINE is None:
        _ENGLISH_ENGINE = EnglishTrOCREngine()
    return _ENGLISH_ENGINE


def get_devanagari_engine() -> DevanagariTrOCREngine:
    global _DEVANAGARI_ENGINE
    if _DEVANAGARI_ENGINE is None:
        _DEVANAGARI_ENGINE = DevanagariTrOCREngine()
    return _DEVANAGARI_ENGINE


class DualEngineOCRPipeline:
    """Dispatches sentence task transcription to either English or Devanagari TrOCR."""

    def __init__(self):
        self.cache = get_trocr_cache()

    def transcribe_task(
        self,
        image_crop: np.ndarray,
        sentence_json: Dict[str, Any],
        use_cache: bool = True,
    ) -> OCRTaskResult:
        """Transcribe sentence task using script-specific TrOCR engine with disk caching."""
        sid = str(sentence_json.get("student_id", ""))
        tid = str(sentence_json.get("task_id", ""))
        cache_key = f"{sid}_{tid}" if (sid and tid) else ""

        if use_cache and cache_key and cache_key in self.cache:
            cached = self.cache[cache_key]
            return OCRTaskResult(
                text=cached.get("text", ""),
                tokens=cached.get("tokens", []),
                token_confidences=cached.get("token_confidences", []),
                mean_confidence=float(cached.get("mean_confidence", 0.0)),
                min_confidence=float(cached.get("min_confidence", 0.0)),
                std_confidence=float(cached.get("std_confidence", 0.0)),
                low_conf_ratio=float(cached.get("low_conf_ratio", 1.0)),
                char_count=int(cached.get("char_count", 0)),
                word_count=int(cached.get("word_count", 0)),
                line_count=int(cached.get("line_count", 0)),
                lines=cached.get("lines", []),
            )

        script = str(sentence_json.get("script", "latin")).lower()
        task_name = str(sentence_json.get("task_name", "")).lower()
        is_hindi = "hindi" in task_name or script == "devanagari"

        physical_lines = sentence_json.get("physical_lines", [])
        crop_bbox = sentence_json.get("crop_bbox")

        if is_hindi:
            engine = get_devanagari_engine()
        else:
            engine = get_english_engine()

        result = engine.transcribe_task(image_crop, physical_lines=physical_lines, crop_bbox=crop_bbox)

        if cache_key:
            self.cache[cache_key] = result.to_dict()

        return result

    def persist_cache(self) -> None:
        """Persist in-memory results to disk."""
        save_trocr_cache(self.cache)

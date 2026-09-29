"""
Context-Aware OCR Engine for Dysgraphic Handwriting
====================================================

Multi-layered probabilistic OCR pipeline that mimics human reading strategies:
1. Read easy words first (visual model)
2. Analyze stroke geometry for partial signals (stroke primitives)
3. Use surrounding context to rescue illegible words (language model re-ranker)
4. Report confidence and alternatives for uncertain words

Designed to work with the existing Dysgraphia Detection BHK feature pipeline.
"""

__version__ = "0.1.0"

from src.ocr.utils import (
    CharHypothesis,
    WordHypothesis,
    LineResult,
    TranscriptionResult,
    ConfidenceTier,
    StrokePrimitive,
    StrokeAnalysis,
    OCRDysgraphiaFeatures,
    render_confidence_overlay,
)
from src.ocr.segmentation import segment_image, LineRegion, WordRegion, normalize_word_image
from src.ocr.stroke_features import extract_stroke_features, compute_char_prior
from src.ocr.char_hypothesis import CRNNModel, logits_to_word_hypotheses
from src.ocr.word_recognizer import WordRecognizer
from src.ocr.language_reranker import LanguageReRanker
from src.ocr.fusion import compute_word_confidence, extract_ocr_dysgraphia_features
from src.ocr.augmentation import DysgraphiaAugmenter
from src.ocr.pipeline import ContextAwareOCRPipeline

__all__ = [
    "ContextAwareOCRPipeline",
    "WordRecognizer",
    "LanguageReRanker",
    "DysgraphiaAugmenter",
    "CRNNModel",
    "CharHypothesis",
    "WordHypothesis",
    "LineResult",
    "TranscriptionResult",
    "ConfidenceTier",
    "StrokePrimitive",
    "StrokeAnalysis",
    "OCRDysgraphiaFeatures",
    "render_confidence_overlay",
    "segment_image",
    "normalize_word_image",
    "extract_stroke_features",
    "compute_char_prior",
    "extract_ocr_dysgraphia_features",
]

"""
End-to-End Context-Aware OCR Pipeline.

Orchestrates the entire multi-layered recognition process:
  Raw Image → Preprocessing → Segmentation → Stroke Analysis →
  Visual CRNN Recognition → Sentence Context Re-ranking →
  Confidence Calibration → Annotated Transcription & BHK Diagnostic Features.
"""

from __future__ import annotations

import logging
from typing import Union, List, Optional, Tuple, Dict
import numpy as np
import cv2

from src.ocr.utils import (
    TranscriptionResult,
    LineResult,
    WordHypothesis,
    OCRDysgraphiaFeatures,
    render_confidence_overlay,
)
from src.ocr.segmentation import segment_image, LineRegion
from src.ocr.word_recognizer import WordRecognizer
from src.ocr.language_reranker import LanguageReRanker
from src.ocr.fusion import finalize_line_confidence, extract_ocr_dysgraphia_features
from src.preprocessing import preprocess_handwriting_image, remove_guide_lines

logger = logging.getLogger(__name__)


class ContextAwareOCRPipeline:
    """
    End-to-end OCR engine for dysgraphic and degraded handwriting.
    """

    def __init__(
        self,
        crnn_model_path: Optional[str] = None,
        use_neural_lm: bool = False,
        neural_lm_name: Optional[str] = None,
        beam_width: int = 15,
        stroke_weight: float = 0.15,
        device: Optional[str] = None,
    ):
        """
        Args:
            crnn_model_path: Checkpoint path for trained CRNN.
            use_neural_lm: Whether to load neural language model (e.g. GPT-2).
            neural_lm_name: HuggingFace model identifier.
            beam_width: CTC and lattice beam search width.
            stroke_weight: Weight for stroke topological prior.
            device: 'cuda' or 'cpu'.
        """
        if crnn_model_path is None:
            import os
            default_path = "models/crnn_iam/checkpoint_best.pth"
            if os.path.exists(default_path):
                crnn_model_path = default_path

        logger.info(f"Initializing ContextAwareOCRPipeline (CRNN checkpoint: {crnn_model_path})...")
        self.word_recognizer = WordRecognizer(
            model_path=crnn_model_path,
            beam_width=beam_width,
            stroke_weight=stroke_weight,
            device=device,
        )

        self.language_reranker = LanguageReRanker(
            neural_model_name=neural_lm_name,
            use_neural_lm=use_neural_lm,
            beam_width=30,
            device=device,
        )

    def preprocess(
        self,
        image_input: Union[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and prepares the handwriting image.

        Args:
            image_input: File path (str) or image array (BGR / Grayscale).

        Returns:
            (original_bgr, binary_mask) where binary_mask has ink=255, bg=0.
        """
        if isinstance(image_input, str):
            orig = cv2.imread(image_input)
            if orig is None:
                raise ValueError(f"Unable to read image at: {image_input}")
        elif isinstance(image_input, np.ndarray):
            orig = image_input.copy()
        else:
            raise TypeError("image_input must be a file path string or numpy array.")

        # Ensure 3-channel BGR for visualization
        if len(orig.shape) == 2:
            bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        else:
            bgr = orig

        # Use unified preprocessing: auto-polarity, binarization, guide line filtering
        bin_mask, _ = preprocess_handwriting_image(bgr)

        return bgr, bin_mask

    def transcribe(
        self,
        image_input: Union[str, np.ndarray],
    ) -> TranscriptionResult:
        """
        Transcribe a handwriting document end-to-end.

        Args:
            image_input: File path or numpy image array.

        Returns:
            TranscriptionResult containing lines, words, confidences,
            diagnostics, and OCR-derived dysgraphia features.
        """
        bgr, binary_mask = self.preprocess(image_input)

        # Step 1: Segmentation (Lines -> Words)
        lines: List[LineRegion] = segment_image(binary_mask)

        processed_lines: List[LineResult] = []
        low_confidence_words: List[Tuple[int, int, WordHypothesis]] = []

        # Step 2-5: Recognize and re-rank line by line
        for line_idx, line in enumerate(lines):
            if not line.words:
                continue

            # Visual recognition with stroke prior -> List[List[WordHypothesis]]
            line_hypotheses = self.word_recognizer.recognize_line(line)

            # Contextual LM Re-ranking across word lattice -> LineResult
            line_result = self.language_reranker.rerank_line(line_hypotheses)

            # Confidence Fusion & Calibration
            line_result = finalize_line_confidence(line_result)
            line_result.bbox = line.bbox

            # Track low-confidence words
            for w_idx, word in enumerate(line_result.words):
                if word.confidence < 0.50:
                    low_confidence_words.append((line_idx, w_idx, word))

            processed_lines.append(line_result)

        # Step 6: Assemble full document transcription
        full_text = "\n".join(l.raw_text for l in processed_lines if l.raw_text)

        mean_conf = 1.0
        if processed_lines:
            mean_conf = float(np.mean([l.line_confidence for l in processed_lines]))

        transcription = TranscriptionResult(
            lines=processed_lines,
            full_text=full_text,
            mean_confidence=mean_conf,
            low_confidence_words=low_confidence_words,
            image_shape=bgr.shape,
        )

        # Step 7: Extract OCR Dysgraphia features
        ocr_features = extract_ocr_dysgraphia_features(transcription)
        transcription.metadata["ocr_dysgraphia_features"] = ocr_features

        return transcription

    def render_overlay(
        self,
        original_image: np.ndarray,
        transcription: TranscriptionResult,
    ) -> np.ndarray:
        """
        Render interactive confidence heatmap boxes and annotations on the original image.
        """
        return render_confidence_overlay(original_image, transcription)

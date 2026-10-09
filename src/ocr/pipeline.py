"""
End-to-End Context-Aware OCR Pipeline.

Orchestrates the entire multi-layered recognition process:
  Raw Image → Preprocessing → Segmentation → Stroke Analysis →
  Visual CRNN Recognition → Sentence Context Re-ranking →
  Confidence Calibration → Annotated Transcription & BHK Diagnostic Features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
import numpy as np
import cv2

from src.ocr.utils import (
    TranscriptionResult,
    LineResult,
    WordHypothesis,
    OCRDysgraphiaFeatures,
    render_confidence_overlay,
    assign_confidence_tier,
)
from src.ocr.char_hypothesis import CharHypothesis
from src.ocr.segmentation import segment_image, LineRegion
from src.ocr.word_recognizer import WordRecognizer
from src.ocr.language_reranker import LanguageReRanker
from src.ocr.fusion import finalize_line_confidence, extract_ocr_dysgraphia_features
from src.ocr.forced_alignment import ForcedAlignmentEngine, DiagnosticCopyTaskResult
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
        backend: str = "trocr",
    ):
        """
        Args:
            crnn_model_path: Checkpoint path for trained CRNN or Transformer.
            use_neural_lm: Whether to load neural language model (e.g. GPT-2).
            neural_lm_name: HuggingFace model identifier.
            beam_width: CTC and lattice beam search width.
            stroke_weight: Weight for stroke topological prior.
            device: 'cuda' or 'cpu'.
            backend: 'transformer' (In-House CNN-Transformer), 'trocr', or 'crnn'.
        """
        if crnn_model_path is None:
            import os
            trans_p = "models/transformer_ocr/checkpoint_best.pth"
            crnn_p = "models/crnn_iam/checkpoint_best.pth"
            if os.path.exists(trans_p):
                crnn_model_path = trans_p
            elif os.path.exists(crnn_p):
                crnn_model_path = crnn_p

        logger.info(f"Initializing ContextAwareOCRPipeline (OCR backend: {backend}, checkpoint: {crnn_model_path})...")
        self.word_recognizer = WordRecognizer(
            model_path=crnn_model_path,
            beam_width=beam_width,
            stroke_weight=stroke_weight,
            device=device,
            backend=backend,
        )

        self.language_reranker = LanguageReRanker(
            neural_model_name=neural_lm_name,
            use_neural_lm=use_neural_lm,
            beam_width=30,
            device=device,
        )

        self.forced_alignment_engine = ForcedAlignmentEngine()

    def preprocess(
        self,
        image_input: Union[str, np.ndarray],
        rotation: Optional[Union[int, str]] = None,
        auto_orient: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and prepares the handwriting image.

        Args:
            image_input: File path (str) or image array (BGR / Grayscale).
            rotation: Optional manual rotation.
            auto_orient: Whether to auto-detect text line orientation.

        Returns:
            (original_bgr, binary_mask) where binary_mask has ink=255, bg=0.
        """
        if isinstance(image_input, str):
            p = Path(image_input)
            if not p.exists():
                raise ValueError(f"Unable to read image at: {image_input}")
            try:
                from PIL import Image, ImageOps
                pil_img = Image.open(str(p))
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode == "RGB":
                    orig = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                elif pil_img.mode == "RGBA":
                    orig = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
                else:
                    orig = np.array(pil_img)
            except Exception:
                orig = cv2.imread(str(p))
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
        bin_mask, _, meta = preprocess_handwriting_image(
            bgr,
            rotation=rotation,
            auto_orient=auto_orient,
            return_metadata=True,
        )
        bgr = meta.get("clean_color", bgr)

        return bgr, bin_mask

    def transcribe(
        self,
        image_input: Union[str, np.ndarray],
        rotation: Optional[Union[int, str]] = None,
        auto_orient: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe a handwriting document end-to-end.

        Args:
            image_input: File path or numpy image array.
            rotation: Optional manual rotation.
            auto_orient: Whether to auto-detect orientation.

        Returns:
            TranscriptionResult containing lines, words, confidences,
            diagnostics, and OCR-derived dysgraphia features.
        """
        bgr, binary_mask = self.preprocess(
            image_input,
            rotation=rotation,
            auto_orient=auto_orient,
        )

        # Step 1: Segmentation (Lines -> Words)
        lines: List[LineRegion] = segment_image(binary_mask)

        processed_lines: List[LineResult] = []
        low_confidence_words: List[Tuple[int, int, WordHypothesis]] = []

        # Step 2-5: Recognize and re-rank line by line
        for line_idx, line in enumerate(lines):
            lx, ly, lw, lh = line.bbox_in_image
            line_crop = bgr[ly : ly + lh, lx : lx + lw]

            # High-Accuracy TrOCR Line-First Pipeline with Valley Projection Alignment
            if (
                self.word_recognizer.backend == "trocr"
                and self.word_recognizer.trocr_engine is not None
                and line_crop.size > 0
            ):
                phys_boxes = [w.bbox_in_image for w in line.words] if line.words else None
                word_records = self.word_recognizer.trocr_engine.recognize_line_with_word_alignment(
                    line_crop,
                    global_origin=(lx, ly),
                    physical_word_boxes=phys_boxes,
                )
                line_words: List[WordHypothesis] = []
                for wr in word_records:
                    w_text = wr["text"]
                    w_conf = wr["confidence"]
                    w_box = wr["bbox"]
                    tier = assign_confidence_tier(w_conf)
                    char_hyps = [CharHypothesis(char=c, confidence=w_conf) for c in w_text]
                    hyp = WordHypothesis(
                        text=w_text,
                        visual_score=float(np.log(max(w_conf, 1e-4))),
                        confidence=w_conf,
                        confidence_tier=tier,
                        bbox=w_box,
                        char_hypotheses=char_hyps,
                    )
                    line_words.append(hyp)

                line_result = LineResult(
                    words=line_words,
                    line_index=line_idx,
                    line_confidence=float(np.mean([w.confidence for w in line_words])) if line_words else 0.85,
                    bbox=line.bbox,
                )
                if hasattr(self.language_reranker, "refine_line_result"):
                    line_result = self.language_reranker.refine_line_result(line_result)
            else:
                if not line.words:
                    continue
                # Fallback to word-level recognition
                line_hypotheses = self.word_recognizer.recognize_line(line)
                line_result = self.language_reranker.rerank_line(line_hypotheses)
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

    def transcribe_with_prompt(
        self,
        image_input: Union[str, np.ndarray],
        prompt_text: Optional[str] = None,
    ) -> Tuple[TranscriptionResult, Optional[DiagnosticCopyTaskResult]]:
        """
        Transcribe handwriting with optional prompt-guided forced alignment
        for copy-task dysgraphia diagnostic error profiling.
        """
        transcription = self.transcribe(image_input)
        copy_task_diag = None

        if prompt_text and prompt_text.strip():
            flat_words = []
            for line in transcription.lines:
                for word in line.words:
                    char_confs = [c.confidence for c in word.char_hypotheses] if word.char_hypotheses else []
                    flat_words.append({
                        "text": word.text,
                        "confidence": word.confidence,
                        "char_confidences": char_confs,
                        "bbox": word.bbox,
                    })

            copy_task_diag = self.forced_alignment_engine.evaluate_copy_task(
                prompt_text=prompt_text,
                transcribed_words=flat_words,
            )
            transcription.metadata["copy_task_diagnostic"] = copy_task_diag

        return transcription, copy_task_diag


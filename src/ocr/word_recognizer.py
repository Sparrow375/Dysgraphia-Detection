"""
Word Recognizer Module — Visual Recognition & Candidate Generation.

Coordinates the visual word recognition process:
  1. Normalizes word image (deskew, aspect ratio scaling, padding)
  2. Runs topological stroke feature extraction to compute P(char | stroke) priors
  3. Feeds word image to CRNN (or TrOCR / heuristic fallback)
  4. Runs stroke-prior-augmented CTC beam search
  5. Returns ranked List[WordHypothesis] for downstream language re-ranking.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.ocr.utils import (
    WordHypothesis,
    CharHypothesis,
    StrokeAnalysis,
    assign_confidence_tier,
    ConfidenceTier,
)
from src.ocr.segmentation import WordRegion, LineRegion, normalize_word_image
from src.ocr.stroke_features import (
    extract_stroke_features,
    compute_char_prior,
    stroke_prior_to_ctc_bias,
)
from src.ocr.char_hypothesis import (
    CRNNModel,
    logits_to_word_hypotheses,
    DEFAULT_ALPHABET,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Word Recognizer Class
# ---------------------------------------------------------------------------

class WordRecognizer:
    """
    Coordinates word-level visual handwriting recognition with stroke priors.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        alphabet: str = DEFAULT_ALPHABET,
        beam_width: int = 15,
        stroke_weight: float = 0.15,
        target_height: int = 64,
        device: Optional[str] = None,
        use_stroke_prior: bool = True,
    ):
        """
        Args:
            model_path: Path to pre-trained CRNN checkpoint (.pth), if available.
            alphabet: Alphabet string matching CRNN output head.
            beam_width: CTC beam search width.
            stroke_weight: Weight α for stroke-prior bias in beam search.
            target_height: Normalized word image height.
            device: 'cuda' or 'cpu'. If None, auto-detected.
            use_stroke_prior: Whether to blend topological stroke priors.
        """
        self.alphabet = alphabet
        self.beam_width = beam_width
        self.stroke_weight = stroke_weight
        self.target_height = target_height
        self.use_stroke_prior = use_stroke_prior

        if device is None:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model: Optional[CRNNModel] = None
        if HAS_TORCH:
            self._init_model(model_path)
        else:
            logger.warning("PyTorch not installed; running in heuristic/fallback OCR mode.")

    def _init_model(self, model_path: Optional[str]):
        """Initialize CRNN model and load weights if provided."""
        from pathlib import Path
        if model_path is None:
            default_ckpt = Path("models/crnn_iam/checkpoint_best.pth")
            if default_ckpt.exists():
                model_path = str(default_ckpt)

        lstm_hidden = 512
        lstm_layers = 3
        state_dict = None

        if model_path and Path(model_path).exists():
            try:
                ckpt = torch.load(model_path, map_location=self.device)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    state_dict = ckpt["model"]
                    lstm_hidden = ckpt.get("lstm_hidden", 512)
                    lstm_layers = ckpt.get("lstm_layers", 3)
                    if "alphabet" in ckpt:
                        self.alphabet = ckpt["alphabet"]
                    best_cer = ckpt.get("best_cer", ckpt.get("CER", None))
                    cer_str = f" (CER={best_cer:.4f})" if best_cer is not None else ""
                    logger.info(f"Loaded trained CRNN weights from {model_path}{cer_str}")
                else:
                    state_dict = ckpt
                    logger.info(f"Loaded CRNN weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load CRNN weights from {model_path}: {e}. Initializing randomly.")

        num_classes = len(self.alphabet) + 1  # +1 for blank
        self.model = CRNNModel(
            n_classes=num_classes,
            input_height=self.target_height,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=0.2,
        )

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        else:
            logger.info("CRNN initialized with default initialization (no checkpoint found).")

        self.model.to(self.device)
        self.model.eval()

    def recognize_word(
        self,
        word_image: np.ndarray,
        word_bbox: Optional[Tuple[int, int, int, int]] = None,
        baseline_slope: float = 0.0,
    ) -> List[WordHypothesis]:
        """
        Recognize a single word image and return top-K ranked hypotheses.

        Args:
            word_image: Binary or grayscale word image patch.
            word_bbox: (x, y, w, h) bounding box in line/page coordinates.
            baseline_slope: Local line baseline slope for deskewing.

        Returns:
            List[WordHypothesis] sorted by visual_score descending.
        """
        if word_image is None or word_image.size == 0 or np.count_nonzero(word_image) < 5:
            return [WordHypothesis(
                text="",
                visual_score=-10.0,
                confidence=0.0,
                bbox=word_bbox,
            )]

        # Ensure bright ink on dark background (matching IAM training polarity)
        if word_image.ndim == 3:
            gray = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = word_image.copy()

        if np.mean(gray) > 127:
            gray = 255 - gray

        # 1. Normalize image
        norm_img = normalize_word_image(
            gray,
            target_height=self.target_height,
            baseline_slope=baseline_slope,
        )

        # 2. Extract stroke features & prior
        stroke_analysis: Optional[StrokeAnalysis] = None
        stroke_bias: Optional[np.ndarray] = None

        if self.use_stroke_prior:
            try:
                stroke_analysis = extract_stroke_features(norm_img)
                char_prior = compute_char_prior(stroke_analysis)
                stroke_bias = stroke_prior_to_ctc_bias(
                    char_prior,
                    alphabet=self.alphabet,
                    weight=self.stroke_weight,
                )
            except Exception as e:
                logger.debug(f"Stroke feature extraction skipped: {e}")

        # 3. Model forward pass
        if HAS_TORCH and self.model is not None:
            # Prepare tensor: (1, 1, H, W) normalized to [0, 1]
            tensor_img = norm_img.astype(np.float32) / 255.0
            tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                log_probs = self.model(tensor_img)  # (T, 1, num_classes)
                logits_np = log_probs.squeeze(1).cpu().numpy()  # (T, num_classes)

            # 4. CTC Beam Search with stroke prior
            hypotheses = logits_to_word_hypotheses(
                logits=logits_np,
                alphabet=self.alphabet,
                beam_width=self.beam_width,
                stroke_bias=stroke_bias,
                word_bbox=word_bbox,
            )
        else:
            # Fallback heuristic generator when PyTorch/weights are not present
            hypotheses = self._heuristic_fallback(norm_img, stroke_analysis, word_bbox)

        # 5. Populate stroke metadata on hypotheses
        if stroke_analysis is not None:
            for h in hypotheses:
                h.metadata["stroke_analysis"] = stroke_analysis

        return hypotheses

    def recognize_line(
        self,
        line: LineRegion,
    ) -> List[List[WordHypothesis]]:
        """
        Recognize all words in a segmented line region.

        Args:
            line: Segmented LineRegion containing word regions.

        Returns:
            List of hypothesis lists, one list per word position in the line.
        """
        line_hypotheses: List[List[WordHypothesis]] = []

        for word_reg in line.words:
            hyps = self.recognize_word(
                word_image=word_reg.image,
                word_bbox=word_reg.bbox_in_image,
                baseline_slope=line.baseline_slope,
            )
            line_hypotheses.append(hyps)

        return line_hypotheses

    def _heuristic_fallback(
        self,
        norm_img: np.ndarray,
        stroke_analysis: Optional[StrokeAnalysis],
        word_bbox: Optional[Tuple[int, int, int, int]],
    ) -> List[WordHypothesis]:
        """
        Lightweight topological fallback when deep neural model is not active.
        Generates plausible character combinations based on stroke primitives.
        """
        if stroke_analysis is None or not stroke_analysis.primitives:
            return [WordHypothesis(
                text="word",
                visual_score=-2.5,
                confidence=0.3,
                confidence_tier=ConfidenceTier.LOW,
                bbox=word_bbox,
            )]

        prior = compute_char_prior(stroke_analysis)
        top_chars = sorted(prior.items(), key=lambda kv: kv[1], reverse=True)
        best_candidate = "".join([c for c, _ in top_chars[:max(2, min(5, len(stroke_analysis.primitives)))]])

        # Build character-level hypotheses from stroke prior
        char_hyps = []
        for char in best_candidate:
            conf = float(prior.get(char, 0.4))
            alts = [(c, round(float(p), 3)) for c, p in top_chars if c != char][:3]
            char_hyps.append(CharHypothesis(char=char, confidence=conf, alternatives=alts))

        primary = WordHypothesis(
            text=best_candidate,
            visual_score=-1.5,
            confidence=0.45,
            confidence_tier=ConfidenceTier.LOW,
            bbox=word_bbox,
            char_hypotheses=char_hyps,
        )

        alt1 = WordHypothesis(
            text=best_candidate[::-1],
            visual_score=-2.2,
            confidence=0.35,
            confidence_tier=ConfidenceTier.LOW,
            bbox=word_bbox,
        )
        primary.alternatives = [alt1]

        return [primary, alt1]

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
from src.ocr.handwriting_transformer import HandwritingTransformerOCR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Word Recognizer Class
# ---------------------------------------------------------------------------

class WordRecognizer:
    """
    Coordinates word-level visual handwriting recognition with stroke priors.
    Supports both CRNN and Vision-to-Sequence Transformer OCR backends.
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
        backend: str = "auto",
    ):
        """
        Args:
            model_path: Path to pre-trained CRNN or Transformer checkpoint (.pth).
            alphabet: Alphabet string matching model output head.
            beam_width: CTC beam search width.
            stroke_weight: Weight α for stroke-prior bias in beam search.
            target_height: Normalized word image height.
            device: 'cuda' or 'cpu'. If None, auto-detected.
            use_stroke_prior: Whether to blend topological stroke priors.
            backend: 'auto', 'transformer', or 'crnn'.
        """
        self.alphabet = alphabet
        self.beam_width = beam_width
        self.stroke_weight = stroke_weight
        self.target_height = target_height
        self.use_stroke_prior = use_stroke_prior
        self.backend = backend.lower()

        if device is None:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model: Optional[Union[CRNNModel, HandwritingTransformerOCR]] = None
        self.trocr_engine = None

        if HAS_TORCH:
            if self.backend in ["trocr", "auto"]:
                try:
                    from src.ocr.trocr_engine import TrOCREngine
                    self.trocr_engine = TrOCREngine(device=self.device)
                    self.backend = "trocr"
                    logger.info("WordRecognizer initialized with TrOCREngine backend!")
                except Exception as e:
                    logger.warning(f"TrOCREngine init failed, falling back: {e}")
                    self._init_model(model_path)
            else:
                self._init_model(model_path)
        else:
            logger.warning("PyTorch not installed; running in heuristic/fallback OCR mode.")

    def _init_model(self, model_path: Optional[str]):
        """Initialize CRNN or Transformer model and load weights if provided."""
        from pathlib import Path
        if model_path is None:
            line_ckpt = Path("models/line_transformer/checkpoint_best.pth")
            transformer_ckpt = Path("models/transformer_ocr/checkpoint_best.pth")
            crnn_ckpt = Path("models/crnn_iam/checkpoint_best.pth")
            if line_ckpt.exists() and self.backend in ["auto", "transformer"]:
                model_path = str(line_ckpt)
            elif transformer_ckpt.exists() and self.backend in ["auto", "transformer"]:
                model_path = str(transformer_ckpt)
            elif crnn_ckpt.exists():
                model_path = str(crnn_ckpt)

        ckpt = None
        if model_path and Path(model_path).exists():
            try:
                ckpt = torch.load(model_path, map_location=self.device)
            except Exception as e:
                logger.warning(f"Could not load checkpoint from {model_path}: {e}")

        # Auto-detect backend
        if self.backend == "auto":
            if ckpt is not None and isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.backend = "transformer"
            elif model_path and "transformer" in str(model_path).lower():
                self.backend = "transformer"
            else:
                self.backend = "crnn"

        num_classes = len(self.alphabet) + 1  # +1 for blank

        if self.backend == "transformer":
            state_dict = ckpt.get("model_state_dict", ckpt) if (ckpt is not None and isinstance(ckpt, dict)) else None
            max_seq = 256
            if ckpt is not None and isinstance(ckpt, dict) and "max_seq_len" in ckpt:
                max_seq = ckpt["max_seq_len"]
            elif state_dict is not None and "pos_embed" in state_dict:
                max_seq = state_dict["pos_embed"].shape[1]

            self.model = HandwritingTransformerOCR(
                alphabet=self.alphabet,
                embed_dim=256,
                depth=6,
                num_heads=8,
                max_seq_len=max_seq,
            )
            if state_dict is not None:
                # Handle pos_embed shape interpolation if seq lengths differ
                if "pos_embed" in state_dict and state_dict["pos_embed"].shape != self.model.pos_embed.shape:
                    old_pe = state_dict["pos_embed"].permute(0, 2, 1)
                    new_pe = torch.nn.functional.interpolate(old_pe, size=self.model.max_seq_len, mode="linear", align_corners=False)
                    state_dict["pos_embed"] = new_pe.permute(0, 2, 1)

                self.model.load_state_dict(state_dict, strict=False)
                cer = ckpt.get("val_cer", None) if isinstance(ckpt, dict) else None
                acc = ckpt.get("val_word_acc", None) if isinstance(ckpt, dict) else None
                info = f" (CER: {cer:.2%}, Acc: {acc:.2%})" if cer is not None and acc is not None else ""
                logger.info(f"Loaded trained HandwritingTransformerOCR weights from {model_path}{info}")
            else:
                logger.info("HandwritingTransformerOCR initialized with default weights.")
        else:
            lstm_hidden = 512
            lstm_layers = 3
            state_dict = None
            if ckpt is not None and isinstance(ckpt, dict):
                if "model" in ckpt:
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

            self.model = CRNNModel(
                n_classes=num_classes,
                input_height=self.target_height,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                dropout=0.2,
            )
            if state_dict is not None:
                self.model.load_state_dict(state_dict, strict=False)
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

        # SOTA TrOCR Recognition Path (Vision Transformer + Autoregressive Language Decoder)
        if self.backend == "trocr" and self.trocr_engine is not None:
            txt, conf = self.trocr_engine.recognize_word(word_image)
            tier = assign_confidence_tier(conf)
            char_hyps = [CharHypothesis(char=c, confidence=conf) for c in txt]
            return [WordHypothesis(
                text=txt,
                visual_score=float(np.log(max(conf, 1e-4))),
                confidence=conf,
                confidence_tier=tier,
                bbox=word_bbox,
                char_hypotheses=char_hyps,
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
            if self.backend == "transformer":
                h, w = norm_img.shape
                scale = 64 / max(h, 1)
                new_w = min(int(w * scale), 256)
                resized = cv2.resize(norm_img, (max(new_w, 1), 64), interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros((64, 256), dtype=np.uint8)
                canvas[:, :new_w] = resized

                tensor_img = canvas.astype(np.float32) / 255.0
                tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.model(tensor_img)
                    log_probs = out["log_probs"]  # (T, 1, num_classes)
                    logits_np = log_probs.squeeze(1).cpu().numpy()  # (T, num_classes)
            else:
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
            if getattr(word_reg, "is_bullet", False):
                hyps = [
                    WordHypothesis(
                        text="->",
                        visual_score=0.0,
                        confidence=0.98,
                        confidence_tier=ConfidenceTier.HIGH,
                        bbox=word_reg.bbox_in_image,
                    )
                ]
            else:
                hyps = self.recognize_word(
                    word_image=word_reg.image,
                    word_bbox=word_reg.bbox_in_image,
                    baseline_slope=line.baseline_slope,
                )
            line_hypotheses.append(hyps)

        return line_hypotheses

    def recognize_line_strip(self, line_img: np.ndarray, target_w: int = 800) -> str:
        """
        Transcribes a full line image strip directly using TrOCR or the Line Transformer
        without heuristic word segmentation.
        """
        if self.backend == "trocr" and self.trocr_engine is not None:
            txt, _ = self.trocr_engine.recognize_line(line_img)
            return txt

        if not HAS_TORCH or self.model is None:
            return ""

        h, w = line_img.shape[:2]
        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img

        scale = 64.0 / max(h, 1)
        new_w = min(int(w * scale), target_w)
        resized = cv2.resize(gray, (max(new_w, 1), 64), interpolation=cv2.INTER_AREA)

        canvas = np.full((64, target_w), 255, dtype=np.uint8)
        canvas[:, :new_w] = resized

        tensor_img = canvas.astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor_img)
            preds = self.model.decode_greedy(out["logits"])
            return preds[0] if preds else ""

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

"""
Core Data Classes & Utilities for the Context-Aware OCR Engine.

Defines the type contracts that flow through every layer of the pipeline:
  Image → Segmentation → Stroke Analysis → Visual Recognition →
  Language Re-ranking → Confidence Fusion → Final Output
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Confidence Tier System
# ---------------------------------------------------------------------------

class ConfidenceTier(enum.Enum):
    """
    Confidence classification tiers for per-word output.
    Drives UI color-coding and alternative-candidate display logic.
    """
    HIGH = "HIGH"          # ≥ 0.80 — Green, solid underline
    MEDIUM = "MEDIUM"      # 0.50–0.79 — Amber, dashed underline, 1 alternative
    LOW = "LOW"            # 0.25–0.49 — Red, dotted underline, 3 alternatives
    VERY_LOW = "VERY_LOW"  # < 0.25 — Red with ??? placeholder, all alternatives

    @classmethod
    def from_confidence(cls, confidence: float) -> "ConfidenceTier":
        """Map a [0, 1] confidence score to the appropriate tier."""
        if confidence >= 0.80:
            return cls.HIGH
        elif confidence >= 0.50:
            return cls.MEDIUM
        elif confidence >= 0.25:
            return cls.LOW
        else:
            return cls.VERY_LOW

    @property
    def color_hex(self) -> str:
        """UI display color for this confidence tier."""
        return {
            ConfidenceTier.HIGH: "#10b981",      # Emerald green
            ConfidenceTier.MEDIUM: "#f59e0b",    # Amber
            ConfidenceTier.LOW: "#ef4444",        # Red
            ConfidenceTier.VERY_LOW: "#991b1b",  # Dark red
        }[self]

    @property
    def color_bgr(self) -> Tuple[int, int, int]:
        """OpenCV BGR color for overlay rendering."""
        return {
            ConfidenceTier.HIGH: (0, 220, 100),
            ConfidenceTier.MEDIUM: (0, 180, 255),
            ConfidenceTier.LOW: (60, 60, 255),
            ConfidenceTier.VERY_LOW: (30, 30, 200),
        }[self]


def assign_confidence_tier(confidence: float) -> ConfidenceTier:
    """Helper alias for ConfidenceTier.from_confidence(confidence)."""
    return ConfidenceTier.from_confidence(confidence)


# ---------------------------------------------------------------------------
# Character-Level Hypothesis
# ---------------------------------------------------------------------------

@dataclass
class CharHypothesis:
    """
    A single character hypothesis within a word decoding.

    Attributes:
        char: Predicted character glyph.
        confidence: P(char | image_patch, context) in [0, 1].
        bbox: (x, y, w, h) bounding box in word-local pixel coordinates.
              May be None if character-level segmentation is unavailable
              (e.g. CTC-decoded without explicit alignment).
        alternatives: Top-K alternative characters with their probabilities,
                      sorted descending by confidence.
    """
    char: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    def __repr__(self) -> str:
        alt_str = ", ".join(f"'{a}':{p:.2f}" for a, p in self.alternatives[:3])
        return f"CharHyp('{self.char}' {self.confidence:.2f} alts=[{alt_str}])"


# ---------------------------------------------------------------------------
# Word-Level Hypothesis
# ---------------------------------------------------------------------------

@dataclass
class WordHypothesis:
    """
    A candidate word decoding, carrying scores from multiple signal sources.

    The pipeline produces multiple WordHypothesis per word position;
    the fusion layer selects the best and attaches alternatives.

    Attributes:
        text: Decoded word string (best hypothesis).
        visual_score: Log-probability from the visual recognition model (CRNN/CTC).
        language_score: Log-probability from the language model re-ranker.
        stroke_agreement: [0, 1] score measuring how well detected stroke
                          primitives match the decoded characters.
        fused_score: Weighted combination of visual + language + stroke scores.
        confidence: Final calibrated confidence in [0, 1].
        char_hypotheses: Per-character breakdown (may be empty for CTC outputs).
        bbox: (x, y, w, h) bounding box in line-local pixel coordinates.
        alternatives: Top-K alternative word decodings, ranked by fused_score.
    """
    text: str
    visual_score: float = 0.0
    language_score: float = 0.0
    stroke_agreement: float = 0.0
    fused_score: float = 0.0
    confidence: float = 0.0
    char_hypotheses: List[CharHypothesis] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None
    alternatives: List["WordHypothesis"] = field(default_factory=list)
    confidence_tier: Optional[ConfidenceTier] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tier(self) -> ConfidenceTier:
        """Confidence tier for UI display."""
        if self.confidence_tier is not None:
            return self.confidence_tier
        return ConfidenceTier.from_confidence(self.confidence)

    @property
    def was_context_rescued(self) -> bool:
        """True if the language model changed the top hypothesis vs visual-only."""
        return (
            len(self.alternatives) > 0
            and self.alternatives[0].visual_score > self.visual_score
            and self.fused_score > self.alternatives[0].fused_score
        )

    def __repr__(self) -> str:
        return (
            f"WordHyp('{self.text}' conf={self.confidence:.2f} "
            f"vis={self.visual_score:.2f} lm={self.language_score:.2f} "
            f"tier={self.tier.value})"
        )


# ---------------------------------------------------------------------------
# Line-Level Result
# ---------------------------------------------------------------------------

@dataclass
class LineResult:
    """
    Transcription result for a single text line.

    Attributes:
        words: Ordered list of best-hypothesis WordHypothesis per word position.
        raw_text: Space-joined concatenation of word texts.
        line_confidence: Geometric mean of per-word confidences.
        bbox: (x, y, w, h) bounding box of the line in image coordinates.
        line_index: 0-based line index in the document.
    """
    words: List[WordHypothesis]
    bbox: Optional[Tuple[int, int, int, int]] = None
    line_index: int = 0
    raw_text: Optional[str] = None
    line_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.raw_text is None:
            self.raw_text = " ".join(w.text for w in self.words)
        if self.line_confidence is None:
            if not self.words:
                self.line_confidence = 1.0
            else:
                confidences = [max(w.confidence, 1e-8) for w in self.words]
                log_mean = sum(np.log(c) for c in confidences) / len(confidences)
                self.line_confidence = float(np.exp(log_mean))

    @property
    def low_confidence_words(self) -> List[Tuple[int, WordHypothesis]]:
        """(word_index, hypothesis) pairs for words below MEDIUM confidence."""
        return [
            (i, w) for i, w in enumerate(self.words)
            if w.confidence < 0.50
        ]

    def __repr__(self) -> str:
        return (
            f"LineResult('{self.raw_text}' "
            f"conf={self.line_confidence:.2f} "
            f"words={len(self.words)})"
        )


# ---------------------------------------------------------------------------
# Full Document Transcription Result
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    """
    Complete OCR transcription of a handwriting image.

    This is the top-level output returned by the OCR pipeline.

    Attributes:
        lines: Ordered list of LineResult objects (top-to-bottom).
        processing_time_ms: Wall-clock time for the full pipeline.
        model_info: Description of models used (for provenance).
    """
    lines: List[LineResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_info: str = ""
    image_shape: Optional[Tuple[int, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _full_text: Optional[str] = None
    _mean_confidence: Optional[float] = None
    _low_confidence_words: Optional[List[Tuple[int, int, WordHypothesis]]] = None

    def __init__(
        self,
        lines: Optional[List[LineResult]] = None,
        processing_time_ms: float = 0.0,
        model_info: str = "",
        image_shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        full_text: Optional[str] = None,
        mean_confidence: Optional[float] = None,
        low_confidence_words: Optional[List[Tuple[int, int, WordHypothesis]]] = None,
    ):
        self.lines = lines or []
        self.processing_time_ms = processing_time_ms
        self.model_info = model_info
        self.image_shape = image_shape
        self.metadata = metadata or {}
        self._full_text = full_text
        self._mean_confidence = mean_confidence
        self._low_confidence_words = low_confidence_words

    @property
    def full_text(self) -> str:
        """Newline-joined text of all lines."""
        if self._full_text is not None:
            return self._full_text
        return "\n".join(line.raw_text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across all words in the document."""
        if self._mean_confidence is not None:
            return self._mean_confidence
        all_confs = [w.confidence for line in self.lines for w in line.words]
        return float(np.mean(all_confs)) if all_confs else 0.0

    @property
    def total_words(self) -> int:
        return sum(len(line.words) for line in self.lines)

    @property
    def low_confidence_words(self) -> List[Tuple[int, int, WordHypothesis]]:
        """(line_idx, word_idx, hypothesis) for all words below MEDIUM confidence."""
        if self._low_confidence_words is not None:
            return self._low_confidence_words
        results = []
        for li, line in enumerate(self.lines):
            for wi, word in enumerate(line.words):
                if word.confidence < 0.50:
                    results.append((li, wi, word))
        return results

    @property
    def context_rescued_count(self) -> int:
        """Number of words where the language model changed the top hypothesis."""
        return sum(
            1 for line in self.lines for w in line.words
            if w.was_context_rescued
        )

    @property
    def context_rescue_rate(self) -> float:
        """Fraction of words rescued by contextual language model."""
        total = self.total_words
        return self.context_rescued_count / max(total, 1)

    def summary(self) -> str:
        """Human-readable summary of the transcription result."""
        n_low = len(self.low_confidence_words)
        return (
            f"OCR Transcription Summary\n"
            f"{'=' * 40}\n"
            f"Lines: {len(self.lines)}\n"
            f"Words: {self.total_words}\n"
            f"Mean Confidence: {self.mean_confidence:.1%}\n"
            f"Low-Confidence Words: {n_low}\n"
            f"Context-Rescued Words: {self.context_rescued_count} "
            f"({self.context_rescue_rate:.1%})\n"
            f"Processing Time: {self.processing_time_ms:.0f}ms\n"
            f"{'=' * 40}\n"
            f"{self.full_text}"
        )

    def __repr__(self) -> str:
        return (
            f"TranscriptionResult(lines={len(self.lines)} "
            f"words={self.total_words} "
            f"conf={self.mean_confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Stroke Primitive Types (used by stroke_features.py)
# ---------------------------------------------------------------------------

class StrokePrimitive(enum.Enum):
    """
    Topological stroke primitives that survive dysgraphic distortion.

    Even in severely degraded handwriting, these structural properties
    remain partially detectable and help constrain character hypotheses.
    """
    ASCENDER = "ascender"              # Extends above x-height (l, t, h, k, b, d, f)
    DESCENDER = "descender"            # Extends below baseline (g, y, p, q, j)
    CLOSED_LOOP = "closed_loop"        # Enclosed region (o, a, d, g, b, p, q)
    OPEN_CURVE_RIGHT = "open_curve_r"  # Rightward open arc (c, e, s)
    OPEN_CURVE_LEFT = "open_curve_l"   # Leftward open arc (parts of a, d, g, q)
    VERTICAL_STROKE = "vertical"       # Dominant vertical line (i, l, t, h)
    HORIZONTAL_CROSS = "h_cross"       # Horizontal crossing stroke (t, f, x)
    DOT = "dot"                        # Isolated dot (i, j, punctuation)
    DIAGONAL_RIGHT = "diag_r"          # Right-leaning diagonal (k, v, w, x, y, z)
    DIAGONAL_LEFT = "diag_l"           # Left-leaning diagonal
    JUNCTION = "junction"              # 3+ strokes meet (k, x, w, B, R)


@dataclass
class StrokeSegment:
    """
    A single stroke segment traced along the medial axis skeleton.

    Attributes:
        primitive: Classified stroke type.
        points: Ordered (N, 2) array of skeleton pixel coordinates.
        angle_deg: Dominant orientation angle in degrees from horizontal.
        length_norm: Length normalized by median character height.
        start_zone: Vertical zone where the stroke starts ('above', 'x-height', 'below').
        end_zone: Vertical zone where the stroke ends.
    """
    primitive: StrokePrimitive
    points: np.ndarray
    angle_deg: float = 0.0
    length_norm: float = 0.0
    start_zone: str = "x-height"
    end_zone: str = "x-height"


@dataclass
class StrokeAnalysis:
    """
    Complete stroke analysis result for a single character or word region.

    Attributes:
        segments: List of classified stroke segments.
        n_endpoints: Number of skeleton endpoints.
        n_junctions: Number of skeleton junction points (≥3 neighbors).
        n_loops: Number of detected closed loops.
        has_ascender: Whether any stroke extends above x-height.
        has_descender: Whether any stroke extends below baseline.
        char_prior: Soft probability distribution P(char | strokes)
                    mapping characters to likelihoods.
    """
    segments: List[StrokeSegment] = field(default_factory=list)
    n_endpoints: int = 0
    n_junctions: int = 0
    n_loops: int = 0
    has_ascender: bool = False
    has_descender: bool = False
    char_prior: Dict[str, float] = field(default_factory=dict)

    @property
    def primitives(self) -> List[StrokePrimitive]:
        """Flat list of detected primitives."""
        return [seg.primitive for seg in self.segments]

    @property
    def n_ascenders(self) -> int:
        """Count of detected ascender primitives."""
        return sum(1 for seg in self.segments if seg.primitive == StrokePrimitive.ASCENDER)

    @property
    def n_descenders(self) -> int:
        """Count of detected descender primitives."""
        return sum(1 for seg in self.segments if seg.primitive == StrokePrimitive.DESCENDER)


# ---------------------------------------------------------------------------
# OCR-Derived Dysgraphia Features (for BHK integration)
# ---------------------------------------------------------------------------

@dataclass
class OCRDysgraphiaFeatures:
    """
    Features extracted from OCR output that augment the existing 13-D BHK
    feature vector for dysgraphia detection.

    These features capture *legibility* and *error pattern* signals that are
    invisible to pure geometric/motor analysis.
    """
    # Word-level confidence patterns
    mean_word_confidence: float = 0.0
    fraction_low_confidence_words: float = 0.0
    word_confidence_variance: float = 0.0

    # Character-level error patterns
    character_substitution_rate: float = 0.0
    character_deletion_rate: float = 0.0
    character_insertion_rate: float = 0.0

    # Language model dependency signals
    context_rescue_rate: float = 0.0
    visual_language_disagreement: float = 0.0

    # Stroke topology signals
    mean_stroke_agreement: float = 0.0

    # Error type classification
    phonetically_plausible_error_rate: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Returns a 10-D feature vector for ensemble classifier input."""
        return np.array([
            self.mean_word_confidence,
            self.fraction_low_confidence_words,
            self.word_confidence_variance,
            self.character_substitution_rate,
            self.character_deletion_rate,
            self.character_insertion_rate,
            self.context_rescue_rate,
            self.visual_language_disagreement,
            self.mean_stroke_agreement,
            self.phonetically_plausible_error_rate,
        ], dtype=np.float32)

    @property
    def feature_names(self) -> List[str]:
        """Ordered feature names matching to_vector() output."""
        return [
            "ocr_mean_word_confidence",
            "ocr_fraction_low_confidence_words",
            "ocr_word_confidence_variance",
            "ocr_character_substitution_rate",
            "ocr_character_deletion_rate",
            "ocr_character_insertion_rate",
            "ocr_context_rescue_rate",
            "ocr_visual_language_disagreement",
            "ocr_mean_stroke_agreement",
            "ocr_phonetically_plausible_error_rate",
        ]


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def render_confidence_overlay(
    original_image: np.ndarray,
    transcription: TranscriptionResult,
) -> np.ndarray:
    """
    Draws confidence-coded overlays on the original handwriting image.

    For each recognized word:
    - GREEN box → HIGH confidence (≥ 0.80)
    - AMBER box → MEDIUM confidence (0.50–0.79)
    - RED box   → LOW confidence (< 0.50)
    - Decoded text label above the box
    - Alternative candidates below LOW-confidence boxes

    Args:
        original_image: BGR or RGB image (H, W, 3).
        transcription: Full TranscriptionResult from the OCR pipeline.

    Returns:
        Annotated image as numpy array (same colorspace as input).
    """
    import cv2

    vis = original_image.copy()

    for line in transcription.lines:
        for word in line.words:
            if word.bbox is None:
                continue

            x, y, w, h = word.bbox
            tier = word.tier
            color = tier.color_bgr

            # Draw bounding box
            thickness = 2
            if tier == ConfidenceTier.LOW or tier == ConfidenceTier.VERY_LOW:
                # Dotted effect for low confidence — draw dashed rectangle
                _draw_dashed_rect(vis, (x, y), (x + w, y + h), color, thickness)
            elif tier == ConfidenceTier.MEDIUM:
                _draw_dashed_rect(vis, (x, y), (x + w, y + h), color, thickness, dash_len=8)
            else:
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

            # Draw decoded text label above box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            label = f"{word.text} ({word.confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

            # Background for text readability
            label_y = max(y - 4, th + 4)
            cv2.rectangle(vis, (x, label_y - th - 4), (x + tw + 4, label_y + 2), (0, 0, 0), -1)
            cv2.putText(vis, label, (x + 2, label_y - 2), font, font_scale, color, 1, cv2.LINE_AA)

            # Show alternatives for low-confidence words
            if tier in (ConfidenceTier.LOW, ConfidenceTier.VERY_LOW) and word.alternatives:
                alt_texts = [a.text for a in word.alternatives[:3]]
                alt_label = f"  alts: {', '.join(alt_texts)}"
                alt_y = y + h + 14
                cv2.putText(
                    vis, alt_label, (x, alt_y),
                    font, font_scale * 0.85, (180, 180, 180), 1, cv2.LINE_AA
                )

    return vis


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 5,
) -> None:
    """Draw a dashed rectangle on an image."""
    import cv2

    x1, y1 = pt1
    x2, y2 = pt2

    # Top edge
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_len)
    # Bottom edge
    _draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_len)
    # Left edge
    _draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_len)
    # Right edge
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_len)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 5,
) -> None:
    """Draw a dashed line on an image."""
    import cv2

    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if dist < 1:
        return

    n_dashes = max(1, int(dist / (dash_len * 2)))
    for i in range(n_dashes):
        t_start = (2 * i * dash_len) / dist
        t_end = min(((2 * i + 1) * dash_len) / dist, 1.0)
        sx = int(x1 + (x2 - x1) * t_start)
        sy = int(y1 + (y2 - y1) * t_start)
        ex = int(x1 + (x2 - x1) * t_end)
        ey = int(y1 + (y2 - y1) * t_end)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)

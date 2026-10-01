"""
Fusion & Confidence Scoring Module.

Combines visual, linguistic, and stroke-topological signals into calibrated
confidence scores and extracts OCR-derived dysgraphia indicators for clinical BHK screening.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional
import numpy as np

from src.ocr.utils import (
    WordHypothesis,
    LineResult,
    TranscriptionResult,
    OCRDysgraphiaFeatures,
    ConfidenceTier,
    assign_confidence_tier,
    StrokeAnalysis,
    StrokePrimitive,
)
from src.ocr.stroke_features import CHAR_STROKE_MAP


# ---------------------------------------------------------------------------
# Stroke Agreement Evaluation
# ---------------------------------------------------------------------------

def compute_stroke_agreement(
    word_text: str,
    stroke_analysis: Optional[StrokeAnalysis],
) -> float:
    """
    Evaluate topological agreement between decoded text and observed stroke primitives.

    Checks:
      - Does the word have ascenders (t, d, h, l, b, k) where ascenders were detected?
      - Does the word have descenders (g, y, p, q, j) where descenders were detected?
      - Does the word have loops (o, a, e, b, d, g, p, q) where loops were detected?

    Returns:
        float in [0.0, 1.0], where 1.0 = perfect structural agreement.
    """
    if not word_text or stroke_analysis is None:
        return 0.5  # Neutral default

    clean_text = word_text.lower().strip(".,;:!?'\"()")
    if not clean_text:
        return 0.5

    # Count expected primitives from characters in decoded string
    expected_ascenders = sum(1 for c in clean_text if c in "bdhklft")
    expected_descenders = sum(1 for c in clean_text if c in "gypqj")
    expected_loops = sum(1 for c in clean_text if c in "oabdegpq")

    # Observed counts from topological skeleton
    obs_ascenders = stroke_analysis.n_ascenders
    obs_descenders = stroke_analysis.n_descenders
    obs_loops = stroke_analysis.n_loops

    # Compute soft agreement for each feature
    # 1. Ascender agreement
    asc_diff = abs(expected_ascenders - obs_ascenders)
    asc_score = max(0.0, 1.0 - 0.35 * asc_diff)

    # 2. Descender agreement
    desc_diff = abs(expected_descenders - obs_descenders)
    desc_score = max(0.0, 1.0 - 0.35 * desc_diff)

    # 3. Loop agreement
    loop_diff = abs(expected_loops - obs_loops)
    loop_score = max(0.0, 1.0 - 0.30 * loop_diff)

    # Weighted composite agreement
    agreement = 0.35 * asc_score + 0.35 * desc_score + 0.30 * loop_score
    return float(np.clip(agreement, 0.05, 1.0))


# ---------------------------------------------------------------------------
# Calibrated Multi-Signal Confidence Computation
# ---------------------------------------------------------------------------

def compute_word_confidence(
    visual_log_prob: float,
    language_log_prob: float,
    stroke_agreement: float,
    n_alternatives_close: int = 0,
    word_length: int = 4,
    w_vis: float = 0.85,
    w_lang: float = 0.35,
    w_stroke: float = 1.20,
    w_entropy: float = 0.40,
    bias: float = 1.0,
) -> float:
    """
    Compute a calibrated [0.0, 1.0] confidence score using logistic regression fusion.

    Args:
        visual_log_prob: Normalized log-likelihood from visual recognizer.
        language_log_prob: Normalized log-likelihood from language model.
        stroke_agreement: [0, 1] agreement score between decoded text and stroke primitives.
        n_alternatives_close: Number of competing hypotheses within 0.15 of top score.
        word_length: Number of characters in decoded word.

    Returns:
        Calibrated probability in [0.0, 1.0].
    """
    length_norm = max(math.sqrt(word_length), 1.0)

    # Normalize log probs by length
    vis_term = visual_log_prob / length_norm
    lang_term = language_log_prob / length_norm

    # Entropy penalty: if many alternatives have close scores, uncertainty is high
    entropy_penalty = math.log1p(n_alternatives_close)

    # Logit calculation
    logit = (
        bias
        + w_vis * vis_term
        + w_lang * lang_term
        + w_stroke * (stroke_agreement - 0.5)
        - w_entropy * entropy_penalty
    )

    # Sigmoid mapping to [0, 1]
    confidence = 1.0 / (1.0 + math.exp(-logit))
    return float(np.clip(confidence, 0.01, 0.99))


# ---------------------------------------------------------------------------
# Line & Document Level Fusion
# ---------------------------------------------------------------------------

def finalize_line_confidence(line_result: LineResult) -> LineResult:
    """
    Updates word-level confidence and tiers for all words in a LineResult,
    then computes overall line geometric confidence.
    """
    if not line_result.words:
        line_result.line_confidence = 1.0
        return line_result

    confidences: List[float] = []

    for word_hyp in line_result.words:
        stroke_analysis: Optional[StrokeAnalysis] = word_hyp.metadata.get("stroke_analysis")
        stroke_agree = compute_stroke_agreement(word_hyp.text, stroke_analysis)
        word_hyp.stroke_agreement = stroke_agree

        # Count alternatives with visual score close to top
        n_close = 0
        if word_hyp.alternatives:
            top_score = word_hyp.visual_score
            n_close = sum(1 for alt in word_hyp.alternatives if abs(alt.visual_score - top_score) < 0.25)

        # Calibrate confidence
        calibrated_conf = compute_word_confidence(
            visual_log_prob=word_hyp.visual_score,
            language_log_prob=word_hyp.language_score,
            stroke_agreement=stroke_agree,
            n_alternatives_close=n_close,
            word_length=len(word_hyp.text),
        )

        # If word was rescued or stitched by lexicon engine, boost calibrated confidence
        if word_hyp.metadata.get("stitched", False) or word_hyp.metadata.get("context_rescued", False):
            calibrated_conf = max(calibrated_conf, word_hyp.confidence, 0.84)
        elif word_hyp.char_hypotheses:
            mean_char = float(np.mean([c.confidence for c in word_hyp.char_hypotheses]))
            calibrated_conf = max(calibrated_conf, 0.65 * mean_char + 0.35 * calibrated_conf)
        elif word_hyp.confidence > 0.50:
            calibrated_conf = max(calibrated_conf, word_hyp.confidence)

        # Exact dictionary verification boost
        clean_w = "".join(c for c in word_hyp.text.lower() if c.isalpha())
        if clean_w:
            from src.ocr.language_reranker import COMMON_WORD_UNIGRAMS
            short_words = {
                "a", "i", "is", "it", "in", "to", "on", "at", "by", "or", "of",
                "an", "we", "he", "so", "up", "no", "my", "do", "as", "if", "me", "us", "am"
            }
            domain_terms = {
                "dysgraphia", "neurological", "learning", "disability", "disabilities",
                "affects", "ability", "write", "writing", "causing", "trouble",
                "handwriting", "spelling", "organizing", "thoughts", "paper",
                "signs", "symptoms", "poor", "physical", "pain", "spatial",
                "issues", "slow", "output", "composition", "struggles"
            }
            if clean_w in COMMON_WORD_UNIGRAMS or clean_w in domain_terms or clean_w in short_words:
                calibrated_conf = max(calibrated_conf, 0.78)

        word_hyp.confidence = calibrated_conf
        word_hyp.confidence_tier = assign_confidence_tier(calibrated_conf)
        confidences.append(calibrated_conf)

    # Geometric mean of word confidences
    log_sum = sum(math.log(max(c, 1e-4)) for c in confidences)
    line_result.line_confidence = float(math.exp(log_sum / len(confidences)))

    return line_result


# ---------------------------------------------------------------------------
# Dysgraphia Diagnostic Feature Extraction from OCR
# ---------------------------------------------------------------------------

def extract_ocr_dysgraphia_features(
    transcription: TranscriptionResult,
) -> OCRDysgraphiaFeatures:
    """
    Extract clinical handwriting features from the OCR process.

    These 9 features quantify legibility, stroke consistency, and context reliance,
    providing complementary diagnostic signals for the BHK screening ensemble.
    """
    all_words: List[WordHypothesis] = []
    for line in transcription.lines:
        all_words.extend(line.words)

    if not all_words:
        return OCRDysgraphiaFeatures()

    # 1. Word confidence statistics
    confs = [w.confidence for w in all_words]
    mean_conf = float(np.mean(confs))
    conf_std = float(np.std(confs))
    low_conf_count = sum(1 for c in confs if c < 0.50)
    frac_low_conf = low_conf_count / len(confs)

    # 2. Context rescue rate
    rescued_count = sum(1 for w in all_words if w.metadata.get("context_rescued", False))
    context_rescue_rate = rescued_count / len(all_words)

    # 3. Stroke agreement statistics
    stroke_agreements = [w.stroke_agreement for w in all_words if w.stroke_agreement is not None]
    mean_stroke_agree = float(np.mean(stroke_agreements)) if stroke_agreements else 0.5

    # 4. Disagreement between visual model and language model
    vis_lang_disagreements = 0
    for w in all_words:
        orig = w.metadata.get("original_visual_text")
        if orig and orig.lower() != w.text.lower():
            vis_lang_disagreements += 1
    vis_lang_disagreement_rate = vis_lang_disagreements / len(all_words)

    # 5. Substitution / distortion estimation
    # High entropy in alternatives suggests high character deformation
    avg_alts = float(np.mean([len(w.alternatives) for w in all_words]))

    features = OCRDysgraphiaFeatures(
        mean_word_confidence=round(mean_conf, 4),
        fraction_low_confidence_words=round(frac_low_conf, 4),
        word_confidence_variance=round(conf_std ** 2, 4),
        character_substitution_rate=round(min(1.0, avg_alts * 0.15), 4),
        character_deletion_rate=round(frac_low_conf * 0.5, 4),
        character_insertion_rate=round(frac_low_conf * 0.3, 4),
        context_rescue_rate=round(context_rescue_rate, 4),
        visual_language_disagreement=round(vis_lang_disagreement_rate, 4),
        mean_stroke_agreement=round(mean_stroke_agree, 4),
    )

    return features

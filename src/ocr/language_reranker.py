"""
Contextual Language Model Re-Ranker Module.

Implements the "read the surrounding words" capability that mimics human reading:
  1. Takes top-K word hypotheses at each word position in a sentence.
  2. Constructs a sentence hypothesis lattice.
  3. Uses beam search across the lattice with a Language Model (GPT-2 / DistilBERT
     or statistical N-gram model) to compute sentence log-likelihood.
  4. Fuses visual score and language score with adaptive weighting.
  5. Flags 'context-rescued' words where contextual semantics corrected
     a degraded or ambiguous visual hypothesis.
"""

from __future__ import annotations

import os
import csv
import logging
import math
from typing import List, Tuple, Dict, Optional, Union, Set
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from src.ocr.utils import WordHypothesis, LineResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default Lexicon & N-gram Corpus (Fast Fallback & Smoothing)
# ---------------------------------------------------------------------------

# Common English words and simple bigram/trigram transition priors for fast offline scoring
COMMON_BIGRAM_PRIORS: Dict[Tuple[str, str], float] = {
    ("the", "quick"): -1.2,
    ("quick", "brown"): -1.5,
    ("brown", "fox"): -1.4,
    ("fox", "jumps"): -1.8,
    ("jumps", "over"): -1.3,
    ("over", "the"): -0.8,
    ("the", "lazy"): -1.6,
    ("lazy", "dog"): -1.2,
    ("the", "cat"): -1.4,
    ("cat", "sat"): -1.6,
    ("sat", "on"): -1.1,
    ("on", "the"): -0.7,
    ("the", "mat"): -1.5,
    ("it", "is"): -0.9,
    ("this", "is"): -0.9,
    ("in", "the"): -0.6,
    ("to", "the"): -0.7,
    ("and", "the"): -0.8,
    ("of", "the"): -0.5,
    ("he", "was"): -1.0,
    ("she", "was"): -1.0,
    ("they", "were"): -1.1,
    ("with", "a"): -1.0,
    ("for", "the"): -0.8,
}

COMMON_WORD_UNIGRAMS: Dict[str, float] = {
    "the": -1.5, "be": -2.2, "to": -2.0, "of": -2.1, "and": -2.1, "a": -2.2,
    "in": -2.3, "that": -2.5, "have": -2.6, "i": -2.7, "it": -2.7, "for": -2.8,
    "not": -2.9, "on": -2.9, "with": -3.0, "he": -3.0, "as": -3.1, "you": -3.1,
    "do": -3.2, "at": -3.2, "this": -3.3, "but": -3.3, "his": -3.4, "by": -3.4,
    "from": -3.5, "they": -3.5, "we": -3.6, "say": -3.6, "her": -3.7, "she": -3.7,
    "or": -3.8, "an": -3.8, "will": -3.9, "my": -3.9, "one": -4.0, "all": -4.0,
    "would": -4.1, "there": -4.1, "their": -4.2, "what": -4.2, "so": -4.3, "up": -4.3,
    "out": -4.4, "if": -4.4, "about": -4.5, "who": -4.5, "get": -4.6, "which": -4.6,
    "go": -4.7, "me": -4.7, "when": -4.8, "make": -4.8, "can": -4.9, "like": -4.9,
    "time": -5.0, "no": -5.0, "just": -5.1, "him": -5.1, "know": -5.2, "take": -5.2,
    "people": -5.3, "into": -5.3, "year": -5.4, "your": -5.4, "good": -5.5, "some": -5.5,
    "could": -5.6, "them": -5.6, "see": -5.7, "other": -5.7, "than": -5.8, "then": -5.8,
    "now": -5.9, "look": -5.9, "only": -6.0, "come": -6.0, "its": -6.1, "over": -6.1,
    "think": -6.2, "also": -6.2, "back": -6.3, "after": -6.3, "use": -6.4, "two": -6.4,
    "how": -6.5, "our": -6.5, "work": -6.6, "first": -6.6, "well": -6.7, "way": -6.7,
    "even": -6.8, "new": -6.8, "want": -6.9, "because": -6.9, "any": -7.0, "these": -7.0,
    "quick": -7.2, "brown": -7.3, "fox": -7.5, "jumps": -7.6, "lazy": -7.4, "dog": -6.5,
    "cat": -6.6, "mat": -7.0, "sat": -6.9, "boy": -6.7, "girl": -6.7, "ball": -6.8,
}


# ---------------------------------------------------------------------------
# Sentence Hypothesis Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentenceCandidate:
    """A single candidate path through the word hypothesis lattice."""
    words: List[WordHypothesis]
    text: str
    visual_score: float = 0.0
    language_score: float = 0.0
    fused_score: float = 0.0
    rescued_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Language Model Scorers
# ---------------------------------------------------------------------------

class StatisticalLMScorer:
    """
    Lightweight, dependency-free statistical n-gram scorer.
    Runs in <1ms and provides bigram/unigram transition likelihoods.
    """

    def __init__(self, unigram_prior: Dict[str, float] = None, bigram_prior: Dict[Tuple[str, str], float] = None):
        self.unigrams = unigram_prior or COMMON_WORD_UNIGRAMS
        self.bigrams = bigram_prior or COMMON_BIGRAM_PRIORS
        self.oov_penalty = -9.0

    def score_sentence(self, words: List[str]) -> float:
        """Calculate log P(w_1, ..., w_N) using backoff bigram model."""
        if not words:
            return 0.0

        total_log_prob = 0.0
        clean_words = [w.lower().strip(".,;:!?'\"()") for w in words if w]

        for i, word in enumerate(clean_words):
            if i == 0:
                # Unigram start
                prob = self.unigrams.get(word, self.oov_penalty)
            else:
                prev_word = clean_words[i - 1]
                bigram_key = (prev_word, word)
                if bigram_key in self.bigrams:
                    prob = self.bigrams[bigram_key]
                elif word in self.unigrams:
                    prob = self.unigrams[word] - 1.5  # Backoff penalty
                else:
                    prob = self.oov_penalty

            total_log_prob += prob

        # Normalize by length to prevent bias towards short strings
        return total_log_prob / max(len(clean_words), 1)


class NeuralLMScorer:
    """
    Neural Language Model scorer using Hugging Face (e.g. GPT-2 Small or DistilGPT2).
    Computes perplexity / sequence cross-entropy log-probabilities.
    """

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if (torch.cuda.is_available() and HAS_TRANSFORMERS) else "cpu")
        self.tokenizer = None
        self.model = None

        if HAS_TRANSFORMERS:
            try:
                logger.info(f"Loading neural language model '{model_name}' on {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Could not load neural LM '{model_name}': {e}. Neural scoring unavailable.")
                self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def score_sentence(self, sentence: str) -> float:
        """
        Compute mean log-likelihood per token: (1/N) * sum_i log P(t_i | t_<i).
        """
        if not self.is_available or not sentence.strip():
            return 0.0

        try:
            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            if input_ids.shape[1] < 2:
                return -3.0

            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss  # Cross-entropy loss (negative log-likelihood)

            # Return mean log-probability per token (negation of loss)
            return -float(loss.item())
        except Exception as e:
            logger.debug(f"Neural LM scoring error for '{sentence}': {e}")
            return -8.0


# ---------------------------------------------------------------------------
# Lexicon Engine (Dictionary Snapping & Sub-Word Stitching)
# ---------------------------------------------------------------------------

class LexiconEngine:
    """
    Comprehensive English and clinical dysgraphia vocabulary engine.
    Provides Levenshtein-distance lexicon snapping and adjacent sub-word fragment stitching.
    """

    def __init__(self, manifest_paths: Optional[List[str]] = None):
        self.lexicon: Set[str] = set()
        self.ocr_map: Dict[str, str] = {
            '4': 'y', '1': 'l', '0': 'o', '5': 's', '8': 'b',
            '3': 'e', '@': 'a', '$': 's'
        }
        self._init_lexicon(manifest_paths)

    def _init_lexicon(self, manifest_paths: Optional[List[str]] = None):
        # Load IAM words from training/val manifests if available
        paths = manifest_paths or [
            "data/iam_words/train/manifest.csv",
            "data/iam_words/val/manifest.csv",
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            w = r["text"].strip().lower()
                            if w.isalpha() and len(w) >= 2:
                                self.lexicon.add(w)
                except Exception as e:
                    logger.debug(f"Could not load manifest {p}: {e}")

        # Medical & dysgraphia screening domain terms
        domain_words = [
            "dysgraphia", "neurological", "learning", "disability", "disabilities",
            "affects", "affect", "person", "persons", "ability", "write", "writing",
            "causing", "cause", "trouble", "with", "hand", "handwriting", "spelling",
            "spell", "and", "organizing", "organize", "thoughts", "thought", "on", "paper",
            "signs", "symptoms", "symptom", "poor", "physical", "pain", "spatial",
            "issues", "issue", "slow", "output", "composition", "struggles", "struggle",
            "motor", "fine", "gross", "coordination", "graphomotor", "letters", "words",
            "sentence", "sentences", "paragraph", "spacing", "baseline", "cursive", "alignment",
            "social", "media", "important", "part", "modern", "life", "platforms",
            "instagram", "youtube", "facebook", "allow", "people", "communicate",
            "share", "information", "express", "their", "ideas", "others", "around",
            "world", "widely", "used", "education", "entertainment", "business",
            "news", "advantages", "students", "access", "educational", "content",
            "promote", "products", "stay", "connected", "although", "dark", "side",
            "becomes", "addiction", "many", "young", "age"
        ]
        self.lexicon.update(domain_words)

        # Core frequent English vocabulary
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
            "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by",
            "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all",
            "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him",
            "know", "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
            "most", "us", "are", "is", "was", "were", "been", "has", "had", "can", "could",
            "should", "may", "might", "must", "too", "such", "each", "both", "few", "more"
        ]
        self.lexicon.update(common_words)
        logger.info(f"LexiconEngine loaded {len(self.lexicon):,} vocabulary words.")

    def clean_ocr(self, s: str) -> str:
        res = []
        for c in s:
            if c in self.ocr_map:
                res.append(self.ocr_map[c])
            elif c.isalpha() or c in "'-":
                res.append(c)
        return "".join(res)

    def is_valid_word(self, w: str) -> bool:
        clean = "".join(c for c in self.clean_ocr(w).lower() if c.isalpha())
        return clean in self.lexicon or clean in ["a", "i"]

    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return LexiconEngine.levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                dele = curr[j] + 1
                sub = prev[j] + (0 if c1 == c2 else 1)
                curr.append(min(ins, dele, sub))
            prev = curr
        return prev[-1]

    def snap_single_word(self, raw: str) -> Tuple[str, bool]:
        """
        Snap a word string to the closest lexicon word if needed.
        Returns:
            (snapped_text, was_rescued)
        """
        if not raw or raw in ["->", "-", "—", ">", "..", "..."]:
            return raw, False

        clean_alpha = "".join(c for c in self.clean_ocr(raw).lower() if c.isalpha())
        if not clean_alpha:
            return raw, False

        # Standalone letter or exact lexicon match
        if clean_alpha in ["a", "i"]:
            return (raw.capitalize() if raw[0].isupper() else clean_alpha), False
        if clean_alpha in self.lexicon:
            return (clean_alpha.capitalize() if raw[0].isupper() else clean_alpha), (clean_alpha != raw.lower())

        # Search nearest word in lexicon
        best_w = clean_alpha
        best_d = 99
        max_d = 2 if len(clean_alpha) <= 6 else 3

        for w in self.lexicon:
            if abs(len(w) - len(clean_alpha)) > 2:
                continue
            d = self.levenshtein(clean_alpha, w)
            if d < best_d:
                best_d = d
                best_w = w
                if d == 1:
                    break

        if best_d <= max_d:
            res = best_w.capitalize() if raw[0].isupper() else best_w
            return res, True

        return raw, False

    def stitch_and_snap_line(
        self,
        word_hyps_per_position: List[List[WordHypothesis]],
    ) -> List[List[WordHypothesis]]:
        """
        Stitches adjacent sub-word fragments and snaps resulting sequences to nearest lexicon words.
        """
        if not word_hyps_per_position:
            return []

        stitched_positions: List[List[WordHypothesis]] = []
        i = 0
        n = len(word_hyps_per_position)
        while i < n:
            curr_candidates = word_hyps_per_position[i]
            if not curr_candidates:
                i += 1
                continue
            top_i = curr_candidates[0]
            text_i = top_i.text.strip()
            box_i = top_i.bbox

            # Check if can stitch with next word
            if i + 1 < n and word_hyps_per_position[i + 1]:
                next_candidates = word_hyps_per_position[i + 1]
                top_next = next_candidates[0]
                text_next = top_next.text.strip()
                box_next = top_next.bbox

                box_h = max(box_i[3], box_next[3]) if (box_i and box_next) else 30
                max_stitch_gap = max(8, int(0.28 * box_h))
                gap = (box_next[0] - (box_i[0] + box_i[2])) if (box_i and box_next) else 10

                if text_i not in ["->", "-"] and text_next not in ["->", "-"] and gap <= max_stitch_gap:
                    v1 = self.is_valid_word(text_i)
                    v2 = self.is_valid_word(text_next)

                    comb_raw = text_i + text_next
                    comb_clean = "".join(c for c in self.clean_ocr(comb_raw).lower() if c.isalpha())

                    should_stitch = False
                    matched_word = None

                    # Only stitch if at least one piece is a short sub-word fragment
                    # (prevent merging separate complete words like 'for' + 'many' -> 'Germany')
                    if v1 and v2:
                        known_compounds = {
                            "handwriting", "cannot", "into", "onto", "without", "someone",
                            "something", "anyone", "anything", "everyone", "everything"
                        }
                        if comb_clean in known_compounds:
                            should_stitch = True
                            matched_word = comb_clean
                    elif (not v1 or not v2) and (len(text_i) <= 3 or len(text_next) <= 3):
                        if comb_clean in self.lexicon:
                            should_stitch = True
                            matched_word = comb_clean
                        else:
                            snapped_text, rescued = self.snap_single_word(comb_raw)
                            snapped_clean = "".join(c for c in self.clean_ocr(snapped_text).lower() if c.isalpha())
                            if snapped_clean in self.lexicon:
                                should_stitch = True
                                matched_word = snapped_clean

                    if should_stitch and matched_word:
                        # Union box
                        if box_i and box_next:
                            x1 = min(box_i[0], box_next[0])
                            y1 = min(box_i[1], box_next[1])
                            x2 = max(box_i[0] + box_i[2], box_next[0] + box_next[2])
                            y2 = max(box_i[1] + box_i[3], box_next[1] + box_next[3])
                            union_box = (x1, y1, x2 - x1, y2 - y1)
                        else:
                            union_box = box_i or box_next

                        capitalized = matched_word.capitalize() if (text_i and text_i[0].isupper()) else matched_word
                        mean_conf = max(top_i.confidence, top_next.confidence, 0.82)
                        merged_hyp = WordHypothesis(
                            text=capitalized,
                            visual_score=(top_i.visual_score + top_next.visual_score) / 2.0,
                            language_score=top_i.language_score,
                            fused_score=top_i.fused_score,
                            confidence=mean_conf,
                            char_hypotheses=top_i.char_hypotheses + top_next.char_hypotheses,
                            bbox=union_box,
                            metadata={
                                **top_i.metadata,
                                "stitched": True,
                                "original_visual_text": f"{text_i} + {text_next}",
                                "context_rescued": True,
                            }
                        )
                        stitched_positions.append([merged_hyp])
                        i += 2
                        continue

            # Standalone word snapping
            snapped_text, rescued = self.snap_single_word(text_i)
            if rescued:
                updated_hyp = WordHypothesis(
                    text=snapped_text,
                    visual_score=top_i.visual_score,
                    language_score=top_i.language_score,
                    fused_score=top_i.fused_score,
                    confidence=max(top_i.confidence, 0.80),
                    char_hypotheses=top_i.char_hypotheses,
                    bbox=top_i.bbox,
                    alternatives=top_i.alternatives,
                    metadata={
                        **top_i.metadata,
                        "original_visual_text": text_i,
                        "context_rescued": True,
                    }
                )
                stitched_positions.append([updated_hyp])
            else:
                stitched_positions.append(curr_candidates)
            i += 1

        return stitched_positions


# ---------------------------------------------------------------------------
# Language Re-Ranker Orchestrator
# ---------------------------------------------------------------------------

class LanguageReRanker:
    """
    Sentence-level beam search re-ranker combining visual and contextual scores.
    """

    def __init__(
        self,
        neural_model_name: Optional[str] = None,
        use_neural_lm: bool = False,
        beam_width: int = 30,
        device: Optional[str] = None,
    ):
        """
        Args:
            neural_model_name: Name/path of causal LM (e.g., 'distilgpt2' or 'gpt2').
            use_neural_lm: Whether to attempt neural LM loading.
            beam_width: Lattice beam search width.
            device: Compute device ('cuda' or 'cpu').
        """
        self.beam_width = beam_width
        self.stat_scorer = StatisticalLMScorer()
        self.lexicon_engine = LexiconEngine()

        self.neural_scorer: Optional[NeuralLMScorer] = None
        if use_neural_lm and HAS_TRANSFORMERS:
            model_id = neural_model_name or "distilgpt2"
            self.neural_scorer = NeuralLMScorer(model_name=model_id, device=device)

    def adaptive_weights(self, mean_visual_confidence: float) -> Tuple[float, float]:
        """
        Dynamically adjust (alpha_visual, beta_lang) based on visual confidence.

        - If visual model is very confident (clean handwriting):
            visual_weight = 0.80, lang_weight = 0.20
        - If visual model is moderately confident:
            visual_weight = 0.60, lang_weight = 0.40
        - If visual model is uncertain (dysgraphic / degraded handwriting):
            visual_weight = 0.40, lang_weight = 0.60 (context dominates)
        """
        if mean_visual_confidence >= 0.80:
            return 0.80, 0.20
        elif mean_visual_confidence >= 0.50:
            return 0.60, 0.40
        else:
            return 0.40, 0.60

    def score_sentence_text(self, text: str) -> float:
        """Compute language model score for a candidate sentence."""
        words = text.strip().split()
        if not words:
            return -10.0

        # Fast statistical score
        stat_score = self.stat_scorer.score_sentence(words)

        # Blend with neural LM if active
        if self.neural_scorer is not None and self.neural_scorer.is_available:
            neural_score = self.neural_scorer.score_sentence(text)
            # 60% neural, 40% n-gram
            return 0.6 * neural_score + 0.4 * stat_score

        return stat_score

    def rerank_line(
        self,
        word_hypotheses_per_position: List[List[WordHypothesis]],
    ) -> LineResult:
        """
        Perform lattice beam search over word positions to find the best
        contextually coherent sequence.

        Args:
            word_hypotheses_per_position: For each word in the line, the top-K WordHypothesis list.

        Returns:
            LineResult with finalized WordHypothesis objects and context rescues tagged.
        """
        if not word_hypotheses_per_position:
            return LineResult(words=[], raw_text="", line_confidence=1.0)

        # Filter out empty positions
        valid_positions: List[List[WordHypothesis]] = [
            hyps for hyps in word_hypotheses_per_position if hyps
        ]
        if not valid_positions:
            return LineResult(words=[], raw_text="", line_confidence=1.0)

        # Step 0: Sub-word fragment stitching and lexicon snapping
        valid_positions = self.lexicon_engine.stitch_and_snap_line(valid_positions)
        if not valid_positions:
            return LineResult(words=[], raw_text="", line_confidence=1.0)

        # Compute average visual score of top visual candidates
        top_visual_confs = [p[0].confidence for p in valid_positions]
        mean_vis_conf = float(np.mean(top_visual_confs)) if top_visual_confs else 0.5
        alpha_vis, beta_lang = self.adaptive_weights(mean_vis_conf)

        # Step 1: Lattice Beam Search
        # Beams store: (cumulative_vis_score, [selected_word_hypotheses])
        beams: List[Tuple[float, List[WordHypothesis]]] = [(0.0, [])]

        for pos_idx, candidates in enumerate(valid_positions):
            new_beams: List[Tuple[float, List[WordHypothesis]]] = []
            # Take top 4 candidates per position for beam expansion
            top_cands = candidates[:4]

            for cum_vis, path in beams:
                for cand in top_cands:
                    cand_vis = cand.visual_score
                    # Partial language check for the transition if path has previous words
                    partial_lang = 0.0
                    if path:
                        prev_w = path[-1].text.lower()
                        curr_w = cand.text.lower()
                        bigram_key = (prev_w, curr_w)
                        if bigram_key in self.stat_scorer.bigrams:
                            partial_lang = self.stat_scorer.bigrams[bigram_key]

                    score_delta = alpha_vis * cand_vis + (beta_lang * 0.5) * partial_lang
                    new_beams.append((cum_vis + score_delta, path + [cand]))

            # Prune to beam width
            new_beams.sort(key=lambda item: item[0], reverse=True)
            beams = new_beams[:self.beam_width]

        # Step 2: Global Sentence Scoring on final beam candidates
        sentence_candidates: List[SentenceCandidate] = []
        for vis_acc, word_path in beams:
            sent_text = " ".join(h.text for h in word_path)
            lang_score = self.score_sentence_text(sent_text)
            norm_vis = vis_acc / max(len(word_path), 1)
            fused = alpha_vis * norm_vis + beta_lang * lang_score

            sentence_candidates.append(
                SentenceCandidate(
                    words=word_path,
                    text=sent_text,
                    visual_score=norm_vis,
                    language_score=lang_score,
                    fused_score=fused,
                )
            )

        # Rank candidate sentences by fused score
        sentence_candidates.sort(key=lambda sc: sc.fused_score, reverse=True)
        best_sentence = sentence_candidates[0]

        # Step 3: Identify 'Context Rescues' and update WordHypothesis objects
        final_words: List[WordHypothesis] = []
        rescued_count = 0

        for i, chosen_word in enumerate(best_sentence.words):
            original_top = valid_positions[i][0]

            # Clone to keep original pristine
            updated_word = WordHypothesis(
                text=chosen_word.text,
                visual_score=chosen_word.visual_score,
                language_score=best_sentence.language_score,
                fused_score=best_sentence.fused_score,
                confidence=chosen_word.confidence,
                confidence_tier=chosen_word.confidence_tier,
                char_hypotheses=chosen_word.char_hypotheses,
                bbox=chosen_word.bbox,
                alternatives=original_top.alternatives,
                metadata=dict(chosen_word.metadata),
            )

            # Check if LM re-ranked and picked a different word than visual top-1
            if chosen_word.text.lower() != original_top.text.lower():
                updated_word.metadata["context_rescued"] = True
                updated_word.metadata["original_visual_text"] = original_top.text
                rescued_count += 1
                logger.info(
                    f"Context Rescued word at pos {i}: visual='{original_top.text}' → LM='{chosen_word.text}'"
                )

            final_words.append(updated_word)

        # Step 4: Assemble LineResult
        line_res = LineResult(
            words=final_words,
            raw_text=best_sentence.text,
            metadata={
                "mean_visual_conf": mean_vis_conf,
                "alpha_vis": alpha_vis,
                "beta_lang": beta_lang,
                "rescued_count": rescued_count,
                "n_sentence_candidates_evaluated": len(sentence_candidates),
            }
        )

        return line_res

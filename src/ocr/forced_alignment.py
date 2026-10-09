"""
Prompt-Guided Forced Alignment & Dysgraphia Error Diagnostic Engine.

In clinical screening (BHK, schools, clinics, and stylus collection),
children copy a KNOWN target sentence or word (the prompt).

This module aligns the OCR hypothesis against the expected prompt text to
extract diagnostic clinical biomarkers that standard OCR cannot detect:
  1. Letter Reversal Profiling (e.g., 'b' <-> 'd', 'p' <-> 'q', 'm' <-> 'w')
  2. Letter Omissions (dropped characters indicating attention or spatial skips)
  3. Spurious Insertions (extra strokes / tremor loops)
  4. Spatial Hesitation & Legibility Gradient across the sentence
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Canonical dysgraphia mirror and inversion letter reversal pairs
REVERSAL_PAIRS = {
    ('b', 'd'), ('d', 'b'),
    ('p', 'q'), ('q', 'p'),
    ('b', 'p'), ('p', 'b'),
    ('d', 'p'), ('p', 'd'),
    ('u', 'n'), ('n', 'u'),
    ('m', 'w'), ('w', 'm'),
    ('s', 'z'), ('z', 's'),
    ('t', 'f'), ('f', 't'),
}

# Phonetic substitutions common in dyslexic-dysgraphic subtypes
PHONETIC_PAIRS = {
    ('c', 'k'), ('k', 'c'),
    ('s', 'c'), ('c', 's'),
    ('f', 'v'), ('v', 'f'),
    ('g', 'j'), ('j', 'g'),
}


@dataclass
class CharAlignment:
    """Character-level alignment record between expected and observed letters."""
    expected_char: str
    observed_char: str
    status: str  # 'match', 'reversal', 'phonetic', 'substitution', 'omission', 'insertion'
    is_reversal: bool = False
    reversal_pair: Optional[Tuple[str, str]] = None
    confidence: float = 1.0


@dataclass
class WordAlignmentResult:
    """Word-level alignment record."""
    expected_word: str
    observed_word: str
    match_status: str  # 'exact_match', 'reversal_present', 'substituted', 'omitted', 'spurious'
    confidence: float = 0.0
    char_alignments: List[CharAlignment] = field(default_factory=list)
    reversal_count: int = 0
    omission_count: int = 0
    insertion_count: int = 0


@dataclass
class DiagnosticCopyTaskResult:
    """Complete diagnostic assessment for a prompt-guided copy-task."""
    prompt_text: str
    transcribed_text: str
    alignment_similarity: float
    total_expected_chars: int
    total_observed_chars: int
    reversal_count: int
    omission_count: int
    insertion_count: int
    substitution_count: int
    reversal_rate: float
    omission_rate: float
    word_alignments: List[WordAlignmentResult] = field(default_factory=list)
    reversals_detected: List[Dict[str, Any]] = field(default_factory=list)
    diagnostic_summary: str = ""
    diagnostic_badge_html: str = ""
    alignment_grid_html: str = ""


class ForcedAlignmentEngine:
    """
    Diagnostic forced alignment engine that maps student OCR transcriptions
    to reference copy prompts.
    """

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def _clean_text(self, text: str) -> str:
        """Standardize spaces and basic punctuation."""
        return re.sub(r'\s+', ' ', text.strip())

    def _align_strings_dp(
        self,
        target: str,
        observed: str,
        char_confidences: Optional[List[float]] = None
    ) -> List[CharAlignment]:
        """
        Global character sequence alignment using modified Needleman-Wunsch
        penalizing reversals less than arbitrary substitutions to highlight them.
        """
        tgt = target if self.case_sensitive else target.lower()
        obs = observed if self.case_sensitive else observed.lower()

        m, n = len(tgt), len(obs)
        dp = np.zeros((m + 1, n + 1), dtype=np.float32)

        GAP_PENALTY = -1.0
        MISMATCH_PENALTY = -2.0
        MATCH_REWARD = 2.0
        REVERSAL_SCORE = 0.5  # Partial credit with specific tag

        for i in range(m + 1):
            dp[i, 0] = i * GAP_PENALTY
        for j in range(n + 1):
            dp[0, j] = j * GAP_PENALTY

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                tc = tgt[i - 1]
                oc = obs[j - 1]

                if tc == oc:
                    sim = MATCH_REWARD
                elif (tc, oc) in REVERSAL_PAIRS:
                    sim = REVERSAL_SCORE
                elif (tc, oc) in PHONETIC_PAIRS:
                    sim = REVERSAL_SCORE * 0.8
                else:
                    sim = MISMATCH_PENALTY

                dp[i, j] = max(
                    dp[i - 1, j - 1] + sim,    # match / substitution
                    dp[i - 1, j] + GAP_PENALTY,  # omission (in target, not in obs)
                    dp[i, j - 1] + GAP_PENALTY   # insertion (in obs, not in target)
                )

        # Traceback
        alignments: List[CharAlignment] = []
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0:
                tc = tgt[i - 1]
                oc = obs[j - 1]
                conf = char_confidences[j - 1] if (char_confidences and j - 1 < len(char_confidences)) else 0.85

                if tc == oc:
                    alignments.append(CharAlignment(
                        expected_char=target[i - 1],
                        observed_char=observed[j - 1],
                        status="match",
                        confidence=conf
                    ))
                    i -= 1
                    j -= 1
                    continue
                elif (tc, oc) in REVERSAL_PAIRS:
                    alignments.append(CharAlignment(
                        expected_char=target[i - 1],
                        observed_char=observed[j - 1],
                        status="reversal",
                        is_reversal=True,
                        reversal_pair=(tc, oc),
                        confidence=conf
                    ))
                    i -= 1
                    j -= 1
                    continue
                elif (tc, oc) in PHONETIC_PAIRS:
                    alignments.append(CharAlignment(
                        expected_char=target[i - 1],
                        observed_char=observed[j - 1],
                        status="phonetic",
                        confidence=conf
                    ))
                    i -= 1
                    j -= 1
                    continue
                elif dp[i, j] == dp[i - 1, j - 1] + MISMATCH_PENALTY:
                    alignments.append(CharAlignment(
                        expected_char=target[i - 1],
                        observed_char=observed[j - 1],
                        status="substitution",
                        confidence=conf
                    ))
                    i -= 1
                    j -= 1
                    continue

            if i > 0 and (j == 0 or dp[i, j] == dp[i - 1, j] + GAP_PENALTY):
                # Omission (expected letter skipped)
                alignments.append(CharAlignment(
                    expected_char=target[i - 1],
                    observed_char="∅",
                    status="omission",
                    confidence=0.0
                ))
                i -= 1
            else:
                # Insertion (spurious letter)
                conf = char_confidences[j - 1] if (char_confidences and j - 1 < len(char_confidences)) else 0.5
                alignments.append(CharAlignment(
                    expected_char="∅",
                    observed_char=observed[j - 1],
                    status="insertion",
                    confidence=conf
                ))
                j -= 1

        alignments.reverse()
        return alignments

    def evaluate_copy_task(
        self,
        prompt_text: str,
        transcribed_words: List[Dict[str, Any]],
    ) -> DiagnosticCopyTaskResult:
        """
        Compares transcribed words against the reference copy-task prompt.

        Args:
            prompt_text: Target text given to the student.
            transcribed_words: List of dicts with 'text', 'confidence', 'char_confidences'.
        """
        cleaned_prompt = self._clean_text(prompt_text)
        prompt_words = cleaned_prompt.split()

        obs_word_list = [w.get("text", "") for w in transcribed_words]
        full_transcription = " ".join(obs_word_list)

        word_alignments: List[WordAlignmentResult] = []
        total_reversals = 0
        total_omissions = 0
        total_insertions = 0
        total_substitutions = 0
        reversals_list: List[Dict[str, Any]] = []

        # Align word-by-word
        max_len = max(len(prompt_words), len(transcribed_words))

        for idx in range(max_len):
            tgt_w = prompt_words[idx] if idx < len(prompt_words) else ""
            obs_dict = transcribed_words[idx] if idx < len(transcribed_words) else {}
            obs_w = obs_dict.get("text", "")
            conf = obs_dict.get("confidence", 0.0)
            char_confs = obs_dict.get("char_confidences", [])

            if tgt_w and obs_w:
                char_aligns = self._align_strings_dp(tgt_w, obs_w, char_confs)
                n_rev = sum(1 for c in char_aligns if c.status == "reversal")
                n_om = sum(1 for c in char_aligns if c.status == "omission")
                n_ins = sum(1 for c in char_aligns if c.status == "insertion")
                n_sub = sum(1 for c in char_aligns if c.status in ("substitution", "phonetic"))

                total_reversals += n_rev
                total_omissions += n_om
                total_insertions += n_ins
                total_substitutions += n_sub

                for c_idx, c in enumerate(char_aligns):
                    if c.is_reversal:
                        reversals_list.append({
                            "word_idx": idx + 1,
                            "expected_word": tgt_w,
                            "observed_word": obs_w,
                            "char_idx": c_idx + 1,
                            "expected_char": c.expected_char,
                            "observed_char": c.observed_char,
                            "pair": f"{c.expected_char} ➔ {c.observed_char}",
                            "confidence": c.confidence,
                        })

                if tgt_w.lower() == obs_w.lower():
                    status = "exact_match"
                elif n_rev > 0:
                    status = "reversal_present"
                elif n_om > 0:
                    status = "omitted"
                else:
                    status = "substituted"

                word_alignments.append(WordAlignmentResult(
                    expected_word=tgt_w,
                    observed_word=obs_w,
                    match_status=status,
                    confidence=conf,
                    char_alignments=char_aligns,
                    reversal_count=n_rev,
                    omission_count=n_om,
                    insertion_count=n_ins
                ))
            elif tgt_w and not obs_w:
                # Completely missed word
                char_aligns = [
                    CharAlignment(expected_char=c, observed_char="∅", status="omission", confidence=0.0)
                    for c in tgt_w
                ]
                total_omissions += len(tgt_w)
                word_alignments.append(WordAlignmentResult(
                    expected_word=tgt_w,
                    observed_word="[OMITTED]",
                    match_status="omitted",
                    confidence=0.0,
                    char_alignments=char_aligns,
                    omission_count=len(tgt_w)
                ))
            elif obs_w and not tgt_w:
                # Spurious extra word
                char_aligns = [
                    CharAlignment(expected_char="∅", observed_char=c, status="insertion", confidence=conf)
                    for c in obs_w
                ]
                total_insertions += len(obs_w)
                word_alignments.append(WordAlignmentResult(
                    expected_word="[NONE]",
                    observed_word=obs_w,
                    match_status="spurious",
                    confidence=conf,
                    char_alignments=char_aligns,
                    insertion_count=len(obs_w)
                ))

        total_tgt_chars = max(sum(len(w) for w in prompt_words), 1)
        total_obs_chars = sum(len(w.get("text", "")) for w in transcribed_words)

        reversal_rate = total_reversals / total_tgt_chars
        omission_rate = total_omissions / total_tgt_chars

        exact_words = sum(1 for w in word_alignments if w.match_status == "exact_match")
        alignment_sim = exact_words / max(len(prompt_words), 1)

        # Generate clinical diagnostic summary
        diagnostic_summary = self._generate_summary(
            reversals_list, total_reversals, total_omissions, total_insertions,
            reversal_rate, omission_rate, alignment_sim
        )

        badge_html = self._generate_badge_html(total_reversals, total_omissions, alignment_sim)
        grid_html = self._generate_grid_html(word_alignments)

        return DiagnosticCopyTaskResult(
            prompt_text=cleaned_prompt,
            transcribed_text=full_transcription,
            alignment_similarity=alignment_sim,
            total_expected_chars=total_tgt_chars,
            total_observed_chars=total_obs_chars,
            reversal_count=total_reversals,
            omission_count=total_omissions,
            insertion_count=total_insertions,
            substitution_count=total_substitutions,
            reversal_rate=reversal_rate,
            omission_rate=omission_rate,
            word_alignments=word_alignments,
            reversals_detected=reversals_list,
            diagnostic_summary=diagnostic_summary,
            diagnostic_badge_html=badge_html,
            alignment_grid_html=grid_html,
        )

    def _generate_summary(
        self,
        reversals: List[Dict[str, Any]],
        n_rev: int,
        n_om: int,
        n_ins: int,
        rev_rate: float,
        om_rate: float,
        sim: float,
    ) -> str:
        findings = []

        if n_rev > 0:
            rev_details = ", ".join(f"'{r['expected_char']}'➔'{r['observed_char']}' in '{r['expected_word']}'" for r in reversals[:3])
            findings.append(f"**Letter Reversals Detected ({n_rev} instances):** Mirror/inversion pattern found ({rev_details}). High clinical correlation with spatial-motor dysgraphia.")
        else:
            findings.append("**Letter Directionality:** Normal. No spatial mirror-reversals (e.g. b/d, p/q) identified.")

        if n_om > 2:
            findings.append(f"**Character Omissions ({n_om} dropped):** Omission rate is {om_rate:.1%}. Indicates letter-skipping, fine motor fatigue, or attentional lapses.")

        if n_ins > 2:
            findings.append(f"**Spurious Stroke Insertions ({n_ins} extra):** Child added extra loops/strokes, consistent with motor hesitation or tremor.")

        if sim >= 0.85:
            findings.append("**Overall Legibility & Fidelity:** High (Word Accuracy: " + f"{sim:.1%}). Writing accurately adheres to target copy.")
        elif sim >= 0.60:
            findings.append("**Overall Legibility & Fidelity:** Moderate (" + f"{sim:.1%}). Noticeable character deformation and spatial irregularity.")
        else:
            findings.append("**Overall Legibility & Fidelity:** Low (< 60%). Significant handwriting degradation, severe letter substitutions, and spatial fragmentation.")

        return "\n\n".join(findings)

    def _generate_badge_html(self, n_rev: int, n_om: int, sim: float) -> str:
        rev_color = "#ef4444" if n_rev > 0 else "#10b981"
        rev_text = f"⚠️ {n_rev} Letter Reversal{'s' if n_rev != 1 else ''}" if n_rev > 0 else "✓ No Reversals"

        om_color = "#f59e0b" if n_om > 1 else "#10b981"
        om_text = f"⚠️ {n_om} Omission{'s' if n_om != 1 else ''}" if n_om > 1 else "✓ Omissions Low"

        sim_color = "#10b981" if sim >= 0.80 else ("#f59e0b" if sim >= 0.50 else "#ef4444")
        sim_text = f"Fidelity: {sim:.0%}"

        return f"""
        <div style="display:flex; gap:12px; margin: 12px 0; flex-wrap:wrap;">
            <div style="background:{rev_color}18; border:1px solid {rev_color}; color:{rev_color}; padding:6px 14px; border-radius:20px; font-weight:600; font-size:13px;">
                {rev_text}
            </div>
            <div style="background:{om_color}18; border:1px solid {om_color}; color:{om_color}; padding:6px 14px; border-radius:20px; font-weight:600; font-size:13px;">
                {om_text}
            </div>
            <div style="background:{sim_color}18; border:1px solid {sim_color}; color:{sim_color}; padding:6px 14px; border-radius:20px; font-weight:600; font-size:13px;">
                {sim_text}
            </div>
        </div>
        """

    def _generate_grid_html(self, alignments: List[WordAlignmentResult]) -> str:
        html_blocks = []
        for w in alignments:
            char_boxes = []
            for c in w.char_alignments:
                if c.status == "match":
                    bg = "#dcfce7"
                    border = "#16a34a"
                    text_color = "#15803d"
                    label = "✓"
                elif c.status == "reversal":
                    bg = "#fee2e2"
                    border = "#ef4444"
                    text_color = "#b91c1c"
                    label = f"REV ({c.expected_char}➔{c.observed_char})"
                elif c.status == "omission":
                    bg = "#fef3c7"
                    border = "#d97706"
                    text_color = "#b45309"
                    label = f"DROP ({c.expected_char})"
                elif c.status == "insertion":
                    bg = "#f3e8ff"
                    border = "#9333ea"
                    text_color = "#7e22ce"
                    label = f"EXTRA ({c.observed_char})"
                else:
                    bg = "#ffedd5"
                    border = "#ea580c"
                    text_color = "#c2410c"
                    label = f"{c.expected_char}➔{c.observed_char}"

                char_boxes.append(f"""
                <div style="display:inline-flex; flex-direction:column; align-items:center; margin:2px 3px; border:1px solid {border}; background:{bg}; border-radius:6px; padding:4px 6px; min-width:28px;">
                    <span style="font-size:16px; font-weight:bold; color:{text_color};">{c.observed_char}</span>
                    <span style="font-size:10px; color:#475569; margin-top:2px;">{label}</span>
                </div>
                """)

            chars_str = "".join(char_boxes)
            status_badge = (
                "<span style='color:#16a34a; font-size:11px;'>● Exact</span>" if w.match_status == "exact_match"
                else ("<span style='color:#ef4444; font-size:11px; font-weight:bold;'>● Reversal</span>" if w.match_status == "reversal_present"
                else "<span style='color:#ea580c; font-size:11px;'>● Mismatch</span>")
            )

            html_blocks.append(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:10px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:13px;">
                    <span><b>Expected:</b> <code style="background:#f1f5f9; padding:2px 6px; border-radius:4px;">{w.expected_word}</code></span>
                    <span><b>Observed:</b> <code style="background:#f1f5f9; padding:2px 6px; border-radius:4px;">{w.observed_word}</code> {status_badge}</span>
                </div>
                <div style="display:flex; flex-wrap:wrap;">
                    {chars_str}
                </div>
            </div>
            """)

        return f"<div style='max-height:420px; overflow-y:auto; padding-right:4px;'>{''.join(html_blocks)}</div>"

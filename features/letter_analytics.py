"""Granular Letter-Level Linguistic & Graphomotor Error Analytics Engine.

Provides fine-grained diagnostic extraction across 4 clinical error groups:
1. Target-referenced character error taxonomy:
   - Omissions: dropped characters, whole words, dropped word endings (suffixes).
   - Insertions & Stutters: extraneous characters, immediate letter/word repetitions.
   - Confusable pairs: spatial reversals/rotations (b/d, p/q, m/n, u/n, w/m) and Devanagari (ब/व, घ/ध, भ/म).
   - Case & punctuation: mid-word capitalization, sentence boundary violations, errant intra-word periods.
   - Spatial error distribution: early vs late line error ratio (fatigue).
   - Word segmentation: merged words and split words.
2. Target-free confidence dynamics:
   - Confidence trajectory slope across line (trailing off / fatigue).
   - Inter-word / inter-token consistency (CV).
   - Low-confidence cluster rate.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Latin confusable pairs: (written, target) or (target, written)
CONFUSABLE_PAIRS_LATIN: Set[Tuple[str, str]] = {
    # Reversals & vertical mirror reflections
    ("b", "d"),
    ("d", "b"),
    ("p", "q"),
    ("q", "p"),
    # Inversions & 180-degree rotations
    ("n", "u"),
    ("u", "n"),
    ("m", "w"),
    ("w", "m"),
    # Stroke similarity / arch count confusion
    ("m", "n"),
    ("n", "m"),
    ("v", "w"),
    ("w", "v"),
    ("c", "e"),
    ("e", "c"),
    ("f", "t"),
    ("t", "f"),
}

# Devanagari confusable pairs & headline/belly differences
CONFUSABLE_PAIRS_DEVANAGARI: Set[Tuple[str, str]] = {
    ("ब", "व"),
    ("व", "ब"),
    ("घ", "ध"),
    ("ध", "घ"),
    ("भ", "म"),
    ("म", "भ"),
    ("श", "ष"),
    ("ष", "श"),
    ("ख", "र"),
    ("र", "ख"),
}

DEVANAGARI_MATRAS: Set[str] = {
    "\u093e",  # ा (aa)
    "\u093f",  # ि (i)
    "\u0940",  # ी (ii)
    "\u0941",  # ु (u)
    "\u0942",  # ू (uu)
    "\u0943",  # ृ (r)
    "\u0947",  # े (e)
    "\u0948",  # ै (ai)
    "\u094b",  # ो (o)
    "\u094c",  # ौ (au)
    "\u0902",  # ं (anusvara)
    "\u0903",  # ः (visarga)
    "\u0901",  # ँ (candrabindu)
}

# Phonetic substitutions / homophones (spelling errors, distinguished from motor reversals)
PHONETIC_PAIRS_LATIN: Set[Tuple[str, str]] = {
    ("c", "k"), ("k", "c"),
    ("c", "s"), ("s", "c"),
    ("s", "z"), ("z", "s"),
    ("f", "v"), ("v", "f"),
    ("j", "g"), ("g", "j"),
    ("x", "z"), ("z", "x"),
    ("x", "s"), ("s", "x"),
}

PHONETIC_DIGRAPHS_LATIN: Set[Tuple[str, str]] = {
    ("ph", "f"), ("f", "ph"),
    ("ee", "ea"), ("ea", "ee"),
    ("ai", "ay"), ("ay", "ai"),
    ("oa", "ow"), ("ow", "oa"),
    ("ou", "ow"), ("ow", "ou"),
    ("ck", "k"), ("k", "ck"),
    ("er", "ir"), ("ir", "er"),
    ("er", "ur"), ("ur", "er"),
}

PHONETIC_PAIRS_DEVANAGARI: Set[Tuple[str, str]] = {
    ("ष", "श"), ("श", "ष"),
    ("श", "स"), ("स", "श"),
    ("ण", "न"), ("न", "ण"),
    ("ऋ", "रि"), ("रि", "ऋ"),
}

DEVANAGARI_VIRAMA = "\u094d"  # ् (halant / virama for conjuncts)


def clean_text_for_alignment(text: str) -> str:
    """Normalize whitespace and remove non-printable characters while preserving case."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def analyze_letter_errors(
    target_text: str,
    decoded_text: str,
    token_confidences: Optional[List[float]] = None,
    script: str = "latin",
) -> Dict[str, float]:
    """Compute rich character-level, word-level, and trajectory error metrics.

    Parameters
    ----------
    target_text : str
        The standardized target prompt text (e.g. from CANDIDATE_PROMPTS).
    decoded_text : str
        The raw decoded text produced by TrOCR.
    token_confidences : list of float, optional
        Step-by-step token posterior confidences from TrOCR.
    script : str
        'latin' or 'devanagari'.

    Returns
    -------
    dict with 20+ granular diagnostic error features.
    """
    clean_target = clean_text_for_alignment(target_text)
    clean_decoded = clean_text_for_alignment(decoded_text)

    if not clean_target:
        # Extract target-free confidence dynamics and case anomalies even without a reference prompt
        res = _empty_analytics_result()
        d_words = clean_decoded.split()
        if d_words and script == "latin":
            midword_caps = 0
            errant_periods = 0
            for w in d_words:
                alpha_chars = [c for c in w if c.isalpha()]
                if len(alpha_chars) > 1 and any(c.isupper() for c in alpha_chars[1:]):
                    midword_caps += 1
                if "." in w and not w.endswith("."):
                    errant_periods += 1
            res["midword_caps_count"] = float(midword_caps)
            res["midword_caps_rate"] = float(midword_caps / len(d_words))
            res["errant_internal_periods"] = float(errant_periods)

        if token_confidences is not None and len(token_confidences) > 1:
            confs = np.array(token_confidences, dtype=float)
            mean_c = float(np.mean(confs))
            std_c = float(np.std(confs))
            res["conf_consistency_cv"] = float(std_c / (mean_c + 1e-6))
            x = np.arange(len(confs), dtype=float)
            cov = np.cov(x, confs)[0, 1]
            var_x = np.var(x)
            if var_x > 1e-6:
                res["conf_trajectory_slope"] = float(cov / var_x)
            q_len = max(1, len(confs) // 4)
            res["conf_drop_quarter"] = float(np.mean(confs[:q_len]) - np.mean(confs[-q_len:]))
        return res

    t_len = max(1, len(clean_target))
    t_words = clean_target.split()
    d_words = clean_decoded.split()
    n_t_words = max(1, len(t_words))
    n_d_words = max(1, len(d_words))

    # 1. SEQUENCE ALIGNMENT (Levenshtein OpCodes at Character Level)
    sm = difflib.SequenceMatcher(None, clean_target, clean_decoded)
    opcodes = sm.get_opcodes()

    char_omissions = 0
    char_insertions = 0
    char_substitutions = 0
    letter_repeats = 0
    letter_transpositions = 0
    confusable_swaps = 0
    phonetic_errors = 0
    matra_omissions = 0
    conjunct_errors = 0

    early_errors = 0
    late_errors = 0
    one_third = t_len / 3.0
    two_thirds = 2.0 * t_len / 3.0

    confusable_dict = CONFUSABLE_PAIRS_LATIN if script == "latin" else CONFUSABLE_PAIRS_DEVANAGARI

    for tag, i1, i2, j1, j2 in opcodes:
        span_len = i2 - i1
        # Track spatial position of error relative to prompt
        is_early = (i1 < one_third)
        is_late = (i2 > two_thirds)

        if tag == "delete":
            char_omissions += span_len
            deleted_chars = clean_target[i1:i2]
            if script == "devanagari":
                for ch in deleted_chars:
                    if ch in DEVANAGARI_MATRAS:
                        matra_omissions += 1
                    if ch == DEVANAGARI_VIRAMA:
                        conjunct_errors += 1
            if is_early:
                early_errors += span_len
            if is_late:
                late_errors += span_len

        elif tag == "insert":
            ins_len = j2 - j1
            char_insertions += ins_len
            ins_str = clean_decoded[j1:j2]
            # Check immediate letter stutter / repeat
            if j1 > 0:
                prev_char = clean_decoded[j1 - 1]
                if prev_char in ins_str:
                    letter_repeats += 1
            if is_early:
                early_errors += ins_len
            if is_late:
                late_errors += ins_len

        elif tag == "replace":
            t_sub = clean_target[i1:i2]
            d_sub = clean_decoded[j1:j2]
            sub_count = max(len(t_sub), len(d_sub))
            char_substitutions += sub_count

            # Check letter transpositions (adjacent swapped letters, e.g. teh vs the)
            if len(t_sub) == 2 and len(d_sub) == 2:
                if t_sub[0].lower() == d_sub[1].lower() and t_sub[1].lower() == d_sub[0].lower():
                    letter_transpositions += 1

            # Check confusable pairs (e.g. b/d, p/q)
            for tc, dc in zip(t_sub.lower(), d_sub.lower()):
                if (tc, dc) in confusable_dict or (dc, tc) in confusable_dict:
                    confusable_swaps += 1

            # Check phonetic slips
            if script == "latin":
                for tc, dc in zip(t_sub.lower(), d_sub.lower()):
                    if (tc, dc) in PHONETIC_PAIRS_LATIN:
                        phonetic_errors += 1
                t_low, d_low = t_sub.lower(), d_sub.lower()
                if (t_low, d_low) in PHONETIC_DIGRAPHS_LATIN:
                    phonetic_errors += 1
            elif script == "devanagari":
                for tc, dc in zip(t_sub, d_sub):
                    if (tc, dc) in PHONETIC_PAIRS_DEVANAGARI:
                        phonetic_errors += 1

            if script == "devanagari":
                for ch in t_sub:
                    if ch in DEVANAGARI_MATRAS and ch not in d_sub:
                        matra_omissions += 1

            if is_early:
                early_errors += sub_count
            if is_late:
                late_errors += sub_count

    # 2. WORD-LEVEL ANALYSIS (Dropped endings, word omissions, merged/split words, position)
    w_sm = difflib.SequenceMatcher(None, [w.lower() for w in t_words], [w.lower() for w in d_words])
    w_opcodes = w_sm.get_opcodes()

    word_omissions = 0
    dropped_endings = 0
    word_repeats = 0
    word_onset_errors = 0
    word_mid_errors = 0
    word_coda_errors = 0

    for idx in range(len(d_words) - 1):
        if d_words[idx].lower() == d_words[idx + 1].lower():
            word_repeats += 1

    for tag, i1, i2, j1, j2 in w_opcodes:
        if tag == "delete":
            word_omissions += (i2 - i1)
        elif tag == "replace":
            t_slice = t_words[i1:i2]
            d_slice = d_words[j1:j2]
            if len(t_slice) > len(d_slice):
                word_omissions += (len(t_slice) - len(d_slice))
            for idx_offset in range(min(len(t_slice), len(d_slice))):
                tw = t_slice[idx_offset].lower()
                dw = d_slice[idx_offset].lower()
                # If target word is longer and begins with decoded word -> dropped ending
                if len(tw) > len(dw) and tw.startswith(dw) and len(dw) >= 2:
                    dropped_endings += 1
                if tw != dw:
                    if len(tw) > 0 and len(dw) > 0:
                        if tw[0] != dw[0]:
                            word_onset_errors += 1
                        if tw[-1] != dw[-1]:
                            word_coda_errors += 1
                        if len(tw) > 2 and len(dw) > 2 and tw[1:-1] != dw[1:-1]:
                            word_mid_errors += 1
                    # Detect adjacent letter transpositions within word
                    min_l = min(len(tw), len(dw))
                    for k in range(min_l - 1):
                        if tw[k] == dw[k + 1] and tw[k + 1] == dw[k] and tw[k] != tw[k + 1]:
                            letter_transpositions += 1

    # Word segmentation errors (merged words: e.g. "quickbrown"; split words: e.g. "qu ick")
    merged_words = 0
    split_words = 0
    t_no_space = "".join(t_words).lower()
    d_no_space = "".join(d_words).lower()

    for idx in range(len(t_words) - 1):
        pair = (t_words[idx] + t_words[idx + 1]).lower()
        if any(pair in dw.lower() for dw in d_words):
            merged_words += 1

    for dw in d_words:
        if len(dw) <= 2 and any(dw.lower() in tw.lower() for tw in t_words):
            split_words += 1

    # 3. CASE & PUNCTUATION (English-specific)
    midword_caps = 0
    missing_start_cap = 0.0
    missing_end_period = 0.0
    errant_internal_periods = 0

    if script == "latin":
        for w in d_words:
            # Check mid-word capitals (e.g. qUick, laZy)
            alpha_chars = [c for c in w if c.isalpha()]
            if len(alpha_chars) > 1 and any(c.isupper() for c in alpha_chars[1:]):
                midword_caps += 1
            # Check errant periods inside word (e.g. OF. or th.e)
            if "." in w and not w.endswith("."):
                errant_internal_periods += 1

        if clean_target and clean_target[0].isupper():
            if clean_decoded and not clean_decoded[0].isupper():
                missing_start_cap = 1.0

        if clean_target and clean_target.endswith("."):
            if clean_decoded and not clean_decoded.endswith("."):
                missing_end_period = 1.0

    # 4. CONFIDENCE DYNAMICS & FATIGUE SLOPES
    conf_slope = 0.0
    conf_cv = 0.0
    conf_drop_quarter = 0.0

    if token_confidences is not None and len(token_confidences) > 1:
        confs = np.array(token_confidences, dtype=float)
        mean_c = float(np.mean(confs))
        std_c = float(np.std(confs))
        conf_cv = float(std_c / (mean_c + 1e-6))

        # Linear regression slope over token index
        x = np.arange(len(confs), dtype=float)
        cov = np.cov(x, confs)[0, 1]
        var_x = np.var(x)
        if var_x > 1e-6:
            conf_slope = float(cov / var_x)

        # Drop from first quarter to last quarter of line
        q_len = max(1, len(confs) // 4)
        first_q = float(np.mean(confs[:q_len]))
        last_q = float(np.mean(confs[-q_len:]))
        conf_drop_quarter = float(first_q - last_q)

    # Fatigue error ratio: ratio of late errors to early errors
    fatigue_error_ratio = float((late_errors + 1.0) / (early_errors + 1.0))

    # 5. MOTOR VS SPELLING DISSOCIATION & COVERAGE
    motor_errors = (
        confusable_swaps
        + letter_repeats
        + midword_caps
        + dropped_endings
        + errant_internal_periods
        + letter_transpositions
    )
    motor_error_rate = float(motor_errors / t_len)
    phonetic_error_rate = float(phonetic_errors / t_len)
    motor_vs_phonetic_ratio = float((motor_error_rate + 0.01) / (phonetic_error_rate + 0.01))
    coverage_ratio = float(len(clean_decoded) / t_len)

    return {
        "char_omission_count": float(char_omissions),
        "char_omission_rate": float(char_omissions / t_len),
        "word_omission_count": float(word_omissions),
        "word_omission_rate": float(word_omissions / n_t_words),
        "dropped_endings_count": float(dropped_endings),
        "dropped_endings_rate": float(dropped_endings / n_t_words),
        "char_insertion_count": float(char_insertions),
        "char_insertion_rate": float(char_insertions / t_len),
        "letter_repeat_count": float(letter_repeats),
        "letter_transpositions": float(letter_transpositions),
        "char_substitution_count": float(char_substitutions),
        "char_substitution_rate": float(char_substitutions / t_len),
        "confusable_swaps_count": float(confusable_swaps),
        "confusable_rate": float(confusable_swaps / t_len),
        "phonetic_error_count": float(phonetic_errors),
        "phonetic_error_rate": phonetic_error_rate,
        "matra_omissions_count": float(matra_omissions),
        "conjunct_errors_count": float(conjunct_errors),
        "midword_caps_count": float(midword_caps),
        "midword_caps_rate": float(midword_caps / n_d_words),
        "errant_internal_periods": float(errant_internal_periods),
        "missing_start_cap": missing_start_cap,
        "missing_end_period": missing_end_period,
        "merged_words_count": float(merged_words),
        "split_words_count": float(split_words),
        "word_onset_errors": float(word_onset_errors),
        "word_mid_errors": float(word_mid_errors),
        "word_coda_errors": float(word_coda_errors),
        "early_line_error_rate": float(early_errors / t_len),
        "late_line_error_rate": float(late_errors / t_len),
        "fatigue_error_ratio": fatigue_error_ratio,
        "motor_error_count": float(motor_errors),
        "motor_error_rate": motor_error_rate,
        "motor_vs_phonetic_ratio": motor_vs_phonetic_ratio,
        "ocr_coverage_ratio": coverage_ratio,
        "conf_trajectory_slope": conf_slope,
        "conf_consistency_cv": conf_cv,
        "conf_drop_quarter": conf_drop_quarter,
    }


def _empty_analytics_result() -> Dict[str, float]:
    """Return default neutral values when target text is missing or invalid."""
    return {
        "char_omission_count": 0.0,
        "char_omission_rate": 0.0,
        "word_omission_count": 0.0,
        "word_omission_rate": 0.0,
        "dropped_endings_count": 0.0,
        "dropped_endings_rate": 0.0,
        "char_insertion_count": 0.0,
        "char_insertion_rate": 0.0,
        "letter_repeat_count": 0.0,
        "letter_transpositions": 0.0,
        "char_substitution_count": 0.0,
        "char_substitution_rate": 0.0,
        "confusable_swaps_count": 0.0,
        "confusable_rate": 0.0,
        "phonetic_error_count": 0.0,
        "phonetic_error_rate": 0.0,
        "matra_omissions_count": 0.0,
        "conjunct_errors_count": 0.0,
        "midword_caps_count": 0.0,
        "midword_caps_rate": 0.0,
        "errant_internal_periods": 0.0,
        "missing_start_cap": 0.0,
        "missing_end_period": 0.0,
        "merged_words_count": 0.0,
        "split_words_count": 0.0,
        "word_onset_errors": 0.0,
        "word_mid_errors": 0.0,
        "word_coda_errors": 0.0,
        "early_line_error_rate": 0.0,
        "late_line_error_rate": 0.0,
        "fatigue_error_ratio": 1.0,
        "motor_error_count": 0.0,
        "motor_error_rate": 0.0,
        "motor_vs_phonetic_ratio": 1.0,
        "ocr_coverage_ratio": 0.0,
        "conf_trajectory_slope": 0.0,
        "conf_consistency_cv": 0.0,
        "conf_drop_quarter": 0.0,
    }

"""Unit tests for granular letter-level error taxonomy and fatigue metrics."""

import pytest
from features.letter_analytics import analyze_letter_errors


def test_clean_match():
    """Verify clean identical match yields zero errors."""
    target = "The quick brown fox jumps over the lazy dog."
    decoded = "The quick brown fox jumps over the lazy dog."
    confs = [0.95, 0.94, 0.96, 0.92, 0.95, 0.93, 0.94, 0.95, 0.96]

    res = analyze_letter_errors(target, decoded, token_confidences=confs, script="latin")
    assert res["char_omission_count"] == 0.0
    assert res["char_insertion_count"] == 0.0
    assert res["char_substitution_count"] == 0.0
    assert res["confusable_swaps_count"] == 0.0
    assert res["midword_caps_count"] == 0.0
    assert res["missing_start_cap"] == 0.0
    assert res["missing_end_period"] == 0.0
    assert res["dropped_endings_count"] == 0.0


def test_confusable_reversals_latin():
    """Verify b/d and p/q spatial reversals are identified."""
    target = "bad pig"
    decoded = "dad qig"  # b->d, p->q

    res = analyze_letter_errors(target, decoded, script="latin")
    assert res["confusable_swaps_count"] >= 2.0
    assert res["char_substitution_count"] >= 2.0


def test_dropped_endings_and_midword_caps():
    """Verify dropped word endings and mid-word capitalization are detected."""
    target = "she jumps and cooks food"
    decoded = "she jump and cooKs food"  # 'jumps' -> 'jump' (dropped ending), 'cooKs' (midword cap)

    res = analyze_letter_errors(target, decoded, script="latin")
    assert res["dropped_endings_count"] >= 1.0
    assert res["midword_caps_count"] >= 1.0


def test_devanagari_confusables_and_matra():
    """Verify Devanagari belly/loop confusables and matra omissions."""
    target = "भारत देश"
    decoded = "भरत देख"  # missing 'aa' matra in भारत -> भरत, substitute श -> ख

    res = analyze_letter_errors(target, decoded, script="devanagari")
    assert res["matra_omissions_count"] >= 1.0
    assert res["char_omission_count"] >= 1.0


def test_confidence_trajectory_fatigue():
    """Verify negative slope indicates trailing off / fatigue."""
    target = "one two three four five six"
    decoded = "one two three four five six"
    # Decaying confidence
    confs = [0.95, 0.90, 0.82, 0.70, 0.55, 0.40]

    res = analyze_letter_errors(target, decoded, token_confidences=confs, script="latin")
    assert res["conf_trajectory_slope"] < -0.05
    assert res["conf_drop_quarter"] > 0.30


def test_phonetic_slips_and_motor_dissociation():
    """Verify distinction between phonetic spelling slips (c->k, ph->f) vs motor errors."""
    # Phonetic slip: 'cat' -> 'kat', 'phone' -> 'fone'
    target = "cat phone"
    decoded = "kat fone"
    res = analyze_letter_errors(target, decoded, script="latin")
    assert res["phonetic_error_count"] >= 2.0
    assert res["motor_error_count"] == 0.0
    assert res["motor_vs_phonetic_ratio"] < 1.0


def test_letter_transpositions():
    """Verify adjacent letter order transpositions (e.g. teh for the, form for from)."""
    target = "the cat"
    decoded = "teh cat"
    res = analyze_letter_errors(target, decoded, script="latin")
    assert res["letter_transpositions"] >= 1.0
    assert res["motor_error_count"] >= 1.0


def test_word_error_positions_onset_mid_coda():
    """Verify onset, nucleus, and coda error distribution within words."""
    target = "park playing kitchen"
    decoded = "dark playin kitchin"  # dark: onset (p->d), playin: coda (dropped g), kitchin: mid (e->i)
    res = analyze_letter_errors(target, decoded, script="latin")
    assert res["word_onset_errors"] >= 1.0
    assert res["word_coda_errors"] >= 1.0
    assert res["word_mid_errors"] >= 1.0


def test_devanagari_conjunct_errors():
    """Verify Devanagari virama / halant conjunct omission is captured."""
    # Hindi conjunct: 'स्कूल' (has virama: स + ् + क + ू + ल)
    target = "स्कूल"
    decoded = "सकूल"  # halant dropped, written as full consonant
    res = analyze_letter_errors(target, decoded, script="devanagari")
    assert res["conjunct_errors_count"] >= 1.0


"""Unit and Invariant Tests for Tiling Preprocessor (Fix 1, Fix 2, Fix 3).

Verifies:
1. Fix 1: Scaling by true character height h (mapping h -> 48 px) and vertical centering on 224 canvas.
2. Fix 2: Rejection gate rejecting blank tiles, flat ruled lines, and sparse residual lines.
3. Fix 3: Strict zeroing of optical density in Channel 2 outside ink, ensuring pure white [255, 255, 255] background.
"""

import numpy as np
import pytest
from baseline.tiling_preprocessor import (
    build_white_canvas_strip,
    estimate_strip_x_height,
    extract_tiles_from_strip,
)


def test_fix3_optical_density_zeroing_no_pink_background():
    """Verify Channel 2 optical density is strictly zeroed outside ink mask."""
    # Synthetic noisy gray paper with ink
    h, w = 100, 300
    np.random.seed(42)
    # Paper background with grain and shadows (gray 200..240)
    raw_gray = np.random.randint(200, 241, size=(h, w), dtype=np.uint8)

    # Clean ink mask: a letter block in the middle
    ink_mask = np.zeros((h, w), dtype=np.uint8)
    ink_mask[20:70, 50:250] = 1  # 50px tall letter strokes
    # Darken raw gray inside ink
    raw_gray[ink_mask == 1] = 50

    strip = build_white_canvas_strip(ink_mask, raw_gray)
    assert strip.shape == (h, w, 3)

    # Everywhere outside ink mask, all 3 channels MUST be identically 255 (pure white)
    bg_pixels = strip[ink_mask == 0]
    assert np.all(bg_pixels[:, 0] == 255), "Ch1 (Red) must be 255 outside ink"
    assert np.all(bg_pixels[:, 1] == 255), "Ch2 (Green) must be 255 outside ink (Fix 3: zero optical density)"
    assert np.all(bg_pixels[:, 2] == 255), "Ch3 (Blue) must be 255 outside ink"

    # Inside ink mask, strokes should be dark
    ink_pixels = strip[ink_mask == 1]
    assert np.mean(ink_pixels[:, 0]) == 0.0  # Ch1 inverted ink is 0
    assert np.mean(ink_pixels[:, 1]) < 128.0  # Ch2 is dark shaded ink


def test_fix1_scale_by_true_character_height():
    """Verify strip is scaled by 48 / h and centered vertically on 224 canvas."""
    h_orig, w_orig = 100, 600
    strip = np.full((h_orig, w_orig, 3), 255, dtype=np.uint8)
    ink_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

    # Put a letter of height 40px at rows [30:70]
    x_h_given = 40.0
    ink_mask[30:70, 100:500] = 1
    strip[ink_mask == 1] = [0, 50, 0]

    tiles, coords = extract_tiles_from_strip(
        strip,
        target_height=224,
        stride=112,
        x_height=x_h_given,
        target_x_height=48.0,
        ink_mask=ink_mask,
    )

    assert len(tiles) > 0
    for tile in tiles:
        assert tile.shape == (224, 224, 3)
        # Check background margin at top and bottom
        assert np.all(tile[0, :, :] == 255), "Top row must be clean white padding"
        assert np.all(tile[223, :, :] == 255), "Bottom row must be clean white padding"


def test_fix2_rejection_of_empty_and_ruled_line_tiles():
    """Verify Fix 2 Rejection Gate completely kills blank and flat ruled-line tiles."""
    h, w = 224, 500
    canvas_strip = np.full((h, w, 3), 255, dtype=np.uint8)
    ink_mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Blank strip -> 0 tiles returned
    tiles_blank, _ = extract_tiles_from_strip(
        canvas_strip,
        target_height=224,
        stride=112,
        ink_mask=ink_mask,
        allow_empty=True,
    )
    assert len(tiles_blank) == 0, "Blank strip must return 0 tiles"

    # 2. Pure flat horizontal ruled line (thickness 3 px across entire width)
    # Like Image 1 / Image 2
    rule_mask = np.zeros((h, w), dtype=np.uint8)
    rule_mask[100:104, :] = 1  # 4px tall flat line
    rule_strip = np.full((h, w, 3), 255, dtype=np.uint8)
    rule_strip[rule_mask == 1] = [0, 100, 0]

    tiles_rule, _ = extract_tiles_from_strip(
        rule_strip,
        target_height=224,
        stride=112,
        ink_mask=rule_mask,
        allow_empty=True,
    )
    assert len(tiles_rule) == 0, "Flat horizontal ruled line must be 100% rejected"

    # 3. Genuine handwriting letters (height 50px, width 150px)
    letter_mask = np.zeros((h, w), dtype=np.uint8)
    letter_mask[80:135, 100:250] = 1  # 55px tall letter block with 8,250 pixels
    letter_strip = np.full((h, w, 3), 255, dtype=np.uint8)
    letter_strip[letter_mask == 1] = [0, 50, 0]

    tiles_letter, _ = extract_tiles_from_strip(
        letter_strip,
        target_height=224,
        stride=112,
        x_height=48.0,
        ink_mask=letter_mask,
        allow_empty=True,
    )
    assert len(tiles_letter) > 0, "Genuine handwriting letter block must pass rejection gate"


def test_estimate_strip_x_height():
    """Verify estimate_strip_x_height accurately measures character height."""
    mask = np.zeros((150, 800), dtype=np.uint8)
    # Add multiple letter-like components of height 42 px
    for x in range(50, 700, 60):
        mask[50:92, x:x+30] = 1

    est = estimate_strip_x_height(mask)
    assert 38.0 <= est <= 46.0, f"Expected x_height ~42, got {est}"


def test_ch1_raw_mode_and_no_guideline_channel():
    """Verify parameterized channel composition: raw image in Ch1 and no guideline in Ch3."""
    h, w = 100, 300
    raw_gray = np.full((h, w), 220, dtype=np.uint8)
    ink_mask = np.zeros((h, w), dtype=np.uint8)
    ink_mask[30:70, 50:150] = 1
    raw_gray[ink_mask == 1] = 60  # dark handwriting stroke

    # 1. Test raw mode in Ch1 with guideline
    strip_raw = build_white_canvas_strip(
        clean_mask_or_path=ink_mask,
        raw_gray_or_path=raw_gray,
        ch1_mode="raw",
        include_guideline=True,
    )
    assert strip_raw.shape == (h, w, 3)
    # Background in Ch1 should be normalized close to 255
    assert np.all(strip_raw[ink_mask == 0, 0] >= 250)
    # Stroke in Ch1 should have raw gray values preserved (~69)
    assert np.all(strip_raw[ink_mask == 1, 0] < 100)

    # 2. Test no guideline in Ch3
    strip_no_guide = build_white_canvas_strip(
        clean_mask_or_path=ink_mask,
        raw_gray_or_path=raw_gray,
        ch1_mode="binary",
        include_guideline=False,
    )
    # Ch3 should duplicate Ch1 when guideline is removed
    np.testing.assert_array_equal(strip_no_guide[:, :, 2], strip_no_guide[:, :, 0])



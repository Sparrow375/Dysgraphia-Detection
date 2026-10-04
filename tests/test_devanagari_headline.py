"""
Unit Test: Devanagari Headline (Shirorekha) Preservation Test
============================================================
Tests that scripts with continuous horizontal top bars (such as Devanagari,
Bengali, Gurmukhi) do NOT have their top headlines mistakenly classified
and removed as paper ruled lines.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.branch_a.line_removal import detect_and_remove_ruled_lines


class TestDevanagariHeadlinePreservation(unittest.TestCase):
    def setUp(self):
        # 600 x 600 canvas (unruled paper background = 0, ink = 1)
        self.canvas = np.zeros((600, 600), dtype=np.uint8)

        # Draw a synthetic Devanagari word:
        # Continuous horizontal headline (shirorekha) at y = 200, length = 180px, thickness = 3px
        y_head = 200
        x_start, x_end = 150, 330
        self.canvas[y_head - 1:y_head + 2, x_start:x_end] = 1

        # Vertical letter stems hanging from the headline (e.g. 4 characters)
        stems = [170, 210, 260, 310]
        for sx in stems:
            # Vertical downward stroke (length 50px, thickness 3px)
            self.canvas[y_head:y_head + 50, sx - 1:sx + 2] = 1
            # Add small loops / curves at the bottom
            self.canvas[y_head + 48:y_head + 52, sx - 15:sx + 15] = 1

        self.initial_ink_count = int(np.sum(self.canvas))

    def test_shirorekha_is_not_removed_as_ruled_line(self):
        """
        Verify that detect_and_remove_ruled_lines preserves the headline
        and does not trigger false positive line detection.
        """
        clean_mask, meta = detect_and_remove_ruled_lines(self.canvas)

        # 1. Ruled paper should NOT be detected
        self.assertFalse(meta.get("ruled_paper_detected", False),
                         "False positive: Devanagari headline falsely identified as ruled notebook paper.")
        self.assertFalse(meta.get("grid_paper_detected", False),
                         "False positive: Devanagari text falsely identified as grid paper.")

        # 2. Check crossing / headline preservation
        # The entire headline should remain intact
        y_head = 200
        x_start, x_end = 150, 330
        headline_pixels = self.canvas[y_head - 1:y_head + 2, x_start:x_end]
        cleaned_headline_pixels = clean_mask[y_head - 1:y_head + 2, x_start:x_end]

        preserved_ratio = np.sum(cleaned_headline_pixels) / np.sum(headline_pixels)
        self.assertGreaterEqual(preserved_ratio, 0.95,
                                f"Shirorekha headline ink was degraded: only {preserved_ratio*100:.1f}% preserved.")

        # 3. Total ink preservation
        final_ink = int(np.sum(clean_mask))
        ink_preservation = final_ink / self.initial_ink_count
        self.assertGreaterEqual(ink_preservation, 0.98,
                                f"Overall text ink preservation below 98%: {ink_preservation*100:.1f}%.")


if __name__ == "__main__":
    unittest.main()

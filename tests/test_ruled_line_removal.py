"""
Test Suite: Ruled Notebook and Grid Line Detection & Removal
Validates:
  1. Detection and elimination of ruled notebook lines on lined paper
  2. Detection and elimination of vertical and horizontal grid lines on graph paper
  3. Strict zero false-positive removal on unlined paper (genuine cursive 't'-crossbars, dashes, word underlines)
  4. Preservation of intersecting handwriting strokes at crossing junctions
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.branch_a.line_removal import detect_and_remove_ruled_lines


def create_synthetic_handwriting_stroke(mask: np.ndarray, x0: int, y0: int, text_type: str = "cursive_t"):
    """Draws synthetic handwriting strokes onto a binary mask (1 = ink, 0 = bg)."""
    if text_type == "cursive_t":
        # Vertical stem of 't' from y0 to y0 + 50
        for y in range(y0, y0 + 51):
            mask[y, x0:x0+3] = 1
        # Horizontal crossbar of 't': length 35px, thickness 2px
        crossbar_y = y0 + 15
        mask[crossbar_y:crossbar_y+2, x0-15:x0+20] = 1
    elif text_type == "descender_g":
        # Descender loop crossing downward
        for y in range(y0, y0 + 70):
            mask[y, x0:x0+3] = 1
        for x in range(x0 - 20, x0 + 1):
            mask[y0 + 70, x:x+2] = 1
    elif text_type == "word_underline":
        # Short underline under a single word (length 110px, less than 10% of image width)
        mask[y0:y0+3, x0:x0+110] = 1


class TestRuledLineDetectionAndRemoval(unittest.TestCase):

    def test_ruled_lined_paper(self):
        """Case (a): Photos of ruled/lined paper."""
        h, w = 600, 1000
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw 5 horizontal ruled lines spanning 90% of page width (thickness 2px)
        rule_y_positions = [100, 180, 260, 340, 420]
        for ry in rule_y_positions:
            mask[ry:ry+2, 50:950] = 1

        # Draw genuine handwriting crossing the lines:
        # A cursive 't' on line 2 (crossbar at y=155, stem crossing rule at y=180)
        create_synthetic_handwriting_stroke(mask, x0=300, y0=140, text_type="cursive_t")
        # A descending stroke 'g' on line 3 crossing through rule at y=260
        create_synthetic_handwriting_stroke(mask, x0=500, y0=230, text_type="descender_g")

        cleaned, meta = detect_and_remove_ruled_lines(mask)

        # Assertions
        self.assertTrue(meta["ruled_paper_detected"], "Ruled paper should be detected")
        self.assertGreaterEqual(meta["horizontal_lines_count"], 4, "Should detect at least 4 horizontal ruled lines")
        self.assertEqual(meta["vertical_lines_count"], 0, "No vertical lines should be detected on lined paper")

        # Confirm cursive 't' crossbar is NOT eaten (crossbar at y=155, x=285..320)
        crossbar_pixels = cleaned[155, 285:320]
        self.assertGreater(np.sum(crossbar_pixels), 20, "Cursive 't' crossbar must not be eaten")

        # Confirm crossing handwriting stroke preserved at crossing intersection
        # Stem of 't' at y=180, x=300:
        self.assertEqual(cleaned[180, 301], 1, "Handwriting stem must be preserved across intersection")
        # Descender 'g' at y=260, x=501:
        self.assertEqual(cleaned[260, 501], 1, "Descender stroke must be preserved across intersection")

        # Confirm ruled lines outside handwriting are removed
        self.assertEqual(cleaned[100, 100], 0, "Ruled line pixel must be removed")
        self.assertEqual(cleaned[420, 800], 0, "Ruled line pixel must be removed")

    def test_unlined_paper(self):
        """Case (b): Unlined paper — ensure ZERO false-positive removals of genuine strokes."""
        h, w = 600, 1000
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw handwriting with multiple cursive 't's, dashes, and word underlines
        create_synthetic_handwriting_stroke(mask, x0=200, y0=150, text_type="cursive_t")
        create_synthetic_handwriting_stroke(mask, x0=450, y0=150, text_type="cursive_t")
        create_synthetic_handwriting_stroke(mask, x0=700, y0=150, text_type="cursive_t")
        create_synthetic_handwriting_stroke(mask, x0=300, y0=250, text_type="word_underline")

        cleaned, meta = detect_and_remove_ruled_lines(mask)

        # Assertions
        self.assertFalse(meta["ruled_paper_detected"], "Unlined paper must NOT be flagged as ruled")
        self.assertFalse(meta["grid_paper_detected"], "Unlined paper must NOT be flagged as grid")
        self.assertEqual(meta["horizontal_lines_count"], 0, "Zero false-positive horizontal lines detected")
        self.assertEqual(meta["vertical_lines_count"], 0, "Zero false-positive vertical lines detected")
        self.assertEqual(meta["total_line_pixels_removed"], 0, "Zero line pixels removed on unlined paper")

        # Verify all ink remains 100% untouched
        np.testing.assert_array_equal(cleaned, mask, "Cleaned mask must be strictly identical to original mask on unlined paper")

    def test_grid_graph_paper(self):
        """Case (c): Grid / Graph paper."""
        h, w = 600, 1000
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw grid: horizontal lines every 60px, vertical lines every 60px (thickness 1px)
        for ry in range(60, h - 50, 60):
            mask[ry, 50:950] = 1
        for rx in range(60, w - 50, 60):
            mask[50:550, rx] = 1

        # Draw handwriting letters in the grid
        create_synthetic_handwriting_stroke(mask, x0=310, y0=130, text_type="cursive_t")

        cleaned, meta = detect_and_remove_ruled_lines(mask, detect_grid=True)

        # Assertions
        self.assertTrue(meta["ruled_paper_detected"], "Ruled paper detected on grid")
        self.assertTrue(meta["grid_paper_detected"], "Grid paper detected")
        self.assertGreaterEqual(meta["horizontal_lines_count"], 5, "Horizontal grid lines detected")
        self.assertGreaterEqual(meta["vertical_lines_count"], 5, "Vertical grid lines detected")
        self.assertGreater(meta["total_line_pixels_removed"], 1000, "Grid lines successfully removed")

        # Cursive 't' crossbar still preserved
        crossbar_pixels = cleaned[145, 295:330]
        self.assertGreater(np.sum(crossbar_pixels), 15, "Handwriting preserved amidst grid removal")


if __name__ == "__main__":
    unittest.main()

"""
Standalone OCR Command-Line Tester.

Run with:
    python test_ocr.py                                   # Runs on default sample
    python test_ocr.py path/to/handwriting_image.jpg     # Runs on custom image

Prints detailed predictions:
  - Document-level text & overall confidence
  - Line-by-line word hypotheses
  - Letter-by-letter character predictions with alternatives & confidence
  - Topological stroke primitives detected
  - Context rescues (where language model altered visual prediction)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import argparse

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.ocr.pipeline import ContextAwareOCRPipeline
from src.ocr.utils import ConfidenceTier


def inspect_transcription(image_path: str, beam_width: int = 15):
    print("=" * 80)
    print(f"🧠 CONTEXT-AWARE OCR INSPECTOR: {image_path}")
    print("=" * 80)

    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return

    pipeline = ContextAwareOCRPipeline(beam_width=beam_width)
    print("⏳ Processing image (preprocessing → segmentation → stroke analysis → CRNN → LM re-rank)...")
    result = pipeline.transcribe(image_path)

    print("\n" + "─" * 80)
    print("📄 FULL DOCUMENT TRANSCRIPTION:")
    print("─" * 80)
    if result.full_text.strip():
        print(result.full_text)
    else:
        print("[No legible text detected]")

    print(f"\nOverall Document Confidence: {result.mean_confidence:.2%}")
    print(f"Total Lines Segmented:       {len(result.lines)}")
    print(f"Total Words Extracted:       {result.total_words}")

    # Detailed Line & Word Breakdown
    print("\n" + "─" * 80)
    print("🔍 WORD-BY-WORD & LETTER-BY-LETTER INSPECTION:")
    print("─" * 80)

    for line_idx, line in enumerate(result.lines):
        print(f"\n[Line {line_idx + 1}] (Confidence: {line.line_confidence:.2%}): \"{line.raw_text}\"")

        for w_idx, word in enumerate(line.words):
            tier_badge = {
                ConfidenceTier.HIGH: "🟢 [HIGH]",
                ConfidenceTier.MEDIUM: "🟡 [MEDIUM]",
                ConfidenceTier.LOW: "🔴 [LOW]",
                ConfidenceTier.VERY_LOW: "⚠️ [VERY LOW]",
            }.get(word.tier, "[UNKNOWN]")

            rescued_flag = " 🔄 (Rescued by Context!)" if word.metadata.get("context_rescued") else ""
            bbox_str = f"bbox={word.bbox}" if word.bbox else ""

            print(f"\n  ├─ Word {w_idx + 1}: \"{word.text}\" {tier_badge} (conf: {word.confidence:.2%}){rescued_flag} {bbox_str}")
            print(f"  │   Scores: Visual={word.visual_score:.2f} | Language={word.language_score:.2f} | Fused={word.fused_score:.2f}")

            # Topological stroke primitives
            stroke_analysis = word.metadata.get("stroke_analysis")
            if stroke_analysis and stroke_analysis.primitives:
                prim_names = [p.value for p in stroke_analysis.primitives]
                print(f"  │   Detected Stroke Primitives: {', '.join(prim_names)}")
                print(f"  │   Ascenders: {stroke_analysis.n_ascenders}, Descenders: {stroke_analysis.n_descenders}, Closed Loops: {stroke_analysis.n_loops}")

            # Letter-by-letter breakdown
            if word.char_hypotheses:
                print(f"  │   Letters Predicted ({len(word.char_hypotheses)} chars):")
                for c_idx, char_h in enumerate(word.char_hypotheses):
                    alt_str = ", ".join([f"'{c}': {p:.1%}" for c, p in char_h.alternatives[:3]])
                    alt_display = f" | alts: [{alt_str}]" if alt_str else ""
                    print(f"  │     Pos {c_idx + 1}: '{char_h.char}' (conf: {char_h.confidence:.1%}){alt_display}")

            # Alternative word candidates considered
            if word.alternatives:
                top_alts = [f"\"{a.text}\" ({a.visual_score:.2f})" for a in word.alternatives[:4]]
                print(f"  │   Alternative Word Candidates: {', '.join(top_alts)}")

    # OCR Clinical Dysgraphia Features
    ocr_feats = result.metadata.get("ocr_dysgraphia_features")
    if ocr_feats:
        print("\n" + "─" * 80)
        print("📊 OCR-DERIVED DYSGRAPHIA DIAGNOSTIC SIGNALS:")
        print("─" * 80)
        print(f"  • Mean Word Confidence:          {ocr_feats.mean_word_confidence:.2%} (Overall legibility)")
        print(f"  • Fraction Low-Confidence Words: {ocr_feats.fraction_low_confidence_words:.2%} (Severe distortion)")
        print(f"  • Word Confidence Variance:      {ocr_feats.word_confidence_variance:.4f} (Consistency)")
        print(f"  • Context Rescue Rate:           {ocr_feats.context_rescue_rate:.2%} (Reliance on context)")
        print(f"  • Mean Stroke Agreement:         {ocr_feats.mean_stroke_agreement:.2%} (Stroke regularity)")
        print(f"  • Visual vs LM Disagreement:     {ocr_feats.visual_language_disagreement:.2%}")

    print("\n" + "=" * 80)
    print("✅ Inspection complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Standalone OCR Tester with Word & Letter Inspection")
    parser.add_argument("image", nargs="?", default=None, help="Path to handwriting image file")
    parser.add_argument("--beam-width", type=int, default=15, help="Beam search width")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        # Check default candidate images
        candidates = [
            "scraped_candidates/images/ENG_CAND_058.jpg",
            "scraped_candidates/images/ENG_CAND_023.jpg",
            "DATASET DYSGRAPHIA HANDWRITING/Potential Dysgraphia/PD (1).jpg",
            "DATASET DYSGRAPHIA HANDWRITING/Low Potential Dysgraphia/LPD (1).jpg",
        ]
        for c in candidates:
            if os.path.exists(c):
                image_path = c
                break

    if not image_path or not os.path.exists(image_path):
        print("⚠️ No input image specified and default candidates not found.")
        print("Usage: python test_ocr.py <path_to_image>")
        return

    inspect_transcription(image_path, beam_width=args.beam_width)


if __name__ == "__main__":
    main()

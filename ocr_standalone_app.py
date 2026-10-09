"""
Standalone Interactive OCR Web Application (Word & Letter Inspector).

Run with:
    python ocr_standalone_app.py

Provides a dedicated, rich web UI specifically focused on:
  1. Full-page transcription with confidence color codes
  2. Interactive word-by-word inspector showing cropped word ink patches
  3. Letter-by-letter character predictions with alternatives & confidence
  4. Stroke topology breakdown (ascenders, descenders, loops)
  5. Alternative word lattice showing beam search candidates
  6. Context rescue tracking (where language model assisted bad handwriting)
"""

from __future__ import annotations

import os
import sys
import json
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import cv2
import gradio as gr

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.ocr.pipeline import ContextAwareOCRPipeline
from src.ocr.utils import ConfidenceTier, TranscriptionResult, WordHypothesis

# Global pipeline instance (cached)
_PIPELINE: Optional[ContextAwareOCRPipeline] = None
_CURRENT_TRANSCRIPTION: Optional[TranscriptionResult] = None
_CURRENT_IMAGE_BGR: Optional[np.ndarray] = None


def get_pipeline(beam_width: int = 15, stroke_weight: float = 0.15) -> ContextAwareOCRPipeline:
    global _PIPELINE
    ckpt_path = "models/transformer_ocr/checkpoint_best.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "models/crnn_iam/checkpoint_best.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = None

    if _PIPELINE is None:
        _PIPELINE = ContextAwareOCRPipeline(
            crnn_model_path=ckpt_path,
            use_neural_lm=False,
            beam_width=beam_width,
            stroke_weight=stroke_weight,
            backend="transformer",
        )
    else:
        _PIPELINE.word_recognizer.beam_width = beam_width
        _PIPELINE.word_recognizer.stroke_weight = stroke_weight

    return _PIPELINE


def process_image(image_input, prompt_text: str = "", beam_width: int = 15, stroke_weight: float = 0.15):
    """
    Main transcription callback with optional prompt-guided forced alignment.
    """
    global _CURRENT_TRANSCRIPTION, _CURRENT_IMAGE_BGR

    placeholder_badge = "<div style='color:#64748b; padding:10px;'><em>Enter a target copy sentence on the left to activate diagnostic reversal and omission analysis.</em></div>"
    placeholder_summary = "*(Optional) Enter the target prompt sentence the student was copying to enable clinical letter-reversal ($b \\leftrightarrow d$, $p \\leftrightarrow q$) detection.*"
    placeholder_grid = ""

    if image_input is None:
        return (
            "<p style='color:#64748b;'>⚠️ Please upload or select a handwriting image.</p>",
            None,
            gr.Dropdown(choices=[], value=None),
            None,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            placeholder_badge,
            placeholder_summary,
            placeholder_grid,
        )

    # Convert RGB input to BGR
    if isinstance(image_input, np.ndarray):
        bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if len(image_input.shape) == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.imread(str(image_input))

    _CURRENT_IMAGE_BGR = bgr

    pipeline = get_pipeline(beam_width=int(beam_width), stroke_weight=float(stroke_weight))
    transcription, copy_task_diag = pipeline.transcribe_with_prompt(bgr, prompt_text=prompt_text)
    _CURRENT_TRANSCRIPTION = transcription

    # 1. HTML Transcription Badges
    html_lines = []
    word_choices = []

    for li, line in enumerate(transcription.lines):
        word_spans = []
        for wi, w in enumerate(line.words):
            tier = w.tier
            if tier == ConfidenceTier.HIGH:
                bg, text_col, border = "#ecfdf5", "#047857", "#10b981"
            elif tier == ConfidenceTier.MEDIUM:
                bg, text_col, border = "#fef3c7", "#b45309", "#f59e0b"
            elif tier == ConfidenceTier.LOW:
                bg, text_col, border = "#fee2e2", "#b91c1c", "#ef4444"
            else:
                bg, text_col, border = "#fef2f2", "#7f1d1d", "#991b1b"

            rescued_icon = " 🔄" if w.metadata.get("context_rescued") else ""
            span = (
                f"<span style='display:inline-block; margin:3px 4px; padding:4px 8px; "
                f"background:{bg}; color:{text_col}; border:1px solid {border}; border-radius:5px; "
                f"font-weight:600; font-size:15px;' title='Word {wi+1} (Line {li+1}): {w.confidence:.1%}{rescued_icon}'>"
                f"{w.text}{rescued_icon} "
                f"<small style='font-size:11px; opacity:0.8;'>({w.confidence:.0%})</small></span>"
            )
            word_spans.append(span)

            # Add to word choices dropdown
            choice_label = f"Line {li+1}, Word {wi+1}: '{w.text}' ({w.confidence:.0%})"
            word_choices.append(choice_label)

        html_lines.append("<div style='margin-bottom:8px;'>" + "".join(word_spans) + "</div>")

    rendered_html = (
        "<div style='background:#f8fafc; padding:16px; border-radius:8px; border:1px solid #e2e8f0; line-height:2.0;'>"
        + ("".join(html_lines) if html_lines else "<em>No text detected.</em>")
        + "</div>"
    )

    # 2. Render Overlay Image
    overlay_bgr = pipeline.render_overlay(bgr, transcription)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # 3. Overall Word Summary Dataframe
    summary_rows = []
    for li, line in enumerate(transcription.lines):
        for wi, w in enumerate(line.words):
            alts_str = ", ".join(a.text for a in w.alternatives[:3]) if w.alternatives else "—"
            chars_predicted = "".join([c.char for c in w.char_hypotheses]) if w.char_hypotheses else w.text
            summary_rows.append({
                "Line": li + 1,
                "Position": wi + 1,
                "Decoded Word": w.text,
                "Letters Sequence": chars_predicted,
                "Confidence": f"{w.confidence:.1%}",
                "Tier": w.tier.name,
                "Context Rescued": "✅ Yes (LM adapted)" if w.metadata.get("context_rescued") else "No",
                "Alternatives": alts_str,
            })
    df_summary = pd.DataFrame(summary_rows)

    # 4. Clinical Metrics
    ocr_feats = transcription.metadata.get("ocr_dysgraphia_features")
    metrics_rows = []
    if ocr_feats:
        metrics_rows = [
            {"Signal": "Mean Word Confidence", "Value": f"{ocr_feats.mean_word_confidence:.2%}", "Clinical Meaning": "Overall legibility index"},
            {"Signal": "Fraction Low Confidence", "Value": f"{ocr_feats.fraction_low_confidence_words:.2%}", "Clinical Meaning": "Proportion of severely degraded words"},
            {"Signal": "Confidence Variance", "Value": f"{ocr_feats.word_confidence_variance:.4f}", "Clinical Meaning": "Inconsistency in legibility across page"},
            {"Signal": "Context Rescue Rate", "Value": f"{ocr_feats.context_rescue_rate:.2%}", "Clinical Meaning": "Words salvaged via sentence context"},
            {"Signal": "Mean Stroke Agreement", "Value": f"{ocr_feats.mean_stroke_agreement:.2%}", "Clinical Meaning": "Topological skeleton primitive agreement"},
            {"Signal": "Visual vs LM Disagreement", "Value": f"{ocr_feats.visual_language_disagreement:.2%}", "Clinical Meaning": "Conflict between visual strokes and grammatical syntax"},
        ]
    df_metrics = pd.DataFrame(metrics_rows)

    # 5. Diagnostic Copy-Task Outputs
    if copy_task_diag is not None:
        diag_badge_html = copy_task_diag.diagnostic_badge_html
        diag_summary_md = copy_task_diag.diagnostic_summary
        diag_grid_html = copy_task_diag.alignment_grid_html
    else:
        diag_badge_html = placeholder_badge
        diag_summary_md = placeholder_summary
        diag_grid_html = placeholder_grid

    # Select first word by default for inspector
    initial_choice = word_choices[0] if word_choices else None
    crop_img, df_chars, df_alts, df_strokes, word_info_text = inspect_selected_word(initial_choice)

    return (
        rendered_html,
        overlay_rgb,
        gr.Dropdown(choices=word_choices, value=initial_choice),
        crop_img,
        df_chars,
        df_alts,
        df_strokes,
        df_metrics,
        word_info_text,
        diag_badge_html,
        diag_summary_md,
        diag_grid_html,
    )


def inspect_selected_word(selected_choice: Optional[str]):
    """
    Called when a user picks a specific word from the dropdown to inspect.
    """
    global _CURRENT_TRANSCRIPTION, _CURRENT_IMAGE_BGR

    empty_df = pd.DataFrame()
    if not selected_choice or _CURRENT_TRANSCRIPTION is None or _CURRENT_IMAGE_BGR is None:
        return None, empty_df, empty_df, empty_df, "Select a word to inspect its letter predictions."

    # Parse line and word index: "Line 1, Word 2: 'quick' (41%)"
    try:
        parts = selected_choice.split(":")
        coords = parts[0].replace("Line", "").replace("Word", "").strip().split(",")
        line_idx = int(coords[0].strip()) - 1
        word_idx = int(coords[1].strip()) - 1
    except Exception:
        return None, empty_df, empty_df, empty_df, "Error parsing selected word coordinates."

    if line_idx >= len(_CURRENT_TRANSCRIPTION.lines) or word_idx >= len(_CURRENT_TRANSCRIPTION.lines[line_idx].words):
        return None, empty_df, empty_df, empty_df, "Word index out of bounds."

    word: WordHypothesis = _CURRENT_TRANSCRIPTION.lines[line_idx].words[word_idx]

    # 1. Crop image patch
    crop_rgb = None
    if word.bbox:
        x, y, w, h = word.bbox
        img_h, img_w = _CURRENT_IMAGE_BGR.shape[:2]
        # Pad slightly
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        cropped = _CURRENT_IMAGE_BGR[y1:y2, x1:x2]
        if cropped.size > 0:
            crop_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # 2. Letter-by-Letter Breakdown Table
    char_rows = []
    if word.char_hypotheses:
        for idx, ch in enumerate(word.char_hypotheses):
            alt_str = ", ".join([f"'{c}' ({p:.1%})" for c, p in ch.alternatives[:4]]) if ch.alternatives else "—"
            char_rows.append({
                "Position": idx + 1,
                "Predicted Letter": ch.char,
                "Letter Confidence": f"{ch.confidence:.1%}",
                "Alternative Candidates": alt_str,
            })
    else:
        # Fallback if character hypotheses not generated
        for idx, ch in enumerate(word.text):
            char_rows.append({
                "Position": idx + 1,
                "Predicted Letter": ch,
                "Letter Confidence": f"{word.confidence:.1%}",
                "Alternative Candidates": "—",
            })
    df_chars = pd.DataFrame(char_rows)

    # 3. Alternative Word Lattice Table
    alt_rows = [{
        "Rank": 1,
        "Candidate Word": word.text,
        "Visual Score": f"{word.visual_score:.2f}",
        "Language Score": f"{word.language_score:.2f}",
        "Fused Score": f"{word.fused_score:.2f}",
        "Selected": "⭐ Top Pick",
    }]
    if word.alternatives:
        for rank, alt in enumerate(word.alternatives[:5], start=2):
            alt_rows.append({
                "Rank": rank,
                "Candidate Word": alt.text,
                "Visual Score": f"{alt.visual_score:.2f}",
                "Language Score": f"{alt.language_score:.2f}",
                "Fused Score": f"{alt.fused_score:.2f}",
                "Selected": "Alternative",
            })
    df_alts = pd.DataFrame(alt_rows)

    # 4. Stroke Topology Primitives Table
    stroke_analysis = word.metadata.get("stroke_analysis")
    stroke_rows = []
    if stroke_analysis:
        prim_counts: Dict[str, int] = {}
        for p in stroke_analysis.primitives:
            prim_counts[p.value] = prim_counts.get(p.value, 0) + 1

        for name, count in prim_counts.items():
            stroke_rows.append({
                "Detected Primitive": name.replace("_", " ").title(),
                "Count": count,
                "Key Characters": "Ascenders (b, d, h, k, l, t)" if "ascender" in name else (
                    "Descenders (g, j, p, q, y)" if "descender" in name else (
                        "Loops (a, b, d, e, g, o, p, q)" if "loop" in name else "Crossings (t, f, x)"
                    )
                )
            })
    df_strokes = pd.DataFrame(stroke_rows)

    # 5. Header Summary Info
    rescue_text = "🔄 **Rescued by Context:** Visual recognizer was uncertain; sentence language model boosted this candidate." if word.metadata.get("context_rescued") else "Normal Visual Prediction."
    info_text = (
        f"### 🔎 Selected: **\"{word.text}\"**\n"
        f"- **Confidence:** {word.confidence:.1%} ({word.tier.name})\n"
        f"- **Visual Log-Likelihood:** {word.visual_score:.2f} | **Language Log-Likelihood:** {word.language_score:.2f}\n"
        f"- **Context Status:** {rescue_text}\n"
    )

    return crop_rgb, df_chars, df_alts, df_strokes, info_text


# Find candidate sample images for quick testing
sample_examples = []
candidates = [
    "scraped_candidates/images/ENG_CAND_058.jpg",
    "scraped_candidates/images/ENG_CAND_023.jpg",
    "scraped_candidates/images/ENG_CAND_015.jpg",
    "DATASET DYSGRAPHIA HANDWRITING/Potential Dysgraphia/PD (1).jpg",
    "DATASET DYSGRAPHIA HANDWRITING/Low Potential Dysgraphia/LPD (1).jpg",
    "reconstructed_dataset/full_page/dysgraphic/user_00006_full.png",
]
for c in candidates:
    if os.path.exists(c):
        sample_examples.append(c)


# Gradio App Layout
custom_css = """
.gradio-container { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.header-box { text-align: center; margin-bottom: 20px; padding: 24px; background: linear-gradient(135deg, #0f172a, #1e293b); border-radius: 12px; color: white; border: 1px solid #334155; }
.header-box h1 { margin: 0 0 6px 0; font-size: 26px; font-weight: 700; color: #38bdf8; }
.header-box p { margin: 0; color: #94a3b8; font-size: 14px; }
.inspector-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
"""

with gr.Blocks(title="Context-Aware OCR — Word & Character Inspector") as demo:
    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            "<h1>🧠 Context-Aware Dysgraphia OCR — Word & Letter Inspector</h1>"
            "<p>Inspect step-by-step how the engine reads bad handwriting: word image segmentation, stroke topology, letter-level probabilities, and language context re-ranking.</p>"
            "<div style='margin-top:10px;'><span style='background:rgba(56,189,248,0.15); color:#38bdf8; border:1px solid #0284c7; padding:4px 12px; border-radius:16px; font-size:12px; font-weight:600;'>⚡ Active Engine: IAM CRNN (28.7M Params • Residual CNN + BiLSTM • Val CER 6.96% • Accuracy 83.8%)</span></div>"
        )

    with gr.Row():
        # LEFT COLUMN: Image Upload & Parameters
        with gr.Column(scale=4):
            gr.Markdown("### 📤 Handwriting Image Input")
            img_input = gr.Image(
                type="numpy",
                label="Upload Photo / Scan of Handwriting",
                sources=["upload", "clipboard"]
            )

            prompt_input = gr.Textbox(
                label="🎯 Target Copy Sentence (Optional — Diagnostic Mode)",
                placeholder="e.g. The quick brown fox jumps over the lazy dog",
                lines=2,
            )
            with gr.Row():
                btn_p1 = gr.Button("📋 Pangram", size="sm")
                btn_p2 = gr.Button("📋 BHK Prompt", size="sm")
                btn_p_clear = gr.Button("✕ Clear", size="sm")

            with gr.Accordion("⚙️ Decoding Engine Settings", open=False):
                beam_width_slider = gr.Slider(
                    minimum=5, maximum=40, value=15, step=1,
                    label="CTC Beam Width (Hypothesis Lattice Breadth)"
                )
                stroke_weight_slider = gr.Slider(
                    minimum=0.0, maximum=0.5, value=0.15, step=0.05,
                    label="Stroke Prior Weight α (Topological Biasing)"
                )

            transcribe_btn = gr.Button("🔍 Transcribe & Inspect Predictions", variant="primary", size="lg")

            if sample_examples:
                gr.Markdown("#### 📁 Quick Test Examples")
                gr.Examples(examples=sample_examples, inputs=img_input)

        # RIGHT COLUMN: Full Output & Overview
        with gr.Column(scale=6):
            gr.Markdown("### 📝 Full Transcription Output")
            full_transcription_html = gr.HTML(
                value="<p style='color:#64748b;'>Upload an image to start decoding.</p>",
                label="Decoded Text with Word Confidence Badges"
            )

            with gr.Tabs():
                with gr.TabItem("🎯 Copy-Task Diagnostic (Reversals)"):
                    diag_badges_html = gr.HTML(
                        value="<div style='color:#64748b; padding:10px;'><em>Enter a target copy sentence on the left to activate letter reversal, omission, and fidelity tracking.</em></div>",
                        label="Clinical Diagnostic Badges"
                    )
                    diag_summary_md = gr.Markdown(value="*(Optional) Enter the target prompt sentence the student was copying to enable clinical letter-reversal ($b \\leftrightarrow d$, $p \\leftrightarrow q$) detection.*")
                    gr.Markdown("#### 🔍 Character-by-Character Alignment & Error Heatmap")
                    diag_grid_html = gr.HTML(value="")

                with gr.TabItem("🗺️ Word Confidence Overlay"):
                    overlay_image = gr.Image(
                        label="Overlay (Green=High Conf, Yellow=Medium, Red=Low/Distorted)",
                        type="numpy"
                    )
                with gr.TabItem("📋 All Words Summary"):
                    summary_table = gr.Dataframe(
                        headers=["Line", "Position", "Decoded Word", "Letters Sequence", "Confidence", "Tier", "Context Rescued", "Alternatives"],
                        interactive=False,
                        label="Document Words Breakdown"
                    )
                with gr.TabItem("📈 Diagnostic Signals"):
                    metrics_table = gr.Dataframe(
                        headers=["Signal", "Value", "Clinical Meaning"],
                        interactive=False,
                        label="OCR Legibility & Clinical Metrics"
                    )

    # BOTTOM SECTION: DEEP DIVE WORD & LETTER INSPECTOR
    gr.Markdown("---")
    gr.Markdown("## 🔬 Word & Letter Deep Dive Inspector")
    gr.Markdown("Select any word from the document below to inspect its cropped image patch, letter-by-letter softmax distribution, and stroke primitives.")

    with gr.Row():
        with gr.Column(scale=4):
            word_dropdown = gr.Dropdown(
                choices=[],
                label="👉 Select Word to Inspect",
                interactive=True
            )
            word_info_markdown = gr.Markdown(value="Select a word above to view details.")
            word_crop_image = gr.Image(
                label="Cropped Word Ink Patch",
                type="numpy"
            )

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.TabItem("🔤 Letter-by-Letter Breakdown"):
                    gr.Markdown("Shows what **characters / letters** the model predicts at each position, along with alternative letters and confidence scores.")
                    char_breakdown_table = gr.Dataframe(
                        headers=["Position", "Predicted Letter", "Letter Confidence", "Alternative Candidates"],
                        interactive=False,
                        label="Character Softmax Hypotheses"
                    )

                with gr.TabItem("🧬 Topological Stroke Primitives"):
                    gr.Markdown("Structural stroke primitives detected from the skeleton that survive dysgraphic distortion.")
                    strokes_table = gr.Dataframe(
                        headers=["Detected Primitive", "Count", "Key Characters"],
                        interactive=False,
                        label="Detected Stroke Anatomy"
                    )

                with gr.TabItem("💡 Word Candidates (Lattice)"):
                    gr.Markdown("Top competing word hypotheses evaluated by the beam search and language model.")
                    alts_table = gr.Dataframe(
                        headers=["Rank", "Candidate Word", "Visual Score", "Language Score", "Fused Score", "Selected"],
                        interactive=False,
                        label="Beam Search Candidates"
                    )

    # Preset button handlers
    btn_p1.click(lambda: "The quick brown fox jumps over the lazy dog", outputs=prompt_input)
    btn_p2.click(lambda: "A quick movement of the enemy will jeopardize six gunboats", outputs=prompt_input)
    btn_p_clear.click(lambda: "", outputs=prompt_input)

    # Event Handlers
    transcribe_btn.click(
        fn=process_image,
        inputs=[img_input, prompt_input, beam_width_slider, stroke_weight_slider],
        outputs=[
            full_transcription_html,
            overlay_image,
            word_dropdown,
            word_crop_image,
            char_breakdown_table,
            alts_table,
            strokes_table,
            metrics_table,
            word_info_markdown,
            diag_badges_html,
            diag_summary_md,
            diag_grid_html,
        ]
    )

    img_input.change(
        fn=process_image,
        inputs=[img_input, prompt_input, beam_width_slider, stroke_weight_slider],
        outputs=[
            full_transcription_html,
            overlay_image,
            word_dropdown,
            word_crop_image,
            char_breakdown_table,
            alts_table,
            strokes_table,
            metrics_table,
            word_info_markdown,
            diag_badges_html,
            diag_summary_md,
            diag_grid_html,
        ]
    )

    word_dropdown.change(
        fn=inspect_selected_word,
        inputs=[word_dropdown],
        outputs=[
            word_crop_image,
            char_breakdown_table,
            alts_table,
            strokes_table,
            word_info_markdown
        ]
    )


if __name__ == "__main__":
    print("🚀 Starting Standalone Context-Aware OCR Inspector on http://127.0.0.1:7861 ...")
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, css=custom_css)

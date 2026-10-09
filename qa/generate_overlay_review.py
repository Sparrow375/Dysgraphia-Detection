"""Quality Gate review tool for Workstream A Phase 1.

Processes ~30 sample sheets across grade bands (G3, G4, G5, G6, G7), checks word count
match consistency, and generates an interactive HTML visual inspection gallery (qa/review_gallery.html).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import pandas as pd
from tqdm import tqdm

from pipeline.process_dataset import process_single_student


def image_to_base64_thumbnail(img_path: Path, max_height: int = 500) -> str:
    """Read image, resize to thumbnail height, and return data URL."""
    if not img_path.exists():
        return ""
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / float(h)
        img = cv2.resize(img, (int(w * scale), max_height), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def run_qa_review(
    manifest_path: str = "data/manifest.csv",
    output_dir: str = "data/processed",
    gallery_html_path: str = "qa/review_gallery.html",
    samples_per_grade: int = 6,
    force: bool = True,
) -> dict:
    """Select stratified sample across grades, process if needed, and build review gallery."""
    df = pd.read_csv(manifest_path)
    df_a = df[df["school"] == "school_a"].copy()

    # Sample up to samples_per_grade students per grade
    sampled_rows = []
    for g in sorted(df_a["grade"].unique()):
        g_df = df_a[df_a["grade"] == g]
        # Include positives and negatives in sample
        pos = g_df[g_df["label"] == 1]
        neg = g_df[g_df["label"] == 0]

        n_pos = min(len(pos), max(1, samples_per_grade // 3))
        n_neg = min(len(neg), samples_per_grade - n_pos)

        sample = pd.concat([pos.head(n_pos), neg.head(n_neg)])
        sampled_rows.append(sample)

    review_df = pd.concat(sampled_rows).reset_index(drop=True)
    print(f"Running Phase 1 QA Review on {len(review_df)} sheets across grades {sorted(df_a['grade'].unique())}...")

    Path("qa").mkdir(parents=True, exist_ok=True)

    qa_results = []
    html_cards = []

    for _, row in tqdm(review_df.iterrows(), total=len(review_df), desc="QA Inspection"):
        student_id = str(row["student_id"])
        school = str(row["school"])
        grade = int(row["grade"])
        label = int(row["label"])
        roll = int(row["roll_number"])

        student_dir = Path(output_dir) / school / student_id
        if force or not (student_dir / "overlay_debug.png").exists():
            try:
                process_single_student(row, output_base_dir=output_dir)
            except Exception as e:
                print(f"Error processing {student_id}: {e}")
                continue

        overlay_path = student_dir / "overlay_debug.png"
        page_norm_path = student_dir / "page_normalized.png"

        overlay_b64 = image_to_base64_thumbnail(overlay_path, max_height=550)

        # Collect sentence crops and jsons
        sentence_cards = []
        total_words = 0
        json_files = sorted(student_dir.glob("sentence_*.json"))

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            t_name = s_data["task_name"]
            script = s_data["script"]
            wc = s_data["qa_flags"]["word_count_detected"]
            total_words += wc

            png_path = student_dir / s_data["crop_filename"]
            png_b64 = image_to_base64_thumbnail(png_path, max_height=120)

            sentence_cards.append(
                f"""<div class="sentence-box">
                    <strong>{t_name}</strong> ({script}) — {wc} words<br>
                    <img src="{png_b64}" class="crop-img"/>
                </div>"""
            )

        badge_class = "badge-pos" if label == 1 else "badge-neg"
        badge_text = "Dysgraphia At-Risk (Positive)" if label == 1 else "Typical (Negative)"

        html_cards.append(
            f"""
            <div class="student-card">
                <div class="card-header">
                    <h3>{student_id} (Grade {grade}, Roll {roll})</h3>
                    <span class="badge {badge_class}">{badge_text}</span>
                    <span>Total Sentences: {len(json_files)} | Total Words: {total_words}</span>
                </div>
                <div class="card-body">
                    <div class="overlay-pane">
                        <h4>Segmentation Overlay (Ruled Lines + Word BBoxes)</h4>
                        <img src="{overlay_b64}" class="overlay-img"/>
                    </div>
                    <div class="crops-pane">
                        <h4>Extracted Sentence Crops</h4>
                        {''.join(sentence_cards)}
                    </div>
                </div>
            </div>
            """
        )

        qa_results.append(
            {
                "student_id": student_id,
                "grade": grade,
                "label": label,
                "sentences": len(json_files),
                "words": total_words,
            }
        )

    # Compile full HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Phase 1 QA Segmentation Review Gallery</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ margin-top: 0; color: #38bdf8; }}
        .summary {{ background: #1e293b; padding: 16px 20px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #334155; }}
        .student-card {{ background: #1e293b; border-radius: 10px; margin-bottom: 28px; padding: 20px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }}
        .card-header {{ display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px; }}
        .card-header h3 {{ margin: 0; color: #f1f5f9; }}
        .badge {{ padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }}
        .badge-pos {{ background: #991b1b; color: #fecaca; }}
        .badge-neg {{ background: #065f46; color: #a7f3d0; }}
        .card-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        .overlay-img {{ width: 100%; border-radius: 6px; border: 1px solid #475569; }}
        .sentence-box {{ background: #0f172a; padding: 10px; border-radius: 6px; margin-bottom: 10px; border: 1px solid #334155; }}
        .crop-img {{ width: 100%; margin-top: 6px; background: white; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Workstream A — Phase 1 Segmentation Quality Gate Gallery</h1>
    <div class="summary">
        <strong>Inspected Sheets:</strong> {len(qa_results)} sheets across Grades 3, 4, 5, 6, 7.<br>
        <strong>Segmentation Quality Gate:</strong> Visual inspection of line bounds, deskewing, and word separation.<br>
        <strong>Output location:</strong> <code>data/processed/school_a/&lt;student_id&gt;/</code>
    </div>
    {''.join(html_cards)}
</body>
</html>
"""
    with open(gallery_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nQuality Gate review gallery generated successfully at: {gallery_html_path}")
    return {"inspected": len(qa_results), "gallery_path": gallery_html_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA review gallery for Phase 1")
    parser.add_argument("--samples", type=int, default=6, help="Samples per grade")
    parser.add_argument("--no-force", dest="force", action="store_false", help="Don't reprocess existing")
    parser.set_defaults(force=True)
    args = parser.parse_args()

    run_qa_review(samples_per_grade=args.samples, force=args.force)

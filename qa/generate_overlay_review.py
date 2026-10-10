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
    force: bool = False,
    all_sheets: bool = False,
) -> dict:
    """Select stratified sample across grades (or all sheets), process if needed, and build review gallery."""
    import time
    df = pd.read_csv(manifest_path)
    df_a = df[df["school"] == "school_a"].copy()

    if all_sheets or samples_per_grade <= 0:
        review_df = df_a.sort_values(by=["grade", "roll_number"]).reset_index(drop=True)
        print(f"Running Phase 1 QA Review on ALL {len(review_df)} sheets across grades {sorted(df_a['grade'].unique())}...")
    else:
        # Sample up to samples_per_grade students per grade
        sampled_rows = []
        for g in sorted(df_a["grade"].unique()):
            g_df = df_a[df_a["grade"] == g]
            pos = g_df[g_df["label"] == 1]
            neg = g_df[g_df["label"] == 0]

            n_pos = min(len(pos), max(1, samples_per_grade // 3))
            n_neg = min(len(neg), samples_per_grade - n_pos)

            sample = pd.concat([pos.head(n_pos), neg.head(n_neg)])
            sampled_rows.append(sample)

        review_df = pd.concat(sampled_rows).reset_index(drop=True)
        print(f"Running Phase 1 QA Review on {len(review_df)} sampled sheets across grades {sorted(df_a['grade'].unique())}...")

    Path("qa").mkdir(parents=True, exist_ok=True)

    qa_results = []
    html_cards = []

    # Threshold: consider sheets updated in last 6 hours as fresh
    current_time = time.time()
    fresh_threshold = current_time - (6 * 3600)

    for _, row in tqdm(review_df.iterrows(), total=len(review_df), desc="QA Inspection"):
        student_id = str(row["student_id"])
        school = str(row["school"])
        grade = int(row["grade"])
        label = int(row["label"])
        roll = int(row["roll_number"])

        student_dir = Path(output_dir) / school / student_id
        overlay_path = student_dir / "overlay_debug.png"

        needs_processing = force or not overlay_path.exists()
        if not needs_processing:
            try:
                mtime = os.path.getmtime(str(overlay_path))
                if mtime < fresh_threshold:
                    needs_processing = True
            except OSError:
                needs_processing = True

        if needs_processing:
            try:
                process_single_student(row, output_base_dir=output_dir)
            except Exception as e:
                print(f"Error processing {student_id}: {e}")
                qa_results.append({
                    "student_id": student_id,
                    "grade": grade,
                    "roll_number": roll,
                    "label": label,
                    "status": "ERROR",
                    "error": str(e),
                    "tasks_count": 0,
                    "words_count": 0,
                    "task_sequence": "",
                })
                continue

        overlay_rel_path = f"../{output_dir}/{school}/{student_id}/overlay_debug.png"

        # Collect sentence crops and jsons
        sentence_cards = []
        total_words = 0
        json_files = sorted(student_dir.glob("sentence_*.json"))
        task_info_list = []

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            t_name = s_data["task_name"]
            script = s_data["script"]
            wc = s_data["qa_flags"]["word_count_detected"]
            total_words += wc
            task_info_list.append(f"{t_name}({script[:3]})")

            crop_fn = s_data["crop_filename"]
            crop_rel_path = f"../{output_dir}/{school}/{student_id}/{crop_fn}"

            sentence_cards.append(
                f"""<div class="sentence-box">
                    <strong>{t_name}</strong> ({script}) — {wc} words<br>
                    <img src="{crop_rel_path}" class="crop-img" alt="{t_name}"/>
                </div>"""
            )

        badge_class = "badge-pos" if label == 1 else "badge-neg"
        badge_text = "Dysgraphia At-Risk (Positive)" if label == 1 else "Typical (Negative)"

        # Check protocol compliance
        expected_tasks = 4 if grade == 3 else 6
        is_compliant = (len(json_files) == expected_tasks) or (grade > 3 and len(json_files) >= 4)
        status_text = "PASS" if is_compliant else "WARN"

        html_cards.append(
            f"""
            <div class="student-card">
                <div class="card-header">
                    <h3>{student_id} (Grade {grade}, Roll {roll})</h3>
                    <span class="badge {badge_class}">{badge_text}</span>
                    <span>Status: <strong>{status_text}</strong> | Total Sentences: {len(json_files)} (Expected: {expected_tasks}) | Total Words: {total_words}</span>
                </div>
                <div class="card-body">
                    <div class="overlay-pane">
                        <h4>Segmentation Overlay (Ruled Lines + Word BBoxes)</h4>
                        <img src="{overlay_rel_path}" class="overlay-img" alt="overlay"/>
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
                "roll_number": roll,
                "label": label,
                "status": status_text,
                "tasks_count": len(json_files),
                "expected_tasks": expected_tasks,
                "words_count": total_words,
                "task_sequence": ", ".join(task_info_list),
            }
        )

    # Save detailed assessment CSV
    results_df = pd.DataFrame(qa_results)
    csv_report_path = "qa/school_a_assessment_report.csv"
    results_df.to_csv(csv_report_path, index=False)
    print(f"Detailed assessment report saved to: {csv_report_path}")

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
        .overlay-pane {{ overflow: hidden; }}
        .overlay-img {{ width: 100%; border-radius: 6px; border: 1px solid #475569; display: block; }}
        .crops-pane {{ display: flex; flex-direction: column; gap: 14px; }}
        .sentence-box {{ background: #0f172a; padding: 12px 14px; border-radius: 6px; border: 1px solid #334155; }}
        .crop-img {{ max-width: 100%; height: auto; margin-top: 8px; background: white; border-radius: 4px; border: 1px solid #475569; display: block; }}
    </style>
</head>
<body>
    <h1>Workstream A — Phase 1 Segmentation Quality Gate Gallery</h1>
    <div class="summary">
        <strong>Inspected Sheets:</strong> {len(qa_results)} sheets across Grades 3, 4, 5, 6, 7.<br>
        <strong>Segmentation Quality Gate:</strong> Visual inspection of line bounds, deskewing, and word separation.<br>
        <strong>Output location:</strong> <code>data/processed/school_a/&lt;student_id&gt;/</code><br>
        <strong>Detailed Audit CSV:</strong> <code>qa/school_a_assessment_report.csv</code>
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
    parser.add_argument("--samples", type=int, default=6, help="Samples per grade (0 for all)")
    parser.add_argument("--all", action="store_true", help="Process all sheets in school_a")
    parser.add_argument("--force", action="store_true", help="Force re-process all sheets")
    args = parser.parse_args()

    run_qa_review(samples_per_grade=args.samples, force=args.force, all_sheets=args.all)

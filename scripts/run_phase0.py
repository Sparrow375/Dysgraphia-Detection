"""
Phase 0 Execution Script: Confirm the Data & Multi-Dataset Rendering
1. Discovers and seeded-randomly samples N files per dataset from:
   - dataSciRep_public (.svc files)
   - DiaGraMo-project (.json files across TSK3, TSK4, TSK15, TSK16)
2. Verifies column layouts, sampling rates, stroke counts
3. Renders static images to outputs/phase0/{sample_id}_rendered.png
4. Writes comprehensive metadata to outputs/phase0/phase0_summary.json
"""

import os
import sys
import glob
import json
import random
import argparse
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders import load_svc, load_diagramo_json
from src.render import render_trajectory_to_image, plot_multimodal_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0: Multi-Dataset Trajectory Rendering")
    parser.add_argument("--samples_per_dataset", type=int, default=35,
                        help="Number of samples to select per dataset (default: 35)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible cohort sampling (default: 42)")
    parser.add_argument("--save_plots", action="store_true", default=False,
                        help="Save multimodal diagnostic plots for all samples (default: False)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("PHASE 0: DATA CONFIRMATION & TRAJECTORY RENDERING")
    print(f"Sampling: {args.samples_per_dataset} samples/dataset | Seed: {args.seed}")
    print("=" * 70)

    datasets_dir = PROJECT_ROOT / "Datasets"
    output_dir = PROJECT_ROOT / "outputs" / "phase0"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # 1. Discover and sample dataSciRep_public
    svc_files = sorted(glob.glob(str(datasets_dir / "dataSciRep_public" / "**" / "*.svc"), recursive=True))
    if not svc_files:
        raise FileNotFoundError("No .svc files found in dataSciRep_public!")
    shuffled_svc = list(svc_files)
    rng.shuffle(shuffled_svc)
    selected_svc = shuffled_svc[:min(args.samples_per_dataset, len(shuffled_svc))]
    print(f"Discovered {len(svc_files)} dataSciRep .svc files; sampled {len(selected_svc)} files.")

    # 2. Discover and sample DiaGraMo-project strictly across text writing tasks (TSK3, TSK4, TSK15, TSK16)
    tasks = ["TSK3", "TSK4", "TSK15", "TSK16"]
    base_n = args.samples_per_dataset // len(tasks)
    rem = args.samples_per_dataset % len(tasks)
    task_allocations = {tsk: base_n + (1 if i < rem else 0) for i, tsk in enumerate(tasks)}

    import re
    drawing_pattern = re.compile(r'_(TSK1|TSK2|TSK5|TSK6|TSK7|TSK8|TSK9|TSK10|TSK11|TSK12|TSK13|TSK14)\(')
    selected_diagramo = []

    for tsk in tasks:
        matches = sorted(glob.glob(str(datasets_dir / "DiaGraMo-project" / "**" / f"*{tsk}*.json"), recursive=True))
        # Strictly verify each file belongs to the text task and matches the exact task boundary
        clean_text_matches = [
            m for m in matches
            if re.search(rf'_{tsk}\(', os.path.basename(m)) and not drawing_pattern.search(os.path.basename(m))
        ]
        if not clean_text_matches:
            print(f"Warning: No clean text files found for task {tsk}")
            continue
        shuffled_matches = list(clean_text_matches)
        rng.shuffle(shuffled_matches)
        n_alloc = task_allocations[tsk]
        task_sample = shuffled_matches[:n_alloc]
        selected_diagramo.extend(task_sample)
        print(f"DiaGraMo {tsk} (Text Handwriting): Discovered {len(clean_text_matches)} files; sampled {len(task_sample)} files.")

    # Explicit assertion check: confirm zero drawing/graphomotor tasks
    for f in selected_diagramo:
        base = os.path.basename(f)
        assert not drawing_pattern.search(base), f"Drawing task unexpectedly found in handwriting cohort: {base}"

    print(f"Total DiaGraMo clean text handwriting sampled: {len(selected_diagramo)} files.")

    # Combined list
    all_selected = [(f, "dataSciRep_public") for f in selected_svc] + \
                   [(f, "DiaGraMo") for f in selected_diagramo]
    print(f"\nTotal cohort size to render and summarize: {len(all_selected)} samples.\n")

    summary_records = []

    for idx, (fpath, dname) in enumerate(all_selected, start=1):
        fname = os.path.basename(fpath)
        try:
            if fpath.endswith(".svc"):
                sample = load_svc(fpath)
            else:
                sample = load_diagramo_json(fpath)
        except Exception as e:
            print(f"[{idx}/{len(all_selected)}] Error loading {fname}: {e}")
            continue

        # Render to image canvas
        rendered = render_trajectory_to_image(sample, target_width=1200, padding=40, line_width=3, invert_y=True)
        img_path = output_dir / f"{sample.sample_id}_rendered.png"
        rendered.save(img_path)

        diag_name = ""
        # Save plots for first 3 samples or if save_plots is True
        if args.save_plots or idx <= 3:
            diag_path = output_dir / f"{sample.sample_id}_multimodal.png"
            plot_multimodal_diagnostics(sample, rendered, str(diag_path))
            diag_name = str(diag_path.name)

        summary_records.append({
            "sample_id": sample.sample_id,
            "dataset": sample.dataset_name,
            "task": sample.task_name,
            "filepath": str(fpath),
            "total_points": len(sample.points),
            "stroke_count": len(sample.strokes),
            "duration_sec": round(sample.total_duration_sec, 2),
            "sampling_rate_hz": round(sample.sampling_rate_hz, 1),
            "in_air_percentage": round(sample.in_air_ratio * 100, 1),
            "pressure_min": float(np.min(sample.pressure)),
            "pressure_max": float(np.max(sample.pressure)),
            "rendered_image": str(img_path.name),
            "multimodal_plot": diag_name,
        })

        if idx % 10 == 0 or idx == len(all_selected):
            print(f"  Processed {idx}/{len(all_selected)}: {sample.sample_id} ({sample.task_name}, {len(sample.strokes)} strokes)")

    summary_json_path = output_dir / "phase0_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print(f"\nPhase 0 Complete! {len(summary_records)} samples rendered and recorded.")
    print(f"Summary written to: {summary_json_path}")


if __name__ == "__main__":
    main()

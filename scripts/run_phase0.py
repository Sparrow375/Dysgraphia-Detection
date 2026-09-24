"""
Phase 0 Execution Script: Confirm the Data
1. Loads sample .svc from dataSciRep_public
2. Loads sample TSK4 (dictation) .json from DiaGraMo
3. Loads sample TSK16 (copy sentence) .json from DiaGraMo
4. Verifies column layouts, sampling rates, stroke counts
5. Renders static images and saves paired multimodal diagnostic figures to outputs/phase0/
"""

import os
import sys
import glob
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders import load_svc, load_diagramo_json
from src.render import render_trajectory_to_image, plot_multimodal_diagnostics


def main():
    print("=" * 70)
    print("PHASE 0: DATA CONFIRMATION & TRAJECTORY RENDERING")
    print("=" * 70)

    datasets_dir = PROJECT_ROOT / "Datasets"
    output_dir = PROJECT_ROOT / "outputs" / "phase0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate samples
    # dataSciRep_public sample
    svc_files = sorted(glob.glob(str(datasets_dir / "dataSciRep_public" / "**" / "*.svc"), recursive=True))
    if not svc_files:
        raise FileNotFoundError("No .svc files found in dataSciRep_public!")
    sample_svc_path = svc_files[0]

    # DiaGraMo TSK4 sample
    tsk4_files = sorted(glob.glob(str(datasets_dir / "DiaGraMo-project" / "**" / "*TSK4*.json"), recursive=True))
    if not tsk4_files:
        raise FileNotFoundError("No TSK4 .json files found in DiaGraMo!")
    sample_tsk4_path = tsk4_files[0]

    # DiaGraMo TSK16 sample
    tsk16_files = sorted(glob.glob(str(datasets_dir / "DiaGraMo-project" / "**" / "*TSK16*.json"), recursive=True))
    if not tsk16_files:
        raise FileNotFoundError("No TSK16 .json files found in DiaGraMo!")
    sample_tsk16_path = tsk16_files[0]

    samples_to_process = [
        ("dataSciRep_public", sample_svc_path, False),
        ("DiaGraMo_TSK4", sample_tsk4_path, False),
        ("DiaGraMo_TSK16", sample_tsk16_path, False),
    ]

    summary_records = []

    for name, fpath, invert_y in samples_to_process:
        print(f"\nProcessing {name}: {os.path.basename(fpath)}")
        if fpath.endswith(".svc"):
            sample = load_svc(fpath)
        else:
            sample = load_diagramo_json(fpath)

        print(f"  Sample ID:        {sample.sample_id}")
        print(f"  Dataset:          {sample.dataset_name}")
        print(f"  Task:             {sample.task_name}")
        print(f"  Total Points:     {len(sample.points):,}")
        print(f"  On-surface Strokes:{len(sample.strokes):,}")
        print(f"  Duration:         {sample.total_duration_sec:.2f} seconds")
        print(f"  Sampling Rate:    {sample.sampling_rate_hz:.1f} Hz")
        print(f"  In-Air Ratio:     {sample.in_air_ratio * 100:.1f}%")
        print(f"  X Range:          [{np.min(sample.x):.1f}, {np.max(sample.x):.1f}]")
        print(f"  Y Range:          [{np.min(sample.y):.1f}, {np.max(sample.y):.1f}]")
        print(f"  Pressure Range:   [{np.min(sample.pressure):.1f}, {np.max(sample.pressure):.1f}]")

        # Physical tablet coordinates have (0,0) at bottom-left; image pixels have (0,0) at top-left.
        # invert_y=True correctly maps top lines of handwriting to the top of the image canvas.
        rendered = render_trajectory_to_image(sample, target_width=1200, padding=40, line_width=3, invert_y=True)

        img_path = output_dir / f"{sample.sample_id}_rendered.png"
        rendered.save(img_path)

        # Plot multimodal diagnostics
        diag_path = output_dir / f"{sample.sample_id}_multimodal.png"
        plot_multimodal_diagnostics(sample, rendered, str(diag_path))
        print(f"  Rendered image saved: {img_path}")
        print(f"  Multimodal plot saved: {diag_path}")

        summary_records.append({
            "name": name,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset_name,
            "task": sample.task_name,
            "total_points": len(sample.points),
            "stroke_count": len(sample.strokes),
            "duration_sec": round(sample.total_duration_sec, 2),
            "sampling_rate_hz": round(sample.sampling_rate_hz, 1),
            "in_air_percentage": round(sample.in_air_ratio * 100, 1),
            "pressure_min": float(np.min(sample.pressure)),
            "pressure_max": float(np.max(sample.pressure)),
            "x_min": float(np.min(sample.x)),
            "x_max": float(np.max(sample.x)),
            "y_min": float(np.min(sample.y)),
            "y_max": float(np.max(sample.y)),
            "rendered_image": str(img_path.name),
            "multimodal_plot": str(diag_path.name),
        })

    summary_json_path = output_dir / "phase0_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\nPhase 0 complete! Summary written to: {summary_json_path}")


if __name__ == "__main__":
    import numpy as np
    main()

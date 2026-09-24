"""
Phase 2 Execution Script: Branch B — Kinematics & Ground-Truth Validation
1. Loads static rendered images from Phase 0 (dataSciRep_public, DiaGraMo TSK4, DiaGraMo TSK16)
2. Runs topological skeleton graph recovery with minimal-direction-change heuristic
3. Reconstructs velocity and pressure profiles via Sigma-Lognormal / Power-Law modeling
4. Validates directly against real recorded sensor signals (.svc and .json)
5. Computes and reports Pearson correlation r and NRMSE
6. Generates side-by-side ground truth vs reconstructed waveform figures
"""

import os
import sys
import glob
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders import load_svc, load_diagramo_json
from src.branch_a.preprocessing import load_and_binarize, zhang_suen_skeletonize, distance_transform_edt
from src.branch_b.stroke_recovery import recover_handwriting_trajectory
from src.branch_b.kinematics import validate_kinematics_against_ground_truth, extract_kinematic_features


def plot_groundtruth_comparison(val_result: dict, output_path: str):
    """
    Renders 3-panel comparative validation plot:
      1. Reconstructed vs Ground-Truth Velocity Profile
      2. Optical Pressure Proxy vs True Stylus Pressure Profile
      3. Summary of Correlation and Error Metrics
    """
    curves = val_result["resampled_curves"]
    grid = np.array(curves["grid"])
    gt_v = np.array(curves["gt_v_norm"])
    rec_v = np.array(curves["rec_v_norm"])
    gt_p = np.array(curves["gt_p_norm"])
    rec_p = np.array(curves["rec_p_norm"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Velocity Comparison
    r_v = val_result["pearson_r_velocity"]
    p_v = val_result["pvalue_velocity"]
    nrmse_v = val_result["nrmse_velocity"]

    axes[0].plot(grid, gt_v, color="#2b5c8f", lw=1.8, label="Ground Truth Recorded Velocity (v_true)")
    axes[0].plot(grid, rec_v, color="#d95f02", lw=1.8, linestyle="--", label=f"Reconstructed Velocity (v_rec, r={r_v:.3f})")
    axes[0].set_title(f"1. Velocity Profile Validation: r = {r_v:.3f} (p = {p_v:.1e}) | NRMSE = {nrmse_v:.3f}",
                      fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Normalized Velocity [0, 1]", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: Pressure Comparison
    r_p = val_result["pearson_r_pressure"]
    p_p = val_result["pvalue_pressure"]
    nrmse_p = val_result["nrmse_pressure"]

    axes[1].plot(grid, gt_p, color="#1b9e77", lw=1.8, label="Ground Truth Stylus Pressure (P_true)")
    axes[1].plot(grid, rec_p, color="#7570b3", lw=1.8, linestyle="--", label=f"Optical Pressure Proxy (P_rec, r={r_p:.3f})")
    axes[1].set_title(f"2. Stylus Pressure Validation: r = {r_p:.3f} (p = {p_p:.1e}) | NRMSE = {nrmse_p:.3f}",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Normalized Trajectory Progression [0, 1]", fontsize=10)
    axes[1].set_ylabel("Normalized Pressure [0, 1]", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=9)

    kin = val_result["kinematic_features"]
    header = f"{val_result['dataset']} | {val_result['sample_id']} ({val_result['task']})"
    subtext = (
        f"GT Strokes: {val_result['gt_stroke_count']} | Recovered Strokes: {val_result['recovered_stroke_count']} | "
        f"Reconstructed NVI Rate: {kin['nvi_rate']:.1f} inv/s | Total NVI: {kin['total_nvi']} | Peak V: {kin['peak_velocity']:.1f}"
    )
    fig.suptitle(f"{header}\n{subtext}", fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("PHASE 2: BRANCH B — KINEMATICS & GROUND-TRUTH VALIDATION")
    print("=" * 70)

    datasets_dir = PROJECT_ROOT / "Datasets"
    phase0_dir = PROJECT_ROOT / "outputs" / "phase0"
    output_dir = PROJECT_ROOT / "outputs" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 0 summary to retrieve matched filepaths
    phase0_summary_path = phase0_dir / "phase0_summary.json"
    if not phase0_summary_path.exists():
        raise FileNotFoundError("Phase 0 summary not found! Run Phase 0 first.")

    with open(phase0_summary_path, "r", encoding="utf-8") as f:
        samples_meta = json.load(f)

    validation_results = []

    print(f"\nEvaluating {len(samples_meta)} ground-truth samples against static reconstructions:\n")

    for meta in samples_meta:
        sid = meta["sample_id"]
        dname = meta["dataset"]
        task = meta["task"]
        rendered_img_path = phase0_dir / meta["rendered_image"]

        print(f"--> Processing: {sid} ({dname} - {task})")

        # 1. Reload raw ground-truth telemetry
        if dname == "dataSciRep_public":
            svc_files = glob.glob(str(datasets_dir / "dataSciRep_public" / "**" / f"{sid}.svc"), recursive=True)
            if not svc_files:
                print(f"  Warning: SVC file not found for {sid}")
                continue
            sample = load_svc(svc_files[0])
        else:
            json_files = glob.glob(str(datasets_dir / "DiaGraMo-project" / "**" / f"{sid}.json"), recursive=True)
            if not json_files:
                print(f"  Warning: JSON file not found for {sid}")
                continue
            sample = load_diagramo_json(json_files[0])

        # 2. Extract static skeleton from rendered image
        binary_mask = load_and_binarize(str(rendered_img_path))
        skeleton = zhang_suen_skeletonize(binary_mask)
        dist_map = distance_transform_edt(binary_mask)

        # 3. Recover stroke order and trajectory via minimal-direction-change heuristic
        recovered_strokes = recover_handwriting_trajectory(skeleton)
        print(f"  Original on-surface strokes: {len(sample.strokes)}")
        print(f"  Recovered strokes from image: {len(recovered_strokes)}")

        # 4. Validate reconstructed kinematics against true ground truth
        val = validate_kinematics_against_ground_truth(sample, recovered_strokes, dist_map)

        if val.get("status") == "success":
            r_v = val["pearson_r_velocity"]
            p_v = val["pvalue_velocity"]
            r_p = val["pearson_r_pressure"]
            p_p = val["pvalue_pressure"]
            nrmse_v = val["nrmse_velocity"]
            nrmse_p = val["nrmse_pressure"]

            print(f"  [RESULT] Velocity Pearson r (whole doc): {r_v:+.4f} (p = {p_v:.2e}, NRMSE = {nrmse_v:.3f})")
            print(f"  [RESULT] Single-Stroke Level 1 Velocity: Mean r = {val['stroke_level_mean_r']:+.4f} (Median = {val['stroke_level_median_r']:+.4f}, Frac > 0.3 = {val['stroke_level_fraction_gt_03']:.1%}, N={val['stroke_level_count']})")
            print(f"  [RESULT] Pressure Pearson r:             {r_p:+.4f} (p = {p_p:.2e}, NRMSE = {nrmse_p:.3f})")
            print(f"  [RESULT] NVI per Stroke:                 {val['kinematic_features']['nvi_per_stroke']:.2f}")
            print(f"  [RESULT] NVI Rate:                       {val['kinematic_features']['nvi_rate']:.2f} inversions/s")
            print(f"  [RESULT] Total Inversions:               {val['kinematic_features']['total_nvi']}")

            # Save diagnostic plot
            plot_path = output_dir / f"{sid}_groundtruth_vs_reconstruction.png"
            plot_groundtruth_comparison(val, str(plot_path))
            print(f"  Comparison plot saved to: {plot_path}")

            # Prepare record for JSON (strip huge resampled curves for compact summary)
            rec = {k: v for k, v in val.items() if k != "resampled_curves"}
            rec["comparison_plot"] = str(plot_path.name)
            validation_results.append(rec)
        else:
            print(f"  Validation error: {val.get('status')}")

    # Export validation summary
    summary_path = output_dir / "phase2_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 2 GROUND-TRUTH VALIDATION SUMMARY (MULTI-LEVEL KINEMATICS)")
    print("=" * 80)
    print(f"{'Sample ID':<26} | {'Doc Vel r':<10} | {'Stroke Mean r':<14} | {'Frac r>0.3':<10} | {'Recovered / GT'}")
    print("-" * 80)
    for r in validation_results:
        print(f"{r['sample_id']:<26} | {r['pearson_r_velocity']:+8.4f}   | {r['stroke_level_mean_r']:+12.4f}   | {r['stroke_level_fraction_gt_03']:8.1%}   | {r['recovered_stroke_count']} / {r['gt_stroke_count']}")

    print(f"\nSaved Phase 2 validation summary to: {summary_path}")
    print("Phase 2 Complete!")


if __name__ == "__main__":
    main()

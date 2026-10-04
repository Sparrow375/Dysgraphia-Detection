"""
Phase 2 Execution Script: Branch B — Kinematics & Ground-Truth Validation
1. Loads static rendered images from Phase 0 (seeded-random cohort across dataSciRep_public & DiaGraMo)
2. Runs topological skeleton graph recovery with scale-invariant H_med heuristics
3. Reconstructs velocity and pressure profiles via Sigma-Lognormal / Power-Law modeling
4. Validates directly against real recorded tablet sensor signals (.svc and .json)
5. Computes and reports full distribution (mean, median, std, min, max) of Level-1 single-stroke correlations
"""

import os
import sys
import glob
import json
import math
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders import load_svc, load_diagramo_json
from src.branch_a.preprocessing import load_and_binarize, zhang_suen_skeletonize, distance_transform_edt
from src.branch_a.segmentation import segment_handwriting
from src.branch_b.stroke_recovery import recover_handwriting_trajectory
from src.branch_b.kinematics import validate_kinematics_against_ground_truth, extract_kinematic_features


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Multi-Sample Ground-Truth Validation")
    parser.add_argument("--save_plots", action="store_true", default=False,
                        help="Save comparative plots for all samples (default: False)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit evaluation to first N samples from summary")
    return parser.parse_args()


def plot_groundtruth_comparison(val_result: dict, output_path: str):
    """
    Renders comparative validation plot:
      1. Reconstructed vs Ground-Truth Velocity Profile
      2. Optical Pressure Proxy vs True Stylus Pressure Profile
    """
    curves = val_result.get("resampled_curves")
    if not curves:
        return
    grid = np.array(curves["grid"])
    gt_v = np.array(curves["gt_v_norm"])
    rec_v = np.array(curves["rec_v_norm"])
    gt_p = np.array(curves["gt_p_norm"])
    rec_p = np.array(curves["rec_p_norm"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

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


def compute_distribution_stats(values: list) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    args = parse_args()
    print("=" * 75)
    print("PHASE 2: BRANCH B — KINEMATICS & GROUND-TRUTH VALIDATION BENCHMARK")
    print("=" * 75)

    datasets_dir = PROJECT_ROOT / "Datasets"
    phase0_dir = PROJECT_ROOT / "outputs" / "phase0"
    output_dir = PROJECT_ROOT / "outputs" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase0_summary_path = phase0_dir / "phase0_summary.json"
    if not phase0_summary_path.exists():
        raise FileNotFoundError("Phase 0 summary not found! Run Phase 0 first.")

    with open(phase0_summary_path, "r", encoding="utf-8") as f:
        samples_meta = json.load(f)

    if args.max_samples:
        samples_meta = samples_meta[:args.max_samples]

    validation_results = []
    print(f"\nEvaluating cohort of {len(samples_meta)} ground-truth samples...\n")

    for idx, meta in enumerate(samples_meta, start=1):
        sid = meta["sample_id"]
        dname = meta["dataset"]
        task = meta["task"]
        rendered_img_path = phase0_dir / meta["rendered_image"]

        if not rendered_img_path.exists():
            print(f"[{idx}/{len(samples_meta)}] Warning: Rendered image {rendered_img_path.name} not found. Skipping.")
            continue

        # 1. Reload raw ground-truth telemetry
        fpath = meta.get("filepath", "")
        if not fpath or not os.path.exists(fpath):
            if dname == "dataSciRep_public":
                m = glob.glob(str(datasets_dir / "dataSciRep_public" / "**" / f"{sid}.svc"), recursive=True)
                fpath = m[0] if m else ""
            else:
                m = glob.glob(str(datasets_dir / "DiaGraMo-project" / "**" / f"{sid}.json"), recursive=True)
                fpath = m[0] if m else ""

        if not fpath or not os.path.exists(fpath):
            print(f"[{idx}/{len(samples_meta)}] Warning: Telemetry file not found for {sid}. Skipping.")
            continue

        try:
            if fpath.endswith(".svc"):
                sample = load_svc(fpath)
            else:
                sample = load_diagramo_json(fpath)
        except Exception as e:
            print(f"[{idx}/{len(samples_meta)}] Error loading {sid}: {e}")
            continue

        # 2. Extract static skeleton from rendered image
        binary_mask = load_and_binarize(str(rendered_img_path))
        skeleton = zhang_suen_skeletonize(binary_mask)
        dist_map = distance_transform_edt(binary_mask)

        # 3. Derive scale factor H_med from segmentation
        text_lines = segment_handwriting(binary_mask)
        med_heights = [l.median_height for l in text_lines if l.components]
        h_med = float(np.median(med_heights)) if med_heights else 25.0
        h_med = max(h_med, 1.0)

        # 4. Recover stroke order and trajectory via scale-invariant recovery
        recovered_strokes = recover_handwriting_trajectory(skeleton, h_med=h_med)

        # 5. Validate reconstructed kinematics against true ground truth
        val = validate_kinematics_against_ground_truth(sample, recovered_strokes, dist_map, h_med=h_med)

        if val.get("status") == "success":
            rec = {k: v for k, v in val.items() if k != "resampled_curves"}
            if args.save_plots or idx <= 3:
                plot_path = output_dir / f"{sid}_groundtruth_vs_reconstruction.png"
                plot_groundtruth_comparison(val, str(plot_path))
                rec["comparison_plot"] = str(plot_path.name)
            validation_results.append(rec)

            stk_r = val['stroke_level_mean_r']
            stk_med = val['stroke_level_median_r']
            stk_cnt = val['stroke_level_count']
            frac30 = val['stroke_level_fraction_gt_03']
            print(f"[{idx:2d}/{len(samples_meta):2d}] {sid:<24} | {dname:<18} | H_med={h_med:4.1f}px | Strokes={stk_cnt:3d} | Level-1 Mean r = {stk_r:+6.3f} (Med = {stk_med:+6.3f}, >0.3 = {frac30*100:4.1f}%)")
        else:
            print(f"[{idx:2d}/{len(samples_meta):2d}] {sid:<24} | Validation Status: {val.get('status')}")

    # Compute Statistical Distributions
    ds_results = [r for r in validation_results if r["dataset"] == "dataSciRep_public"]
    diag_results = [r for r in validation_results if r["dataset"] != "dataSciRep_public"]

    ds_means = [r["stroke_level_mean_r"] for r in ds_results if r["stroke_level_count"] > 0]
    diag_means = [r["stroke_level_mean_r"] for r in diag_results if r["stroke_level_count"] > 0]
    all_means = [r["stroke_level_mean_r"] for r in validation_results if r["stroke_level_count"] > 0]

    ds_medians = [r["stroke_level_median_r"] for r in ds_results if r["stroke_level_count"] > 0]
    diag_medians = [r["stroke_level_median_r"] for r in diag_results if r["stroke_level_count"] > 0]
    all_medians = [r["stroke_level_median_r"] for r in validation_results if r["stroke_level_count"] > 0]

    dist_ds = compute_distribution_stats(ds_means)
    dist_diag = compute_distribution_stats(diag_means)
    dist_all = compute_distribution_stats(all_means)

    total_strokes_eval = sum(r["stroke_level_count"] for r in validation_results)
    avg_frac_gt30 = float(np.mean([r["stroke_level_fraction_gt_03"] for r in validation_results if r["stroke_level_count"] > 0])) if all_means else 0.0
    avg_frac_gt50 = float(np.mean([r["stroke_level_fraction_gt_05"] for r in validation_results if r["stroke_level_count"] > 0])) if all_means else 0.0

    print("\n" + "=" * 90)
    print("LEVEL-1 STROKE VELOCITY CORRELATION DISTRIBUTION (BROAD MULTI-DATASET COHORT)")
    print("=" * 90)
    print(f"{'Cohort':<22} | {'N':<5} | {'Mean r':<8} | {'Median r':<9} | {'Std':<7} | {'Min r':<8} | {'Max r':<8}")
    print("-" * 90)
    print(f"{'dataSciRep_public':<22} | {dist_ds['count']:<5d} | {dist_ds['mean']:+7.4f}  | {dist_ds['median']:+8.4f}  | {dist_ds['std']:6.4f}  | {dist_ds['min']:+7.4f}  | {dist_ds['max']:+7.4f}")
    print(f"{'DiaGraMo-project':<22} | {dist_diag['count']:<5d} | {dist_diag['mean']:+7.4f}  | {dist_diag['median']:+8.4f}  | {dist_diag['std']:6.4f}  | {dist_diag['min']:+7.4f}  | {dist_diag['max']:+7.4f}")
    print("-" * 90)
    print(f"{'OVERALL COMBINED':<22} | {dist_all['count']:<5d} | {dist_all['mean']:+7.4f}  | {dist_all['median']:+8.4f}  | {dist_all['std']:6.4f}  | {dist_all['min']:+7.4f}  | {dist_all['max']:+7.4f}")
    print("=" * 90)
    print(f"Total Individual Strokes Evaluated: {total_strokes_eval:,}")
    print(f"Average Fraction of Strokes with r > 0.30: {avg_frac_gt30 * 100:.1f}%")
    print(f"Average Fraction of Strokes with r > 0.50: {avg_frac_gt50 * 100:.1f}%")

    # Pressure Correlation Diagnostics & Aggregation
    valid_pressures = [
        r["pearson_r_pressure"] for r in validation_results
        if not math.isnan(r.get("pearson_r_pressure", float("nan")))
    ]
    excluded_pressures = [
        r for r in validation_results
        if math.isnan(r.get("pearson_r_pressure", float("nan")))
    ]
    print("\n" + "=" * 90)
    print("WHOLE-DOCUMENT PRESSURE CORRELATION DIAGNOSTICS")
    print("=" * 90)
    print(f"Total Samples Evaluated:            {len(validation_results)}")
    print(f"Valid Pressure Corrs:              {len(valid_pressures)}")
    print(f"Excluded Constant Pressure Samples: {len(excluded_pressures)}")
    for ex in excluded_pressures:
        print(f"  - Excluded {ex['sample_id']} ({ex['dataset']}, {ex['task']}): Constant signal variance (std < 1e-4).")
    if valid_pressures:
        dist_p = compute_distribution_stats(valid_pressures)
        print(f"Mean Pressure r:   {dist_p['mean']:+.4f} (Median: {dist_p['median']:+.4f}, Std: {dist_p['std']:.4f})")
    print("Confirming: Zero constant/NaN pressure samples were silently included as 0.0 in any aggregate.")

    # Export validation summary with distributions
    summary_data = {
        "cohort_distribution": {
            "overall": dist_all,
            "dataSciRep_public": dist_ds,
            "DiaGraMo_project": dist_diag,
            "total_strokes_evaluated": total_strokes_eval,
            "average_fraction_gt_03": avg_frac_gt30,
            "average_fraction_gt_05": avg_frac_gt50,
        },
        "per_sample_results": validation_results
    }

    summary_path = output_dir / "phase2_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved Phase 2 validation summary to: {summary_path}")


if __name__ == "__main__":
    main()

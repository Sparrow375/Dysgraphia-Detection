"""
Phase 1 Execution Script: Branch A — BHK Static Feature Extraction
1. Runs full Branch A pipeline over a batch of real photographed handwriting
   (Low Potential Dysgraphia vs Potential Dysgraphia) and rendered tablet samples.
2. Extracts all 6 BHK features:
   - Size Covariance
   - Height-Ratio Consistency
   - Baseline Drift
   - Inter-Letter Spacing Entropy
   - Stroke-Width Variance
   - Telescoping / Overlap
3. Produces visual preprocessing/segmentation diagnostic comparisons.
4. Generates statistical comparison figures and exports CSV.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.branch_a.features import extract_bhk_features


def visualize_extraction_stages(image_path: str, output_path: str, title: str):
    """
    Renders 4-stage visual diagnostic:
      1. Original Input Image
      2. Adaptive Binary Mask
      3. Zhang-Suen 1-Pixel Skeleton
      4. Segmented Letter Bounding Boxes & Baseline Fits
    """
    res = extract_bhk_features(image_path)
    binary = res["binary_mask"]
    skel = res["skeleton"]
    lines = res["text_lines"]

    orig_img = Image.open(image_path).convert("RGB")
    orig_arr = np.array(orig_img)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Stage 1: Original
    axes[0, 0].imshow(orig_arr)
    axes[0, 0].set_title("1. Original Static Handwriting", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Stage 2: Binary Mask
    axes[0, 1].imshow(binary, cmap="gray")
    axes[0, 1].set_title(f"2. Adaptive Binary Mask ({np.sum(binary):,} ink px)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Stage 3: Skeleton
    # Invert for white background, red skeleton
    skel_rgb = np.ones((skel.shape[0], skel.shape[1], 3), dtype=np.uint8) * 255
    skel_rgb[skel > 0] = [217, 95, 2]  # Orange-red skeleton
    axes[1, 0].imshow(skel_rgb)
    axes[1, 0].set_title(f"3. 1-Pixel Skeleton ({np.sum(skel):,} centerline px)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Stage 4: Segmentation & Baselines
    axes[1, 1].imshow(binary, cmap="gray", alpha=0.3)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for l_idx, line in enumerate(lines):
        col = colors[l_idx % len(colors)]
        x_pts, y_pts = [], []
        for comp in line.components:
            y0, y1, x0, x1 = comp.bbox
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=col, linewidth=1.2)
            axes[1, 1].add_patch(rect)
            x_pts.append(comp.x_center)
            y_pts.append(comp.y_bottom)

        if len(x_pts) >= 2:
            p = np.polyfit(x_pts, y_pts, 1)
            x_line = np.linspace(min(x_pts), max(x_pts), 50)
            axes[1, 1].plot(x_line, np.polyval(p, x_line), color="red", linestyle="--", linewidth=1.5)

    tot_letters = sum(len(l.components) for l in lines)
    axes[1, 1].set_title(f"4. Segmentation: {tot_letters} letters, {len(lines)} lines (Baselines in red)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Metrics summary in suptitle
    m = res["metrics"]
    metrics_str = (
        f"Size CV: {m['size_covariance_score']:.2f} | "
        f"Height IQR: {m['height_iqr_ratio']:.2f} | "
        f"Baseline Drift: {m['baseline_drift_score']:.2f} | "
        f"Spacing Entropy: {m['spacing_entropy']:.2f} | "
        f"Stroke-Width CV: {m['stroke_width_cv']:.2f} | "
        f"Telescoping: {m['telescoping_score']:.2f}"
    )
    fig.suptitle(f"{title}\n{metrics_str}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("PHASE 1: BRANCH A — BHK STATIC FEATURE EXTRACTION")
    print("=" * 70)

    datasets_dir = PROJECT_ROOT / "Datasets"
    output_dir = PROJECT_ROOT / "outputs" / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    lpd_dir = datasets_dir / "DATASET DYSGRAPHIA HANDWRITING" / "Low Potential Dysgraphia"
    pd_dir = datasets_dir / "DATASET DYSGRAPHIA HANDWRITING" / "Potential Dysgraphia"

    lpd_files = sorted(glob.glob(str(lpd_dir / "*.jpg")))[:25]
    pd_files = sorted(glob.glob(str(pd_dir / "*.jpg")))[:25]

    print(f"Loaded {len(lpd_files)} Low Potential Dysgraphia (Control) samples.")
    print(f"Loaded {len(pd_files)} Potential Dysgraphia (Dysgraphic) samples.")

    # Rendered samples from Phase 0
    phase0_dir = PROJECT_ROOT / "outputs" / "phase0"
    rendered_files = sorted(glob.glob(str(phase0_dir / "*_rendered.png")))
    print(f"Loaded {len(rendered_files)} rendered tablet samples from Phase 0.")

    records = []

    # Process Malay LPD
    for f in lpd_files:
        try:
            res = extract_bhk_features(f)
            rec = {
                "sample_id": Path(f).stem,
                "dataset": "Malay_Handwriting",
                "label": "Control (LPD)",
                "is_dysgraphic": 0,
                **{k: round(v, 4) for k, v in res["metrics"].items() if isinstance(v, (int, float))},
            }
            records.append(rec)
        except Exception as e:
            print(f"  Error on {f}: {e}")

    # Process Malay PD
    for f in pd_files:
        try:
            res = extract_bhk_features(f)
            rec = {
                "sample_id": Path(f).stem,
                "dataset": "Malay_Handwriting",
                "label": "Dysgraphic (PD)",
                "is_dysgraphic": 1,
                **{k: round(v, 4) for k, v in res["metrics"].items() if isinstance(v, (int, float))},
            }
            records.append(rec)
        except Exception as e:
            print(f"  Error on {f}: {e}")

    # Process Rendered Tablet samples
    for f in rendered_files:
        try:
            res = extract_bhk_features(f)
            rec = {
                "sample_id": Path(f).stem.replace("_rendered", ""),
                "dataset": "Rendered_Tablet",
                "label": "Tablet_Rendered",
                "is_dysgraphic": -1,
                **{k: round(v, 4) for k, v in res["metrics"].items() if isinstance(v, (int, float))},
            }
            records.append(rec)
        except Exception as e:
            print(f"  Error on {f}: {e}")

    df = pd.DataFrame(records)
    csv_path = output_dir / "bhk_features_batch.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved batch feature matrix ({len(df)} rows) to: {csv_path}")

    # Statistical comparison: LPD vs PD
    malay_df = df[df["dataset"] == "Malay_Handwriting"]
    lpd_sub = malay_df[malay_df["is_dysgraphic"] == 0]
    pd_sub = malay_df[malay_df["is_dysgraphic"] == 1]

    feature_cols = [
        ("size_covariance_score", "Size Covariance"),
        ("height_iqr_ratio", "Height-Ratio Dispersion"),
        ("baseline_drift_score", "Baseline Drift"),
        ("spacing_entropy", "Spacing Entropy"),
        ("stroke_width_cv", "Stroke-Width CV"),
        ("telescoping_score", "Telescoping / Overlap"),
        ("acute_turns_score", "Acute Turns / Tremor"),
        ("left_margin_score", "Left Margin Drift"),
        ("line_collision_score", "Line Collisions"),
    ]

    print("\n" + "=" * 70)
    print("STATISTICAL VALIDATION: Low Potential (LPD) vs Potential Dysgraphia (PD)")
    print("=" * 70)
    print(f"{'Feature':<28} | {'LPD Mean (±Std)':<18} | {'PD Mean (±Std)':<18} | {'p-value':<10} | {'Status'}")
    print("-" * 88)

    stat_summary = []
    for col, name in feature_cols:
        lpd_vals = lpd_sub[col].dropna()
        pd_vals = pd_sub[col].dropna()

        m_lpd, s_lpd = lpd_vals.mean(), lpd_vals.std()
        m_pd, s_pd = pd_vals.mean(), pd_vals.std()

        # Mann-Whitney U test (non-parametric)
        stat, p_val = mannwhitneyu(pd_vals, lpd_vals, alternative="greater")
        direction_ok = m_pd > m_lpd
        sig = "p < 0.05" if p_val < 0.05 else "n.s."
        flag = "[ELEVATED IN PD]" if direction_ok else "[LOWER]"

        print(f"{name:<28} | {m_lpd:6.3f} (±{s_lpd:5.3f})   | {m_pd:6.3f} (±{s_pd:5.3f})   | {p_val:8.4f}   | {flag} {sig}")

        stat_summary.append({
            "feature": name,
            "col": col,
            "lpd_mean": m_lpd,
            "lpd_std": s_lpd,
            "pd_mean": m_pd,
            "pd_std": s_pd,
            "p_value": p_val,
            "elevated_in_pd": direction_ok,
        })

    # Generate Statistical Comparison Bar Plot (3x3 grid for 9 features)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (col, name) in enumerate(feature_cols):
        ax = axes[idx]
        means = [lpd_sub[col].mean(), pd_sub[col].mean()]
        stds = [lpd_sub[col].std(), pd_sub[col].std()]
        bars = ax.bar(["Control (LPD)", "Dysgraphic (PD)"], means, yerr=stds, capsize=6,
                       color=["#4575b4", "#d73027"], alpha=0.85, edgecolor="black", width=0.5)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Feature Score", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Annotate values
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h / 2.0, f"{h:.2f}",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    fig.suptitle("Branch A: BHK Static Features Comparison\nControl (LPD) vs Dysgraphic (PD) on Malay Handwriting Dataset",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    comp_plot_path = output_dir / "bhk_statistical_comparison.png"
    plt.savefig(comp_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved statistical comparison plot to: {comp_plot_path}")

    # Generate Visual Pipeline Stages for 1 LPD and 1 PD sample
    sample_lpd = lpd_files[0]
    sample_pd = pd_files[0]

    stage_lpd_path = output_dir / "diagnostic_stages_LPD.png"
    stage_pd_path = output_dir / "diagnostic_stages_PD.png"

    visualize_extraction_stages(sample_lpd, str(stage_lpd_path), "Control Sample (LPD (1).jpg)")
    visualize_extraction_stages(sample_pd, str(stage_pd_path), "Dysgraphic Sample (PD (1).jpg)")

    print(f"Saved LPD visual stages to: {stage_lpd_path}")
    print(f"Saved PD visual stages to: {stage_pd_path}")
    print("\nPhase 1 Complete!")


if __name__ == "__main__":
    main()

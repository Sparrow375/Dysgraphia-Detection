"""
Phase 3 Execution Script: Consolidated Pipeline Batch Evaluation
1. Initializes unified DysgraphiaFeaturePipeline (Branch A + Branch B)
2. Processes batch of samples across all datasets:
   - Malay Photographed Dataset (LPD vs PD)
   - Rendered Tablet Datasets (dataSciRep_public, DiaGraMo TSK4, DiaGraMo TSK16)
3. Generates consolidated feature CSV (14 features per sample)
4. Computes cross-modal feature correlation matrix
5. Generates normalized multimodal profile comparison figure
"""

import os
import sys
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import DysgraphiaFeaturePipeline


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list, output_path: str):
    """Generates cross-modal correlation heatmap between BHK and Kinematic features."""
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient (r)", fontsize=10)

    # Format ticks
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    clean_labels = [c.replace("_", " ").title() for c in feature_cols]
    ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(clean_labels, fontsize=9)

    # Annotate correlation values in cells
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Cross-Modal Feature Correlation Matrix\n(Branch A: BHK Static Features vs Branch B: Kinematic Proxies)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_consolidated_profile(df: pd.DataFrame, feature_cols: list, output_path: str):
    """
    Plots normalized comparative profile of Controls (LPD) vs Dysgraphic (PD)
    across the full 14-feature space.
    """
    malay_df = df[df["dataset"] == "Malay_Handwriting"].copy()
    lpd = malay_df[malay_df["label"] == "Control (LPD)"][feature_cols]
    pd_sub = malay_df[malay_df["label"] == "Dysgraphic (PD)"][feature_cols]

    # Z-score normalize using overall Malay distribution
    means = malay_df[feature_cols].mean()
    stds = malay_df[feature_cols].std().replace(0, 1.0)

    lpd_z = (lpd - means) / stds
    pd_z = (pd_sub - means) / stds

    lpd_z_mean = lpd_z.mean()
    pd_z_mean = pd_z.mean()

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(feature_cols))
    width = 0.38

    ax.bar(x - width/2, lpd_z_mean, width, label="Control (LPD)", color="#4575b4", alpha=0.85, edgecolor="black")
    ax.bar(x + width/2, pd_z_mean, width, label="Dysgraphic (PD)", color="#d73027", alpha=0.85, edgecolor="black")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    clean_labels = [c.replace("_", " ").title() for c in feature_cols]
    ax.set_xticklabels(clean_labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Standardized Score (Z-Score Deviation)", fontsize=10)
    ax.set_title("Multimodal Feature Profiles: Control (LPD) vs Dysgraphic (PD)\n(Branch A: Spatial BHK Indices | Branch B: Neuromotor Kinematics)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("PHASE 3: CONSOLIDATED MULTIMODAL PIPELINE EVALUATION")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "outputs" / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DysgraphiaFeaturePipeline(compute_kinematics=True)

    datasets_dir = PROJECT_ROOT / "Datasets"
    phase0_dir = PROJECT_ROOT / "outputs" / "phase0"

    # 1. Gather samples
    lpd_files = sorted(glob.glob(str(datasets_dir / "DATASET DYSGRAPHIA HANDWRITING" / "Low Potential Dysgraphia" / "*.jpg")))[:25]
    pd_files = sorted(glob.glob(str(datasets_dir / "DATASET DYSGRAPHIA HANDWRITING" / "Potential Dysgraphia" / "*.jpg")))[:25]
    rendered_files = sorted(glob.glob(str(phase0_dir / "*_rendered.png")))

    sample_queue = []
    for f in lpd_files:
        sample_queue.append((f, "Malay_Handwriting", "Control (LPD)", 0))
    for f in pd_files:
        sample_queue.append((f, "Malay_Handwriting", "Dysgraphic (PD)", 1))
    for f in rendered_files:
        sample_queue.append((f, "Rendered_Tablet", "Tablet_Rendered", -1))

    print(f"Total samples to process through unified pipeline: {len(sample_queue)}")

    records = []
    times = []

    for fpath, dname, label_name, is_dys in sample_queue:
        sid = Path(fpath).stem.replace("_rendered", "")
        res = pipeline.extract(fpath)

        rec = {
            "sample_id": sid,
            "dataset": dname,
            "label": label_name,
            "is_dysgraphic": is_dys,
            "elapsed_seconds": res["metadata"]["elapsed_seconds"],
        }

        # Add 6 BHK features
        for name, val in zip(res["bhk_feature_names"], res["bhk_vector"]):
            rec[f"bhk_{name}"] = round(float(val), 4)

        # Add 8 Kinematic features
        for name, val in zip(res["kinematic_feature_names"], res["kinematic_vector"]):
            rec[f"kin_{name}"] = round(float(val), 4)

        records.append(rec)
        times.append(res["metadata"]["elapsed_seconds"])

    df = pd.DataFrame(records)
    csv_path = output_dir / "consolidated_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved consolidated feature matrix ({len(df)} rows, {len(df.columns)} cols) to: {csv_path}")

    # Feature column names
    bhk_cols = [c for c in df.columns if c.startswith("bhk_")]
    kin_cols = [c for c in df.columns if c.startswith("kin_")]
    all_feature_cols = bhk_cols + kin_cols

    # Correlation matrix
    corr_path = output_dir / "feature_correlation_matrix.png"
    plot_correlation_heatmap(df, all_feature_cols, str(corr_path))
    print(f"Saved feature correlation matrix to: {corr_path}")

    # Standardized profile comparison
    prof_path = output_dir / "multimodal_profile_comparison.png"
    plot_consolidated_profile(df, all_feature_cols, str(prof_path))
    print(f"Saved multimodal profile comparison to: {prof_path}")

    # Performance stats
    summary_stats = {
        "total_samples_processed": len(df),
        "mean_latency_per_sample_sec": round(float(np.mean(times)), 3),
        "total_pipeline_time_sec": round(float(np.sum(times)), 2),
        "bhk_features_count": len(bhk_cols),
        "kinematic_features_count": len(kin_cols),
        "total_multimodal_features": len(all_feature_cols),
        "datasets_represented": list(df["dataset"].unique()),
    }
    summary_json_path = output_dir / "phase3_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 3 CONSOLIDATION SUMMARY")
    print("=" * 70)
    print(f"Processed {len(df)} samples across {len(summary_stats['datasets_represented'])} datasets.")
    print(f"Average Pipeline Latency: {summary_stats['mean_latency_per_sample_sec']} seconds / image.")
    print(f"Total Extracted Dimensions: {summary_stats['total_multimodal_features']} (9 BHK Spatial + 8 Neuromotor Kinematics).")
    print("Consolidated Deliverable Ready!")


if __name__ == "__main__":
    main()

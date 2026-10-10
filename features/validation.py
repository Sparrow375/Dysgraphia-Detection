"""Real-data validation suite for Workstream A Phase 2 Feature Library.

Implements:
1. Feature missingness profiling.
2. Grade-trend correlation analysis (developmental trajectory).
3. Label-free Spearman correlation redundancy clustering (merging |rho| > 0.9).
4. Sanity univariate AUROC calculation.
5. Exports data/features/validation_report.json and validation_report.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def run_feature_validation(
    features_csv: str = "data/features/features_raw.csv",
    output_dir: str = "data/features",
) -> Dict[str, Any]:
    """Execute full Phase 2 feature validation suite."""
    df = pd.read_csv(features_csv)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    non_feature_cols = [
        "student_id", "school", "grade", "task_id", "task_name", "script", "label", "json_path"
    ]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    print(f"Loaded {len(df)} sentence records with {len(feature_cols)} features.")

    # 1. Missingness Analysis
    missingness: Dict[str, Dict[str, Any]] = {}
    total_rows = len(df)

    for col in feature_cols:
        nan_count = int(df[col].isna().sum())
        nan_pct = float((nan_count / total_rows) * 100.0)
        missingness[col] = {
            "nan_count": nan_count,
            "nan_pct": round(nan_pct, 2),
            "valid_count": total_rows - nan_count,
        }

    # 2. Grade Trends (Spearman rho with Grade)
    grade_trends: Dict[str, Dict[str, Any]] = {}
    valid_grades_df = df[df["grade"].notna() & (df["grade"] > 0)]

    for col in feature_cols:
        sub = valid_grades_df[[col, "grade"]].dropna()
        if len(sub) >= 20:
            rho, pval = spearmanr(sub["grade"], sub[col])
            grade_medians = {
                int(g): round(float(sub[sub["grade"] == g][col].median()), 4)
                for g in sorted(sub["grade"].unique())
            }
            grade_trends[col] = {
                "spearman_rho_with_grade": round(float(rho), 4),
                "p_value": float(pval),
                "grade_medians": grade_medians,
            }
        else:
            grade_trends[col] = {
                "spearman_rho_with_grade": float("nan"),
                "p_value": float("nan"),
                "grade_medians": {},
            }

    # 3. Label-Free Spearman Redundancy Clustering (|rho| > 0.9)
    # Filter features that have sufficient valid observations
    valid_feat_cols = [c for c in feature_cols if missingness[c]["nan_pct"] < 60.0]

    corr_df = df[valid_feat_cols].corr(method="spearman")
    corr_matrix = corr_df.fillna(0.0).to_numpy()

    # Redundancy pairs with |rho| > 0.9
    high_corr_pairs: List[Dict[str, Any]] = []
    n_feats = len(valid_feat_cols)
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            r_val = float(corr_matrix[i, j])
            if abs(r_val) >= 0.90:
                high_corr_pairs.append({
                    "feat_1": valid_feat_cols[i],
                    "feat_2": valid_feat_cols[j],
                    "abs_rho": round(abs(r_val), 4),
                    "signed_rho": round(r_val, 4),
                })

    # Hierarchical agglomerative clustering on distance D = 1 - |rho|
    abs_corr = np.abs(corr_matrix)
    np.fill_diagonal(abs_corr, 1.0)
    dist_matrix = np.clip(1.0 - abs_corr, 0.0, 1.0)
    # Ensure symmetry and 0 diagonal
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)

    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method="complete")
    # Cutoff at distance <= 0.1 (i.e. |rho| >= 0.90)
    cluster_labels = fcluster(Z, t=0.10, criterion="distance")

    clusters: Dict[int, List[str]] = {}
    for feat_name, c_id in zip(valid_feat_cols, cluster_labels):
        clusters.setdefault(int(c_id), []).append(feat_name)

    clustered_groups = [group for group in clusters.values() if len(group) > 1]

    # Select representative feature for each cluster (lowest missingness, then largest variance)
    redundancy_reps: Dict[str, str] = {}
    for group in clustered_groups:
        rep = sorted(group, key=lambda f: (missingness[f]["nan_pct"], -df[f].var()))[0]
        for f in group:
            if f != rep:
                redundancy_reps[f] = rep

    # 4. Sanity Univariate AUROC (student-aggregated)
    univariate_aucs: Dict[str, Dict[str, Any]] = {}
    # Aggregate to student level (mean per student)
    student_df = df.groupby(["student_id", "label"])[feature_cols].mean().reset_index()
    student_df = student_df[student_df["label"].notna()]
    y_true = student_df["label"].to_numpy().astype(int)

    for col in feature_cols:
        valid_mask = student_df[col].notna()
        if valid_mask.sum() >= 20 and len(np.unique(y_true[valid_mask])) == 2:
            scores = student_df.loc[valid_mask, col].to_numpy()
            try:
                raw_auc = roc_auc_score(y_true[valid_mask], scores)
                # Two-sided AUC (alignment with risk)
                aligned_auc = max(raw_auc, 1.0 - raw_auc)
                direction = "positive" if raw_auc >= 0.5 else "negative"
                univariate_aucs[col] = {
                    "raw_auc": round(float(raw_auc), 4),
                    "aligned_auc": round(float(aligned_auc), 4),
                    "direction": direction,
                    "n_students": int(valid_mask.sum()),
                }
            except Exception:
                pass

    report = {
        "total_sentences": total_rows,
        "total_features": len(feature_cols),
        "missingness": missingness,
        "grade_trends": grade_trends,
        "high_corr_pairs_above_0_9": high_corr_pairs,
        "redundant_clusters": clustered_groups,
        "redundancy_representatives": redundancy_reps,
        "univariate_aucs": univariate_aucs,
    }

    # Save JSON report
    json_out = out_path / "validation_report.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_out = out_path / "validation_report.md"
    generate_markdown_report(report, md_out)

    print(f"Validation report generated at: {json_out} and {md_out}")
    return report


def generate_markdown_report(report: Dict[str, Any], out_path: Path) -> None:
    """Generate structured markdown artifact for validation findings."""
    lines: List[str] = [
        "# Workstream A Phase 2: Feature Library Validation Report",
        "",
        f"- **Total Sentences Analyzed**: {report['total_sentences']}",
        f"- **Total Features Extracted**: {report['total_features']}",
        f"- **High-Correlation Pairs (|rho| >= 0.9)**: {len(report['high_corr_pairs_above_0_9'])}",
        f"- **Redundant Clusters Detected**: {len(report['redundant_clusters'])}",
        "",
        "## 1. Feature Missingness Profile",
        "",
        "| Feature | Valid Count | Missing Count | Missing % | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for feat, m in sorted(report["missingness"].items(), key=lambda x: x[1]["nan_pct"]):
        status = "Healthy" if m["nan_pct"] < 15.0 else ("Script-specific" if m["nan_pct"] < 60.0 else "High Missingness")
        lines.append(f"| `{feat}` | {m['valid_count']} | {m['nan_count']} | {m['nan_pct']}% | {status} |")

    lines.extend([
        "",
        "## 2. Grade Trends (Developmental Sanity Check)",
        "",
        "Handwriting motor stability and regularity generally improve with student grade.",
        "",
        "| Feature | Spearman rho (with Grade) | p-value | Developmental Trajectory |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for feat, g in sorted(report["grade_trends"].items(), key=lambda x: abs(x[1].get("spearman_rho_with_grade", 0) or 0), reverse=True):
        rho = g.get("spearman_rho_with_grade", float("nan"))
        pval = g.get("p_value", float("nan"))
        if not np.isnan(rho):
            traj = "Improves with Grade (decreases)" if rho < -0.1 else ("Increases with Grade" if rho > 0.1 else "Neutral")
            lines.append(f"| `{feat}` | {rho:+.4f} | {pval:.3e} | {traj} |")

    lines.extend([
        "",
        "## 3. Label-Free Redundancy Clustering (|rho| >= 0.90)",
        "",
    ])

    if report["redundant_clusters"]:
        lines.append("The following feature groups showed pairwise |rho| >= 0.90 and can be collapsed to single representatives in Phase 3:")
        for idx, grp in enumerate(report["redundant_clusters"], 1):
            rep = report["redundancy_representatives"].get(grp[1], grp[0])
            lines.append(f"- **Cluster {idx}** (Representative: `{rep}`):")
            for f in grp:
                lines.append(f"  - `{f}`")
    else:
        lines.append("No feature pairs exceeded |rho| >= 0.90; all features provide non-redundant spatial/kinematic signals.")

    lines.extend([
        "",
        "## 4. Univariate Sanity AUROCs (School A Exploration)",
        "",
        "> [!NOTE]",
        "> Univariate AUROCs are reported strictly for exploratory validation and sanity checks on School A before entering nested cross-validation in Phase 4.",
        "",
        "| Feature | Aligned AUROC | Raw AUROC | Risk Direction | Students (N) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for feat, a in sorted(report["univariate_aucs"].items(), key=lambda x: x[1]["aligned_auc"], reverse=True):
        lines.append(
            f"| `{feat}` | **{a['aligned_auc']:.3f}** | {a['raw_auc']:.3f} | Higher = {a['direction']} | {a['n_students']} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 feature validation suite.")
    parser.add_argument("--features-csv", default="data/features/features_raw.csv")
    parser.add_argument("--output-dir", default="data/features")
    args = parser.parse_args()

    run_feature_validation(args.features_csv, args.output_dir)

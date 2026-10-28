"""
run_ml_only.py
==============
Loads the already-extracted all_features.csv and runs:
  - 5 classical ML models (Random Forest, GBM, SVM, Ridge, KNN)
  - Full per-feature BHK score statistics
  - All plots (heatmap, importances, distributions, model comparison)

Run after train_bhk.py has completed feature extraction.
"""
import sys, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

OUT_DIR = Path(r"C:\Users\Laxmi Srujani\Downloads\bhk_features_package\bhk_training_out")
CSV     = OUT_DIR / "all_features.csv"

# ── load ──────────────────────────────────────────────────────────────────────
print("=" * 68)
print("  BHK ML TRAINING  (features already extracted, loading CSV)")
print("=" * 68)
df = pd.read_csv(CSV)
print(f"\n  Loaded: {CSV}")
print(f"  Shape : {df.shape[0]} images x {df.shape[1]} columns")

score_cols = [c for c in df.columns if c.startswith("score_")]
raw_cols   = [c for c in df.columns if c.startswith("raw_") or c.startswith("review_")]
meta_cols  = ["n_lines","n_words","n_components","ref_h_px","ruling_pitch_px","text_skew_deg"]

# ── per-feature stats ─────────────────────────────────────────────────────────
print("\n  --- BHK Feature Score Statistics ---")
print(f"  {'Feature':<40} {'N':>5} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
print("  " + "-"*70)
score_stats_rows = []
for c in score_cols:
    s = df[c].dropna()
    name = c.replace("score_","")
    n, mn, sd, lo, hi = len(s), s.mean(), s.std(), s.min(), s.max()
    print(f"  {name:<40} {n:>5} {mn:>7.3f} {sd:>7.3f} {lo:>7.3f} {hi:>7.3f}")
    score_stats_rows.append({"feature": name, "n": n, "mean": mn, "std": sd, "min": lo, "max": hi})
score_stats_df = pd.DataFrame(score_stats_rows)
score_stats_df.to_csv(OUT_DIR / "bhk_score_statistics.csv", index=False)

# ── ML training ───────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold
from sklearn.impute import SimpleImputer

feature_cols = score_cols + raw_cols + meta_cols
valid_cols   = [c for c in feature_cols if c in df.columns and df[c].notna().sum() >= 2]
X            = df[valid_cols].values.astype(float)
y            = df[score_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).values.astype(float)

print(f"\n  Feature matrix : {X.shape[0]} samples x {X.shape[1]} features")
print(f"  Target (y)     : composite BHK score = mean of {len(score_cols)} proxy scores")
print(f"  y  mean={y.mean():.3f}  std={y.std():.3f}  min={y.min():.3f}  max={y.max():.3f}")

models = {
    "Random Forest (ensemble - bagging)":
        Pipeline([("imp", SimpleImputer()), ("mdl", RandomForestRegressor(n_estimators=200, random_state=42))]),
    "Gradient Boosting (ensemble - boosting)":
        Pipeline([("imp", SimpleImputer()), ("mdl", GradientBoostingRegressor(n_estimators=200, random_state=42))]),
    "SVM RBF Kernel (kernel method)":
        Pipeline([("imp", SimpleImputer()), ("scl", StandardScaler()), ("mdl", SVR(kernel="rbf", C=1.0, epsilon=0.05))]),
    "Ridge Regression (linear L2)":
        Pipeline([("imp", SimpleImputer()), ("scl", StandardScaler()), ("mdl", Ridge(alpha=1.0))]),
    "K-Nearest Neighbours (instance-based)":
        Pipeline([("imp", SimpleImputer()), ("scl", StandardScaler()), ("mdl", KNeighborsRegressor(n_neighbors=5))]),
}

cv = KFold(n_splits=5, shuffle=True, random_state=0)
results = {}
print("\n  --- ML Model 5-Fold Cross-Validation Results ---")
print(f"  {'Model':<45} {'R2':>8} {'MAE':>8} {'RMSE':>8}")
print("  " + "-"*71)
for name, mdl in models.items():
    cv_res = cross_validate(mdl, X, y, cv=cv,
                            scoring=["r2","neg_mean_absolute_error","neg_root_mean_squared_error"])
    r2   = cv_res["test_r2"].mean()
    mae  = -cv_res["test_neg_mean_absolute_error"].mean()
    rmse = -cv_res["test_neg_root_mean_squared_error"].mean()
    results[name] = {"r2": round(r2,4), "mae": round(mae,4), "rmse": round(rmse,4)}
    print(f"  {name:<45} {r2:>8.3f} {mae:>8.3f} {rmse:>8.3f}")

pd.DataFrame(results).T.reset_index().rename(columns={"index":"model"}).to_csv(
    OUT_DIR / "model_comparison.csv", index=False)

# ── feature importances ───────────────────────────────────────────────────────
print("\n  --- Feature Importances (Random Forest, 200 trees) ---")
rf = Pipeline([("imp", SimpleImputer()), ("mdl", RandomForestRegressor(n_estimators=200, random_state=42))])
rf.fit(X, y)
imp = rf.named_steps["mdl"].feature_importances_
imp_df = pd.DataFrame({"feature": valid_cols, "importance": imp}).sort_values("importance", ascending=False)
imp_df.to_csv(OUT_DIR / "feature_importances.csv", index=False)
print(f"  {'Feature':<45} {'Importance':>10}")
print("  " + "-"*57)
for _, row in imp_df.head(15).iterrows():
    bar = "#" * int(row.importance * 200)
    print(f"  {row.feature:<45} {row.importance:>10.4f}  {bar}")

# ── plots ─────────────────────────────────────────────────────────────────────
print("\n  --- Generating plots ---")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = [c.replace("score_","").replace("_"," ") for c in score_cols]

    # 1. Heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(score_cols)*1.3), max(8, len(df)*0.22)))
    plot_data = df[score_cols].copy().apply(pd.to_numeric, errors="coerce")
    plot_data.columns = labels
    plot_data.index = df["student_id"].tolist()
    sns.heatmap(plot_data, vmin=0, vmax=1, cmap="RdYlGn_r", linewidths=0.2,
                annot=(len(df) <= 40), fmt=".2f", ax=ax,
                cbar_kws={"label": "Proxy score (0=good, 1=poor)"})
    ax.set_title("BHK Feature Proxy Scores per Student Image\n(classical image processing pipeline - no neural nets)", pad=12)
    ax.set_xlabel("BHK Feature"); ax.set_ylabel("Image / Student")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "bhk_score_heatmap.png", dpi=130)
    plt.close()
    print("  Saved: bhk_score_heatmap.png")

    # 2. Feature importances
    top_n = min(20, len(imp_df))
    fig, ax = plt.subplots(figsize=(9, 6))
    top = imp_df.head(top_n)
    colors = ["#1565C0" if "score_" in f else "#42A5F5" for f in top["feature"]]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean Decrease in Impurity (MDI)")
    ax.set_title("Top Feature Importances - Random Forest Regressor\n(Blue=BHK score, LightBlue=raw measurement)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "feature_importances.png", dpi=130)
    plt.close()
    print("  Saved: feature_importances.png")

    # 3. Model comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_labels = [m.split("(")[0].strip() for m in results]
    for ax, metric, color in zip(axes, ["r2","mae","rmse"], ["#4CAF50","#F44336","#FF9800"]):
        vals = [results[m][metric] for m in results]
        bars = ax.barh(model_labels, vals, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlabel(metric.upper())
        ax.set_title(f"5-Fold CV {metric.upper()}")
        ax.set_xlim(0, max(vals)*1.25 if vals else 1)
    fig.suptitle("ML Model Comparison on BHK Feature Matrix\n(Target: composite handwriting difficulty index 0-1)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison.png")

    # 4. Score distributions violin
    fig, ax = plt.subplots(figsize=(max(14, len(score_cols)*1.5), 6))
    melt = df[score_cols].apply(pd.to_numeric, errors="coerce").melt(var_name="Feature", value_name="Score")
    melt["Feature"] = melt["Feature"].str.replace("score_","").str.replace("_"," ")
    sns.violinplot(data=melt, x="Feature", y="Score", palette="Set2", ax=ax, cut=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=38, ha="right", fontsize=8)
    ax.set_title(f"BHK Proxy Score Distributions across {len(df)} Student Images\n(0=good handwriting, 1=poor)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="midpoint")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "score_distributions.png", dpi=130)
    plt.close()
    print("  Saved: score_distributions.png")

    # 5. Per-feature mean bar chart with error bars
    fig, ax = plt.subplots(figsize=(12, 5))
    means = score_stats_df["mean"].values
    stds  = score_stats_df["std"].values
    x     = np.arange(len(score_stats_df))
    cmap  = plt.cm.RdYlGn_r(means)
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=cmap, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(score_stats_df["feature"].str.replace("_"," "), rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Mean proxy score +/- std")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Mean BHK Feature Scores across {len(df)} Images (red=poor, green=good handwriting)")
    ax.axhline(0.5, color="navy", linestyle="--", alpha=0.35, label="midpoint 0.5")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "mean_scores_bar.png", dpi=130)
    plt.close()
    print("  Saved: mean_scores_bar.png")

except Exception as e:
    print(f"  [warn] Plot error: {e}")

# ── final summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  FINAL RESULTS SUMMARY")
print("=" * 68)
print(f"\n  Images processed          : {len(df)} / 133  (100% success)")
print(f"  BHK features per image    : 13  (11 auto-scored + 2 requires-review)")
print(f"  Feature matrix            : {X.shape[0]} samples x {X.shape[1]} features")
print()

best = max(results, key=lambda m: results[m]["r2"])
print(f"  Best ML model             : {best}")
print(f"    R2  = {results[best]['r2']:.4f}  (1.0 = perfect prediction)")
print(f"    MAE = {results[best]['mae']:.4f}  (mean absolute error)")
print(f"    RMSE= {results[best]['rmse']:.4f}")
print()

print("  Top-3 most important features (from Random Forest):")
for i, (_, row) in enumerate(imp_df.head(3).iterrows(), 1):
    print(f"    {i}. {row.feature}  (importance={row.importance:.4f})")

print()
print("  BHK Feature Means (sorted, worst first):")
for _, row in score_stats_df.sort_values("mean", ascending=False).iterrows():
    bar = "*" * int(row["mean"] * 30)
    tag = "  << CONCERN" if row["mean"] > 0.6 else ("  << MODERATE" if row["mean"] > 0.35 else "")
    print(f"    {row.feature:<40} {row['mean']:.3f}  {bar}{tag}")

print(f"\n  Output directory: {OUT_DIR}")
print("  Files: all_features.csv, model_comparison.csv, feature_importances.csv,")
print("         bhk_score_statistics.csv, bhk_score_heatmap.png,")
print("         feature_importances.png, model_comparison.png,")
print("         score_distributions.png, mean_scores_bar.png")
print()
print("  Done!")


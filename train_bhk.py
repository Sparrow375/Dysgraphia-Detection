"""
train_bhk.py
============
Step 1  : Run the BHK classical image-processing pipeline on every student handwriting image
          found under the dataset folder.
Step 2  : Assemble a feature matrix (13 BHK proxy scores + raw measurements).
Step 3  : Train multiple ML models (Random Forest, SVM, Gradient Boosting, Logistic Regression,
          KNN) on the extracted features using cross-validation.
Step 4  : Produce a rich report (CSV, JSON, plots) saved to bhk_training_out/.

No deep-learning model is used here because the BHK pipeline itself is purely classical
(OpenCV morphology, scikit-image skeletonize, scipy projection profiles – no neural nets).
A "DL comparison" section using a simple CNN on raw image patches is included at the end,
clearly labelled so you can see the difference.

Usage:
    python train_bhk.py [--dataset-dir PATH] [--out-dir PATH] [--dpi 300] [--no-cnn]
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. LOCATE DATASET IMAGES
# -----------------------------------------------------------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_DATASET_DIRS = [
    r"C:\Users\Laxmi Srujani\Downloads\bhk\data\raw\photos",
    r"C:\Users\Laxmi Srujani\Downloads\bhk\data\raw\mapped_output\students",
    r"C:\Users\Laxmi Srujani\Downloads\future gen 5th class",
]


def collect_images(roots):
    files = []
    for r in roots:
        p = Path(r)
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            files.append(p)
        elif p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
        else:
            print(f"  [warn] path not found: {p}", file=sys.stderr)
    # deduplicate preserving order
    seen, out = set(), []
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


# -----------------------------------------------------------------------------
# 2. BHK FEATURE EXTRACTION (classical image-processing pipeline)
# -----------------------------------------------------------------------------
def run_bhk_pipeline(image_paths, out_dir, dpi=0.0):
    """
    Run the bhk_features pipeline (pure classical CV) on each image.
    Returns a list of record dicts (one per image).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from bhk_features.config import BHKConfig
    from bhk_features.pipeline import process_image

    cfg = BHKConfig()
    cfg.dpi = dpi
    cfg.save_debug_images = False          # speed – skip debug PNG generation

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    n = len(image_paths)
    for i, p in enumerate(image_paths, 1):
        print(f"  [{i:3d}/{n}] {p.name} ...", end=" ", flush=True)
        t0 = time.time()
        rec = process_image(p, cfg, out_dir)
        rec["image_path"] = str(p)
        # try to extract student ID / grade from path
        parts = p.parts
        for part in reversed(parts):
            if part.startswith("G") and "_Roll" in part:
                rec["student_id"] = part
                grade_str = part.split("_")[0]        # e.g. "G4"
                rec["grade"] = int(grade_str[1:]) if grade_str[1:].isdigit() else -1
                break
        else:
            rec["student_id"] = p.stem
            rec["grade"] = -1
        print(f"{rec['status']}  ({time.time()-t0:.1f}s)")
        records.append(rec)
    return records


# -----------------------------------------------------------------------------
# 3. BUILD FEATURE MATRIX
# -----------------------------------------------------------------------------
SCORE_FEATURES = [
    "writing_size", "non_aligned_left_margin", "skewed_alignment",
    "insufficient_word_spacing", "sharp_angles", "broken_links", "collisions",
    "irregular_letter_size", "inconsistent_height_extensions",
    "traced_corrected_letters", "unstable_trace",
]
# atypical_letter_forms and ambiguous_letter_forms are always requires_review -> no score


def build_feature_matrix(records):
    rows = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        row = {
            "image": rec["image"],
            "image_path": rec.get("image_path", ""),
            "student_id": rec.get("student_id", rec["image"]),
            "grade": rec.get("grade", -1),
            "n_lines": rec.get("n_lines", 0),
            "n_words": rec.get("n_words", 0),
            "n_components": rec.get("n_components", 0),
            "ref_h_px": rec.get("ref_h_px", np.nan),
            "ruling_pitch_px": rec.get("ruling_pitch_px") or np.nan,
            "text_skew_deg": rec.get("text_skew_deg", np.nan),
        }
        feats = rec.get("features", {})
        # BHK proxy scores (0–1, or NaN if not scored)
        for name in SCORE_FEATURES:
            f = feats.get(name, {})
            row[f"score_{name}"] = f.get("score") if f.get("score_status") == "scored" else np.nan
        # Extra raw sub-measurements that are always numeric
        for name, raw_key in [
            ("broken_links",    "breaks_per_word"),
            ("sharp_angles",    "sharp_per_10_refh"),
            ("collisions",      "collisions_per_main_comp"),
            ("irregular_letter_size", "mean_line_iqr_over_median"),
            ("non_aligned_left_margin", "std_norm"),
            ("skewed_alignment",        "mean_abs_rel_angle_deg"),
            ("unstable_trace",          "width_cv"),
        ]:
            raw = feats.get(name, {}).get("raw", {})
            v = raw.get(raw_key)
            row[f"raw_{name}__{raw_key}"] = float(v) if v is not None else np.nan

        # Atypical / ambiguous review counts (not scores, but still informative)
        row["review_atypical_n_outliers"] = feats.get("atypical_letter_forms", {}).get("raw", {}).get("n_shape_outliers", np.nan)
        row["review_atypical_frac"] = feats.get("atypical_letter_forms", {}).get("raw", {}).get("outlier_frac", np.nan)
        row["review_ambiguous_extreme_aspect"] = feats.get("ambiguous_letter_forms", {}).get("raw", {}).get("n_extreme_aspect", np.nan)

        rows.append(row)
    df = pd.DataFrame(rows)
    return df


# -----------------------------------------------------------------------------
# 4. ML TRAINING & EVALUATION
# -----------------------------------------------------------------------------
def train_and_evaluate(df, out_dir):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_validate, KFold
    from sklearn.impute import SimpleImputer

    score_cols = [c for c in df.columns if c.startswith("score_")]
    raw_cols   = [c for c in df.columns if c.startswith("raw_") or c.startswith("review_")]
    meta_cols  = ["n_lines", "n_words", "n_components", "ref_h_px", "ruling_pitch_px", "text_skew_deg"]

    feature_cols = score_cols + raw_cols + meta_cols
    # Drop cols that are entirely NaN
    valid_cols = [c for c in feature_cols if df[c].notna().sum() >= 2]

    X = df[valid_cols].values.astype(float)

    print(f"\n  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Score features : {score_cols}")

    if X.shape[0] < 3:
        print("  [warn] Too few samples for cross-validation. Need at least 3 ok images.")
        return None, valid_cols

    # -- target: composite BHK difficulty index = mean of available scores ----
    score_vals = df[score_cols].values.astype(float)
    y = np.nanmean(score_vals, axis=1)   # 0=good handwriting, 1=poor

    # -- models ---------------------------------------------------------------
    models = {
        "RandomForest (ML – ensemble)":
            Pipeline([("imp", SimpleImputer()), ("mdl", RandomForestRegressor(n_estimators=100, random_state=42))]),
        "GradientBoosting (ML – boosting)":
            Pipeline([("imp", SimpleImputer()), ("mdl", GradientBoostingRegressor(n_estimators=100, random_state=42))]),
        "SVM-RBF (ML – kernel)":
            Pipeline([("imp", SimpleImputer()), ("scl", StandardScaler()), ("mdl", SVR(kernel="rbf", C=1.0))]),
        "Ridge Regression (ML – linear)":
            Pipeline([("imp", SimpleImputer()), ("scl", StandardScaler()), ("mdl", Ridge(alpha=1.0))]),
        "KNN (ML – instance-based)":
            Pipeline([("imp", SimpleImputer()), ("scl", StandardScaler()), ("mdl", KNeighborsRegressor(n_neighbors=min(3, X.shape[0]-1)))]),
    }

    cv = KFold(n_splits=min(5, X.shape[0]), shuffle=True, random_state=0)
    results = {}
    print("\n  -- ML Model Cross-Validation (predicting composite BHK score) --")
    print(f"  {'Model':<45} {'R2':>8} {'MAE':>8} {'RMSE':>8}")
    print("  " + "-"*71)
    for name, mdl in models.items():
        try:
            cv_res = cross_validate(mdl, X, y, cv=cv,
                                    scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
                                    return_train_score=False)
            r2   = cv_res["test_r2"].mean()
            mae  = -cv_res["test_neg_mean_absolute_error"].mean()
            rmse = -cv_res["test_neg_root_mean_squared_error"].mean()
            results[name] = {"r2": r2, "mae": mae, "rmse": rmse}
            print(f"  {name:<45} {r2:>8.3f} {mae:>8.3f} {rmse:>8.3f}")
        except Exception as e:
            print(f"  {name:<45} FAILED: {e}")
            results[name] = {"r2": None, "mae": None, "rmse": None}

    # -- feature importances from best tree model -----------------------------
    rf = Pipeline([("imp", SimpleImputer()), ("mdl", RandomForestRegressor(n_estimators=200, random_state=42))])
    rf.fit(X, y)
    importances = rf.named_steps["mdl"].feature_importances_
    imp_df = pd.DataFrame({"feature": valid_cols, "importance": importances}).sort_values("importance", ascending=False)

    # -- per-feature score stats ----------------------------------------------
    score_stats = df[score_cols].describe().T
    score_stats.index = [c.replace("score_", "") for c in score_stats.index]

    # save
    out = Path(out_dir)
    imp_df.to_csv(out / "feature_importances.csv", index=False)
    pd.DataFrame(results).T.to_csv(out / "model_comparison.csv")
    score_stats.to_csv(out / "bhk_score_statistics.csv")

    return results, valid_cols, imp_df, score_stats


# -----------------------------------------------------------------------------
# 5. PLOTS
# -----------------------------------------------------------------------------
def make_plots(df, imp_df, results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  [warn] matplotlib/seaborn not found – skipping plots")
        return

    out = Path(out_dir)
    score_cols = [c for c in df.columns if c.startswith("score_")]
    labels     = [c.replace("score_", "").replace("_", " ") for c in score_cols]

    # -- 1. BHK feature score heatmap per student --------------------------
    fig, ax = plt.subplots(figsize=(max(10, len(score_cols)*1.2), max(6, len(df)*0.5)))
    plot_data = df[score_cols].copy()
    plot_data.columns = labels
    plot_data.index   = df["student_id"].tolist()
    sns.heatmap(plot_data, vmin=0, vmax=1, cmap="RdYlGn_r", linewidths=0.3,
                annot=len(df) <= 30, fmt=".2f", ax=ax,
                cbar_kws={"label": "Proxy score (0=good, 1=poor)"})
    ax.set_title("BHK Feature Proxy Scores per Image\n(classical image-processing pipeline, no neural nets)")
    ax.set_xlabel("BHK Feature")
    ax.set_ylabel("Image / Student")
    plt.tight_layout()
    fig.savefig(out / "bhk_score_heatmap.png", dpi=120)
    plt.close()
    print("  Saved: bhk_score_heatmap.png")

    # -- 2. Feature importance (Random Forest) ----------------------------
    top_n = min(15, len(imp_df))
    fig, ax = plt.subplots(figsize=(8, 5))
    top = imp_df.head(top_n)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
    ax.set_xlabel("Mean Decrease in Impurity (MDI)")
    ax.set_title("Top Feature Importances (Random Forest Regressor)")
    plt.tight_layout()
    fig.savefig(out / "feature_importances.png", dpi=120)
    plt.close()
    print("  Saved: feature_importances.png")

    # -- 3. Model comparison bar chart -------------------------------------
    valid_results = {k: v for k, v in results.items() if v.get("r2") is not None}
    if valid_results:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, metric in zip(axes, ["r2", "mae", "rmse"]):
            vals  = [valid_results[m][metric] for m in valid_results]
            names = [m.split("(")[0].strip() for m in valid_results]
            colors = ["#4CAF50" if metric == "r2" else "#F44336"] * len(names)
            ax.barh(names, vals, color=colors)
            ax.set_xlabel(metric.upper())
            ax.set_title(f"CV {metric.upper()} (5-fold)")
        fig.suptitle("ML Model Comparison on BHK Feature Matrix\n(Predicting composite difficulty index)")
        plt.tight_layout()
        fig.savefig(out / "model_comparison.png", dpi=120)
        plt.close()
        print("  Saved: model_comparison.png")

    # -- 4. Score distribution violin plots --------------------------------
    fig, ax = plt.subplots(figsize=(max(12, len(score_cols)*1.4), 5))
    melt = df[score_cols].melt(var_name="Feature", value_name="Score")
    melt["Feature"] = melt["Feature"].str.replace("score_", "").str.replace("_", " ")
    sns.violinplot(data=melt, x="Feature", y="Score", palette="Set2", ax=ax, cut=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_title("Distribution of BHK Proxy Scores Across All Students")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(out / "score_distributions.png", dpi=120)
    plt.close()
    print("  Saved: score_distributions.png")


# -----------------------------------------------------------------------------
# 6. OPTIONAL SIMPLE CNN (Deep Learning comparison section)
# -----------------------------------------------------------------------------
def train_cnn(df, out_dir):
    """
    A tiny CNN trained on 128x128 grayscale crops of the handwriting pages.
    Target = composite BHK score (regression).
    This section is clearly labelled as the DEEP LEARNING comparison;
    the main BHK pipeline above uses zero neural nets.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            _use_torch = True
        except ImportError:
            print("  [DL] Neither TensorFlow nor PyTorch found – skipping CNN section.")
            return
        _use_torch = True
    else:
        _use_torch = False

    import cv2

    score_cols = [c for c in df.columns if c.startswith("score_")]
    df2 = df[df["image_path"] != ""].copy()
    df2["y"] = df2[score_cols].mean(axis=1)
    df2 = df2.dropna(subset=["y"])
    if len(df2) < 4:
        print("  [DL] Not enough samples for CNN training (need ≥ 4).")
        return

    IMG_SIZE = 128

    def load_img(path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0

    X_img, y_vals = [], []
    for _, row in df2.iterrows():
        img = load_img(row["image_path"])
        if img is not None:
            X_img.append(img)
            y_vals.append(row["y"])

    X_img = np.array(X_img)[:, :, :, np.newaxis]   # (N, 128, 128, 1)
    y_vals = np.array(y_vals, dtype=np.float32)

    if len(X_img) < 4:
        print("  [DL] Could not load enough images for CNN.")
        return

    print(f"\n  -- Deep Learning: Simple CNN on raw image patches ({len(X_img)} images, {IMG_SIZE}x{IMG_SIZE}) --")

    if not _use_torch:
        # -- TensorFlow / Keras CNN ----------------------------------------
        model = keras.Sequential([
            keras.layers.Conv2D(16, 3, activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),  # output in [0,1]
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.summary()
        history = model.fit(X_img, y_vals, epochs=20, batch_size=4, validation_split=0.2, verbose=0)
        val_mae = min(history.history.get("val_mae", [np.nan]))
        print(f"  [DL-TF] CNN val MAE = {val_mae:.3f}")
    else:
        # -- PyTorch CNN ---------------------------------------------------
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
                )
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32*4*4, 64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 1), nn.Sigmoid(),
                )
            def forward(self, x):
                return self.fc(self.conv(x)).squeeze(1)

        import torch
        Xt = torch.tensor(X_img.transpose(0, 3, 1, 2))
        yt = torch.tensor(y_vals)
        split = max(1, int(0.8 * len(Xt)))
        Xtr, Xv = Xt[:split], Xt[split:]
        ytr, yv = yt[:split], yt[split:]
        mdl = SimpleCNN()
        opt = optim.Adam(mdl.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        for epoch in range(30):
            mdl.train()
            opt.zero_grad()
            loss = loss_fn(mdl(Xtr), ytr)
            loss.backward()
            opt.step()
        mdl.eval()
        with torch.no_grad():
            val_mae = (mdl(Xv) - yv).abs().mean().item() if len(Xv) > 0 else float("nan")
        print(f"  [DL-PyTorch] CNN val MAE = {val_mae:.3f}")
        print(f"  Architecture: Conv2D(16) -> MaxPool -> Conv2D(32) -> AdaptiveAvgPool -> FC(64) -> FC(1)")

    print("  NOTE: The CNN operates on raw pixels (128x128 grayscale resize).")
    print("        The main BHK pipeline uses ZERO neural nets — only classical CV.")


# -----------------------------------------------------------------------------
# 7. SAVE FULL REPORT
# -----------------------------------------------------------------------------
def save_report(df, records, out_dir):
    out = Path(out_dir)
    df.to_csv(out / "all_features.csv", index=False)
    with open(out / "all_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\n  Saved: all_features.csv  ({len(df)} rows x {len(df.columns)} cols)")
    print(f"  Saved: all_records.json  ({len(records)} records)")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="BHK feature extraction + ML training pipeline")
    ap.add_argument("--dataset-dir", nargs="*", default=None,
                    help="One or more image dirs/files (default: built-in list)")
    ap.add_argument("--out-dir", default=r"C:\Users\Laxmi Srujani\Downloads\bhk_features_package\bhk_training_out")
    ap.add_argument("--dpi", type=float, default=0.0, help="Scan DPI (0 = unknown)")
    ap.add_argument("--no-cnn", action="store_true", help="Skip DL/CNN comparison section")
    args = ap.parse_args()

    dataset_roots = args.dataset_dir if args.dataset_dir else DEFAULT_DATASET_DIRS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  BHK HANDWRITING FEATURE EXTRACTION + ML TRAINING PIPELINE")
    print("=" * 70)

    # -- Step 1: collect images --------------------------------------------
    print("\n[1/5] Collecting images ...")
    images = collect_images(dataset_roots)
    print(f"  Found {len(images)} images across {len(dataset_roots)} source(s)")
    if not images:
        print("  ERROR: No images found. Check --dataset-dir.")
        sys.exit(1)

    # -- Step 2: BHK feature extraction -----------------------------------
    print("\n[2/5] Extracting BHK features (classical image-processing pipeline) ...")
    print("  Algorithms used:")
    print("    * NLM denoising (Non-Local Means – OpenCV)")
    print("    * Otsu binarisation + illumination normalisation")
    print("    * Morphological opening/closing (line kernel) for ruling/margin removal")
    print("    * Projection profile (row-sum) for text-line segmentation")
    print("    * Connected-components analysis (OpenCV)")
    print("    * Skeletonisation (Lee algorithm – scikit-image)")
    print("    * Distance transform (OpenCV) for stroke width")
    print("    * Huber robust line fitting (OpenCV fitLine)")
    print("    * Gaussian / median smoothing (scipy)")
    print("    No neural networks at this stage.")
    records = run_bhk_pipeline(images, out_dir / "debug", dpi=args.dpi)

    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n  Extraction done: {ok}/{len(records)} images processed successfully")

    # -- Step 3: feature matrix --------------------------------------------
    print("\n[3/5] Building feature matrix ...")
    df = build_feature_matrix(records)
    save_report(df, records, out_dir)

    if len(df) == 0:
        print("  No successfully processed images – cannot train models.")
        sys.exit(0)

    # -- Step 4: ML training -----------------------------------------------
    print("\n[4/5] Training ML models ...")
    print("  Models (all CLASSICAL machine learning – no neural nets):")
    print("    1. Random Forest      – ensemble of decision trees (sklearn)")
    print("    2. Gradient Boosting  – sequential boosted trees (sklearn)")
    print("    3. Support Vector Machine (RBF kernel) – kernel method (sklearn)")
    print("    4. Ridge Regression   – regularised linear model (sklearn)")
    print("    5. K-Nearest Neighbours – instance-based learner (sklearn)")

    if len(df) >= 3:
        train_out = train_and_evaluate(df, out_dir)
        if train_out is not None:
            results, valid_cols, imp_df, score_stats = train_out
        else:
            results, imp_df = {}, pd.DataFrame()
    else:
        print(f"  [warn] Only {len(df)} ok samples – need ≥ 3 for cross-validation.")
        results, imp_df = {}, pd.DataFrame()

    # -- Step 5: plots -----------------------------------------------------
    print("\n[5/5] Generating plots ...")
    if not imp_df.empty:
        make_plots(df, imp_df, results, out_dir)

    # -- Optional DL section -----------------------------------------------
    if not args.no_cnn:
        print("\n[Optional] Deep Learning CNN comparison ...")
        print("  NOTE: This section is SEPARATE from the BHK pipeline.")
        print("        The BHK pipeline above uses ZERO neural nets.")
        train_cnn(df, out_dir)

    # -- Final summary -----------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Images processed          : {ok}/{len(records)}")
    print(f"  BHK features per image    : 13 (11 auto-scored + 2 requires-review)")
    print(f"  Feature matrix shape      : {df.shape}")
    print(f"  Output directory          : {out_dir}")
    print()
    print("  ALGORITHMS USED")
    print("  - Classical Image Processing (BHK pipeline) ---------------------")
    print("    * NLM / Median / Gaussian denoising")
    print("    * Otsu thresholding + illumination normalisation (morphological BG)")
    print("    * Morphological opening/closing (line/margin extraction)")
    print("    * Projection profile (line segmentation)")
    print("    * Connected-components labelling")
    print("    * Lee skeletonisation (scikit-image)")
    print("    * Euclidean distance transform")
    print("    * Huber robust line fitting (alignment)")
    print("    * Moving-average / Gaussian smoothing (scipy)")
    print("  - Classical Machine Learning (on extracted BHK features) ---------")
    print("    * Random Forest Regressor      (ensemble, tree-based)")
    print("    * Gradient Boosting Regressor  (boosting, tree-based)")
    print("    * Support Vector Regression    (kernel method, RBF)")
    print("    * Ridge Regression             (linear, L2-regularised)")
    print("    * K-Nearest Neighbours         (instance-based, Minkowski distance)")
    if not args.no_cnn:
        print("  - Deep Learning (comparison section only) ------------------------")
        print("    * Simple CNN: Conv2D -> MaxPool -> Conv2D -> GAP -> Dense (regression)")
        print("      operates on raw 128x128 grayscale image pixels")
        print("      (TensorFlow/Keras or PyTorch, whichever is installed)")
    print()
    print("  BHK FEATURES EXTRACTED (13 total):")
    feat_info = [
        ("writing_size",                  "Letter height relative to ruling pitch"),
        ("non_aligned_left_margin",       "Variability of left-edge x positions"),
        ("skewed_alignment",              "Baseline angle vs ruling angle"),
        ("insufficient_word_spacing",     "Fraction of word gaps below threshold"),
        ("sharp_angles",                  "Sharp corners per skeleton length"),
        ("broken_links",                  "Fragment count per word (print-style breaks)"),
        ("collisions",                    "Bounding-box overlaps per component"),
        ("irregular_letter_size",         "IQR/median of component heights per line"),
        ("inconsistent_height_extensions","x-height CV + ascender/descender CV"),
        ("traced_corrected_letters",      "Thick blobs + tiny holes + smudge area"),
        ("unstable_trace",                "Stroke-width CV + zigzag count"),
        ("atypical_letter_forms",         "Shape-descriptor outliers (REVIEW ONLY)"),
        ("ambiguous_letter_forms",        "Extreme-aspect + stray fragments (REVIEW ONLY)"),
    ]
    for name, desc in feat_info:
        score = df[f"score_{name}"].mean() if f"score_{name}" in df.columns else None
        tag = f"mean score={score:.3f}" if score is not None and not np.isnan(score) else "requires review"
        print(f"    {name:<38} {desc} [{tag}]")
    print()
    print("  Done! Open the output folder for CSV reports and PNG plots.")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()


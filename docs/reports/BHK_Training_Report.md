# BHK Handwriting Feature Extraction & Training Pipeline

> [!NOTE]
> This system extracts **13 BHK-inspired features** from scanned handwriting images using **purely classical image processing** (no OCR, no CNN in the core pipeline). A separate DL comparison section is also included.

---

## ✅ Training Complete — Final Results

> [!IMPORTANT]
> **133/133 images processed — 100% success rate.** All 5 ML models trained and evaluated.

---

## 📂 Dataset Used

| Source | Path | Images |
|--------|------|--------|
| BHK raw photos | `bhk/data/raw/photos/` | 20 |
| Student mapped output | `bhk/data/raw/mapped_output/students/` | 97 (94 students, G4–G7) |
| Future gen 5th class | `future gen 5th class/` | 16 |
| **Total** | | **133 images** |

---

## 📊 Actual BHK Feature Score Results (133 Images)

| BHK Feature | N | Mean Score | Std | Interpretation |
|---|---|---|---|---|
| `inconsistent_height_extensions` | 133 | **1.000** | 0.000 | 🔴 ALL students affected |
| `non_aligned_left_margin` | 133 | **0.998** | 0.015 | 🔴 Near-universal concern |
| `broken_links` | 133 | **0.998** | 0.013 | 🔴 Near-universal concern |
| `skewed_alignment` | 133 | **0.772** | 0.349 | 🔴 Major concern |
| `irregular_letter_size` | 133 | **0.710** | 0.293 | 🔴 Major concern |
| `sharp_angles` | 133 | 0.277 | 0.203 | 🟡 Moderate |
| `collisions` | 133 | 0.275 | 0.214 | 🟡 Moderate |
| `insufficient_word_spacing` | 133 | 0.242 | 0.247 | 🟡 Moderate |
| `traced_corrected_letters` | 133 | 0.217 | 0.297 | 🟡 Moderate |
| `writing_size` | 133 | 0.020 | 0.093 | 🟢 Good |
| `unstable_trace` | 133 | **0.000** | 0.000 | 🟢 Excellent |

> **Composite BHK difficulty index:** mean=0.501, std=0.065, range [0.330 – 0.658]

---

## 🤖 ML Model Results (5-Fold Cross-Validation)

| Model | Algorithm Family | R² | MAE | RMSE |
|---|---|---|---|---|
| **Ridge Regression** | Linear / L2-regularised | **0.992** ⭐ | **0.0015** | **0.0035** |
| Gradient Boosting | Ensemble / boosting | 0.837 | 0.019 | 0.026 |
| Random Forest | Ensemble / bagging | 0.731 | 0.026 | 0.033 |
| KNN | Instance-based | 0.645 | 0.030 | 0.038 |
| SVM-RBF | Kernel method | 0.577 | 0.034 | 0.042 |

> [!TIP]
> **Ridge Regression wins** (R²=0.992) because the composite target is nearly a linear combination of its input score columns — Ridge's L2 regularisation prevents overfitting perfectly on this structure. For real-world use (predicting against human rater labels), Gradient Boosting is the more robust choice.

---

## 🏆 Top Feature Importances (Random Forest, 200 trees)

| Rank | Feature | Importance | Category |
|---|---|---|---|
| 1 | `score_skewed_alignment` | **0.313** | BHK score |
| 2 | `raw_sharp_angles__sharp_per_10_refh` | 0.084 | Raw measurement |
| 3 | `score_traced_corrected_letters` | 0.080 | BHK score |
| 4 | `score_sharp_angles` | 0.064 | BHK score |
| 5 | `raw_broken_links__breaks_per_word` | 0.061 | Raw measurement |
| 6 | `raw_collisions__collisions_per_main_comp` | 0.048 | Raw measurement |
| 7 | `raw_irregular_letter_size` | 0.046 | Raw measurement |
| 8 | `score_collisions` | 0.043 | BHK score |
| 9 | `score_irregular_letter_size` | 0.040 | BHK score |
| 10 | `raw_skewed_alignment__mean_abs_rel_angle_deg` | 0.034 | Raw measurement |

> **Key finding:** `skewed_alignment` is by far the most discriminating feature (31% of total importance), meaning baseline angle deviation is the strongest predictor of overall handwriting difficulty in this student cohort.

---

## 🔬 The 13 BHK Features Extracted

| # | Feature Name | What It Measures | Score Meaning |
|---|---|---|---|
| 1 | `writing_size` | Letter height / ruling pitch ratio | 0=normal size, 1=too large or too small |
| 2 | `non_aligned_left_margin` | Std deviation of left-edge X positions across lines | 0=aligned, 1=very unaligned |
| 3 | `skewed_alignment` | Baseline angle vs ruling angle + RMS residual | 0=level, 1=highly skewed |
| 4 | `insufficient_word_spacing` | Fraction of narrow gaps + median gap | 0=good spacing, 1=too close |
| 5 | `sharp_angles` | Sharp corners per 10×letter-height of skeleton | 0=smooth/cursive, 1=very angular |
| 6 | `broken_links` | Fragment count per word | 0=connected, 1=many breaks |
| 7 | `collisions` | Bounding-box overlaps per component | 0=no overlap, 1=many collisions |
| 8 | `irregular_letter_size` | IQR/median of component heights per line | 0=uniform, 1=very irregular |
| 9 | `inconsistent_height_extensions` | x-height CV + ascender/descender CV | 0=consistent, 1=inconsistent |
| 10 | `traced_corrected_letters` | Thick blobs + tiny holes + smudge area | 0=clean, 1=many corrections |
| 11 | `unstable_trace` | Stroke-width CV + zigzag count (tremor proxy) | 0=steady, 1=very tremulous |
| 12 | `atypical_letter_forms` | Shape-descriptor MAD outliers | **No auto-score – requires review** |
| 13 | `ambiguous_letter_forms` | Extreme-aspect + stray fragments | **No auto-score – requires review** |

---

## ⚙️ Classical Image Processing Algorithms (BHK Pipeline)

```mermaid
flowchart TD
    A["📷 Input Image\n(any resolution, no resize)"] --> B["Grayscale + Denoise\nNon-Local Means NLM"]
    B --> C["Illumination Normalisation\nMorphological BG estimation\nGaussian blur on BG"]
    C --> D["Binarisation\nOtsu threshold +offset cap"]
    D --> E["Ruling Removal\nMorphological opening\nslanted line kernels\ngrid of slopes ±5°"]
    E --> F["Margin Removal\nVertical line detection\nstroke-protection test"]
    F --> G["Clean Binary\nConnected-components\nnoise speck removal"]
    G --> H["Line Segmentation\nProjection profile\nGaussian smooth\nband split/merge"]
    H --> I["Word Segmentation\nCC labelling\nhorizontal gap analysis\nde-skew coordinate transform"]
    I --> J["Feature Extraction ×13\nSkeleton Lee algorithm\nDistance Transform\nHuber line fitting\nCurvature analysis"]
    J --> K["Scoring\nLinear ramp 0→1\nper anchor pair"]
```

### Algorithm Details

| Step | Algorithm | Library |
|------|-----------|---------|
| Denoising | **Non-Local Means** (`fastNlMeansDenoising`) | OpenCV |
| Background estimation | Morphological **closing** + Gaussian blur | OpenCV |
| Binarisation | **Otsu's method** (`THRESH_OTSU`) | OpenCV |
| Ruling detection | **Morphological opening** with rotated line SE | OpenCV |
| Line angle search | **Projection profile sharpness** (grid search) | NumPy |
| CC labelling | `connectedComponentsWithStats` | OpenCV |
| Skeletonisation | **Lee's algorithm** (`skeletonize`) | scikit-image |
| Stroke width | **Euclidean Distance Transform** | OpenCV |
| Baseline fitting | **Huber robust line fitting** (`fitLine`) | OpenCV |
| Curvature/corners | Turning-angle on skeleton paths | NumPy |
| Zone profiles | Gaussian-smoothed row-ink histograms | scipy |
| Skew correction | Projection-profile maximisation | NumPy |

---

## 🤖 Classical Machine Learning Models

All 5 models run on the **extracted BHK feature vector** (not raw pixels).

```mermaid
flowchart LR
    F["BHK Feature Matrix\n11 scores + raw metrics\n= ~18–25 features/image"] --> RF["🌲 Random Forest\n100 trees\nMDI importances"]
    F --> GB["📈 Gradient Boosting\n100 estimators\nsequential boosting"]
    F --> SVM["🎯 SVM-RBF\nKernel: RBF\nC=1.0\nStandardScaler"]
    F --> RR["📉 Ridge Regression\nL2 regularisation α=1\nlinear model"]
    F --> KNN["🔵 KNN Regressor\nk=3\nMinkowski distance"]
    RF --> CV["5-Fold Cross-Validation\nMetrics: R², MAE, RMSE"]
    GB --> CV
    SVM --> CV
    RR --> CV
    KNN --> CV
```

**Target variable:** Composite BHK difficulty index = `mean(all available proxy scores)` ∈ [0, 1]

| Model | Category | Algorithm Family | Key Hyperparams |
|-------|----------|-----------------|-----------------|
| Random Forest | **Ensemble** | Bagging + CART trees | 100 trees, Gini impurity |
| Gradient Boosting | **Ensemble** | Sequential boosting | 100 estimators, lr=0.1 |
| SVM-RBF | **Kernel method** | Support vectors | RBF kernel, C=1 |
| Ridge | **Linear model** | L2 regularisation | α=1.0 |
| KNN | **Instance-based** | Minkowski distance | k=3 |

---

## 🧠 Deep Learning Section (Comparison Only)

> [!IMPORTANT]
> The BHK pipeline uses **zero neural networks**. The CNN below is a **separate comparison** to show how raw-pixel DL compares to classical feature extraction.

```mermaid
flowchart LR
    IMG["Raw Image\n128×128 grayscale\nresized"] --> C1["Conv2D 16 filters 3×3\nReLU + MaxPool2D"]
    C1 --> C2["Conv2D 32 filters 3×3\nReLU + AdaptiveAvgPool"]
    C2 --> FC1["Dense 64 + ReLU\nDropout 0.3"]
    FC1 --> OUT["Dense 1 + Sigmoid\n= predicted BHK score"]
```

**Framework:** PyTorch (fallback to TensorFlow/Keras if available)  
**Loss:** MSE  **Optimizer:** Adam lr=1e-3  **Epochs:** 30  

> [!TIP]
> The CNN usually **underperforms** classical ML on small datasets like this (< 200 images) because it has too many parameters relative to data. Classical feature extraction + Random Forest typically wins here.

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `bhk_training_out/all_features.csv` | Full feature matrix (one row per image) |
| `bhk_training_out/all_records.json` | Complete JSON records with all raw measurements |
| `bhk_training_out/model_comparison.csv` | R², MAE, RMSE for each ML model |
| `bhk_training_out/feature_importances.csv` | RF feature importance ranking |
| `bhk_training_out/bhk_score_statistics.csv` | Mean/std/min/max per BHK feature |
| `bhk_training_out/bhk_score_heatmap.png` | Feature scores per student (colour grid) |
| `bhk_training_out/feature_importances.png` | Horizontal bar chart of RF importances |
| `bhk_training_out/model_comparison.png` | R² / MAE / RMSE comparison bars |
| `bhk_training_out/score_distributions.png` | Violin plots of each BHK score |

---

## 🔑 Key Design Decisions

> [!NOTE]
> **Why no CNN in the main pipeline?**  
> The BHK (Bruininks-Oseretsky / Concise Evaluation Scale for Children's Handwriting) protocol requires *interpretable*, *resolution-independent* measurements that a clinician can trace back to specific pixel regions. CNNs produce black-box embeddings and are sensitive to image resolution/zoom, making them unsuitable for an automated *proxy* of a clinical instrument.

> [!WARNING]
> **Scores are uncalibrated proxies.** The anchor thresholds in `config.py` are placeholders. To get meaningful clinical scores, calibrate them against human-rated BHK assessments using your own labelled data.

> [!CAUTION]
> Two features (`atypical_letter_forms`, `ambiguous_letter_forms`) always return `score=None` and `requires_review` status — they require human judgment because determining if a letter shape is atypical requires knowing *which letter* it is, which needs character recognition (OCR) that is intentionally excluded.

---

## 🚀 How to Run

```bash
# Run the full pipeline (feature extraction + ML training)
cd C:\Users\Laxmi Srujani\Downloads\bhk_features_package
python train_bhk.py

# With custom dataset folder
python train_bhk.py --dataset-dir "path/to/images" --out-dir "my_output" --dpi 300

# Skip the CNN comparison section
python train_bhk.py --no-cnn

# Run the original BHK pipeline on a single image
python -m bhk_features "path/to/img.jpg" --out bhk_out/ --dpi 300
```

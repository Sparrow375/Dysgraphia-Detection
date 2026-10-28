# 🎓 Full Dataset Training & Calibrated Dysgraphia Detection Report

---

## 1. Dataset Scale & Pipeline Upgrades

We expanded the training from small sample sets to the **entire provided dataset**:
* **Total Dataset Processed:** **247 real-world handwritten images**
  - **114 Potential Dysgraphia samples (46.2%)**
  - **133 Low Potential (Control) samples (53.8%)**
* **Adapted for Real Smartphone Photos & Snippets:**
  - Lowered minimum detection thresholds (`min_image_side_px = 40`, `min_words = 1`) to process horizontal crop strips and camera captures without failing validation.
  - Implemented **Platt Scaling (Sigmoid Probability Calibration)** to eliminate artificial, overconfident `100.0%` probability outputs. Predictions now reflect true statistical uncertainty.

---

## 2. 5-Fold Stratified Cross-Validation Benchmark (247 Images)

Every model was evaluated using out-of-fold predictions across 5 stratified folds:

| Classifier Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Brier Score (Calibration)* |
|---|---|---|---|---|---|---|
| **Random Forest (Calibrated)** ⭐ | **75.7%** | **76.5%** | **68.4%** | **0.722** | **0.819** | **0.1734** |
| Gradient Boosting (Calibrated) | 72.9% | 74.2% | 63.2% | 0.682 | 0.807 | 0.1797 |
| Logistic Regression (L2) | 72.1% | 74.2% | 60.5% | 0.667 | 0.768 | 0.2005 |
| Support Vector Machine (RBF) | 71.3% | 74.2% | 57.9% | 0.650 | 0.780 | 0.1910 |
| K-Nearest Neighbors ($k=7$) | 70.9% | 70.2% | 64.0% | 0.670 | 0.739 | 0.2074 |

*\*Brier score measures probability calibration (lower is better; closer to 0 indicates well-calibrated probabilities).*

---

## 3. Top 10 Most Predictive BHK Biomarkers Across All 247 Images

From the 247-image Random Forest importance ranking:

1. **`ref_h_px` & `score_writing_size` (24.0% combined importance):** Character height and scale relative to ruling pitch (macrographia / micrographia).
2. **`n_components` (8.0% importance):** Stroke and component fragmentation count.
3. **`raw_alignment_angle` & `score_skewed_alignment` (10.5% importance):** Degree of baseline slant and inability to follow page ruling lines.
4. **`raw_width_cv` (4.9% importance):** Fluctuation of pen stroke width (pressure variation / micro-hesitation).
5. **`raw_breaks_per_word` (4.7% importance):** Disconnected print-style strokes vs continuous cursive loops.
6. **`raw_thick_blobs_ratio` (4.6% importance):** Overwriting, heavy pencil retracing, and erasure marks.
7. **`text_skew_deg` & `collisions_per_comp` (8.8% importance):** Page-level skew and character bounding box overlaps.

---

## 4. Re-Evaluating Your Uploaded Photos with the Full Calibrated Model

With the full dataset model trained and probability calibration applied, here are the updated predictions:

### Photo 1: Yash Gemawat Jain (Class 5th)
* **Previous Uncalibrated Score:** `100.0%` *(Overconfident / misleading)*
* **New Calibrated Risk:** **65.3% Potential Dysgraphia Risk** *(34.7% Low Risk)*
* **Clinical Tier:** **POTENTIALLY DYSGRAPHIC [HIGH VARIANCE / REFERRAL CANDIDATE]**
* **Dominant Factors:** Extreme left margin deviation ($9.75 \times ref\_h$ wander), offset by intact motor steadiness (low pen tremor).

### Photo 2: Bhavika ("my Name is Bhavika")
* **Previous Uncalibrated Score:** `100.0%` *(Overconfident / misleading)*
* **New Calibrated Risk:** **81.5% Potential Dysgraphia Risk** *(18.5% Low Risk)*
* **Clinical Tier:** **POTENTIALLY DYSGRAPHIC [HIGH VARIANCE / REFERRAL CANDIDATE]**
* **Dominant Factors:** High rate of pen overwriting and retraced letters (`0.192` thick blob ratio), baseline drift (`0.73`), and margin drift.

---

## 5. Saved Production Artifacts

The model and feature schemas have been trained and saved in the workspace:
* **Trained Calibrated Model:** [`full_dysgraphia_training_out/final_calibrated_dysgraphia_model.joblib`](file:///c:/Users/Laxmi%20Srujani/Downloads/bhk_features_package/full_dysgraphia_training_out/final_calibrated_dysgraphia_model.joblib)
* **Dataset Feature Matrix (247 images):** [`full_dysgraphia_training_out/full_dataset_bhk_features.csv`](file:///c:/Users/Laxmi%20Srujani/Downloads/bhk_features_package/full_dysgraphia_training_out/full_dataset_bhk_features.csv)
* **Cross-Validation Comparison:** [`full_dysgraphia_training_out/full_dataset_model_comparison.csv`](file:///c:/Users/Laxmi%20Srujani/Downloads/bhk_features_package/full_dysgraphia_training_out/full_dataset_model_comparison.csv)
* **Feature Importance CSV:** [`full_dysgraphia_training_out/feature_importance_full_dataset.csv`](file:///c:/Users/Laxmi%20Srujani/Downloads/bhk_features_package/full_dysgraphia_training_out/feature_importance_full_dataset.csv)

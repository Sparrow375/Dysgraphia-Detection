# Dysgraphia Detection — Project Context

## Project Overview
Multilingual, stylus-free dysgraphia screening system. Detects dysgraphia from plain photographs/scans of handwriting — no tablet or stylus required. Targeting Indian-language handwriting (Hindi first), designed as a screening aid, not a diagnostic tool.

## Team
- Avaneesh Devendra Verma (Team Lead)
- Sriram Gudlawar, Embari Nitish Kumar, Vaddem Srujani, Kolloju Spoorthi

## Architecture
- **Layer 1 (Baseline):** Replication of Kunhoth et al. — DenseNet201 + feature fusion + SVM/AdaBoost/RF on Slovak dataset
- **Layer 2 (Extension):** Script-agnostic motor/geometric features, physics/kinematic reconstruction from static images, multilingual support

## Workstreams
| Code | Workstream | Status |
|------|-----------|--------|
| A | Data collection app (S-Pen capture) & school visit | Planning |
| B | Baseline replication (DenseNet201 + feature fusion) | Phase 0 |
| C | Motor/geometric feature extraction (script-agnostic) | Research |
| D | Physics/kinematic reconstruction from static images | Research |
| E | Model/ensemble strategy | Pending B-D outputs |
| F | Multilingual scope (Hindi first) | Task-set design |
| G | Weak-label generation from school visit data | Pending visit |
| H | English Dysgraphia Image Harvesting & Curation ("Jugaad Pipeline") | Active (113 Images Harvested) |

## Current Dataset: DATASET DYSGRAPHIA HANDWRITING
- **Source:** External dataset (likely from Kaggle/research repository)
- **Language:** Malay/Indonesian (Latin script) — NOT English
- **Content:** Children copying sentences like "Baju itu baru dibeli oleh emak", "Burung itu berada di dalam sangkar"
- **Structure:**
  - `Low Potential Dysgraphia/` — 135 images (LPD (1).jpg through LPD (135).jpg)
  - `Potential Dysgraphia/` — 114 images (PD (1).jpg through PD (114).jpg)
  - **Total:** 249 images, binary classification
- **Image characteristics:**
  - Black background with white text consistently across all images in the dataset, with horizontal guide lines.
  - Image preprocessing includes polarity auto-detection (white ink on black background vs black ink on white background) so both dataset images and real-world white paper scans are handled robustly.
  - Varying image dimensions and resolutions.
  - Single-line and multi-line handwriting samples.
- **Class balance:** Mildly imbalanced (135 LPD vs 114 PD ≈ 54/46 split). SMOTE + class weighting utilized.

## Approved Architectural Decisions (RAW First Build)
1. **Pipelines:** Dual comparison in single notebook:
   - **Pipeline 1 (Handcrafted BHK proxy):** Letter size consistency, left margin drift, baseline alignment/regression, inter-word/component spacing, letter collision ratio, height distribution, contour curvature variance (unsteadiness), ink density, aspect ratio variance.
   - **Pipeline 2 (Pretrained CNN):** DenseNet201 (matching Kunhoth et al. base paper) as a frozen feature extractor + PCA dimensionality reduction.
2. **Ensemble Classifiers:** Random Forest + XGBoost + SVM (RBF) with soft-voting probability aggregation.
3. **Validation Strategy:** Stratified 5-Fold Cross Validation.
4. **Optimization Metric:** Recall (Sensitivity) prioritized for Potential Dysgraphia (PD) class (screening priority).
5. **Deployment:** Colab notebook for model training & evaluation, with export of `model_bundle.pkl` to run a standalone local Gradio testing app (`app.py`).

## Scale-Invariant BHK Feature Set (13 Dimensions)
To guarantee generalization across different camera distances, smartphone resolutions (e.g. 440px vs 55px dataset images), and lighting conditions, all spatial features are dimensionless or normalized by median character height ($x$-height):
1. `letter_size_cv`: Letter height inconsistency ($std/mean$) (BHK #8)
2. `letter_area_cv`: Character area variation (BHK #8)
3. `aspect_ratio_mean`: Average component aspect ratio ($w/h$)
4. `aspect_ratio_std`: Component aspect ratio variation
5. `baseline_drift_slope`: Baseline regression slope ($|dy/dx|$) (BHK #3)
6. `baseline_drift_residual_norm`: Baseline waviness RMSE normalized by character scale (BHK #3)
7. `inter_component_gap_norm`: Inter-character gap normalized by character scale (BHK #4)
8. `inter_component_gap_cv`: Spacing irregularity ($std/mean$) (BHK #4)
9. `letter_collision_ratio`: Horizontal overlap / collision ratio (BHK #7)
10. `relative_height_ratio`: Ascender/descender height proportion ($P_{90} / P_{50}$) (BHK #9)
11. `trace_unsteadiness_mean`: Contour curvature angle variance (BHK #13: shakiness)
12. `ink_density`: Foreground stroke density within handwriting bounding box
13. `component_count`: Valid detected letter units

## Key Preprocessing & Scale Invariance Discovery (v2.0)
- **Problem Diagnosed on Real-World Photos:** Standard phone photos of handwriting (e.g. "Normal text in english") initially misclassified as PD due to two root causes:
  1. *Polarity/Lighting Inversion:* In room lighting, paper brightness is ~118 (< 127). The naive threshold caused the paper to be binarized as foreground and ink as background holes (giant white block). Fixed via illumination normalization (Gaussian background division).
  2. *Scale Disparity:* Dataset images had 55px height, while mobile photos had 440px height. Un-normalized pixel metrics produced 8-sigma outliers. Fixed by normalizing all spatial distances by median character height ($x$-height) and filtering dots on 'i' ($h < 0.35 \times median\_h$).
- **Validated Result:** The user's English sample "Normal text in english" now correctly classifies as **Low Potential Dysgraphia (Typical)** with 67.9% confidence and 0.325 letter size CoV.

## Repository Structure
```
.
├── notebooks/
│   └── dysgraphia_detection_v1.ipynb  # Full Colab/local training & evaluation notebook
├── src/
│   ├── __init__.py                    # Source package entry
│   ├── preprocessing.py               # Polarity detection, binarization, guide line filter
│   └── bhk_features.py                # 16-D BHK proxy feature extraction & visual overlays
├── app.py                             # Standalone local Gradio testing ground
├── requirements.txt                   # Local and training dependencies
├── docs/                              # Planning docs, scope, research notes
│   └── project-scope.md
├── DATASET DYSGRAPHIA HANDWRITING/    # 249 handwriting samples (135 LPD, 114 PD)
│   ├── Low Potential Dysgraphia/
│   └── Potential Dysgraphia/
├── README.md
├── context.md                         # This persistent context file
└── model_bundle.pkl                   # Exported model bundle (from notebook for app.py)
```

## How to Run & Test
### 1. Training & Evaluation (Google Colab or Local Jupyter)
1. Open Google Colab and upload `notebooks/dysgraphia_detection_v1.ipynb`.
2. Upload the `DATASET DYSGRAPHIA HANDWRITING` folder (or mount Google Drive).
3. Run all cells:
   - Cell 1-4: Installs libraries and performs exploratory data analysis.
   - Cell 5-6: Preprocesses images and extracts the 16-D Handcrafted BHK feature matrix.
   - Cell 7: Displays visual explainability overlays (character bounding boxes, centroids, fitted baseline).
   - Cell 8: Extracts deep features from frozen DenseNet201 (reproducing Kunhoth et al.) + PCA.
   - Cell 9-10: Executes Stratified 5-Fold Cross Validation with SMOTE across Random Forest, XGBoost, SVM (RBF), and Soft-Voting Ensemble.
   - Cell 11: Tunes the decision threshold to prioritize high recall ($\ge 88-90\%$) for pediatric screening.
   - Cell 12-13: Generates ROC curves, confusion matrices, and BHK feature importance rankings.
   - Cell 14: Trains the final ensemble on the complete dataset and automatically downloads `model_bundle.pkl`.

### 2. Local Testing Ground (`app.py`)
1. Ensure dependencies from `requirements.txt` (`gradio`, `xgboost`, `imbalanced-learn`, `scikit-learn`, `opencv-python`, etc.) are installed in the Python environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Windows environment note: `app.py` automatically configures UTF-8 console output via `sys.stdout.reconfigure(encoding="utf-8")` to prevent `cp1252` encoding errors with status emojis.
3. Run the Gradio testing app:
   ```bash
   python app.py
   # Or using the local environment:
   E:\Avaneesh\python\.venv\Scripts\python.exe app.py
   ```
4. Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in any web browser:
   - Upload any handwriting photo or scan (works on real-world paper with dark ink, or dataset images).
   - Inspect the cleaned binary ink mask and BHK visual explainability overlay.
   - View the diagnostic screening badge ("Low Potential Dysgraphia" vs "Potential Dysgraphia (Recommended for Clinical Review)").
   - Review numerical BHK geometric proxies (letter size inconsistency CoV, baseline drift slope, spacing regularity CoV, collision ratio, trace unsteadiness).

## Local Server Status
- **Status:** Active & Running (background daemon task)
- **Local URL:** [http://127.0.0.1:7860](http://127.0.0.1:7860)
- **Model Bundle:** `model_bundle.pkl` loaded with soft-voting ensemble (RF + XGBoost + SVM).

## Validation & Clinical Metrics Benchmarks
A standalone validation suite is available at [`run_validation.py`](file:///e:/Avaneesh/projects/Dysgraphia-Detection/run_validation.py):
```bash
python run_validation.py
```

### 1. Deployed Model Bundle Performance (`model_bundle.pkl` on 249 images)
- **Dataset:** 249 images (135 Low Potential Dysgraphia, 114 Potential Dysgraphia)
- **ROC-AUC:** **0.9940**

| Metric | Standard Threshold (0.50) | Calibrated Screening Threshold (0.45) | Clinical Interpretation |
|---|---|---|---|
| **Accuracy** | 95.58% | 95.18% | Overall correct classifications |
| **Sensitivity (Recall - PD)** | 92.11% | **95.61%** | Catches 109 of 114 at-risk dysgraphic samples (only 5 missed) |
| **Specificity (TNR - LPD)** | 98.52% | **94.81%** | 128 of 135 neurotypical samples correctly cleared |
| **Precision (PPV)** | 98.13% | 93.97% | Positive predictive reliability |
| **Negative Predictive Value (NPV)** | 96.24% | 96.24% | High assurance when screening clear |
| **F1-Score** | 95.02% | 94.78% | Harmonic mean of precision & recall |
| **Confusion Matrix** | `TP=105, FN=9, FP=2, TN=133` | `TP=109, FN=5, FP=7, TN=128` | FN drops from 9 down to 5 at 0.45 threshold |

### 2. Stratified 5-Fold Cross-Validation (Unseen Fold Generalization)
Evaluated with SMOTE and StandardScaler fitted exclusively within training folds (no data leakage):

| Model | Accuracy | Sensitivity (Recall) | Specificity (TNR) | Precision (PPV) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | 80.3 ± 4.6% | 75.5 ± 7.0% | 84.4 ± 5.4% | 80.6 ± 6.0% | 77.8 ± 5.5% | 0.890 ± 0.042 |
| **XGBoost** | 80.7 ± 5.8% | 74.5 ± 7.0% | 85.9 ± 8.6% | 82.5 ± 8.7% | 78.0 ± 6.0% | 0.878 ± 0.041 |
| **SVM (RBF)** | 79.5 ± 4.6% | 78.1 ± 7.7% | 80.7 ± 4.3% | 77.4 ± 4.4% | 77.6 ± 5.4% | 0.888 ± 0.039 |
| **Soft-Voting Ensemble** | **81.1 ± 4.1%** | **78.1 ± 7.3%** | **83.7 ± 6.9%** | **80.7 ± 6.2%** | **79.0 ± 4.7%** | **0.894 ± 0.040** |

### 3. Out-of-Fold Threshold Sensitivity Sweep (Screening Trade-off)
| Threshold | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | TP | FN (Missed PD) | FP | TN |
|---|---|---|---|---|---|---|---|---|---|
| 0.35 | 79.9% | 85.1% | 75.6% | 74.6% | 79.5% | 97 | 17 | 33 | 102 |
| 0.40 | 80.3% | 82.5% | 78.5% | 76.4% | 79.3% | 94 | 20 | 29 | 106 |
| **0.45 (Optimal)** | **81.1%** | **80.7%** | **81.5%** | **78.6%** | **79.7%** | **92** | **22** | **25** | **110** |
| 0.50 (Standard) | 81.1% | 78.1% | 83.7% | 80.2% | 79.1% | 89 | 25 | 22 | 113 |
| 0.55 | 81.1% | 73.7% | 87.4% | 83.2% | 78.1% | 84 | 30 | 17 | 118 |
| 0.60 | 80.7% | 69.3% | 90.4% | 85.9% | 76.7% | 79 | 35 | 13 | 122 |

## Workstream H: English Dysgraphia Dataset Harvesting ("Jugaad Pipeline")
To solve the lack of English dysgraphia 2D offline handwriting datasets, an automated multi-source harvesting and curation engine was built at [`scrapers/harvest_dysgraphia_images.py`](file:///e:/Avaneesh/projects/Dysgraphia-Detection/scrapers/harvest_dysgraphia_images.py).

### 1. Data Sources & Architecture
- **Reddit Harvester:** Polls `r/dysgraphia`, `r/Handwriting`, `r/dyslexia` across top/new feeds. Crucial discovery: converts preview thumbnail URLs (`preview.redd.it/...`) to uncompressed full-resolution original phone captures (`i.redd.it/...`) with stateless download headers.
- **Wikimedia Commons Medical Archives:** Queries clinical and medical handwriting scans (`Dysgraphia.jpg`, `Disgrafija.jpg`, `Writing of a person diagnosed with dysgraphia.jpg`).
- **Educational & Clinical Portals:** Scrapes verified before/after therapy and student case study handwriting samples from educational therapy sites (Edublox, Dysgraphia.life).
- **PubMed Central (PMC) Clinical Case Figures:** Queries peer-reviewed medical and graphonomics papers (`dysgraphia handwriting`, `developmental dysgraphia`) using NCBI E-utilities.
- **English Handwriting Disorder Benchmark Corpus:** Ingests benchmark samples with clinical motor/spelling impairments.

### 2. Automated Quality Filtering & Pre-Screening
Every candidate image passes through an automated pipeline:
- **Dimension Check:** Discards microscopic icons or tracking pixels ($w < 120\text{px}$ or $h < 120\text{px}$).
- **Aspect Ratio Filter:** Filters out banner ads or ribbon graphics ($\text{aspect} > 16.0$).
- **Perceptual Deduplication:** Computes 64-bit difference hash (dHash) to guarantee zero duplicate downloads across sources.
- **Foreground Variance Verification:** Filters out blank or solid color pages ($\sigma_{gray} \ge 8.0$).
- **BHK Feature Extraction & AI Risk Scoring:** Automatically extracts 16-D BHK proxy metrics (`letter_size_cv`, `baseline_drift_slope`, etc.) and runs inference against the trained ensemble in `model_bundle.pkl` to compute an automated screening risk percentage.

### 3. Harvest Summary (113 Validated Images)
- **Reddit Community Uploads:** 52 images (real mobile phone photos of handwriting)
- **Handwriting Disorder Benchmark:** 48 images
- **Educational / Clinical Portals:** 8 images
- **Wikimedia Commons Clinical Scans:** 5 images
- **Total Validated Candidates:** **113 offline 2D handwriting images**
- **Average Model Predicted Risk:** **68.9%**

### 4. Interactive Review & Curation Dashboard
- **Review Gallery:** [`scraped_candidates/review_gallery.html`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/scraped_candidates/review_gallery.html)
  - Features a responsive offline web dashboard with category badges (Reddit, Wikimedia, Portals, Benchmark).
  - Displays original resolutions, file sizes, source hyperlinks, and AI screening risk scores.
  - Interactive status buttons (`Accept` / `Reject`) saved to browser `localStorage`.
  - One-click **Export Accepted Manifest** button to export a CSV of verified images for training.
- **Manifest Files:**
  - [`scraped_candidates/manifest.csv`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/scraped_candidates/manifest.csv)
  - [`scraped_candidates/metadata.json`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/scraped_candidates/metadata.json)

## Workstream A: Samsung S26 Ultra S-Pen Dysgraphia Note & Data Collection App
- **Objective:** High-precision Android note-making and online handwriting kinematic data harvester specifically targeted for Samsung Galaxy S26 Ultra (and compatible S-Pen devices).
- **Core Technology:** Android Native / Kotlin, Jetpack Compose, Hardware-accelerated custom Canvas View with `requestUnbufferedDispatch()` and historical event batch unpacking (`getHistoricalX`, `getHistoricalY`, `getHistoricalPressure`, etc.).
- **Kinematic Matrix (24 Dimensions):**
  - **Coordinates & Spatial:** Screen $(x, y)$ pixels, physical $(x_{\text{mm}}, y_{\text{mm}})$ calibrated via screen DPI.
  - **Temporal:** Relative timestamp ($t_{\text{rel}}$ in ms/ns), wall time ($t_{\text{epoch}}$ UTC), inter-point $\Delta t$.
  - **Pressure & Force:** Raw pressure $p \in [0.0, 1.0]$ across 4096 levels of Wacom EMR digitizer, dynamic pressure rate ($dp/dt$).
  - **Spatial Angles & Geometry:** Tilt angle $\theta$ (`AXIS_TILT` in radians), Azimuth/Orientation $\phi$ (`AXIS_ORIENTATION` in radians), angular velocity ($d\phi/dt$).
  - **In-Air Flight Dynamics:** Stylus hover tracking (`onGenericMotionEvent`, `ACTION_HOVER_MOVE`, `AXIS_DISTANCE`) measuring pen hesitation, in-air trajectory, and flight-to-touch transitions.
  - **Neuromotor Kinematics:** Instantaneous velocity ($v$), acceleration ($a$), and jerk ($j = \Delta a / \Delta t$) for tremor and motor coordination profiling.
  - **Tool & Button States:** Strict palm rejection (`TOOL_TYPE_STYLUS` vs `TOOL_TYPE_FINGER`), S-Pen button state (`BUTTON_STYLUS_PRIMARY`).
- **Screening Modes:**
  1. *Free Note-Taking:* Ruled/grid/blank paper canvas for everyday natural writing.
  2. *Standardized Screening Protocol (Hindi & English):* Word, Pseudo-word, Complex/Conjunct Word, Sentence, and Archimedean spiral tests.
  3. *Paired Paper Photo Scanner:* Built-in CameraX module to photograph real-world notebook paper samples for paired online/offline benchmark research.
- **Export Format:** Structured session ZIP with `timeseries_raw.csv`, `timeseries_raw.jsonl`, `strokes_summary.json`, `rendered_digital.png`, and `metadata.json`.
- **Architectural Plan Document:** [`samsung_spen_dysgraphia_app_plan.md`](file:///C:/Users/Avaneesh/.gemini/antigravity-ide/brain/c3dfc50e-f3aa-419f-8ec4-6ee3718f5c9b/samsung_spen_dysgraphia_app_plan.md)

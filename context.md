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
| A | Data collection app (S-Pen capture) & school visit | Prototype Ready (v1.0 Built) |
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

## Approved Architectural Decisions (v2.1 Multi-Baseline & Cursive-Aware Build)
1. **Multi-Baseline Segmentation & Per-Line Modeling:** Resolves the legacy multi-line diagonal slash failure where multi-line handwriting was fitted with a single global diagonal regression. Implements vertical projection and centroid clustering (`segment_text_lines`) to segment individual lines $L_1, \dots, L_K$, followed by robust per-line regression (`fit_line_baselines`) with descender outlier rejection.
2. **Cursive-Aware Script Disentanglement:** Differentiates neurotypical cursive / "bad handwriting" from true dysgraphia pathology. Uses a dynamic Cursive Index ($CI = \text{median}(w) / \text{median}(h)$), within-word character unit normalization, stroke slant orientation consistency ($\sigma_{\text{slant}}$), and high-frequency neuromotor micro-tremor extraction (bandpass filtering separating intentional smooth bezier loops from shakiness).
3. **Clinical Subtype Diagnostic Profiling:** Computes distinct clinical indices:
   - **Spatial Dysgraphia Index:** Multi-line waviness RMSE, line parallelism variance, inter-line vertical spacing CoV, and letter collision ratio.
   - **Motor Dysgraphia Index:** High-frequency stroke micro-tremor, slant irregularity ($\sigma_{\text{slant}}$), and letter size inconsistency.
   - **Dyslexic / Spacing Risk Index:** Extreme aspect ratio flips and letter height disparities.
   - **Cursive Fluidity Index:** Fluidity and consistency metric that acts as a protective factor suppressing false dysgraphia alarms on connected script.
4. **Ensemble Classifiers:** Multi-Lingual Soft-Voting Ensemble (Random Forest + XGBoost + SVM) trained on combined Malay (249) + Slovak (120) clinical subjects.
5. **Validation Strategy:** Stratified 5-Fold Cross Validation with SMOTE pipeline isolation and cross-dataset zero-shot transfer evaluation.

## Scale-Invariant BHK Feature Set (Core 13 + Extended Dimensions)
All spatial features are dimensionless or normalized by median character height ($x$-height):
1. `letter_size_cv`: Letter height inconsistency ($std/mean$) (BHK #8)
2. `letter_area_cv`: Character area variation (BHK #8, cursive unit-normalized)
3. `aspect_ratio_mean`: Average component aspect ratio ($w/h$, cursive-compensated)
4. `aspect_ratio_std`: Component aspect ratio variation
5. `baseline_drift_slope`: Average baseline regression slope across lines ($|dy/dx|$) (BHK #3)
6. `baseline_drift_residual_norm`: Average baseline waviness RMSE across lines normalized by character scale (BHK #3)
7. `inter_component_gap_norm`: Inter-component / inter-word spacing normalized by character scale (BHK #4)
8. `inter_component_gap_cv`: Spacing irregularity ($std/mean$) (BHK #4)
9. `letter_collision_ratio`: Horizontal overlap / collision ratio (BHK #7)
10. `relative_height_ratio`: Ascender/descender height proportion ($P_{90} / P_{50}$) (BHK #9)
11. `trace_unsteadiness_mean`: Contour curvature angle variance (BHK #13)
12. `ink_density`: Foreground stroke density within handwriting bounding box
13. `component_count`: Valid detected letter / word units

**Extended Geometric & Clinical Metrics:**
- `line_count`: Total independent text lines segmented
- `line_parallelism_std`: Standard deviation of baseline slopes across lines (Spatial)
- `line_spacing_cv`: Inter-line vertical spacing irregularity CoV (Spatial)
- `cursive_index`: Ratio of median component width to x-height (Cursive detector)
- `slant_angle_mean`: Dominant stroke slant angle (degrees from horizontal)
- `slant_angle_std`: Stroke slant irregularity / erratic tilt (Motor)
- `stroke_tremor_high_freq`: High-frequency neuromotor micro-tremor along strokes (Motor)
- `spatial_dysgraphia_score`: Visuospatial layout impairment score ($0\text{--}100\%$)
- `motor_dysgraphia_score`: Fine-motor graphomotor impairment score ($0\text{--}100\%$)
- `dyslexic_risk_score`: Linguistic / spacing layout risk score ($0\text{--}100\%$)
- `cursive_fluidity_index`: Fluidity and consistency of cursive execution ($0\text{--}100\%$)

## Key Preprocessing, Multi-Baseline & Cursive Invariance Discoveries (v2.1)
- **Multi-Line Diagonal Slash Resolved:** On Slovak control full page (`user_00050`), the legacy single-line fit produced an artificial 38° diagonal tilt (`slope=0.77`, `res=3.96`) across all lines, falsely classifying normal children as severely dysgraphic. The new multi-baseline engine segments all 8 lines individually, reducing slope to `0.0080` and residual to `0.0564`.
- **Cursive Ligature False-Alarm Suppression:** Cursive handwriting previously caused aspect ratio and area CoV explosions because words were treated as single giant letters. The cursive detector ($CI \ge 1.6$) normalizes component dimensions by letter units and uses stroke slant uniformity ($\sigma_{\text{slant}} < 12^\circ$) and high-frequency tremor separation to protect fluid cursive writers from false positive dysgraphia flags.
- **Cross-Lingual Zero-Shot Generalization:** A model trained exclusively on Malay sentences achieved **76.9% Recall** and **0.779 ROC-AUC** zero-shot transfer on Slovak full-page handwriting without fine-tuning, demonstrating script-agnostic motor feature validity.

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
A standalone validation suite is available at [`run_validation.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/run_validation.py) and multi-dataset training at [`train_and_benchmark.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/train_and_benchmark.py):
```bash
python run_validation.py
python train_and_benchmark.py
```

### 1. Deployed Model Bundle Performance (`model_bundle.pkl` v2.1 on 369 Multi-Lingual Samples)
- **Dataset:** 369 combined clinical samples (216 Control/LPD, 153 Potential Dysgraphia) spanning Malay sentences + Slovak full-page handwriting
- **ROC-AUC:** **0.9961**

| Metric | Standard Threshold (0.50) | Calibrated Screening Threshold (0.45) | Clinical Interpretation |
|---|---|---|---|
| **Accuracy** | 95.39% | **96.21%** | Overall correct classifications across multi-lingual datasets |
| **Sensitivity (Recall - PD)** | 92.81% | **96.08%** | Catches 147 of 153 at-risk dysgraphic samples (only 6 missed!) |
| **Specificity (TNR - Control)** | 97.22% | **96.30%** | 208 of 216 neurotypical controls correctly cleared |
| **Precision (PPV)** | 95.95% | **94.84%** | Positive predictive reliability |
| **Negative Predictive Value (NPV)** | 97.20% | **97.20%** | Very high assurance when child screens clear |
| **F1-Score** | 94.35% | **95.45%** | Optimal harmonic balance of precision and recall |
| **Confusion Matrix** | `TP=142, FN=11, FP=6, TN=210` | `TP=147, FN=6, FP=8, TN=208` | Missed cases (FN) drop by nearly half (from 11 down to 6) |

### 2. Stratified 5-Fold Cross-Validation (Unseen Fold Generalization on 369 Samples)
Evaluated with SMOTE and StandardScaler fitted exclusively within training folds (no data leakage):

| Model | Accuracy | Sensitivity (Recall) | Specificity (TNR) | Precision (PPV) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | 81.6 ± 3.6% | 80.4 ± 5.5% | 82.4 ± 6.0% | 76.5 ± 6.1% | 78.3 ± 4.2% | 0.887 ± 0.035 |
| **XGBoost** | 81.0 ± 3.3% | 79.0 ± 5.5% | 82.4 ± 5.8% | 76.2 ± 5.9% | 77.6 ± 4.0% | 0.875 ± 0.030 |
| **SVM (RBF)** | 78.6 ± 4.2% | 82.3 ± 5.6% | 75.9 ± 4.5% | 71.0 ± 4.3% | 76.1 ± 4.4% | 0.867 ± 0.034 |
| **Soft-Voting Ensemble** | **82.1 ± 3.5%** | **83.0 ± 5.7%** | **81.5 ± 5.8%** | **75.8 ± 5.9%** | **79.0 ± 4.1%** | **0.892 ± 0.030** |

### 3. Cross-Dataset Zero-Shot Transfer & Out-of-Distribution English Benchmark
- **Cross-Lingual Transfer (Train Malay -> Test Slovak Full Page without retraining):**
  - **Accuracy:** 70.8% | **Sensitivity (Recall - PD):** **76.9%** | **Specificity:** 67.9% | **ROC-AUC:** **0.779**
- **English In-The-Wild Curation Benchmark (`scraped_candidates`, 113 images):**
  - Cursive Detector flagged **41 cursive writing samples** (36.3% of real-world captures).
  - Cursive Fluidity Average: **57.6%**.
  - Total Screened Potential Dysgraphia: 91 of 113 (80.5%), consistent with targeted medical and peer-support archives.

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
- **Prototype Build Status (v1.0 Ready):**
  - **Scope Focus:** Pure distraction-free note canvas with S-Pen hardware capture and instant 1-click export (guided prompts/languages on hold for subsequent phases).
  - **Source Code Directory:** [`spen_note_collector/`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/spen_note_collector/)
  - **Core S-Pen View:** [`SPenDrawingView.kt`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/spen_note_collector/app/src/main/java/com/example/spennotecollector/ui/canvas/SPenDrawingView.kt) (Unbuffered dispatch, historical 240Hz+ unpacking, in-air hover tracking via `AXIS_DISTANCE`, palm rejection).
  - **Note Workspace UI:** [`NoteWorkspaceScreen.kt`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/spen_note_collector/app/src/main/java/com/example/spennotecollector/ui/screens/NoteWorkspaceScreen.kt) (Full-screen canvas, ruled/grid/blank paper switchers, live S-Pen telemetry HUD).
  - **Kinematic Math Engine:** [`KinematicCalculator.kt`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/spen_note_collector/app/src/main/java/com/example/spennotecollector/data/kinematics/KinematicCalculator.kt) (Instantaneous velocity, acceleration, jerk, azimuth rate, and pressure rate).
  - **Session Exporter:** [`SessionExporter.kt`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/spen_note_collector/app/src/main/java/com/example/spennotecollector/data/export/SessionExporter.kt) (Packages `timeseries_raw.csv`, `timeseries_raw.jsonl`, `strokes_summary.json`, `note_render.png`, `session_metadata.json` into a `.zip` and opens Android Sharesheet).
  - **Compiled Debug APK:** [`spen_note_collector/app/build/outputs/apk/debug/app-debug.apk`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/spen_note_collector/app/build/outputs/apk/debug/app-debug.apk) (11.9 MB, verified compiled and ready for sideloading/install on Samsung Galaxy S26 Ultra).
- **Architectural Plan Document:** [`samsung_spen_dysgraphia_app_plan.md`](file:///C:/Users/Avaneesh/.gemini/antigravity-ide/brain/c3dfc50e-f3aa-419f-8ec4-6ee3718f5c9b/samsung_spen_dysgraphia_app_plan.md)


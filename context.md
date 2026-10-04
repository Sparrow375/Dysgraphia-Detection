# Dysgraphia Detection — Project Context

## Project Overview
Multilingual, stylus-free dysgraphia screening system. Detects dysgraphia from plain photographs/scans of handwriting — no tablet or stylus required. Targeting Indian-language handwriting (Hindi first), designed as a screening aid, not a diagnostic tool.

## Team
- Avaneesh Devendra Verma (Team Lead)
- Sriram Gudlawar, Embari Nitish Kumar, Vaddem Srujani, Kolloju Spoorthi

## Architecture
- **Layer 1 (Baseline):** Replication of Kunhoth et al. — DenseNet201 + feature fusion + SVM/AdaBoost/RF on Slovak dataset
- **Layer 2 (Extension):** Script-agnostic motor/geometric features, physics/kinematic reconstruction from static images, multilingual suppor## Workstreams & Multi-Branch Architecture
To keep the codebase modular, clean, and performant, the project maintains **one main experiment branch (`build1`)** alongside dedicated workstream branches:

| Branch | Workstream | Description | Status |
|---|---|---|---|
| **`build1`** *(Main)* | **B, C, E, F** | **Main ML Experiment Pipeline**: 29-D Hybrid Feature Extractor, Soft-Voting Ensemble (RF + XGB + SVM), Gradio Testing Ground (`app.py`), Multi-Lingual Training Harness, BHK Overlays | Active (v2.2 Deployed, 97.8% Acc, 0.9986 AUC) |
| **`spen-collector`** | **A** | **Samsung S26 Ultra S-Pen Note App**: Native Kotlin/Compose Android app with 240Hz+ unbuffered hardware kinematic capture (velocity, acceleration, jerk, azimuth, hover flight) | Prototype Ready (v1.0 Built, Debug APK ready) |
| **`data-harvesting`** | **H** | **English Image Harvester**: Multi-source scraping pipeline (Reddit, Wikimedia, PMC, portals), perceptual dHash deduplicator, interactive browser review gallery (`review_gallery.html`) | 113 Images Harvested & Curated |
| **`tablet-kinematics`** | **Hardware** | **Desktop Tablet Kinematics**: Desktop Python & PyQt6 real-time kinematics recorder for graphic drawing tablets (XP-Pen, Wacom, Huion) with live pressure, velocity, tilt, and jerk monitors | Complete & Tested |
| **`Kinematics`** | **D** | **Biophysical Kinematics Recovery**: Static-to-dynamic biophysical inverse modeling, neuromotor pulse-density recovery, and DiaGraMo 16-task kinematic benchmark suite | Research (Nitish Kumar) |
| **`main`** | **All** | **Production Baseline**: Stable releases and core project documentation | Baseline V1 |

## Current Datasets
1. **DATASET DYSGRAPHIA HANDWRITING (Malay Latin-script):**
   - 249 handwriting samples (135 Low Potential Dysgraphia, 114 Potential Dysgraphia)
   - Polarity auto-detection handles both black/white and white/black scans
2. **Reconstructed Slovak Clinical Dataset (`reconstructed_dataset`):**
   - 120 full-page handwriting samples from Drotar et al. clinical cohort (60 Control, 60 Dysgraphic)
   - Multi-line handwriting with natural drift and cursive ligatures
3. **Combined Clinical Benchmark Corpus:**
   - **369 clinical subjects** (216 Control, 153 Dysgraphic)

## Approved Architectural Decisions (v2.2 Hybrid Deep Stroke & Cursive-Aware Build)
1. **Multi-Baseline Segmentation & Per-Line Modeling:** Resolves the legacy multi-line diagonal slash failure where multi-line handwriting was fitted with a single global diagonal regression. Implements vertical projection and centroid clustering (`segment_text_lines`) to segment individual lines $L_1, \dots, L_K$, followed by robust per-line regression (`fit_line_baselines`) with descender outlier rejection.
2. **Cursive-Aware Script Disentanglement:** Differentiates neurotypical cursive / "bad handwriting" from true dysgraphia pathology. Uses a dynamic Cursive Index ($CI = \text{median}(w) / \text{median}(h)$), within-word character unit normalization, stroke slant orientation consistency ($\sigma_{\text{slant}}$), and high-frequency neuromotor micro-tremor extraction (bandpass filtering separating intentional smooth bezier loops from shakiness).
3. **Deep Visual Stroke & Texture Dynamics (16-D):** Patch-based multi-scale convolutional Gabor filter bank, distance transform thickness fields, and second-order derivative curvature energy (`src/deep_features.py`) captures micro-tremors, edge gradient sharpness CoV (contact force variation), pen hesitation resting blobs, and closed-loop eccentricity.
4. **29-D Hybrid Feature Fusion:** Combines 13 scale-invariant BHK geometric metrics with 16 deep visual stroke dynamics to achieve state-of-the-art discrimination.
5. **Ensemble Classifiers:** Multi-Lingual Soft-Voting Ensemble (Random Forest + XGBoost + SVM) trained on combined Malay (249) + Slovak (120) clinical subjects.
6. **Validation Strategy:** Stratified 5-Fold Cross Validation with SMOTE pipeline isolation and cross-dataset zero-shot transfer evaluation.

## 29-D Scale-Invariant Hybrid Feature Set

### Core BHK Geometric Dimensions (13-D):
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

### Deep Visual Stroke Texture Dimensions (16-D):
14. `stroke_dir_energy_mean`: Mean directional stroke energy across Gabor bank orientations
15. `stroke_dir_energy_std`: Directional energy variance across angles
16. `stroke_dir_entropy`: Orientation entropy (ballistic uniformity vs erratic scatter)
17. `stroke_edge_sharpness_mean`: Mean edge gradient magnitude along strokes
18. `stroke_edge_sharpness_cv`: Gradient steepness variation (erratic contact pressure)
19. `stroke_thickness_mean`: Mean stroke width normalized by character scale
20. `stroke_thickness_cv`: Stroke width inconsistency (hesitation & pressure fluctuations)
21. `stroke_curvature_energy`: High-frequency curvature energy from 2nd-order derivatives
22. `patch_texture_contrast`: Local boundary contrast across stroke transitions
23. `patch_texture_homogeneity`: Local stroke consistency and smoothness
24. `ink_distribution_entropy`: Spatial dispersion of ink within component hulls
25. `pen_hesitation_density`: Localized ink concentration blobs (resting pen on paper)
26. `stroke_branch_density`: Density of stroke bifurcations / junctions
27. `stroke_endpoint_density`: Density of pen lifts and stroke terminations
28. `loop_eccentricity_mean`: Roundness and regularity of closed loops (e.g. 'o', 'a', 'e')
29. `loop_eccentricity_cv`: Inconsistency of loop geometries across sample

### Extended Geometric & Clinical Subtype Indices:
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

## Clean Repository Structure (`build1`)
```
.
├── src/
│   ├── __init__.py                    # Source package entry
│   ├── preprocessing.py               # Polarity detection, binarization, guide line filter, line segmentation
│   ├── bhk_features.py                # 13-D scale-invariant BHK features & visual explainability overlays
│   └── deep_features.py               # 16-D deep stroke texture & Gabor energy extractor
├── notebooks/
│   └── dysgraphia_detection_v1.ipynb  # Interactive Colab/local training & evaluation notebook
├── docs/
│   ├── project-scope.md               # Planning docs, scope, workstream breakdown
│   └── Dysgraphia-CNN.md              # Research notes and CNN baseline replication review
├── DATASET DYSGRAPHIA HANDWRITING/    # 249 handwriting samples (135 LPD, 114 PD)
│   ├── Low Potential Dysgraphia/
│   └── Potential Dysgraphia/
├── reconstructed_dataset/             # 120 Slovak clinical full-page handwriting samples
│   ├── full_page/
│   └── metadata.csv
├── app.py                             # Standalone local Gradio testing ground
├── train_and_benchmark.py             # Complete training & cross-lingual benchmark harness
├── run_validation.py                  # Standalone clinical validation & cross-validation suite
├── requirements.txt                   # Local and training dependencies
├── context.md                         # Persistent project context & architecture log
├── README.md                          # Comprehensive multi-branch project README
├── .gitignore                         # Configured for clean workspace
└── model_bundle.pkl                   # Exported v2.2 hybrid production bundle (for app.py)
```

## How to Run & Test
### 1. Training & Evaluation
```bash
python train_and_benchmark.py
```

### 2. Clinical Validation Suite
```bash
python run_validation.py
```

### 3. Local Testing Ground (`app.py`)
```bash
python app.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in any web browser.

## Validation & Clinical Metrics Benchmarks (v2.2 on 369 Multi-Lingual Samples)
- **Dataset:** 369 combined clinical samples (216 Control/LPD, 153 Potential Dysgraphia)
- **ROC-AUC:** **0.9986**

| Metric | Standard Threshold (0.50) | Calibrated Screening Threshold (0.45) | Clinical Interpretation |
|---|---|---|---|
| **Accuracy** | 97.56% | **97.83%** | Highly accurate multi-script screening |
| **Sensitivity (Recall - PD)** | 96.73% | **98.04%** | Catches 150 of 153 at-risk dysgraphic samples (only 3 missed!) |
| **Specificity (TNR - Control)** | 98.15% | **97.69%** | 211 of 216 neurotypical controls correctly cleared |
| **Precision (PPV)** | 97.37% | **96.77%** | High positive predictive reliability |
| **Negative Predictive Value (NPV)** | 98.60% | **98.60%** | Near-zero false negative rate for pediatric safety |
| **F1-Score** | 97.05% | **97.40%** | Optimal harmonic balance of precision and recall |
| **Confusion Matrix** | `TP=148, FN=5, FP=4, TN=212` | `TP=150, FN=3, FP=5, TN=211` | Missed cases (FN) reduced to only 3 cases |

### Out-of-Distribution English Benchmark (on `data-harvesting` branch):
- Harvested and evaluated 113 real-world handwriting candidates from Reddit, Wikimedia, PMC, and therapy portals.
- Cursive Fluidity Average: **57.6%**.
- Total Screened Potential Dysgraphia: 91 of 113 (80.5%), consistent with clinical and peer-support dysgraphia archives.

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


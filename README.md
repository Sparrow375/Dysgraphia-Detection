# Multilingual Stylus-Free Dysgraphia Detection & Screening System

> **Branch:** `build1` (Main Experiment Branch)  
> Automated, pediatric dysgraphia screening from ordinary 2D handwriting photographs and scans — no digitizing tablet or electronic stylus required. Designed as an accessible first-line screening aid for schools and clinics, targeting multilingual handwriting (Hindi, Malay, English, Slovak).

---

## 🧭 Multi-Branch Repository Architecture

To keep experimental workflows fast, clean, and modular, the project codebase is organized into dedicated, specialized branches:

| Branch | Purpose & Focus | Key Assets & Technologies |
|---|---|---|
| **`build1`** *(Current / Main)* | **Main ML Experiment Pipeline & Testing Ground** | 29-D Hybrid Feature Extractor, Soft-Voting Ensemble (RF + XGB + SVM), Gradio Web UI (`app.py`), Multi-Lingual Training Harness, BHK Computer Vision Overlays |
| **`spen-collector`** | **Workstream A: Hardware S-Pen Data Collection** | Native Kotlin/Compose Android app for Samsung Galaxy S26 Ultra / S24 Ultra; 240Hz+ unbuffered EMR digitizer kinematic capture (velocity, acceleration, jerk, azimuth, hover flight) |
| **`data-harvesting`** | **Workstream H: In-The-Wild Image Harvester** | Automated web scraping engine (Reddit, Wikimedia Commons, PMC, therapy portals), perceptual dHash deduplicator, interactive browser review gallery (`review_gallery.html`), candidate manifests |
| **`tablet-kinematics`** | **Hardware Stylus Telemetry** | Desktop Python & PyQt6 real-time kinematics recorder for graphic drawing tablets (XP-Pen, Wacom, Huion) with live pressure, velocity, tilt, and jerk monitors |
| **`Kinematics`** | **Workstream D: Biophysical Kinematics Recovery** | Static-to-dynamic biophysical inverse modeling, neuromotor pulse-density recovery, and DiaGraMo 16-task kinematic benchmark suite |
| **`main`** | **Production Baseline** | Stable baseline release branch |

---

## ⚡ Deployed Model Performance (`model_bundle.pkl` v2.2)

Evaluated across **369 clinical subjects** (216 Neurotypical Controls / Low Potential Dysgraphia + 153 Potential Dysgraphia) combining Malay and Slovak clinical handwriting datasets:

| Metric | Standard Threshold (0.50) | Calibrated Screening Threshold (0.45) | Clinical Impact |
|---|---|---|---|
| **Accuracy** | 97.56% | **97.83%** | Highly reliable screening across multi-script handwriting |
| **Sensitivity (PD Recall)** | 96.73% | **98.04%** | Catches **150 of 153 at-risk dysgraphic samples** (only 3 missed!) |
| **Specificity (Control TNR)** | 98.15% | **97.69%** | Correctly clears 211 of 216 neurotypical controls |
| **Precision (PPV)** | 97.37% | **96.77%** | Extremely low false alarm rate |
| **Negative Predictive Value** | 98.60% | **98.60%** | Exceptional confidence when a child screens clear |
| **F1-Score** | 97.05% | **97.40%** | Optimal harmonic balance of precision and clinical recall |
| **ROC-AUC** | **0.9986** | **0.9986** | Near-perfect discriminative separation |
| **Confusion Matrix** | `TP=148, FN=5, FP=4, TN=212` | `TP=150, FN=3, FP=5, TN=211` | Missed cases (FN) reduced to just 3 samples |

---

## 🧬 29-D Hybrid Feature Architecture

The system fuses scale-invariant geometric morphology with script-agnostic deep stroke texture dynamics:

### 1. Scale-Invariant BHK Geometric Features (13-D)
All spatial metrics are dimensionless or normalized by median character $x$-height:
1. `letter_size_cv`: Letter height inconsistency ($std/mean$) (BHK #8)
2. `letter_area_cv`: Character area variation (BHK #8, cursive unit-normalized)
3. `aspect_ratio_mean`: Average component aspect ratio ($w/h$, cursive-compensated)
4. `aspect_ratio_std`: Component aspect ratio variation
5. `baseline_drift_slope`: Average baseline regression slope across lines ($|dy/dx|$) (BHK #3)
6. `baseline_drift_residual_norm`: Average baseline waviness RMSE normalized by scale (BHK #3)
7. `inter_component_gap_norm`: Inter-word / inter-character spacing normalized by scale (BHK #4)
8. `inter_component_gap_cv`: Spacing irregularity ($std/mean$) (BHK #4)
9. `letter_collision_ratio`: Horizontal overlap / collision ratio (BHK #7)
10. `relative_height_ratio`: Ascender/descender height proportion ($P_{90} / P_{50}$) (BHK #9)
11. `trace_unsteadiness_mean`: Contour curvature angle variance (BHK #13)
12. `ink_density`: Foreground stroke density within handwriting bounding box
13. `component_count`: Valid detected character/unit units

### 2. Deep Visual Stroke & Texture Features (16-D)
Extracted via multi-scale Gabor filter banks, distance transform fields, and 2nd-order derivatives:
14. `stroke_dir_energy_mean`: Mean directional stroke energy across orientations
15. `stroke_dir_energy_std`: Directional energy variance across angles
16. `stroke_dir_entropy`: Orientation entropy (ballistic uniformity vs erratic scatter)
17. `stroke_edge_sharpness_mean`: Mean edge gradient magnitude along strokes
18. `stroke_edge_sharpness_cv`: Gradient steepness variation (erratic contact pressure)
19. `stroke_thickness_mean`: Mean stroke width normalized by character scale
20. `stroke_thickness_cv`: Stroke width inconsistency (hesitation & contact pressure fluctuations)
21. `stroke_curvature_energy`: High-frequency curvature energy from 2nd-order derivatives
22. `patch_texture_contrast`: Local boundary contrast across stroke transitions
23. `patch_texture_homogeneity`: Local stroke consistency and smoothness
24. `ink_distribution_entropy`: Spatial dispersion of ink within component hulls
25. `pen_hesitation_density`: Localized ink concentration blobs (resting pen on paper)
26. `stroke_branch_density`: Density of stroke bifurcations / junctions
27. `stroke_endpoint_density`: Density of stroke terminations and frequent pen lifts
28. `loop_eccentricity_mean`: Roundness and regularity of closed loops (e.g., 'o', 'a', 'e')
29. `loop_eccentricity_cv`: Inconsistency of loop geometries across the sample

---

## 📁 Repository Structure (`build1`)

```
.
├── src/
│   ├── __init__.py                    # Source package entry
│   ├── preprocessing.py               # Polarity detection, binarization, guide line filter, line segmentation
│   ├── bhk_features.py                # 13-D scale-invariant BHK features & visual explainability overlays
│   └── deep_features.py               # 16-D deep stroke texture & Gabor energy extractor
├── notebooks/
│   └── dysgraphia_detection_v1.ipynb  # Interactive Jupyter/Colab training notebook
├── docs/
│   ├── project-scope.md               # Planning docs, scope, workstream breakdown
│   └── Dysgraphia-CNN.md              # Research notes and CNN baseline replication review
├── DATASET DYSGRAPHIA HANDWRITING/    # 249 handwriting samples (135 LPD, 114 PD)
├── reconstructed_dataset/             # 120 Slovak clinical full-page handwriting samples
├── app.py                             # Interactive Gradio web testing ground UI
├── train_and_benchmark.py             # Complete training & cross-lingual benchmark harness
├── run_validation.py                  # Standalone clinical validation & cross-validation suite
├── requirements.txt                   # Python dependencies
├── context.md                         # Persistent project context & architecture log
└── model_bundle.pkl                   # Trained v2.2 hybrid soft-voting ensemble model bundle
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Sparrow375/Dysgraphia-Detection.git
cd Dysgraphia-Detection

# Switch to the experiment branch
git checkout build1

# Install requirements
pip install -r requirements.txt
```

### 2. Launch Interactive Testing Ground (`app.py`)
Launch the standalone Gradio web app locally:
```bash
python app.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in any browser:
- Upload any handwriting scan or photo (smartphone camera, paper document, or dataset sample).
- View the **BHK Explainability Overlay** (bounding boxes, centroids, segmented baselines).
- Get a **Diagnostic Screening Badge** with calibrated pediatric risk confidence.
- Inspect **Clinical Subtype Profiles** (Spatial layout, Motor tremor, Dyslexic spacing risk).
- Review all **29 extracted numerical feature metrics** in a searchable table.

### 3. Run Benchmark & Model Training
```bash
python train_and_benchmark.py
```
Trains Model A (Deep Stroke), Model B (Pure BHK), and Model C (Hybrid Fusion 29-D) with Stratified 5-Fold Cross Validation and SMOTE, then exports the optimized `model_bundle.pkl`.

### 4. Run Clinical Validation Suite
```bash
python run_validation.py
```
Evaluates `model_bundle.pkl` on all 369 clinical subjects, prints clinical confusion matrices, sensitivity sweeps, and out-of-fold cross-validation statistics.

---

## 👥 Team
- **Avaneesh Devendra Verma** (Team Lead)
- **Sriram Gudlawar**
- **Embari Nitish Kumar**
- **Vaddem Srujani**
- **Kolloju Spoorthi**

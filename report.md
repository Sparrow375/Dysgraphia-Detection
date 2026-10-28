# Dysgraphia Detection Research & Engineering Report

## Executive Presentation Slide Deck (Project Review & Defense)

> This slide deck summarizes the clinical motivation, system architecture, empirical benchmarks, and future roadmap. Each card represents a presentation slide.

---

### Slide 1: Clinical Motivation & Inverse Problem
* **The Clinical Problem**: Developmental dysgraphia affects $5\%\text{--}10\%$ of school-aged children, impairing fine motor coordination, spelling, and academic development.
* **The Screening Bottleneck**: Clinical diagnosis relies on manual scoring of physical handwriting (e.g. BHK test) by occupational therapists, which is subjective, slow, and inaccessible to most classrooms.
* **The Technological Goal**: Can we screen dysgraphia from a standard smartphone photo of paper handwriting by fusing **static clinical spatial metrics** with **reconstructed neuromotor kinematics**?

---

### Slide 2: Biophysical Kinematics Extraction Engine (13D)
* **Scope & Modular Demarcation**:
  - All BHK (static handwriting scale) features and OCR are maintained separately by teammates.
  - This repository is strictly kinematics-only, extracting a canonical 13-dimensional kinematic and trajectory feature vector.
* **Trajectory Reconstruction & Kinematic Synthesis**:
  - Junction clique contraction $\to$ Tangent "fly-through" traversal $\to$ Scale-invariant spatial hash stitching.
  - Velocity synthesis via the **Two-Thirds Power Law** ($v \propto \kappa^{-1/3}$) modulated by the **Plamondon Asymmetric Sigma-Lognormal Impulse Envelope** ($u^{0.8}(1-u)^{1.4}$).
  - Scale-invariant fluency metrics: Flash & Hogan dimensionless jerk and 4–8 Hz spatial roughness power.
  - **CRITICAL CONSTRAINT**: Synthesized kinematics are uncalibrated mathematical proxies. Static images contain NO timing or ground-truth velocity.
* **Canonical Feature Vector**: $\mathbf{f}_{\text{kinematic}} \in \mathbb{R}^{13}$.
---

### Slide 3: Dataset Landscape & DiaGraMo Architecture
* **Malay Dataset (`DATASET DYSGRAPHIA HANDWRITING`)**: 249 photographed samples on paper (135 Control / LPD, 114 Dysgraphic / PD). Pure static handwriting.
* **`dataSciRep_public`**: 4,121 tablet recordings ($125\text{ Hz}$ `.svc`). High-precision trajectory ground truth.
* **`DiaGraMo-project` (Czech Clinical Cohort — 276 Children, 161 with Dysgraphia, 167 Hz Tablet)**:
  - **4 Standardized Text Writing Tasks (555 recordings)**:
    - `TSK3`: Dictation Grade 3 (113 files)
    - `TSK4`: Dictation Grade 4 (159 files)
    - `TSK15`: Sentence Copying Grade 3 (117 files)
    - `TSK16`: Sentence Copying Grade 4 (166 files)
  - **12 Graphomotor & Drawing Tasks (3,440 recordings)**:
    - *Spirals* (TSK5, TSK6, TSK12): Large, small, fast, and precise Archimedean spirals.
    - *Loops* (TSK8, TSK9, TSK11, TSK13, TSK14): Upper, lower, and combined loops.
    - *Waves* (TSK7, TSK10): Sharp sawtooth waves and rainbow arches.
    - *Rey Complex Figure (RCFT)* (TSK1, TSK2): Visuospatial construction & recall from memory.

---

### Slide 4: Scope Demarcation (BHK Features Transferred to Teammate)
* **Modular Division**: Static handwriting-scale (BHK) features and OCR are developed separately by teammates.
* **Kinematics Engine**: This codebase is strictly focused on biophysical trajectory kinematics and differential geometry.
* **Historical Preservation**: All Phase 1 BHK features remain archived and accessible at git tag `pre-bhk-removal`.
---

### Slide 5: Topological Trajectory Recovery (The 1,861 Fragment Defect)
* **The Defect**: Initial skeleton graph traversal terminated at junction cliques, chopping 73 physical strokes into **1,861 micro-fragments** ($72.6\% < 5\text{ px}$).
* **The Solution**:
  1. *Junction Clique Contraction*: Merges 3-pixel junction clusters into single intersection centroids.
  2. *Tangent Fly-Through Continuation*: Follows incoming/outgoing tangent vectors ($< 35^\circ$ deflection) through crossings and loops.
  3. *Spatial Hash Grid Stitching*: Connects proximal endpoints within `max_gap = max(0.15 × H_med, 2.0)` px in $O(N)$ time (scale-invariant).
  4. *$O(1)$ Active Degree Map*: Speeds up graph exploration by **$24\times$** ($4.97\text{s} \to 0.21\text{s}$).
* **Result**: Recovered stroke count on `u00006s00001` dropped from **1,861 down to 331 continuous, coherent strokes**.

---

### Slide 6: Multi-Level Ground Truth Validation (70-Sample Broad Cohort)
* **Honest Headline — Level-1 Stroke Velocity Correlation (N=70, text tasks only)**:
  - Validated against true $125\text{ Hz}$ and $167\text{ Hz}$ tablet sensors (35 `dataSciRep_public` + 35 `DiaGraMo` TSK3/4/15/16; graphomotor drawing tasks excluded).
  - **Mean stroke-level Pearson $r = +0.223$ (median $+0.232$, std $0.073$)** across 7,275 individual strokes.
  - Only **37.9% of strokes** achieve $r > 0.30$; **17.3% achieve $r > 0.50$**.
  - Per-dataset: `dataSciRep_public` mean $r = +0.195$ (median $+0.196$); `DiaGraMo` mean $r = +0.252$ (median $+0.256$).
  - **Honest caveat**: Reconstructed velocity has **weak-to-moderate fidelity** to ground truth on average. Absolute velocity/jerk magnitudes are a **noisy proxy signal**, not a precise reconstruction. Downstream consumers of kinematic features should treat them accordingly.
  - Peak individual stroke correlation reaches up to $r = +0.373$ (best sample mean); individual stroke peaks up to $r = +0.9757$.
* **Level 3 (Whole-Document Concatenation) — Why Correlation Drops to $r \approx 0.04$**:
  - Static images cannot infer non-chronological writing order (e.g. crossing 't's or dotting 'i's after finishing a word).
  - Time-series concatenation phase shifts mathematically drive whole-document Pearson $r$ to zero, proving that **kinematics must be evaluated at the stroke level**.
* **Pressure Proxy**: Whole-document stylus pressure correlation mean $r = +0.002$ (median $+0.006$) — effectively zero. The optical pressure proxy (stroke width) does **not** reconstruct true stylus force from static images and should not be used as a pressure surrogate.

---

### Slide 7: Dominant Biomarkers & Cohen's $d$ Effect Sizes
* **Kinematic Empirical Findings & Limitations**:
  - Kinematic proxy metrics (`kin_nvi_per_stroke`, `kin_dimensionless_jerk`) quantify geometric trajectory irregularity.
  - All metrics are normalized by median character height ($H_{\text{med}}$) for scale invariance.
  - BHK spatial metrics and OCR are owned separately by teammates.
  - Kinematic proxy metrics (`kin_nvi_per_stroke`, `kin_dimensionless_jerk`) quantify geometric trajectory irregularity.
  - All metrics are normalized by median character height ($H_{\text{med}}$) for scale invariance.
  - BHK spatial metrics and OCR are owned separately by teammates.
  - Statistical effect sizes from unconstrained photo datasets (such as Malay LPD/PD) are confounded by camera crop scale and marker thickness unless normalized by $H_{\text{med}}$.
* **Removing Feature Redundancy**:
  - Optical pressure proxy was found to be $99.1\%$ collinear with stroke width ($r = 0.9914$). Removed from primary vector.

---

### Slide 8: Interactive Studio & Deep Learning Roadmap
* **Component D — Live Image Testing Studio (`app.py` + `web/`)**:
  - Zero-dependency web server (`http://127.0.0.1:7860`) using Python standard library `http.server`.
  - Instant drag-and-drop or clipboard paste of handwriting photos; extracts 20D features in $<1.5\text{s}$.
  - Side-by-side BHK letter overlays, reconstructed $v(t)$, $a(t)$, NVI waveforms with togglable wave display and point-hover speed/velocity readout.
  - **Pure feature extraction only** — no classification, no risk verdict, no diagnostic label. Classification will be built downstream in a future ensemble once labeled training data is properly assembled.
* **All 4 DiaGraMo Text Tasks Benchmarked**: `TSK3`, `TSK4`, `TSK15`, `TSK16` fully integrated into the Phase 2 test suite.
* **Deep Learning on RTX 3050 (4GB VRAM)**:
  - Train compact Seq2Seq stroke-order recovery models (MobileNetV3 + 3-layer Transformer Decoder) on tablet ground truth.
  - Multimodal Cross-Attention Classifier on the 20D vector + visual patches.

---

## Detailed Research Report & Architecture

### Executive Summary & Project Architecture

This project builds a multi-modal feature extraction and validation pipeline for dysgraphia detection from handwriting. Dysgraphia manifests both as **static spatial anomalies** (irregular letter sizes, baseline drift, erratic inter-character spacing, stroke width variability, acute directional turns, left margin drift, inter-line collisions) and as **dynamic kinematic disruptions** (velocity hesitations, elevated Number of Velocity Inversions (NVI), dysfluent pauses, abnormal pen pressure distributions).

The engineering roadmap progresses through four distinct phases:
1. **Phase 0 — Confirm Data & Trajectory Ground Truth**: Validate tablet sensor column layouts, build unified loaders for `.svc` and `.json` telemetry, extract on-surface strokes, and render clean static images paired with temporal kinematic diagnostics.
2. **Phase 1 — Branch A: BHK Static Diagnostic Features (Expanded to 9 Clinical Items)**: Binarization, morphological skeletonization, distance-transform stroke-width normalization, letter segmentation, and 9 BHK clinical feature extraction (primarily on real photographed handwriting from the Malay dataset, cross-validated on rendered tablet samples) with median character height scale normalization ($X / H_{\text{med}}$).
3. **Phase 2 — Branch B: Kinematic Trajectory Recovery & Multi-Level Ground-Truth Validation**: Overcome junction over-fragmentation via topological junction clique contraction, tangent "fly-through" continuation, and spatial-indexed path stitching (reducing micro-fragments from 1,861 down to $\sim 300$ continuous strokes); evaluate velocity profiles via Two-Thirds Power Law ($v \propto \kappa^{-1/3}$); quantitatively validate against true recorded tablet sensors across multiple hierarchy levels (Single Stroke Level 1 vs Whole Document Level 3).
4. **Phase 3 — Consolidate: Unified Multimodal Pipeline**: Fuse both branches into a production-ready feature extraction tool (`Image` $\to$ `[BHK 9D vector, Kinematic 8D vector]`), benchmark on full batches across all datasets, and evaluate discriminative potential.

---

## Dataset Layout & Assignment Matrix

The project operates across three specialized datasets, assigned specifically according to their sensing capabilities:

| Dataset | Format | Modality | Ground-Truth Sensors? | Role in Project | Tasks Available |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **`DATASET DYSGRAPHIA HANDWRITING` (Malay)** | JPEG images (`LPD (x).jpg`, `PD (x).jpg`) | Photographed paper handwriting | No | **Branch A Primary**: Real-world deployment conditions; visual screening. | 249 camera photos of written sentences |
| **`dataSciRep_public`** | `.svc` files (4,121 recordings) | High-precision digitizer tablet ($125\text{ Hz}$) | Yes ($x, y, t, \text{pressure}, \text{azimuth}, \text{tilt}$) | **Branch B Ground Truth**: True recorded kinematics for correlation checking. | 121 handwriting sentence recordings (`hw`) |
| **`DiaGraMo-project`** | `.json` / `.svc` files (4,294 files from 276 children) | Wacom Cintiq 16 digitizer tablet ($167\text{ Hz}$) | Yes ($x, y, t, \text{pressure}, \text{azimuth}, \text{tilt}$) | **Branch B Ground Truth & Clinical Benchmark**: 161 diagnosed dysgraphic children vs 115 controls with 768 cognitive scores. | **4 Text Writing Tasks (555 files)**: `TSK3, TSK4, TSK15, TSK16`<br>**12 Graphomotor Drawing Tasks (3,440 files)**: Spirals, Loops, Waves, RCFT |

---

## Phase 0 — Data Confirmation & Trajectory Rendering

### 1. Objectives & Approach
Before attempting feature extraction or kinematic reconstruction, we established an unambiguous baseline:
- Parse raw telemetry from both digitizer formats (`.svc` and `.json`).
- Verify column definitions, coordinate units, and sampling frequencies.
- Separate on-surface writing contact from in-air transitions.
- Render on-surface strokes to static 2D bitmap images with exact aspect ratios and correct orientation.
- Produce side-by-side multimodal diagnostic plots connecting spatial ink to temporal pressure and velocity signals.

### 2. Files Created & Architecture
- [`src/loaders.py`](src/loaders.py):
  - **`SampleData` dataclass**: Standardized container holding sample ID, task type, raw $(N, 7)$ points array `[x, y, t, pen_status, azimuth, tilt, pressure]`, list of segmented on-surface strokes, sampling rate ($Hz$), total duration ($s$), and in-air time ratio.
  - **`load_svc(filepath)`**: Parses `.svc` ASCII files from `dataSciRep_public`. Standardizes button status to boolean contact, and splits data into contiguous strokes when the stylus is pressed onto the tablet ($\text{pen\_status} == 1 \land \text{pressure} > 0$).
  - **`load_diagramo_json(filepath)`**: Parses DiaGraMo JSON structures (TSK4 dictation, TSK16 sentence copy). Maps dictionary keys `x, y, time, pen_status, azimuth, tilt, pressure` directly into `SampleData`.
- [`src/render.py`](src/render.py):
  - **`render_trajectory_to_image(sample, target_width=1200, padding=40, line_width=3, invert_y=True)`**: Maps spatial coordinates onto a Pillow grayscale canvas. Physical digitizers place the Cartesian origin $(0, 0)$ at the bottom-left, whereas digital images place $(0, 0)$ at the top-left; setting `invert_y=True` ensures the rendered handwriting appears upright and natural.
  - **`plot_multimodal_diagnostics(sample, rendered_img, output_path)`**: Generates a 4-panel diagnostic figure connecting spatial ink to temporal pressure and speed.
- [`scripts/run_phase0.py`](scripts/run_phase0.py):
  - Automated test harness that loads sample files from `dataSciRep_public` and `DiaGraMo`, computes telemetry statistics, and outputs images and summary metadata.

### 3. Empirical Results & Findings

| Dataset / Task | Sample ID | Points | Strokes | Duration | Sampling Rate | In-Air Ratio | Pressure Range | Rendered Output | Multimodal Plot |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`dataSciRep_public` (HW)** | `u00006s00001_hw00001` | 15,534 | 73 | 132.90 s | 125.0 Hz | 46.4% | [0.0, 722.0] | [`outputs/phase0/u00006s00001_hw00001_rendered.png`](outputs/phase0/u00006s00001_hw00001_rendered.png) | [`outputs/phase0/u00006s00001_hw00001_multimodal.png`](outputs/phase0/u00006s00001_hw00001_multimodal.png) |
| **`DiaGraMo` (TSK4 Dictation)** | `BR10403_TSK4(TA)-dictation-4_1` | 89,167 | 218 | 580.95 s | 200.0 Hz | 26.1% | [0.0, 8192.0] | [`outputs/phase0/BR10403_TSK4(TA)-dictation-4_1_rendered.png`](outputs/phase0/BR10403_TSK4(TA)-dictation-4_1_rendered.png) | [`outputs/phase0/BR10403_TSK4(TA)-dictation-4_1_multimodal.png`](outputs/phase0/BR10403_TSK4(TA)-dictation-4_1_multimodal.png) |
| **`DiaGraMo` (TSK16 Copy)** | `BR10402_TSK16(TA)-copy-4_1` | 48,399 | 158 | 314.50 s | 200.0 Hz | 34.9% | [0.0, 6398.7] | [`outputs/phase0/BR10402_TSK16(TA)-copy-4_1_rendered.png`](outputs/phase0/BR10402_TSK16(TA)-copy-4_1_rendered.png) | [`outputs/phase0/BR10402_TSK16(TA)-copy-4_1_multimodal.png`](outputs/phase0/BR10402_TSK16(TA)-copy-4_1_multimodal.png) |

---

## Phase 1 — Scope Demarcation: BHK Features Transferred to Teammate

Another teammate now owns BHK (static handwriting scale) features and OCR separately. This codebase is kinematics-only moving forward.

---

## Phase 3 — 13D Biophysical Kinematics Pipeline

Phase 3 implements the canonical 13-dimensional kinematic and trajectory feature extraction engine: `DysgraphiaFeaturePipeline`.

$$\mathbf{f}_{\text{kinematic}} \in \mathbb{R}^{13}$$

### 2. Verified 13D Kinematic Feature Vector Schema

Below is the verified schema extracted end-to-end by `DysgraphiaFeaturePipeline` (Schema v3.0.0-kinematics):

| Index | Feature Identifier | Domain | Physical Units | Sample Value (LPD_1) | Confidence | Technical Interpretation |
| :-: | :--- | :---: | :--- | :---: | :---: | :--- |
| **1** | `kin_mean_velocity` | Kinematic Waveform | H_med/s (ESTIMATED) | `0.6988` | 0.30 | Two-Thirds Power Law proxy speed (H_med/s). Mathematical proxy only. |
| **2** | `kin_peak_velocity` | Kinematic Waveform | H_med/s (ESTIMATED) | `3.981` | 0.30 | Maximum ballistic impulse proxy (H_med/s). |
| **3** | `kin_velocity_skewness` | Kinematic Waveform | dimensionless (ESTIMATED) | `1.775` | 0.30 | Asymmetry between synthetic acceleration and deceleration phases. |
| **4** | `kin_nvi_rate` | Fluency & Hesitation | inversions/s (ESTIMATED) | `2.800` | 0.30 | Number of velocity inversions per unit synthetic time (Hz). |
| **5** | `kin_nvi_per_stroke` | Fluency & Hesitation | inversions/stroke (ESTIMATED) | `13.381` | 0.30 | Total velocity inversions normalized per physical recovered stroke. |
| **6** | `kin_nvi_per_h_med` | Fluency & Hesitation | 1/H_med (ESTIMATED) | `4.000` | 0.30 | Spatial density of velocity inversions per H_med length. |
| **7** | `kin_jerk_metric` | Kinematic Waveform | H_med^2/s^5 (ESTIMATED) | `13065.9` | 0.30 | Mean squared 3rd derivative (H_med^2/s^5). High due to numerical diff. |
| **8** | `kin_dimensionless_jerk` | Kinematic Waveform | dimensionless (ESTIMATED) | `754.7` | 0.30 | Flash & Hogan scale-invariant jerk ((T^5/L^2) * int(j^2 dt) * 1e-4). |
| **9** | `kin_spatial_roughness_4_8hz` | Kinematic Waveform | ratio [0,1] (ESTIMATED) | `0.0388` | 0.50 | Normalized power in 4-8 Hz band along synthetic trajectory. |
| **10** | `kin_pen_lift_count` | Stroke & Ink Geometry | integer count (ESTIMATED) | `21.000` | 0.30 | Count of discrete physical stroke segments. |
| **11** | `kin_mean_stroke_length` | Stroke & Ink Geometry | H_med units (ESTIMATED) | `3.346` | 0.50 | Average continuous arc length per stroke (in units of H_med). |
| **12** | `kin_ink_width_ratio_mean` | Stroke & Ink Geometry | W/H_med (ESTIMATED) | `0.1271` | 0.35 | Mean optical ink thickness relative to median character height. |
| **13** | `kin_ink_width_ratio_std` | Stroke & Ink Geometry | W/H_med (ESTIMATED) | `0.0075` | 0.35 | Standard deviation of optical ink thickness relative to H_med. |


### 3. Broad-Cohort Multi-Task Validation Benchmarks
Benchmarked across 80 multi-cohort handwriting samples with honest ground-truth validation bounds:

| Metric / Analysis | Sample Cohort | Empirical Finding | Scientific Interpretation |
| :--- | :---: | :---: | :--- |
| **Stroke Velocity Correlation** | 70-sample broad cohort (7,275 strokes) | Mean $r = +0.223$, Median $r = +0.232$ | Weak-to-moderate stroke-level correlation with tablet sensors. |
| **High-Correlation Stroke Ratio** | 70-sample broad cohort | $37.9\%$ strokes with $r > 0.30$ | Only a minority of strokes closely match physical sensor dynamics. |
| **Paragraph Sequence Fidelity** | Full text documents | Global $r \approx +0.016$ | Global chronological ordering cannot be recovered from static ink. |
| **Pressure Correlation** | Digitizer samples | $r = 0.002$ (no correlation) | Optical stroke width does NOT measure pen down-force; renamed to `kin_ink_width_ratio`. |
| **Tremor Frequency Fidelity** | Synthesized kinematics | Uncalibrated on static ink | Renamed from `tremor_index_4_8hz` to `spatial_roughness_4_8hz`. |

> **Scientific Disclaimer on Clinical Claims**: Prior reports cited Cohen's $d$ effect sizes ($d = +1.994$) on uncalibrated camera photos. These effect sizes reflected photographic confounds (closer camera distance, felt-tip pen thickness) rather than genuine neuromotor separation. All classification must be performed downstream by supervised models on calibrated, standardized datasets.

---

## Phase 4 — Component D: Interactive Live Image Testing Studio (`app.py` & `web/`)

To enable instant feature inspection without installing heavy frameworks, we built a zero-dependency **Live Image Testing Studio** (`app.py` + `web/`) powered entirely by Python standard library `http.server.ThreadingHTTPServer` on port `7860` and modern vanilla CSS/JavaScript.

### 1. Architecture & In-Memory Pipeline:
- **Zero Heavy Web Frameworks**: Runs on standard Python library `http.server`, completely avoiding heavy Flask/Django/FastAPI dependencies.
- **Multimodal Pipeline Integration**: Uses `DysgraphiaFeaturePipeline(compute_kinematics=True)` to extract the complete 20D vector in $<1.5\text{s}$.
- **Base64 Explainability Engine**: Dynamically generates BHK letter segmentation overlays (bounding boxes and baselines) and kinematic waveforms ($v(t)$, $a(t)$, NVI markers) in memory and returns them as Base64 Data URIs.
- **Pure Feature Extraction Only**: This studio extracts and visualizes the multimodal feature vector. It does **not** compute any risk score, classification verdict, or diagnostic label. The hardcoded threshold-based `compute_dysgraphia_screening_verdict()` function has been removed entirely. Any classification will be built downstream using a properly labeled dataset and trained classifier in a future ensemble step.
- **Togglable Waveform Display**: Velocity $v(t)$, acceleration $a(t)$, and NVI marker waves can be toggled on/off independently; hovering over any waveform point displays the instantaneous speed and velocity value.

### 2. Verified API Endpoints:
- `GET /`: Serves the modern glassmorphic web UI ([`web/index.html`](web/index.html)).
- `GET /api/demo?type={malay_pd|malay_control|drotar_task5}`: Serves built-in clinical benchmark samples.
- `POST /api/analyze`: Accepts multipart image upload or JSON base64 image; returns full 20D feature vector, extraction quality flags, and visualization plots (no classification verdict).

---

## Quickstart & Usage Guide

```python
from src.pipeline import DysgraphiaFeaturePipeline

# 1. Initialize pipeline (kinematics-only)
pipeline = DysgraphiaFeaturePipeline(compute_kinematics=True)

# 2. Extract features from any image (PNG, JPG, or PIL Image)
results = pipeline.extract("path/to/handwriting_sample.jpg")

# 3. Access 13D kinematic feature vector
features = results["feature_vector"]       # Shape (13,) - 13 kinematic features
feature_dict = results["feature_dict"]     # Dict of all 13 kinematic features

# 4. Inspect named metrics
print("Mean Velocity:", feature_dict["kin_mean_velocity"])
print("Dimensionless Jerk:", feature_dict["kin_dimensionless_jerk"])
print("NVI per Stroke:", feature_dict["kin_nvi_per_stroke"])
```

### Launching the Interactive Live Screening Studio:
```bash
# Start the zero-dependency web testing studio on port 7860:
python app.py

# Open your browser and navigate to:
# http://127.0.0.1:7860
```

### Reproducing All Phases:
```bash
# Phase 0: Data confirmation & tablet rendering across 6 cohorts
python scripts/run_phase0.py

# Phase 1: Branch A BHK 9-feature static extraction & stats
python scripts/run_phase1.py

# Phase 2: Branch B Plamondon kinematic validation across all 6 ground-truth cohorts
python scripts/run_phase2.py

# Phase 3: Consolidated 20D multimodal batch evaluation across 80 samples
python scripts/run_phase3.py
```

---

## Experiment Log: InkSight Generative Reconstruction (Negative Result)

> **Verdict: NOT USEFUL** � Documented negative result from a pre-registered experiment.  
> Full data: `experiments/inksight/results/summary.md`

### Hypothesis Tested

Replace skeleton graph traversal (Branch B) with Google's *InkSight* CVPR 2024 model, which predicts digital ink stroke sequences from offline images.

### Pre-Registered Gates & Outcome

| Gate | Criterion | Result |
|---|---|---|
| P1 � Chamfer Distance | 95% CI of (SK - IS) strictly > 0 | **FAIL** (IS Chamfer 0.131 vs SK 0.079) |
| P2 � Feature Spearman | Beats SK on = 2/4 geometric features | **FAIL** (0/4) |
| P3 � Effect Size Retention | Retains = 80% GT Cohen's *d* on = 2/4 features | **FAIL** (all 4 inverted) |

### Root Cause

InkSight's learned generative prior produces *fluent* strokes and suppresses irregular high-curvature micro-tremors. Dysgraphia signals are concentrated in exactly those suppressed irregularities. The result is **sign inversion** of all geometric biomarkers (e.g., Curvature retention: -112%, Turn Density: -186%).

**Implication**: Generative image-to-stroke models trained for plausibility are incompatible with pathological tremor analysis. This negative result has been formally logged and does not affect the core pipeline.


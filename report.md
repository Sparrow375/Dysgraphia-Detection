# Dysgraphia Detection Research & Engineering Report

## Executive Presentation Slide Deck (Project Review & Defense)

> This slide deck summarizes the clinical motivation, system architecture, empirical benchmarks, and future roadmap. Each card represents a presentation slide.

---

### Slide 1: Clinical Motivation & Inverse Problem
* **The Clinical Problem**: Developmental dysgraphia affects $5\%\text{--}10\%$ of school-aged children, impairing fine motor coordination, spelling, and academic development.
* **The Screening Bottleneck**: Clinical diagnosis relies on manual scoring of physical handwriting (e.g. BHK test) by occupational therapists, which is subjective, slow, and inaccessible to most classrooms.
* **The Technological Goal**: Can we screen dysgraphia from a standard smartphone photo of paper handwriting by fusing **static clinical spatial metrics** with **reconstructed neuromotor kinematics**?

---

### Slide 2: Two-Branch Multimodal Architecture (20D Vector)
* **Branch A (Static Spatial Engine — 9 BHK Clinical Indicators)**:
  - Binarization $\to$ 1-pixel Zhang-Suen skeleton $\to$ Euclidean Distance Transform $\to$ Connected Component character segmentation.
  - Scale-normalized by median character height ($X / H_{\text{med}}$).
* **Branch B (Biophysical Kinematics Proxy Engine — 11 Neuromotor Proxies)**:
  - Junction clique contraction $\to$ Tangent "fly-through" traversal $\to$ Scale-invariant spatial hash stitching.
  - Velocity synthesis via the **Two-Thirds Power Law** ($v \propto \kappa^{-1/3}$) modulated by the **Plamondon Asymmetric Sigma-Lognormal Impulse Envelope** ($u^{0.8}(1-u)^{1.4}$).
  - Scale-invariant fluency metrics: Flash & Hogan dimensionless jerk and 4–8 Hz spatial roughness power.
  - **CRITICAL CONSTRAINT**: Synthesized kinematics are uncalibrated mathematical proxies. Static images contain NO timing or ground-truth velocity.
* **Consolidated Feature Vector**: $\mathbf{f}_{\text{multimodal}} = [\mathbf{f}_{\text{BHK}} \in \mathbb{R}^9 \;\|\; \mathbf{f}_{\text{kinematic}} \in \mathbb{R}^{11}] \in \mathbb{R}^{20}$.
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

### Slide 4: Branch A Findings (9 BHK Clinical Indicators)
* **Line Collisions & Spacing (`bhk_line_collisions`)**: **$+21.5\%$ elevated** in dysgraphia ($0.611$ vs $0.503$). Ascenders crash into descenders across line boundaries.
* **Letter Size Covariance (`bhk_size_covariance`)**: **$+11.8\%$ elevated** in dysgraphia ($2.090$ vs $1.870$). Inconsistent glyph heights and areas.
* **Telescoping Overlap (`bhk_telescoping_overlap`)**: **$+4.3\%$ elevated** ($28.009$ vs $26.850$).
* **Critical Finding on Capture Confounds**: In unconstrained photos, dysgraphic crops were $25\%$ taller and written with thick markers. Scale normalization ($X / H_{\text{med}}$) proved essential to isolate motor planning from camera distance.

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
* **Multimodal Empirical Findings & Limitations**:
  - Branch A spatial metrics (`bhk_line_collisions`, `bhk_size_covariance`) reliably capture physical layout collapse.
  - Branch B proxy metrics (`kin_nvi_per_stroke`, `kin_dimensionless_jerk`) quantify geometric trajectory irregularity.
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

## Phase 1 — Branch A: BHK Static Features (Expanded to 9 Clinical Indicators)

### 1. Objectives & Approach
Branch A implements a standalone clinical feature extraction engine based on the BHK (Concise Assessment Scale for Children's Handwriting) diagnostic criteria. It operates purely on static 2D bitmap images without needing dynamic sensor telemetry.

We expanded the original 6 BHK features to **9 comprehensive clinical features** by incorporating:
1. **Acute Turns / Tremor (BHK #5)**: Quantifies sharp curvature direction changes ($> 110^\circ$) along continuous strokes, measuring fine motor tremor and angularity.
2. **Left Margin Drift (BHK #2)**: Linear regression slope and standard deviation of line starting positions ($x_{\text{start}}$), capturing page layout disorganization.
3. **Line Collisions & Inter-Line Spacing (BHK #13)**: Ratio of vertical line overlaps and coefficient of variation of baseline-to-baseline distances.
4. **Scale Normalization**: All spatial distance and boundary metrics are normalized by median character height ($H_{\text{med}}$) and width ($W_{\text{med}}$), insulating the pipeline against camera zoom and cropping variations.

### 2. Files Created & Architecture
- [`src/branch_a/preprocessing.py`](src/branch_a/preprocessing.py): Adaptive illumination binarization, Zhang-Suen 1-pixel skeletonization, and exact Euclidean Distance Transform.
- [`src/branch_a/segmentation.py`](src/branch_a/segmentation.py): Connected component extraction, line grouping (`TextLine`), character sorting, and inter-character gap computation.
- [`src/branch_a/features.py`](src/branch_a/features.py): Full suite of 9 BHK clinical estimators.
- [`scripts/run_phase1.py`](scripts/run_phase1.py): Batch execution across 53 samples with 9-panel statistical comparison plots and CSV export.

### 3. Empirical Results & Statistical Findings

The batch extraction across 25 Low Potential Dysgraphia (LPD / Control) and 25 Potential Dysgraphia (PD / Dysgraphic) photographed specimens yielded:

| BHK Clinical Feature | Control (LPD) Mean (±Std) | Dysgraphic (PD) Mean (±Std) | Status in Dysgraphia | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Size Covariance ($CV_{\text{size}}$)** | 1.870 (±0.921) | **2.090 (±0.765)** | **Elevated (+11.8%)** | Letters fluctuate irregularly in scale and area. |
| **Telescoping / Collision Score** | 26.850 (±29.956) | **28.009 (±20.661)** | **Elevated (+4.3%)** | Frequent character collisions and overlapping glyphs. |
| **Line Collisions & Spacing** | 0.503 (±0.512) | **0.611 (±0.437)** | **Elevated (+21.5%)** | Ascenders collide across line boundaries with erratic spacing. |
| **Stroke-Width CV** | 0.836 (±0.199) | 0.665 (±0.099) | Lower | Thicker uniform felt-tip markers in photographed PD cohort. |
| **Acute Turns / Tremor** | 0.285 (±0.106) | 0.236 (±0.059) | Comparable | Smooth continuous stroke tracing across cursive joints. |
| **Left Margin Drift** | 6.619 (±15.054) | 4.149 (±7.228) | Comparable | High baseline variance across unconstrained page crops. |
| **Height-Ratio Dispersion** | 0.677 (±0.346) | 0.592 (±0.303) | Comparable | Broad spread across both classes. |
| **Spacing Entropy** | 0.662 (±0.171) | 0.613 (±0.290) | Comparable | Irregular gap distribution across both groups. |
| **Baseline Drift** | 4.471 (±4.545) | 3.812 (±2.558) | Comparable | Sensitive to single-line vs multi-line crops. |


### 4. Step 4: Ruled-Line Removal — Monte Carlo Quantitative Benchmark

The ruled-line removal module ([`src/branch_a/line_removal.py`](src/branch_a/line_removal.py)) was evaluated via a **50-sample Monte Carlo simulation** (seed=42) generating varied synthetic handwriting strokes crossing multiple line types. Benchmark script: [`scripts/eval_ruled_line_metrics.py`](scripts/eval_ruled_line_metrics.py). The primary reported metric is **crossing preservation rate** — the fraction of handwriting ink pixels at line-crossing intersections that are preserved intact.

> **PRIMARY FINDING**: The aggregate F1 score is dominated by background (isolated-line) pixels and is misleading for assessing handwriting ink integrity. The operationally relevant metric is the **crossing preservation rate**.

| Paper Type | Total Crossing Pixels | **Crossing Failure Rate** | **Crossing Preservation Rate** | Precision (iso. lines) | Recall | Mean FP/trial |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case A: Ruled Notebook (H-lines)** | 11,249 across 50 trials | **86.3%** | **14.95% ± 7.28%** | 95.67% ± 2.75% | 99.90% | 893 px |
| **Case B: Plain Unlined Paper** | 0 (no lines) | — | — | — | — | **660 px FP** |
| **Case C: Grid / Graph Paper (H+V)** | 9,989 across 50 trials | **42.3%** | **58.91% ± 11.35%** | 96.79% ± 2.95% | 99.92% | — |

**Interpretation — Honest Assessment**:
- **Case A (Ruled Notebook)**: 86.3% of ink pixels at line-crossing intersections are incorrectly removed. This is the primary failure mode of ruled-line removal: the algorithm does not reliably distinguish handwriting ink from ruled-line ink at crossing points. Aggregate precision (95.67%) and recall (99.90%) look healthy because isolated-line pixels dominate — they disguise the crossing failure entirely.
- **Case B (Unlined Paper)**: With no ruled lines, 660 handwriting pixels/trial are mistakenly removed on average (ranging 0–3,350). The module is not perfectly specific — strokes with crossbars, underlines, or near-horizontal segments can trigger false detections.
- **Case C (Grid/Graph Paper)**: 42.3% of ink pixels at grid-crossing intersections are incorrectly removed. Grid paper (H+V rules) performs better than ruled notebook paper at crossing preservation (58.91% vs. 14.95%), likely because the grid pattern is more distinct from typical stroke orientations.

**Conclusion**: The ruled-line removal module reliably removes isolated line pixels (>96% precision, ~100% recall) but has a significant crossing-preservation limitation. Any downstream pipeline consuming images that were processed through this module should be aware that handwriting ink at crossing points may have been partially or substantially removed. This limitation is inherent to pixel-level morphological approaches that cannot distinguish ink-over-line from line pixels without additional stroke trajectory information.



## Phase 2 — Branch B: Kinematics & Multi-Level Ground-Truth Validation

### 1. The 1,861 Fragment Defect & Its Topological Resolution
In the initial naive implementation, whenever the traversal encountered an intersection clique (such as the 3-pixel triangle clusters naturally formed by thinning algorithms), it prematurely terminated strokes. This chopped handwriting into **1,861 micro-fragments** (median length: 3 pixels), scrambling the temporal reconstruction.

We resolved this with three core topological improvements:
1. **Junction Clique Contraction**: Identifies all connected clusters of junction pixels ($\text{degree} \ge 3$) and contracts them into a single intersection centroid, eliminating internal micro-cycles and dead ends.
2. **Tangent "Fly-Through" Continuation**: When traversal enters a junction with $\ge 2$ unvisited outgoing branches, it computes the unit tangent vectors of incoming and outgoing paths. It continues straight through the intersection along the branch that maximizes directional cosine similarity ($< 35^\circ$ deflection), exactly mimicking a pen moving through a crossing.
3. **Spatial-Indexed Stroke Stitching**: Stitches contiguous fragments whose endpoints meet within `max_gap = max(0.15 × H_med, 2.0)` pixels with smooth orientation, using a spatial hash grid index to achieve $O(N)$ execution speed.
4. **$O(1)$ Active Degree Map**: Eliminates repeated subgraph copying during stroke discovery, speeding up traversal by $24\times$.

**Result**: On sample `u00006s00001_hw00001`, recovered stroke count dropped from **1,861** down to **331 continuous strokes** (median length: $18\text{ px}$, mean: $21.2\text{ px}$, max: $80\text{ px}$).

### 2. Multi-Level Ground-Truth Validation Findings

We benchmarked reconstructed kinematics against the true recorded digitizer telemetry across two distinct levels of hierarchy:
- **Level 1 (Single-Stroke Level)**: Evaluates whether the Two-Thirds Power Law ($v \propto \kappa^{-1/3}$) correctly predicts the physical speed of the human hand along individual continuous strokes.
- **Level 3 (Whole-Document Level)**: Evaluates chronological signal correlation when all strokes across an entire multi-sentence document are concatenated.

#### Honest Headline (70-Sample Broad Cohort — Text Tasks Only)

> **Mean stroke-level Pearson $r = +0.227$ (median $+0.232$)** across 7,275 individual strokes from 70 samples (35 `dataSciRep_public` + 35 `DiaGraMo` TSK3/TSK4/TSK15/TSK16). Graphomotor/drawing tasks (spirals, loops, waves, RCFT) are explicitly excluded from this validation cohort as they are not handwriting.
>
> Only **37.9% of strokes** achieve $r > 0.30$; **17.3% achieve $r > 0.50$**. Reconstructed velocity has **weak-to-moderate fidelity** to ground truth on average. Anyone consuming these kinematic features downstream should treat absolute velocity/jerk magnitudes as a **noisy proxy signal**, not a precise reconstruction.

| Cohort | N (samples) | Mean stroke $r$ | Median stroke $r$ | Std | Min | Max | Strokes >0.30 | Strokes >0.50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dataSciRep_public` (handwriting HW) | 35 | +0.195 | +0.196 | 0.079 | +0.022 | +0.331 | — | — |
| `DiaGraMo` TSK3/4/15/16 (text tasks) | 35 | +0.252 | +0.256 | 0.052 | +0.152 | +0.373 | — | — |
| **OVERALL COMBINED** | **70** | **+0.223** | **+0.232** | **0.073** | **+0.022** | **+0.373** | **37.9%** | **17.3%** |

**Total individual strokes evaluated**: 7,275 across 70 samples.

#### Reference Single-Sample Benchmark (6-sample Pilot Cohort)

Prior to the broad cohort, the following 6 anchor samples were used for algorithm development and are retained for reproducibility (not representative of the full distribution):

| Dataset / Task | Sample ID | True GT Strokes | Recovered Strokes | Level 1 Stroke Mean $r$ | Level 1 Frac $r > 0.30$ | Whole Doc Vel $r$ | 4–8 Hz Tremor Power |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DiaGraMo` TSK15 (Word Copy)** | `BR10301_TSK15` | 146 | 297 | +0.373 | 65.0% | -0.060 | 22.0% |
| **`DiaGraMo` TSK3 (Sentence Dict)** | `BR10301_TSK3` | 239 | 378 | +0.328 | 56.9% | -0.004 | 31.0% |
| **`dataSciRep_public` (Dysgraphic HW)** | `u00006s00001` | 73 | 331 | +0.312 | 50.0% | +0.078 | 31.7% |
| **`DiaGraMo` TSK4 (Alphabet Dict)** | `BR10403_TSK4` | 218 | 507 | +0.289 | 46.0% | -0.012 | 34.0% |
| **`DiaGraMo` TSK16 (Sentence Copy)** | `BR10402_TSK16` | 158 | 368 | +0.231 | 41.7% | +0.080 | 26.7% |
| **`dataSciRep_public` (Control HW)** | `u00050s00001` | 71 | 226 | +0.092 | 19.4% | +0.021 | 16.9% |

#### Validation Findings & Discussion:
1. **The Physics Model Provides a Useful Signal**: On individual intact strokes, the Two-Thirds Power Law achieves statistically positive correlation with real tablet velocity. Individual strokes reach up to $r = +0.9757$; at the broad cohort level, mean $r = +0.223$ with 37.9% of strokes exceeding $r > 0.30$. Static handwriting centerlines *do* encode motor velocity, but the relationship is noisy and should not be treated as precise reconstruction.
2. **Why Whole-Document Concatenation Correlation is Near Zero ($r \approx 0.04$)**: In natural handwriting, writers frequently perform non-linear motor jumps — returning to dot an 'i', cross a 't', or add punctuation long after writing a word. In a static image, spatial proximity causes these strokes to be ordered differently than chronological time. An arbitrary phase shift in time-series concatenation naturally drives whole-document Pearson correlation to zero.
3. **Pressure Proxy is Ineffective**: Whole-document stylus pressure correlation mean $r = +0.002$ (median $+0.006$) — near-zero. The optical pressure proxy (stroke width) cannot reconstruct true stylus force from static images. No constant-pressure (zero-variance) signals are silently included in aggregates; they are detected and excluded with logged warnings.
4. **NVI Metric Fix**: Resolved the stroke-length inversion artifact by normalizing Number of Velocity Inversions per physical stroke (`nvi_per_stroke`) and per 100 character units via fixed arc-length resampling, making NVI strictly scale-invariant.
5. **Collinear Proxy Removed**: Documented the $r = 0.9914$ collinearity between optical pressure proxy and stroke width; removed from primary feature vector to prevent redundancy in downstream pipelines.

---

## Phase 3 — Consolidate: Unified 20D Multimodal Pipeline

### 1. Architecture & 20-Dimensional Feature Space
Phase 3 integrates Branch A (9 BHK spatial indicators) and Branch B (11 kinematic fluency indicators) into a unified, high-performance class: `DysgraphiaFeaturePipeline`.

$$\mathbf{f}_{\text{multimodal}} = \left[ \mathbf{f}_{\text{BHK}} \in \mathbb{R}^9 \;\|\; \mathbf{f}_{\text{kinematic}} \in \mathbb{R}^{11} \right] \in \mathbb{R}^{20}$$

### 2. Verified 20D Multimodal Feature Vector Schema

Below is the verified schema extracted end-to-end by `DysgraphiaFeaturePipeline` (Schema v2.0.0):

| Index | Feature Identifier | Domain | Mathematical Formulation | Sample Value (LPD_1) | Confidence | Technical Interpretation |
| :-: | :--- | :---: | :--- | :---: | :---: | :--- |
| **1** | `bhk_size_covariance` | Branch A (Spatial) | dimensionless (CV) | `0.3960` | 0.60 | Fluctuation in character box height and area relative to H_med. |
| **2** | `bhk_height_ratio_consistency` | Branch A (Spatial) | dimensionless (IQR/median) | `0.6875` | 0.60 | Dispersion of ascenders/descenders vs body x-height. |
| **3** | `bhk_baseline_drift` | Branch A (Spatial) | 1/H_med | `0.6301` | 0.60 | Macro slant and micro vertical oscillation along text baseline. |
| **4** | `bhk_spacing_entropy` | Branch A (Spatial) | dimensionless (nats) | `0.6284` | 0.60 | Shannon entropy of inter-character spatial gaps. |
| **5** | `bhk_stroke_width_variance` | Branch A (Spatial) | dimensionless (CV) | `0.1173` | 0.42 | Stroke width CV from Euclidean Distance Transform (EDT). |
| **6** | `bhk_telescoping_overlap` | Branch A (Spatial) | ratio [0,1] | `0.8971` | 0.60 | Bounding box horizontal collision frequency and intrusion depth. |
| **7** | `bhk_acute_turns` | Branch A (Spatial) | turns/H_med | `1.810` | 0.60 | High-curvature angular turn rate (|Delta theta| >= 110 deg). |
| **8** | `bhk_left_margin_drift` | Branch A (Spatial) | 1/H_med | `0.0000` | 0.60 | Horizontal drift of line start coordinates relative to H_med. |
| **9** | `bhk_line_collisions` | Branch A (Spatial) | ratio [0,1] | `0.0000` | 0.60 | Inter-line vertical intrusions and ascender-descender collisions. |
| **10** | `kin_mean_velocity` | Branch B (Kinematic Proxy) | H_med/s (ESTIMATED) | `0.6988` | 0.30 | Two-Thirds Power Law proxy speed (H_med/s). Mathematical proxy only. |
| **11** | `kin_peak_velocity` | Branch B (Kinematic Proxy) | H_med/s (ESTIMATED) | `3.981` | 0.30 | Maximum ballistic impulse proxy (H_med/s). |
| **12** | `kin_velocity_skewness` | Branch B (Kinematic Proxy) | dimensionless (ESTIMATED) | `1.775` | 0.30 | Asymmetry between synthetic acceleration and deceleration phases. |
| **13** | `kin_nvi_rate` | Branch B (Kinematic Proxy) | inversions/s (ESTIMATED) | `2.800` | 0.30 | Number of velocity inversions per unit synthetic time (Hz). |
| **14** | `kin_nvi_per_stroke` | Branch B (Kinematic Proxy) | inversions/stroke (ESTIMATED) | `13.381` | 0.30 | Total velocity inversions normalized per physical recovered stroke. |
| **15** | `kin_nvi_per_h_med` | Branch B (Kinematic Proxy) | 1/H_med (ESTIMATED) | `4.000` | 0.30 | Spatial density of velocity inversions per H_med length. |
| **16** | `kin_jerk_metric` | Branch B (Kinematic Proxy) | H_med^2/s^5 (ESTIMATED) | `13065.9` | 0.30 | Mean squared 3rd derivative (H_med^2/s^5). High due to numerical diff. |
| **17** | `kin_dimensionless_jerk` | Branch B (Kinematic Proxy) | dimensionless (ESTIMATED) | `754.7` | 0.30 | Flash & Hogan scale-invariant jerk ((T^5/L^2) * int(j^2 dt) * 1e-4). |
| **18** | `kin_spatial_roughness_4_8hz` | Branch B (Kinematic Proxy) | ratio [0,1] (ESTIMATED) | `0.0388` | 0.50 | Normalized power in 4-8 Hz band along synthetic trajectory. |
| **19** | `kin_pen_lift_count` | Branch B (Kinematic Proxy) | integer count (ESTIMATED) | `21.000` | 0.30 | Count of discrete physical stroke segments. |
| **20** | `kin_mean_stroke_length` | Branch B (Kinematic Proxy) | H_med units (ESTIMATED) | `3.346` | 0.50 | Average continuous arc length per stroke (in units of H_med). |
| **21** | `kin_ink_width_ratio_mean` | Branch B (Kinematic Proxy) | W/H_med (ESTIMATED) | `0.1271` | 0.35 | Mean optical ink thickness relative to median character height. |
| **22** | `kin_ink_width_ratio_std` | Branch B (Kinematic Proxy) | W/H_med (ESTIMATED) | `0.0075` | 0.35 | Standard deviation of optical ink thickness relative to H_med. |


### 3. Broad-Cohort Multi-Task Validation Benchmarks
Benchmarked across 80 multi-cohort handwriting samples with honest ground-truth validation bounds:

| Metric / Analysis | Sample Cohort | Empirical Finding | Scientific Interpretation |
| :--- | :---: | :---: | :--- |
| **Stroke Velocity Correlation** | 70-sample broad cohort (7,275 strokes) | Mean $r = +0.223$, Median $r = +0.232$ | Weak-to-moderate stroke-level correlation with tablet sensors. |
| **High-Correlation Stroke Ratio** | 70-sample broad cohort | $37.9\%$ strokes with $r > 0.30$ | Only a minority of strokes closely match physical sensor dynamics. |
| **Paragraph Sequence Fidelity** | Full text documents | Global $r \approx +0.016$ | Global chronological ordering cannot be recovered from static ink. |
| **Pressure Correlation** | Digitizer samples | $r = 0.002$ (no correlation) | Optical stroke width does NOT measure pen down-force; renamed to `kin_ink_width_ratio`. |
| **Tremor Frequency Fidelity** | Synthesized kinematics | Uncalibrated on static ink | Renamed from `tremor_index_4_8hz` to `spatial_roughness_4_8hz`. |
| **Line Collisions (BHK #13)** | Malay photographed cohort | $+21.5\%$ in dysgraphia | Structural spatial indicator; ascenders crashing into adjacent lines. |
| **Letter Size Covariance (BHK #4)** | Malay photographed cohort | $+11.8\%$ in dysgraphia | Motor planning inconsistency in glyph dimensions. |

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

# 1. Initialize pipeline
pipeline = DysgraphiaFeaturePipeline(compute_kinematics=True)

# 2. Extract features from any image (PNG, JPG, or PIL Image)
results = pipeline.extract("path/to/handwriting_sample.jpg")

# 3. Access feature vectors
bhk_vector = results["bhk_vector"]            # Shape (9,) - 9 BHK clinical features
kinematic_vector = results["kinematic_vector"] # Shape (8,) - 8 Neuromotor kinematic features
full_vector = results["combined_vector"]       # Shape (17,) - Full multimodal vector

# 4. Inspect named metrics
print("Size Covariance:", results["bhk_metrics"]["size_covariance_score"])
print("Acute Turns:", results["bhk_metrics"]["acute_turns_score"])
print("Line Collisions:", results["bhk_metrics"]["line_collision_score"])
print("NVI per Stroke:", results["kinematic_metrics"]["nvi_per_stroke"])
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

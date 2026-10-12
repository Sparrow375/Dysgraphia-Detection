# Comprehensive Architecture Blueprint - Dysgraphia Detection System

## 1. System Objective, Scope & Clinical Rationale

### 1.1 Objective & Vision
The **Multilingual Dysgraphia Screening System** provides an automated, objective, and clinically validated screening engine for developmental dysgraphia in school-aged children (Grades 3–8). 

Unlike commercial tablet-based systems that rely on expensive digitizing tablets (e.g., Wacom Cintiq) and high-frequency stylus sampling (200 Hz pressure, altitude, azimuth, and velocity), this system operates **exclusively on static 2D pen/pencil-on-paper artifacts** captured via standard smartphone cameras or flatbed office scanners. This eliminates hardware barriers, enabling ubiquitous, cost-effective screening across low-resource community schools and public health initiatives.

### 1.2 Target Cohort & Protocol Prompts
The screening protocol targets bilingual children writing in both **Devanagari (Hindi)** and **Latin (English)** scripts across structured motor and cognitive tasks:
1. **Sentence 1 (`sentence_01_copy_hindi`)**: Copying Hindi — Devanagari script, baseline copying task.
2. **Sentence 2 (`sentence_02_copy_english`)**: Copying English — Latin script, baseline copying task (often multi-line).
3. **Sentence 3 (`sentence_03_dictated_hindi`)**: Dictation Hindi — Devanagari script, testing auditory-motor integration under time pacing.
4. **Sentence 4 (`sentence_04_dictated_english`)**: Dictation English — Latin script, testing auditory-motor integration under time pacing.
5. **Sentence 5 (`sentence_05_own_hindi`)**: Free Writing / Own Hindi — Expressive cognitive load, Grades 4–8.
6. **Sentence 6 (`sentence_06_own_english`)**: Free Writing / Own English — Expressive cognitive load, Grades 4–8.

### 1.3 Clinical Decision Framework: The Two-Tier Screening Cascade
To address the clinical trade-off between missing children in need (False Negatives) and overwhelming clinical specialists with false alarms (False Positives), the system is architected around a **Two-Tier Screening Cascade**:
- **Tier 1 (Community Triage)**: Prioritizes Sensitivity ($\ge 90\%$) with Specificity $\ge 80\%$. Used in primary schools to identify any child exhibiting motor or cognitive handwriting vulnerabilities for preliminary evaluation.
- **Tier 2 (Specialist Referral)**: Prioritizes Specificity ($\ge 90\%$) with Sensitivity $\ge 70\%$. Flags severe, multi-domain motor breakdown for formal occupational therapy or neurological assessment.

---

## 2. Complete Repository Directory Hierarchy

```
Dysgraphia-Detection/
├── .git/                             # Version control metadata
├── .gitattributes                    # Git LFS & line-ending tracking rules
├── .gitignore                        # Git exclusion rules
├── .dvc/                             # Data Version Control configuration
├── .dvcignore                        # DVC exclusion rules
├── README.md                         # Quick-start documentation and overview
├── architechture.md                  # Comprehensive architectural blueprint and system reference
├── context.md                        # Persistent project memory, sprint logs, and benchmark metrics
├── task_list.md                      # Active sprint tasks and historical phase trackers
├── requirements.txt                  # Python runtime dependencies
├── configs/
│   └── config.yaml                   # Global configuration (seeds, paths, thresholds, protocol definitions)
├── data/
│   ├── manifest.csv                  # Master cohort manifest (215 students, paths, labels, splits)
│   ├── folds/
│   │   ├── nested_folds.json         # Nested 5-repeat 5-fold outer/inner split indices
│   │   └── student_folds_lookup.csv  # Flat lookup table of outer fold assignments per student
│   ├── school_a/
│   │   ├── master_mapping.csv        # Cross-reference table for School A student metadata
│   │   └── raw/                      # Raw paper scans for School A (Grades 3, 4, 5, 6, 7)
│   ├── school_b/
│   │   └── raw/                      # Raw paper scans for School B (Grades 3, 4, 8)
│   ├── processed/                    # Segmented crops, masks, and metadata schemas
│   │   ├── school_a/{student_id}/    # Sentence crops, clean ink masks, review_status.json
│   │   └── school_b/{student_id}/    # Sentence crops, clean ink masks, review_status.json
│   ├── features/
│   │   ├── features_raw.csv          # Unnormalized 37-feature table across 1,145 sentence tasks
│   │   └── features_meta.json        # Mathematical definitions, units, and descriptions
│   └── datasets/
│       ├── dataset_hindi.csv         # Normalized Devanagari tasks (570 tasks, 72 cols)
│       ├── dataset_english.csv       # Normalized Latin tasks (575 tasks, 66 cols)
│       ├── dataset_combined.csv      # Unified sentence dataset (1,145 tasks, 74 cols)
│       ├── dataset_student_level.csv # Multi-phenotype student matrix (215 students, 312 cols)
│       └── datasets_meta.json        # Dataset transformation metadata and cell distributions
├── docs/
│   ├── Workstream A v1 Implementation Plan (1).md
│   ├── data-versioning-dvc.md        # DVC remote storage setup and management
│   ├── git-lfs-guide.md              # Git Large File Storage policies
│   └── project-scope.md              # Clinical scope, protocol background, and literature review
├── pipeline/                         # Image processing, segmentation, and dataset compilation
│   ├── __init__.py
│   ├── manifest.py                   # Master cohort ingestion, verification, and manifest builder
│   ├── folds.py                      # Leakage-free Stratified Group K-Fold partition generator
│   ├── preprocess.py                 # Auto-rotation, perspective warp, deskewing, Otsu binarization
│   ├── rules.py                      # Ruled line detection, trajectory tracking, stroke-preserving reduction
│   ├── segment.py                    # Connected component clustering, EasyOCR script ID, task matching
│   ├── skeleton.py                   # Medial axis thinning, branch junction/endpoint detection, pruning
│   ├── dataset_assembly.py           # Robust within-cohort z-scoring and fold imputation
│   └── process_dataset.py            # Cohort-wide batch orchestration for preprocessing & cropping
├── features/                         # Mathematical handwriting feature extraction library
│   ├── __init__.py                   # Feature module exports
│   ├── baseline.py                   # Linear baseline slope, RMSE, and trajectory drift
│   ├── rule_offset.py                # Ruled-line adherence offsets and margin deviations
│   ├── slant.py                      # Tangent angle circular mean, variance, and slant dispersion
│   ├── curvature.py                  # Savitzky-Golay smoothed curvature κ, tortuosity, jerk proxy
│   ├── gaps.py                       # Inter-word spacing mean, CV, crowding (<0.3h), hesitation (>2h)
│   ├── size.py                       # Word height CV, macro/micrographia (h/r), ascender & matra ratios
│   ├── hindi.py                      # Devanagari Shirorekha headline RMS deviation, breaks, tilt variance
│   ├── fragmentation.py              # Connected components, junctions, endpoints per unit width
│   ├── utils.py                      # Shared utilities: x-height estimation, clean ink mask loader
│   ├── extractor.py                  # Multi-core batch cohort feature extractor
│   └── validation.py                 # Real-data feature correlations, univariate AUROCs, grade trends
├── models/                           # Machine learning, nested CV, ensembling & evaluation suite
│   ├── __init__.py
│   ├── cv.py                         # 5-repeat 5-fold Stratified Group K-Fold nested CV engine
│   ├── candidates.py                 # Estimator definitions (Elastic-Net, LR, RF, HistGBM, Stacking)
│   ├── feature_selection.py          # Stability selection via randomized subsampled Lasso
│   ├── evaluator.py                  # Comprehensive metrics: AUROC, AUPRC, Brier, Sens@Spec, Bootstrap
│   ├── student_features.py           # Multi-phenotype student-level feature aggregation (312 cols)
│   ├── student_experiments.py        # 5-repeat 5-fold nested benchmark runner on student level
│   ├── experiments.py                # Sentence-level benchmark suite per dataset alias
│   ├── cross_school_experiments.py   # External validation (A→B, B→A), pooled models, domain shift
│   ├── ablations.py                  # Drop-one-family ablation and feature importance analysis
│   ├── generate_gallery.py           # Diagnostic gallery builder for misclassified students
│   └── run_full_benchmark.py         # Unified cross-dataset benchmark execution runner
├── qa/                               # Quality assurance, interactive inspection & web review application
│   ├── app.py                        # Asynchronous Python (aiohttp) backend for review & annotation
│   ├── inspection.py                 # Diagnostic transparency inspector for layers and features
│   ├── generate_overlay_review.py    # Static review gallery generator (HTML side-by-side cards)
│   ├── review_gallery.html           # Pre-rendered visual review cards for rapid quality check
│   └── web/                          # Single-page web application frontend
│       ├── index.html                # Multi-layer canvas review interface
│       ├── style.css                 # Modern responsive styling and dark mode design system
│       └── app.js                    # Canvas rendering, layer toggles, and annotation controls
├── reports/                          # Structured benchmark outputs, metrics JSONs, and CSV tables
│   ├── student_benchmark_results.csv # 5-repeat 5-fold nested CV results on student level
│   ├── student_benchmark_results.json# Detailed per-fold student evaluation metrics
│   ├── benchmark_all_datasets.csv    # Cross-dataset comparative summary table
│   ├── benchmark_hindi.csv           # Sentence-level Hindi benchmark results
│   ├── benchmark_english.csv         # Sentence-level English benchmark results
│   ├── benchmark_combined.csv        # Sentence-level Combined benchmark results
│   └── misclassified_gallery.html    # Qualitative visual analysis of false positives & false negatives
├── scratch/                          # Diagnostic scripts, algorithm prototypes, and visual validation crops
└── tests/                            # Automated unit, regression, and mathematical invariant tests
    ├── __init__.py
    ├── test_phase0.py                # Manifest schema, file existence, and fold leakage tests
    ├── test_phase1.py                # Preprocessing, deskewing, line detection, segmentation, skeletons
    ├── test_phase3.py                # Dataset assembly, robust z-scoring, and fold imputation tests
    ├── test_phase4.py                # Nested CV, candidate model fitting, student feature aggregation
    ├── test_phase4_stage2.py         # Stability selection and student-level model pipeline tests
    ├── test_qa_app.py                # REST API endpoint and static routing tests
    ├── test_features_synthetic.py    # Controlled perturbation sensitivity tests (Spearman ρ ≥ 0.8)
    └── test_features_invariance.py   # Scale (0.7x) and rotation (±3°) invariance tests
```

---

## 3. End-to-End Data Flow Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Raw Image Capture (Photo / Scan)                  │
│                     Dimensions: ~2000×2500 px, Grayscale               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               1. Preprocessing (`pipeline/preprocess.py`)              │
│  • Auto-orientation detection & rotation to portrait                   │
│  • Quadrilateral page corner detection & perspective rectifying warp   │
│  • Normalized deskewed grayscale (`page_normalized.png`)               │
│  • Page-level Otsu binarization (`binary_ink`)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│         2. Ruled Line Reduction & Stroke Preservation                  │
│                        (`pipeline/rules.py`)                           │
│  • Multi-line trajectory detection via horizontal opening ($25\times 1$)│
│  • Polynomial trajectory tracking $y_{\text{line}}(x)$ across page     │
│  • Continuous vertical run-length filtering $R_v(y, x)$                │
│  • 2D neighborhood max filtering: $(9 \times 3)$ rectangular dilation  │
│  • Erasure along trajectories where $\max R_v \le 6$ px                │
│  • Absolute preservation where $\max R_v \ge 7$ px                     │
│  • Vertical margin removal ($\ge 120$ px) & speckle filter ($\le 4$ px)│
│  • Clean Ink Mask (`ink_clean`) & Ruled Line Mask (`rule_mask`)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             3. Line Segmentation & Sentence Task Assembly              │
│                       (`pipeline/segment.py`)                          │
│  • Nearest-neighbor connected component vertical clustering            │
│  • Deep OCR script classification (EasyOCR `hi+en` + Shirorekha logic) │
│  • Protocol state machine: Groups lines into 6 standard tasks          │
│  • Word bounding boxes & robust linear baseline fit ($y = mx + c$)     │
│  • Output: `{task_id}.png`, `{task_id}_clean_ink.png`, `{task_id}.json`│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               4. Mathematical Feature Extraction Library               │
│                             (`features/`)                              │
│  • 8 Feature Families (Baseline, Rule, Slant, Curvature, Gaps,         │
│    Size, Hindi, Fragmentation)                                         │
│  • Dimensionless scale normalization by x-height $h$ and rule space $r$│
│  • Batch generation: `data/features/features_raw.csv` (1,145 × 37 cols)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│         5. Dataset Assembly & Robust Within-Cohort Z-Scoring           │
│                   (`pipeline/dataset_assembly.py`)                     │
│  • Stratified robust z-score: $z = (x - \text{med}) / (1.4826 \cdot \text{MAD})$│
│  • Computed within homogeneous $(\text{Grade} \times \text{Script} \times \text{Task})$ cells │
│  • Task one-hot indicators (`task_copy`, `task_dictated`, `task_own`)  │
│  • Tables: `dataset_hindi.csv`, `dataset_english.csv`, `combined.csv`  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│         6. Student-Level Multi-Phenotype Feature Engineering           │
│                     (`models/student_features.py`)                     │
│  • Central tendencies (mean, median across all tasks)                  │
│  • Cognitive load gradients ($\Delta_{\text{dictation} - \text{copy}}$)│
│  • Cross-script motor divergence ($\Delta_{\text{english} - \text{hindi}}$)│
│  • Peak motor breakdown episodes ($\max$ fragmentation, $\max$ wobble) │
│  • Temporal fatigue degradation ($\Delta_{\text{late} - \text{early}}$)│
│  • Output: `data/datasets/dataset_student_level.csv` (215 × 312 cols)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             7. Machine Learning & Clinical Decision Engine             │
│                              (`models/`)                               │
│  • Zero-leakage 5-repeat 5-fold Stratified Group K-Fold nested CV      │
│  • Stability feature selection: Curated Clean 15 multi-phenotype set   │
│  • Multi-model candidates: Regularized LR, Elastic-Net, RF, HistGBM    │
│  • Two-Tier Screening Cascade:                                         │
│      Tier 1: Community Triage (Sensitivity $\ge 90\%$)                 │
│      Tier 2: Specialist Referral (Specificity $\ge 90\%$)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│           8. Quality Assurance & Clinical Review Web Platform          │
│                                (`qa/`)                                 │
│  • Multi-layer live canvas viewer: Grayscale | Rules | Clean Ink |     │
│    Baselines | Skeleton Graph                                          │
│  • Instant boundary micro-adjustments and ground truth labeling        │
│  • Automated transparency report gallery (`review_gallery.html`)       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deep Component Specifications & Technical Details

### 4.1 Ingestion & Cohort Stratification (`pipeline/manifest.py`, `pipeline/folds.py`)
- **Manifest Builder (`pipeline/manifest.py`)**:
  - Ingests raw scans across School A (Future Gen, 115 students across Grades 3, 4, 5, 6, 7) and School B (Kanyashala, 100 students across Grades 3, 4, 8).
  - Merges ground truth labels: School A ground truth verified via specialist assessment (`future_gen_handwritten_labels.csv`); School B verified via cross-school screening protocols.
  - Generates `data/manifest.csv` with strict verification that all referenced file paths exist on the local file system.
- **Nested Cross-Validation Partitioning (`pipeline/folds.py`, `models/cv.py`)**:
  - **Zero Student Leakage Invariant**: Because a single student produces multiple sentence crops across different tasks and languages, naive sample-level splitting causes catastrophic data leakage. The system enforces strict grouped partitioning using `StratifiedGroupKFold(n_splits=5, groups=student_id)`.
  - **Outer Loop**: 5 repeats of 5-fold cross-validation. In each fold, 80% of students are used for training and 20% are completely held out for evaluation.
  - **Inner Loop**: 3-fold Stratified Group K-Fold on the training split to select hyperparameters (regularization strength $C$, tree depth, $L_1$ ratios) and operational clinical thresholds.

### 4.2 Image Normalization & Preprocessing (`pipeline/preprocess.py`)
- **Auto-Orientation**: Analyzes aspect ratio and horizontal versus vertical stroke gradients to ensure sheets are positioned in standard portrait orientation ($H > W$).
- **Perspective Rectification**: Identifies outer sheet corners using Canny edge detection, morphological dilation, and quadrilateral contour approximation. Warps the page quadrilateral to a standardized 2000 px width coordinate space via homography transformation. If sheet corners are obscured by table backgrounds, applies a safe 2% margin trim.
- **Grayscale Illumination Normalization**: Divides the raw image by a heavily blurred background model ($\sigma \approx 50$) to neutralize uneven smartphone lighting, ambient shadows, and paper discoloration while preserving fine pencil strokes.
- **Binarization**: Computes page-level Otsu thresholding to produce an initial binary ink mask (`binary_ink`), avoiding destructive contrast clipping.

### 4.3 Definitive Ruled-Line Reduction & Stroke Preservation (`pipeline/rules.py`)
Notebook guidelines present a fundamental dilemma in static handwriting analysis: leaving ruled lines intact causes artificial component merging, distorted skeletons, and false rule adherence features; conversely, naive thresholding or subtraction slices horizontal gaps through letters, amputating descenders (`g`, `y`, `p`, `q`), severing closed loops (`o`, `e`, `a`, `d`), and destroying Devanagari headlines (Shirorekhas).

The definitive engine resolves this through an invariant-driven architecture:
1. **Multi-Line Trajectory Tracking**:
   - Detects all horizontal ruling lines across the page using a $25 \times 1$ horizontal morphological opening and projection peak analysis with peak distance $d=6$, capturing both primary notebook lines and fine secondary guidelines in 4-line notebooks.
   - For each guideline, fits a polynomial trajectory $y_{\text{line}}(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3$ across confirmed non-text columns, modeling paper bowing, deskew angle, and camera lens barrel distortion with sub-pixel precision.
2. **2D Continuous Vertical Run-Length Neighborhood Filtering**:
   - Computes continuous vertical ink run lengths $R_v(y, x)$ for every pixel:
     $$R_v(y, x) = \text{number of contiguous black ink pixels in column } x \text{ containing } (y, x)$$
   - Evaluates the local maximum vertical run length within a $(9 \times 3)$ neighborhood using vectorized morphological dilation:
     $$\max R_v(y, x) = \max_{|\Delta x| \le 4, |\Delta y| \le 1} R_v(y + \Delta y, x + \Delta x)$$
3. **Physical vs Handwriting Separation Invariant**:
   - **Printed Guideline Invariant**: Notebook ruling lines have physical printed thickness $\le 5-6\text{ px}$. In margins and inter-word gaps, $\max R_v \le 6\text{ px}$.
   - **Handwriting Invariant**: Genuine handwriting stems, curved letter loops, and Devanagari headlines connect into two-dimensional character bodies within $\pm 4\text{ px}$ horizontally, guaranteeing $\max R_v \ge 7\text{ px}$.
4. **Vectorized Trajectory-Guided Erasure**:
   - For each pixel $(y, x)$ within $\pm 4\text{ px}$ of a detected line trajectory:
     $$\text{Ink}(y, x) = \begin{cases} 0 & \text{if } \max R_v(y, x) \le 6 \\ \text{Ink}(y, x) & \text{if } \max R_v(y, x) \ge 7 \end{cases}$$
   - This guarantees that 100% of printed lines in margins and inter-word gaps are eradicated, while **zero handwriting stems are severed, zero letter loops are hollowed, and zero Shirorekhas are broken**.
5. **Margin Line & Pepper Noise Filtering**:
   - Removes vertical notebook margin lines ($\ge 120\text{ px}$ tall via vertical morphological opening).
   - Removes isolated pepper noise speckles ($\le 4\text{ px}$).

### 4.4 Text Segmentation & Script Identification (`pipeline/segment.py`)
- **Connected Component Line Clustering**:
  - Filters out page headers and student roll-number boxes ($y < 320\text{ px}, x > 850\text{ px}$).
  - Groups connected text components into physical lines using nearest-neighbor vertical centroid clustering bounded by $0.65 \cdot r$ (where $r$ is the median ruling spacing, typically $70-90\text{ px}$).
- **Script Classification**:
  - Processes each line with deep-learning OCR (EasyOCR `hi+en`).
  - Lines with confident Devanagari Unicode characters are classified as `devanagari`; Latin characters as `latin`.
  - Fallback logic checks for topological Shirorekha headline continuity if OCR confidence is low.
- **Protocol State Machine**:
  - Matches physical text lines to the 6 canonical protocol tasks based on script transitions and grade expectations (Grade 3 produces 4 tasks; Grades 4–8 produce 6 tasks).
  - Multi-line tasks (such as English copy) are automatically merged into a single coherent sentence crop.
- **Per-Word Baseline Extraction**:
  - Identifies word bounding boxes and computes the bottom anchor point $(x_{\text{center}}, y_{\text{bottom}})$ of each word.
  - Fits a robust linear baseline ($y = mx + c$) using RANSAC regression, calculating per-word vertical residuals and baseline slope.
- **Persistence**:
  - Outputs `{task_id}.png` (grayscale natural crop), `{task_id}_clean_ink.png` (stroke-preserved binary mask), and `{task_id}.json` (geometric bounding boxes, word anchors, baseline parameters, and script tags).

### 4.5 Topological Skeletonization & Graph Analysis (`pipeline/skeleton.py`)
- **Medial Axis Thinning**: Applies iterative topological thinning (`skimage.morphology.skeletonize`) on the clean ink mask to extract 1-pixel-wide medial axes.
- **Neighborhood Convolution ($3 \times 3$)**:
  - Evaluates the 8-connected neighborhood sum $N(y, x) = \sum_{i,j \in \{-1,0,1\}} S(y+i, x+j) - 1$:
    - **Endpoints**: $N(y, x) = 1$ (stroke terminations, pen lifts).
    - **Continuation Points**: $N(y, x) = 2$ (regular stroke trajectory).
    - **Junctions**: $N(y, x) \ge 3$ (stroke intersections, crossings, loops).
- **Spur Pruning**: Prunes terminal skeleton spurs shorter than $0.15 \cdot h$ (where $h$ is the local x-height) to eliminate minor stroke boundary noise without pruning legitimate serifs.
- **NetworkX Topological Graph**: Compiles the skeleton into a spatial graph structure, computing connected components, path tortuosity, and branch length distributions.

---

## 5. Mathematical Feature Extraction Library (`features/`)

Every feature module strictly follows a functional contract: accepts `(sentence_json, image_crop)` and returns a flat dictionary of `feature_name -> float | np.nan`. All spatial metrics are rendered **dimensionless** by normalizing against x-height $h$ or ruling spacing $r$.

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE EXTRACTION TAXONOMY (37 FEATURES)                   │
├──────────────────────────┬────────────────────────────────────────────────────────────┤
│ Family                   │ Features & Mathematical Formulations                       │
├──────────────────────────┼────────────────────────────────────────────────────────────┤
│ 1. Baseline Adherence    │ • baseline_rmse_norm: $\text{RMSE} / h$ from RANSAC line   │
│    (`features/baseline.py`)│ • baseline_slope_mean: Mean line slope $\tan(\theta)$      │
│                          │ • baseline_slope_std: Standard deviation across lines      │
├──────────────────────────┼────────────────────────────────────────────────────────────┤
│ 2. Rule Adherence        │ • rule_offset_mean: $\frac{1}{N}\sum (y_{\text{word}} -    │
│    (`features/rule_offset.py`)│   y_{\text{rule}}) / r$                                   │
│                          │ • rule_offset_std: Standard deviation of rule offsets      │
├──────────────────────────┼────────────────────────────────────────────────────────────┤
│ 3. Slant & Orientation   │ • slant_mean_deg: Circular mean of stroke tangents $\phi$  │
│    (`features/slant.py`) │ • slant_circular_std_deg: Circular standard deviation      │
│                          │ • slant_abrupt_changes_per_h: Slant sign flips per $h$     │
├──────────────────────────┼────────────────────────────────────────────────────────────┤
│ 4. Curvature & Tremor    │ • jerk_proxy: $\frac{1}{L}\int |\frac{d\kappa}{ds}| h^2 ds$│
│    (`features/curvature.py`)│ • tangent_variance_short_wavelength: Power at $\lambda < 0.5h$│
│                          │ • kappa_sign_changes_per_h: Inflection count per $h$       │
│                          │ • tortuosity_median: Median path length / chord length     │
├──────────────────────────┼────────────────────────────────────────────────────────────┤
│ 5. Inter-Word Spacing    │ • gap_mean: Mean inter-word gap / $h$                      │
│    (`features/gaps.py`)  │ • gap_cv: $\sigma_{\text{gap}} / \mu_{\text{gap}}$         │
│                          │ • gap_fraction_below_0_3h: Fraction $< 0.3h$ (crowding)    │
│                          │ • gap_fraction_above_2h: Fraction $> 2.0h$ (hesitation)    │
├──────────────────────────┼────────────────────────────────────────────────────────────┤
│ 6. Letter Sizing & Ratios│ • word_height_cv: $\sigma_{\text{height}} / \mu_{\text{height}}$│
│    (`features/size.py`)  │ • h_over_r: $h / r$ (macrographia / micrographia)          │
│                          │ • word_width_per_char: Total width / expected char count   │
│                          │ • ascender_descender_ratio: $(H_{\text{asc}} + H_{\text{desc}})/h$│
│                          │ • matra_ratio: $(H_{\text{upper}} + H_{\text{lower}}) / h$ │
├──────────────────────────┼────────────────────────────────────────────────────────────┤
│ 7. Devanagari Mechanics  │ • shirorekha_rms_deviation_norm: Headline wobble / $h$     │
│    (`features/hindi.py`) │ • shirorekha_breaks_per_word: Discontinuities per word     │
│                          │ • shirorekha_tilt_var: Variance of headline angles         │
├──────────────────────────┼────────────────────────────────────────────────────────────┤
│ 8. Stroke Fragmentation  │ • components_per_unit_width: $N_{\text{cc}} / (W / h)$     │
│    (`features/fragmentation.py`)│ • junctions_per_unit_width: $N_{\text{junc}} / (W / h)$│
│                          │ • endpoints_per_unit_width: $N_{\text{end}} / (W / h)$     │
│                          │ • words_written_ratio: $N_{\text{written}} / N_{\text{prompt}}$│
└──────────────────────────┴────────────────────────────────────────────────────────────┘
```

---

## 6. Dataset Normalization & Student-Level Engineering

### 6.1 Robust Within-Cohort Z-Scoring (`pipeline/dataset_assembly.py`)
Children develop handwriting motor control nonlinearly across elementary school: Grade 3 children naturally exhibit larger handwriting, slower stroke formation, and greater baseline wobble than Grade 8 students. Comparing raw feature values directly across grades introduces developmental confounding.

The system applies **Robust Within-Cohort Z-Scoring**:
$$z = \frac{x - \text{median}(X_{\text{cohort}})}{1.4826 \cdot \max(\text{MAD}(X_{\text{cohort}}), 10^{-6})}$$
where the median and Median Absolute Deviation (MAD) are computed strictly within homogeneous cohorts defined by:
$$\text{Cohort Cell} = (\text{Grade} \times \text{Script} \times \text{Task})$$
All z-scores are symmetrically clipped to $[-5.0, 5.0]$ to guard against extreme outliers.

### 6.2 Multi-Phenotype Student-Level Feature Engineering (`models/student_features.py`)
Handwriting dysgraphia is not a single deficit but a multi-phenotype developmental disorder characterized by:
1. **Motor Execution Instability**: High intra-individual variability across tasks.
2. **Cognitive Load Collapse**: Severe motor degradation when auditory or expressive cognitive load is introduced (Dictation vs Copying).
3. **Cross-Script Motor Divergence**: Disproportionate difficulty in complex cursive or headline scripts (Latin vs Devanagari).
4. **Episodic Motor Breakdown**: Severe, intermittent loss of motor control during handwriting execution.
5. **Motor Fatigue**: Degradation of motor precision from the beginning to the end of a writing session.

To capture these phenotypes, `models/student_features.py` aggregates the 1,145 sentence records into a **312-dimensional student matrix** across 215 students:

| Aggregation Dimension | Formula / Definition | Clinical Phenomenon |
|---|---|---|
| **Central Tendency** | $\mu = \text{mean}(f)$, $\sigma = \text{std}(f)$, $\text{CV} = \sigma / (\mu + \epsilon)$ | General motor ability and execution jitter |
| **Cognitive Load Gradient** | $\Delta_{\text{dict} - \text{copy}} = f_{\text{dictation}} - f_{\text{copy}}$ | Auditory-motor working memory strain |
| **Compositional Load** | $\Delta_{\text{own} - \text{copy}} = f_{\text{own}} - f_{\text{copy}}$ | Expressive language generation burden |
| **Cross-Script Divergence**| $\Delta_{\text{eng} - \text{hindi}} = f_{\text{latin}} - f_{\text{devanagari}}$ | Script motor asymmetry (headline vs loop) |
| **Episodic Breakdown** | $\max(f_{\text{fragmentation}})$, $\max(f_{\text{shirorekha\_wobble}})$ | Acute motor collapse episodes |
| **Fatigue Degradation** | $\Delta_{\text{fatigue}} = \text{mean}(f_{\text{tasks } 5,6}) - \text{mean}(f_{\text{tasks } 1,2})$ | Muscle exhaustion & motor stamina loss |

### 6.3 The Curated Clean 15 Multi-Phenotype Feature Set
Extensive stability selection (via randomized subsampled Lasso) and diagnostic false-negative auditing identified the top 15 most robust, invariant multi-phenotype features:
1. `components_per_unit_width_dict_mean`: Stroke fragmentation under auditory dictation pacing.
2. `components_per_unit_width_max`: Peak stroke fragmentation episode across the session.
3. `components_per_unit_width_std`: Variance in stroke fragmentation across tasks.
4. `components_per_unit_width_eng_minus_hindi`: Cross-script fragmentation divergence.
5. `endpoints_per_unit_width_mean`: Density of pen lifts and stroke interruptions.
6. `junctions_per_unit_width_mean`: Density of abnormal stroke re-tracing and overlaps.
7. `junctions_per_unit_width_own_mean`: Stroke re-tracing under free expressive composition.
8. `words_written_ratio_eng_minus_hindi`: Relative language completion drop under English.
9. `words_written_ratio_dict_minus_copy`: Task completion drop under dictation pacing.
10. `matra_ratio_std`: Inconsistency in Devanagari vowel sign (matra) sizing.
11. `matra_ratio_max`: Severe vertical protrusion of Devanagari matras.
12. `shirorekha_rms_deviation_norm_max`: Peak headline vertical wobble in Devanagari.
13. `tortuosity_median_cv`: Inconsistency in stroke path curvature and directness.
14. `word_width_per_char_std`: Irregularity in letter horizontal proportions and spacing.
15. `gap_fraction_above_2h_mean`: Frequency of abnormal inter-word hesitations ($> 2.0h$).

---

## 7. Machine Learning, Nested Cross-Validation & Benchmark Suite

### 7.1 Cross-Validation Rigor & Leakage Prevention (`models/cv.py`)
- **Outer Partitioning**: 5 repeats of 5-fold Stratified Group K-Fold on `student_id`.
- **Inner Tuning**: 3-fold Stratified Group K-Fold within each training partition to optimize hyperparameters and threshold bounds.
- **Student Weight Balancing**: Sample weights balance positive vs control base rates while equalizing student task contributions:
  $$w_i = \frac{1}{N_{\text{sentences\_for\_student}}} \times w_{\text{class}}$$
- **Probability Calibration**: Platt scaling / isotonic calibration fitted strictly on out-of-fold inner validation predictions to ensure predicted probabilities match empirical risk.
- **Confidence Intervals**: 1,000-sample empirical cluster bootstrap (clustered by `student_id`) computing 95% confidence intervals for AUROC, AUPRC, Brier score, and Sensitivities at 80% and 90% Specificity.

### 7.2 Candidate Estimator Architectures (`models/candidates.py`)
1. **Regularized Logistic Regression ($L_2$)**:
   - High bias, low variance linear baseline with balanced class weights.
   - Operates on standardized robust z-scores; highly robust against overfitting on small medical cohorts.
2. **Elastic-Net Logistic Regression ($L_1 / L_2$)**:
   - Balances sparse feature selection ($L_1$) with grouped collinearity shrinkage ($L_2$).
   - Regularization grid: $C \in [0.05, 5.0]$, $L_1\text{-ratio} \in [0.2, 0.7]$.
3. **Support Vector Classifier (SVM-RBF)**:
   - Radial basis function kernel capturing nonlinear geometric boundaries.
   - Grid: $C \in [0.1, 10.0]$, $\gamma \in [\text{'scale'}, 0.01, 0.05]$.
4. **Random Forest Classifier**:
   - Ensemble of 100 decorrelated decision trees with shallow depth ($d \in [3, 5]$), $\text{min\_samples\_leaf} \ge 3$, and subsampled feature partitions ($m = \sqrt{p}$).
5. **Histogram-Based Gradient Boosting (HistGBM)**:
   - Fast, regularized boosted trees with shallow depth ($d=2$), learning rate $\eta = 0.05$, and $L_2$ leaf regularization.
6. **Multi-Model Stacking Ensemble**:
   - Blends predictions across diverse inductive biases:
     $$P_{\text{ensemble}} = 0.50 \cdot P_{\text{GBM}} + 0.30 \cdot P_{\text{RF}} + 0.20 \cdot P_{\text{LR}}$$

### 7.3 Multi-Center Cross-School Evaluation (`models/cross_school_experiments.py`)
To test generalization against distribution shift across schools and paper formats:
- **External Validation (School A $\to$ School B)**: Models trained on School A (115 students) are evaluated on School B (100 students) with zero retraining.
- **Reverse Transfer (School B $\to$ School A)**: Models trained on School B are evaluated on School A.
- **Multi-Center Pooled Nested CV**: Pooled 5-repeat 5-fold evaluation across all 215 students (36 positives, 179 controls).
- **Domain Shift Analysis**: Employs Kolmogorov-Smirnov tests ($\alpha = 0.05$) to detect features that shift across school environments, verifying that the Clean 15 set remains domain-invariant.

---

## 8. Quality Assurance & Web Application Architecture (`qa/`)

The `qa/` subsystem provides an interactive platform for visual inspection, transparency verification, and human-in-the-loop annotation adjustments.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          QA WEB PLATFORM ARCHITECTURE                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Frontend Single-Page App (`qa/web/`)                                 │
│   ┌─────────────────────┐  ┌────────────────────┐  ┌────────────────┐ │
│   │  Student Roster     │  │  Multi-Layer       │  │ Diagnostic     │ │
│   │  • School Filters   │  │  Canvas Canvas     │  │ Feature Panel  │ │
│   │  • Grade Filters    │  │  • Raw Grayscale   │  │ • 37 Metrics   │ │
│   │  • Risk Badges      │  │  • Rule Overlay    │  │ • Norm Z-Scores│ │
│   │  • Status (Verified)│  │  • Clean Ink Mask  │  │ • Warning Flags│ │
│   └──────────┬──────────┘  │  • Baseline Vectors│  └───────┬────────┘ │
│              │             │  • Skeleton Graph  │          │          │
│              │             └─────────▲──────────┘          │          │
│              │                       │                     │          │
│              ▼                       ▼                     ▼          │
│   REST API Backend (`qa/app.py` - Python `aiohttp` on port 8090)       │
│   • GET  `/api/students`                  • GET  `/api/stats`          │
│   • GET  `/api/student/{id}`              • POST `/api/student/{id}/...`│
│   • GET  `/api/student/{id}/inspection`   • POST `/api/verify`         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 8.1 Backend API Server (`qa/app.py`)
An asynchronous Python service (`aiohttp`) providing RESTful endpoints:
- `GET /api/stats`: Cohort-wide completion rates, verified counts, positive base rates.
- `GET /api/students`: Filterable student roster with school, grade, status, and prediction scores.
- `GET /api/student/{id}`: Detailed metadata, task sentence bounding boxes, crop URLs, and clean ink mask URLs.
- `GET /api/student/{id}/inspection`: Deep multi-layer diagnostics, clean ink vs raw crop feature differentials, and topological graph statistics.
- `POST /api/student/{id}/sentence/{task_id}`: Live manual bounding box adjustment, triggering real-time re-cropping and feature re-computation.
- `POST /api/student/{id}/label`: Instant ground truth label override and clinical notes persistence.
- `POST /api/student/{id}/verify`: Mark student record as clinically verified.

### 8.2 Frontend Multi-Layer Canvas (`qa/web/index.html`, `style.css`, `app.js`)
- **Layer Toggle Pills**: Enables instant 1-click toggling between:
  - *Layer 1 (Raw Grayscale)*: Natural scan with notebook paper grain and lighting.
  - *Layer 2 (Rule Mask)*: Coral red overlay of detected ruling line trajectories.
  - *Layer 3 (Clean Ink)*: Stroke-preserved binary ink mask.
  - *Layer 4 (Baselines)*: Word bounding boxes, bottom anchor points, and fitted RANSAC regression lines.
  - *Layer 5 (Skeleton Graph)*: Medial axis skeleton with cyan junctions and red endpoints.
- **Keyboard Shortcuts**:
  - `L`: Cycle dysgraphia label (Unknown $\to$ Normal $\to$ Dysgraphic).
  - `V`: Toggle verification status.
  - `Delete` / `Backspace`: Remove spurious task crops.
  - `+`: Add custom task bounding box.
  - `ArrowLeft` / `ArrowRight`: Navigate previous / next student.

---

## 9. Automated Testing & Verification Suite (`tests/`)

The repository maintains an automated regression and mathematical verification suite across 8 dedicated test modules:

```
tests/
├── test_phase0.py              # Manifest integrity, schema validation, and fold partition safety
├── test_phase1.py              # Preprocessing, deskewing, line detection, rule reduction, skeletons
├── test_phase3.py              # Dataset assembly, robust z-scoring, and fold imputation
├── test_phase4.py              # Stratified Group K-Fold split safety, candidate model fitting
├── test_phase4_stage2.py       # Stability selection, student feature aggregation, benchmark pipeline
├── test_qa_app.py              # REST API endpoint responses and static asset routing
├── test_features_synthetic.py  # Synthetic perturbation validation (Spearman ρ ≥ 0.8)
└── test_features_invariance.py # Scale invariance (0.7x) and rotation invariance (±3°)
```

### 9.1 Mathematical Sensitivity Validation (`tests/test_features_synthetic.py`)
Validates that feature extractors reflect controlled synthetic handwriting perturbations:
- **Baseline Wobble**: Injecting vertical sinusoidal oscillations ($A \in [0, 20]\text{ px}$) verifies Spearman rank correlation $\rho \ge 0.80$ with `baseline_rmse_norm`.
- **Slant Variation**: Applying affine shear angles ($\theta \in [-30^\circ, +30^\circ]$) verifies $\rho \ge 0.80$ with `slant_mean_deg`.
- **Stroke Tremor**: Adding high-frequency Perlin noise to stroke contours verifies $\rho \ge 0.80$ with `jerk_proxy`.
- **Inter-Word Spacing**: Randomly scaling inter-word spaces verifies $\rho \ge 0.80$ with `gap_cv`.

### 9.2 Geometric Invariance Validation (`tests/test_features_invariance.py`)
Validates that features remain invariant under benign imaging variations:
- **Scale Invariance**: Rescaling sentence crops by $0.7\times$ verifies that scale-normalized features change by $< 5\%$.
- **Rotation Invariance**: Rotating crops by $\pm 3^\circ$ verifies that baseline-corrected features remain stable within $< 3\%$.

---

## 10. Data Structures & JSON Schemas

### 10.1 Sentence Schema (`{task_id}.json`)
```json
{
  "student_id": "SCHA_G3_Roll12",
  "task_id": "sentence_01_copy_hindi",
  "task_name": "copy_hindi",
  "script": "devanagari",
  "crop_bbox": [48, 412, 1892, 194],
  "lines_count": 1,
  "words": [
    {
      "word_index": 0,
      "bbox": [52, 420, 210, 160],
      "bottom_point": [157, 580],
      "script": "devanagari",
      "baseline_y": 574.2,
      "residual": 5.8
    }
  ],
  "baseline": {
    "slope": 0.0031,
    "intercept": 573.8,
    "rmse": 3.84
  },
  "ruling": {
    "spacing_r": 82.4,
    "nearest_rule_y": 576.0
  },
  "x_height_h": 46.2,
  "qa_flags": {
    "word_count_mismatch": false,
    "excessive_slant": false,
    "low_contrast": false
  }
}
```

### 10.2 Review Status Schema (`review_status.json`)
```json
{
  "student_id": "SCHA_G3_Roll12",
  "school": "school_a",
  "grade": 3,
  "label": 1,
  "verified": true,
  "verified_at": "2026-10-10T16:22:04Z",
  "reviewer_notes": "Severe stroke fragmentation in dictation task; matra misalignment.",
  "tasks_verified": {
    "sentence_01_copy_hindi": true,
    "sentence_02_copy_english": true,
    "sentence_03_dictated_hindi": true,
    "sentence_04_dictated_english": true
  }
}
```

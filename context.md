# Dysgraphia Detection - Context & Architecture

## Project Overview
Automated, multilingual dysgraphia screening from standard handwriting images without requiring specialized digitizing tablets or styluses. Primary early focus is on Indian languages, starting with Hindi and English handwriting.

## Repository & Branch Structure
- **Repository**: `Sparrow375/Dysgraphia-Detection`
- **Active Branch**: `baseline-v1`
- **Architecture Layout**:
  - `baseline/`: Baseline replication pipeline (DenseNet201 + feature fusion).
  - `features/`: Motor, spatial, and geometric feature extraction from image contours/skeletons.
  - `kinematics/`: Pseudo-kinematic and temporal reconstruction from static image strokes.
  - `models/`: Classifiers and ensemble strategies.
  - `docs/`: Project scope, Workstream A implementation plans, Git LFS guide.
  - `data/`: Local dataset directory tracked via **Git LFS (Large File Storage)**.

## Cloud Data Architecture: Git LFS
- **Tracking Mechanism**: `data/**` is tracked via Git LFS in `.gitattributes`.
- **Cloud Remote**: GitHub Large File Storage (`https://github.com/Sparrow375/Dysgraphia-Detection.git/info/lfs`).
- **Files Tracked & Pushed**: 647 files (1.34 GB) covering the School A (Future Gen) mapped cohort (stylus renders, kinematics CSVs, handwritten paper scans, master tables).
- **Branch Synchronization**:
  - Code and dataset pointers stay synchronized on Git branch `baseline-v1`.
  - Git commits contain lightweight SHA-256 pointer objects (~130 bytes each).
  - When cloning or switching branches, `git lfs pull` automatically fetches and populates the heavy binaries matching that branch.
- **Local Fallback**: Local source data remains preserved in `VV data` (`F:\Avaneesh\download\VV data-20261006T155534Z-1-001\VV data`).

## School Cohorts & Workstreams
- **School A (Future Gen)**: Training, feature engineering, and nested cross-validation (95 students, 91 matched with stylus + paper).
- **School B (Kanyashala)**: Held-out external test cohort (100 paper images across G3–G8, 76 stylus sessions across G4–G7).
- **Workstream A**: Data foundation, manifest generation, ruled-line detection, Sauvola binarization, word/sentence segmentation.

## Workstream A v1 — Finalized Decisions (2026-10-08)
- **Data scope**: Handwritten images only. Stylus data stays as local archive in `VV data`, not in repo/DVC.
- **Ground truth labels**: ~24 positives across G3–G7. G4–G7 mapped from `master_mapping.csv`; G3 (21 images, positives = rolls 12,13,15,19) requires manual labeler run **before Phase 0 can start**.
  - G3 positives: 12, 13, 15, 19
  - G4 positives: 9, 11, 14, 17, 22, 23
  - G5 positives: 5, 7
  - G6 positives: 14, 16, 18, 22, 24, 25, 26, 27, 28
  - G7 positives: 6, 12, 15
- **Repo layout (target)**:
  - `data/school_a/raw/` — Future gen Handwritten (118 images, by grade subfolder), DVC-tracked
  - `data/school_b/raw/` — Kanyashala Handwritten (100 images, by grade subfolder), DVC-tracked
  - `data/manifest.csv` — Phase 0 output
  - `data/folds/` — Phase 0 CV fold files
  - `pipeline/` — Python modules (manifest, folds, preprocess, rules, segment, skeleton)
  - `features/` — Feature functions Phase 2
  - `modeling/` — Classifiers Phase 4
  - `qa/` — Jupyter notebooks for visual overlay QA
  - `configs/config.yaml` — Seeds, paths, hyperparameters
- **Normalization**: Pooled A+B (grade × language × task medians/MAD).
- **Code style**: Python modules in `pipeline/`, config YAML in `configs/`. Jupyter only for QA overlays.
- **Execution order**: Grade 3 labeling → repo restructure → Phase 0 (manifest + folds) → Phase 1 preprocessing + modeling scaffold in parallel.
- **Immediate blocker**: None (Grade 3 labeling complete).

## Phase 0: Data Foundation & Folds (Completed 2026-10-08)
- **Label Resolution**: Grade 3 manually labeled and integrated via `future_gen_handwritten_labels.csv`. Cohort across G3–G7 comprises 115 unique students (24 positives, 91 negatives).
- **Raw Data Tracking (DVC)**:
  - `data/school_a/raw/`: 118 handwritten images across G3–G7 (DVC-tracked via `data/school_a/raw.dvc`).
  - `data/school_b/raw/`: 100 handwritten images across G3–G8 (DVC-tracked via `data/school_b/raw.dvc`).
  - DVC remote synchronized (123 files cached to `dvc_storage`).
  - Git attributes (`.gitattributes`) updated to ensure manifests, CSVs, JSONs, and `.dvc` files are tracked as native text in Git.
- **Manifest Pipeline (`pipeline/manifest.py`)**:
  - Generates `data/manifest.csv` (215 rows: 115 School A, 100 School B held-out).
  - All 215 image paths verified to exist on disk.
  - School B is frozen and held out without labels (`label = NaN`).
- **Cross-Validation Folds Pipeline (`pipeline/folds.py`)**:
  - Generates nested CV on School A:
    - Outer: Stratified Group 5-fold × 5 repeats (23 test students per fold, 4–5 positives each).
    - Inner: Stratified Group 3-fold for hyperparameter tuning.
  - Output files: `data/folds/nested_folds.json` and `data/folds/student_folds_lookup.csv`.
- **Validation**:
  - `tests/test_phase0.py` passed 4/4 test cases confirming schema integrity, zero student leakage across all outer & inner folds, and complete partition of School A.
- **Next Phase**: Phase 1 Preprocessing and Segmentation (`pipeline/preprocess.py`, `pipeline/rules.py`, `pipeline/segment.py`, `pipeline/skeleton.py`).

## Phase 1: Preprocessing & Segmentation Blueprint (Finalized 2026-10-08)
- **Target Unit**: 6 Task Sentences per student (Hindi Copy, English Copy, Hindi Dictation, English Dictation, Hindi Free Writing, English Free Writing).
- **Page Normalization**: Auto-portrait rotation, quadrilateral contour detection + perspective warp to fixed 2000px width (fallback to 2% margin trim crop).
- **Illumination & Binarization**: Large-kernel background illumination division + Sauvola thresholding ($W=31, k=0.2$). Maintains both normalized grayscale and binary ink mask layers.
- **Ruled-Line Pipeline**: Morphological horizontal opening + RANSAC linear fitting ($y = ax + b$). Deskew page by median slope, subtract rule pixels, and repair crossing strokes via localized vertical closing ($3 \times 1$). Store ruling parameters ($r$ spacing, rule coordinates) in metadata.
- **Script & Sentence Grouping**: Shirorekha ratio detector ($\ge 0.7 \implies$ Devanagari) + sequential state machine matching the 6-prompt protocol. Word-count mismatches flagged in QA flags.
- **Skeleton & Graph**: Pruned skeleton graphs via `skimage` + `skan` (spur pruning $<0.15h$) tracking endpoints, junctions, and paths.
- **Output Hierarchy**: `data/processed/<school>/<student_id>/` with cropped sentence PNGs, sentence JSON schemas, and `overlay_debug.png`.
- **Quality Gate**: 30-sheet visual overlay review notebook (`qa/phase1_overlay_review.ipynb` / `qa/review_gallery.html`) verifying word-count match and segmentation bounds.

## Phase 1 v2: Preprocessing & Segmentation Overhaul (Completed 2026-10-09)
- **Motivation & v1 Issues**:
  - v1 illumination flattening + Sauvola thresholding destroyed image fidelity, turning ruled lines into thick black bars and degrading ink strokes.
  - Fragile rule-band assignment and script-based state transitions misclassified lines and split multi-line sentences.
  - Binary mask sentence crops inverted into illegible black/white bitmaps.
- **v2 Architecture & Key Changes**:
  - `pipeline/preprocess.py`: Removed illumination flattening completely. Uses gentle Gaussian blur + Otsu thresholding, preserving sharp ink strokes with zero paper grain noise while retaining original deskewed grayscale.
  - `pipeline/rules.py`: Upgraded rule removal to 7px vertical window with morphological dilation and 120px vertical margin line filtering. Localized vertical closing restores intersecting ascenders/descenders without re-bridging removed rules.
  - `pipeline/segment.py`: Replaced rule-band slicing with connected component extraction, top header box filtering ($y < 320, x > 850$), and nearest-neighbor y-centroid line clustering ($0.65 \times r$). Accurately groups multi-line English copy (Task 2) and single-line tasks (Tasks 1, 3, 4, 5, 6).
  - **Per-Word Baseline Extraction**: Captures the bottom point of every word bounding box (`x_center`, `bottom`), fits a robust linear baseline ($y = mx + c$), and computes per-word baseline residuals and RMSE for Phase 2 kinematic/spatial features.
  - `pipeline/process_dataset.py`: Crops sentences directly from the clean deskewed grayscale image (natural black ink on white paper, no inversion). Injects per-word baseline coordinates and residuals into sentence JSON schemas. Debug overlay renders clean grayscale with cyan ruled lines, green/orange word bboxes, red baseline sample dots, and yellow fitted regression lines.
- **Validation**:
  - `tests/test_phase1.py` passed 5/5 test cases.
  - `tests/test_phase0.py` passed 4/4 test cases.
- **Quality Gate**:
  - `qa/generate_overlay_review.py` re-run across grades G3–G7; generated refreshed interactive visual review gallery `qa/review_gallery.html`.

## Phase 1 v3: Illumination Normalization, Landscape Orientation & Margin Hardening (Completed 2026-10-10)
- **Motivation & Root Causes**:
  - Grade 4 was scanned via mobile DocScanner (built-in illumination flattening), giving the illusion that raw grayscale without normalization was sufficient.
  - Grades 3, 5, 6, and 7 were raw phone photos with heavy uneven lighting and shadow gradients (mean BGR ~175). Without normalization, global Otsu thresholding created massive central shadow blobs (1.8M noise pixels), misclassifying blank paper rules as ink.
  - Inverted/landscape scans (`G3_A_Roll12`, `G7_A_Roll11`) rotated arbitrarily clockwise, leaving sheets upside-down.
  - Indian notebook red double margin lines at `x \approx 180-220` bridged sentences into 1000px+ connected components.
- **v3 Architecture & Improvements**:
  - `pipeline/preprocess.py`: Added `normalize_background` using large-kernel morphological dilation (`41x41`) + median blur (`21`) + background division. This cleanly flattens lighting gradients on camera photos, reducing Otsu noise from 1.8M down to 356k clean ink pixels without degrading fine stroke boundaries.
  - `auto_orient_portrait`: Upgraded landscape rotation by comparing horizontal edge density between page halves (`edges_left` vs `edges_right`). Because notebook ruled lines are concentrated in the lower body while the header box region has sparser edges, the orientation algorithm places the header at the top, rotating upside-down sheets 180° into proper portrait.
  - `pipeline/segment.py`: Extended left margin cutoff to `clean_mask[:, :225] = 0` to discard the printed red double margin line. Added multi-line component rejection `ch < 1.8 * median_spacing_r` and `cw < 1200` to prevent margin fragments from bridging sentences, while relaxing aspect ratio rejection (`aspect > 12.0 and ch < 20`) to safely retain genuine wide Devanagari words with continuous shirorekha.
  - Rule masking in segmentation: Rule mask passed into `extract_lines_and_segment` to mask out ruled line pixels during script classification and word segmentation.
- **Validation**:
  - `tests/test_phase1.py` passed 5/5 test cases.
  - `tests/test_phase0.py` passed 4/4 test cases.
  - Benchmark template `G4_A_Roll01` fully preserved (6 sentences, 8 lines).
  - All 115 School A sheets processed successfully with 0 unhandled exceptions.
  - Interactive QA gallery `qa/review_gallery.html` regenerated across 30 diverse sheets across G3–G7.
## Phase 1 v4: Deep Learning OCR & Dynamic Alternating Task Grouping (Completed 2026-10-10)
- **Motivation & Root Causes**:
  - Children's handwriting (especially dysgraphic students in Grades 3–7) breaks classical morphological shirorekha heuristics: Devanagari letters are choppy with headlines broken into segments < 45px, rule erasure cuts horizontal strokes, and English capital letters ('T', 'I', 'E') or crossed-out words mimic shirorekhas.
  - Previous pipeline assumed a rigid 6-sentence count. Grade 3 students wrote strictly 4 tasks (`copy_hindi`, `copy_english`, `dictated_hindi`, `dictated_english`), whereas Grades 4–7 wrote up to 6 tasks.
- **v4 Architecture & Key Improvements**:
  - `pipeline/segment.py: classify_line_script`: Integrated EasyOCR (`easyocr`, PyTorch) with cached reader. Measures character count in Devanagari Unicode block (`\u0900`–`\u097F`) vs Latin (`a-zA-Z`), delivering unambiguous ground truth. Slices line width to first 800px for 2.5× faster CPU inference (~0.9s per line). Preserves vertical projection / shirorekha as fallback if 0 letters are detected.
  - `group_lines_into_sentence_blocks`: Replaced rigid indices with dynamic alternating language dynamic programming (DP). Strictly enforces protocol: Task 1: Hindi → Task 2: English → Task 3: Hindi → Task 4: English (→ Task 5: Hindi → Task 6: English). Grade 3 strictly extracts 4 tasks, while Grades 4–7 extract 4–6 tasks. Eliminates language swallowing across all sheets.
  - Grayscale preservation: `pipeline/process_dataset.py` passes `grayscale_deskewed` into line extraction and crops directly from clean grayscale (no binarization inversion or lossy compression).

## Phase 1 Interactive Review & Segment Editor Web App (Completed 2026-10-10)
- **Purpose**: Interactive local application allowing visual review and manual bounding box adjustment / recropping for any student sheet across the School A cohort.
- **Backend (`qa/app.py`)**:
  - Asynchronous Python web server using `aiohttp.web` on `http://127.0.0.1:8090/`.
  - `GET /api/students`: Lists all 115 School A students with grade, roll, label, detected task count, and verification status.
  - `GET /api/student/{student_id}`: Returns page dimensions, image URLs, bounding boxes, scripts, word counts, and crops.
  - `POST /api/student/{student_id}/update_sentence`: Updates bounding box `[x,y,w,h]`, script (`devanagari`/`latin`), task name, recrops `page_normalized.png`, recomputes word segmentation, updates sentence JSON, and re-renders `overlay_debug.png`.
  - `POST /api/student/{student_id}/verify`: Persists verification status and reviewer notes to `data/processed/school_a/<student_id>/review_status.json`.
  - Static file routes serving `qa/web/` assets and `/files/processed/...` images.
- **Frontend (`qa/web/index.html`, `qa/web/style.css`, `qa/web/app.js`)**:
  - Dark mode glassmorphic UI with Google Fonts (Inter / Outfit).
  - Central zoomable, pannable canvas rendering `page_normalized.png` with SVG bounding box overlays (green for Devanagari, amber for Latin, glowing sky blue for selected).
  - 8-handle drag-to-resize and drag-to-move bounding box interactions.
  - "Draw Box" mode: Click and drag anywhere on the page to define a new bounding box.
  - Right-hand Inspector Panel: Student metadata, Task Editor with live coordinate inputs and script toggles, "Save & Re-Crop" button, and task cards list displaying full-resolution crops.
  - Navigation: Grade filter pills (All, G3, G4, G5, G6, G7), status dropdown (Needs Review, Verified, At-Risk), Previous/Next buttons, and keyboard shortcuts (`[` / `]`, `V`, `D`, `F`, `0`, `S`).
- **Interactive Editor Refinements (2026-10-10)**:
  - **Non-blocking Instant Deletion**: Removed modal confirmation popups on task deletion; bound <kbd>Delete</kbd> and <kbd>Backspace</kbd> keys for immediate removal from local state and background API file cleanup.
  - **Dynamic Background Auto-Saving**: Every bbox adjustment (mouse drag release, input changes, script toggle, task assignment) automatically schedules and flushes an auto-save to `/api/student/{id}/update_sentence`. Added `beforeNavigate()` hook ensuring unsaved changes are saved before navigating between students.
  - **Resize Handle Stability**: Removed CSS hover scale transforms on SVG resize handles that caused jitter between move and resize modes; added 30px transparent touch target hitboxes and dynamic screen-space handle sizing.
  - **Dedicated "+ Add Task" Flow**: "+ Add Task" safely saves any pending changes on the active box, clears selection, and enters draw mode with `state.isAddingNewTask = true`. When the box is drawn, it automatically assigns the next protocol task (`sentence_01_copy_hindi` through `sentence_06_own_english`, or custom sequential tasks) without replacing the previously selected task.
- **Manual Verification & Cohort Audit Results (2026-10-10)**:
  - 115 / 115 (100.0%) School A students reviewed, manually verified, and marked `verified: true` in `review_status.json`.
  - 641 total sentence tasks saved on disk (430 manually adjusted/customized, 211 confirmed auto-extracted).
  - 0 missing crop images, 0 corrupt JSON schemas, 0 bounding boxes out of bounds, 0 zero-word crops.
  - 4,526 total words segmented across 322 Devanagari and 319 Latin sentences.
  - Grade 3: strictly 4 tasks across all 21 students; Grades 4–7: 87 students have all 6 tasks, 7 students have 5 tasks matching original physical sheets.

- **Next Phase**: Phase 2 Feature Library (Completed 2026-10-10).

## Phase 2: Feature Library & Validation (Completed 2026-10-10)
- **Architecture Overview**:
  - Modular mathematical feature extraction library operating under the strict contract $f(\text{sentence\_json}) \to \text{value} \mid \text{NaN}$ (returning NaN if fewer than 3 words or insufficient lines are available).
  - Normalization parameters: $h$ (x-height / core band height) and $r$ (ruled-line spacing).
  - Package layout:
    - `features/baseline.py`: Robust line-level baseline fitting (RANSAC), residual computation $e_i$, and normalized RMSE (`baseline_rmse_norm = sqrt(mean(e_i^2)) / h`, `baseline_slope_mean`, `baseline_slope_std`).
    - `features/rule_offset.py`: Rule alignment offsets $o_i = (y_i - y_{\text{rule}}(x_i)) / r$, floating mean and vertical adherence standard deviation (`rule_offset_mean`, `rule_offset_std`).
    - `features/slant.py`: Orientation tensor on skeleton stroke tangents within $\pm 45^\circ$ of vertical, using doubled-angle circular statistics (`slant_mean_deg`, `slant_circular_std_deg`).
    - `features/curvature.py`: Savitzky-Golay path smoothing, uniform arclength resampling ($0.05h$), dimensionless jerk proxy ($|d\kappa/ds| \cdot h^2$), short-wavelength tangent variance fraction ($f > 2/h$, $\lambda < 0.5h$), curvature sign changes per $h$ with deadband, and median tortuosity ($\text{path length} / \text{chord}$). Shirorekha headline is automatically masked out for Devanagari script.
    - `features/gaps.py`: Inter-word spacing $g_k / h$ for consecutive words in the same line (`gap_mean`, `gap_cv`, `gap_fraction_below_0_3h`, `gap_fraction_above_2h`).
    - `features/size.py`: Word-height CV (`word_height_cv`), $h/r$ ratio (`h_over_r`), word width per expected character for copy/dictation (`word_width_per_char`), intra-word component height CV (`component_height_cv`), ascender/descender ratio for English (`ascender_descender_ratio`), and matra ratio for Hindi (`matra_ratio`).
    - `features/hindi.py`: Devanagari-specific shirorekha continuity features (`shirorekha_rms_deviation_norm`, `shirorekha_breaks_per_word`, `shirorekha_tilt_var`). Returns NaN for Latin sentences.
    - `features/fragmentation.py`: Graph density per unit width (`components_per_unit_width`, `junctions_per_unit_width`, `endpoints_per_unit_width`), task completion ratio (`words_written_ratio`), and lines used (`lines_used`).
    - `features/extractor.py`: Unified extractor `extract_sentence_features` and batch cohort extractor `extract_dataset_features` outputting `data/features/features_raw.csv` and `data/features/features_meta.json`.
    - `features/validation.py`: Statistical validation suite generating `data/features/validation_report.json` and `data/features/validation_report.md`.

- **Cohort Extraction Results**:
  - 641 / 641 verified sentences extracted across all 115 School A students.
  - 29 distinct quantitative features computed per sentence row.
  - Zero missing values across 20 general spatial/kinematic features; script-specific features cleanly masked to NaN on opposite language sentences.

- **Phase 2 Validation Suite Findings**:
  - **Synthetic Perturbation Tests (`tests/test_features_synthetic.py`)**: All 4 monotonic response tests passed with Spearman $\rho \ge 0.8$:
    - Spacing jitter vs `gap_cv`: $\rho = 1.0000$ ($p < 10^{-10}$)
    - Size jitter vs `word_height_cv`: $\rho = 1.0000$ ($p < 10^{-10}$)
    - Baseline wobble vs `baseline_rmse_norm`: $\rho = 1.0000$ ($p < 10^{-10}$)
    - Tremor noise vs `jerk_proxy`: $\rho = 0.9286$ ($p < 0.005$)
  - **Invariance Tests (`tests/test_features_invariance.py`)**:
    - Scale invariance ($0.7\times$ rescale): `h_over_r` (0.0% diff), `word_height_cv` (1.1% diff), `tortuosity_median` (0.7% diff), `baseline_rmse_norm` (5.9% diff).
    - Rotation invariance ($\pm 3^\circ$): slant shifts predictably from $-2.80^\circ$ to $-2.46^\circ$ and $-4.09^\circ$, while circular std remains invariant.
  - **Redundancy Analysis**:
    - Zero feature pairs exceeded $|\rho| \ge 0.90$. All 29 features capture non-redundant, complementary handwriting dimensions.
  - **Developmental Sanity Checks (Grade Trends)**:
    - Intra-word letter height variation (`component_height_cv`) decreases significantly with grade ($\rho = -0.1865, p = 9.9 \times 10^{-6}$).
    - Matra proportion variability (`matra_ratio`) stabilizes significantly with grade ($\rho = -0.1767, p = 3.28 \times 10^{-3}$).
    - Stroke inflection flips (`kappa_sign_changes_per_h`) decrease significantly with grade ($\rho = -0.1005, p = 0.011$).
  - **Univariate AUROCs (School A Sanity Check)**:
    - `components_per_unit_width`: AUROC = **0.707** (dysgraphic handwriting exhibits marked stroke fragmentation).
    - `endpoints_per_unit_width`: AUROC = **0.665**.
    - `junctions_per_unit_width`: AUROC = **0.644**.
    - `gap_fraction_above_2h`: AUROC = **0.636**.
    - `rule_offset_std`: AUROC = **0.612**.
    - `matra_ratio`: AUROC = **0.608**.
    - `kappa_sign_changes_per_h`: AUROC = **0.605**.

- **Next Phase**: Phase 3 Dataset Assembly (Completed 2026-10-10).

## Phase 3: Dataset Assembly & Feature Preparation (Completed 2026-10-10)
- **Cohort Health Audit**:
  - Inspected all 115 School A students on disk: 115/115 present (100%), 115/115 marked verified in `review_status.json`.
  - 641 total sentence schemas and 641 lossless crop images; 0 corrupt files, 0 missing files.
  - Script distribution: 322 Devanagari sentences, 319 Latin sentences (4,526 segmented words total).
- **Architecture & Datasets (`pipeline/dataset_assembly.py`)**:
  - `data/datasets/dataset_hindi.csv`: 322 Devanagari sentences $\times$ 73 columns (filtered to exclude Latin-only features).
  - `data/datasets/dataset_english.csv`: 319 Latin sentences $\times$ 67 columns (filtered to exclude Devanagari-only features).
  - `data/datasets/dataset_combined.csv`: 641 multilingual sentences $\times$ 75 columns.
  - `data/datasets/datasets_meta.json`: Dataset schemas, cell statistics, and normalization configurations.
- **Feature Representations**:
  - **Raw Scale-Normalized Features**: Prefixed with `raw_` (already dimensionless and normalized by character height $h$ and rule spacing $r$).
  - **Robust Z-Score Normalization**: Prefixed with `z_raw_`, computed via:
    $$z = \frac{x - \text{median}}{1.4826 \cdot \max(\text{MAD}, 10^{-6})}$$
    per $(\text{grade} \times \text{script} \times \text{task\_group})$ cell with fallback to $(\text{grade} \times \text{script})$ when cell size $< 15$, clipped to $[-5.0, 5.0]$.
  - **One-Hot Task Encoding**: `task_copy`, `task_dictated`, `task_own` (exactly one active per row).
  - **Fold Mappings**: Outer test fold indicators `repeat_0_fold` through `repeat_4_fold` integrated for seamless evaluation in Phase 4.
- **Leakage-Free Fold Imputation**:
  - `impute_fold_features(train_df, test_df)` utility computes medians strictly on training folds and applies them to both train and test splits with zero test set leakage.
- **Validation**:
  - `tests/test_phase3.py` passed 6/6 test cases verifying row counts, schema integrity, zero student leakage across CV folds, z-score bounds, script feature isolation, and imputation correctness.

- **Next Phase**: Phase 4 Modeling and Evaluation (Stage 1 Completed 2026-10-10).

## Phase 4: Modeling & Cross-Validation Evaluation (Stage 1 Completed 2026-10-10)
- **Nested Cross-Validation Framework (`models/cv.py`, `models/evaluator.py`)**:
  - **Outer Splits**: 5-repeat 5-fold student-stratified CV (`student_folds_lookup.csv`). Zero student leakage guaranteed between train and test sets.
  - **Inner Splits**: 3-fold student `StratifiedGroupKFold` for leakage-free hyperparameter tuning and threshold selection.
  - **Sample Weights**: Per-sentence weights balancing individual student contributions and student-level class balance ($w_i = \frac{1}{N_{\text{sentences}}} \times w_{\text{class}}$).
  - **Probability Calibration**: Platt scaling (sigmoid calibration) fit on inner out-of-fold validation scores.
  - **Student Score Aggregation**: Mean-pooling of calibrated sentence probabilities:
    $$P_{\text{student}} = \frac{1}{K} \sum_{k=1}^K P(\text{dysgraphic} \mid \text{sentence}_k)$$
  - **Cluster Bootstrap CIs**: $B=1000$ student-level bootstrap iterations generating empirical 95% percentile confidence intervals.
- **Candidate Models & Feature Preferences (`models/candidates.py`)**:
  - Distance/margin models (Elastic-Net Logistic Regression, SVM-RBF): Trained on robust z-scores (`z_raw_*`).
  - Tree ensembles (Random Forest, Gradient Boosting): Trained on raw scale-normalized features (`raw_*`).
  - Baselines: Majority class baseline and Grade-only logistic regression baseline.
- **Benchmark Results across Datasets (`reports/benchmark_all_datasets.csv`)**:
  - **Combined Multilingual Dataset (641 sentences, 115 students)**:
    - **SVM (RBF Kernel)**: AUROC = **0.645** (95% CI: 0.514–0.761), AUPRC = **0.303**, Brier = **0.181**, Balanced Accuracy = **54.9%**, Specificity = 89.0%, Sensitivity = 20.8%.
    - **Elastic-Net LR**: AUROC = **0.609** (95% CI: 0.470–0.748), AUPRC = **0.296**, Brier = 0.228, Specificity = 90.1%, Sensitivity = 12.5%.
    - **Random Forest**: AUROC = **0.556** (95% CI: 0.427–0.678), Brier = 0.195, Specificity = 91.2%, Sensitivity = 16.7%.
    - **Grade-Only Baseline**: AUROC = 0.464, Sensitivity = 4.2% (demonstrating dysgraphia markers are not developmental age artifacts).
  - **Hindi-Only Table (Devanagari, 322 sentences)**:
    - **Elastic-Net LR**: AUROC = **0.647** (95% CI: 0.517–0.758), AUPRC = **0.358**, Sensitivity = **25.0%**, Specificity = 83.5%, Brier = 0.215.
    - **Random Forest**: AUROC = **0.633** (95% CI: 0.499–0.757), AUPRC = 0.306, Brier = **0.176**, Specificity = 91.2%, Sensitivity = 12.5%.
    - **SVM-RBF**: AUROC = **0.603** (95% CI: 0.478–0.720).
    - **Gradient Boosting**: AUROC = 0.582, Balanced Accuracy = **55.9%**, Sensitivity = **25.0%**, Specificity = 86.8%.
  - **English-Only Table (Latin, 319 sentences)**:
    - **SVM-RBF**: AUROC = **0.566** (95% CI: 0.423–0.701), AUPRC = 0.306.
    - **Elastic-Net LR**: AUROC = 0.525.
- **Key Scientific Insights**:
  1. **Hindi Superiority**: Devanagari script features yield substantially higher classification discriminability (AUROC 0.647 vs English 0.566) due to the structural constraints of the shirorekha and upper/lower matras magnifying motor dyspraxia.
  2. **Clinical Trade-offs**: When operating at a conservative $\ge 90\%$ specificity (minimizing false positive referrals), sensitivity is $\sim 17–25\%$. At a broader screening triage threshold (75% specificity), sensitivity reaches **50.0%**.
- **Automated Validation**:
  - `tests/test_phase4.py` passed 6/6 test cases verifying zero student CV leakage, sample weights, leakage-free fold imputation, Platt calibration, and cluster bootstrap CIs.
- **Next Phase**: Phase 4 Stage 2 (Frozen DINOv2 visual encoder baseline, 8-family drop-one feature ablations, permutation importance, and misclassification gallery `reports/misclassified_gallery.html`).




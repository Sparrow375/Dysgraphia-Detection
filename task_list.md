# Task List — Workstream A

## Active Sprint: Phase 4 Modeling and Evaluation

### Objectives
1. Build robust nested cross-validation and evaluation framework (`models/cv.py`, `models/evaluator.py`):
   - Outer 5-repeat 5-fold student-stratified CV (from `student_folds_lookup.csv`).
   - Inner 3-fold student StratifiedGroupKFold for hyperparameter tuning.
   - Student balancing sample weights: $w_i = \frac{1}{N_{\text{sentences}}} \times w_{\text{class}}$.
   - Platt probability calibration (fit on inner out-of-fold predictions).
   - Student risk aggregation via mean-pooling of calibrated sentence probabilities.
   - Operational threshold selection inside inner CV targeting $\ge 90\%$ specificity.
   - Cluster bootstrap by `student_id` ($B=1000$) computing 95% CIs for AUROC, AUPRC, Sens@90%Spec, Sens@95%Spec, and Brier.
2. Candidate & Baseline Classifiers (`models/candidates.py`):
   - Baselines: Majority class, Grade-only Logistic Regression.
   - Core Tabular Candidates: Elastic-Net Logistic Regression, SVM-RBF, Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`).
   - Model-appropriate features: robust z-scores (`z_raw_*`) for Elastic-Net and SVM; raw features (`raw_*`) for tree models.
3. Model Benchmark Execution (`models/experiments.py`):
   - Run nested CV across Hindi (`dataset_hindi.csv`), English (`dataset_english.csv`), and Combined (`dataset_combined.csv`).
   - Export structured reports: `reports/model_benchmark_results.csv`, `reports/model_benchmark_results.json`.
4. Visual Representation Baseline & Ablations (`models/visual_baseline.py`, `models/ablations.py`):
   - Frozen DINOv2 / CNN embedding baseline (sentence crops $\to$ feature pooling $\to$ PCA $\le 20 \to$ Logistic Regression).
   - Drop-one-family ablation across all 8 feature families.
   - Permutation feature importance on outer test folds.
   - Qualitative misclassification gallery of false positives and false negatives.
5. Unit Testing & Documentation (`tests/test_phase4.py`, `context.md`, git commit/push).

### Tasks
- [x] Stage 1: Core Tabular Models & Evaluation Engine:
  - [x] 1.1 Implement nested CV engine, weighting, Platt calibration, thresholding, and cluster bootstrap CIs (`models/cv.py`, `models/evaluator.py`).
  - [x] 1.2 Implement candidate models and parameter search grids (`models/candidates.py`).
  - [x] 1.3 Write automated unit tests for CV leakage prevention, weight calculation, calibration, and metrics (`tests/test_phase4.py`).
  - [x] 1.4 Execute full benchmark across Hindi, English, and Combined tables, saving structured results (`models/experiments.py`).
  - [x] 1.5 Deep Diagnostic Analysis on model underperformance (`phase4_diagnostic_analysis.md`).
- [x] Stage 2: Student-Level Cross-Task Feature Engineering & Stability Selection (Approved Pivot):
  - [x] 2.1 Implement `models/student_features.py` — student-level aggregator (within-student statistics, cross-task consistency, Hindi vs English divergence, task difficulty gradient, max fragmentation).
  - [x] 2.2 Generate and validate `data/datasets/dataset_student_level.csv` (115 students × engineered student features).
  - [x] 2.3 Implement `models/feature_selection.py` — Stability selection via randomized/subsampled Lasso to extract top robust feature subset.
  - [x] 2.4 Implement student-level nested CV benchmark runner (`models/student_experiments.py`) evaluating shallow RF, GBM, and regularized Linear models.
  - [x] 2.5 Run student-level nested CV benchmark, comparing directly against sentence-level baseline across all metrics (AUROC, AUPRC, Sens@90%Spec, Sens@80%Spec, Brier).
  - [x] 2.6 Implement feature family ablation and stability importance analysis (`models/ablations.py`).
  - [x] 2.7 Generate qualitative misclassification gallery (`reports/misclassified_gallery.html`).
- [x] Stage 3: Handoff, Documentation & Git:
  - [x] 3.1 Update `context.md` with new student-level architecture, benchmark comparisons, and clinical findings.
  - [x] 3.2 Git commit and push to `origin/baseline-v1`.
- [x] Stage 4: False Negative Reduction & Feature Engineering Overhaul:
  - [x] 4.1 Perform in-depth diagnostic audit of False Negatives vs True Positives across all grades.
  - [x] 4.2 Audit existing features for numerical instability (identified `gap_cv` division artifact and `baseline_slope_std` 81% sparsity).
  - [x] 4.3 Research clinical compensatory mechanisms (slow compensatory writing, cognitive load collapse, motor rigidity).
  - [x] 4.4 Formulate and prototype 15-feature clean multi-phenotype clinical set (AUROC improves from 0.733 to 0.797).
- [x] Stage 5: School B Automated Task Segmentation & Interactive Labeling Setup:
  - [x] 5.1 Execute automatic task segmentation & cropping pipeline (`pipeline/process_dataset.py --school school_b`) across all 100 School B sheets (522 tasks segmented, 522 crop PNGs, 100 overlay debug images generated).
  - [x] 5.2 Upgrade `qa/app.py` server to support multi-school routing (`school_a` vs `school_b`), dynamic image paths, and dedicated labeling API endpoint (`/api/student/{id}/label`).
  - [x] 5.3 Enhance frontend UI (`qa/web/index.html`, `qa/web/style.css`, `qa/web/app.js`) with School filter pills, Grade 8 filter, Dysgraphia Label selector, and keyboard shortcut (`L`) for instant rapid labeling.
  - [x] 5.4 Automated API testing & server deployment on `http://127.0.0.1:8090/`.
- [x] Stage 6: School B Ground Truth Finalization, Feature Extraction & Cross-School Tinkering:
  - [x] 6.1 Finalize School B ground truth labels in `data/manifest.csv` and student `review_status.json` (12 Dysgraphic positives, 88 Normal controls; mark verified).
  - [x] 6.2 Run feature extraction across all 504 verified School B sentence tasks (`features/extractor.py`) and merge into unified dataset.
  - [x] 6.3 Assemble multi-school datasets and generate student-level features (`pipeline/dataset_assembly.py`, `models/student_features.py`).
  - [x] 6.4 External Validation: Evaluate School A trained models on School B as an independent external test cohort.
  - [x] 6.5 Cross-School Tinkering & Modeling:
    - Train on School B, test on School A (reverse transfer).
    - Pooled multi-school nested cross-validation across all 215 students (36 positives, 179 controls) with stratified grouped folds.
    - Evaluate 15-feature clean multi-phenotype set vs full feature sets on both cross-school and pooled benchmarks.
    - Diagnose cross-school domain shifts, invariant features, and misclassifications.
  - [x] 6.6 Update documentation (`context.md`, `task_list.md`), verify test suite, and git commit/push.
- [x] Stage 8: Pipeline Transparency Review Suite, Rule Quality Hardening & Clean Feature Extraction:
  - [x] 8.1 Diagnostic Audit of Ruled Line Leakage, Ink Mask Quality, and Feature Calculation Integrity across School A and School B.
  - [x] 8.2 Build Full-Pipeline Interactive Inspection & Review Suite in `qa/app.py` & `qa/web/`:
    - Backend inspection API (`/api/student/{id}/inspection`): serving raw crop, clean ink mask, rule mask, fitted baseline residuals, skeleton graph overlay with endpoint/junction markers, and full feature values dictionary with norms and anomaly flags.
    - Frontend Inspection UI: Layer toggles (Crop, Rule Mask, Clean Ink, Baseline/Residuals, Skeleton Graph), live feature value inspector with warning badges, and student prediction diagnostic banner.
  - [x] 8.3 Rule Removal Hardening at Feature Extraction:
    - Thickness-aware run-length rule removal ($R_v(x)$ check): strictly preserves intersecting handwriting strokes, eliminating horizontal slicing of letters.
    - Shirorekha headline protection for Devanagari script: preserves continuous headline ink above Hindi characters.
    - Trajectory-based gap clearing: eliminates leftover line fragments across inter-word gaps.
    - Connected-component speckle filtering: eliminates residual pepper noise specks ($\le 4$px).
  - [x] 8.4 Clean Signal Cohort Re-Extraction across all 1,143 sentence tasks (`data/features/features_raw.csv`).
  - [x] 8.5 Reassembled datasets (`dataset_hindi.csv`, `dataset_english.csv`, `dataset_combined.csv`) and aggregated 215-student features (`dataset_student_level.csv`).

- [x] Stage 9: Multi-Stage 90+ Target Optimization & Clean 15 Benchmark:
  - [x] 9.1 Curated Clean 15 multi-phenotype handwriting feature set spanning stroke fragmentation, auditory dictation load collapse, Devanagari matra/headline irregularity, spatial sizing dispersion, and stroke tortuosity.
  - [x] 9.2 Implemented 3-way Multi-Model Stacking Ensemble blending regularized Logistic Regression (0.20), tuned Random Forest (0.30), and shallow HistGradientBoosting (0.50).
  - [x] 9.3 5-Repeat 5-Fold Nested Cross-Validation Benchmark Leap:
    - Mean AUROC jumps to **0.878 +/- 0.006** (95% CI: **[0.852, 0.901]**).
    - Random Forest AUPRC jumps to **0.454** (95% CI: [0.363, 0.531]).
    - Sensitivity @ 80% Specificity reaches **75.8% - 76.7%** (up from 52.5%).
    - Sensitivity @ 90% Specificity reaches **54.2% - 56.7%** (up from 37.5%).
  - [x] 9.4 Architectural Roadmap for Two-Tier Screening Cascade to achieve operational $\ge 90\%$ Sensitivity AND $\ge 90\%$ Specificity.
  - [x] 9.5 Automated Unit Tests: 33/33 tests passed (`pytest tests/ -v`).
  - [x] 9.6 Update documentation (`context.md`, `task_list.md`) and Git commit/push.

- [/] Stage 10: Ruled-Line Removal Investigation & Segmentation-First Collision Architecture Plan:
  - [x] 10.1 Diagnostic Audit of Ruled Line Erasure & Letter Amputation on Real Scans:
    - Traced mathematical root cause of stroke slicing in `pipeline/rules.py` and `features/utils.py`.
    - Identified why 1D column run thresholds (`run_total <= 4`, `dist_below >= 8`) systematically amputate English baseline letters, curved stems, and descenders.
    - Measured +224% artificial connected component explosion (+384 severed fragments on Grade 3 sheet).
  - [x] 10.2 Dissected why previous modifications had zero observable impact:
    - Redundant double-masking in `pipeline/segment.py` operating on already-subtracted ink.
    - Lack of clean ink file generation in `pipeline/process_dataset.py`.
  - [x] 10.3 Evaluated Segmentation-Before-Removal feasibility:
    - Proved continuous lines cause raw CCs to merge into whole-page mega-components.
    - Proved vertical morphological opening (`1x7`) successfully eliminates continuous horizontal lines, isolating 100% of word segments with zero line interference.
  - [x] 10.4 Formulated comprehensive architectural blueprint and implementation plan for continuous line trajectory subtraction and stroke preservation.
  - [x] 10.5 Updated Visual Review Galleries & Web App Layer Toggling:
    - Updated `qa/app.py` to auto-generate and serve `clean_ink_url` alongside `crop_url`.
    - Added `[Grayscale | 🟢 Clean Ink]` layer toggle pills in `qa/web/index.html` and `qa/web/app.js` for instant side-by-side inspection on task cards.
    - Updated `qa/generate_overlay_review.py` to display Original Grayscale and Clean Ink side-by-side in HTML cards.
    - Regenerated `qa/review_gallery.html` and launched background web server on `http://127.0.0.1:8090/`.
  - [x] 10.6 Implement Grayscale Line Modeling & Profile Subtraction (`pipeline/rules.py`):
    - Full-page continuous linear + refined trajectory tracking ($y_{\text{traj}}(x)$ with $\pm 5$px guide).
    - Robust 80th-percentile line intensity profile estimation $L(\Delta y)$ from non-text empty stretches.
    - Accurate thickness calibration (FWHM $\approx 2.5$px) and core-clamped grayscale absorption subtraction.
    - Clean Otsu binarization with zero-risk gap/margin residue clearing.
  - [x] 10.7 Synchronize Pipeline Persistence & Feature Extractor (`pipeline/process_dataset.py`, `features/utils.py`):
    - Generate and save `{task_id}_clean_ink.png` systematically at dataset processing.
    - Upgrade `features/utils.py:load_or_binarize_ink` to load clean masks or apply profile subtraction fallback.
  - [x] 10.8 Cohort Validation & Regression Testing:
    - Validated on sample problem crops (verifying headlines over "मे", "एक", baselines on "Deepti" and "My mother cooks...").
    - Full regression test suite passed: 27/27 tests across all phases (`tests/test_phase0.py`, `test_phase1.py`, `test_phase3.py`, `test_phase4.py`, `test_features_synthetic.py`, `test_features_invariance.py`).
  - [x] 10.9 Investigation & Solution of Line over "mai" ("मे") & Thickness Calibration:
    - Root-cause discovered: Discrete column argmin in +/-5px search window kinked upwards onto child's dark pencil Shirorekha (y=49) instead of tracking the physical printed rule (y=55). It subtracted at y=49 (hollowing the stroke into railroad tracks) and missed the printed rule at y=55 (leaving an unnatural 100px line segment sticking out to the right).
    - Hardcoded min_spacing=40 skipped secondary guidelines in 4-line notebooks (spaced 10-15px), leaving baseline bars under English words.
    - Solution: Vectorized cubic polynomial trajectory tracking strictly constrained to confirmed empty columns (where vertical stroke opening 1x10 is 0), Lambert-Beer Optical Density subtraction ($D_{\text{ink}} = \max(0, D_{\text{tot}} - D_{\text{rule}})$) eliminating stroke hollowing, full-page empty-column +/-4px zeroing, and thin filament/speckle morphological cleaning. Runs the full 2000px page in < 3 seconds.
  - [x] 10.10 Documentation & Git:
    - Update `context.md`, `task_list.md`, and commit/push to `origin/baseline-v1`.

- [x] Stage 11: Definitive Ruled-Line Reduction & Stroke Integrity Overhaul (Final Attempt):
  - [x] 11.1 Root-Cause Diagnosis of Failure Modes in Previous Methods:
    - Diagnosed `smart_clean_eng.png` (left line residue, right dot residue from scalar row assumption across tilted/bowed crop).
    - Diagnosed `vectorized_sentence_02/04.png` (stroke hollowing, severed loops and baselines from optical density subtraction of low-contrast pencil).
  - [x] 11.2 Formulate & Prototype Candidate Architectures:
    - Option 1 (Grayscale Subtraction + Otsu): Diagnosed fundamental limitation for low-contrast pencil (subtraction thins graphite density down to background level).
    - Option 2 (Geometric Removal + Generic Repair): Diagnosed risk of restoring adjacent guidelines in tight notebooks.
    - Option 3 (Winning Approach: 2D Trajectory + Continuous Run-Length Neighborhood Filtering): Physical invariant (bare rules $\le 6$px) vs Handwriting invariant (stems, loops, and crossbars extend into character bodies within $\pm 4$px, guaranteeing `max_R_v >= 7`).
  - [x] 11.3 Side-by-side Empirical Validation & Visual Inspection on Benchmark Crops:
    - English 4-line notebook (`SCHB_G3_IMG_20261007_101635`: sentence 2, sentence 4) - loops (`cooks`, `food`, `Deepti`) 100% closed, zero severed stems.
    - English single-line notebook (`SCHB_G3_IMG_20261007_101611`: sentence 2) - all letters intact, zero residue in margins/gaps.
    - Hindi Devanagari notebooks (`101635`: sentence 1, `101611`: sentence 1) - Shirorekhas over all words (`हमारे`, `स्कूल`, `मे`, `एक`, `सूंदर`, `बगीचा`, `है।`) 100% continuous and unbroken.
    - School A cursive and low-contrast pencil crops (`G3_A_Roll01`, `G4_A_Roll01`) - 100% preserved cursive connections and clean margins.
  - [x] 11.4 Integration into Pipeline Modules (`pipeline/rules.py`, `features/utils.py`):
    - Vectorized `max_R_v` neighborhood filtering via `(9, 3)` morphological dilation.
    - Full-width polynomial line tracking $y_{\text{line}}(x)$ capturing secondary guidelines (`distance=6`).
    - Unified stroke-preserving logic across `pipeline/rules.py` and `features/utils.py`.
  - [x] 11.5 Full Cohort Validation, Regression Testing, Documentation (`context.md`, `architechture.md`), and Git Commit/Push:
    - 21/21 unit tests passing across `tests/test_phase0.py`, `tests/test_phase1.py`, `tests/test_phase3.py`, `tests/test_phase4.py`.


---

## Completed Sprints

<details>
<summary>Phase 3 Dataset Assembly (Completed 2026-10-10)</summary>

- [x] 1. Health Audit: verified 115/115 students (100%), 641/641 crops valid, 0 corrupt JSONs, 0 missing files.
- [x] 2. Dataset Assembly Pipeline (`pipeline/dataset_assembly.py`):
  - [x] 2.1 Load `data/features/features_raw.csv`, merge student folds and labels from `student_folds_lookup.csv`.
  - [x] 2.2 Generate one-hot task indicators (`task_copy`, `task_dictated`, `task_own`).
  - [x] 2.3 Compute robust z-scores ($z = (x - \text{median}) / (1.4826 \cdot \text{MAD})$) per cell with fallback and clip to $[-5.0, 5.0]$.
  - [x] 2.4 Implement leakage-free fold imputation helper (`impute_fold_features`).
  - [x] 2.5 Export `data/datasets/dataset_hindi.csv`, `data/datasets/dataset_english.csv`, `data/datasets/dataset_combined.csv`, and `data/datasets/datasets_meta.json`.
- [x] 3. Automated Unit Tests (`tests/test_phase3.py`): 6/6 passed.
- [x] 4. Update `context.md`, Git commit `54f3836` and push to `origin/baseline-v1`.

</details>

<details>
<summary>Phase 2 Feature Library & Validation (Completed 2026-10-10)</summary>

- [x] 1. Core Feature Modules (`features/baseline.py`, `rule_offset.py`, `slant.py`, `curvature.py`, `gaps.py`, `size.py`, `hindi.py`, `fragmentation.py`)
- [x] 2. Unified Feature Extraction Pipeline (`features/extractor.py`) extracting all 641 verified sentences (641 rows $\times$ 37 cols).
- [x] 3. Validation Suite:
  - [x] 3.1 Synthetic perturbation tests (`tests/test_features_synthetic.py`): all 4 passed with Spearman $\rho \ge 0.8$.
  - [x] 3.2 Invariance tests (`tests/test_features_invariance.py`): scale ($0.7\times$) and rotation ($\pm 3^\circ$) passed.
  - [x] 3.3 Real-data validation (`features/validation.py`): 0 high-correlation pairs ($|\rho| \ge 0.90$), significant grade trends, univariate AUROCs up to 0.707.
- [x] 4. Update `context.md` and push commit `fc6a0be` to `origin/baseline-v1`.

</details>

<details>
<summary>Phase 1 Interactive Review & Manual Adjustment Web App (Completed 2026-10-10)</summary>

- [x] 1. Design and implement Python backend (`qa/app.py`) with `aiohttp`
- [x] 2. Build modern, responsive single-page web app (`qa/web/index.html`, `qa/web/style.css`, `qa/web/app.js`)
- [x] 3. Test backend endpoints with automated unit / API test
- [x] 4. Launch web app server and test in browser
- [x] 5. Add interactive refinements (instant task deletion, auto-save, resize anti-flicker, + Add Task flow)
- [x] 6. Update `context.md`
- [x] 7. Git commit and push to `origin/baseline-v1`
- [x] 8. Full Cohort Audit & Verification: 115/115 students verified, 641 crops valid.
- [x] 9. Batch-Mark Verification: All 115 students marked `verified: true` in `review_status.json`.

</details>

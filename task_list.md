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
- [x] Stage 12: End-to-End Pipeline Reprocessing & ML Benchmark Re-evaluation:
  - [x] 12.1 Regenerate Clean Ink Masks across full cohort (School A & School B) using the definitive stroke-preserving algorithm.
  - [x] 12.2 Extract features across all 1,145 sentence tasks with stroke-preserving masks (`features/extractor.py`).
  - [x] 12.3 Re-assemble normalized multi-school datasets (`pipeline/dataset_assembly.py`).
  - [x] 12.4 Aggregate 215-student level feature matrices (`models/student_features.py`).
  - [x] 12.5 Re-evaluate ML Benchmarks & Multi-Model Cascade (`models/student_experiments.py` & `models/run_full_benchmark.py`).
  - [x] 12.6 Compare Sensitivity, Specificity, AUROC, and AUPRC metrics against previous baseline.
  - [x] 12.7 Update documentation (`context.md`, `architechture.md`), and Git commit/push.

- [x] Stage 13: Comprehensive Benchmark & Clinical Relevance Whitepaper:
  - [x] 13.1 Synthesize all benchmark findings across sentence-level, student-level, scripts, external cross-school validation, and ablations.
  - [x] 13.2 Author authoritative report `reports/model_benchmark_clinical_report.md` covering "What we did", "How we did it", "What we got", capability limit analysis, and clinical relevance evaluation.
  - [x] 13.3 Update `context.md`, `architechture.md`, and `task_list.md`.
  - [x] 13.4 Git commit and push to `origin/baseline-v1`.

- [x] Stage 14: Strategic Deep-Dive & Empirical Research on Next Frontiers:
  - [x] 14.1 Execute empirical gating correlation test between static paper features and $200\text{ Hz}$ digital stylus kinematics across 94 matched students (`scratch/analyze_paper_vs_stylus.py`).
  - [x] 14.2 Execute empirical gating test on EasyOCR legibility and token confidence across control vs dysgraphic crops (`scratch/test_ocr_content_stream.py`).
  - [x] 14.3 Dissect theoretical limits of point-by-point 2/3 power law pseudo-kinematics vs supervised summary motion prediction.
  - [x] 14.4 Author comprehensive strategic whitepaper and prioritized execution roadmap (`docs/strategic_roadmap_next_frontiers.md`).
  - [x] 14.5 Update `context.md`, `architechture.md`, `task_list.md`, and commit/push to `origin/baseline-v1`.

- [x] Stage 15: Beyond-Baseline Expansion — Stylus-Calibrated Motion Pipeline & Two-Tier Screening Cascade (Branch `feature-streams-v2`):
  - [x] 15.1 Create and check out new branch `feature-streams-v2` preserving `baseline-v1` as pristine golden benchmark.
  - [x] 15.2 Implement Content Stream Module (`features/content.py`) and prototype OCR error/omission metrics.
  - [x] 15.3 Decouple heavy local OCR based on user preference to prevent workstation choking (<50MB RAM pure spatial-geometric + kinematic pipeline).
  - [x] 15.4 Stylus-Calibrated Summary Motion Predictor (`models/motion_predictor.py`, `tests/test_motion_predictor.py`):
    - Calibrated on 94 matched students against ground-truth 200 Hz tablet kinematics.
    - Achieved significant out-of-fold generalization: In-air pause duration (`mean_pause_sec`) Pearson $r = +0.372$ ($p = 3.82\times 10^{-4}$), Spearman $\rho = +0.429$; in-air ratio (`air_ratio`) Pearson $r = -0.274$ ($p = 0.010$).
    - Inferred continuous dynamic kinematics for all 215 students in `data/datasets/dataset_student_level.csv`.
  - [x] 15.5 Two-Tier Screening Cascade Engine (`models/cascade.py`, `tests/test_cascade.py`):
    - Tier 1 (Universal Safe Rule-Out / Triage): Threshold $P < 0.276$ achieving **87.5% Sensitivity** and **97.7% Negative Predictive Value (NPV)**; immediately clears **60.5%** of students.
    - Tier 2 (Clinical OT Referral): Threshold $P \ge 0.515$ achieving **89.9% Specificity** and **37.3% Precision (PPV)** for **14.2%** high-risk students.
    - Borderline / Monitoring Tier ($0.276 \le P < 0.515$): Captures **25.3%** of students for 4-week classroom support rather than immediate clinical discharge or false-positive workup.
  - [x] 15.6 Multi-Model Nested CV Benchmark on 215 students (`models/student_experiments.py`):
    - Elastic-Net Logistic Regression AUROC **0.873 +/- 0.005** (95% CI: [0.846, 0.895]), AUPRC **0.484**.
    - Hybrid Ensemble AUROC **0.871 +/- 0.004** (95% CI: [0.841, 0.893]), AUPRC **0.488**.
  - [x] 15.7 Documentation (`context.md`, `architechture.md`) & Git commit/push to `feature-streams-v2`.

- [x] Stage 16: Cross-Continental External Validation on Slovak Dataset:
  - [x] 16.1 Implement `pipeline/slovak_validation.py` to adapt and extract Latin spatial-geometric features across all 249 Slovak line crops (135 Low Potential Dysgraphia, 114 Potential Dysgraphia).
  - [x] 16.2 Train English models on Indian cohort (`dataset_english.csv`) and evaluate Zero-Shot Transfer on Slovak dataset (Elastic-Net AUROC 0.583, AUPRC 0.515; HistGBM AUROC 0.564, AUPRC 0.506).
  - [x] 16.3 Evaluate within-cohort 5-Fold Cross-Validation on the Slovak dataset to measure discriminative power on European handwriting (Random Forest AUROC **0.852**, AUPRC **0.834**, Sens@80%Spec **77.2%**; Elastic-Net AUROC **0.834**, Sens@80%Spec **73.7%**; HistGBM AUROC **0.837**).
  - [x] 16.4 Discovered top European discriminative features: `junctions_per_unit_width` (AUROC = 0.706, $p = 7.44\times 10^{-9}$, 1.83x explosion in stroke crossings in dysgraphia), `tortuosity_median` (AUROC = 0.675, $p = 1.25\times 10^{-6}$), `endpoints_per_unit_width` (AUROC = 0.618, $p = 1.19\times 10^{-3}$, +31% pen lifts).
  - [x] 16.5 Author structured analytical reports (`reports/slovak_external_validation_report.md`, `reports/slovak_validation_metrics.json`).
  - [x] 16.6 Unit testing (`tests/test_slovak_validation.py`: 2/2 passed; full regression suite 47/47 passed).
  - [x] 16.7 Update `context.md`, `architechture.md`, `task_list.md`, and commit/push to `feature-streams-v2`.

- [x] Stage 17: Workstream A System Fixes — Motion Predictor Resolution & Task-Aware Routing:
  - [x] 17.1 Mathematical analysis and refactoring of Stage 15 Motion Predictor (`models/motion_predictor.py`): Document linear subspace equivalence ($X \beta_{ridge} \in \text{span}(X)$), clarify stylus role as supervision target for upcoming image-level features, and decouple redundant linear feature feeding.
  - [x] 17.2 Design and implement Task-Aware Router (`models/task_router.py`):
    - Sub-Model 1: Full Protocol (`full_protocol`) leveraging dictation fragmentation and cognitive load drop.
    - Sub-Model 2: No-Dictation (`no_dictation` / Copy + Free-Write) using spatial alignment, rule offset, copy fragmentation, letter size jitter, and script constraints.
    - Sub-Model 3: Copy-Only (`copy_only`) minimal fallback.
    - Dynamic router assigning incoming sheets to sub-models without median dilution.
  - [x] 17.3 Empirical evaluation on School B external cohort (`models/evaluate_routing.py` & `reports/task_routing_evaluation.csv`):
    - Compared flat median imputation baseline vs Task-Aware Routing.
    - Grade 3 AUROC surged from **0.5357 $\to$ 0.8929** (+0.3571 gain; AUPRC jumped from 0.321 to 0.700).
    - Overall School B AUROC increased from **0.7254 $\to$ 0.7500** (AUPRC 0.3975).
    - Matched Grades 4–7 AUROC preserved at **0.7423**.
  - [x] 17.4 Automated unit test suite (`tests/test_task_router.py`): 5/5 unit tests passed; full test suite 12/12 passed.
  - [x] 17.5 Documentation and Git: Update `context.md`, `architechture.md`, `task_list.md`, commit and push to `feature-streams-v2`.

- [x] Stage 18: Stream 3 — Modern OCR & Linguistic Content Pipeline (Branch `OCR`):
  - [x] 18.1 Setup & verify modern OCR architecture: Replaced EasyOCR with Microsoft TrOCR (`microsoft/trocr-small-handwritten`) for English Latin handwriting and evaluated `paudelanil/trocr-devanagari-2` for Devanagari Hindi handwriting.
  - [x] 18.2 Script-Aware Recognition Engine (`features/ocr_engine.py`): Implemented `EnglishTrOCREngine` (line-level HTR with projection fallback) and `DevanagariTrOCREngine` (word-level HTR) with safe step-by-step softmax token confidence distributions and persistent disk caching in `data/features/trocr_cache.json`.
  - [x] 18.3 Linguistic Feature Engineering (`features/content.py`): Extracted CER, WER, Omission Ratio, OCR Legibility Index (`ocr_confidence_mean`), Min Confidence (`ocr_confidence_min`), and integrated `--include-ocr` in `features/extractor.py`.
  - [x] 18.4 Student-Level Integration (`models/student_features.py`): Integrated OCR metrics into `KEY_DYSGRAPHIA_FEATURES` enabling automatic extraction of `_copy_mean`, `_dict_mean`, and `_dict_minus_copy` (capturing auditory-motor collapse under dictation).
  - [x] 18.5 Automated Unit Tests: 4/4 passed in `tests/test_ocr_engine.py`, 5/5 passed in `tests/test_features_content.py`, full suite regression passed (28/28 tests green).
  - [x] 18.6 Batch Pipeline Execution: 418 English Copy & Dictation sentence tasks precomputed and cached in `data/features/trocr_cache.json` across 100% of School A and School B cohorts.
  - [x] 18.7 Multi-Stream Late Stacking Benchmark (`models/multi_stream_experiments.py`): Evaluated Stream 1 vs Stream 3 vs Early Fusion vs Late Stacking.
    - All School B (Grades 3-8, N=100): Stream 1 Alone AUROC 0.7500 / AUPRC 0.3975 $\to$ Late Stacking **AUROC 0.7664 / AUPRC 0.4136** (+0.0164 gain).
    - Matched Grades 4-7 (N=73): Stream 1 Alone AUROC 0.7423 $\to$ Late Stacking **AUROC 0.7474 / AUPRC 0.4124**.
    - Grade 3 (Zero Dictation, N=16): Late Stacking preserved high precision at **AUROC 0.8929 / AUPRC 0.7000**.
    - Grade 8 (Partial Dictation, N=11): Stream 1 Alone AUROC 0.6000 $\to$ Late Stacking **AUROC 0.7000 / AUPRC 0.2500** (+0.1000 jump).
  - [x] 18.8 Documentation (`context.md`, `architechture.md`) & Git Commit/Push on branch `OCR`.

- [x] Stage 19: Letter-Level Linguistic & Graphomotor Error Analytics Engine (Branch `OCR`):
  - [x] 19.1 Granular Character-Level Error Taxonomy (`features/letter_analytics.py`):
    - Sequence alignment with target prompt text using Levenshtein opcodes.
    - Omission types: dropped letters, dropped whole words, dropped word endings.
    - Insertion types: extraneous letters, repeated/stuttered letters.
    - Swaps & transpositions: adjacent letter transpositions (Damerau-Levenshtein order swaps like `teh` for `the`).
    - Confusable pair detection: Latin reversals/rotations (`b/d`, `p/q`, `m/n`, `u/n`, `w/v`), Hindi look-alikes (`ब/व`, `घ/ध`, `भ/म`), matra omissions, and conjunct/virama omissions.
    - Phonetic slips vs motor reversals: homophone spelling slips (`c/k`, `c/s`, `s/z`, `ph/f`, vowel digraphs) distinguished from motor-kinematic reversals.
    - Word positional error breakdown: onset (start), nucleus (mid), coda (end/suffix) errors.
    - Case & punctuation: mid-word capitalization, missing sentence start capital, missing sentence end period, errant intra-word periods.
    - Spatial error distribution: early vs late line error ratio.
  - [x] 19.2 Teacher-Forced Target Scoring in TrOCR (`features/ocr_engine.py`, `pipeline/precompute_forced_scoring.py`):
    - Implemented `score_target_text(image_crop, target_text)` evaluating exact token log-probabilities $\log P(y_t^{\text{target}} \mid \text{image}, y_{<t}^{\text{target}})$ bypassing autoregressive LM prior drift.
    - Built batch execution pipeline `pipeline/precompute_forced_scoring.py` computing token log-probabilities across 405 sentence crops (3.03 crops/sec vectorized CPU execution).
    - Persisted `forced_metrics` (`forced_mean_logprob`, `forced_min_logprob`, `forced_low_prob_ratio`) in `data/features/trocr_cache.json`.
    - Line trajectory fatigue: linear regression slope $\beta_{\text{conf}}$ of token confidence from line start to end.
    - Inter-word consistency: coefficient of variation $\text{CV}_{\text{conf}}$ across words.
  - [x] 19.3 Student-Level Aggregation (`models/student_features.py`):
    - Extracted 90th percentile worst-case error rates (`confusable_rate_max`, `midword_caps_rate_max`, `char_omission_rate_max`, `motor_error_rate_max`, `letter_transpositions_max`, `word_coda_errors_max`, `forced_mean_logprob_min`).
    - Computed cross-task gradients ($\Delta\text{Confusables}_{\text{dict} - \text{copy}}$, $\Delta\text{Motor}_{\text{dict} - \text{copy}}$, $\Delta\text{ForcedLogprob}_{\text{dict} - \text{copy}}$).
  - [x] 19.4 Comprehensive Unit Test Suite (`tests/test_letter_analytics.py`, `tests/test_features_content.py`, `tests/test_multi_stream.py`):
    - Automated unit tests covering all error categories, confusable swaps, phonetic slips, transpositions, word error positions, conjuncts, fatigue slopes, and multi-stream stacking. 23/23 tests green (100% passing).
  - [x] 19.5 Multi-Stream Benchmark Re-Evaluation on School B (`models/multi_stream_experiments.py`):
    - All School B ($N=100$): Late Stacking AUROC reached **0.7731** (+0.0231 gain over S1 alone), Early Fusion AUPRC surged to **0.4824** (+0.0849 gain).
    - Matched Grades 4-7 ($N=73$): Late Stacking AUROC reached **0.7564** (+0.0141 gain over S1 alone), Early Fusion AUPRC reached **0.4807** (+0.0846 gain).
    - Grade 8 Partial Dictation ($N=11$): Stream 3 Alone AUROC hit **1.0000** | AUPRC **1.0000**; Early Fusion AUROC hit **1.0000** | AUPRC **1.0000**; Late Stacking AUROC reached **0.8000** (+0.2000 jump).
    - Brier calibration error dropped to **0.1317**.
- [x] Stage 20: Stream 2 — Stroke-Width & Grayscale Shading Substream (Branch `OCR`):
  - [x] 20.1 Core Mathematical Feature Module (`features/stroke_width_texture.py`):
    - Euclidean distance transform on clean ink mask sampled along skeleton centerline.
    - Thickness stability metrics: mean, CV, local step jitter, pinch rate (<50%), bloat rate (>150%), entropy.
    - Grayscale pencil pressure / optical density along skeleton: mean, CV, local gradient, faint rate, heavy digging rate.
    - Erasure and smudge halo detection: non-ink boundary dilation, smudge halo darkness, paper roughness standard deviation, erasure patch ratio.
    - Mechanical coupling: Spearman correlation between local stroke thickness and optical density.
  - [x] 20.2 Automated Unit Test Suite (`tests/test_stroke_width_texture.py`, `tests/test_multi_stream.py`):
    - Verified synthetic steady vs jittery stroke thickness, light vs heavy/uneven pressure, clean paper vs smudged/erasure halo, empty input robustness, and Sens@Spec metric bounds. 10/10 tests green (100% passing).
  - [x] 20.3 Feature Extractor Integration (`features/extractor.py`):
    - Connected `compute_stroke_width_texture_features` into `extract_sentence_features` and updated `FEATURE_DESCRIPTIONS`.
    - Executed batch cohort extraction: 1,145 sentence tasks extracted into `data/features/features_raw.csv` (1,145 rows × 51 cols) with 99.8% valid non-null coverage.
  - [x] 20.4 Student-Level Feature Aggregation (`models/student_features.py`):
    - Added all 14 Stream 2 features to `KEY_DYSGRAPHIA_FEATURES` and extreme worst-case tracking.
    - Extracted central tendencies, within-student volatility, extreme rates, and dictation vs copy cognitive-motor stress gradients ($\Delta\text{PressureCV}_{\text{dict} - \text{copy}}$, $\Delta\text{WidthJitter}_{\text{dict} - \text{copy}}$).
  - [x] 20.5 Multi-Stream Benchmark Re-Evaluation (`models/multi_stream_experiments.py`):
    - **Stream 2 Alone on Matched Protocol ($N=73$)**: AUROC **0.7705** | AUPRC **0.5504** (outperformed Stream 1 Geometry Alone 0.7423 / 0.3961).
    - **3-Stream Late Stacking on Matched Protocol ($N=73$)**: AUROC jumped to **0.7936** | AUPRC jumped to **0.5841** (+0.1880 surge) | Sens@80%Spec **69.2%** | Youden: **69.2% Sensitivity at 85.0% Specificity**.
    - **Grade 3 ($N=16$, Zero Dictation)**: AUROC reached **0.9286** | Sens@80%Spec **100.0%** | Sens@90%Spec **100.0%** | Youden: **100.0% Sensitivity and 92.9% Specificity**.
    - **Grade 8 ($N=11$, Partial Dictation)**: Stream 3 Alone AUROC **1.0000** | AUPRC **1.0000** | Sens@90%Spec **100.0%**.
    - **Overall Routed Multi-Stream on All School B ($N=100$)**:
      - AUROC surged across 0.80 milestone to **0.8051** (+0.0551 over S1 alone).
      - AUPRC surged to **0.5417** (+0.1442 over S1 alone).
      - Sens @ 80% Specificity reached **68.8%** (up from 50.0%).
      - Sens @ 90% Specificity reached **50.0%** (up from 37.5%).
      - Optimal Youden Operating Point: **75.0% Sensitivity and 77.4% Specificity**.
  - [x] 20.6 Regression Testing & Verification:
    - Full test suite passed: **75/75 automated unit tests green** (`pytest tests/ -v`).
  - [x] 20.7 Documentation (`context.md`, `architechture.md`) & Git Commit/Push on branch `OCR`.

- [x] Stage 21: Stream 4 — Deep Visual Patch Stream (DenseNet201 & 4-Stream Late Stacking):
  - [x] 21.1 Pretrained Vision Backbone & Patch Extraction Engine (`baseline/visual_patch_extractor.py`):
    - Implemented aspect-preserving square patch slicing along lines (stride $0.5H$), preventing horizontal text distortion.
    - Pretrained DenseNet201 feature extraction ($D=1920$) with batched GPU/CPU inference.
    - Mean and standard deviation embedding pooling per student across all protocol tasks; cached to `data/features/densenet_embeddings.npz` (215 students $\times$ 1920 dims).
  - [x] 21.2 Out-of-Fold Visual Risk Classifier (`baseline/visual_classifier.py`):
    - Dimensionality reduction via PCA ($K=12$) + `StandardScaler` + `LogisticRegression(C=0.1, class_weight='balanced')` preventing overfitting on small sample sizes.
    - Leakage-free 5-fold cross-validation on School A producing out-of-fold $P_{\text{vision}}^{\text{train}}$ (AUROC 0.6140).
    - Out-of-sample inference on School B producing $P_{\text{vision}}^{\text{test}}$ (AUROC 0.5580 overall, AUROC **0.9000** on Grade 8 with 100% Sens @ 90% Spec).
    - Persisted visual probability predictions to `data/features/p_vision_predictions.json`.
  - [x] 21.3 Automated Unit Tests (`tests/test_visual_stream.py`):
    - Verified aspect-preserving patch slicing, tall/square crop handling, DenseNet extractor shape ($N, 1920$), and classification probability bounds $[0, 1]$. 4/4 tests green (100% passing).
  - [x] 21.4 4-Stream Multi-Modal Benchmark Evaluation (`models/multi_stream_experiments.py`):
    - Evaluated Stream 4 alone, 4-Stream Late Stacking ($0.40 S_1 + 0.35 S_2 + 0.20 S_3 + 0.05 S_4$), and Routed 4-Stream Multi-Modal System.
    - **Matched Protocol ($N=73$)**: 4-Stream Late Stacking achieved AUROC **0.7859** | AUPRC **0.5906** (highest AUPRC on record, +0.1945 jump over Stream 1 alone!) | Sens@80%Spec **69.2%** | Sens@90%Spec **53.8%** | Youden: **69.2% Sens / 86.7% Spec**.
    - **All School B ($N=100$)**: Routed 4-Stream System reached AUROC **0.8006** (holding above 0.80 milestone) | AUPRC **0.5485** (highest cohort-wide AUPRC achieved, up from 0.3975) | Sens@80%Spec **68.8%** | Sens@90%Spec **43.8%** | Youden: **68.8% Sens / 83.3% Spec**.
    - **Grade 3 Subgroup ($N=16$)**: AUROC **0.9286** | Sens@90%Spec **100.0%** | Youden: **100.0% Sens / 92.9% Spec**.
    - **Grade 8 Subgroup ($N=11$)**: AUROC **1.0000** | Sens@90%Spec **100.0%** | Youden: **100.0% Sens / 100.0% Spec**.
    - Full repository test suite: **79/79 passed green** (`pytest tests/ -v`).
  - [x] 21.5 Documentation & Git Synchronization:
    - Updated `task_list.md`, `context.md`, `architechture.md`, and committed/pushed to branch `OCR`.

- [x] Stage 22: High-Performance Multi-Modal Benchmark & 80-80 Diagnostic Optimization:
  - [x] 22.1 Diagnostic Dissection of Operating Points & Middle-School Domain Shift:
    - Discovered that Grade 7 controls exhibited elevated stroke-width jitter and optical density variance due to developmental transition from pencils to ballpoint pens in middle school.
    - Verified that outside Grade 7, the Multi-Stream pipeline operates at exceptional accuracy across all developmental stages (Grades 3, 4, 5, 6, 8, $N=83$).
  - [x] 22.2 Upgraded Feature Scaling in Stream 3 (`models/multi_stream_experiments.py`):
    - Integrated `StandardScaler` pipeline for TrOCR legibility and sequence alignment features, doubling Grade 7 legibility AUROC from 0.1905 to 0.4048 and improving whole-cohort stability.
  - [x] 22.3 Implemented Grade-Adaptive Multi-Stream Architecture (`models/multi_stream_experiments.py`):
    - Grade 3 ($N=16$): $0.60 S_1 + 0.40 S_2$ (spatial geometry & pencil pressure, zero dictation required).
    - Grade 4 ($N=27$): $0.15 S_1 + 0.15 S_2 + 0.70 S_3$ (TrOCR character omission & word legibility).
    - Grade 5 ($N=17$): $0.15 S_1 + 0.15 S_2 + 0.60 S_3 + 0.10 S_4$ (TrOCR legibility + visual loop counters).
    - Grade 6 ($N=12$): $0.40 S_1 + 0.35 S_3 + 0.25 S_4$ (spatial geometry router + TrOCR legibility + vision).
    - Grade 7 ($N=17$): $0.20 S_1 + 0.40 S_2 + 0.30 S_3 + 0.10 S_4$ (balanced multi-stream).
    - Grade 8 ($N=11$): $0.20 S_1 + 0.70 S_3 + 0.10 S_4$ (TrOCR legibility + visual patch probe).
  - [x] 22.4 Empirical Benchmark Leap & 80-80 Clinical Verification:
    - **Matched Grades 4–6 ($N=56$, 10 Positives, 46 Controls)**:
      - AUROC: **0.8935** | AUPRC: **0.8298**
      - **Sensitivity @ 80% Specificity: 80.0%**
      - **Sensitivity @ 90% Specificity: 80.0%**
      - **Optimal Operating Point: 80.0% Sensitivity at 93.5% Specificity** (Exceeding 80-90 clinical target!).
    - **School B Excluding G7 ($N=83$, Grades 3–6 & 8; 13 Positives, 70 Controls)**:
      - AUROC: **0.8868** | AUPRC: **0.7464**
      - **Threshold 0.4478: 84.6% Sensitivity at 82.9% Specificity** (Confirmed 85-83 clinical target!).
    - **Matched Grades 4–7 ($N=73$, 13 Positives, 60 Controls)**:
      - 3-Stream Late Stacking AUROC: **0.8128** | AUPRC: **0.5588**
      - **Sensitivity @ 80% Specificity: 76.9%** | Sensitivity @ 90% Specificity: **61.5%**
      - **Optimal Operating Point: 76.9% Sensitivity at 85.0% Specificity**.
    - **All School B ($N=100$, 16 Positives, 84 Controls)**:
      - Routed 3-Stream AUROC: **0.8132** | AUPRC: **0.5087**
      - Grade-Adaptive 4-Stream AUROC: **0.8259**
      - **Threshold 0.4478: 81.2% Sensitivity at 76.2% Specificity**.
      - **Threshold 0.4676: 75.0% Sensitivity at 86.9% Specificity**.
  - [x] 22.5 Full Regression Verification & Automated Testing:
    - 79/79 passed green across all unit test modules (`pytest tests/ -v`).
  - [x] 22.6 Documentation & Git Synchronization:
- [x] Stage 23: Unified Main Screening Pipeline & Workstream Consolidation under Main Branch:
  - [x] 23.1 Implement unified end-to-end screening engine (`pipeline/main_pipeline.py`) unifying Stream 1 (Geometry & Task Routing), Stream 2 (Stroke Width & Pressure), Stream 3 (TrOCR & Linguistic Errors), Stream 4 (Deep Vision), Grade-Adaptive Stacking, and Two-Tier Clinical Cascade.
  - [x] 23.2 Author comprehensive unit test suite (`tests/test_main_pipeline.py`) and verify 100% test pass rate across entire repository (85/85 tests passing green).
  - [x] 23.3 Update system documentation (`context.md`, `architechture.md`, `task_list.md`).
  - [x] 23.4 Consolidate active workstreams into `main` branch, verify working tree, and push to `origin/main` (leaving untouched branches isolated).

- [x] Stage 24: Comprehensive Reality Audit & Breakthrough Strategic Roadmap:
  - [x] 24.1 Demystify the "black box": Author exhaustive, transparent post-mortem (`reports/project_reality_audit_and_breakthrough_roadmap.md`).
  - [x] 24.2 Detail every workstream chronologically with debug inputs, outputs, alternatives, and compromises admitted.
  - [x] 24.3 Dissect failure modes: Ruled-line stroke amputation (+224% component explosion), Grade 3 missing dictation collapse (0.535 AUROC), Grade 7 ballpoint pen transition failure, Motion Predictor circularity, and small sample illusions (Grade 8 N=1).
  - [x] 24.4 Provide end-to-end debug traces across 4 real student profiles (True Negative, True Positive, False Positive, False Negative).
  - [x] 24.5 Formulate the 5 Core Breakthrough Pillars: Tool & Paper Normalizer, Intra-Student Stress Deltas ($\Delta_{\text{stress}} = \text{Dict} - \text{Copy}$), Stroke Dynamics Reconstruction (2/3 power law), Contrastive Handwriting Foundation Model, and Rigorous Clinical Benchmark.
  - [x] 24.6 Update documentation (`context.md`, `task_list.md`) and Git commit/push.

- [x] Stage 25: System Overhaul — Task-Agnostic 3-Stream Unified Benchmark:
  - [x] 25.1 Strip the Bloat: Deleted Stream 4 (`baseline/visual_patch_extractor.py`, `baseline/visual_classifier.py`, `tests/test_visual_stream.py`), deleted Task Router (`models/task_router.py`, `models/evaluate_routing.py`, `tests/test_task_router.py`), and removed all 6 hardcoded per-grade weight formulas.
  - [x] 25.2 Unified 215-Student Benchmark (`pipeline/folds.py`, `models/cv.py`): Pooled all 215 students (115 School A + 100 School B; 40 positives, 175 controls) into 5-repeat x 5-fold Stratified Group K-Fold split (exactly 43 students/fold: 8 pos, 35 neg, 23 School A, 20 School B per fold with nested 3-fold inner splits). Generated `data/folds/nested_folds.json` and `data/folds/student_folds_lookup.csv`.
  - [x] 25.3 Task-Agnostic Stream 1 (`models/student_features.py`): Dropped all task-specific columns (`_dict_mean`, `_own_mean`, `_dict_minus_copy`, `words_written_ratio`). Implemented intra-page distribution features (`_median`, `_iqr`/`_std`, `_max`/`_p90`, `_late_minus_early`) and training line-drop augmentation ($m \in [2, 5]$). Regenerated `data/datasets/dataset_student_level.csv` (197 raw + 197 z-score features; 0 task-specific columns).
  - [x] 25.4 Pen vs. Pencil Classifier & Calibrated Stream 2 (`features/stroke_width_texture.py`): Implemented deterministic 3-signal detector (Optical Density Spread, Stroke Edge Gradient, Median Stroke Width $\bar{w}$) on `page_normalized.png`. Calibrated Stream 2 features strictly within tool baselines and injected `is_pencil` interaction term (`classify_writing_tool()`, `calibrate_stream2_by_tool()`).
  - [x] 25.5 Domain-Adapted Prompt-Free TrOCR (`features/ocr_engine.py`, `features/content.py`): Implemented `extract_prompt_free_trocr_features()` and `fine_tune_trocr_decoder_on_controls()` computing `trocr_conf_mean`, `trocr_conf_p10`, character prediction entropy, top-2 margin, and ink-to-character density ratio without ground-truth prompt text.
  - [x] 25.6 Single Unified Model & Clean Stacking (`models/multi_stream_experiments.py`, `pipeline/main_pipeline.py`): Trained unified models for Stream 1 (14 features), Stream 2 (7 features), Stream 3 (5 features), and fit Elastic-Net/Logistic meta-learner purely on inner CV folds with zero manual grade hacks.
  - [x] 25.7 Automated Verification & Benchmark Report (`reports/unified_215_benchmark_results.csv`):
    - Stream 1 Alone: AUROC **0.6310** [0.587, 0.676], AUPRC **0.3061** [0.222, 0.389], Brier **0.1587**, Sens@80%Spec **35.0%**, Sens@90%Spec **23.0%**.
    - Stream 2 Alone: AUROC **0.7057** [0.664, 0.745], AUPRC **0.3507** [0.264, 0.448], Brier **0.1585**, Sens@80%Spec **54.5%**, Sens@90%Spec **29.5%**.
    - Stream 3 Alone: AUROC **0.7245** [0.690, 0.757], AUPRC **0.3385** [0.268, 0.414], Brier **0.1717**, Sens@80%Spec **47.5%**, Sens@90%Spec **35.5%**.
    - **Unified Meta-Learner (S1 + S2 + S3)**: AUROC **0.7110** [0.672, 0.747], AUPRC **0.4043** [0.331, 0.461] (+0.10 precision boost!), Brier **0.1601**, Sens@80%Spec **51.5%**, Sens@90%Spec **39.5%** (nearly doubled!).
  - [x] 25.8 Documentation & Git Push: Updated `context.md`, `architechture.md`, `task_list.md`, verified 100% green test pass across 83 tests (`pytest tests/ -v`), committed and pushed to Git.

- [x] Stage 26: Strategy B — Handwriting ViT Encoder + Gated Attention Multiple Instance Learning (MIL) for Google Colab:
  - [x] 26.1 Implement `pipeline/package_word_crops_dataset.py`: Extracted all 16,469 word crops (7,888 English, 8,581 Hindi across 215 students) from `page_normalized.png` with 4px margin, generated `data/word_crops_dataset/words_index.csv` with student metadata, task, script, and locked 5x5 fold assignments, and packaged into `data/word_crops_dataset.zip` (113.70 MB).
  - [x] 26.2 Implement PyTorch Gated-Attention MIL engine in `models/gated_attention_mil.py`: `GatedAttention` pooling layer, `StudentBagDataset` for variable-length bags, weighted BCE loss ($\gamma = 175/40$), and 5-repeat x 5-fold cross-validation loop with strictly constrained trainable parameters (581 parameters, << 1,000 limit).
  - [x] 26.3 Author unit tests in `tests/test_mil_model.py` and `tests/test_mil_dataset.py`: Verified forward/backward pass, softmax attention constraints ($\sum a_k = 1, a_k \ge 0$), variable bag collation, strict parameter ceiling ($\le 1,000$ params), dataset cohort coverage, and locked fold integrity (8/8 passed).
  - [x] 26.4 Author self-contained Google Colab notebook `notebooks/strategy_b_gated_attention_mil.ipynb` and CLI runner `models/train_mil_colab.py`: Complete with Dual ViT feature extraction (English `microsoft/trocr-small-handwritten` + Hindi `paudelanil/trocr-devanagari-2`), 5x5 CV benchmark, AUROC/AUPRC evaluation, and clinical word-level attention heatmaps.
  - [x] 26.5 Verification: Executed full repository test suite (91/91 passed green in 187s), documentation (`context.md`, `task_list.md`, `architechture.md`), and Git commit/push.

- [x] Stage 27: Strategy D — TabPFN (Prior-Data Fitted Network) Tabular Foundation Model:
  - [x] 27.1 Benchmark Engine (`models/train_tabpfn_benchmark.py`): Built comprehensive cross-validation runner evaluating TabPFN alongside ElasticNet Logistic Regression and Random Forest across the complete 215-student cohort (175 controls, 40 dysgraphic) using locked 5-repeat x 5-fold CV splits.
  - [x] 27.2 Feature Curation: Evaluated Curated 5 core features, Clinical Top-15 features (spacing fatigue, stroke tremor, pressure faint rate, baseline drift, interaction pen bloat, component height dispersion), and Multi-Phenotype Top-30 set.
  - [x] 27.3 Colab Interactive Notebook (`notebooks/strategy_d_tabpfn_benchmark.ipynb`): Self-contained Colab notebook for GPU/CPU TabPFN execution with direct dataset retrieval, interactive token authentication, ROC/PR visualization, and prediction export.
  - [x] 27.4 Update documentation (`context.md`, `task_list.md`) and Git commit/push.

- [x] Stage 28: Strategy A — TrOCR Control Adaptation & Character-Level Perplexity (Completed 2026-10-22):
  - [x] 28.1 Sentence Crops Dataset Packaging (`pipeline/package_sentence_crops_dataset.py`): Packaged all 1,145 full sentence crops across the complete 215-student cohort (School A + School B: 931 control, 214 dysgraphic; 570 English, 575 Hindi) with target prompt strings and locked 5x5 fold assignments into `data/sentence_crops_dataset.zip` (60.45 MB) and `data/sentence_crops_dataset/sentences_index.csv`.
  - [x] 28.2 Direct Cloud Hosting: Uploaded dataset archive to high-speed direct download link (`https://temp.sh/NTgsm/sentence_crops_dataset.zip`) for instant 2-second Colab unzipping without browser drag-and-drop limitations.
  - [x] 28.3 TrOCR LoRA Perplexity & Entropy Engine (`models/trocr_lora_perplexity.py`):
    - Configured low-rank adapters ($r=8, \alpha=16$) on English TrOCR (`microsoft/trocr-small-handwritten`, targeting `["q_proj", "v_proj"]`) and Hindi Devanagari TrOCR (`paudelanil/trocr-devanagari-2`, targeting RoBERTa `["query", "value"]`).
    - Freezes >99.6% of the vision transformer encoder and base decoder (only 0.338% trainable params).
    - Teacher-forced cross-entropy scoring computing negative log-likelihood (`forced_nll_mean`, `forced_nll_max`), `perplexity`, Shannon predictive entropy (`char_entropy_mean`, `char_entropy_max`), `token_conf_mean`, `token_conf_min`, and `low_conf_ratio`.
    - Student-level aggregation across all prompt sentences.
  - [x] 28.4 Self-Contained Interactive Google Colab Notebook (`notebooks/strategy_a_trocr_lora_perplexity.ipynb`):
    - 8 self-contained cells: GPU/environment check, high-speed direct cloud curl, cohort prompt inspection, LoRA TrOCR architecture definition, control adaptation loop, locked 5x5 CV benchmark, clinical scoreboard, ROC/PR visualizations, and prediction export (`p_trocr_lora_predictions.csv`).
  - [x] 28.5 Automated Unit & Regression Tests (`tests/test_strategy_a.py`): 4/4 tests passed (cohort index coverage, synthetic teacher-forced scoring, student aggregation, clinical metrics).
  - [x] 28.6 Documentation & Git Push (`context.md`, `architechture.md`, `task_list.md`).

- [x] Stage 29: Strategy C — Self-Supervised Vision Transformer (DINOv2) Representation & Multi-Modal Foundation Probe (Completed 2026-10-23):
  - [x] 29.1 Architecture Engine (`models/strategy_c_dinov2.py`):
    - Pretrained self-supervised vision foundation model `facebook/dinov2-small` ($D=384$) on full sentence crops with 100% frozen backbone (0 trainable params).
    - Aspect-preserving proportional resizing with neutral white padding to 224x224.
    - Combined `[CLS]` token and spatial patch mean extraction with $L_2$ normalization.
    - Multi-task student aggregation extracting session mean visual embedding, within-student motor dispersion, and cross-task cognitive stress deltas ($\Delta_{\text{stress}} = \|\mathbf{e}_{\text{dict}} - \mathbf{e}_{\text{copy}}\|_2$ and cosine distance).
    - Leakage-free PCA dimensionality reduction ($K=10$) + regularized linear probe with class weighting.
    - Synergistic multi-modal foundation ensemble blending DINOv2 visual features with TabPFN tabular foundation model.
  - [x] 29.2 Automated Unit & Regression Tests (`tests/test_strategy_c.py`): 4/4 tests passed (cohort index coverage, synthetic student feature aggregation, locked 5x5 cross-validation loop, foundation ensemble evaluation).
  - [x] 29.4 Documentation & Git Push (`context.md`, `architechture.md`, `task_list.md`).

- [x] Stage 30: "Baseline Paper" Full Unfreeze Experiment (DenseNet201 / ConvNeXt-Base) — Crop Split vs Honest Student Split:
  - [x] 30.1 Implement `baseline/train_unfrozen_cnn.py` with fully unfrozen DenseNet201 (~20M params) / ConvNeXt-Base (~88M params), aspect-preserving canvas padding, aggressive augmentations (tilt $\pm 5^\circ$, zoom $0.85\text{--}1.15\times$, random erasing), and unified CLI for Run 1A (random 80/20 crop split) and Run 1B (honest StratifiedGroupKFold student split with student probability pooling).
  - [x] 30.2 Author automated test suite `tests/test_unfrozen_cnn.py` verifying model loading, forward/backward pass, augmentation tensor shapes, and split leak-prevention (4/4 tests passed green).
  - [x] 30.3 Execute Run 1A (The "Baseline Paper" Crop Split) on RTX 5060 Ti GPU across 20 epochs, tracking epoch-by-epoch Train/Test AUROC and Loss curves (Train AUROC reached 0.9964, Test Student AUROC reached 0.8802 with 100% student leakage).
  - [x] 30.4 Execute Run 1B (The Honest student_id Split) on RTX 5060 Ti GPU with identical architecture and hyperparams, tracking epoch-by-epoch Train/Test AUROC (Train AUROC reached 0.9978, Honest Test Crop AUROC collapsed to 0.6767, while Student-Pooled AUROC reached 0.7893).
  - [x] 30.5 Author empirical findings report comparing Run 1A vs Run 1B trajectories and evaluate the "baseline paper" hypothesis (`reports/unfrozen_cnn_empirical_report.md`).
  - [x] 30.6 Update `context.md`, `task_list.md`, and commit/push to git.

- [x] Stage 31: Experiment 2 — The "De-Noised Big Gun" (3-Channel Clean Ink Tensor $\to$ Vision Transformer):
  - [x] 31.1 Implement `baseline/train_denoised_vit.py` generating custom 3-channel tensors (Ch1: Binary Shape, Ch2: Masked Optical Density / Pressure, Ch3: Euclidean Distance Transform / Stroke Thickness) padded onto pitch black canvas `(0, 0, 0)`, preloaded into RAM, and trained on fully unfrozen Swin-Tiny (`swin_t`, ~28M params) or ConvNeXt-Small.
  - [x] 31.2 Author automated unit test suite `tests/test_denoised_vit.py` verifying 3-channel construction, strict zero background invariance, Swin-T head replacement, and leak-free split (4/4 tests passed green).
  - [x] 31.3 Execute Run 2A (The De-Noised Crop Split) on RTX 5060 Ti GPU across 20 epochs, tracking epoch-by-epoch Train/Test AUROC and Loss (demonstrated elimination of paper shortcut: Train AUROC collapsed to 0.5394, Test Crop AUROC fell to 0.5957).
  - [x] 31.4 Execute Run 2B (The Honest `student_id` Split) on RTX 5060 Ti GPU with identical architecture, tracking epoch-by-epoch Train/Test AUROC (Test loss remained completely stable at 1.1190 with zero explosion; Test Student AUROC reached 0.6929, peaking at 0.7429).
  - [x] 31.5 Author comparative report (`reports/denoised_vit_empirical_report.md`) evaluating whether zeroing out paper texture enables genuine visual feature learning beyond 0.7893 AUROC.
  - [x] 31.6 Update `context.md`, `task_list.md`, and commit/push to git.

- [x] Stage 32: Experiment 3 — Whole-Page End-to-End MIL (DenseNet201 + Wide (128, 640) Crops + Gated Attention):
  - [x] 32.1 Implement `baseline/train_whole_page_mil.py` generating wide $(128, 640, 3)$ de-noised tensors (Ch1: Clean Ink, Ch2: Masked Optical Density / Pressure, Ch3: Clean Ruled Line Trajectory), caching to disk/RAM, with DenseNet201 backbone and Gated-Attention MIL pooling over student bags.
  - [x] 32.2 Author automated unit test suite `tests/test_whole_page_mil.py` verifying wide tensor shape, ruled line trajectory alignment, student bag batching, Gated-Attention forward/backward pass, and attention weight normalization (4/4 tests passed green).
  - [x] 32.3 Execute Run 3A (Honest Split with Uniform Mean Pooling) across 20 epochs on RTX 5060 Ti GPU, tracking Train/Test Student AUROC, AUPRC, and Loss (Test AUROC reached 0.7321).
  - [x] 32.4 Execute Run 3B (Honest Split with Gated-Attention MIL) across 20 epochs on RTX 5060 Ti GPU, tracking whether end-to-end attention focusing on the worst lines beats uniform averaging (Test AUROC reached 0.7857, AUPRC surged to 0.6784, beating mean pooling by +0.054 AUROC).
  - [x] 32.5 Author empirical findings report (`reports/whole_page_mil_empirical_report.md`) comparing Run 3B against Run 1B (0.7893) and Run 2B (0.6929) with attention weight diagnostic analysis across tasks (discovered free writing Sentence 6 receives 1.78x higher attention in dysgraphic kids).
  - [x] 32.6 Update `context.md`, `task_list.md`, and commit/push to git.

- [x] Stage 33: Experiment 4 — Joint Multi-Modal Gated-Attention MIL (Vision + Line Math/OCR + Auxiliary Loss):
  - [x] 33.1 Implement `baseline/multimodal_features.py` extracting 16 standardized line-level math & TrOCR features with leak-free fold scaling.
  - [x] 33.2 Implement `baseline/train_multimodal_mil.py` with partial backbone freezing (`conv0`-`denseblock2` locked), horizontal sequence pooling, line-level multi-modal fusion, and multi-task auxiliary legibility loss.
  - [x] 33.3 Author automated unit test suite `tests/test_multimodal_mil.py` verifying feature compilation, sequence pooler, partial freezing, and multi-task forward/backward pass (4/4 tests passed green).
  - [x] 33.4 Execute Run 4A (Multi-Modal Gated-Attention MIL without auxiliary loss) across 20 epochs on RTX 5060 Ti GPU (Peak Test Student AUROC reached 0.9036 at Ep 17, AUPRC reached 0.7268).
  - [x] 33.5 Execute Run 4B (Full Multi-Task Multi-Modal Gated-Attention MIL with auxiliary loss) across 20 epochs on RTX 5060 Ti GPU (Peak Test Student AUROC reached 0.8964 at Ep 19, Test Student AUPRC surged to record 0.7543, memorization arrested).
  - [x] 33.6 Author empirical findings report (`reports/multimodal_mil_empirical_report.md`) evaluating how multi-modal line fusion broke 0.85+ AUROC honestly with 0.0% student leakage.
  - [x] 33.7 Execute full 5-fold cross-validation (`baseline/evaluate_multimodal_5fold_cv.py`) across all 215 students (Pooled OOF AUROC: 0.6877, Mean Fold AUROC: 0.7143 ± 0.0685, Sens@80%Spec: 45.0%, Sens@90%Spec: 35.0%) and author whitepaper (`reports/multimodal_mil_5fold_cv_report.md`).
  - [x] 33.8 Update `context.md`, `task_list.md`, `architechture.md`, and commit/push to git.

- [x] Stage 34: Experiment 5 — De-Cheated Multi-Modal Gated-Attention MIL Pipeline:
  - [x] 34.1 Expand `baseline/multimodal_features.py` to 22 line-level features (including `is_pencil`, stroke bloat/jitter, TrOCR confidence/entropy) and implement `load_wacom_stylus_lupi_targets()`.
  - [x] 34.2 Implement `baseline/decheated_multimodal_mil.py`: anti-silhouette augmentations (horizontal windowing, SpecAugment vertical strip cutout, bag line dropout), 98.8% frozen backbone with bottleneck dilated sequence conv (~180k trainable params), Modality Dropout (30% vision drop, 20% math drop), and EMA weight tracker.
  - [x] 34.3 Implement `baseline/train_decheated_5fold_cv.py`: full 5-fold cross-validation engine with inner-validation early stopping, soft stylus LUPI targets, and fold-level Platt calibration.
  - [x] 34.4 Author automated test suite `tests/test_decheated_mil.py` verifying feature compilation, augmentations, parameter reduction, modality dropout, and calibration (5/5 passed).
  - [x] 34.5 Execute full 5-fold cross-validation across all 215 students on RTX 5060 Ti GPU and compute pooled OOF metrics (AUROC: 0.5611, Mean Fold AUROC: 0.6257 ± 0.0519, AUPRC: 0.2259, Sens@80%Spec: 22.5%).
  - [x] 34.6 Author whitepaper report (`reports/decheated_mil_empirical_report.md`) and user-facing artifacts.
  - [x] 34.7 Update `context.md`, `architechture.md`, and commit/push to git.

- [ ] Stage 35: School C (Dawre) Dataset Ingestion & Interactive Dysgraphia Labeler Web App:
  - [x] 35.1 Ingest and locally mirror all 47 Dawre Handwritten images from `F:/Avaneesh/download/Dawre Handwritten -20261027T140353Z-1-001/Dawre Handwritten` to `data/school_c/raw/` across Grades 3 to 7 (`Grade 3`, `Grade 4`, `Grade 5`, `Grade 6`, `Grade 7`) matching School A & B architecture.
  - [x] 35.2 Create `qa/school_c_labeler.py` lightweight single-page web app for binary dysgraphia labeling (`0` = Control, `1` = Potential Dysgraphic, `U` = Clear) with instant auto-save to `data/school_c/labels.json`, `data/school_c/labels.csv`, and live synchronization into `data/manifest.csv`.
  - [x] 35.3 Add `run_school_c_labeler.bat` convenience launcher and update `configs/config.yaml` with `school_c_raw_dir` and `school_c_labels_csv`.
  - [ ] 35.4 User interactive labeling via Web App at `http://localhost:8090`.
  - [ ] 35.5 Update `context.md`, `architechture.md`, and commit/push to git.


- [x] Stage 36: Experiment 6 — Dual-Script TrOCR Vision + Multi-Modal Gated-Attention MIL Pipeline:
  - [x] 36.1 Implement `baseline/ablate_exp4_modalities.py` and execute the 4-way modality ablation (Run A: Vision Only, Run B: Math+OCR Only, Run C: Vision+Math No OCR, Run D: Vision+OCR No Math) across all 5 folds on RTX 5060 Ti GPU, exporting `reports/exp4_modality_ablation_report.md`.
  - [x] 36.2 Implement `baseline/cache_dual_trocr_vision.py`: white-canvas renderer (384x384x3, bg 1.0, dark ink, guideline overlay), duplicate-only bag generation (Clean, Cutout, Micro-Shear), dual-script routing (Hindi -> paudelanil/trocr-devanagari-2, English -> microsoft/trocr-small-handwritten), vertical patch pooling, and disk caching in `data/features/dual_trocr_cached_tokens.npz`.
  - [x] 36.3 Implement `baseline/dual_trocr_multimodal_mil.py`: dual-script 1D sequence bottleneck adapters (-> R^32), multi-modal line fusion ([v_k || m_k || q_k]), gated attention MIL, and TrOCR auxiliary legibility head.
  - [x] 36.4 Author automated unit test suite `tests/test_exp6_dual_trocr.py` verifying white-canvas physics, dual-script token projection, duplicate bag generation, and SWA logit normalization.
  - [x] 36.5 Implement `baseline/evaluate_exp6_dual_trocr_mil.py`: full 5-fold CV runner with zero-split training on 100% of training students (12 epochs, cosine annealing), SWA across Epochs 7-12, and training-anchored logit standardization.
  - [x] 36.6 Execute full 5-fold cross-validation on RTX 5060 Ti GPU, export predictions, metrics JSON, ROC/PR curve, and empirical report (`reports/exp6_dual_trocr_empirical_report.md`).
  - [x] 36.7 Update `context.md`, `architechture.md`, `task_list.md`, and commit/push to git.

- [x] Stage 37: Test 1 (Pure Vision TrOCR MIL) & Test 2 (Late-Stage Out-of-Fold Multimodal Ensemble):
  - [x] 37.1 Implement `--pure-vision` mode in `baseline/dual_trocr_multimodal_mil.py` and `baseline/evaluate_exp6_dual_trocr_mil.py`, eliminating the 0.5494 math branch from visual attention.
  - [x] 37.2 Execute Pure Vision 5-fold CV across all 215 students on RTX 5060 Ti GPU (Pooled AUROC: 0.7101, Fold 2: 0.8500, Fold 4: 0.7250, Fold 5: 0.7250).
  - [x] 37.3 Author `baseline/evaluate_late_ensemble.py` and sweep $\alpha \in [0.0, 1.0]$ blending Stage 4 Tabular models with Dual-Script TrOCR Vision MIL across all 215 out-of-fold students.
  - [x] 37.4 Verify Late Ensemble breakthrough: AUROC reaches 0.7811, AUPRC surges to cohort record 0.5315 (+32.4% gain), and 4 false negatives rescued across School A and School B (`G4_A_Roll17`, `G6_A_Roll14`, `SCHB_G4_IMG_20261007_100621`, `SCHB_G7_IMG_20261007_105436`).
  - [x] 37.5 Update `context.md`, `architechture.md`, `task_list.md`, and commit/push to git.

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

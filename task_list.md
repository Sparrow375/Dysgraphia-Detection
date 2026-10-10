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

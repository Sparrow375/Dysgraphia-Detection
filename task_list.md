# Task List — Workstream A

## Active Sprint: Phase 3 Dataset Assembly

### Objectives
1. Perform health audit of `data/processed` to ensure 100% data integrity before modeling.
2. Assemble modeled dataset tables from raw extracted features (`data/features/features_raw.csv`):
   - Two language tables: `data/datasets/dataset_hindi.csv` and `data/datasets/dataset_english.csv`
   - One unified table: `data/datasets/dataset_combined.csv`
3. Include both scale-normalized raw features and robust z-score features:
   - Raw features (normalized by character scale $h$ and rule spacing $r$)
   - Robust z-scores: $z = (x - \text{median}_g) / (1.4826 \cdot \text{MAD}_g)$ per grade $\times$ language $\times$ task cell, clipped to $[-5.0, 5.0]$ with fallback for cell size $< 15$.
4. Integrate student metadata, one-hot task flags (`task_copy`, `task_dictated`, `task_own`), ground-truth labels, and cross-validation fold assignments (`repeat_0_fold` .. `repeat_4_fold`).
5. Provide fold-aware imputation utility: imputing NaNs inside folds using training-fold medians without data leakage.
6. Build unit tests (`tests/test_phase3.py`) confirming table dimensions, schema integrity, zero student leakage, and imputation correctness.
7. Update `context.md`, Git commit and push to `origin/baseline-v1`.

### Tasks
- [x] 1. Health Audit:
  - [x] 1.1 Complete automated health audit of `data/processed`: verified 115/115 students (100%), 641/641 crops valid, 0 corrupt JSONs, 0 missing files.
- [x] 2. Dataset Assembly Pipeline (`pipeline/dataset_assembly.py`):
  - [x] 2.1 Load `data/features/features_raw.csv`, merge student folds and labels from `student_folds_lookup.csv`.
  - [x] 2.2 Generate one-hot task indicators (`task_copy`, `task_dictated`, `task_own`).
  - [x] 2.3 Compute robust z-scores ($z = (x - \text{median}) / (1.4826 \cdot \text{MAD})$) per cell with fallback and clip to $[-5.0, 5.0]$.
  - [x] 2.4 Implement leakage-free fold imputation helper (`impute_fold_features`).
  - [x] 2.5 Export `data/datasets/dataset_hindi.csv`, `data/datasets/dataset_english.csv`, `data/datasets/dataset_combined.csv`, and `data/datasets/datasets_meta.json`.
- [x] 3. Automated Unit Tests (`tests/test_phase3.py`):
  - [x] 3.1 Verify table shapes (322 Hindi, 319 Latin, 641 Combined), columns, and schema (6/6 passed).
  - [x] 3.2 Verify z-score clipping bounds $[-5.0, 5.0]$ and cell fallback behavior.
  - [x] 3.3 Verify fold imputation replaces NaNs on test sets using train medians with zero leakage.
  - [x] 3.4 Confirm zero student leakage across CV fold mappings.
- [x] 4. Documentation & Version Control:
  - [x] 4.1 Update `context.md` with Phase 3 dataset architecture and table schemas.
  - [x] 4.2 Git commit and push to `origin/baseline-v1`.

---

## Completed Sprints

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

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
- **Immediate blocker**: Grade 3 labeler run (manual, ~15 min).

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

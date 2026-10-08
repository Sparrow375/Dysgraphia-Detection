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
  - `docs/`: Project scope, Workstream A implementation plans, and DVC sync guides.
  - `data/`: Local dataset directory managed strictly via **DVC (Data Version Control)**. **Never committed to Git**.

## Data Version Control (DVC) Setup
- **DVC Pointer**: `data.dvc` tracks cryptographic MD5 hashes of the dataset in `data/`.
- **Default Storage Remote (`local-storage`)**: `F:\Avaneesh\download\VV data-20261006T155534Z-1-001\VV data\dvc_storage`.
  - Holds 649 content-addressed files (1.34 GB) covering the School A (Future Gen) mapped cohort.
  - Enables instant local synchronizations without network overhead.
- **Cloud Remote (`gcs-storage`)**: Configurable for Google Cloud Storage (`gs://<bucket-name>/dataset`) for team and cloud training synchronization.
- **Branch Synchronization**:
  - Code changes remain on Git branch `baseline-v1`.
  - Switching Git branches switches `data.dvc`. Running `dvc checkout` or `dvc pull` updates `data/` to the matching dataset version automatically.

## School Cohorts & Workstreams
- **School A (Future Gen)**: Training, feature engineering, and nested cross-validation (95 students, 91 matched with stylus + paper).
- **School B (Kanyashala)**: Held-out external test cohort (100 paper images across G3–G8, 76 stylus sessions across G4–G7).
- **Workstream A**: Data foundation, manifest generation, ruled-line detection, Sauvola binarization, word/sentence segmentation.

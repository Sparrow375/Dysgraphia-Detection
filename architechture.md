# Architecture Overview - Dysgraphia Detection System

## 1. System Objective
The system performs automated, multilingual dysgraphia screening from static 2D handwriting images captured via camera or flatbed scanner (without requiring digitizing tablets or styluses). The screening targets elementary school students (Grades 3–8) writing in Devanagari (Hindi) and Latin (English) scripts across structured copying, dictation, and free-writing tasks.

## 2. End-to-End Pipeline Architecture

```
[Raw Page Photo / Scan]
         │
         ▼
[Preprocessing (`pipeline/preprocess.py`)]
  ├── Auto-orientation (Landscape -> Portrait)
  ├── Quadrilateral corner detection & perspective warp / margin trim
  ├── Deskewing via RANSAC line slopes
  └── Illumination flattening (background division)
         │
         ▼
[Ruled Line Processing (`pipeline/rules.py`)]
  ├── Primary line detection & spacing estimation ($r$) for alignment features
  ├── Trajectory tracking across full page width (polynomial fitting on horizontal openings)
  ├── Vectorized 2D continuous vertical run-length filtering ($R_v(y, x)$ via $(9,3)$ dilation)
  ├── Stroke preservation ($max\_R_v \ge 7$ kept; bare lines $max\_R_v \le 6$ erased)
  └── Generation of clean ink masks (`_clean_ink.png`) & rule reference layers
         │
         ▼
[Task & Word Segmentation (`pipeline/segment.py`)]
  ├── Connected component clustering
  ├── Dynamic OCR script detection (Devanagari vs Latin via EasyOCR)
  ├── Protocol state machine / dynamic programming line grouping
  ├── Word bounding-box extraction
  └── Per-word physical baseline extraction & residual computation
         │
         ▼
[Feature Extraction Library (`features/`)]
  ├── Baseline alignment & RMSE (`features/baseline.py`)
  ├── Ruled-line adherence offsets (`features/rule_offset.py`)
  ├── Tangent angle & slant circular dispersion (`features/slant.py`)
  ├── Savitzky-Golay curvature, tortuosity & jerk proxy (`features/curvature.py`)
  ├── Inter-word spacing & gap dispersion (`features/gaps.py`)
  ├── Character size variability & $h/r$ ratios (`features/size.py`)
  ├── Devanagari Shirorekha continuity & matra ratios (`features/hindi.py`)
  └── Stroke graph fragmentation: components, junctions, endpoints (`features/fragmentation.py`)
         │
         ▼
[Dataset Assembly & Normalization (`pipeline/dataset_assembly.py`)]
  ├── Dimensionless scale normalization (by x-height $h$ and rule spacing $r$)
  ├── Robust within-cohort z-scoring per (Grade × Script × Task)
  └── Leakage-free fold imputation
         │
         ▼
[Student-Level Multi-Phenotype Aggregation (`models/student_features.py`)]
  ├── Cross-task central tendencies (mean, median)
  ├── Task difficulty load gradients (Dictation vs Copying)
  ├── Cross-script motor divergence (English vs Hindi)
  └── Peak motor breakdown episodes (maximum fragmentation, peak wobble)
         │
         ▼
[Classification & Multi-Tier Cascade Modeling (`models/`)]
  ├── 5-repeat 5-fold student-stratified nested cross-validation (`models/cv.py`)
  ├── Stability feature selection (`models/feature_selection.py`)
  ├── Multi-model stacking (GBM + RF + Regularized LR)
  └── Two-Tier Clinical Screening Cascade (Triage -> Specialist Verification)
         │
         ▼
[Visual Inspection & Review Web Server (`qa/`)]
  ├── Asynchronous Python web backend (`qa/app.py`)
  ├── Interactive multi-layer canvas (`qa/web/`): Grayscale, Clean Ink, Rules, Skeletons, Baselines
  └── Ground truth verification & interactive crop boundary adjustment
```

## 3. Core Modules & Responsibilities
- `pipeline/preprocess.py`: Image normalization, perspective correction, deskewing.
- `pipeline/rules.py`: Ruled line modeling, trajectory tracking, and stroke-preserving line reduction.
- `pipeline/segment.py`: Text line clustering, script classification, task grouping, word segmentation, and baseline fitting.
- `pipeline/process_dataset.py`: Batch cohort execution and persistence of crops, masks, and metadata.
- `features/`: 8 modular mathematical feature extraction families adhering to $f(\text{sentence}) \to \text{value} \mid \text{NaN}$.
- `models/`: Student-level feature aggregation, stability selection, candidate estimators, nested CV engine, and evaluation metrics.
- `qa/`: Human-in-the-loop review interface, interactive annotation adjustments, and transparency diagnostic tools.

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

- **Next Phase**: Phase 2 Feature Library (baseline wobble & residuals, rule offset, slant tensor, curvature & jerk proxy, inter-word/inter-character gaps, shirorekha continuity features).


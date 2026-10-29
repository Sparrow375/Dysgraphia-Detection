# Task List — Workstream A

## Active Sprint: Ground-Truth Re-Classifications & Phenotypic Gating Analysis

### Objectives
1. **Apply Expert Re-Classifications (`pipeline/apply_user_label_reclassifications.py`)**:
   - `G5_A_Roll07`: 1 -> 0 (Negative).
   - `SCHB_G3_IMG_20261007_101635`: 1 -> 0 (Negative).
   - `SCHB_G4_IMG_20261007_100350`: 0 -> 1 (Positive / Dysgraphic).
2. **Re-Benchmark 5x5 Cross-Validation on Cleaned Cohort (37 Positives, 178 Controls)**:
   - Re-evaluate Dual-ViT Standalone, English OCR Standalone, and Tri-Modal Champion.
   - Update `reports/exp7_bilingual_ocr_fusion_results.json` and `reports/exp7_bilingual_ocr_fusion_report.md`.
3. **Phenotypic Gating Investigation (Is OCR an AND or an OR?)**:
   - Analyze mathematical and clinical trade-offs between Linear Blend (soft AND), Hard Max-Pooling (unconstrained OR), and Asymmetric Non-Penalizing Gating.
   - Document case studies: `G7_A_Roll02` multi-line tile collisions, `SCHB_G6_IMG_20261007_102238` oversized crop, and `G3_A_Roll12` word spacing deficits.
4. **Synchronize Memory & Git Remote**:
   - Update `context.md` (Stage 52), `task_list.md`, and commit/push to `origin/main`.

### Sprint Tasks
- [x] Task 1: Dataset Ground Truth Re-Classification
  - [x] 1.1 Updated `data/manifest.csv`, `review_status.json`, `dataset_student_level.csv`, and `sentences_index.csv`.
  - [x] 1.2 Updated `exp7_all_models_oof_probs.npz` with new binary label vector.
  - [x] 1.3 Regenerated `data/features/exp7_student_model_scores.json`.
- [x] Task 2: Re-Benchmark 5x5 Cross-Validation
  - [x] 2.1 Dual-ViT alone jumped from 0.8440 to **0.8620 AUROC** (AUPRC = **0.6165**).
  - [x] 2.2 Tri-Modal Champion jumped from 0.8519 to **0.8709 AUROC** (AUPRC = **0.6530**).
  - [x] 2.3 English OCR standalone jumped from 0.6326 to **0.6540 AUROC**.
  - [x] 2.4 Sens @ 80% Spec reached **75.7% (28/37)**; Sens @ 90% Spec reached **56.8% (21/37)**.
- [x] Task 3: Phenotypic Gating vs Linear Blend Investigation
  - [x] 3.1 Investigated why linear blend penalizes pure motor dysgraphics with clean spelling.
  - [x] 3.2 Proved empirically that hard Max-Pooling drops AUROC (0.871 -> 0.797) due to phonetic ESL false alarms.
  - [x] 3.3 Formulated Asymmetric Non-Penalizing Gating: Vision anchors the prediction; OCR acts as a unidirectional rescue term.
  - [x] 3.4 Diagnosed `G7_A_Roll02` multi-line box wrapping and strike-through artifacts.
- [x] Task 4: Memory Updates & Git Synchronization
  - [x] 4.1 Updated `context.md` (Stage 52).
  - [x] 4.2 Updated `task_list.md`.
  - [x] 4.3 Commit and push to `origin/main`.

---

## Completed Sprints

<details>
<summary>Comprehensive Error Analysis & QA Web App Multimodal Integration (Completed 2026-10-29)</summary>

## Previous Sprint: Comprehensive Error Analysis & QA Web App Multimodal Integration

### Objectives
1. Full-Cohort Error Analysis on Unified Model (215 Students).
2. Export Multimodal Score & Status Cache (`pipeline/export_student_model_scores.py`).
3. QA Web App Live Integration (`qa/app.py`, `qa/web/index.html`, `qa/web/app.js`).
4. Comprehensive Documentation & Git Synchronization.

</details>


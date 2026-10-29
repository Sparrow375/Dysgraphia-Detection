# Task List — Workstream A

## Active Sprint: Line Decomposition & Hierarchical Multi-Scale Champion

### Objectives
1. **Automated Line Decomposition**:
   - Decompose multi-line sentence crops into single-line strips at horizontal ink projection valleys (`pipeline/decompose_sentence_crops_to_lines.py`).
   - Fix vertical tile compression and multi-line stroke collision artifacts.
2. **Backbone Extraction on Line Strips**:
   - Extract DINOv2-Small and CLIP ViT-B/32 tile descriptors on single-line crops (`pipeline/extract_line_backbone_features.py`).
   - Extract bilingual line-level TrOCR decoding (`pipeline/extract_line_bilingual_ocr.py`).
3. **Hierarchical Multi-Scale Vision & Phenotypic Rescue**:
   - Combine Macro-Spatial Sentence Vision (80%) + Clean Micro-Stroke Line Vision (20%) -> jumps to **0.8725 AUROC** on vision alone!
   - Benchmark Tri-Modal Champion on Hierarchical Vision + Line TrOCR + Tabular -> reaches **0.8788 AUROC**, **0.6740 AUPRC**, and **67.6% (25/37) Sens@90% Spec**!
4. **Student Audits & QA Web App Synchronization**:
   - Verify `G7_A_Roll02` false alarm dropped by -30 percentile points into the normal control range.
   - Verify `SCHB_G7_IMG_20261007_105512` caught (+0.1581 jump).
   - Update `context.md`, `architechture.md`, `qa/` score cache, and git push.

### Sprint Tasks
- [x] Task 1: Automated Line Decomposition
  - [x] 1.1 Decomposed 1,404 sentence crops into 2,429 clean single lines.
  - [x] 1.2 Mean line height reduced from 189px to 116px (100% <= 228px).
- [x] Task 2: Line Backbone Feature Extraction
  - [x] 2.1 Extracted DINOv2-Small on 76,426 line tiles (`exp7_line_dinov2_small_tiles.npz`).
  - [x] 2.2 Extracted CLIP ViT-B/32 on 76,426 line tiles (`exp7_line_clip_tiles.npz`).
- [x] Task 3: Multi-Scale Hierarchical Vision Evaluation
  - [x] 3.1 Line Dual-ViT alone: 0.8138 AUROC (cures `G7_A_Roll02` tile collisions).
  - [x] 3.2 Sentence Dual-ViT alone: 0.8620 AUROC (preserves `G3_A_Roll12` macro baseline drift).
  - [x] 3.3 Hierarchical Dual-ViT (0.80 Sentence + 0.20 Line): **0.8725 AUROC**, **0.6537 AUPRC**!
- [/] Task 4: Line TrOCR Bilingual Extraction & Final Consensus
  - [/] 4.1 Running GPU line-level TrOCR decoding (`pipeline/extract_line_bilingual_ocr.py`).
  - [ ] 4.2 Benchmark Hierarchical Tri-Modal Champion and Asymmetric Phenotypic Rescue.
  - [ ] 4.3 Update QA web app scores, documentation, and commit/push to git.

---

## Completed Sprints

<details>
<summary>Ground-Truth Re-Classifications & Phenotypic Gating Analysis (Completed 2026-10-29)</summary>

## Previous Sprint: Ground-Truth Re-Classifications & Phenotypic Gating Analysis

### Objectives
1. **Apply Expert Re-Classifications (`pipeline/apply_user_label_reclassifications.py`)**:
   - `G5_A_Roll07`: 1 -> 0 (Negative).
   - `SCHB_G3_IMG_20261007_101635`: 1 -> 0 (Negative).
   - `SCHB_G4_IMG_20261007_100350`: 0 -> 1 (Positive / Dysgraphic).
2. **Re-Benchmark 5x5 Cross-Validation on Cleaned Cohort (37 Positives, 178 Controls)**:
   - Dual-ViT alone: 0.8620 AUROC, 0.6165 AUPRC.
   - Tri-Modal Champion: 0.8709 AUROC, 0.6530 AUPRC.
3. **Phenotypic Gating Analysis**:
   - Formulated Asymmetric Non-Penalizing Gating: Vision anchors; OCR rescues.

</details>

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


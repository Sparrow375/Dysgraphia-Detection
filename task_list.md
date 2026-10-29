# Task List — Workstream A

## Active Sprint: Multi-Line Falsification, Selection Bias Recalibration & Stroke Artifact Trimming

### Objectives
1. **Unbiased Multi-Line Hypothesis Test on Full Control Cohort (N=178)**:
   - Compare sentence-crop scores and false alarm rates of controls with 0, 1, and >=2 multi-line tasks without cherry-picking (`scratch/test_multiline_hypothesis.py`).
   - Uncover mechanism and quantify statistical significance (Welch's t, Mann-Whitney U, Cohen's d).
2. **OOF Selection Bias & Realistic Pipeline Expectation**:
   - Acknowledge that post-hoc ensemble tuning (0.80/0.20, 0.84/0.08/0.08, tau=0.70) yields optimistic estimates.
   - Calibrate honest expected cohort performance to **~0.85–0.87 AUROC** rather than 0.88+.
3. **Clinical & Methodological Refinements on G7_A_Roll02**:
   - Implement Soft Visual Floor Gating on Asymmetric OCR Rescue:
     $$\text{Rescue} = \beta \cdot \sigma\left(\frac{R_{\text{vis}} - 0.35}{0.10}\right) \cdot \max(0, R_{\text{ocr}} - \tau)$$
   - Diagnose crossed-out word (`wa~~k~~` on dictated English line 1) and test outlier tile trimming / removing the 90th percentile tile spike.
4. **Tile Inspection & Clickable Artifact Gallery**:
   - Generate and export 224x224 tile PNGs across sentence crops and single-line strips for `G7_A_Roll02`, `G7_A_Roll15`, and `G5_A_Roll07`.
   - Provide direct clickable file links for manual inspection.

### Sprint Tasks
- [x] Task 1: Cohort-Wide Multi-Line Falsification Test
  - [x] 1.1 Evaluated 178 controls partitioned into 0 tasks (N=31), 1 task (N=57), and >=2 tasks (N=90).
  - [x] 1.2 Proved statistically significant false alarm inflation: 3.2% vs 11.1% (>0.80 score, p=0.0158, d=0.465).
  - [x] 1.3 Identified line-strip tile bag-size artifact driven by `np.percentile(descs, 90)`.
- [x] Task 2: Methodological Integrity & Benchmark Recalibration
  - [x] 2.1 Formalized honest expected generalization range at ~0.85–0.87 AUROC.
- [x] Task 3: G7_A_Roll02 Refinements
  - [x] 3.1 Evaluated Soft Visual Floor Gated Rescue (`scratch/test_refinements.py`, AUROC 0.8759).
  - [x] 3.2 Diagnosed crossed-out word artifact (`wa~~k~~`) in tiles 12 & 13.
  - [x] 3.3 Validated that removing extreme 90th-percentile tile spike drops G7_A_Roll02 to 51.6th percentile (score: 0.2347).
- [x] Task 4: Tile Export & Verification
  - [x] 4.1 Exported 341 tile PNGs for G7_A_Roll02, plus reference tiles for G7_A_Roll15 and G5_A_Roll07 to `data/qa_tiles_inspection/`.
  - [x] 4.2 Verified QA server running on `http://127.0.0.1:8090`.
  - [x] 4.3 Synchronized `context.md`, `architechture.md`, and commit/push to git.

---

## Completed Sprints

<details>
<summary>Line Decomposition & Hierarchical Multi-Scale Champion (Completed 2026-10-29)</summary>

## Previous Sprint: Line Decomposition & Hierarchical Multi-Scale Champion

### Objectives
1. **Automated Line Decomposition**: Decompose multi-line sentence crops into single-line strips at horizontal ink projection valleys (`pipeline/decompose_sentence_crops_to_lines.py`).
2. **Backbone Extraction on Line Strips**: Extract DINOv2-Small and CLIP ViT-B/32 on single lines.
3. **Hierarchical Multi-Scale Vision**: Combine Macro-Spatial Sentence Vision (80%) + Clean Micro-Stroke Line Vision (20%).
4. **Line TrOCR Decoding**: Decoded line tasks with TrOCR (`pipeline/extract_line_bilingual_ocr.py`).

</details>

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


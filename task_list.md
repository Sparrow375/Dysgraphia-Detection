# Task List — Workstream A

## Active Sprint: Production Inference Engine & Interactive Upload/Crop Web App

### Objectives
1. **Answer Production Status**: Clarify the status of the production line (line representation promoted; full end-to-end single-image inference being deployed now).
2. **Build Production Inference Engine (`pipeline/inference_engine.py`)**:
   - Fit and freeze the reference cohort 5x5 CV probe heads (StandardScaler + PCA + LogisticRegression) for CLIP, DINOv2, and Dual-ViT across all configurations (`no_guideline`, `baseline`, `raw_ch1`, `raw_no_guideline`).
   - Implement single-crop feature extraction, physical x-height scaling ($h \to 48\text{px}$), rejection gate, and ensemble inference.
   - Return probabilities, cohort percentiles, clinical tiers, and base64 tile images for visual inspection.
3. **Build Interactive Web Application (`qa/inference_app.py` + `qa/web/inference.html`)**:
   - Drag & drop image upload.
   - Interactive HTML5 canvas manual crop selector with draggable handles and live crop preview.
   - Model and preprocessing configuration selectors (CLIP, DINOv2, Dual-ViT; No Guideline, Baseline, Raw).
   - Real-time screening dashboard with clinical risk gauge (Tier 1 vs Tier 2), score cards, and tile inspection gallery (showing Ch1, Ch2, Ch3, and RGB tensor).
4. **Unit & Integration Testing**:
   - Verify inference on sample crops from School A and School B.
   - Test server launch and API endpoints.
5. **Update Documentation & Git**:
   - Update `context.md`, `architechture.md`, and commit/push to git.

### Sprint Tasks
- [x] Task 1: Build `pipeline/inference_engine.py` with frozen probe ensembles and live extraction
- [x] Task 2: Build `qa/web/inference.html` with interactive canvas crop tool and results dashboard
- [x] Task 3: Build `qa/inference_app.py` aiohttp server with `/api/screen` and `/api/health`
- [x] Task 4: Verify end-to-end inference on sample handwriting images and run unit tests
- [x] Task 5: Update `context.md`, `architechture.md`, and commit/push to git

---

## Completed Sprints

<details>
<summary>Channel Ablation Suite — Raw Images in Ch1 & Ruled Guideline Removal (Completed 2026-10-30)</summary>

### Objectives
1. Parameterized channel composition in `baseline/tiling_preprocessor.py`.
2. Extracted DINOv2 and CLIP on RTX 5060 Ti GPU across 3 ablation conditions.
3. Test 2 (`no_guideline`) achieved all-time records: Line CLIP Joint 0.8969, Line Dual-ViT 0.8845, and Tri-Modal Champion 0.8892 AUROC (AUPRC 0.6926, Sens@90: 70.3%, $p=0.0260$).
4. Promoted `no_guideline` representation to production.

</details>

---

## Completed Sprints

<details>
<summary>Native Stroke Scaling, Residual-Line Rejection Gate, & Background De-noising (Completed 2026-10-30)</summary>

### Objectives
1. Fix 1: True character height scaling ($h \to 48\text{px}$, vertical centering on 224px canvas).
2. Fix 2: Empty & residual ruled-line tile rejection gate ($\ge 1,250$ ink px, span $\ge 28\text{px}$, std $\ge 8.0\text{px}$).
3. Fix 3: Pure white canvas background ($D_{\text{clean}} = D \cdot M$, zero pink grain).
4. GPU re-extraction on 215 students: Hierarchical Tri-Modal Champion reached 0.8841 AUROC and 64.9% Sens @ 90% Spec.

</details>

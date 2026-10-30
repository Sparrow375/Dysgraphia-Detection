# Task List — Workstream A

## Active Sprint: Channel Ablation Suite — Raw Image in Ch1 & Ruled Guideline Removal

### Objectives
1. **Support Parameterized Channel Composition (`baseline/tiling_preprocessor.py`)**:
   - `ch1_mode`: `"binary"` (clean inverted binary ink) vs `"raw"` (natural grayscale image fed directly into Channel 1).
   - `include_guideline`: `True` (0.7 gray guideline reference in Ch3) vs `False` (ruled guideline channel completely removed).
2. **Update Extraction CLI (`pipeline/extract_line_backbone_features.py`)**:
   - Add `--ch1-mode [binary, raw]`, `--no-guideline`, and `--tag <str>`.
3. **Execute Test 1: Raw Image in Channel 1 (`--ch1-mode raw`)**:
   - Extract DINOv2-Small and CLIP ViT-B/32 on RTX 5060 Ti GPU.
   - Evaluate 5x5 repeated cross-validation on 215 students.
4. **Execute Test 2: Guideline Channel Removed (`--no-guideline`)**:
   - Extract DINOv2-Small and CLIP ViT-B/32 on RTX 5060 Ti GPU.
   - Evaluate 5x5 repeated cross-validation on 215 students.
5. **Execute Test 3: Raw Image + Guideline Removed (`--ch1-mode raw --no-guideline`)**:
   - Extract DINOv2-Small and CLIP ViT-B/32 on RTX 5060 Ti GPU.
   - Evaluate 5x5 repeated cross-validation on 215 students.
6. **Synthesize Comparative Scoreboard & Clinical Analysis**:
   - Compare Line CLIP, Line DINOv2, Dual-ViT, Hierarchical Vision, and Tri-Modal Champion across all variants.
   - Update `context.md`, `architechture.md`, and commit/push to git.

### Sprint Tasks
- [x] Task 1: Update `baseline/tiling_preprocessor.py` with `ch1_mode` and `include_guideline`
- [x] Task 2: Update `pipeline/extract_line_backbone_features.py` with CLI flags and tagging
- [x] Task 3: Run Test 1 (Raw Image in Ch1) on GPU & evaluate 5x5 CV
- [x] Task 4: Run Test 2 (No Guideline in Ch3) on GPU & evaluate 5x5 CV
- [x] Task 5: Run Test 3 (Raw in Ch1 + No Guideline in Ch3) on GPU & evaluate 5x5 CV
- [x] Task 6: Compile comparative ablation scoreboard & analyze findings
- [x] Task 7: Update `context.md`, `architechture.md`, and commit/push to git

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

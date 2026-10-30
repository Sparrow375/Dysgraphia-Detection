# Task List — Workstream A

## Active Sprint: Native Stroke Scaling, Residual-Line Rejection Gate, & Background De-noising

### Objectives
1. **Fix 1: Scale by True Character Height ($h$), NOT Arbitrary Crop Box Height ($H$)**:
   - Set resize factor so that estimated character x-height $h$ (`x_height_h`) maps to a fixed target of $48\text{ px}$.
   - Center line vertically on a clean white $224 \times 224$ canvas.
   - Guarantees 1:1 physical magnification across all students, lines, and grades (~4.6 letters per tile, preserving intra-word context).
2. **Fix 2: Empty & Residual-Line Tile Rejection Gate**:
   - Minimum foreground ink pixel count: $\ge 1,250$ pixels ($\ge 2.5\%$ of 50,176).
   - Vertical span gate: $y_{\max} - y_{\min} \ge 28\text{ px}$ ($\ge 0.6 \times$ target x-height).
   - Vertical projection variance / std gate: $\text{std}(y \mid \text{ink}) \ge 8.0\text{ px}$ to 100% reject flat ruled lines.
3. **Fix 3: Strip Pink Background Grain Out of Channel 2**:
   - Strictly zero optical density outside binary ink mask ($D_{\text{clean}} = D \cdot M$).
   - Enforce pure white background $[255, 255, 255]$ everywhere outside ink strokes.
4. **End-to-End Line Decomposition & Clean Mask Synchronization**:
   - Synchronized `features/utils.py:load_or_binarize_ink` to prevent returning mismatched mask shapes on recropped tasks.
   - Updated `pipeline/extract_line_backbone_features.py` and `baseline/tiling_preprocessor.py`.
5. **Validation, GPU Re-Extraction & Benchmark Evaluation**:
   - Unit tests in `tests/test_tiling_preprocessor.py` passing (4/4 passed).
   - Re-extracted clean line DINOv2 (32,509 tiles in 64.6s) and CLIP (32,509 tiles in 41.9s) on RTX 5060 Ti GPU.
   - Evaluated 5x5 CV benchmark: Line CLIP AUROC surged to 0.8643, Hierarchical Tri-Modal Champion reached **0.8841 AUROC**, **0.6650 AUPRC**, and **64.9% Sens @ 90% Spec (24/37 detected)**.
   - Exported refreshed inspection gallery for `G7_A_Roll02` (empty line 1 has 0 tiles; line 3 "lazy dog" has exactly 4 clean tiles).

### Sprint Tasks
- [x] Task 1: Update `baseline/tiling_preprocessor.py` with Fix 1, Fix 2, and Fix 3
- [x] Task 2: Synchronize clean ink mask handling in line decomposition & tile extraction
- [x] Task 3: Comprehensive unit test suite in `tests/test_tiling_preprocessor.py`
- [x] Task 4: Re-extract line backbone features on GPU and run 5x5 CV benchmark
- [x] Task 5: Export updated inspection tiles for verification
- [x] Task 6: Synchronize `context.md`, `architechture.md`, and commit/push to git

---

## Completed Sprints

<details>
<summary>Multi-Line Falsification, Selection Bias Recalibration & Stroke Artifact Trimming (Completed 2026-10-29)</summary>

### Objectives
1. Unbiased Multi-Line Hypothesis Test on Full Control Cohort (N=178).
2. OOF Selection Bias & Realistic Pipeline Expectation (~0.85-0.87 AUROC).
3. Soft Visual Floor Gated Rescue on G7_A_Roll02.
4. Export inspection tiles to `data/qa_tiles_inspection/`.

</details>

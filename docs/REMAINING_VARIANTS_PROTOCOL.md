# Protocol for Pre-Committed Exploration Variants & Stopping Rules

**Status:** RATIFIED & PRE-COMMITTED  
**Date:** 2026-10-28  
**Scope:** Strict Protocol for Any Further Model Variants on the 215-Student Screening Cohort  

---

## 1. Multiple Testing Problem & The "One More Look" Hazard

Every time a new neural architecture, layer depth, or hyperparameter configuration is evaluated on the existing 215 students, it constitutes another "look" at the same data. On a dataset with 40 positives, the standard error of AUROC is approximately $\pm 0.035$ (95% CI width $\approx \pm 0.07$).

Evaluating 10–20 uncommitted model variations will inevitably surface one that scores $0.86$ or $0.87$ purely through empirical variance, creating a false illusion of progress.

To prevent this, the research team establishes a **pre-committed, closed candidate list of exactly four variants**, accompanied by a strict **paired-bootstrap decision rule**. No other architectures will be benchmarked on this cohort.

---

## 2. The Four Pre-Committed Variants

### Variant 1: Intermediate DINOv2 Layers (Layer 6 / Layer 9 vs Layer 12)
- **Hypothesis**: In self-supervised Vision Transformers, the final transformer blocks (Layers 11–12) specialize in global semantic grouping, which may attenuate high-frequency stroke texture, pencil grain, and micro-tremor. Intermediate blocks (Layer 6 or 9) preserve finer local geometric fidelity.
- **Specification**:
  - Backbone: `dinov2_vits14`
  - Extraction: Hidden state activations from Transformer Block 6 or Block 9 (prior to final norm and classification head).
  - Preprocessing, tiling, spatial moments ($\mu, \sigma, GeM$), PCA ($K=24$), and Ridge probe ($C=0.1$) remain identical to the frozen pipeline.

### Variant 2: Scaled DINOv2-Base (`dinov2_vitb14`)
- **Hypothesis**: Model capacity scaling. ViT-Base increases embedding dimensionality from $D=384$ to $D=768$ and parameter count from 21M to 86M. We test whether higher-dimensional patch representations improve separation or merely overfit the 215-student sample space.
- **Specification**:
  - Backbone: `dinov2_vitb14`
  - Output token dimension: $D = 768$ ($3D = 2,304$ per tile, student vector $D = 9,216$).
  - Probe: Identical PCA ($K=24$) and Ridge probe ($C=0.1$).

### Variant 3: SigLIP 2 (`google/siglip-base-patch16-224`)
- **Hypothesis**: Contrastive vision-language representation with pairwise sigmoid loss (SigLIP) rather than softmax cross-entropy (CLIP). SigLIP has demonstrated significantly stronger fine-grained localization and dense visual feature quality than OpenAI CLIP. We test whether SigLIP provides a superior multimodal visual stream for handwriting consensus.
- **Specification**:
  - Backbone: `google/siglip-base-patch16-224`
  - Tiling: Same 3-channel native stroke tiles ($224 \times 224$).
  - Spatial patch moments pooled across tiles.

### Variant 4: DINOv2 on Squashed Sentence Crops (The ViT Tiling Ablation)
- **Hypothesis**: In Experiment 7, DenseNet201 showed a $+0.06$ AUROC improvement when transitioning from squashed crops ($0.6993$) to native stroke tiles ($0.7591$). We isolate whether DINOv2's strong performance ($0.8427$) depends on native stroke-scale tiling or if its attention mechanism is invariant to anisotropic stretching.
- **Specification**:
  - Whole sentence crops anisotropically squashed to $224 \times 224$ (Exp 1–4 format).
  - Backbone: `dinov2_vits14` with identical spatial moment pooling and Ridge probe.

---

## 3. Strict Paired-Bootstrap Decision Rule

Because every variant is evaluated on the exact same 215 students under the exact same 5-repeat 5-fold CV splits (`data/folds/student_folds_lookup.csv`), differences between models must be evaluated using **paired bootstrap resampling** ($B = 2,000$ iterations):

$$\Delta \text{AUROC} = \text{AUROC}_{\text{variant}} - \text{AUROC}_{\text{frozen\_baseline (0.8427)}}$$

### Adoption Threshold
A variant will **ONLY** replace the frozen `DINOv2-Small` primary model if all three conditions are satisfied:
1. $\Delta \text{AUROC} > 0.0000$ (positive mean gain).
2. The lower bound of the 95% two-sided paired bootstrap confidence interval is **strictly positive**:
   $$\text{CI}_{95\%}^{\text{lower}}(\Delta \text{AUROC}) > 0.0000$$
3. Two-sided empirical permutation or bootstrap $p < 0.05$.

### Statistical Tie Rule
If a variant achieves a nominal increase (e.g. $0.852$ vs $0.843$, $\Delta = +0.009$) but its 95% CI spans zero (e.g. $[-0.025, +0.043]$), it is declared a **statistical tie**. 
Under the tie rule, **no change to the frozen baseline is made**, preserving system simplicity, runtime speed, and parameter parsimony.

---

## 4. Execution Protocol & Verification Harness
Whenever any of the four pre-committed variants is executed:
1. Features must be cached in `data/features/exp7_{variant_name}.npz`.
2. Out-of-fold probability vectors must be merged into `data/features/exp7_all_models_oof_probs.npz`.
3. Paired bootstrap statistics must be generated via `baseline/compute_paired_bootstrap.py` and appended to `reports/exp7_paired_bootstrap_results.json`.
4. Scoreboards must be generated directly from the JSON files via `baseline/generate_report_tables.py`.

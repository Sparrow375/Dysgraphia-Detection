# Formal Pipeline Freeze & One-Shot Evaluation Protocol

**Status:** APPROVED & FROZEN  
**Date of Freeze:** 2026-10-28  
**Cohort Scope:** Unified 215-Student Bilingual Screening Cohort (Grades 3–8; 40 Positives, 175 Controls)  
**Parent Commit:** `fd40ed1` (tracked on `origin/main`)  

---

## 1. Rationale & Policy

Following the Experiment 7 representation bake-off, paired bootstrap hypothesis testing, shortcut falsification audits, and exhaustive error analysis, the core screening architecture has reached an empirical ceiling on the existing 215-student dataset:
- Self-supervised ViT patch representations achieve **0.8427 AUROC** (DINOv2 standalone) and **0.8549 AUROC** (Dual-ViT fixed 50/50 rank ensemble).
- Paired bootstrap ($B=2000$) demonstrates that the $+0.012$ ensemble gain is a statistical tie ($p = 0.2435$), and that the remaining false negatives (3 students missed across all models) reflect subtle, compensated handwriting or truncated tasks rather than an architectural deficiency.
- Continued tuning on the same 215 students poses an unacceptable risk of overfitting to label noise and subjective screening thresholds.

**Freezing Rule:**  
From this date forward, all preprocessing, tiling geometry, backbone model weights, spatial pooling descriptors, dimensionality reduction, and classification probe hyperparameters are **strictly locked**. Any new school data (e.g., School C, School D, or external clinic datasets) must be evaluated as a **sealed, one-shot holdout test**.

---

## 2. Frozen Architectural Specifications

### 2.1 Preprocessing & Native Stroke-Scale Tiling
- **Input Imagery**: Grayscale or RGB handwriting crops corresponding to standard protocol tasks (Hindi copying/dictation/own, English copying/dictation/own).
- **Target Scale**: Native resolution preserved at 150 DPI (no squashing or anisotropic resizing).
- **Tile Dimensions**: Fixed $224 \times 224$ pixels, horizontal stride of 224 pixels (non-overlapping).
- **Composite 3-Channel Tensor ($224 \times 224 \times 3$)**:
  - **Channel 0 (Binarized Stroke Mask)**: Sauvola/Otsu ink mask ($I(x,y) \in \{0, 1\}$).
  - **Channel 1 (Grayscale Optical Density)**: Inverted normalized grayscale ($G(x,y) = 1.0 - \frac{\text{raw}}{255.0} \in [0.0, 1.0]$), preserving graphite pressure variation, micro-tremor, and line shading.
  - **Channel 2 (Guideline Distance Field)**: Euclidean distance transform to nearest ruled guideline ($D(x,y) \in [0.0, 1.0]$), normalized by line spacing.
- **Tile Filtering Rule**:
  - Tiles with foreground ink density $< 0.5\%$ ($\sum I(x,y) < 250$ pixels) are discarded as blank margin noise.
- **Padding**: White background padding ($255$) if crop width $< 224$ px.

### 2.2 Visual Foundation Backbones (Zero Gradient Updates)
The system operates exclusively on frozen feature representations. No fine-tuning or end-to-end backpropagation is permitted.

#### Primary Standalone Backbone: DINOv2-Small (`dinov2_vits14`)
- **Hub Repository**: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`
- **Architecture**: Vision Transformer Small with $14 \times 14$ patch size.
- **Token Grid**: $16 \times 16 = 256$ spatial patch tokens per $224 \times 224$ tile.
- **Embedding Dimension**: $D = 384$.
- **Feature Extraction Layer**: Final block output patch tokens (excluding `[CLS]` token), producing tensor $T_{\text{patch}} \in \mathbb{R}^{256 \times 384}$.

#### Complementary Multimodal Backbone: OpenAI CLIP ViT-B/32 (`clip_vit_b32`)
- **Repository**: Official `clip` package (`ViT-B/32`).
- **Token Grid**: $7 \times 7 = 49$ spatial patch tokens per $224 \times 224$ tile.
- **Embedding Dimension**: $D = 512$.
- **Role**: Captures macro stroke topology and visual-linguistic structural layout.

### 2.3 Spatial Patch Descriptor & Student Aggregation Engine
To summarize spatial variability without destroying stroke distribution information, spatial moments are extracted across patch tokens:

1. **Tile Descriptor ($e_{\text{tile}} \in \mathbb{R}^{3D}$)**:
   For patch tokens $t_1, \dots, t_P \in \mathbb{R}^D$:
   - Spatial Mean: $\mu = \frac{1}{P} \sum_{p=1}^P t_p \in \mathbb{R}^D$
   - Spatial Standard Deviation: $\sigma = \sqrt{\frac{1}{P} \sum_{p=1}^P (t_p - \mu)^2} \in \mathbb{R}^D$
   - Spatial Generalized Mean (GeM, $p=3$): $g = \left( \frac{1}{P} \sum_{p=1}^P |t_p|^3 \right)^{1/3} \in \mathbb{R}^D$
   - Concatenate: $e_{\text{tile}} = [\mu, \sigma, g] \in \mathbb{R}^{1152}$ (for DINOv2 $D=384$).

2. **Student Script Descriptor ($e_{\text{student, script}} \in \mathbb{R}^{2304}$)**:
   Across all tiles $T_{\text{script}}$ belonging to a student for a language (Hindi or English):
   - Tile Mean: $\mu_{\text{tiles}} = \text{mean}_{t \in T} (e_{\text{tile}}) \in \mathbb{R}^{1152}$
   - Tile Standard Deviation: $\sigma_{\text{tiles}} = \text{std}_{t \in T} (e_{\text{tile}}) \in \mathbb{R}^{1152}$
   - Concatenate: $e_{\text{student, script}} = [\mu_{\text{tiles}}, \sigma_{\text{tiles}}] \in \mathbb{R}^{2304}$.

3. **Multilingual Student Vector ($X_{\text{student}} \in \mathbb{R}^{4608}$)**:
   - Joint Feature Concatenation: $X_{\text{student}} = [e_{\text{student, hindi}}, e_{\text{student, english}}] \in \mathbb{R}^{4608}$.

### 2.4 Linear Probe & Dimensionality Reduction
The linear probe maps the high-dimensional spatial descriptor to a calibrated dysgraphia risk probability:
- **Feature Standardization**: `StandardScaler(with_mean=True, with_std=True)` (fitted on training data).
- **Dimensionality Reduction**: Principal Component Analysis (`PCA(n_components=24, random_state=42)`).
- **Classifier**: $L_2$-regularized Logistic Regression:
  - `C = 0.1`
  - `penalty = 'l2'`
  - `solver = 'lbfgs'`
  - `class_weight = 'balanced'`
  - `max_iter = 1000`
  - `random_state = 42`

### 2.5 Consensus Ensemble Rule: Dual-ViT Fixed 50/50 Rank Average
For deployment requiring the ensemble:
1. Obtain predicted out-of-fold probability $p_{\text{dino}}$ from the DINOv2-Small probe.
2. Obtain predicted out-of-fold probability $p_{\text{clip}}$ from the CLIP-ViT-B/32 probe.
3. Compute fractional cohort ranks:
   $$R_{\text{dino}} = \frac{\text{Rank}(p_{\text{dino}})}{N}, \quad R_{\text{clip}} = \frac{\text{Rank}(p_{\text{clip}})}{N}$$
4. Calculate final consensus risk score:
   $$S_{\text{consensus}} = 0.50 \cdot R_{\text{dino}} + 0.50 \cdot R_{\text{clip}}$$
*Note: No post-hoc soft-probability weighting or fold-specific search is permitted.*

---

## 3. Benchmark Scoreboard Under Frozen Pipeline

All models evaluated strictly on the same 215 students under 5-repeat 5-fold cross-validation:

| Model | Representation | 5x5 CV AUROC | AUPRC | Sens @ 80% Spec | Sens @ 90% Spec | Paired Bootstrap vs DINOv2 (p-val) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dual-ViT 50/50 Rank** | DINOv2 + CLIP (Fixed Rank) | **0.8549** | **0.5496** | 75.0% (30/40) | **42.5% (17/40)** | $+0.0123$ ($p = 0.2435$) | **FROZEN ENSEMBLE** |
| **DINOv2-Small (Joint)** | Frozen ViT-S/14 Patches | **0.8427** | **0.5078** | **77.5% (31/40)** | 40.0% (16/40) | Baseline Reference | **FROZEN PRIMARY** |
| **DINOv2-Small (Blend)** | 50/50 Script Decision Blend | 0.8343 | 0.5633 | 70.0% (28/40) | 47.5% (19/40) | $-0.0084$ ($p = 0.5280$) | Reference |
| **OpenAI CLIP (Blend)** | Frozen ViT-B/32 Patches | 0.8151 | 0.5132 | 75.0% (30/40) | 45.0% (18/40) | $-0.0276$ ($p = 0.2615$) | Reference |
| **DenseNet201 (Tiled)** | Frozen CNN Patches | 0.7591 | 0.4726 | 52.5% (21/40) | 42.5% (17/40) | $\mathbf{-0.1023}$ ($\mathbf{p = 0.0000}$) | Ablated |
| **DenseNet201 (Squashed)**| Squashed Sentence Crops | 0.6993 | 0.4274 | 50.0% (20/40) | 27.5% (11/40) | $-0.1434$ ($p = 0.0000$) | Ablated |
| **Stage 4 Tabular Stack** | 30 Handcrafted BHK Features | 0.6870 | 0.4001 | 47.5% (19/40) | 37.5% (15/40) | $\mathbf{-0.1121}$ ($\mathbf{p = 0.0075}$) | Ablated |

---

## 4. One-Shot External Holdout Protocol

When new school cohorts arrive (e.g. School C / School D):
1. **Zero Contamination**: The new dataset must remain completely untouched during model training.
2. **Single Evaluation Run**:
   - The frozen scaler, PCA ($K=24$), and Ridge probe fitted on the 215-student reference cohort will transform the new student features.
   - Predictions are generated in a single forward pass.
3. **Reportable Metrics**:
   - AUROC and AUPRC against external ground truth.
   - Sensitivity at the frozen operational 80% and 90% specificity thresholds ($0.3459$ and $0.6188$ for DINOv2; $0.6786$ and $0.7884$ for Dual-ViT).
4. **No Iterative Tweaking**: Any hyperparameter search or weight adjustments based on new school test scores invalidates the holdout.


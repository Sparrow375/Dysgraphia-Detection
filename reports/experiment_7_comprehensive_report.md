# Experiment 7: Frozen-Feature Representation Bake-Off & Multi-Stream Ensembles
## Technical Report & Screening Benchmark Evaluation

**Project**: Automated Pediatric Dysgraphia Screening Across Dual Scripts (Hindi & English)  
**Cohort**: 215 Students (40 Screening-Flagged Positives, 175 Peer Controls; Grades 3–8 across School A and School B)  
**Evaluation Protocol**: Locked 5-Repeat $\times$ 5-Fold Stratified Group CV (25 Outer Folds, 0.0% Leakage)  
**Hardware Harness**: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM), AMD Ryzen 5 7600X, 32 GB DDR5 RAM  
**Repository State**: Commit `6b2f0d9` on `main`  
**Date**: October 2026  

---

## 1. Executive Summary & Context

### 1.1 The "Data-Starved" Insight
Experiments 1 through 6 focused primarily on training and fine-tuning convolutional and sequence networks (DenseNet, ResNet, CRNN, TrOCR) directly on raw handwriting crops. In a screening cohort of 215 students, each training fold contains only $\approx 172$ students ($\approx 32$ flagged cases). Backpropagation across deep architectures in this regime led to two systematic failures:
1. **Representational memorization**: Deep heads overfit to idiosyncratic student flourishes rather than learning generalizable motor control deficits.
2. **Aspect ratio distortion**: Squashing wide multi-word sentences into $224 \times 224$ global boxes compressed delicate character loops and destroyed physical stroke scale.

### 1.2 The Experiment 7 Shift
Experiment 7 transitioned to a **Frozen-Feature Representation Bake-Off**:
* **Completely Frozen Foundation Encoders**: No fine-tuning gradients propagated through vision backbones.
* **Native Stroke-Scale Tiling**: Sentence crops are tiled at native 150 DPI resolution ($224 \times 224$ sliding patches with 50% overlap), preserving character geometry and line context.
* **Spatial Patch-Token Pooling**: Rather than relying on semantic `[CLS]` tokens, spatial token descriptors (mean $\mu$, standard deviation $\sigma$, and Generalized Mean $GeM$ pooling) capture stroke texture, line raggedness, and pressure density.
* **Regularized Linear Probes**: Standardized Ridge and Logistic Regression heads ($L_2$ regularization with 3-fold inner-CV parameter tuning) prevent probe overfitting.

### 1.3 Key Empirical Findings
1. **ViT Representation Regime (~0.82–0.85 AUROC)**: Frozen self-supervised (DINOv2) and multimodal contrastive (OpenAI CLIP) ViT patch representations decisively outperform supervised CNNs and OCR encoders.
2. **Deconfounding Tiling vs. Probing**:
   * Squashed DenseNet fine-tuned (Exp 1–4): $\approx 0.62$ AUROC.
   * Squashed DenseNet frozen linear probe: **0.6993 AUROC**.
   * Tiled DenseNet frozen linear probe: **0.7591 AUROC** ($+0.06$ AUROC gain directly attributable to native tiling).
3. **The Style-Invariance Paradox**: OCR sequence-to-sequence pretraining (Microsoft TrOCR-Base) actively erases dysgraphic stroke irregularities, collapsing to **0.6560 AUROC** (English alone: **0.5713**).
4. **ViT Ensemble Statistical Tie**: Fusing DINOv2 with CLIP yields **0.8529–0.8549 AUROC** (depending on whether fixed 50/50 rank averaging or nested fold-tuning is used). Because this sits well within the $\pm 0.055$ bootstrap confidence intervals of standalone DINOv2 ($0.8427$), we characterize this as a **statistical tie among ViT representations in the 0.82–0.85 AUROC regime**, rather than an isolated "record".

---

## 2. Experimental Protocol & Architectural Framework

### 2.1 The Unified Cohort Constraint
In strict accordance with project rules (`guide.md`), **all evaluations operate across the entire unified cohort of 215 students**. No per-grade tuning is used, and no model is evaluated on single-grade slices.

```
       UNIFIED COHORT DEMOGRAPHICS (N = 215 Students)
┌─────────────────────────────────────────────────────────────┐
│  School A: 115 Students (24 Positives, 91 Controls)         │
│  School B: 100 Students (16 Positives, 84 Controls)         │
│  Grade 3: 37  |  Grade 4: 51  |  Grade 5: 33                │
│  Grade 6: 41  |  Grade 7: 42  |  Grade 8: 11                │
└─────────────────────────────────────────────────────────────┘
```
> [!NOTE]
> Positive labels are **screening-flagged** via standardized peer-deviation protocols and teacher assessments, not clinical neuropsychological DSM-5 diagnoses. We adhere strictly to screening terminology.

### 2.2 Locked 5x5 Repeated Stratified Group CV Harness
* **Repeats**: 5 independent seeds (`repeat_0` through `repeat_4`).
* **Folds**: 5 stratified folds per repeat, grouped strictly by `student_id` (0.0% student leakage).
* **Inner CV Tuning**: All standardizers, PCA reductions, and regularization parameters ($C$) are tuned strictly inside 3-fold inner CV splits on training fold partitions.

### 2.3 Native Stroke-Scale Tiling Engine
Wide sentence crops are normalized to a uniform line height of 224 pixels while strictly preserving the horizontal aspect ratio. A sliding window extracts $224 \times 224$ tiles with 50% overlap ($112\text{ px}$ step). Each tile contains:
* Channel 1: Clean binarized Otsu stroke mask.
* Channel 2: Natural grayscale optical density.
* Channel 3: Ruled guideline distance field.

---

## 3. Single-Source-of-Truth Empirical Scoreboard

All metrics in this table are drawn directly from the respective evaluation JSON files in `reports/`:

| Rank | Model / Architecture | Pipeline Variant | 5x5 CV AUROC | AUPRC | Sens @ 80% Spec | Sens @ 90% Spec | Primary JSON Source |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Dual-ViT Ensemble** | **Fixed 50/50 Rank Average** | **0.8549** | **0.5496** | **75.0% (30/40)** | **42.5% (17/40)** | `multistream_ensemble` |
| **2** | **Dual-ViT Ensemble** | **Nested Fold-Tuned Soft** | **0.8529** | **0.5708** | **75.0% (30/40)** | **45.0% (18/40)** | `multistream_ensemble` |
| **3** | **DINOv2-Small (ViT-S/14)** | **Joint Concatenated Model** | **0.8427** | **0.5078** | **77.5% (31/40)** | **40.0% (16/40)** | `exp7_dinov2_small_probe` |
| **4** | **DINOv2-Small (ViT-S/14)** | **50/50 Decision Blend** | **0.8343** | **0.5633** | **70.0% (28/40)** | **47.5% (19/40)** | `exp7_dinov2_small_probe` |
| **5** | **Intermediate Feature Head** | **DINOv2 + Tabular Macro10** | **0.8289** | **0.5091** | **77.5% (31/40)** | **32.5% (13/40)** | `macro10_fusion` |
| **6** | **OpenAI CLIP (ViT-B/32)** | **50/50 Decision Blend** | **0.8151** | **0.5132** | **75.0% (30/40)** | **45.0% (18/40)** | `exp7_clip_probe` |
| **7** | **OpenAI CLIP (ViT-B/32)** | **Joint Concatenated Model** | **0.8109** | **0.4449** | **72.5% (29/40)** | **50.0% (20/40)** | `exp7_clip_probe` |
| **8** | **DenseNet201 (CNN-201)** | **Tiled 50/50 Decision Blend** | **0.7591** | **0.4726** | **52.5% (21/40)** | **42.5% (17/40)** | `exp7_densenet201_probe` |
| **9** | **DenseNet201 (CNN-201)** | **Squashed Sentence Ablation** | **0.6993** | **0.4274** | **50.0% (20/40)** | **37.5% (15/40)** | `ablate_squashed_densenet` |
| **10** | **Stage 4 Tabular Stack** | **30 Handcrafted BHK Features** | **0.6870** | **0.4001** | **47.5% (19/40)** | **37.5% (15/40)** | `verify_stage4_benchmark` |
| **11** | **Writer-Style Encoder** | **SupCon Metric Learning** | **0.6823** | **0.3804** | **57.5% (23/40)** | **40.0% (16/40)** | `candidate2_writer_style` |
| **12** | **Microsoft TrOCR-Base** | **50/50 Decision Blend** | **0.6560** | **0.3949** | **47.5% (19/40)** | **35.0% (14/40)** | `exp7_trocr_probe` |
| **13** | **Candidate 4 VLM Rubric** | **Qwen2-VL-2B Clinical Rubric** | **0.6166** | **0.2484** | **7.5% (3/40)** | **5.0% (2/40)** | `exp7_vlm_rubric` |
| **14** | **Maturity Gap ($\Delta G$)** | **Chronological Age Deficit** | **0.5714** | **0.3010** | **35.0% (14/40)** | **20.0% (8/40)** | `maturity_gap` |

---

## 4. Detailed Breakdown of Candidates & Ablations

### 4.1 Candidate 1: DINOv2-Small (Self-Supervised ViT-S/14)
* **Pretraining Objective**: Masked Image Modeling in latent feature space (predicting teacher patch representations) combined with multi-crop DINO self-distillation and KoLeo regularizer.
* **Empirical Script Breakdown**:
  * Hindi Model Alone: AUROC = **0.8151**, AUPRC = **0.4882**, Sens@80% = **67.5% (27/40)**.
  * English Model Alone: AUROC = **0.7330**, AUPRC = **0.4543**, Sens@80% = **57.5% (23/40)**.
  * Joint Model (Concatenated scripts): AUROC = **0.8427**, AUPRC = **0.5078**, Sens@80% = **77.5% (31/40)**.
  * Decision Blend (0.5 Hindi + 0.5 English): AUROC = **0.8343**, AUPRC = **0.5633**, Sens@80% = **70.0% (28/40)**.
* **Mechanism**: Because DINOv2's loss forces patch representations to encode local geometric and spatial context rather than an abstract image-level classification token, the spatial descriptors capture stroke boundary regularities and graphite density variations across the $14 \times 14$ patches.

### 4.2 Deconfounding the Tiling vs. Probing Effect (DenseNet201 Ablation)
To cleanly separate the gain from native stroke-scale tiling from the gain of using a frozen linear probe:
* **Squashed DenseNet Fine-Tuned (Exp 1–4)**: $\approx 0.62$ AUROC (backpropagation across all layers overfits on 172 students).
* **Squashed DenseNet Frozen Linear Probe**: **0.6993 AUROC** ($+0.08$ gain purely from freezing the backbone and regularizing the head).
* **Tiled DenseNet Frozen Linear Probe**: **0.7591 AUROC** ($+0.06$ additional gain purely from native stroke-scale tiling).
* **Conclusion**: Both levers matter: freezing the representation avoids small-sample overparameterization, and native tiling prevents letter-loop distortion. DINOv2 further improves upon tiled DenseNet by $+0.0836$ AUROC, proving self-supervised ViT patch features carry superior signal.

### 4.3 Candidate 2: Contrastive Writer-Style Metric Learning
* **Method**: SupCon metric learning ($1152 \to 512 \to 128$) trained strictly inside training folds on student identity ($N=172$), treating lines from the same child as positive pairs.
* **Outcome**: Decision Blend AUROC = **0.6823**, AUPRC = **0.3804**.
* **Finding**: Contrastive writer identification dropped performance by $\approx 0.16$ AUROC compared to frozen DINOv2. Disentangling 172 individual identities forces the network to specialize in individual stylistic idiosyncrasies (e.g. personal slant angle), discarding the shared motor control signals that correlate with screening flags.

### 4.4 Candidate 3: Handwriting Maturity Gap Analysis
* **Method**: Train Ridge regression on typical control children ($N=175$) to predict chronological grade $G \in [3, 8]$ from handwriting, computing the deficit $\Delta G = G - \hat{G}$.
* **Outcome**: Chronological grade is highly predictable from handwriting among controls (Spearman $\boldsymbol{\rho = 0.7882}$, $p = 2.5 \times 10^{-38}$). However, the raw deficit $\Delta G$ yields only **0.5714 AUROC**.
* **Finding**: The signal driving the screening flag is not simply "general motor immaturity". Older flagged children in Grades 6–8 exhibit mature macroscopic handwriting traits (smaller scale, pen usage), meaning their predicted grade does not collapse enough for $\Delta G$ to serve as a standalone detector.

### 4.5 Candidate 4: VLM Pediatric Clinical Rubric Scorer
* **Method**: Batched inference (`batch_size=4`) across all 1,145 sentence crops using `Qwen/Qwen2-VL-2B-Instruct` on RTX 5060 Ti GPU (elapsed: 15.2 min, 0.86s/task, cached in `data/features/vlm_rubric_cache.json`).
* **Outcome**: Standalone AUROC = **0.6166**, AUPRC = **0.2484**.
* **BHK Correlation**: `vlm_letter_formation` correlates significantly with physical `components_per_unit_width` (Spearman $\rho = -0.193$, $p = 0.0045$).
* **Finding**: While qualitative rubric features align with stroke fragmentation, coarse quantization (predominantly 3.0 vs. 4.0) limits standalone discriminative power.

---

## 5. Falsification Diagnostics & Confound Checks

### 5.1 Demographic & Confound Baselines
* **School-Only Predictor**: AUROC = **0.5400** (pure noise).
* **Grade-Only Predictor**: AUROC = **0.5217** (pure noise).
* **Combined [School + Grade] 5-Fold CV**: AUROC = **0.4804**, AUPRC = **0.1867** (identical to random base rate $40/215 = 0.186$).
* **Permutation Null Distribution ($N=50$)**: Mean null AUROC = **0.5000 $\pm$ 0.0698**, Max null = 0.6644. Empirical significance: **$p < 0.02$** ($0/50$ permutations exceeded the true score; true score sits $4.9\sigma$ above null mean).

### 5.2 Out-of-Distribution Transfer
* **Cross-School Transfer (School B $\to$ School A)**: AUROC = **0.7482**, AUPRC = **0.5513**. Visual features generalize zero-shot across completely unseen school environments and cameras.
* **Cross-Instrument Transfer (Primary Pencil $\to$ Middle Pen)**: AUROC = **0.7684**, AUPRC = **0.3819**. A probe trained purely on pencil handwriting detects screening flags in ballpoint pen writers at 0.768 AUROC.

---

## 6. Multi-Stream Ensembles & Weighting Methodology

### 6.1 How Were Ensemble Weights Chosen?
To address potential optimization bias:
1. **Fixed 50/50 Rank Average**: AUROC = **0.8549**, AUPRC = **0.5496**. Zero parameter tuning on out-of-fold predictions.
2. **Fixed 50/50 Soft Average**: AUROC = **0.8536**, AUPRC = **0.5590**.
3. **Nested Fold-Tuned Soft Blend**: AUROC = **0.8529**, AUPRC = **0.5708**. Blend weights were searched strictly over training partitions in a nested loop (yielding fold weights of $[0.6, 0.7, 0.5, 0.7, 0.6]$).
4. **Heuristic 60/40 Soft Blend**: AUROC = **0.8540**, AUPRC = **0.5671**.

### 6.2 Statistical Verdict on the "Record"
The difference between standalone DINOv2 ($0.8427$) and the Dual-ViT ensemble ($0.8529–0.8549$) is $+0.010–0.012$ AUROC, which is well within the bootstrap $95\%$ confidence interval ($\pm 0.055$). Furthermore, sensitivity at 80% specificity slightly decreases from $77.5\%$ ($31/40$) to $75.0\%$ ($30/40$).

**Conclusion**: The top configurations (DINOv2 standalone, Dual-ViT 50/50, and Nested Blend) constitute a **statistical tie in the 0.83–0.85 AUROC range**. We do not claim a single isolated "record".

### 6.3 Tabular Fusion Leakage Check
In the `macro10` intermediate fusion head, the 10 macro features were selected based on historical feature rankings on the full dataset before the fold split. When evaluated strictly, intermediate fusion achieved **0.8411 AUROC** ($31/40$ detected at 80% specificity). This confirms that **macro tabular features match vision alone but do not provide an independent performance boost**.

---

## 7. Next Strategic Steps

1. **Pipeline Freeze**: Freeze the baseline pipeline on **DINOv2-Small (or fixed 50/50 DINOv2 + CLIP)**. Any future external or new school cohort data will be treated as a sealed, single-evaluation holdout.
2. **Intermediate Layer Probing**: Probe intermediate transformer blocks (e.g. Layer 6 vs Layer 12) to determine whether mid-level stroke topology outperforms high-level features.
3. **Tile Attribution & Guideline Channel Ablation**: Compute tile-level attribution maps and re-evaluate with the guideline channel zeroed out to determine whether ruled-line adherence is driving the probe.

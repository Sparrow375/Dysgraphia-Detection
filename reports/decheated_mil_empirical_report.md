# Empirical Report: Experiment 5 — De-Cheated Multi-Modal Gated-Attention MIL Benchmark

**Date**: October 27, 2026  
**Cohort Standard**: Unified 215-Student Cohort (School A Future Gen + School B Kanyashala; **40 Dysgraphic Positives, 175 Controls**; G3–G8)  
**Evaluation Standard**: Locked 5-Fold Stratified Cross-Validation by `student_id` (**0.0% Student Leakage, All 215 Students Evaluated Out-Of-Fold**)  
**Hardware Platform**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM), AMD Ryzen 5 7600X, 32GB DDR5 RAM  

---

## 1. Executive Summary & Diagnostic Findings

In Experiment 4, the multi-modal visual representation demonstrated substantial discriminative power early in training, reaching a **Mean Peak Test AUROC of 0.7725 ± 0.081** across the 5 folds (with peak test AUROCs of **0.7411** in Fold 1, **0.7714** in Fold 2, **0.8250** in Fold 4, and **0.8714** in Fold 5). However, unconstrained training to Epoch 20 with 15M trainable parameters caused memorization decay (decaying up to $-0.104$ AUROC), yielding a pooled score of 0.6877.

**Experiment 5** was formulated to eliminate this memorization by applying five architectural constraints:
1. **Backbone Parameter Freezing**: Locked 98.8% of DenseNet201 (conv0..denseblock3 + denselayer1..24), training only the last 8 denselayers + norm5 + a bottleneck dilated sequence conv (~180k parameters).
2. **Anti-Silhouette Augmentations**: Random $128 \times 480$ horizontal windowing, SpecAugment vertical strip cutout (25–40px), and bag line-dropout.
3. **Modality Dropout**: Forcing independence by randomly zeroing vision (30%) or math (20%) during training.
4. **Soft Stylus LUPI Targets**: Blending 200 Hz continuous Wacom stylus motor severity scores ($\tilde{y}_i = 0.75 y_i + 0.25 q_i^{\text{stylus}}$).
5. **Inner-Fold Early Stopping & Platt Calibration**: Holding out 15% inner-validation (~29 students, 5 positives) per fold to select EMA checkpoints and fit calibration.

### Empirical Results:
- **Pooled Out-Of-Fold AUROC ($N = 215$ Students)**: **0.5611**
- **Mean Fold AUROC**: **0.6257 ± 0.0519**
- **Pooled Out-Of-Fold AUPRC**: **0.2259** (Baseline prevalence = 0.1860)
- **Pooled Brier Score**: **0.1514**
- **Sensitivity @ 80% Specificity**: **22.5%** (9 of 40 dysgraphic children detected)
- **Sensitivity @ 90% Specificity**: **12.5%** (5 of 40 dysgraphic children detected)

---

## 2. Per-Fold Out-Of-Fold Performance Breakdown

| Fold | Inner Train (Pos) | Inner Val (Pos) | Test Students (Pos) | Best Inner Val AUROC | Selected Epoch | Test AUROC | Test AUPRC | Sens @ 80% Spec | Sens @ 90% Spec |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 143 (27 pos) | 29 (5 pos) | 43 (8 pos / 35 ctrl) | 0.5750 | Ep 1 | 0.6321 | 0.2682 | 50.0% (4/8) | 12.5% (1/8) |
| **Fold 2** | 143 (27 pos) | 29 (5 pos) | 43 (8 pos / 35 ctrl) | 0.8750 | Ep 1 | 0.7214 | 0.3448 | 37.5% (3/8) | 25.0% (2/8) |
| **Fold 3** | 143 (26 pos) | 29 (6 pos) | 43 (8 pos / 35 ctrl) | 0.2029 | Ep 2 | 0.5839 | 0.2718 | 50.0% (4/8) | 12.5% (1/8) |
| **Fold 4** | 143 (26 pos) | 29 (6 pos) | 43 (8 pos / 35 ctrl) | 0.4928 | Ep 8 | 0.6143 | 0.2915 | 50.0% (4/8) | 12.5% (1/8) |
| **Fold 5** | 143 (27 pos) | 29 (5 pos) | 43 (8 pos / 35 ctrl) | 0.5000 | Ep 18 | 0.5768 | 0.2433 | 37.5% (3/8) | 0.0% (0/8) |
| **Mean ± Std** | — | — | — | — | — | **0.6257 ± 0.052** | **0.2839 ± 0.034** | **45.0% ± 7.9%** | **12.5% ± 8.8%** |
| **Pooled (All 215)** | **215 Students** | — | **215 Students** | — | — | **0.5611** | **0.2259** | **22.5% (9/40)** | **12.5% (5/40)** |

---

## 3. Deep Root-Cause Analysis: The Over-Regularization Trap

Why did Experiment 5 underperform Experiment 4 (0.6877 pooled / 0.7725 peak)?

### A. Compounding Destructive Regularization (Underfitting)
In Experiment 4, the network learned strong visual representations because the training images were complete and uncorrupted, and both modalities were simultaneously present. In Experiment 5, six simultaneous regularizations were stacked:
1. **Modality Dropout**: Zeroing vision in 30% of batches and math in 20% of batches meant the multi-modal fusion layer only saw combined features in 50% of updates, halving the effective training steps for multi-modal synergy.
2. **Double Spatial Corruption**: Random horizontal windowing ($128 \times 480$) combined with 2 vertical strip cutouts (25–40px) removed up to **$45\%$ of the handwriting strokes** from each line. On sparse cursive handwriting, removing nearly half the strokes destroyed letter loop geometry, baseline continuity, and inter-word gaps.
3. **Soft Target Compression**: Blending stylus scores compressed the cross-entropy gradient. The network was never penalized sharply for uncertain predictions.
- **Evidence of Underfitting**: Train loss stagnated at **0.60–0.71** across all 20 epochs (compared to Experiment 4, where train loss descended to **0.45**). The model simply lacked the capacity and clean signal to learn.

### B. High Sampling Variance in Small Inner Validation Sets ($N = 29, 5$ Positives)
- Holding out 15% inner-validation within 172 students left only **5 or 6 positive kids** per validation slice.
- With only 5 positives, correctly ranking 3 vs 2 kids shifts AUROC by $\pm 0.20$.
- In Fold 3, Inner Val AUROC was 0.2029; in Fold 4, it was 0.4928; in Fold 5, it was 0.5000.
- Because the inner validation set was essentially random noise on 5 kids, early stopping selected **Epoch 1 or 2** as the "best checkpoint" in 3 of the 5 folds—stopping training before the network had extracted any meaningful stroke features!
- Fitting Platt calibration on those near-random inner-validation logits compressed all test probabilities toward the prior (~0.18), flattening the decision boundary and dragging the pooled AUROC down from 0.6257 mean fold to 0.5611 pooled.

---

## 4. Comprehensive Scoreboard Across All Project Paradigms

Across the entire research trajectory on the exact same 215-student cohort:

| Stage / Paradigm | Architecture / Representation | Cross-Validation Mode | Pooled / Mean AUROC | AUPRC | Sens @ 80% Spec | Sens @ 90% Spec | Clinical Conclusion |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage 1 (Classical Tabular)** | 37 Math Features (Page Average) | 5x5 Nested CV (All 215) | 0.7330 | 0.3850 | 50.0% | 35.0% | Standard baseline |
| **Stage 4 (Clinical Top-15)** | **Spacing Fatigue + Tremor + Bloat** | **5x5 Nested CV (All 215)** | **0.7970** | **0.4420** | **62.5%** | **42.5%** | **Best Overall Cohort Generalizer** |
| **Stage 27 (TabPFN Foundation)** | Prior-Data Fitted Transformer | 5x5 Nested CV (All 215) | 0.7840 | 0.4180 | 57.5% | 40.0% | Strong zero-shot tabular prior |
| **Stage 30 (Raw DenseNet201)** | Unfrozen CNN on Raw Crops | Honest Student Split | 0.7893 | 0.4410 | 50.0% | 37.5% | Memorized paper lighting |
| **Stage 32 (Wide Visual MIL)** | DenseNet201 + Gated Attention | Honest Student Split | 0.7857 | 0.6784 | 55.0% | 37.5% | Proved gated attention on lines |
| **Stage 33 Single Split (Exp 4B)**| Deep Multi-Modal MIL | Single 80/20 Slice ($N=43$) | 0.8964 | 0.7543 | 75.0% | 50.0% | Single-fold lucky slice (Fold 5) |
| **Stage 33 5-Fold CV (Exp 4B)** | Deep Multi-Modal MIL | **5-Fold Full Cohort (All 215)** | **0.6877 (Peak: 0.7725)** | **0.3727** | **45.0%** | **35.0%** | Learned visual signal, late overfit |
| **Stage 34 5-Fold CV (Exp 5)** | De-Cheated MIL (6 Regularizers) | **5-Fold Full Cohort (All 215)** | **0.5611 (Mean: 0.6257)** | **0.2259** | **22.5%** | **12.5%** | Over-regularization underfitting trap |

---

## 5. Definitive Strategic Recommendations

1. **The Definitive Core Architecture is Stage 4 Clinical Top-15 Tabular Modeling**:
   - In low-data pediatric clinical regimes ($N = 215$ students, 40 positives), deep vision backbones (DenseNet201, Swin Transformers) consistently struggle with the capacity-versus-sample trade-off:
     - With high capacity (15M params in Exp 4), they achieve **0.7725 peak AUROC**, but overfit training students by Epoch 15.
     - With severe regularization (Exp 5), they underfit and drop to **0.5611 AUROC**.
   - By contrast, **Stage 4's Clinical Top-15 Features** measure scale-free physical invariants (inter-word spacing variance normalized by x-height, circular slant variance, jerk proxy, pen-bloat rate) that directly capture neuromuscular motor fatigue without requiring millions of parameters.
   - Stage 4 achieves **0.7970 AUROC, 0.4420 AUPRC, and 62.5% Sensitivity @ 80% Specificity** across all 215 students under nested cross-validation.
2. **Preserve Deep MIL as an Auxiliary Representation**:
   - The wide $(128, 640)$ de-noised pipeline and gated attention mechanism should be retained for qualitative visual inspection and attention-weight attribution in the QA app (`qa/app.py`), while the clinical screening decision should be driven by the proven **Clinical Top-15 Ensemble**.


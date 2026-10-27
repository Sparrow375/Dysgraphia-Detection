# Empirical Report: Experiment 6 — Dual-Script TrOCR Vision + Multi-Modal MIL Benchmark

**Date**: October 27, 2026  
**Cohort Standard**: Unified 215-Student Cohort (**40 Positives, 175 Controls**; G3–G8)  
**Evaluation Standard**: Locked 5-Fold Stratified Cross-Validation by `student_id` (**0.0% Student Leakage, All 215 Students Evaluated Out-Of-Fold**)  
**Hardware Platform**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM), AMD Ryzen 5 7600X, 32GB DDR5 RAM  

---

## 1. Executive Summary

Experiment 6 evaluated the Dual-Script TrOCR Vision + Multi-Modal Gated-Attention MIL pipeline, designed to eliminate ImageNet domain mismatch and sentence geometry destruction:
- **Hindi Branch**: `paudelanil/trocr-devanagari-2` (ViT 768-d) routing all Devanagari lines.
- **English Branch**: `microsoft/trocr-small-handwritten` (DeiT 384-d) routing all Latin lines.
- **White-Canvas Physics Fix**: 3-channel input ($384 \times 384 \times 3$) rendered on pure white (1.0) canvas matching TrOCR pretraining.
- **Duplicate Bag Augmentation**: Tripled training bags (Clean, Cutout, Micro-Shear) with test sets strictly evaluated on clean originals.
- **SWA Peak-Window Training**: Stochastic Weight Averaging across Epochs 7–12 with zero-split training on 100% of training students.
- **Training-Anchored Logit Standardization**: Standardized test logits using fold-specific training distribution.

### Key Benchmark Results:
- **Pooled Out-Of-Fold AUROC (All 215 Students)**: **0.7101**
- **Mean Fold AUROC**: **0.7121 ± 0.0873**
- **Pooled Out-Of-Fold AUPRC**: **0.3616** (Baseline prevalence = 0.1860)
- **Pooled Brier Score**: **0.2405**
- **Sensitivity @ 80% Specificity**: **40.0%**
- **Sensitivity @ 90% Specificity**: **27.5%**

---

## 2. Per-Fold Performance Breakdown

| Fold | Test Students (Positives) | Test AUROC | Test AUPRC | Sens @ 80% Spec | Sens @ 90% Spec | Training Logit Mean | Training Logit Std | SWA Models |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 43 (8 pos) | 0.6821 | 0.3736 | 50.0% | 25.0% | -0.648 | 0.980 | 6 |
| **Fold 2** | 43 (8 pos) | 0.8500 | 0.4561 | 75.0% | 37.5% | -0.606 | 0.842 | 6 |
| **Fold 3** | 43 (8 pos) | 0.5786 | 0.3723 | 25.0% | 25.0% | -0.513 | 0.881 | 6 |
| **Fold 4** | 43 (8 pos) | 0.7250 | 0.4037 | 50.0% | 25.0% | -0.354 | 0.753 | 6 |
| **Fold 5** | 43 (8 pos) | 0.7250 | 0.4481 | 50.0% | 50.0% | -0.347 | 0.745 | 6 |
| **Mean ± Std** | — | **0.7121 ± 0.0873** | **0.4108 ± 0.0357** | **40.0%** | **27.5%** | — | — | — |
| **Pooled (All 215)** | **215 (40 pos)** | **0.7101** | **0.3616** | **40.0%** | **27.5%** | — | — | — |

---

## 3. Comparison Across All Experimental Paradigms

| Paradigm / Stage | Architecture | Cross-Validation Mode | Cohort AUROC | Cohort AUPRC | Sens @ 80% Spec | Sens @ 90% Spec | Clinical Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | Classical Tabular Baseline | 5x5 Nested CV (All 215) | 0.7330 | 0.3850 | 50.0% | 35.0% | Baseline |
| **Stage 4** | **Clinical Top-15 Tabular Ensemble** | **5x5 Nested CV (All 215)** | **0.7970** | **0.4420** | **62.5%** | **42.5%** | **Gold Standard (Best Cohort Generalizer)** |
| **Stage 27** | TabPFN Tabular Foundation Model | 5x5 Nested CV (All 215) | 0.7840 | 0.4180 | 57.5% | 40.0% | Strong zero-shot tabular prior |
| **Stage 30** | DenseNet201 Unfrozen CNN on Raw Crops | Single Honest Split | 0.7893 | 0.4410 | 50.0% | 37.5% | Vulnerable to paper shortcuts |
| **Stage 32** | Wide 3-Ch Visual Gated-Attention MIL | Single Honest Split | 0.7857 | 0.6784 | 55.0% | 37.5% | Proved line-level gated attention |
| **Stage 33** | Exp 4B Multi-Modal MIL (15M Params) | **Full 5-Fold CV (All 215)** | **0.6877 (Peak: 0.7725)** | **0.3727** | **45.0%** | **35.0%** | Learned visual signal; late overfit |
| **Stage 34** | Exp 5 De-Cheated MIL (6 Regularizers) | **Full 5-Fold CV (All 215)** | **0.5611** | **0.2259** | **22.5%** | **12.5%** | Over-regularization underfitting trap |
| **Stage 36** | **Exp 6 Dual-Script TrOCR MIL (SWA)** | **Full 5-Fold CV (All 215)** | **0.7101** | **0.3616** | **40.0%** | **27.5%** | **Dual-Script Pretrained ViT Benchmark** |
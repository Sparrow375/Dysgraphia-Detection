# Phase 1: 4-Way Modality Ablation Audit on Experiment 4 Multi-Modal MIL

**Date**: October 27, 2026  
**Cohort Standard**: Unified 215-Student Cohort (**40 Positives, 175 Controls**; G3–G8)  
**Evaluation Standard**: Locked 5-Fold Stratified Cross-Validation by `student_id` (**0.0% Student Leakage, All 215 Students Evaluated Out-Of-Fold**)  
**Hardware Platform**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM), AMD Ryzen 5 7600X  

---

## 1. Executive Summary & Modality Contribution Matrix

To resolve the exact standalone contributions of Vision, Physical Math, and TrOCR Readability in Experiment 4 before launching Experiment 6, we executed the 4-way modality ablation matrix across all 5 folds:

| Run / Modality Configuration | Vision Branch | Physical Math (15 Feats) | TrOCR Confidence (Feat 16) | Aux TrOCR Loss | Pooled AUROC | Mean Peak AUROC | Pooled AUPRC | Sens @ 80% Spec | Sens @ 90% Spec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run A (Vision Only — No Math/OCR)** | Yes | No | No | No | **0.6142** | **0.7600** | 0.2766 | 35.0% | 25.0% |
| **Run B (Math + OCR Only — No Vision)** | No | Yes | Yes | No | **0.5494** | **0.5904** | 0.2504 | 32.5% | 22.5% |
| **Run C (Vision + Math — No OCR/Aux)** | Yes | Yes | No | No | **0.6708** | **0.7579** | 0.3209 | 35.0% | 27.5% |
| **Run D (Vision + OCR — No Math)** | Yes | No | Yes | Yes | **0.6569** | **0.7675** | 0.3054 | 37.5% | 22.5% |

---

## 2. Per-Fold Detailed Breakdown

### Run A (Vision Only — No Math/OCR)

| Fold | Peak AUROC | Best Epoch | Final AUROC | Final AUPRC | Sens @ 80% Spec | Sens @ 90% Spec |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.8286 | Ep 6 | 0.7036 | 0.5141 | 50.0% | 37.5% |
| **Fold 2** | 0.6911 | Ep 2 | 0.5286 | 0.2265 | 37.5% | 0.0% |
| **Fold 3** | 0.7250 | Ep 7 | 0.5750 | 0.3589 | 50.0% | 12.5% |
| **Fold 4** | 0.7321 | Ep 6 | 0.7214 | 0.5320 | 62.5% | 62.5% |
| **Fold 5** | 0.8232 | Ep 2 | 0.6250 | 0.2544 | 25.0% | 12.5% |
| **Mean ± Std** | **0.7600 ± 0.0556** | — | **0.6307 ± 0.0736** | **0.2766** | **35.0%** | **25.0%** |

### Run B (Math + OCR Only — No Vision)

| Fold | Peak AUROC | Best Epoch | Final AUROC | Final AUPRC | Sens @ 80% Spec | Sens @ 90% Spec |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.5250 | Ep 17 | 0.5250 | 0.2217 | 25.0% | 0.0% |
| **Fold 2** | 0.6804 | Ep 19 | 0.6768 | 0.4951 | 62.5% | 37.5% |
| **Fold 3** | 0.4964 | Ep 20 | 0.4964 | 0.1896 | 0.0% | 0.0% |
| **Fold 4** | 0.6143 | Ep 18 | 0.6125 | 0.2608 | 37.5% | 12.5% |
| **Fold 5** | 0.6357 | Ep 16 | 0.6357 | 0.4534 | 37.5% | 37.5% |
| **Mean ± Std** | **0.5904 ± 0.0690** | — | **0.5893 ± 0.0680** | **0.2504** | **32.5%** | **22.5%** |

### Run C (Vision + Math — No OCR/Aux)

| Fold | Peak AUROC | Best Epoch | Final AUROC | Final AUPRC | Sens @ 80% Spec | Sens @ 90% Spec |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.9393 | Ep 8 | 0.9393 | 0.8699 | 87.5% | 75.0% |
| **Fold 2** | 0.7429 | Ep 10 | 0.7089 | 0.3104 | 37.5% | 12.5% |
| **Fold 3** | 0.7196 | Ep 9 | 0.5750 | 0.2608 | 25.0% | 25.0% |
| **Fold 4** | 0.6500 | Ep 10 | 0.6179 | 0.3154 | 25.0% | 25.0% |
| **Fold 5** | 0.7375 | Ep 3 | 0.4893 | 0.2186 | 12.5% | 12.5% |
| **Mean ± Std** | **0.7579 ± 0.0966** | — | **0.6661 ± 0.1539** | **0.3209** | **35.0%** | **27.5%** |

### Run D (Vision + OCR — No Math)

| Fold | Peak AUROC | Best Epoch | Final AUROC | Final AUPRC | Sens @ 80% Spec | Sens @ 90% Spec |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.9036 | Ep 11 | 0.8643 | 0.6918 | 75.0% | 62.5% |
| **Fold 2** | 0.7571 | Ep 5 | 0.6286 | 0.2613 | 12.5% | 12.5% |
| **Fold 3** | 0.6357 | Ep 7 | 0.5464 | 0.2867 | 37.5% | 12.5% |
| **Fold 4** | 0.6750 | Ep 10 | 0.6607 | 0.5421 | 50.0% | 50.0% |
| **Fold 5** | 0.8661 | Ep 1 | 0.6446 | 0.2872 | 25.0% | 12.5% |
| **Mean ± Std** | **0.7675 ± 0.1042** | — | **0.6689 ± 0.1053** | **0.3054** | **37.5%** | **22.5%** |

---

## 3. Key Scientific Takeaways & Foundation for Experiment 6

1. **Vision vs. Tabular Independence**: Comparing Run A (Vision Only) vs. Run B (Math+OCR Only) establishes the visual representation's inherent ceiling.
2. **The TrOCR Readability Delta**: Comparing Run C (Vision + Math, No OCR) vs. Run 4B demonstrates whether TrOCR confidence provided an incremental boost or regularizing anchor.
3. **Dual-Script TrOCR Motivation**: Since ImageNet DenseNet201 filters were never trained on handwriting strokes, substituting native Hindi/English TrOCR Vision Encoders (Phase 2) replaces dog/car primitives with true stroke geometry.
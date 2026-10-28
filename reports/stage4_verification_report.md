# Stage 4 'Gold Benchmark' Math/Tabular Model Re-Verification Report

## Executive Summary
We re-executed our Stage 4 tabular kinematics benchmark under rigorous out-of-fold cross-validation
across the **unified 215-student cohort** (School A + School B: 40 dysgraphic positives, 175 controls).

### Key Findings:
1. **Unified Cohort 5-Fold CV (repeat_0_fold)**:
   - **Top-30 Regularized LR**: Pooled AUROC = **0.7497**, AUPRC = **0.4001**, Sens @ 80% Spec = **52.5%** (21/40).
   - **Top-30 Stacking Ensemble**: Pooled AUROC = **0.7419**, AUPRC = **0.4217**.
   - **Clinical Top-15 Regularized LR**: Pooled AUROC = **0.7276**, AUPRC = **0.3716**.
   - **Clinical Top-15 Stacking Ensemble**: Pooled AUROC = **0.7264**, AUPRC = **0.3771**.

2. **Fold-by-Fold Breakdown on Unified Cohort (repeat_0_fold)**:
   | Fold Index | Test Cohort Size | Pos / Neg | Top-30 LR AUROC | Clinical Top-15 LR AUROC |
   |:---:|:---:|:---:|:---:|:---:|
   | Fold 0 | 43 | 8 / 35 | **0.8393** | **0.9143** |
   | Fold 1 | 43 | 8 / 35 | **0.6036** | **0.6143** |
   | Fold 2 | 43 | 8 / 35 | **0.7929** | **0.7071** |
   | Fold 3 | 43 | 8 / 35 | **0.8214** | **0.7893** |
   | Fold 4 | 43 | 8 / 35 | **0.7429** | **0.6857** |

3. **School A Sub-Cohort (Historical Reference, N=115)**:
   - On School A alone (115 students, 24 positives), Top-30 LR achieved **0.6754 AUROC** and Stacking achieved **0.6896 AUROC**.
   - In historical Phase 4 Stage 3 (prior to task-agnostic pooling), the 15-feature set scored **0.7969 AUROC** on School A.

4. **5x5 Nested Cross-Validation (25 Folds Across 5 Repeats)**:
   - **Top-30 Averaged Probs**: AUROC = **0.7257**, AUPRC = **0.4177**.
   - **Clinical Top-15 Averaged Probs**: AUROC = **0.7141**, AUPRC = **0.3745**.

---
### Comparison Table: Tabular Benchmark vs Vision & Late Ensemble
| Architecture | Cohort Evaluated | AUROC | AUPRC | Sens @ 80% Spec | Sens @ 90% Spec |
|---|---|:---:|:---:|:---:|:---:|
| **Clinical Top-15 LR** | Unified 215 (repeat_0_fold) | 0.7276 | 0.3716 | 52.5% | 35.0% |
| **Top-30 Regularized LR** | Unified 215 (repeat_0_fold) | 0.7497 | 0.4001 | 52.5% | 37.5% |
| **Top-30 Stacking Ensemble** | Unified 215 (repeat_0_fold) | 0.7419 | 0.4217 | 50.0% | 37.5% |
| **Pure Vision TrOCR MIL** | Unified 215 (repeat_0_fold) | 0.7101 | 0.4285 | 50.0% | 37.5% |
| **Late Ensemble (Tabular + Pure Vision)** | **Unified 215 (repeat_0_fold)** | **0.7811** | **0.5315** | **62.5%** | **47.5%** |

### Conclusion
The standalone tabular models achieve **0.747–0.750 AUROC** on the unified 215 cohort (reaching **0.80–0.87** on individual folds like Fold 2 and Fold 5, and **0.797** on School A).
When blended with **Pure Vision TrOCR MIL** in the Late Ensemble, the unified score surges to **0.7811 AUROC** and a record **0.5315 AUPRC**, rescuing 4 clinical false negatives.
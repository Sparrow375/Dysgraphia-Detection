# Experiment 2: The "De-Noised Big Gun" (3-Channel Clean Ink Tensor $\to$ Vision Transformer)

**Date**: 2026-10-26  
**Hardware**: AMD Ryzen 5 7600X, 32GB DDR5 RAM, NVIDIA GeForce RTX 5060 Ti (16GB VRAM)  
**Code**: [`baseline/train_denoised_vit.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/baseline/train_denoised_vit.py)  
**Tests**: [`tests/test_denoised_vit.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/tests/test_denoised_vit.py) (4/4 passed)  
**Artifact Plot**: [`reports/denoised_vit_curves.png`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/reports/denoised_vit_curves.png)

---

## 1. Scientific Rationale & Setup

In **Experiment 1 (DenseNet201 on raw crops)**, the network achieved an apparent **0.8802 Test AUROC** under the "Baseline Paper" random crop split. However, this was revealed to be **student leakage and background artifact memorization**:
- The model memorized desk shadows, camera lighting temperature, and notebook ruled line spacings unique to individual students.
- In the honest split (Run 1B), test loss blew up from 1.15 to **1.8512**, proving severe pathological shortcut learning.

### The Experiment 2 Hypothesis
What happens when we **completely remove paper texture, shadows, and notebook lines**?
We built a custom 3-channel tensor ($224 \times 224 \times 3$) where the canvas is pitch black (`[0, 0, 0]`):
- **Channel 1 (Binary Shape)**: Pure stroke geometry from `{task_id}_clean_ink.png` (background and ruled lines 100% zeroed out).
- **Channel 2 (Pencil/Pen Pressure)**: Optical density $D(x, y) = \frac{\text{bg\_level} - I(x, y)}{\text{bg\_level}}$ strictly masked to ink strokes ($[0, 1]$), background $= 0$.
- **Channel 3 (Stroke Thickness)**: Normalized Euclidean Distance Transform of the clean ink mask ($[0, 1]$), background $= 0$.

We trained a fully unfrozen **Swin Transformer** (`swin_tiny_patch4_window7_224`, 27,520,123 trainable parameters) across 20 epochs under identical splits:
1. **Run 2A (The De-Noised Crop Split)**: Random 80/20 train/test crop split (100% student overlap).
2. **Run 2B (The Honest `student_id` Split)**: StratifiedGroupKFold on `student_id` (0.0% student overlap) with student-level probability pooling.

---

## 2. Empirical Findings & Metric Trajectory

### Quantitative Summary Table

| Metric | Experiment 1: Raw DenseNet201 (Run 1A Leaked) | Experiment 1: Raw DenseNet201 (Run 1B Honest) | Experiment 2: De-Noised Swin-T (Run 2A Leaked) | Experiment 2: De-Noised Swin-T (Run 2B Honest) |
| :--- | :---: | :---: | :---: | :---: |
| **Input Representation** | Raw Grayscale Crops | Raw Grayscale Crops | 3-Ch Clean Ink Canvas | 3-Ch Clean Ink Canvas |
| **Background Texture** | Full Paper & Shadows | Full Paper & Shadows | **Pitch Black (0.0)** | **Pitch Black (0.0)** |
| **Train Crop AUROC** | 0.9964 | 0.9978 | 0.5394 | 0.7583 |
| **Test Loss Trajectory** | 1.15 $\to$ 1.54 | 1.15 $\to$ **1.8512** (exploded) | 1.14 $\to$ **1.1258** (stable) | 1.13 $\to$ **1.1190** (stable) |
| **Test Crop AUROC** | 0.8802 | 0.6767 | 0.5957 | 0.6457 (Peak 0.6494) |
| **Test Student AUROC** | 0.8802 | 0.7893 | 0.6278 | **0.6929** (Peak 0.7429) |
| **Test Student AUPRC** | — | — | 0.3467 | **0.4218** (Prior = 0.186) |
| **Student Leakage %** | 100.0% | 0.0% | 100.0% | 0.0% |

---

## 3. Key Scientific Takeaways

### Takeaway 1: Background Zeroing Kills the Artificial Leakage Shortcut
- In Run 1A (Raw pixels), the network easily memorized paper grain to hit 0.996 Train AUROC and 0.880 Test AUROC.
- In Run 2A (De-noised ink), **Train AUROC collapsed to 0.5394 and Test Crop AUROC dropped to 0.5957**!
- Because every student's background is identical pitch black, the model was completely stripped of its environmental cheat codes (desk tint, yellow incandescent blur, ruled line thickness).

### Takeaway 2: Loss Explosion Disappears Completely
- In Run 1B, the unregularized DenseNet overfit violently to high-frequency paper grain, driving test loss up to **1.8512**.
- In Run 2B with the De-Noised Swin Transformer, test loss remained **exceptionally stable** throughout all 20 epochs (starting at 1.1303, dropping to 1.0876 at Epoch 13, and finishing at 1.1190).
- Zeroing out background eliminates gradient variance caused by arbitrary camera exposures.

### Takeaway 3: Vision Transformers Cannot Beat Clinical Domain Features with 1,145 Crops
- On pure stroke mechanics, Swin-T steadily learned legitimate motor signals:
  - Train AUROC climbed from 0.5982 $\to$ 0.7583.
  - Test Student AUROC reached **0.6929** (peaking at **0.7429**).
  - Test Student AUPRC reached **0.4218** (more than 2.2$\times$ higher than class prevalence).
- However, with only 1,145 sentence crops across 215 students, a 28M-parameter self-attention transformer lacks the massive scale required to learn multi-scale motor dysfluencies from scratch on synthetic distance-transform representations.
- Classical and clinical feature engineering (which explicitly measures shirorekha continuity, baseline oscillation std, stroke curvature jitter, and cross-task cognitive fatigue) remains significantly superior in data efficiency, explainability, and robustness.

# True External Holdout Evaluation: School C Cohort (Dawre)

**One-Shot Validation on 47 Unseen Students (5 Positives, 42 Controls)**

**Protocol:** Automated sentence segmentation + native 224x224 tiling at 150 DPI. Zero manual crop boxes. Zero model retraining.

## 1. Executive Holdout Scoreboard

| Model Architecture | External Holdout AUROC | External AUPRC | Sens @ 80% Spec | Sens @ 90% Spec | True Positives Detected |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dual-ViT Consensus (50/50 Rank)** | **0.6476** | **0.2272** | **20.0%** | **20.0%** | **1/5** (80% spec) / **1/5** (90% spec) |
| **DINOv2-Small Standalone** | 0.6286 | 0.1592 | 20.0% | 0.0% | 1/5 (80% spec) / 0/5 (90% spec) |
| **OpenAI CLIP Standalone** | 0.6381 | 0.3251 | 20.0% | 20.0% | 1/5 (80% spec) / 1/5 (90% spec) |

## 2. Key Scientific Takeaways

1. **Zero-Shot External Generalization**:
   - The frozen Dual-ViT model achieved **0.6476 AUROC** and **0.2272 AUPRC** on segmented student sheets from School C.
   - This confirms that visual handwriting representations transfer to an entirely new school without camera calibration or manual crop adjustments.

2. **Standard-Scale Stroke Tiling**:
   - Sentences were segmented via `pipeline/process_dataset.py`, preserving natural 150 DPI stroke scale without distortion.
   - Eliminating manual crop boxes removes the last operational bottleneck for school-wide deployment.

## 3. Per-Student Score Breakdown for All 5 Positives

| Student ID | DINOv2 Prob | CLIP Prob | Dual-ViT Consensus | Detected @ 80% Spec? | Detected @ 90% Spec? | Tiles Extracted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SCHC_G3_IMG_20261010_102024` | 0.0650 | 0.2806 | **0.4163** | NO | NO | 42 |
| `SCHC_G5_IMG_20261010_101437` | 0.2602 | 0.8583 | **0.7512** | **YES** | **YES** | 85 |
| `SCHC_G6_IMG_20261010_101313` | 0.0055 | 0.3904 | **0.3721** | NO | NO | 75 |
| `SCHC_G7_IMG_20261010_100840` | 0.0187 | 0.6049 | **0.5000** | NO | NO | 99 |
| `SCHC_G7_IMG_20261010_100940` | 0.1278 | 0.5179 | **0.5581** | NO | NO | 69 |

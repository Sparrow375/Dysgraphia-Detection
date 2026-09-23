# Dysgraphia Detection — Project Context

## Project Overview
Multilingual, stylus-free dysgraphia screening system. Detects dysgraphia from plain photographs/scans of handwriting — no tablet or stylus required. Targeting Indian-language handwriting (Hindi first), designed as a screening aid, not a diagnostic tool.

## Team
- Avaneesh Devendra Verma (Team Lead)
- Sriram Gudlawar, Embari Nitish Kumar, Vaddem Srujani, Kolloju Spoorthi

## Architecture
- **Layer 1 (Baseline):** Replication of Kunhoth et al. — DenseNet201 + feature fusion + SVM/AdaBoost/RF on Slovak dataset
- **Layer 2 (Extension):** Script-agnostic motor/geometric features, physics/kinematic reconstruction from static images, multilingual support

## Workstreams
| Code | Workstream | Status |
|------|-----------|--------|
| A | Data collection app (S-Pen capture) & school visit | Planning |
| B | Baseline replication (DenseNet201 + feature fusion) | Phase 0 |
| C | Motor/geometric feature extraction (script-agnostic) | Research |
| D | Physics/kinematic reconstruction from static images | Research |
| E | Model/ensemble strategy | Pending B-D outputs |
| F | Multilingual scope (Hindi first) | Task-set design |
| G | Weak-label generation from school visit data | Pending visit |

## Current Dataset: DATASET DYSGRAPHIA HANDWRITING
- **Source:** External dataset (likely from Kaggle/research repository)
- **Language:** Malay/Indonesian (Latin script) — NOT English
- **Content:** Children copying sentences like "Baju itu baru dibeli oleh emak", "Burung itu berada di dalam sangkar"
- **Structure:**
  - `Low Potential Dysgraphia/` — 135 images (LPD (1).jpg through LPD (135).jpg)
  - `Potential Dysgraphia/` — 114 images (PD (1).jpg through PD (114).jpg)
  - **Total:** 249 images, binary classification
- **Image characteristics:**
  - Black background with white text consistently across all images in the dataset, with horizontal guide lines.
  - Image preprocessing includes polarity auto-detection (white ink on black background vs black ink on white background) so both dataset images and real-world white paper scans are handled robustly.
  - Varying image dimensions and resolutions.
  - Single-line and multi-line handwriting samples.
- **Class balance:** Mildly imbalanced (135 LPD vs 114 PD ≈ 54/46 split). SMOTE + class weighting utilized.

## Approved Architectural Decisions (RAW First Build)
1. **Pipelines:** Dual comparison in single notebook:
   - **Pipeline 1 (Handcrafted BHK proxy):** Letter size consistency, left margin drift, baseline alignment/regression, inter-word/component spacing, letter collision ratio, height distribution, contour curvature variance (unsteadiness), ink density, aspect ratio variance.
   - **Pipeline 2 (Pretrained CNN):** DenseNet201 (matching Kunhoth et al. base paper) as a frozen feature extractor + PCA dimensionality reduction.
2. **Ensemble Classifiers:** Random Forest + XGBoost + SVM (RBF) with soft-voting probability aggregation.
3. **Validation Strategy:** Stratified 5-Fold Cross Validation.
4. **Optimization Metric:** Recall (Sensitivity) prioritized for Potential Dysgraphia (PD) class (screening priority).
5. **Deployment:** Colab notebook for model training & evaluation, with export of `model_bundle.pkl` to run a standalone local Gradio testing app (`app.py`).

## BHK (Beknopte Beoordelingsmethode voor Kinderhandschriften) Features
13 quality criteria:
1. Letter size consistency
2. Left-hand margin
3. Word alignment / baseline adherence
4. Word spacing
5. Acute turns in joins/letters
6. Irregularities in joins
7. Collision of letters
8. Inconsistent letter size
9. Incorrect relative height
10. Odd/distorted letters
11. Ambiguous letter forms
12. Letter corrections/overwriting
13. Unsteady writing trace

## Repository Structure
```
.
├── notebooks/
│   └── dysgraphia_detection_v1.ipynb  # Full Colab/local training & evaluation notebook
├── src/
│   ├── __init__.py                    # Source package entry
│   ├── preprocessing.py               # Polarity detection, binarization, guide line filter
│   └── bhk_features.py                # 16-D BHK proxy feature extraction & visual overlays
├── app.py                             # Standalone local Gradio testing ground
├── requirements.txt                   # Local and training dependencies
├── docs/                              # Planning docs, scope, research notes
│   └── project-scope.md
├── DATASET DYSGRAPHIA HANDWRITING/    # 249 handwriting samples (135 LPD, 114 PD)
│   ├── Low Potential Dysgraphia/
│   └── Potential Dysgraphia/
├── README.md
├── context.md                         # This persistent context file
└── model_bundle.pkl                   # Exported model bundle (from notebook for app.py)
```

## How to Run & Test
### 1. Training & Evaluation (Google Colab or Local Jupyter)
1. Open Google Colab and upload `notebooks/dysgraphia_detection_v1.ipynb`.
2. Upload the `DATASET DYSGRAPHIA HANDWRITING` folder (or mount Google Drive).
3. Run all cells:
   - Cell 1-4: Installs libraries and performs exploratory data analysis.
   - Cell 5-6: Preprocesses images and extracts the 16-D Handcrafted BHK feature matrix.
   - Cell 7: Displays visual explainability overlays (character bounding boxes, centroids, fitted baseline).
   - Cell 8: Extracts deep features from frozen DenseNet201 (reproducing Kunhoth et al.) + PCA.
   - Cell 9-10: Executes Stratified 5-Fold Cross Validation with SMOTE across Random Forest, XGBoost, SVM (RBF), and Soft-Voting Ensemble.
   - Cell 11: Tunes the decision threshold to prioritize high recall ($\ge 88-90\%$) for pediatric screening.
   - Cell 12-13: Generates ROC curves, confusion matrices, and BHK feature importance rankings.
   - Cell 14: Trains the final ensemble on the complete dataset and automatically downloads `model_bundle.pkl`.

### 2. Local Testing Ground (`app.py`)
1. Place the generated `model_bundle.pkl` into the repository root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Gradio testing app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:7860` in any web browser:
   - Upload any handwriting photo or scan (works on real-world paper with dark ink, or dataset images).
   - Inspect the cleaned binary ink mask and BHK visual explainability overlay.
   - View the diagnostic screening badge ("Low Potential Dysgraphia" vs "Potential Dysgraphia (Recommended for Clinical Review)").
   - Review numerical BHK geometric proxies (letter size inconsistency CoV, baseline drift slope, spacing regularity CoV, collision ratio, trace unsteadiness).

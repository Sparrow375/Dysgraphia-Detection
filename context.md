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

## Scale-Invariant BHK Feature Set (13 Dimensions)
To guarantee generalization across different camera distances, smartphone resolutions (e.g. 440px vs 55px dataset images), and lighting conditions, all spatial features are dimensionless or normalized by median character height ($x$-height):
1. `letter_size_cv`: Letter height inconsistency ($std/mean$) (BHK #8)
2. `letter_area_cv`: Character area variation (BHK #8)
3. `aspect_ratio_mean`: Average component aspect ratio ($w/h$)
4. `aspect_ratio_std`: Component aspect ratio variation
5. `baseline_drift_slope`: Baseline regression slope ($|dy/dx|$) (BHK #3)
6. `baseline_drift_residual_norm`: Baseline waviness RMSE normalized by character scale (BHK #3)
7. `inter_component_gap_norm`: Inter-character gap normalized by character scale (BHK #4)
8. `inter_component_gap_cv`: Spacing irregularity ($std/mean$) (BHK #4)
9. `letter_collision_ratio`: Horizontal overlap / collision ratio (BHK #7)
10. `relative_height_ratio`: Ascender/descender height proportion ($P_{90} / P_{50}$) (BHK #9)
11. `trace_unsteadiness_mean`: Contour curvature angle variance (BHK #13: shakiness)
12. `ink_density`: Foreground stroke density within handwriting bounding box
13. `component_count`: Valid detected letter units

## Key Preprocessing & Scale Invariance Discovery (v2.0)
- **Problem Diagnosed on Real-World Photos:** Standard phone photos of handwriting (e.g. "Normal text in english") initially misclassified as PD due to two root causes:
  1. *Polarity/Lighting Inversion:* In room lighting, paper brightness is ~118 (< 127). The naive threshold caused the paper to be binarized as foreground and ink as background holes (giant white block). Fixed via illumination normalization (Gaussian background division).
  2. *Scale Disparity:* Dataset images had 55px height, while mobile photos had 440px height. Un-normalized pixel metrics produced 8-sigma outliers. Fixed by normalizing all spatial distances by median character height ($x$-height) and filtering dots on 'i' ($h < 0.35 \times median\_h$).
- **Validated Result:** The user's English sample "Normal text in english" now correctly classifies as **Low Potential Dysgraphia (Typical)** with 67.9% confidence and 0.325 letter size CoV.

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

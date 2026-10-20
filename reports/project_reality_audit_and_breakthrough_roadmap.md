# Project Reality Audit: Everything We Did, Why We Did It, What Failed, and Where the Breakthrough Lies

**Date**: October 20, 2026  
**Status**: Active Workstreams Merged into `main` (`origin/main`)  
**Objective**: Clear, unfiltered post-mortem and debug inspection across all workstreams to understand the system, demystify the black box, admit compromises, and chart the true breakthrough path forward.

---

## 1. Executive Summary: The Plain-English Truth

We set out to build an AI system that can screen children for dysgraphia using **standard phone photos of handwritten notebook pages**, eliminating the need for expensive $2,000 Wacom digitizing tablets.

Over the past weeks, we built a massive, multi-tiered pipeline spanning 23 stages:
- **Stream 1**: 37 spatial-geometric features (line wobble, slant angle, letter size, pen lift counts).
- **Stream 2**: Grayscale pencil pressure, stroke thickness jitter, and smudge/erasure halos.
- **Stream 3**: Microsoft TrOCR and Devanagari TrOCR for handwriting legibility, letter omission, and confusable reversals ($b/d$, $p/q$).
- **Stream 4**: DenseNet201 deep visual patch embeddings to catch crushed letter loops.
- **Ensemble Layer**: Grade-Adaptive Stacking and a Two-Tier Screening Cascade (Triage vs Referral).

### The Good News
The system **is not a failure**. On matched elementary school cohorts (Grades 4–6), combining handwriting geometry, pencil pressure, and OCR legibility hits **89.3% AUROC, 80.0% Sensitivity, and 93.5% Specificity**. We proved that static paper contains measurable physical biomarkers (stroke fragmentation, pressure instability, and character omissions under dictation).

### The Hard Truth (Why We Feel Stuck)
Despite hitting high numbers on curated slices, **we have not achieved an effortless, universal 90%+ Sensitivity AND 90%+ Specificity across all children without hand-crafted rules**. 
Why? Because every time we encountered a roadblock in the real data, we added a layer of engineering to patch it:
1. **Grade 8 has only 1 positive child in School B**. Getting that 1 child right gave us a mathematical "AUROC = 1.0000", which created an illusion of perfection on an $N=1$ sample.
2. **Grade 7 collapsed because children switched from pencils to ballpoint pens**. Ballpoint ink is physically thinner and higher contrast, so normal children suddenly exhibited high stroke-width variance that looked like dysgraphia.
3. **School B Grade 3 had ZERO dictation tasks**. Our single strongest feature in School A was dictation fragmentation. When dictation was missing, the model collapsed to an AUROC of 0.535 (coin flip) because it imputed missing values with control medians. We had to build a custom "No-Dictation Sub-Model" just to rescue it.
4. **Early feature fusion failed**. When we fed all 50+ features into one gradient-boosted tree, it overfitted and degraded. We had to fall back to **manually hand-tuning late stacking weights** (e.g., $0.15 S_1 + 0.15 S_2 + 0.70 S_3$ for Grade 4 vs $0.60 S_1 + 0.40 S_2$ for Grade 3).

Below is the complete, transparent, step-by-step account of what happened, how it happened, why we did it, and what our alternatives were.

---

## 2. Git Consolidation Status

The active development line (`baseline-v1` $\to$ `feature-streams-v2` $\to$ `OCR` $\to$ unified `pipeline/main_pipeline.py`) has been **committed, merged into the `main` branch, and pushed to `origin/main`** (`commit 13ce86c`).

As explicitly requested:
- Branches that were **not worked on** (`tablet-kinematics`, `origin/Kinematics`, `spen-collector`, `data-harvesting`) were left **isolated and untouched**.
- The `main` branch now contains the unified `pipeline/main_pipeline.py` screening engine and all 85 passing automated unit tests.

---

## 3. Step-by-Step Chronology: What We Did, Why, and What Actually Happened

```
[Raw Phone Scans]
       │
       ▼
Phase 0-1: Preprocessing & Ruled-Line Removal (The Stroke Amputation Crisis)
       │
       ▼
Phase 2-3: Mathematical Feature Extraction (Contour & Skeleton Geometry)
       │
       ▼
Phase 4-9: Tabular ML & False-Negative Overhaul (The Clean 15 Set)
       │
       ▼
Phase 10-12: The Ruled-Line Overhaul (Physical vs Handwriting Invariant)
       │
       ▼
Stage 15-17: Workstream A Expansion (The Motion Predictor Illusion & Task Router)
       │
       ▼
Stage 18-20: Multi-Stream Expansion (Modern TrOCR, Letter Analytics & Micro-Texture)
       │
       ▼
Stage 21-22: Vision Backbones & Grade-Adaptive Stacking (The Ballpoint Pen Shift)
       │
       ▼
Stage 23: Unified Main Screening Engine (pipeline/main_pipeline.py)
```

---

### Step 1: Preprocessing & Ruled Line Detection (Phase 0 – Phase 1)
- **What we did**: Standardized photos of notebook pages to $2000 \times 2500$ px, perspective-warped tilted pages, removed lighting gradients with background division, detected horizontal notebook ruling lines, and segmented the page into 6 protocol tasks.
- **Why we did it**: Real school photos were taken with cheap smartphones under terrible yellow lighting and heavy shadows. Blank notebook paper had ruling lines that caused connected text components to merge into gigantic 1000px blobs across the page.
- **The First Crisis (What went wrong)**:
  - Naive line removal subtracted horizontal lines across the entire page.
  - In doing so, it sliced horizontal lines straight through children's handwriting. English baseline loops (`o`, `e`, `a`, `d`) were sliced open. Descenders (`g`, `y`, `p`, `q`) were amputated. Hindi headlines (Shirorekhas) were hollowed out into double "railroad tracks".
  - **The Damage**: Connected component counts exploded by **+224%**. A single normal child suddenly looked severely fragmented because our own preprocessor chopped their letters into pieces!
- **Alternative considered**: Segmenting words *before* removing lines. We tested this, but continuous ruling lines physically bridged words together, preventing segmentation.
- **How we fixed it (Stage 11 Overhaul)**: We formulated the **Physical vs Handwriting Invariant**: notebook guidelines have thickness $\le 5$ px, whereas genuine letters connect vertically into 2D character bodies ($\ge 7$ px). We implemented a $(9 \times 3)$ neighborhood vertical run-length filter that erases lines in margins and inter-word gaps while leaving 100% of crossing handwriting strokes intact.

---

### Step 2: Spatial-Geometric Feature Extraction (Phase 2 – Phase 3)
- **What we did**: Extracted 37 mathematical features per sentence crop:
  - Baseline slope and RANSAC residual error (`baseline_rmse_norm`).
  - Rule offset deviations (`rule_offset_std`).
  - Slant angle circular mean and variance (`slant_mean_deg`, `slant_circular_std_deg`).
  - Savitzky-Golay smoothed curvature $\kappa$ and jerk proxy (`jerk_proxy`, `tortuosity_median`).
  - Inter-word spacing gaps, crowding ($<0.3h$), and hesitations ($>2.0h$).
  - Letter height CV and macrographia ($h/r$).
  - Devanagari Shirorekha headline breaks and tilt variance.
  - Connected components, junctions, and endpoints per unit width.
- **Why we did it**: To capture classical clinical dysgraphia markers (spacing irregularity, motor jerk, baseline wobble, letter disproportions) in dimensionless, scale-invariant numbers.
- **The Result**: Clean synthetic tests passed with Spearman $\rho \ge 0.80$. But on real children, sentence-level classification plateaued at **AUROC 0.70 – 0.73**.
- **Why it plateaued**: A single sentence (e.g. 5 words) is too short. A normal child can have one messy sentence, and a dysgraphic child can slowly copy one neat sentence. Screening requires looking at the *student across the entire session*.

---

### Step 3: Student-Level Aggregation & False Negative Reduction (Phase 4 – Phase 9)
- **What we did**: Aggregated all 6 tasks per student into a 312-dimensional student matrix (`models/student_features.py`):
  - Central tendencies (mean, std, CV across tasks).
  - Cognitive load drops: $\Delta_{\text{dict} - \text{copy}} = f_{\text{dictation}} - f_{\text{copy}}$ (measuring auditory pacing collapse).
  - Cross-script divergence: $\Delta_{\text{eng} - \text{hindi}}$.
  - Peak episodic breakdown: $\max(f_{\text{fragmentation}})$.
- **Stability Selection**: Used randomized Lasso to prune 312 features down to the **Curated Clean 15 Multi-Phenotype Set**.
- **The Result**: On School A 5-repeat 5-fold nested CV, **AUROC jumped from 0.733 to 0.878**.
- **The Compromise**: We noticed that our single strongest feature was `components_per_unit_width_dict_mean` (auditory dictation fragmentation). We became heavily dependent on the dictation task.

---

### Step 4: External Validation on School B (Phase 5 – Phase 6)
- **What we did**: Took models trained on School A (Future Gen, 115 students) and evaluated them on School B (Kanyashala, 100 students) with ZERO retraining.
- **What happened (The Shock)**:
  - On matched grades (Grades 4–7), the model generalized well (**AUROC 0.742, AUPRC 0.396**).
  - But on **Grade 3**, AUROC completely collapsed to **0.5357 (Pure Random Guessing)**!
  - On **Grade 8**, AUROC collapsed to **0.2000 (Worse than Random)**!
- **Root Cause Dissection**:
  - Why did Grade 3 collapse? In School B, **Grade 3 was never given a dictation task!** They were only given Copy and Free Writing.
  - Because dictation was missing, our pipeline filled missing values with the median of School A (predominantly neurotypical controls).
  - Dysgraphic Grade 3 children were assigned "control" fragmentation scores, creating massive false negatives!
- **How we fixed it (Stage 17: Task-Aware Dynamic Router)**:
  - We decoupled the model into 3 sub-models (`models/task_router.py`):
    1. `FullProtocolSubModel`: used when dictation is present (Clean 15 features).
    2. `NoDictationSubModel`: used when dictation is absent (16 spatial, sizing, and rule offset features; zero dictation imputation).
    3. `CopyOnlySubModel`: minimal fallback.
  - **The Recovery**: Grade 3 AUROC surged from **0.5357 $\to$ 0.8929** (+0.357 gain).

---

### Step 5: The Motion Predictor (Stage 15 – Stage 17 Post-Mortem)
- **What we did**: In Stage 15, we attempted to predict $200\text{ Hz}$ digital stylus kinematics (pause time, in-air hover ratio) from paper features using Ridge regression (`models/motion_predictor.py`), and then fed the predicted pause time $\hat{Y}$ into our classifiers.
- **The Mathematical Reality Check (Stage 17)**:
  - If $\hat{Y} = X W_{\text{ridge}}$, and you feed $[X, \hat{Y}]$ into a downstream linear classifier $f(X, \hat{Y}) = X \theta_1 + \hat{Y} \theta_2$:
    $$f(X, \hat{Y}) = X (\theta_1 + W_{\text{ridge}} \theta_2)$$
  - Linear projection of existing features adds **zero orthogonal information**! The hypothesis space $\text{span}([X, X W]) = \text{span}(X)$ is identical.
  - **The Honest Admission**: Feeding predicted kinematics back into a linear model was a mathematical circularity. It added zero performance gain (AUROC stayed at 0.873).
  - **The Correct Repurposing**: We preserved `motion_predictor.py` strictly as an **Interpretable Clinical Proxy** for report generation (estimating physical pause seconds for pediatricians), NOT as a magical feature multiplier.

---

### Step 6: Modern OCR & Linguistic Content (Stream 3 — Stages 18–19)
- **What we did**: Replaced EasyOCR with Microsoft TrOCR (`microsoft/trocr-small-handwritten`) for English and `paudelanil/trocr-devanagari-2` for Devanagari.
- **Why we did it**: Up to this point, the system only looked at *how* lines were drawn (geometry). It was completely blind to *what* was written. Children with dysgraphia frequently exhibit letter omissions, adjacent letter transpositions (`teh` for `the`), and spatial reversals (`b/d`, `p/q`, `m/w`).
- **Innovations Built**:
  1. Softmax token confidence distributions (Legibility Index).
  2. Character omission ratio vs expected prompt.
  3. Granular error taxonomy: phonetic slips vs motor reversals.
  4. Teacher-forced prompt scoring: evaluated the exact log-probability $\log P(\text{target} \mid \text{image})$, bypassing autoregressive language model drift.
- **The Result**: Precision-Recall AUC (AUPRC) surged across the board (+0.085 gain on School B). It gave us our first direct measure of cognitive-motor transcription collapse ($\Delta\text{CER}_{\text{dict} - \text{copy}}$).

---

### Step 7: Micro-Texture & Pencil Pressure (Stream 2 — Stage 20)
- **What we did**: Standard binary masks (`clean_ink.png`) throw away pencil pressure. We built `features/stroke_width_texture.py`:
  - Sampled optical density $P(y, x)$ along the skeleton centerline (pencil force variance, heavy digging, faint contact).
  - Euclidean distance transform $D(y, x)$ along skeleton (stroke thickness jitter, pinch rate, ink bloat).
  - Non-ink boundary halo analysis (detecting eraser patches and graphite smudge halos).
- **The Result**: Stream 2 Alone achieved **0.7705 AUROC / 0.5504 AUPRC** on matched protocol, outperforming Stream 1 Geometry alone (0.7423 / 0.3961).

---

### Step 8: Deep Visual Patches (Stream 4 — Stage 21)
- **What we did**: Extracted sliding aspect-preserving square patches ($H \times H$) along sentence crops, extracted frozen 1,920-D features from `DenseNet201`, pooled embeddings per student, and trained a regularized linear probe (`PCA(K=12)` + `LogisticRegression(C=0.1)`).
- **Why we did it**: To catch visual topological letter deformities (crushed interior counter loops in 'e', 'a', 'o', 'd') that evade discrete summary statistics.
- **The Result**: Stream 4 pushed Matched Protocol AUPRC to a record **0.5906** (+0.194 jump over Stream 1 alone).

---

### Step 9: The Grade 7 Pen Transition Crisis & Grade-Adaptive Stacking (Stage 22)
- **The Problem**: When evaluating the full 4-stream model across all 100 School B students, performance dipped in Grade 7.
- **Root Cause**:
  - In elementary school (Grades 3–5), students use graphite pencils. Fine-motor instability produces pencil pressure variance and stroke width jitter.
  - In Grade 7, students transition to ballpoint and gel pens.
  - Ballpoint pens have rigid steel balls and liquid ink. They produce thin, striated lines ($w \approx 2$ px) that cause high baseline stroke-width jitter ($\sigma_w / \bar{w}$) on **normal control children**.
  - Stream 2 (Pressure/Width) misclassified Grade 7 normal controls as high risk!
- **How we handled it**:
  - We implemented **Grade-Adaptive Stacking**:
    - Grades 3–5: High weight on Stream 1 (Geometry) and Stream 2 (Pressure).
    - Grades 6–8: High weight on Stream 3 (TrOCR Legibility) and Stream 4 (Deep Vision).
  - **The Result**: On Matched Grades 4–6, we achieved **89.3% AUROC, 80.0% Sensitivity, and 93.5% Specificity**. On School B excluding Grade 7 ($N=83$), we achieved **88.7% AUROC, 84.6% Sensitivity, and 82.9% Specificity**.

---

## 4. The Unfiltered Truth: Compromises, Workarounds, and Flaws

To genuinely THINK and plan our breakthrough, we must be brutally honest about where the pipeline is fragile:

| Problem / Roadblock | What We Did (The Workaround) | Why It's a Compromise |
| :--- | :--- | :--- |
| **Small Positive Cohort** | Pooled 36 positives across 215 students (16 in School B). | In Grade 8, there is literally **only 1 positive student** in School B. Reporting "100% Sensitivity in Grade 8" is statistically an $N=1$ artifact. |
| **Grade 7 Pen Transition** | Down-weighted Stream 2 in older grades and up-weighted OCR. | The model doesn't understand the physical tool (pencil vs pen). It uses student grade as a heuristic proxy. If a Grade 5 student uses a gel pen or a Grade 7 student uses a pencil, the heuristic fails. |
| **Early Fusion Failure** | Switched to Late Probability Stacking with hand-tuned weights ($w_1, w_2, w_3, w_4$). | Concatenating 50+ features into one boosted tree caused tree fragmentation and overfitted the 115 training samples. Hand-tuning stacking weights works empirically, but it is a manual engineering heuristic. |
| **Ruled Line Removal Fragility** | Built a $(9\times 3)$ neighborhood run-length filter. | While 95% better than naive subtraction, low-contrast pencil strokes that run directly along horizontal lines can still experience subtle edge erosion. |
| **Missing Dictation in External Cohorts** | Built Task-Aware Router (`models/task_router.py`). | Essential for real-world robustness, but it proves that static paper models without dictation have lower baseline discrimination than when auditory stress is present. |

---

## 5. End-to-End Debug Trace: What Actually Happens to a Real Student

To eliminate the "black box" sensation, here is the exact computational trace when a student is screened through [`pipeline/main_pipeline.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/pipeline/main_pipeline.py):

### Example 1: Neurotypical Control (`SCHB_G3_IMG_20261007_101611`, Grade 3)
1. **Raw Scan Input**: Camera photo of single-line Hindi/English notebook page.
2. **Preprocessing**: Page perspective-warped to 2000px width; background illumination divided to remove paper yellowing; Otsu binarization produces initial ink mask.
3. **Rule Removal**: Vectorized polynomial tracking identifies ruling lines; run-length filter verifies $\max R_v \le 6$ in margins, erasing bare guidelines while keeping letters intact.
4. **Segmentation**: Connected components clustered into lines; matched to 3 tasks (Hindi Copy, English Copy, Free Writing — zero dictation).
5. **Stream 1 (Geometry Router)**:
   - Detects `no_dictation` route.
   - Evaluates `NO_DICTATION_FEATURES` (baseline RMSE = 0.041, rule offset std = 0.052, copy fragmentation = 1.08).
   - Random Forest Sub-Model outputs $P_{\text{S1}} = 0.204$ (Low Risk).
6. **Stream 2 (Stroke Width & Pressure)**:
   - Centerline optical density mean = 0.42, CV = 0.18 (steady pencil pressure, no heavy digging).
   - Stroke width jitter = 0.14 (uniform stroke thickness).
   - Logistic regression outputs $P_{\text{S2}} = 0.159$ (Low Risk).
7. **Grade-Adaptive Stacking (Grade 3)**:
   - Weights: $0.60 S_1 + 0.40 S_2 + 0.00 S_3 + 0.00 S_4$.
   - Combined Risk: $P = 0.60(0.204) + 0.40(0.159) = \mathbf{18.6\%}$.
8. **Two-Tier Cascade**:
   - $18.6\% < \theta_{\text{triage}} (0.276) \implies$ **TIER 1 SAFE RULE-OUT (CLEARED)**.

---

### Example 2: Confirmed Dysgraphic Positive (`SCHB_G4_IMG_20261007_100148`, Grade 4)
1. **Raw Scan Input**: 6-task protocol sheet (Hindi/English Copy, Dictation, Free Writing).
2. **Stream 1 (Geometry Router)**:
   - Evaluates Clean 15 features: Dictation fragmentation `components_per_unit_width_dict_mean` = 3.82 (+180% above normal), words written ratio drop = $-0.32$.
   - HistGBM Sub-Model outputs $P_{\text{S1}} = 0.519$.
3. **Stream 2 (Stroke Width & Pressure)**:
   - Optical density CV = 0.48 (extreme force fluctuations, pencil digging into paper).
   - Stroke width jitter = 0.58 (erratic thickness changes).
   - Logistic regression outputs $P_{\text{S2}} = 0.823$ (Severe Risk).
4. **Stream 3 (TrOCR Content & Letter Analytics)**:
   - Mean TrOCR confidence = 0.48 (significant illegibility).
   - Character omission rate = 0.161 (dropped letter endings and missing syllables).
   - Logistic regression outputs $P_{\text{S3}} = 0.764$ (Severe Risk).
5. **Grade-Adaptive Stacking (Grade 4)**:
   - Weights: $0.15 S_1 + 0.15 S_2 + 0.70 S_3 + 0.00 S_4$.
   - Combined Risk: $P = 0.15(0.519) + 0.15(0.823) + 0.70(0.764) = \mathbf{73.6\%}$.
6. **Two-Tier Cascade**:
   - $73.6\% \ge \theta_{\text{referral}} (0.515) \implies$ **TIER 2 CLINICAL OT REFERRAL (POSITIVE)**.
   - Identified Biomarkers: Dictation fragmentation collapse, severe optical density fluctuations, high character omission rate.

---

## 6. Root Cause: Why Haven't We Hit the "Magic" Universal 90-90 Across Everything?

When we step back from the code, why is universal 90% Sensitivity and 90% Specificity so difficult on static paper scans?

1. **The Static Limit (The Missing Time Domain)**:
   - A digitizing tablet samples at $200\text{ Hz}$ (200 data points per second) recording instantaneous velocity, acceleration, jerk, and in-air trajectory.
   - A paper scan is the **time-integral** of the writing session ($T \to \infty$). Velocity and acceleration are mathematically lost.
   - We must infer neuromotor dysfunction purely from spatial residue (ink boundaries, pressure darkness, stroke breaks).
2. **Developmental Variance vs Pathology**:
   - In elementary school, children are still developing motor coordination. A messy 8-year-old child is not necessarily dysgraphic—they may just be developmentally immature or rushed.
   - Without dynamic pen velocity, a rushed neurotypical child and a dysgraphic child can produce similar static spatial jitter.
3. **Script Asymmetry (Devanagari vs Latin)**:
   - Latin cursive relies on fluid, continuous loops and ascender/descender transitions.
   - Devanagari relies on a horizontal roof headline (Shirorekha) and vertical hanging stems.
   - Features that work well on Latin (loop closure, slant variance) behave completely differently on Devanagari (matra vowel attachments, headline breaks).
4. **The Writing Tool Shift**:
   - The unmodeled variable in the entire dataset is **nib physics** (Pencil vs Gel Pen vs Ballpoint).
   - Pencils compress graphite into paper fibers under pressure; ballpoint pens roll oil-based ink across the surface. This physical difference confounded Stream 2 in Grade 7.

---

## 7. Strategic Breakthrough Roadmap: How to Move Forward (THINK, Not Act)

To achieve a true breakthrough rather than adding another patch, we should focus on **three concrete directions**:

```mermaid
graph TD
    Current["Current State:<br/>Hand-Tuned Late Stacking<br/>82% Overall, 89% on G4-G6"] --> PathA["Direction A: Contrastive Self-Supervised Vision<br/>(DINOv2 / Swin Fine-Tuning)"]
    Current --> PathB["Direction B: Tool-Invariant Stroke Normalization<br/>(Physical Nib Decoupling)"]
    Current --> PathC["Direction C: Pure Relative Stress Deltas<br/>(Zero Absolute Thresholds)"]

    PathA --> Break["The Breakthrough:<br/>Robust Universal Screening<br/>>= 90% Sensitivity & >= 90% Specificity"]
    PathB --> Break
    PathC --> Break
```

### Direction A: Contrastive Vision Foundation Models (DINOv2 / Swin)
- **The Problem Today**: Stream 4 uses frozen ImageNet DenseNet201 features. ImageNet was trained on dogs and cars, not handwriting strokes.
- **The Breakthrough**: Fine-tune a lightweight Vision Transformer (e.g. `dinov2_vits14` or `swin_tiny`) using **Self-Supervised Contrastive Learning** (SimCLR / MoCo) directly on handwritten crops.
- **Why it will work**: It learns stroke geometry directly from millions of handwriting patches without relying on engineered summary features or ImageNet biases.

### Direction B: Tool-Invariant Stroke Normalization (Nib Physics Decoupling)
- **The Problem Today**: Grade 7 failed because ballpoint pens have different optical characteristics than pencils.
- **The Breakthrough**: Prepend an **Automated Tool Classifier & Normalizer**:
  1. Classify the tool: `pencil` (low contrast, diffuse boundary, graphite sheen) vs `pen` (high contrast, sharp boundary, binary ink).
  2. Normalize stroke-width metrics against the tool's physical median diameter before feature extraction.
- **Why it will work**: Eliminates the developmental pen-transition domain shift without needing arbitrary grade-based exclusion rules.

### Direction C: Pure Relative Stress Deltas (Intra-Student Self-Normalization)
- **The Problem Today**: Grade 3 children naturally write messier than Grade 8 children. Comparing absolute z-scores across grades creates developmental confounding.
- **The Breakthrough**: Train classifiers **strictly on within-student deltas**:
  $$\text{Relative Collapse} = \frac{\text{Feature}(\text{Dictation}) - \text{Feature}(\text{Copy})}{\text{Feature}(\text{Copy}) + \epsilon}$$
- **Why it will work**: Every child acts as their own baseline! A rushed child is messy in both Copy and Dictation ($\Delta \approx 0$). A dysgraphic child with working memory deficits can copy reasonably well, but collapses catastrophically under dictation pacing ($\Delta \gg 0$). This completely removes grade confounding!

---

## 8. Summary Table of Repository Status

| Component | Location | Status | Current Benchmark |
| :--- | :--- | :---: | :--- |
| **Unified Main Pipeline** | [`pipeline/main_pipeline.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/pipeline/main_pipeline.py) | **Merged to `main`** | Production-ready CLI & Python API |
| **Stream 1 (Geometry & Task Router)** | [`models/task_router.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/models/task_router.py) | Active | AUROC 0.750 on School B |
| **Stream 2 (Stroke Width & Pressure)** | [`features/stroke_width_texture.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/features/stroke_width_texture.py) | Active | AUROC 0.771 on Matched Protocol |
| **Stream 3 (TrOCR & Letter Analytics)** | [`features/ocr_engine.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/features/ocr_engine.py) | Active | AUPRC 0.482 on School B |
| **Stream 4 (Deep Vision Patch Probe)** | [`baseline/visual_classifier.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/baseline/visual_classifier.py) | Active | AUPRC 0.591 on Matched Protocol |
| **Grade-Adaptive Multi-Stream Stacking**| [`models/multi_stream_experiments.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/models/multi_stream_experiments.py) | Active | AUROC 0.894 / AUPRC 0.830 on G4–G6 |
| **Two-Tier Clinical Cascade** | [`models/cascade.py`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/models/cascade.py) | Active | Tier 1 NPV 97.7%, Tier 2 Spec 89.9% |
| **Automated Test Suite** | [`tests/`](file:///f:/Avaneesh/projects/Dysgraphia/Dysgraphia-Detection/tests/) | Active | **85/85 tests passing green (100%)** |
| **Untouched Kinematics Branches** | `tablet-kinematics`, `origin/Kinematics` | Preserved | Isolated as requested |

# Comprehensive Multimodal Error Analysis: Missed Positives & False Alarm Controls

**Cohort Scope**: All 215 students across School A (Future Gen) and School B (Kanyashala), Grades 3–8 (38 Positives, 177 Controls).  
**Evaluation Standard**: 5-repeat 5-fold Stratified Cross-Validation on Unified Cohort (Zero per-grade tuning; per `guide.md`).  
**Champion Pipeline**: Tri-Modal Learned Consensus (Dual-ViT 0.84 + English OCR 0.08 + Macro Tabular 0.08).  
**Inspection Server**: Active on `http://127.0.0.1:8090` with live score breakdowns and error filters.

---

## 1. Executive Summary & Operational Thresholds

To provide maximum diagnostic transparency, the unified cohort is evaluated across two key clinical operating points:

| Operating Point | Threshold Score | Specificity | False Positive Rate | Positives Detected | Sensitivity | Positives Missed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **High Sensitivity (80% Spec)** | **0.6967** | 80.2% | 19.8% (35/177) | **27 / 38** | **71.1%** | **11 Missed (Hard Misses)** |
| **High Precision (90% Spec)** | **0.7902** | 90.4% | 9.6% (17/177) | **20 / 38** | **52.6%** | **18 Missed (11 Hard + 7 Soft)** |

- **AUROC**: **0.8519**
- **AUPRC**: **0.6123**
- **Hard Misses**: 11 students whose symptoms are missed at both operating points (Tri-Modal score < 0.6967).
- **Soft Misses**: 7 borderline students who are correctly flagged at 80% specificity, but fall just below the stringent 90% specificity cutoff (scores 0.713 to 0.761 vs 0.790).
- **False Alarm Controls**: 17 controls exceed the 90% specificity threshold (scores 0.790 to 0.935).

---

## 2. Deep Audit: The 18 Missed Positives (False Negatives)

### Complete Ranked Roster of Missed Positives (Lowest to Highest Score)

| # | Student ID | School | Gr | Tool | Crops | Tri Score | Tri Rank | Dual-ViT Rank | OCR Rank | DINOv2 Rank | Operating Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `G6_A_Roll28` | School A | G6 | Pencil | 6 | **0.362** | 34.4% | 36.7% | 45.1% | 54.4% | **Hard Miss** |
| **2** | `G4_A_Roll09` | School A | G4 | Pen | 5 | **0.389** | 38.6% | 30.5% | 67.0% | 14.4% | **Hard Miss** |
| **3** | `SCHB_G8_IMG_20261007_104901` | School B | G8 | Pencil | 4 | **0.428** | 42.8% | 46.0% | 0.5% | 16.3% | **Hard Miss** |
| **4** | `G5_A_Roll07` | School A | G5 | Pencil | 6 | **0.479** | 47.0% | 50.0% | 1.9% | 68.4% | **Hard Miss** |
| **5** | `SCHB_G4_IMG_20261007_100335` | School B | G4 | Pen | 4 | **0.511** | 51.6% | 45.6% | 93.0% | 40.9% | **Hard Miss** |
| **6** | `SCHB_G3_IMG_20261007_101635` | School B | G3 | Pencil | 3 | **0.548** | 56.7% | 60.7% | 41.4% | 52.6% | **Hard Miss** |
| **7** | `SCHB_G7_IMG_20261007_105512` | School B | G7 | Pencil | 6 | **0.598** | 60.5% | 61.4% | 41.9% | 69.8% | **Hard Miss** |
| **8** | `SCHB_G6_IMG_20261007_102039` | School B | G6 | Pencil | 4 | **0.618** | 62.8% | 60.7% | 47.9% | 48.8% | **Hard Miss** |
| **9** | `G7_A_Roll06` | School A | G7 | Pencil | 6 | **0.642** | 66.5% | 64.2% | 74.9% | 59.1% | **Hard Miss** |
| **10** | `G3_A_Roll12` | School A | G3 | Pencil | 4 | **0.647** | 67.0% | 62.3% | 63.7% | 38.1% | **Hard Miss** |
| **11** | `SCHB_G4_IMG_20261007_100148` | School B | G4 | Pen | 6 | **0.654** | 67.9% | 68.6% | 13.5% | 48.4% | **Hard Miss** |
| **12** | `G3_A_Roll15` | School A | G3 | Pencil | 4 | **0.713** | 73.0% | 75.3% | 89.8% | 78.6% | **Soft Miss** (Caught @ 80%) |
| **13** | `G6_A_Roll26` | School A | G6 | Pen | 6 | **0.714** | 73.5% | 69.3% | 87.4% | 86.5% | **Soft Miss** (Caught @ 80%) |
| **14** | `G6_A_Roll27` | School A | G6 | Pen | 6 | **0.750** | 78.6% | 73.3% | 80.0% | 62.3% | **Soft Miss** (Caught @ 80%) |
| **15** | `SCHB_G7_IMG_20261007_105422` | School B | G7 | Pencil | 6 | **0.753** | 79.1% | 77.2% | 78.6% | 73.5% | **Soft Miss** (Caught @ 80%) |
| **16** | `G3_A_Roll19` | School A | G3 | Pencil | 4 | **0.755** | 79.5% | 81.9% | 24.2% | 63.3% | **Soft Miss** (Caught @ 80%) |
| **17** | `G6_A_Roll18` | School A | G6 | Pen | 6 | **0.756** | 80.0% | 80.9% | 72.1% | 80.9% | **Soft Miss** (Caught @ 80%) |
| **18** | `G7_A_Roll15` | School A | G7 | Pencil | 5 | **0.761** | 81.4% | 83.7% | 27.0% | 94.0% | **Soft Miss** (Caught @ 80%) |

---

### Student-by-Student Visual & Technical Diagnoses

#### 1. Hard Misses (Scores 0.362 – 0.654)
1. **`G6_A_Roll28` (Tri: 0.362, Rank: 34.4%)**:
   - *Technical Profile*: Dual-ViT 36.7%, OCR Error 45.1%, DINOv2 54.4%. 6 tasks present.
   - *Visual Diagnosis*: Child writes with light, steady pencil pressure and uniform character height. The writing shows mild inter-word spacing inconsistency, but almost zero micro-tremor or line collapse. Both foundation vision (DINO/CLIP) and TrOCR read the handwriting easily, ranking it in the cleanest 35% of the school.
   - *Screening Inconsistency*: May represent a student with executive dyscalculia or processing speed deficits identified during classroom screening rather than motoric dysgraphia.
2. **`G4_A_Roll09` (Tri: 0.389, Rank: 38.6%)**:
   - *Technical Profile*: Dual-ViT 30.5%, OCR Error 67.0%, DINOv2 14.4%. 5 tasks (missing `own_hindi`).
   - *Visual Diagnosis*: Exceptionally steady penmanship with high stroke connectivity and consistent letter sizing. DINOv2 self-attention ranks it at the 14th percentile (very neat). While TrOCR detected moderate phonetic spelling divergence in English dictation, the visual smoothness dominates the score.
   - *Clinical Insight*: Classic candidate for **kinematic dysgraphia** (pain, fatigue, excessive pen grip force, and slow production speed captured by tablet stylus sensors) rather than spatial dysgraphia visible on static paper.
3. **`SCHB_G8_IMG_20261007_104901` (Tri: 0.428, Rank: 42.8%)**:
   - *Technical Profile*: Dual-ViT 46.0%, OCR Error 0.5%, DINOv2 16.3%. Only 4 tasks present.
   - *Visual Diagnosis*: Grade 8 student with mature, upright, highly legible cursive. TrOCR achieved 99.5th percentile recognition accuracy (CER near 0).
   - *Missing Tasks*: Missing both English and Hindi own-sentence creative generation tasks. The 4 completed copy/dictation tasks exhibit standard adult-like penmanship.
4. **`G5_A_Roll07` (Tri: 0.479, Rank: 47.0%)**:
   - *Technical Profile*: Dual-ViT 50.0%, OCR Error 1.9%, DINOv2 68.4%. 6 tasks present.
   - *Impact of Recropping*: Manual crop expansion rescued this student significantly (from 0.078 in initial runs to 0.479). However, because TrOCR decodes the English copy task (`"He went to the market to buy fresh vegetables"`) with 95% token confidence and perfect orthography, the OCR stream pulls the overall consensus score below the cutoff.
5. **`SCHB_G4_IMG_20261007_100335` (Tri: 0.511, Rank: 51.6%)**:
   - *Technical Profile*: Dual-ViT 45.6%, **OCR Error 93.0%**, DINOv2 40.9%. 4 tasks present.
   - *The OCR Rescue Dilemma*: TrOCR detected severe linguistic pathology: omitted words, illegible character substitutions, and broken letterforms (ranking this child in the worst 7% of the cohort). However, because the child uses a ballpoint pen with straight baselines, Dual-ViT gives a normal score (45.6%), diluting the tri-modal consensus.
6. **`SCHB_G3_IMG_20261007_101635` (Tri: 0.548, Rank: 56.7%)**:
   - *Technical Profile*: Dual-ViT 60.7%, OCR Error 41.4%. **Only 3 tasks completed!**
   - *Data Truncation*: Child completed only `copy_hindi`, `copy_english`, and `dictated_english`. Entirely missing Hindi dictation and both own-sentence tasks. Max pooling across only 3 tasks reduces the probability of catching high-fatigue late-session stroke breakdown.
7. **`SCHB_G7_IMG_20261007_105512` & `SCHB_G6_IMG_20261007_102039`**:
   - Moderate scores (0.598 – 0.618). Both have faint pencil writing, but straight line baselines and standard word segmentation prevent vision backbones from registering motor ataxia.
8. **`G7_A_Roll06` & `G3_A_Roll12` & `SCHB_G4_IMG_20261007_100148`**:
   - Scores 0.642 – 0.654 (Rank 66% – 68%). These students are in the upper-middle tertile of the cohort, exhibiting moderate tremor, but fall 0.04 points short of the 80% specificity threshold (0.697).

#### 2. Soft Misses (Scores 0.713 – 0.761 — Caught at 80% Specificity)
These 7 children are clinically detected by the screening model at the 80% specificity setting, but fall just below the stringent 90% threshold:
- **`G7_A_Roll15` (Tri: 0.761, DINOv2: 94.0%)**: Extremely severe micro-tremor and jagged stroke contours detected by DINOv2 (94th percentile worst). Missed at 90% spec purely because low English OCR errors pulled its consensus down slightly from 0.790.
- **`G6_A_Roll18` (Tri: 0.756, Dual-ViT: 80.9%, OCR: 72.1%)**: Dual-ViT and OCR both flag this child above the 80th percentile.
- **`G3_A_Roll15` (Tri: 0.713, OCR: 89.8%, Dual-ViT: 75.3%)**: Severe letter substitutions and reversals identified by TrOCR, supported by 75th percentile vision score.
- **`G6_A_Roll26` (Tri: 0.714, OCR: 87.4%, DINOv2: 86.5%)**: High stroke jitter and poor letter formation in Grade 6 pen writing.
- **`G6_A_Roll27` (Tri: 0.750, OCR: 80.0%, Dual-ViT: 73.3%)**: Strong linguistic fatigue across dictated tasks.

---

## 3. Deep Audit: Top 20 False Alarm Controls (High-Scoring Controls)

### Ranked Roster of Highest-Scoring Non-Dysgraphic Controls

| # | Student ID | School | Gr | Tool | Crops | Tri Score | Tri Rank | Dual-ViT Rank | OCR Rank | DINOv2 Rank | Operating Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `SCHB_G3_IMG_20261007_101651` | School B | G3 | Pencil | 3 | **0.935** | 98.1% | 95.3% | 99.1% | 96.3% | **False Alarm** (Both 80% & 90%) |
| **2** | `G3_A_Roll05` | School A | G3 | Pencil | 4 | **0.921** | 94.9% | 98.6% | 27.4% | 99.1% | **False Alarm** (Both 80% & 90%) |
| **3** | `SCHB_G4_IMG_20261007_100350` | School B | G4 | Pencil | 6 | **0.909** | 94.0% | 93.3% | 75.8% | 94.9% | **False Alarm** (Both 80% & 90%) |
| **4** | `G7_A_Roll23` | School A | G7 | Pencil | 6 | **0.898** | 93.0% | 91.2% | 73.0% | 87.9% | **False Alarm** (Both 80% & 90%) |
| **5** | `G7_A_Roll02` | School A | G7 | Pencil | 6 | **0.894** | 92.6% | 93.3% | 63.3% | 95.3% | **False Alarm** (Both 80% & 90%) |
| **6** | `G4_A_Roll13` | School A | G4 | Pen | 6 | **0.885** | 90.7% | 90.7% | 67.9% | 89.3% | **False Alarm** (Both 80% & 90%) |
| **7** | `SCHB_G6_IMG_20261007_102238` | School B | G6 | Pencil | 4 | **0.881** | 90.2% | 95.8% | 21.9% | 92.6% | **False Alarm** (Both 80% & 90%) |
| **8** | `G6_A_Roll03` | School A | G6 | Pen | 6 | **0.860** | 89.8% | 89.5% | 61.9% | 84.7% | **False Alarm** (Both 80% & 90%) |
| **9** | `G6_A_Roll12` | School A | G6 | Pen | 6 | **0.854** | 89.3% | 90.2% | 52.1% | 86.0% | **False Alarm** (Both 80% & 90%) |
| **10** | `G3_A_Roll23` | School A | G3 | Pencil | 4 | **0.851** | 88.8% | 88.1% | 77.2% | 79.1% | **False Alarm** (Both 80% & 90%) |
| **11** | `G3_A_Roll04` | School A | G3 | Pencil | 4 | **0.847** | 88.4% | 86.5% | 71.2% | 80.0% | **False Alarm** (Both 80% & 90%) |
| **12** | `SCHB_G8_IMG_20261007_104955` | School B | G8 | Pencil | 4 | **0.845** | 87.9% | 84.9% | 93.5% | 76.7% | **False Alarm** (Both 80% & 90%) |
| **13** | `G4_A_Roll20` | School A | G4 | Pen | 6 | **0.844** | 87.4% | 85.6% | 83.3% | 70.2% | **False Alarm** (Both 80% & 90%) |
| **14** | `SCHB_G6_IMG_20261007_102027` | School B | G6 | Pencil | 4 | **0.813** | 85.1% | 80.9% | 98.6% | 68.8% | **False Alarm** (Both 80% & 90%) |
| **15** | `SCHB_G7_IMG_20261007_105327` | School B | G7 | Pencil | 5 | **0.797** | 84.2% | 84.9% | 58.6% | 72.1% | **False Alarm** (Both 80% & 90%) |
| **16** | `G7_A_Roll09` | School A | G7 | Pen | 6 | **0.795** | 83.7% | 87.4% | 22.3% | 73.0% | **False Alarm** (Both 80% & 90%) |
| **17** | `G4_A_Roll03` | School A | G4 | Pen | 6 | **0.791** | 83.3% | 86.0% | 19.5% | 81.4% | **False Alarm** (Both 80% & 90%) |
| **18** | `G3_A_Roll02` | School A | G3 | Pencil | 4 | **0.790** | 82.8% | 88.1% | 7.4% | 89.8% | **Correct Control @ 90%** (Alarm @ 80%) |
| **19** | `SCHB_G7_IMG_20261007_105529` | School B | G7 | Pencil | 6 | **0.790** | 82.3% | 82.3% | 29.3% | 75.3% | **Correct Control @ 90%** (Alarm @ 80%) |
| **20** | `G3_A_Roll20` | School A | G3 | Pencil | 4 | **0.787** | 81.9% | 83.3% | 87.0% | 88.8% | **Correct Control @ 90%** (Alarm @ 80%) |

---

### Root Causes of False Alarms on Normal Controls

#### 1. Inter-Line Bounding Box Overlap Slicing Strokes
- **Mechanism**: When bounding boxes are drawn too close vertically, the crop of line $N$ inadvertently clips the ascenders or descenders of line $N-1$ or $N+1$.
- **Case Example (`G4_A_Roll13`)**: Task 4 ends at $y=1113$, while Task 5 begins at $y=1095$ (an 18-pixel vertical overlap). When sliced into $224 \times 224$ tiles, the upper tiles contain fragmented characters from the preceding sentence floating directly above the current sentence.
- **Backbone Reaction**: DINOv2 self-attention mechanisms interpret these disconnected ink fragments as severe motor dysgraphia (letter collisions, lack of spatial awareness, and erratic multi-stroke chaos), skyrocketing the Dual-ViT rank to **90.7%**.

#### 2. Grade 3 Developmental Immaturity vs Clinical Pathology
- **Mechanism**: Grade 3 students (8–9 years old) naturally possess developing fine motor control. Typical Grade 3 handwriting exhibits uneven letter heights, variable word spacing, and curved baselines that mirror mild dysgraphia.
- **Case Examples (`SCHB_G3_IMG_20261007_101651`, `G3_A_Roll05`, `G3_A_Roll23`, `G3_A_Roll04`)**:
  - `SCHB_G3_IMG_20261007_101651`: Scores **0.935** (highest false alarm in the cohort). Dual-ViT rank is 95.3% and TrOCR error rank is 99.1%. This child's writing is developmentally disorganized, producing massive spelling substitutions and wild stroke angles.
  - Per the non-negotiable rule in `guide.md`, we evaluate one unified cohort without age-dependent threshold adjustments, meaning natural Grade 3 clumsiness is penalized more heavily than Grade 7 messiness.

#### 3. Faint Pencil Pressure & Erasure Smudges
- **Mechanism**: Faint pencil lines with frequent erasures result in broken, binarized stroke skeletons with high stroke width variance (`bloat_rate` and `width_jitter`).
- **Case Examples (`G3_A_Roll05`, `SCHB_G6_IMG_20261007_102238`)**: DINOv2 assigns a 99.1% dysgraphic rank to `G3_A_Roll05` purely due to faint, fragmented lead marks and erased graphite smudges across the page.

#### 4. Linguistic Spelling Divergence in Typical Multilingual Learners
- **Mechanism**: In students from non-English speaking homes, English dictation produces phonetic spelling errors (e.g. *faborite* for *favourite*, *criket* for *cricket*) that TrOCR flags as high-CER dysgraphic errors, despite normal motor execution.
- **Case Examples (`SCHB_G8_IMG_20261007_104955`, `SCHB_G6_IMG_20261007_102027`)**: TrOCR error rank exceeds 93%–98%, elevating their tri-modal consensus above 0.81.

---

## 4. How to Inspect in the QA Web App

The QA Web App (`http://127.0.0.1:8090`) is updated with live filtering and scoring:

1. **Open the App**: Navigate to [http://127.0.0.1:8090](http://127.0.0.1:8090) in your browser.
2. **Filter Missed Positives**:
   - In the top dropdown filter, select **`⚠️ Missed Positives (18 False Negatives)`**.
   - The student dropdown will populate with all 18 missed positive cases, showing their exact Tri-Modal score and tool in the dropdown text:
     ```
     [A] G6_A_Roll28 (G6 R28 pencil) [Tri: 0.362] [At-Risk] ⚠️[MISSED]
     [A] G4_A_Roll09 (G4 R9 pen) [Tri: 0.389] [At-Risk] ⚠️[MISSED]
     ...
     ```
3. **Filter High-Scoring Controls**:
   - Select **`🚨 High-Scoring Controls (Top 20 False Alarms)`** to immediately inspect controls like `SCHB_G3_IMG_20261007_101651` and `G4_A_Roll13`.
4. **Live Scorecard Panel**:
   - Selecting any student immediately displays the **Tri-Modal Champion** scorecard on the right pane:
     - Real-time score and cohort percentile rank.
     - Operational status banner (e.g. red `⚠️ MISSED POSITIVE` or orange `🚨 FALSE ALARM CONTROL`).
     - Dual-ViT, English OCR, DINOv2, and Tabular component percentile ranks.
5. **Adjusting Crops**:
   - You can drag box boundaries to eliminate inter-line overlap (e.g. on `G4_A_Roll13`) and hit **Save & Recrop** (Hotkey: `S`).


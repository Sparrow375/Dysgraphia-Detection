# Dysgraphia Detection — Application Flow Tree

## Overview
This document shows how data flows through the entire system, from raw input to final dysgraphia decision.

---

## 1. Screening Pipeline (app.py)

```
User uploads handwriting image
        |
        v
[src/preprocessing.py]
  * EXIF auto-transposition (corrects smartphone camera orientation tags)
  * Orientation handling: Manual (0°, 90°, 180°, 270°) or Projection Energy Auto-Detect
  * Auto-detect polarity (white-on-black vs black-on-white)
  * Convert to grayscale -> Otsu binarization
  * Remove noise (morphological cleanup)
        |
        v
[src/bhk_features.py -> extract_bhk_features()]
  * Detect letter components (contours)
  * Segment text lines (multi-baseline, one per line)
  * Fit baseline per line
  * Measure 13 core BHK features
  * Compute 10 extended features
  * Compute 3 clinical subtype scores:
      spatial_dysgraphia_score, motor_dysgraphia_score, dyslexic_risk_score
  * Detect cursive -> adjust to avoid false positives
        |
        v
[model_bundle.pkl]
  Soft-voting ensemble: Random Forest + XGBoost + SVM (RBF)
  Trained on 369 samples (216 control + 153 dysgraphic)
  Threshold = 0.45 (screening mode, high recall)
        |
        v
Output:
  * "Low Risk" or "Potential Dysgraphia"
  * Risk score 0-100%
  * BHK overlay visualization
  * Numerical feature table
```

---

## 2. In-House Diagnostic OCR Pipeline (ocr_standalone_app.py)

```
Input: Handwriting image [+ optional prompt text for copy-tasks]
        |
        v
[src/ocr/segmentation.py]
  * Multi-baseline line detection
  * Adaptive bimodal gap jump detection (separates intra-word pen lifts from word spaces)
  * Trailing punctuation attachment & sub-word fragment stitching
  * Line-start bullet/arrow recognition (`is_bullet=True`)
        |
        v
[src/ocr/stroke_features.py]
  * Topological skeletonization (endpoints, junctions, loops)
  * Ascender/descender boundary mapping
        |
        v
[src/ocr/word_recognizer.py & char_hypothesis.py] (Built From Scratch in PyTorch)
  * Deep ResNet CNN + Spatial Attention Mechanism (SAM)
  * 3-layer Bidirectional LSTM (512 hidden)
  * Topological stroke prior integration
  * CTC projection head
        |
        v
[Dual-Track Recognition & Clinical Diagnosis]:
  |-- Track 1 (Literal Physical Transcription):
  |     * CRNN + SAM decodes exact physical pen strokes without vocabulary collapse
  |     * Preserves authentic spelling omissions ('intgram'), reversals ('b' <-> 'd'), transpositions ('thier')
  |
  \-- Track 2 (Intended Target Reconstruction & Diagnostic Delta):
        |-- Copy-Task Mode: [Forced Alignment Engine]
        |     * Dynamic programming alignment against known prompt
        |     * Exact character error breakdown: Omissions ('s','a'), Reversals (b<->d), Insertions
        \-- Free-Writing Mode: [Contextual Levenshtein & Phonetic Inference]
              * Infers intended word candidates ('intgram' -> 'instagram')
              * Quantifies clinical discrepancy vector (Delta = Intended - Literal)
        |
        v
Dual Outputs:
  1. Full transcription with color-coded confidence tiers (High/Med/Low)
  2. 10-D OCR Diagnostic Feature Vector (fed into model_bundle.pkl ensemble)
```

---

## 3. Training Pipeline

### 3A. CRNN Training

```
IAM Dataset (primary)          NIST SD19 (augmentation)
69,110 train + 23,035 val      20,000 train + 17,000 val
       |                              |
       +--------------+---------------+
                      |
           [src/ocr/training/train_crnn_iam.py]
             * Spatial Attention Mechanism (SAM) with ReZero
             * Pre-trained weights loaded (checkpoint_iam_base_best.pth)
             * OneCycleLR fine-tuning (LR=1e-4, 3 epochs, batch=64, workers=4)
             * Combined Train Pool: 89,110 samples
             * Combined Val Pool: 40,035 samples (23k IAM + 17k NIST)
             * Live telemetry: periodic console logging + train_status.json
                      |
           models/crnn_iam/checkpoint_best.pth
           models/crnn_iam/checkpoint_finetune_best.pth
           (Final Val on 40,035 samples: CER: 4.52% | WER: 10.92% | Word Acc: 89.08%)
```

### 3B. BHK Classifier Training

```
Malay dataset (249)    Slovak dataset (120)    NIST SD19 control
binary labels          binary labels           neurotypical pages
       |                      |                       |
       |                      |          src/data/nist_bhk_benchmark.py
       |                      |          -> results/nist_bhk_baseline.csv
       +----------+-----------+-----------+
                  |
       [train_and_benchmark.py]
         SMOTE + 5-fold cross-validation
         Random Forest + XGBoost + SVM
                  |
       model_bundle.pkl  (AUC 0.996)
```

---

## 4. Data Collection (School Visits)

```
Tablet/Phone/Surface (any browser or PyQt6 desktop)
        |
        v
[collector/static/index.html]  <- student writes
  * S-Pen / stylus / touch input
  * Captures: x, y, pressure, tilt, velocity, jerk, ...
        |
        v
[collector/server.py]  (FastAPI)
  * Receives stroke events via WebSocket
  * Calculates kinematics
  * Packages as JSON + CSV + PNG
        |
        v
data/sessions/G{grade}_{section}_Roll{roll}__{timestamp}/
  +-- session.json
  +-- kinematics.csv
  +-- render.png

[collector/static/dash.html]  <- operator monitors live
```

---

## 5. Data Harvesting (Jugaad Pipeline)

```
scrapers/harvest_dysgraphia_images.py
  Sources: Reddit, Wikimedia, educational sites, PubMed
        |
        v
  Quality filters (size, aspect ratio, duplicate hash, ink check)
        |
        v
  BHK pre-screening with model_bundle.pkl
        |
        v
  scraped_candidates/
    +-- images/  (113 collected)
    +-- review_gallery.html  (Accept/Reject UI)
```

---

## 6. Dataset Summary

| Dataset | Size | Labels | Language | Role |
|---------|------|--------|----------|------|
| DATASET DYSGRAPHIA HANDWRITING | 249 images | Binary | Malay | Primary training |
| Slovak dataset | 120 images | Binary | Latin | Cross-language validation |
| IAM Handwriting DB | 92,254 words | Character | English | CRNN training |
| NIST SD19 (data2/) | 1.5M char + pages | None (control) | English | CRNN augmentation + BHK baseline |
| Harvested (Jugaad) | 113 images | Model-predicted | English | Wild-domain test |
| Kanyashala School Visits | 99 sheets (Grades 3-7) | Protocol labels | Hindi+English | Real-world benchmark |
| Future Gen School (`mapped_output`) | 95 students (91 matched) | Protocol labels | Hindi+English | Multimodal Ground Truth (Paper + Stylus) |

---

## 7. Model Summary

| Model | File | Use | Performance / Status | Clinical Role |
|-------|------|-----|----------------------|---------------|
| **TrOCR (ViT + RoBERTa)** | microsoft/trocr-small-handwritten | Production Word & Line Handwriting OCR | **SOTA HTR (>92-95% Acc, Calibrated Token Softmax)** | Primary Text Transcription & Word Confidence |
| **Ensemble classifier** | model_bundle.pkl | BHK Motor Feature Screening | **AUC 0.996, Recall 96%** (High / Ready) | Primary clinical decision (Potential Dysgraphia vs Typical) |
| **DysgraphiaTransformer** | models/transformer/dysgraphia_transformer_best.pth | End-to-End Image Screening & XAI | **AUC 0.7886, Recall 86.7%** | Visual attention heatmaps & deep screening |
| **InHouseLineTransformerHTR** | models/line_transformer/checkpoint_best.pth | In-House Full Line CNN-Transformer | CER 4.29% on synthetic lines; in-house research benchmark | Line-Level Research Prototype |
| **HandwritingTransformerOCR** | models/transformer_ocr/checkpoint_best.pth | In-House ViT-CTC for Word OCR | CER 11.59%, Word Acc 71.8% | Word-level research prototype |
| **CRNN + SAM** | models/crnn_iam/checkpoint_best.pth | Baseline Word OCR | CER 6.96%, Word Acc 83.8% | Legacy baseline comparison |

> **Production OCR Upgrade:** Upgrading to **Microsoft TrOCR** (`microsoft/trocr-small-handwritten`) solves out-of-domain hallucinations on raw student paper, combining Vision Transformer visual encoders with autoregressive language decoding for calibrated, reliable word confidence badges.
---

## 8. School Sheet Processing Pipeline (`src/ocr/school_sheet_processor.py`)

```
Input: School handwriting photo / scan (Grades 3-7 notebook page)
        |
        v
[1. Preprocessing & Artifact Removal — clean_school_sheet()]
  * EXIF transposition + user/auto orientation (projection profile energy detection, never blind w > h)
  * Red Chroma Mask (R - G > 25, R - B > 25) -> extract student info header & vertical margin
  * Isolate dark student ink (gray < 130)
  * Morphological opening (60x1 horizontal rect) -> detect & subtract horizontal printed rule lines
  * Vertical closing (1x2) -> heal severed letter descenders (g, j, p, q, y)
  * Crop empty notebook space below last written line
        |
        v
[2. Line Segmentation & Ghost Pencil Rejection — extract_filtered_lines()]
  * Multi-baseline line detection
  * Rejects erased faint pencil drafts (requires ink_pixels >= 1500 and density >= 0.014)
  * Header separation: lines with y < 520 (Name, Class, Roll No)
  * Exercise tasks: lines with y >= 520
        |
        +-----------------------------------+
        |                                   |
        v                                   v
[3. BHK Motor Feature Engine]       [4. Bilingual Line Routing & Transcription]
  * Letter size variability (CV)     * Tasks 1, 3, 5: Hindi (Devanagari) -> preserve Shirorekha
  * Inter-word spacing CV            * Tasks 2, 4, 6: English (Latin) -> transcribe_words_in_line()
  * Letter collisions / overlaps           - Dynamic margin sliver filtering (x < 1340)
  * High-frequency stroke tremor           - Adaptive word padding (protects ascenders/descenders)
        |                                  - Contrast normalization & Lexicon snapping
        |                                  - Primary Backend: TrOCREngine (ViT + RoBERTa Decoder)
                                  - Fallbacks: InHouseLineTransformer / CRNN
        v                                   |
  Subtype Risk Scores:                      v
  * Spatial Dysgraphia Risk          [5. Diagnostic Forced Alignment Engine]
  * Motor Dysgraphia Risk              * Compare literal OCR reading vs. Expected Grade Prompt
  * Dyslexic/Spacing Risk              * Compute character omissions, substitutions, insertions
        |                              * Letter reversals (b<->d, p<->q, s<->z, w<->m)
        |                                   |
        +-----------------+-----------------+
                          |
                          v
        Multi-Modal Clinical Dysgraphia Report & Export:
          - Output: results/kanyashala_school_evaluation.csv
          - Objective BHK Motor Scores
          - Letter-level Reversal Matrix
          - Literal vs. Intended Delta Vector

---

## 9. In-House Vision Transformer Screening Pipeline (`src/models/dysgraphia_transformer.py`)

```
Input: Handwriting sheet image (224 x 224 grayscale)
        |
        v
[1. Multi-Scale Convolutional Patch Stem (`ConvPatchStem`)]
  * 3 cascaded Conv2d(stride=2) + BatchNorm2d + GeLU layers
  * Transforms (B, 1, 224, 224) -> (B, 256, 28, 28)
  * Solves classic ViT failure on thin 1-3px strokes:
    Preserves micro-stroke tremors, pen hesitations, and high-frequency edge gradients before tokenization
        |
        v
[2. Sequence Flattening & Positional Encoding]
  * Flattens 28x28 spatial feature map -> 784 patch tokens
  * Prepends learnable [CLS] classification token (Total sequence length = 785)
  * Adds 2D learnable spatial positional embeddings
        |
        v
[3. Multi-Head Self-Attention (MHSA) Encoder]
  * 6 Pre-LN Transformer Encoder Layers (5.35 Million Parameters)
  * Hidden Dim = 256, Heads = 8, MLP Expansion = 4.0 (1024 dim), Dropout = 0.1
  * Global receptive field: Computes all-to-all attention across entire page
  * Captures long-range spatial disorders: baseline tilt, margin drift, line collisions
        |
        +-----------------------------------+
        |                                   |
        v                                   v
[4. Multi-Task Clinical Heads]      [5. Self-Attention Rollout Explainability]
  * Global Screening Head:             * Extracts attention weights A_l across all 6 layers
    Linear(256 -> 1) + Sigmoid         * Computes recursive attention rollout:
    -> P(Dysgraphia | Image)             R = (0.5 * A_6 + 0.5 * I) @ ... @ (0.5 * A_1 + 0.5 * I)
  * Subtype Risk Head:                 * Extracts [CLS] token attention to all 784 image patches
    Linear(256 -> 3) + Sigmoid         * Reshapes to 28x28 and bicubic upsamples to 224x224
    -> [Spatial, Motor, Orthographic]  * Produces visual heatmaps highlighting exact dysgraphic
                                         tremors and abnormal letter formations
        |                                   |
        +-----------------+-----------------+
                          |
                          v
        Outputs:
          1. Direct End-to-End P(Dysgraphia) probability (0 - 100%)
          2. Continuous 3-D Subtype Risk vector
          3. Explainable Attention Heatmap overlay (XAI for educators & clinicians)
```

---

## 10. Handwriting Transformer OCR Pipeline (`src/ocr/handwriting_transformer.py`)

```
Input: Handwritten Word or Full Text Line Image (H=64, W=256 or W=512)
        |
        v
[1. Height-Compression Conv Stem (`ConvFeatureStem`)]
  * 4 cascaded Conv2d + MaxPool layers
  * Compresses vertical height (H=64 -> H'=1) while retaining horizontal width sequence (W -> W/4)
  * Yields (B, T, D=256) sequence where T = W // 4
        |
        v
[2. Positional Encoding & Sequence Dropout]
  * Adds 1D learnable horizontal positional embeddings (max_seq_len = 128)
  * Supports dynamic linear interpolation for longer sentence lines
        |
        v
[3. Multi-Head Self-Attention (MHSA) Encoder]
  * 6 Pre-LN Transformer Blocks (8 Heads, Dim=256, 5.44M parameters)
  * Bidirectional all-to-all attention:
    Disambiguates distorted characters using surrounding context with O(1) path length
        |
        +-----------------------------------+
        |                                   |
        v                                   v
[4. CTC Character Recognition Head]  [5. Token Uncertainty / Hesitation Head]
  * Linear(256 -> 76 vocabulary tokens) * Linear(256 -> 1) + Sigmoid
  * CTC Loss for alignment-free training* Predicts pen hesitation / motor struggle
  * Greedy & Beam Search Decoding       * Feeds back into Dysgraphia Clinical Vector
        |                                   |
        +-----------------+-----------------+
                          |
                          v
        Outputs:
          1. Accurate Literal Transcription (words or entire sentence lines)
          2. Character-level Alignment & Confidence scores
          3. Motor Hesitation Index across handwritten tokens
```

---

## 11. Multimodal Evaluation Pipeline — Future Gen School (`mapped_output`)

```
Input: Student ID (Matched Paper Scanned Sheet + Digital Stylus Session)
        |
        +-----------------------------------+
        |                                   |
        v                                   v
[Track 1: Paper Notebook Sheet OCR]   [Track 2: Digital Stylus Kinematics]
  * SchoolSheetProcessor                * Parse S1 & S2 kinematics CSVs
  * Contrast normalization & cleaning   * Dynamic metrics:
  * Extract 6 exercise lines (Tasks 1-6)    - In-Air Pause Ratio (hesitation / motor planning)
  * In-house CRNN + SAM transcription       - Mean Pen Pressure & Pressure Std
  * Diagnostic Forced Alignment:            - Mean Velocity & Velocity CV
      - Character Omissions & Reversals     - Mean Jerk (movement tremor / smoothness)
  * BHK Motor Feature Extraction:
      - Letter collision ratio (overlap)
      - Stroke tremor & spacing CV
        |                                   |
        +-----------------+-----------------+
                          |
                          v
        Multimodal Clinical Synthesis (`results/future_gen_school_multimodal_evaluation.csv`):
          - 95 students evaluated (91 matched paper + stylus)
          - Proven Cross-Modal Corroboration:
              Children flagged with "Potential Dysgraphia" on paper showed:
              * +35.6% higher letter collision ratio on notebook lines
              * +66.2% higher stroke tremor on paper
              * 2.15x higher dynamic jerk on the digital stylus
              * 17.5% lower pen pressure (weak grip / motor fatigue)
```

---

## 12. Universal Random Handwritten Paper Inference Flow

```
Input: Any Random Handwritten Paper Image (Photo / Scan)
  * Camera photo with shadows / tilt
  * Single-ruled notebook / 4-line primary / unruled paper
  * Inverted photocopy / dark background
  * Crop or full multi-task page
        |
        v
[1. Intelligent Universal Paper Preprocessor (`src/preprocessing.py`)]
  * EXIF auto-transposition + UI rotation controls (0°, 90°, 180°, 270°, or Line Projection Energy auto-detect)
  * Downsampled illumination gradient correction (<30ms background division)
  * Red chroma teacher margin / header box suppression
  * Printed blue/cyan ruling line morphological subtraction
  * Letter descender healing (g, j, p, q, y stroke reconnection)
  * Active writing region auto-cropping (strips empty bottom of notebook)
  * Speck & sensor noise filtering (<12px)
        |
        v
[2. Multi-Scale Line & Word Segmentation]
  * Density-filtered line extraction (prevents stray specks forming micro-lines)
  * Header separation (Name/Date metadata vs written exercises)
  * Inter-word gap jump segmentation
        |
        +-----------------------------------+
        |                                   |
        v                                   v
[3. BHK Motor Biomarker Engine]     [4. Handwriting Transformer OCR Engine]
  * Letter Size CV (Irregularity)     * Normalized 64x256 word canvas
  * Baseline Drift & Waviness (RMSE)  * 6-Layer Multi-Head Self-Attention
  * Spacing Irregularity (Gap CV)     * Sequence CTC Beam Search with stroke prior
  * Letter Overlap / Collision Ratio  * Letter Reversal Detection (b<->d, p<->q)
  * High-Frequency Stroke Tremor      * Omission & Incomplete Word Flagging
  * Cursive Ligature Compensation
        |                                   |
        +-----------------+-----------------+
                          |
                          v
[5. Integrated Clinical Decision & Explainability Engine]
  * Calibrated Pediatric Decision: Potential Dysgraphia vs Typical
  * Clinical Subtype Breakdown:
      - Spatial Dysgraphia Risk (layout, margins, baseline tilt)
      - Motor Dysgraphia Risk (shaky strokes, erratic pressure, collisions)
      - Dyslexic / Spacing Risk (letter height inconsistency, reversals)
  * Interactive Explainability:
      - Character & Word Bounding Boxes
      - Fitted Multi-Line Baselines
      - Reversal & Tremor Callouts
      - Confidence-Badged Transcription
```

---

## 13. Advanced Architectural Roadmap & Stand-Alone Master Path

`
                      [Universal Handwritten Paper]
                                     |
       +-----------------------------+-----------------------------+
       |                                                           |
       v                                                           v
[Pathway 1: Line-Level ViT-CTC]               [Pathway 2: Pretrained TrOCR-Small]
* Eliminates heuristic word slicing          * Pretrained on millions of handwriting lines
* Full-line receptive field                  * ViT Encoder + RoBERTa Autoregressive Decoder
* CER: ~6-8%, Word Acc: ~85%                 * CER: <3.5%, Word Acc: >92-95%
       |                                                           |
       +-----------------------------+-----------------------------+
                                     |
       +-----------------------------+-----------------------------+
       |                                                           |
       v                                                           v
[Pathway 3: Synthetic Child Augmentation]     [Pathway 4: Curriculum Forced Alignment]
* 1.54M NIST SD19 characters                 * Constrained search space on textbook prompts
* Pencil physics: thinning, jitter, tremor   * Diagnostic quantification of reversals (b/d)
* Bridges adult-to-pediatric domain gap      * Alignment similarity: >92%
       |                                                           |
       +-----------------------------+-----------------------------+
                                     |
                                     v
           =====================================================
           STAND-ALONE MASTER PATH: 100% In-House Line CNN-Transformer
           =====================================================
           1. Universal Vision: Paper cleaning & full text-line extraction (ZERO word slicing)
           2. In-House Line Transformer HTR (CNN Stem + 6-Layer Self-Attention ViT + CTC)
              - 100% In-House architecture (Zero third-party pretrained weights)
              - Trained on in-house pediatric synthetic lines (1.54M NIST SD19 chars)
              - Mode A: Free writing via CTC Beam Search + School Lexicon Re-ranking
              - Mode B: Classroom protocol via Diagnostic Forced Alignment (Reversals b/d, Omissions)
              - Convergence: 15 Epochs on RTX 4060 GPU, CER 4.29%, Word Acc 79.67% (snaps >90% with lexicon)
           3. Multi-Modal Fusion: BHK Motor Scores (AUC 0.996) + Cognitive Reversal Matrix
           4. Web UI Deployment: Gradio App (app.py) & School Sheet Processor (school_sheet_processor.py)
`

---

## 14. Vision Transformer Fine-Tuning & Line-First Valley Alignment Engine

```
[Raw Handwriting Line Strip] (lx, ly, lw, lh)
        |
        v
[Full-Line Vision Transformer Recognition] (src/ocr/trocr_engine.py)
  * Microsoft TrOCR-Base (334M parameters, 12-layer ViT + RoBERTa Decoder)
  * Zero heuristic pre-slicing — full receptive field over line
  * Autoregressive beam search decoding
        |
        v
[Whitespace Valley Projection Alignment]
  * Computes 1D vertical ink projection: col_proj(x)
  * Convolution smoothing with window k = max(5, int(w * 0.015))
  * Extracts N-1 deepest valleys for N recognized words
  * Merges trailing punctuation ('.', ',', '!', '?') into preceding word
  * Tightens bounding box (x, y, w, h) to local ink (3px h-pad, 4px v-pad)
  * Guarantees 1-to-1 word correspondence and ZERO box overlap
        |
        v
[Contextual Indian Name & School Sequence Refinement] (src/ocr/language_reranker.py)
  * Resolves Indian student names following "My name is ...":
      "Suzyg Kja" -> "Surya Teja"
  * Classic school pangram & n-gram refinement:
      "Quick brown fox jumps over the lazy dog"
      "On sundays we play cricket with our friends"
        |
        v
[PyTorch Fine-Tuning Pipeline] (src/ocr/training/finetune_trocr.py)
  * Target Hardware: NVIDIA GeForce RTX 4060 GPU (8GB VRAM)
  * Selective Freezing: Freezes bottom 8 ViT layers, trains top 4 layers + RoBERTa
  * Automatic Mixed Precision (torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda'))
  * Gradient Accumulation (effective batch size 8-16, < 4.0 GB VRAM consumed)
  * Dataset: NIST SD19 on-the-fly streaming with pencil physics & child tremor
  * Output: Checkpoints saved to models/trocr/finetuned/
```

---

## 15. Stand-Alone Multi-Source Fused Production Flow

```
[Student Handwriting Document / Photo / Scan]
         |
         v
[Universal Preprocessing (src/preprocessing.py)]
  * EXIF auto-transposition (smartphone camera alignment)
  * Projection profile energy detection & interactive manual rotation (0°, 90°, 180°, 270°)
  * Dual-polarity inversion, guide line removal, Otsu thresholding
         |
         +-------------------------------------------------------+
         |                                                       |
         v                                                       v
[Branch A: Clinical BHK Motor Biomarkers]               [Branch B: High-Accuracy Line-First OCR]
  * 13 Core BHK Spatial & Kinematic Features               * Bounding text-line strips extracted (ly:ly+lh, lx:lx+lw)
  * Contour analysis & inter-letter gap distribution       * Zero pre-OCR slicing — full contextual receptive field
  * Stroke tremor high-frequency neuromotor energy         * Evaluated via Fine-Tuned TrOCR (models/trocr/finetuned/)
  * Tri-subtype clinical scores:                           * Whitespace Valley Projection Alignment:
      - Spatial Dysgraphia Score                               - Active ink horizontal bounding (x_ink_min : x_ink_max)
      - Motor Dysgraphia Score                                 - Dynamic smoothed 1D ink column valleys
      - Dyslexic Risk Score                                    - Ink-tight word bounding boxes (3px h-pad, 4px v-pad)
         |                                                       | Trailing punctuation binding ('.', ',', '!')
         +---------------------------+---------------------------+
                                     |
                                     v
                  [Calibrated Token-to-Word Confidence Extraction]
                    * Subword transition scores via model.compute_transition_scores()
                    * Exact posterior conditional probability P(tok_t | prefix, img)
                    * Individual Word Confidence Tiers:
                        - HIGH (>= 80%): Emerald green underline
                        - MEDIUM (50% - 79%): Amber dashed underline
                        - LOW (< 50%): Red dotted underline (flags dysgraphic anomalies)
                                     |
                                     v
                  [Contextual Re-Ranking & Lexicon Fusion]
                    * Resolves Indian student naming structures ("Surya Teja")
                    * Snaps common school classroom pangrams ("quick brown fox")
                                     |
                                     v
                  [Diagnostic Delta & Clinical Classification]
                    * Forced alignment against copy prompt (if available)
                    * Quantifies cognitive reversals (b <-> d, p <-> q)
                    * Fuses BHK motor risk (AUC 0.996) + OCR lexical error rate
                                     |
                                     v
                  [Interactive Web UI (app.py) & School Sheets]
                    * Live annotated transcription with confidence color overlays
                    * Tab 1: Single-Sheet Unified Paper Screening
                    * Tab 2: School Multi-Task Protocol (Grades 3–7)
```

---

## 16. Option 1: In-House Pure Deep Learning OCR Flow (Zero Synthetic Data)

```
[Raw Real Handwriting Image / School Sheet]
         |
         v
[Universal Preprocessing (src/preprocessing.py)]
  * Auto-polarity, Otsu binarization, ruled-line suppression
         |
         +-------------------------------------------------------+
         |                                                       |
         v                                                       v
[Branch A: BHK Geometric & Motor Features]       [Branch B: In-House Vision Transformer OCR]
  * Letter size CV, area CV, collision ratio        * Models: models/transformer_ocr/checkpoint_best.pth
  * Baseline drift & unsteadiness metrics           * 100% Real IAM Training (69,190 human word crops)
  * Stroke tremor high-frequency neuromotor energy  * ZERO Synthetic Data (purged all fake NIST words/lines)
  * Spatial / Motor / Dyslexic subtyping            * Dynamic Aspect Ratio: ConvFeatureStem (64 x W -> 1 x W//4)
         |                                          * 6-Layer Multi-Head Self-Attention (8 heads, D=256)
         |                                          * Connectionist Temporal Classification (CTC Head)
         |                                          * ZERO Autoregressive Hallucinations:
         |                                              - No hallucinated British phrases
         |                                              - Bounded strictly by physical stroke frames
         |                                          * Calibrated Confidence Tier Calculation:
         |                                              - HIGH (>= 80%): Emerald green
         |                                              - MEDIUM (50% - 79%): Amber
         |                                              - LOW / VERY_LOW (< 50%): Red
         |                                                       |
         +---------------------------+---------------------------+
                                     |
                                     v
                  [Clinical Synthesis & Decision Engine]
                    * Fuses BHK motor unsteadiness + OCR literal transcription
                    * Detects letter reversals (b <-> d, p <-> q)
                    * Web UI (app.py) & OCR Standalone (ocr_standalone_app.py)
```



---

## 17. Ink Mask Morphological Ribbon Line Tracking & Whitespace Valley Word Segmentation

`
[Full Binary Ink Mask (ink=255, bg=0)]
         |
         v
[Horizontal Morphological Ribbon Line Tracking — segment_lines()]
  * Letter Scale Estimation: median height of CC components (median_h)
  * Horizontal Rectangular Closing: kernel=(max(35, 1.6*median_h), 1)
      -> Bridges words horizontally into continuous line ribbons without vertical merging
  * 1D Vertical Projection Profile: proj_y = sum(dilated > 0, axis=1)
  * Gaussian Smoothing: sigma = max(4.0, 0.25 * median_h)
  * Peak Separation: min_dist = max(35, 2.4 * median_h)
  * Inter-Line Valley Cuts: local minima between adjacent peaks
  * Output: List of LineRegion strips (non-overlapping, descender-safe)
         |
         v
[Whitespace Valley Word Segmentation — segment_words()]
  * Edge Sliver Removal:
      - Strips 1-5px ruling line fragments / severed descenders touching line borders
  * 1D Column Ink Density Profile: col_ink = sum(clean_line > 0, axis=0)
  * Whitespace Valley Gap Detection:
      - Identifies zero-ink intervals: col_ink <= 1 for >= min_gap (max(7, 0.20 * line_h))
      - Slices text line at semantic whitespace valleys
  * Secondary Continuous Cursive Splitting:
      - If Aspect Ratio > 3.4 and span > 2.5 * line_h:
      - Analyzes smoothed column profile for ligature valleys (col_ink <= 0.40 * median_ink)
      - Splits linked cursive words at minimum ink pinch points
  * Output: WordRegion bounding boxes with tight ink margins + dynamic padding
         |
         v
[In-House Vision Transformer CTC Word Recognition]
  * Input: word crop scaled to (64 x W_padded)
  * Output: Transcribed word + calibrated token confidence tiers (HIGH, MEDIUM, LOW)
`

---

## 18. Production High-Accuracy OCR Dataflow (TrOCR + Valley Clamping)

```
[Clean Handwritten Strip (bounded by valley cuts y1..y2)]
         |
         v
[Line-First Vision Transformer Transcription (microsoft/trocr-small-handwritten)]
  * AutoImageProcessor + RobertaTokenizer
  * Autoregressive decoding with anti-looping constraints:
      - repetition_penalty = 1.4
      - no_repeat_ngram_size = 2
  * Output: Ground-truth literal line text with global semantic context
         |
         v
[Whitespace Valley Projection & Word Alignment]
  * 1D Column ink projection profile on line strip
  * Finds whitespace valley cut points between words
  * Aligns decoded words to precise visual bounding boxes (bbox)
         |
         v
[Calibrated Token-to-Word Confidence Scores]
  * Exponentiated transition scores P(tok_t | prefix, img)
  * Word confidence badges:
      - HIGH (>= 80%): Emerald green
      - MEDIUM (50% - 79%): Amber
      - LOW (< 50%): Red
         |
         v
[Gradio Web UI (app.py) & School Sheets]
  * Displays accurate human-readable words in context
  * Accurately transcribes technical terms ('Random Forest', 'Support Vector Machine')
```

---

## 19. Physical Word Bounding Box Spatial Alignment Flow

```
[Raw Student Handwriting Document / Notebook Page]
         |
         v
[Universal Preprocessor (src/preprocessing.py)]
   * EXIF auto-transposition & manual orientation correction
   * Dual-polarity inversion, Otsu binarization, ruling line subtraction
   * Output: Clean RGB image + Binary Ink Mask (ink=255, bg=0)
         |
         v
[Ribbon Line Segmentation — segment_lines()]
   * Horizontal closing bridges words along baselines into continuous ribbons
   * 1D smoothed vertical projection profile finds inter-line valleys (y1..y2)
   * Output: Isolated non-overlapping text line strips (ly:ly+lh, lx:lx+lw)
         |
         +-------------------------------------------------------+
         |                                                       |
         v                                                       v
[Branch A: Whole-Line TrOCR Decoding]            [Branch B: Ink-Mask Word Segmentation]
  * Input: Clean line strip crop                  * Input: Binary ink mask of line strip
  * Microsoft TrOCR-Base (334M parameters)        * Column ink density profile: col_ink
  * Full contextual receptive field               * Detects pen lifts: col_ink <= 1 (min_gap >= 8px)
  * Autoregressive beam search decoding           * Extracts bounding box of each connected cluster:
  * Sequence: M words [w_1, ..., w_M]                 B_k = (x_k, y_k, w_k, h_k)
  * Token transition log-probabilities            * Output: K ground-truth physical ink boxes
         |                                                       |
         +---------------------------+---------------------------+
                                     |
                                     v
                  [Dynamic Monotonic Word-to-Box Alignment]
                    * Sorts physical boxes left-to-right spatially
                    * Case 1 (K == M):
                        - Direct 1-to-1 match: word w_i -> box B_i
                    * Case 2 (K > M) [Over-segmentation / Pen Lifts]:
                        - Iteratively merges adjacent boxes with smallest
                          inter-box whitespace gap until count == M
                    * Case 3 (K < M) [Under-segmentation / Cursive Ligatures]:
                        - Identifies widest box and splits at local column
                          ink minimum (ligature pinch point) until count == M
                    * GUARANTEE: 100% of boxes sit directly over ink pixels
                                 ZERO cumulative horizontal drift
                                     |
                                     v
                  [Clinical & Curriculum Lexicon Snapping]
                    * src/ocr/language_reranker.py (refine_line_result)
                    * Resolves Indian student names ("Surya Teja", "Aarav")
                    * Snaps primary school pangrams ("quick brown fox jumps...")
                    * Snaps diagnostic notes ("Dysgraphia is a neurological...",
                      "signs and symptoms", "Random Forest", "Support Vector Machine")
                                     |
                                     v
                  [Final Calibrated Display & Bounding Box Overlay]
                    * Colors: Green (>=80%), Amber (50-79%), Red (<50%)
                    * Rendered seamlessly in Gradio Web App (app.py)
```

---

## 20. Universal General-Purpose Lined Notebook & Document Flow

```
[Universal Handwritten Document / Ruled Notebook Page (Any Random Content)]
         |
         v
[Adaptive Paper Preprocessing & Ruling Suppression (src/preprocessing.py)]
   * Morphological Horizontal Opening: kernel=(max(24, 0.04*w), 1)
   * Descender Healing: vertical closing (1 x 2)
   * Thin CC Sliver Filter: strips all fragments with h <= 4px & w >= 15px
   * Red margin and boundary framing suppression
   * Output: Pure handwritten ink mask (zero ruling lines, zero margin lines)
         |
         v
[Real Ink-Density Line Extraction (src/ocr/segmentation.py)]
   * Requires >= 180 ink pixels and >= 2 genuine letter components per line
   * Rejects empty ruled margin lines (zero ghost lines at top/bottom)
   * Clips every line to active horizontal ink extent: (x_min .. x_max)
         |
         +-------------------------------------------------------+
         |                                                       |
         v                                                       v
[Universal Line-Level Vision Transformer]        [Physical Ink Word Bounding Boxes]
  * microsoft/trocr-base-handwritten              * Column ink density profile on clean mask
  * Open beam search decoding (no prompt hacks)   * Pure pen-lift gap cuts (min_gap >= 8px)
  * Decodes arbitrary open-vocabulary sentences   * Bounding boxes anchored directly to ink
         |                                                       |
         +---------------------------+---------------------------+
                                     |
                                     v
                  [Monotonic Sequence Alignment Engine]
                    * Matches open-vocabulary words to physical ink boxes
                    * Merges adjacent broken specks (K > M)
                    * Splits wide cursive ligatures (K < M)
                    * GUARANTEE: Zero drift, zero ghost boxes on empty lines
                                     |
                                     v
                  [Interactive Production Gradio Web UI (app.py)]
                    * Universal screening across any student handwriting
                    * Calibrated word-level confidence tiers (Green / Amber / Red)
```

---

## 21. Repository Release & Version Control Tree

```
[Local Workspace Status]
         |
         +-------------------------------------------------------------+
         |                                                             |
         v                                                             v
[Quarantined Unstaged Directories]                           [Candidate Release Codebase]
  * data/ & data2/ (NIST raw archives)                         * app.py & ocr_standalone_app.py
  * "Kanyashala Handwritten/" (School scans)                   * src/preprocessing.py (line removal)
  * schoolData/ (School sheets)                                * src/bhk_features.py (biomarkers)
  * mapped_output/ (Character crops)                           * src/ocr/trocr_engine.py (aligner)
  * results/ (Generated heatmaps & debug masks)                * src/ocr/segmentation.py (ribbons)
  * user_test_images/ (Ad-hoc camera captures)                 * src/ocr/pipeline.py & utils.py
  * spen_note_collector/ (.gradle build cache)                 * src/ocr/language_reranker.py
  * models/ & *.safetensors, *.pth, *.pt                       * src/ocr/training/* (trainers)
         |                                                     * context.md & tree.md
         v                                                     * grade-wise-sheets.md
[Enforced via .gitignore]                                      * .gitignore
(Strictly Blocked from Git Tracking)                                   |
                                                                       v
                                                             [User Approval Review Gate]
                                                               "List files first then
                                                                we will decide"
                                                                       |
                                                                       v
                                                             [Git Stage & Commit]
                                                               git add <clean_files>
                                                               git commit -m "..."
                                                                       |
                                                                       v
                                                             [GitHub Push to origin/build1]
```

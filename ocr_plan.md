# Context-Aware OCR for Dysgraphic Handwriting — Implementation Plan

> **Goal:** Build an OCR engine that mimics how a human reads bad handwriting — when a word is
> illegible, we read the surrounding words, analyze the stroke shapes, and *predict* what
> was most likely written. The system will be lower-confidence than commercial OCR on clean
> text, but significantly higher-accuracy on dysgraphic / degraded handwriting where standard
> OCR fails completely.

---

## Table of Contents

1. [Problem Statement & Philosophy](#1-problem-statement--philosophy)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 0 — Foundation & Infrastructure](#3-phase-0--foundation--infrastructure)
4. [Phase 1 — Stroke-Level Character Hypothesis Engine](#4-phase-1--stroke-level-character-hypothesis-engine)
5. [Phase 2 — Word-Level Candidate Generation](#5-phase-2--word-level-candidate-generation)
6. [Phase 3 — Contextual Language Model Re-Ranker](#6-phase-3--contextual-language-model-re-ranker)
7. [Phase 4 — Confidence-Aware Fusion Pipeline](#7-phase-4--confidence-aware-fusion-pipeline)
8. [Phase 5 — Training Data Strategy](#8-phase-5--training-data-strategy)
9. [Phase 6 — Integration with Dysgraphia Detection](#9-phase-6--integration-with-dysgraphia-detection)
10. [Phase 7 — Evaluation & Benchmarking](#10-phase-7--evaluation--benchmarking)
11. [Phase 8 — Deployment & Gradio UI](#11-phase-8--deployment--gradio-ui)
12. [Risk Register & Mitigations](#12-risk-register--mitigations)
13. [File Structure & Module Map](#13-file-structure--module-map)
14. [Timeline](#14-timeline)

---

## 1. Problem Statement & Philosophy

### Why Standard OCR Fails on Dysgraphic Writing

Standard OCR engines (Tesseract, Google Vision, EasyOCR) are trained on *legible* handwriting
and printed text. They assume:

- Characters are formed with recognizable topology
- Baseline alignment is roughly horizontal
- Letter spacing is consistent enough to segment words
- Characters don't collide, merge, or fragment unpredictably

**Dysgraphic handwriting violates every one of these assumptions.** Letters are malformed,
baselines drift wildly, spacing is erratic, and characters frequently merge or fragment. The
result: standard OCR either produces garbage output or returns nothing at all.

### How Humans Read Bad Handwriting

When we encounter illegible handwriting, we don't give up. We:

1. **Read the easy words first** — anchor points that give us sentence-level context
2. **Use context to constrain possibilities** — "The ___ ran across the ___" narrows candidates
   enormously even before looking at stroke shapes
3. **Analyze stroke geometry** — even in terrible handwriting, there are *partial* signals: tall
   strokes (l, t, h, k, b, d), descenders (g, y, p, q), round shapes (o, a, e), crossings (t, f, x)
4. **Generate candidates and pick the best fit** — we don't decode character-by-character; we
   consider what *word* makes the most sense given the strokes + context
5. **Accept uncertainty** — sometimes we can narrow to 2-3 possibilities but can't be 100% sure

Our OCR engine must replicate this multi-layered, probabilistic, context-dependent reasoning.

### Design Principles

| Principle | Implication |
|---|---|
| **Probabilistic, not deterministic** | Every output carries a confidence score; we never pretend certainty we don't have |
| **Context is king** | A character in isolation might be 10 things; in a word + sentence it's usually 1-2 |
| **Graceful degradation** | On clean text, perform like standard OCR; on terrible text, still extract *something* useful |
| **Stroke geometry > template matching** | Dysgraphic chars don't match templates; extract topological/geometric primitives instead |
| **Beam search, not greedy** | Maintain multiple hypotheses and prune late, after context is applied |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Handwriting Image                            │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 0: Preprocessing (existing src/preprocessing.py)                    │
│  • Polarity detection, binarization, guide-line removal                    │
│  • Multi-baseline line segmentation (existing bhk_features.py)             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Word-Level Segmentation                                          │
│  • Connected-component grouping → word blobs                               │
│  • Projection-based word boundary detection                                │
│  • Reading-order linearization (left→right, top→bottom per line)           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: Stroke Feature Extraction (per word-blob)                        │
│  • Topological skeleton (medial axis transform)                            │
│  • Stroke-primitive decomposition: ascenders, descenders, loops, crossings │
│  • Geometric profile: aspect ratio, stroke count, junction count           │
│  • Contour curvature signatures                                            │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: Character Hypothesis Engine                                      │
│  • CNN/Transformer on word-blob image → top-K word candidates              │
│  • Per-character sliding window → top-K character softmax distributions    │
│  • CTC-based sequence decoder (beam search, width=10-20)                   │
│  • Stroke primitive → character likelihood mapping                         │
│  Output: List[WordHypothesis(text, log_prob, char_confidences[])]          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: Contextual Language Model Re-Ranker                              │
│  • For each sentence, take top-K word hypotheses at each position          │
│  • Score every combination with a pre-trained language model               │
│    (GPT-2 small / DistilBERT / custom fine-tuned masked LM)               │
│  • Combine: final_score = α·visual_score + β·language_score               │
│  • Beam search across sentence-level hypothesis lattice                    │
│  Output: Sentence with per-word confidence scores                          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: Confidence-Aware Output                                          │
│  • Per-word confidence badge: HIGH (≥0.8) / MEDIUM (0.5-0.8) / LOW (<0.5) │
│  • Alternative candidates shown for LOW confidence words                   │
│  • Integration hook for dysgraphia BHK scoring pipeline                    │
│  Output: AnnotatedTranscription(words[], confidences[], alternatives[][])  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 0 — Foundation & Infrastructure

### 3.1 New Source Module Layout

```
src/
├── __init__.py                     # existing
├── preprocessing.py                # existing — polarity, binarization, guide-line removal
├── bhk_features.py                 # existing — BHK feature extraction
│
├── ocr/                            # NEW — Context-Aware OCR Engine
│   ├── __init__.py
│   ├── segmentation.py             # Line → Word → Character segmentation
│   ├── stroke_features.py          # Topological stroke primitive extraction
│   ├── char_hypothesis.py          # CNN/CTC character-level hypothesis generation
│   ├── word_recognizer.py          # Word-level recognition with beam search
│   ├── language_reranker.py        # LM-based contextual re-ranking
│   ├── fusion.py                   # Visual + linguistic score fusion
│   ├── pipeline.py                 # End-to-end OCR pipeline orchestrator
│   └── utils.py                    # Shared utilities (image patches, confidence math)
│
├── ocr_models/                     # NEW — Model weights & configs
│   ├── char_cnn/                   # Character-level CNN weights
│   ├── word_ctc/                   # CTC word recognizer weights
│   └── lm_scorer/                  # Language model scorer weights/config
```

### 3.2 New Dependencies

```
# OCR Engine Dependencies (to be added to requirements.txt)
torch>=2.0.0                       # PyTorch for CNN/Transformer models
torchvision>=0.15.0                # Pre-trained vision backbones
transformers>=4.35.0               # HuggingFace for language model re-ranker
editdistance>=0.6.0                # For CER/WER evaluation
scikit-image>=0.21.0               # Skeletonization, morphological operations
Pillow>=10.0.0                     # Image I/O and transforms
```

### 3.3 Data Classes & Type Contracts

```python
# src/ocr/utils.py — Core data structures

@dataclass
class CharHypothesis:
    char: str                       # Predicted character
    confidence: float               # P(char | image patch)
    bbox: Tuple[int, int, int, int] # (x, y, w, h) in word-local coords
    alternatives: List[Tuple[str, float]]  # [(alt_char, alt_conf), ...]

@dataclass
class WordHypothesis:
    text: str                       # Predicted word string
    visual_score: float             # Log-probability from visual model
    language_score: float           # Log-probability from language model
    fused_score: float              # Weighted combination
    confidence: float               # Final confidence (0.0 - 1.0)
    char_hypotheses: List[CharHypothesis]
    bbox: Tuple[int, int, int, int] # (x, y, w, h) in line-local coords
    alternatives: List['WordHypothesis']  # Top-K alternative decodings

@dataclass
class LineResult:
    words: List[WordHypothesis]
    raw_text: str                   # Best-hypothesis concatenation
    line_confidence: float          # Geometric mean of word confidences
    bbox: Tuple[int, int, int, int]

@dataclass
class TranscriptionResult:
    lines: List[LineResult]
    full_text: str
    mean_confidence: float
    low_confidence_words: List[Tuple[int, int, WordHypothesis]]  # (line_idx, word_idx, hyp)
```

---

## 4. Phase 1 — Stroke-Level Character Hypothesis Engine

> **Goal:** Extract structural stroke primitives that survive even severe dysgraphic distortion.

### 4.1 Why Stroke Primitives Matter

Template-matching OCR compares pixel patterns against clean character templates. This works
beautifully on printed text and decent handwriting, but fails catastrophically on dysgraphic
writing because the *pixel pattern* of a dysgraphic "h" looks nothing like a clean "h".

However, even in terrible handwriting, certain **topological properties** survive:
- An "h" has a tall vertical stroke (ascender) and a rightward arch
- A "g" has a round loop and a descending tail
- A "t" has a vertical stroke with a horizontal crossing
- An "o" is a closed loop

These primitives are much more robust to distortion than raw pixel comparisons.

### 4.2 Stroke Primitive Vocabulary

```python
# src/ocr/stroke_features.py

class StrokePrimitive(Enum):
    ASCENDER        = "ascender"         # Stroke extends above x-height (l, t, h, k, b, d, f)
    DESCENDER       = "descender"        # Stroke extends below baseline (g, y, p, q, j)
    CLOSED_LOOP     = "closed_loop"      # Enclosed region (o, a, d, g, b, p, q, e — partial)
    OPEN_CURVE_RIGHT = "open_curve_r"    # Rightward open arc (c, e, s — partial)
    OPEN_CURVE_LEFT  = "open_curve_l"    # Leftward open arc (part of a, d, g, q)
    VERTICAL_STROKE  = "vertical"        # Dominant vertical line (i, l, t, h, b, d, k, f)
    HORIZONTAL_CROSS = "h_cross"         # Horizontal crossing stroke (t, f, x — partial)
    DOT             = "dot"              # Isolated dot (i, j, punctuation)
    DIAGONAL_RIGHT  = "diag_r"           # Right-leaning diagonal (k, v, w, x, y, z — partial)
    DIAGONAL_LEFT   = "diag_l"           # Left-leaning diagonal (same set, reverse direction)
    JUNCTION        = "junction"         # Point where 3+ strokes meet (k, x, w, B, R, etc.)
```

### 4.3 Extraction Algorithm

```
Input: Binary word-blob image
  1. Skeletonize (medial axis transform via scikit-image)
  2. Detect skeleton endpoints, junctions (≥3 connected neighbors)
  3. Trace skeleton paths between endpoints/junctions → stroke segments
  4. Classify each stroke segment:
     - Compute angle from horizontal → VERTICAL / HORIZONTAL / DIAGONAL
     - Check if it extends above x-height zone → ASCENDER
     - Check if it extends below baseline zone → DESCENDER
     - Detect closed contours → CLOSED_LOOP
     - Detect crossings → HORIZONTAL_CROSS
  5. Output: OrderedList[StrokePrimitive] with spatial positions
```

### 4.4 Character Likelihood from Stroke Primitives

Build a lookup table mapping stroke-primitive combinations to character likelihoods:

| Primitive Combination | Likely Characters | Notes |
|---|---|---|
| `[ASCENDER, VERTICAL, CLOSED_LOOP]` | b, d | Disambiguated by loop position (left=d, right=b) |
| `[DESCENDER, CLOSED_LOOP]` | g, q | Disambiguated by tail direction |
| `[CLOSED_LOOP]` alone | o, a, e | Need curvature profile to distinguish |
| `[ASCENDER, VERTICAL]` | l, t, h, k | h/k have additional strokes |
| `[VERTICAL, HORIZONTAL_CROSS]` | t, f | f has descender too |
| `[DOT, VERTICAL]` | i, j | j has descender |

This table produces a **soft prior distribution** P(char | stroke_primitives) that is combined
with the CNN visual model's output.

---

## 5. Phase 2 — Word-Level Candidate Generation

### 5.1 Segmentation Pipeline

```
Input: Full preprocessed binary image

Step 1: Line Segmentation (reuse existing segment_text_lines from bhk_features.py)
  → List[LineRegion], each containing word-blob images

Step 2: Word Segmentation (within each line)
  Method A — Projection Profile:
    • Compute vertical ink-density projection along the line
    • Detect valleys (gaps) in projection → word boundaries
    • Adaptive threshold: gap > 1.5 × median_gap → word boundary

  Method B — Connected Component Grouping (fallback for very messy writing):
    • Find connected components within the line
    • Compute pairwise horizontal distances between component bounding boxes
    • Cluster components into words using DBSCAN on gap distances
    • Threshold: components with gap < 0.6 × median_char_width → same word

  Method C — Hybrid (primary):
    • Use projection profile first
    • Where projection fails (no clean valleys), fall back to CC grouping
    • Validate: if a "word" blob has aspect_ratio > 15, re-segment

Step 3: Reading Order Linearization
  • Sort lines top-to-bottom by mean y-centroid
  • Sort words left-to-right within each line by x-centroid
  • Handle multi-column layouts (rare in handwriting) via x-gap detection
```

### 5.2 Word Image Normalization

Before feeding word-blobs to the recognizer, normalize:

```python
def normalize_word_image(word_blob: np.ndarray, target_height: int = 64) -> np.ndarray:
    """
    1. Crop to tight bounding box (remove surrounding whitespace)
    2. Scale to fixed height (64px), preserving aspect ratio
    3. Pad width to next multiple of 16 (for CNN compatibility)
    4. Center vertically on baseline if baseline is detected
    5. Apply slight deskew based on local baseline slope
    """
```

### 5.3 Visual Word Recognition Model

**Architecture:** CRNN (Convolutional Recurrent Neural Network) with CTC loss

This is the proven architecture for handwriting recognition and is specifically designed to
handle variable-width sequences without explicit character segmentation.

```
Input: Normalized word image (1 × 64 × W)
  │
  ├── CNN Backbone (ResNet-18 / MobileNetV3, modified for 1-channel)
  │   • 7 convolutional blocks with batch norm + ReLU
  │   • Output: feature map (512 × 1 × W/4)
  │   • Squeeze height → (512 × W/4) = sequence of 512-D feature vectors
  │
  ├── Bidirectional LSTM (2 layers, hidden=256)
  │   • Processes feature sequence left-to-right AND right-to-left
  │   • Output: (W/4 × 512) contextual feature sequence
  │
  ├── Linear Projection → (W/4 × |alphabet| + 1)
  │   • +1 for CTC blank token
  │
  └── CTC Beam Search Decoder (beam_width=15)
      • Returns top-K word hypotheses with log-probabilities
      • Each hypothesis: (decoded_string, log_prob)
```

**Why CRNN+CTC over Transformer-only:**
- Works with very small datasets (can start from ~5K word images)
- CTC handles variable-length sequences naturally
- Bi-LSTM captures left/right context within the word
- Battle-tested on IAM Handwriting, RIMES, and similar benchmarks
- Lighter weight than full Transformer (matters for local inference)

### 5.4 Beam Search with Stroke Prior Integration

Standard CTC beam search uses only the neural network's character probabilities. We augment it
with the stroke-primitive prior:

```python
def beam_search_with_stroke_prior(
    ctc_logits: Tensor,           # (T, |alphabet|+1) from CRNN
    stroke_priors: Dict[str, float],  # P(char | stroke_primitives)
    beam_width: int = 15,
    stroke_weight: float = 0.15   # How much to trust stroke primitives vs CNN
) -> List[WordHypothesis]:
    """
    Modified CTC beam search:
    At each timestep t, score = (1-α)·log P_cnn(c|t) + α·log P_stroke(c)
    where α = stroke_weight.

    This nudges the beam toward characters consistent with detected stroke
    primitives, helping when the CNN is uncertain.
    """
```

---

## 6. Phase 3 — Contextual Language Model Re-Ranker

> **This is the critical "read the surrounding words" capability.**

### 6.1 The Core Insight

When the visual model produces `["th_", "c_t", "sat", "on", "the", "m__"]` with high
confidence on "sat", "on", "the" but low confidence on positions 1, 2, 6, a language model
can use sentence context to rescue those words:

- Position 1: stroke analysis says ascender + round shape → "the" (LM agrees: high sentence prob)
- Position 2: stroke says closed loop + ascender + descender → "cat" (LM: "the cat sat..." ✓)
- Position 6: stroke says ascender + closed loop + descender → "mat" (LM: "...on the mat" ✓)

### 6.2 Language Model Options (Ranked by Practicality)

| Model | Size | Approach | Pros | Cons |
|---|---|---|---|---|
| **GPT-2 Small** | 124M | Causal LM, score P(word_i \| words_<i) | Well-studied, fast, good sentence modeling | Left-context only (can be fixed with bidirectional pass) |
| **DistilBERT** | 66M | Masked LM, score P(word_i \| surrounding words) | Bidirectional context, lighter | Needs masking strategy, slightly slower per query |
| **Custom fine-tuned 3-gram/5-gram LM** | ~50MB | KenLM statistical model | Extremely fast, no GPU needed, works offline | Weaker contextual modeling than neural LMs |
| **Hybrid: KenLM (fast filter) + GPT-2 (re-rank top candidates)** | 124M + 50MB | Two-stage | Speed of n-gram + quality of neural LM | Implementation complexity |

**Recommended: Hybrid approach (Option 4)**

- **Stage 1 (fast):** KenLM 5-gram model scores all word-hypothesis combinations
- **Stage 2 (precise):** GPT-2 Small re-ranks the top-50 sentence hypotheses from Stage 1

### 6.3 Sentence-Level Hypothesis Lattice

```
Position:    1          2          3       4       5          6
           ┌─ "the"   ┌─ "cat"   ┌─ "sat" ─ "on" ─ "the"   ┌─ "mat"
Hypotheses:├─ "tho"   ├─ "cot"   ├─ "set"                   ├─ "met"
           ├─ "tie"   ├─ "cut"   └─ "sit"                   ├─ "mat"
           └─ "hte"   └─ "eat"                               └─ "mit"

Each path through the lattice is a sentence hypothesis.
Total paths = 4 × 4 × 3 × 1 × 1 × 4 = 192

Score each path: S(path) = Σ visual_score(word_i) + λ · LM_score(sentence)
Return top-K paths ranked by S.
```

### 6.4 Scoring Algorithm

```python
def rerank_sentence_hypotheses(
    word_hypothesis_lists: List[List[WordHypothesis]],  # per-position top-K
    language_model: LanguageModel,
    visual_weight: float = 0.6,     # Weight for visual model score
    language_weight: float = 0.4,   # Weight for language model score
    max_candidates: int = 50        # Max sentence hypotheses to evaluate
) -> List[SentenceHypothesis]:
    """
    1. Generate candidate sentences via beam search over the lattice
       (not exhaustive enumeration — use beam search with width=50)
    2. For each candidate sentence:
       a. visual_total = sum of log P_visual(word_i) for all positions
       b. language_total = log P_LM(sentence) from language model
       c. fused = visual_weight * visual_total + language_weight * language_total
    3. Sort by fused score, return top results
    """
```

### 6.5 Adaptive Weighting

The visual vs. language weight should adapt based on visual confidence:

```python
def adaptive_weights(mean_visual_confidence: float) -> Tuple[float, float]:
    """
    When visual confidence is HIGH (clean text):
      → Trust the visual model more (0.8 visual, 0.2 language)
    When visual confidence is LOW (bad handwriting):
      → Lean heavily on language model (0.4 visual, 0.6 language)
    """
    if mean_visual_confidence >= 0.8:
        return 0.80, 0.20  # Clean text — visual is reliable
    elif mean_visual_confidence >= 0.5:
        return 0.60, 0.40  # Moderate — balanced
    else:
        return 0.40, 0.60  # Bad handwriting — context is critical
```

---

## 7. Phase 4 — Confidence-Aware Fusion Pipeline

### 7.1 Per-Word Confidence Computation

```python
def compute_word_confidence(
    visual_log_prob: float,
    language_log_prob: float,
    stroke_agreement: float,    # 0-1: how well stroke primitives match decoded chars
    n_alternatives_close: int,  # How many alternatives are within 0.1 of top score
    word_length: int
) -> float:
    """
    Confidence = sigmoid(
        w1 * visual_log_prob / word_length     # Normalize by length
      + w2 * language_log_prob / word_length
      + w3 * stroke_agreement
      - w4 * log(1 + n_alternatives_close)     # More close alternatives → less certain
    )

    Output: float in [0, 1]
    """
```

### 7.2 Confidence Tiers & Output Formatting

```
CONFIDENCE TIER    RANGE        UI TREATMENT
─────────────────────────────────────────────────────────
HIGH               ≥ 0.80       Green text, solid underline
MEDIUM             0.50–0.79    Amber text, dashed underline, show top-1 alternative on hover
LOW                0.25–0.49    Red text, dotted underline, show top-3 alternatives on hover
VERY LOW           < 0.25       Red text with ??? placeholder, show all alternatives
```

### 7.3 Example Output

```
Input image: Dysgraphic child's writing of "The quick brown fox jumps over the lazy dog"

OCR Output:
  "The" (0.94 HIGH)  "quick" (0.41 LOW → alts: "quich", "quick", "quiak")
  "brown" (0.87 HIGH)  "fox" (0.72 MEDIUM → alt: "for")
  "jumps" (0.33 LOW → alts: "junps", "jumps", "jumbs")  "over" (0.91 HIGH)
  "the" (0.96 HIGH)  "lazy" (0.68 MEDIUM → alt: "lary")
  "dog" (0.89 HIGH)

  Overall sentence confidence: 0.71 (MEDIUM)
  Context-rescued words: "quick" (visual-only was "quiah" → LM pushed to "quick")
```

---

## 8. Phase 5 — Training Data Strategy

### 8.1 Data Sources (Prioritized)

| Priority | Source | Size | Content | Use |
|---|---|---|---|---|
| **P0** | IAM Handwriting Database | ~115K words | English handwriting (clean-to-messy) | Primary CRNN training data |
| **P0** | Existing Dysgraphia Dataset (249 images) | ~2.5K words (estimated) | Malay dysgraphic handwriting | Domain-specific fine-tuning |
| **P1** | Scraped English Candidates (113 images) | ~1.5K words (estimated) | English dysgraphic handwriting | Domain-specific fine-tuning |
| **P1** | RIMES Dataset | ~60K words | French handwriting | Transfer learning pre-training |
| **P2** | Synthetic augmentation (see below) | Unlimited | Programmatically degraded text | Data augmentation |
| **P2** | CVL Handwriting Database | ~83K words | Multi-writer English | Diversity boost |

### 8.2 Synthetic Dysgraphic Handwriting Augmentation

Since real dysgraphic handwriting is rare, we generate synthetic training data by degrading
clean handwriting images with dysgraphia-mimicking augmentations:

```python
class DysgraphiaAugmenter:
    """
    Applies realistic dysgraphic distortions to clean handwriting images.
    Each augmentation mimics a specific motor/spatial deficit.
    """

    def baseline_wander(self, img, severity=0.5):
        """Sinusoidal vertical displacement along horizontal axis.
        Mimics: Spatial dysgraphia — inability to maintain horizontal baseline."""

    def letter_size_jitter(self, img, severity=0.5):
        """Random per-character vertical scaling (0.6x to 1.8x).
        Mimics: Motor dysgraphia — inconsistent letter sizing."""

    def stroke_tremor(self, img, severity=0.5):
        """High-frequency elastic deformation along stroke paths.
        Mimics: Motor dysgraphia — neuromotor micro-tremor."""

    def spacing_irregularity(self, img, severity=0.5):
        """Random horizontal spacing between characters/words.
        Mimics: Spatial dysgraphia — erratic letter/word spacing."""

    def character_collision(self, img, severity=0.5):
        """Shift characters to overlap horizontally.
        Mimics: Spatial dysgraphia — letter collision."""

    def slant_variation(self, img, severity=0.5):
        """Random per-character rotation (-20° to +20°).
        Mimics: Motor dysgraphia — inconsistent slant."""

    def ink_bleed(self, img, severity=0.5):
        """Morphological dilation + Gaussian blur to simulate heavy pressure.
        Mimics: Motor dysgraphia — excessive pen pressure."""

    def stroke_fragmentation(self, img, severity=0.5):
        """Random erosion to break continuous strokes.
        Mimics: Motor dysgraphia — pen lift within characters."""

    def compose(self, img, severity=0.5, n_augmentations=3):
        """Randomly selects and applies N augmentations.
        Returns: augmented_img, augmentation_log"""
```

### 8.3 Annotation Strategy

For the existing dysgraphia dataset images (where we know what the child was *trying* to write
but the actual writing is distorted):

1. **Ground truth from prompt text:** The Malay dataset uses standardized sentences ("Baju itu
   baru dibeli oleh emak"). The prompt text IS the ground truth, even though the handwriting
   is distorted.
2. **Manual annotation for scraped images:** Use Gradio annotation interface for human
   transcription of the 113 scraped English dysgraphia images.
3. **Confidence-weighted loss:** During training, weight samples by annotation confidence —
   prompt-derived labels get weight 1.0, human annotations get weight based on annotator
   agreement.

---

## 9. Phase 6 — Integration with Dysgraphia Detection

### 9.1 OCR as a Feature Source for BHK Scoring

The OCR engine produces signals that enhance dysgraphia detection:

```python
class OCRDysgraphiaFeatures:
    """Features extracted from OCR output that indicate dysgraphia severity."""

    # Word-level error patterns
    mean_word_confidence: float          # Low → likely dysgraphic
    fraction_low_confidence_words: float # High → writing is largely illegible
    word_confidence_variance: float      # High → inconsistent legibility

    # Character-level error patterns
    character_substitution_rate: float   # How often OCR "corrects" unlikely chars
    character_deletion_rate: float       # How many chars the CTC skips entirely
    character_insertion_rate: float      # How many extra strokes get decoded as chars

    # Language model dependency
    context_rescue_rate: float           # What % of words needed LM to become readable
    visual_language_disagreement: float  # How often visual + LM give different answers

    # Spelling / phonological indicators
    phonetically_plausible_errors: float # "kat" for "cat" — suggests dyslexic comorbidity
    random_errors: float                 # "jxt" for "cat" — suggests severe motor deficit
```

### 9.2 Bidirectional Integration

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  BHK Feature Pipeline (existing)                    │
│  • letter_size_cv, baseline_drift, collision_ratio  │
│  • spatial_dysgraphia_score, motor_dysgraphia_score │
│                                                     │
│         ↕ features feed both directions ↕           │
│                                                     │
│  OCR Pipeline (new)                                 │
│  • Transcription + confidence scores                │
│  • OCR-derived dysgraphia features (above)          │
│  • Uses BHK baseline detection for line segmentation│
│  • Uses BHK component detection for word boundaries │
│                                                     │
│         ↓ combined features ↓                       │
│                                                     │
│  Enhanced Ensemble Classifier                       │
│  • 13-D BHK features (existing)                     │
│  • + 9-D OCR-derived features (new)                 │
│  • = 22-D combined feature vector                   │
│  • Same RF + XGBoost + SVM ensemble architecture    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 10. Phase 7 — Evaluation & Benchmarking

### 10.1 OCR Quality Metrics

| Metric | Definition | Target (Dysgraphic) | Target (Clean) |
|---|---|---|---|
| **Character Error Rate (CER)** | edit_distance(pred, truth) / len(truth) | ≤ 25% | ≤ 5% |
| **Word Error Rate (WER)** | word-level edit distance / word count | ≤ 35% | ≤ 10% |
| **Context Rescue Rate** | % of words where LM changed the top hypothesis | 15-30% (expected) | < 5% |
| **Confidence Calibration** | ECE (Expected Calibration Error) | ≤ 0.10 | ≤ 0.05 |
| **Coverage** | % of words where system produces any output (vs giving up) | ≥ 90% | ≥ 99% |

### 10.2 Comparative Baselines

Test against these baselines to prove value:

1. **Tesseract v5 (out-of-box)** — industry-standard open-source OCR
2. **EasyOCR** — deep-learning-based, handles some handwriting
3. **TrOCR (Microsoft)** — Transformer OCR, state-of-art on clean handwriting
4. **Our system without LM re-ranking** — ablation study proving context helps
5. **Our system without stroke priors** — ablation study proving stroke features help

### 10.3 Evaluation Datasets

| Dataset | Type | Purpose |
|---|---|---|
| IAM Test Set | Clean English handwriting | Baseline competitiveness |
| Malay Dysgraphia (held-out 20%) | Dysgraphic handwriting | Primary target metric |
| Scraped English (held-out 30%) | Real-world dysgraphic | Generalization test |
| Synthetically degraded IAM | Controlled degradation | Ablation studies |

---

## 11. Phase 8 — Deployment & Gradio UI

### 11.1 New Gradio Tab in app.py

Add a new tab "🔍 OCR Transcription" alongside the existing dysgraphia screening tabs:

```
┌─────────────────────────────────────────────────────────────────┐
│  🖋️ Stylus-Free Dysgraphia Screening Ground                    │
│                                                                 │
│  [📊 Diagnostic Screening]  [🔍 OCR Transcription]  [📋 BHK]  │
│                                                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐ │
│  │ Upload Handwriting      │  │ Transcription Output         │ │
│  │ Sample                  │  │                              │ │
│  │ [IMAGE]                 │  │ "The quick brown fox jumps   │ │
│  │                         │  │  over the lazy dog"          │ │
│  │ [🔍 Transcribe]        │  │                              │ │
│  │                         │  │ Word Confidence Map:         │ │
│  │                         │  │ [CONFIDENCE HEATMAP OVERLAY] │ │
│  │                         │  │                              │ │
│  │                         │  │ Low-Confidence Words:        │ │
│  │                         │  │ • "quick" (0.41) → quich,   │ │
│  │                         │  │   quick, quiak               │ │
│  │                         │  │ • "jumps" (0.33) → junps,   │ │
│  │                         │  │   jumps, jumbs               │ │
│  └─────────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Visual Overlay Features

```python
def render_confidence_overlay(
    original_image: np.ndarray,
    transcription: TranscriptionResult
) -> np.ndarray:
    """
    Draws on the original image:
    1. Green boxes around HIGH confidence words
    2. Amber boxes around MEDIUM confidence words
    3. Red boxes around LOW confidence words
    4. Decoded text label above each word box
    5. Alternative candidates shown below LOW confidence words
    6. Dashed lines connecting words to their decoded text
    """
```

---

## 12. Risk Register & Mitigations

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Insufficient dysgraphic training data | Model can't generalize to real dysgraphic writing | HIGH | Synthetic augmentation pipeline (Phase 5.2); progressive fine-tuning from clean → synthetic → real |
| R2 | Word segmentation fails on severely colliding characters | Entire words get merged or split wrong | HIGH | Fall back to line-level CTC (skip word segmentation, decode full lines); collision-aware segmentation using BHK collision_ratio signal |
| R3 | Language model hallucinations | LM "corrects" an unusual but correct word to a common one | MEDIUM | Cap language_weight at 0.6; never let LM override HIGH-confidence visual predictions; add domain-specific vocabulary |
| R4 | Computational cost too high for local inference | Users can't run on modest hardware | MEDIUM | Use MobileNetV3 backbone; KenLM instead of neural LM for Stage 1; quantize models to INT8; batch efficiently |
| R5 | CTC blank token dominance on very distorted text | Model outputs empty strings for hardest words | HIGH | Lower blank penalty in beam search; train with curriculum learning (clean → progressively harder) |
| R6 | Language model favors English; won't work on Malay/Hindi | Multilingual gap | MEDIUM | Use multilingual LM (mBERT) or train language-specific KenLM models; keep visual model language-agnostic |
| R7 | Over-reliance on stroke primitives for non-Latin scripts | Hindi / Devanagari has different topological structure | MEDIUM | Make stroke primitive vocabulary extensible; start with Latin-only, add Devanagari primitives in Phase 2+ |

---

## 13. File Structure & Module Map

```
Dysgraphia-Detection/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                    # EXISTING — reuse
│   ├── bhk_features.py                     # EXISTING — reuse segment_text_lines(), fit_line_baselines()
│   │
│   └── ocr/                                # ──── NEW MODULE TREE ────
│       ├── __init__.py                      # Package init, version
│       ├── pipeline.py                      # End-to-end orchestrator (image → TranscriptionResult)
│       ├── segmentation.py                  # Line/word/character segmentation
│       │   ├── segment_lines()              #   Reuses bhk_features.segment_text_lines
│       │   ├── segment_words()              #   Projection + CC hybrid
│       │   └── normalize_word_image()       #   Height normalization, deskew
│       ├── stroke_features.py               # Stroke primitive extraction
│       │   ├── skeletonize_and_trace()      #   Medial axis → stroke graph
│       │   ├── classify_strokes()           #   Stroke → primitive labels
│       │   └── stroke_to_char_prior()       #   Primitive combo → char distribution
│       ├── char_hypothesis.py               # Character-level hypothesis generation
│       │   ├── CRNNModel (nn.Module)        #   CNN + BiLSTM + CTC
│       │   ├── beam_search_ctc()            #   Standard CTC beam search
│       │   └── beam_search_with_prior()     #   Stroke-augmented beam search
│       ├── word_recognizer.py               # Word-level recognition coordinator
│       │   ├── recognize_word()             #   Image → List[WordHypothesis]
│       │   └── recognize_line()             #   Line image → List[WordHypothesis]
│       ├── language_reranker.py             # Language model contextual re-ranking
│       │   ├── KenLMScorer                  #   Fast n-gram scoring
│       │   ├── NeuralLMScorer               #   GPT-2 / DistilBERT scoring
│       │   └── rerank_sentence()            #   Lattice beam search + LM re-scoring
│       ├── fusion.py                        # Score fusion and confidence
│       │   ├── adaptive_weights()           #   Dynamic visual/language weighting
│       │   ├── compute_word_confidence()    #   Multi-signal confidence score
│       │   └── calibrate_confidence()       #   Post-hoc calibration (Platt scaling)
│       ├── augmentation.py                  # Synthetic dysgraphic augmentation
│       │   └── DysgraphiaAugmenter          #   All degradation transforms
│       ├── training/                        # Training scripts
│       │   ├── train_crnn.py                #   CRNN training loop
│       │   ├── train_lm.py                  #   Language model fine-tuning
│       │   └── evaluate.py                  #   CER/WER/calibration evaluation
│       └── utils.py                         # Data classes, visualization helpers
│           ├── CharHypothesis               #   Character-level result
│           ├── WordHypothesis               #   Word-level result
│           ├── LineResult                   #   Line-level result
│           ├── TranscriptionResult          #   Full document result
│           └── render_confidence_overlay()  #   Visual overlay rendering
│
├── models/                                  # NEW — Model weights directory
│   ├── crnn_iam_base.pth                    #   CRNN pre-trained on IAM
│   ├── crnn_dysgraphia_ft.pth              #   CRNN fine-tuned on dysgraphic data
│   ├── kenlm_en_5gram.binary               #   English 5-gram language model
│   └── gpt2_scorer/                         #   GPT-2 Small weights for re-ranking
│
├── app.py                                   # MODIFIED — Add OCR Transcription tab
├── requirements.txt                         # MODIFIED — Add torch, transformers, etc.
└── ocr_plan.md                              # THIS FILE
```

---

## 14. Timeline

| Phase | Duration | Deliverable | Dependencies |
|---|---|---|---|
| **Phase 0: Foundation** | 1 week | Module scaffolding, data classes, dependency setup | None |
| **Phase 1: Stroke Features** | 2 weeks | `stroke_features.py` with primitive extraction + char likelihood tables | Phase 0 |
| **Phase 2: Word Recognition** | 3 weeks | `segmentation.py` + `char_hypothesis.py` + CRNN trained on IAM | Phase 0 |
| **Phase 3: Language Re-ranker** | 2 weeks | `language_reranker.py` with KenLM + GPT-2 scoring | Phase 2 |
| **Phase 4: Fusion Pipeline** | 1 week | `fusion.py` + `pipeline.py` end-to-end working | Phase 1, 2, 3 |
| **Phase 5: Training Data** | 2 weeks (parallel) | Augmentation pipeline, manual annotation of scraped images | Phase 0 |
| **Phase 6: Dysgraphia Integration** | 1 week | OCR-derived features → enhanced ensemble classifier | Phase 4 |
| **Phase 7: Evaluation** | 1 week | CER/WER benchmarks, ablation studies, baseline comparisons | Phase 4, 5 |
| **Phase 8: Gradio UI** | 1 week | OCR tab in app.py with confidence overlay visualization | Phase 4 |

**Total estimated timeline: 8-10 weeks** (Phases 1-3 partially parallelizable, Phase 5 runs in background)

### Critical Path

```
Phase 0 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 7
                                        ↑          ↑
Phase 0 ──→ Phase 1 ────────────────────┘          │
Phase 0 ──→ Phase 5 (parallel) ────────────────────┘
```

---

> [!IMPORTANT]
> **Start with Phase 0 + Phase 2 simultaneously.** The CRNN word recognizer is the backbone
> of the entire system. Get it working on IAM first (clean handwriting), then progressively
> fine-tune on harder data. The stroke features and language model are *enhancements* that
> improve accuracy on the hardest cases — the system should produce useful output even without
> them.

> [!TIP]
> **Quick Win:** Before building the full CRNN, test TrOCR (pre-trained Transformer OCR from
> HuggingFace) as a baseline recognizer. If TrOCR already handles some dysgraphic text
> reasonably, we can use it as the visual backbone and focus engineering effort on the
> language model re-ranker and confidence calibration — the parts that are truly novel.

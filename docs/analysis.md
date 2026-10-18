# Dysgraphia Detection - Strategic Analysis & Next Frontiers

## Part 1: Fixes to the Existing System (Workstream A)

### 1. Fix why Stage 15 (Motion Predictor) didn't improve accuracy
In Stage 15, `models/motion_predictor.py` trained a linear Ridge regression on existing paper features to predict stylus pauses, and then fed those predicted values back into a linear classifier alongside the same paper features.
Mathematically, feeding a linear combination of $X$ back into a linear model alongside $X$ adds zero new information—which is why AUROC stayed flat at 0.873. For reference stylus data to actually help, it has to supervise new image-level features (like grayscale shading or stroke width), not re-weight existing 15 features.

### 2. Task-Aware Routing (Fixing the School B Grade 3 Drop)
On School B, external validation is 0.815 AUROC on Grades 4–7, but drops to 0.694 when Grade 3 and 8 are included.
That happens because School B Grade 3 sheets have zero dictation tasks, while the #1 most important feature (`components_per_unit_width_dict_mean`) relies on dictation. Imputing missing dictation features with medians dilutes the signal.
**The Fix**: Train separate sub-models based on which tasks are present on the page (e.g., a Copy + Free-Write model for sheets without dictation, and a Full Protocol model when all tasks exist). That makes the system much more reliable when scanning real school notebooks that don't have all 6 tasks.

## Part 2: New Paper-Only Streams to Build Next
Right now, a kid who writes very slowly to keep their shapes neat (which accounted for half of false negatives in Stage 3) fools Workstream A because their static binary skeletons look normal. To catch them from paper alone, we can add three orthogonal streams from the exact same scan:

### Stream 2: Grayscale Micro-Texture & Stroke-Width Stream (Pure Math, <50MB RAM)
When you binarize the crop into black-and-white (`clean_ink.png`), you erase how hard the kid pressed the pencil and how the stroke thickness fluctuates. By using both `clean_ink.png` and the grayscale `page_normalized.png`:
- **Stroke-Width Jitter (Distance Transform)**: Compute the Euclidean distance transform on `clean_ink.png` and sample it along the skeleton path. Smooth writers maintain steady line thickness; struggling writers have sudden pinching and bloating along a single stroke as their hand speed stalls and jerks.
- **Grayscale Pressure / Shading Variance**: Sample the normalized grayscale pixel intensity along the skeleton centerline. Kids with poor motor control press unevenly—digging the pencil lead dark into the paper on downstrokes and barely leaving a faint trail on curves.
- **Erasure & Smudge Halo Detection**: Struggling writers erase and rewrite constantly. Measuring background grayscale variance immediately surrounding word bounding boxes catches heavy pencil smudging and paper abrasion that Otsu thresholding wipes out.
- **How to use Reference Stylus Data here**: Use the 94 matched School A stylus records offline to check which grayscale/width formulas correlate best with true 200 Hz stylus pressure variance and velocity drops.

### Stream 3: The Text / Spelling Stream (Zero Extra Inference RAM)
In Stage 14 (`scratch/test_ocr_content_stream.py`), you already proved that OCR confidence and recognized word counts drop by $2.7\times$ on struggling writers, including the slow compensatory writers who fooled Workstream A. You paused it in Stage 15 only to avoid running heavy deep-learning OCR during feature extraction.
- **The Engineering Workaround**: You already run EasyOCR (hi+en) inside `pipeline/segment.py` during Phase 1 preprocessing to classify Hindi vs. English lines!
- Instead of throwing the EasyOCR output away after checking the script, save the recognized text tokens and confidence scores directly into `{task_id}.json` during Phase 1.
- Then `features/extractor.py` stays 100% lightweight (<50MB RAM) and can compute:
  - OCR Legibility Confidence (mean and minimum character confidence per sentence).
  - Character / Word Error Rate (CER / WER) against the known target sentences for Tasks 1–4.
  - Dictation Spelling Breakdown ($\Delta\text{CER}_{\text{dict} - \text{copy}}$): Kids who can copy letter shapes visually in Task 1/2, but fall apart when trying to spell and form letters from audio in Task 3/4.

### Stream 4: Deep Visual Patch Stream (baseline/ Late Fusion)
Skeleton math reduces a whole sentence down to summary numbers, which means it cannot see backwards letters, crushed character loops, or messy overwriting.
- The `baseline/` directory is already set up for DenseNet201 + feature fusion.
- Instead of concatenating 1,000 deep features into your tabular CSV (which will instantly overfit 215 students), train a small vision model (or frozen pretrained backbone + linear head) on cropped word/sentence patches to output a single out-of-fold visual risk score $P_{\text{vision}}$ per student.

### Multi-Stream Fusion (Late Stacking)
$$P_{\text{student}} = w_1 P_{\text{Geometry (Stream 1)}} + w_2 P_{\text{Grayscale/Width (Stream 2)}} + w_3 P_{\text{OCR Content (Stream 3)}} + w_4 P_{\text{Vision (Stream 4)}}$$

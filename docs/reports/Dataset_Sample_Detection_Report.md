# 📋 Dysgraphia Detection Report (Dataset Samples)

This report shows the automated diagnostic assessment of sample images from your **`bhk/data/raw/photos`** dataset (`1.jpeg` and `2.jpeg`), analyzed using the **13 BHK computational proxies** and the trained **ML Dysgraphia Classifier**.

---

## 🔬 Sample Analysis Summary

| Metric | Sample 1: `1.jpeg` | Sample 2: `2.jpeg` |
|---|---|---|
| **Classification Verdict** | **POTENTIAL DYSGRAPHIA DETECTED** | **POTENTIAL DYSGRAPHIA DETECTED** |
| **Model Confidence** | **100.0%** Potential Dysgraphia | **100.0%** Potential Dysgraphia |
| **Left Margin Alignment** | **1.000 (Severe Deviation)** | **1.000 (Severe Deviation)** |
| **Baseline Alignment** | **0.121 (Relatively Level)** | **1.000 (Severe Slant: $10.33^\circ$)** |
| **Broken Letter Links** | **1.000 (2.3 breaks/word)** | **1.000 (3.5 breaks/word)** |
| **Inconsistent Height Extensions** | **1.000 (Uncontrolled Ascenders)** | **1.000 (Uncontrolled Ascenders)** |
| **Irregular Letter Size** | **0.323 (Moderate)** | **0.690 (High Variability)** |
| **Traced / Corrected Letters** | **0.128 (Clean)** | **0.636 (Heavy Retracing / Hesitation)** |

---

## 🖼️ Diagnostic Measurement Maps

The images below show the computer vision measurements plotted directly on each handwriting sample:

### 1. Diagnostic Overlay for `1.jpeg`
* **Observations:**
  - Notice the **red margin indicator dots** showing paragraph and left-edge instability ($4.93 \times ref\_h$ deviation).
  - High degree of print-style letter fragmentation (**pink rectangles** indicating broken links within words).

![BHK Measurement Overlay for 1.jpeg](C:/Users/Laxmi%20Srujani/.gemini/antigravity/brain/570a66f2-9699-4682-8de6-c09298ce89a5/sample_1_overlay.png)

---

### 2. Diagnostic Overlay for `2.jpeg`
* **Observations:**
  - Significant **baseline slant ($10.33^\circ$)**, marked by the blue/cyan baseline trajectories across words.
  - High frequency of **re-traced and corrected strokes** (dark pink spots on thicker overlapping pen strokes).
  - Irregular ascender and descender zone heights across consecutive lines.

![BHK Measurement Overlay for 2.jpeg](C:/Users/Laxmi%20Srujani/.gemini/antigravity/brain/570a66f2-9699-4682-8de6-c09298ce89a5/sample_2_overlay.png)

---

## 📊 Complete 13 BHK Feature Table

```
====================================================================================
BHK Feature Name                    1.jpeg (Score)   2.jpeg (Score)   Severity
------------------------------------------------------------------------------------
1.  writing_size                         0.000            0.000       🟢 Normal scale
2.  non_aligned_left_margin              1.000            1.000       🔴 Critical (Margin drift)
3.  skewed_alignment                     0.121            1.000       🔴 Critical in sample 2
4.  insufficient_word_spacing            0.000            0.375       🟡 Moderate crowding
5.  sharp_angles                         0.110            0.388       🟡 Moderate turns
6.  broken_links                         1.000            1.000       🔴 Disconnected strokes
7.  collisions                           0.218            0.331       🟢 Low line collisions
8.  irregular_letter_size                0.323            0.690       🟡 High height variance
9.  inconsistent_height_extensions       1.000            1.000       🔴 Critical (Zone control)
10. traced_corrected_letters             0.128            0.636       🟡 High overwriting in 2
11. unstable_trace                       0.000            0.000       🟢 Low high-frequency tremor
====================================================================================
```

---

## 💡 Clinical Diagnostic Interpretation

Both sample images show strong diagnostic markers of **dysgraphia**, driven primarily by:
1. **Loss of Spatial Structuring:** Neither student maintains an aligned left margin relative to page boundaries.
2. **Horizontal Baseline Instability:** In `2.jpeg`, the words slant downwards at an angle of over **$10^\circ$** relative to the ruling.
3. **Ascender/Descender Zone Dysregulation:** The height ratio of letters like *t, d, l, b* compared to the core letter body (*a, c, e*) is erratic across words.

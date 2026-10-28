# 🩺 Dysgraphia Assessment: Sample 2 (Bhavika)

**Student Metadata Detected:**
- **Text:** *"my Name is Bhavika"* (English & Devanagari Hindi dictation)
- **Sample Type:** Ruled notebook with printed vertical double margin

---

## 🔍 Classification Verdict

```
###########################################################################
 CLASSIFICATION RESULT: POTENTIAL DYSGRAPHIA DETECTED
 Model Confidence: 100.0% Potential Dysgraphia | 0.0% Normal
###########################################################################
```

---

## 🔬 BHK Feature Scores (0.0 to 1.0)

| # | BHK Feature Name | Score (0–1) | Status | Clinical Findings |
|---|---|---|---|---|
| 1 | **Traced / Corrected Letters** | **0.953** | 🔴 **CRITICAL DEFICIT** | Heavy pen overwriting, retraced strokes, erased & written-over words (e.g. line 3 ghost text). |
| 2 | **Non-Aligned Left Margin** | **1.000** | 🔴 **CRITICAL DEFICIT** | Wandering margin ($1.49 \times ref\_h$ deviation); ignores the printed vertical margin line. |
| 3 | **Broken Letter Links** | **1.000** | 🔴 **CRITICAL DEFICIT** | $2.37$ breaks per word; disconnected segmented letter components. |
| 4 | **Inconsistent Height Extensions**| **1.000** | 🔴 **CRITICAL DEFICIT** | Inconsistent ascender/descender heights (e.g., in *"follBoll"*, *"friends"*). |
| 5 | **Skewed Alignment** | **0.653** | 🟡 **MODERATE** | Baseline angle deviation ($1.33^\circ$); words sink below the ruling. |
| 6 | **Irregular Letter Size** | **0.435** | 🟡 **MODERATE** | Uncontrolled variation in letter heights within the same line. |
| 7 | **Sharp Angles** | **0.431** | 🟡 **MODERATE** | Abrupt direction changes in curved letter forms. |
| 8 | **Insufficient Word Spacing** | **0.157** | 🟢 **NORMAL** | Spacing between words is mostly adequate. |
| 9 | **Collisions** | **0.169** | 🟢 **NORMAL** | Very few overlapping strokes between adjacent letters. |
| 10 | **Writing Size** | **0.000** | 🟢 **NORMAL** | Appropriate overall letter size relative to the notebook ruling pitch. |
| 11 | **Unstable Trace (Tremor)** | **0.000** | 🟢 **STEADY** | No high-frequency tremor detected ($width\_cv = 0.177$). |

---

## 🖼️ Visual Diagnostic Measurement Overlay

The visual overlay shows:
- **Dark pink circles** highlighting heavy overwriting and ink accumulation where the student hesitated or retraced letters.
- **Red indicator dots** at line beginnings showing margin inconsistency.
- **Line numbering confusion:** Student wrote numbered items (*1., 2., 3., 4.*) overlapping the margin and alternating between Hindi and English sentences.

![BHK Measurement Overlay for Bhavika Sample](C:/Users/Laxmi%20Srujani/.gemini/antigravity/brain/570a66f2-9699-4682-8de6-c09298ce89a5/bhavika_image_overlay.png)

---

## 💡 Clinical Diagnostic Comparison

Comparing this sample (*Bhavika*) with the previous sample (*Yash*):

1. **Prominent Retracing & Correction (Bhavika's signature marker):**
   - In *Bhavika's* sample, **`traced_corrected_letters` jumps to 0.953** (compared to 0.274 for Yash). 
   - Notice line 3 where an entire phrase was faintly erased or overwritten (*"After the game we eat fruits together"*), as well as heavy retracing on *"follBoll"* and Hindi conjunct characters.
2. **Orthographic / Spelling Difficulties (Dyslexic-Dysgraphic overlap):**
   - Phonetic spelling errors and letter substitutions: *"cricket"* written as *"ourket"*, *"friends"* written as *"fruends"*, *"football"* written as *"follBoll"*, *"fruits"* written as *"frouits"*.
3. **Spatial & Margin Deficits:**
   - Both students exhibit **`non_aligned_left_margin = 1.000`**, failing to anchor sentences against the margin line.

# Dysgraphia Handwriting Analysis — Canonical Feature Schema (v3.0.0-kinematics)

> **Architectural Role**: Pure mathematical and biophysical feature extraction engine (13-D). This module **does not classify, score diagnostic risk, or output clinical verdicts**. Static handwriting-scale features (BHK) and OCR are developed separately by teammates; this repository is strictly kinematics-only. Extracted feature vectors are exported for downstream multimodal ensembling alongside OCR-based features and supervised classifiers trained on validated clinical ground truth.

> **CRITICAL LIMITATION**:
> A static 2D image contains **no timing or physical velocity information**.
> All kinematic outputs (`kin_*`) are **ESTIMATES** derived from recovered 2D skeleton paths using the Two-Thirds Power Law ($v \propto \kappa^{-1/3}$) and the Plamondon Sigma-Lognormal model. They are **NOT measurements** of actual pen speed, timing, or physical acceleration. Every kinematic output carries an explicit confidence score in $[0, 1]$ reflecting the geometric quality of the recovered skeleton path. Absolute speed numbers must never be described as measured physical velocities.

---

## 1. Canonical Feature Vector Overview (13-Dimensional)

The extracted feature vector is an ordered numerical vector ($\mathbb{R}^{13}$). Every feature is scale-normalized with reference to the document's median character height ($H_{\text{med}}$) or formulated as a scale-free dimensionless ratio.

| Index | Feature Key | Category | Provenance / Standard Citation | Units | Stability & Reliability Label | Description |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `kin_mean_velocity` | Neuromotor Kinematics (ESTIMATED) | Two-Thirds Power Law ($v \propto \kappa^{-1/3}$) | $H_{\text{med}} / \text{s}$ (ESTIMATED) | `unstable_low_resolution` | Reconstructed mean movement speed proxy along trajectory. Degrades below 300px image height. |
| **1** | `kin_peak_velocity` | Neuromotor Kinematics (ESTIMATED) | Plamondon Sigma-Lognormal Impulse Theory | $H_{\text{med}} / \text{s}$ (ESTIMATED) | `unstable_low_resolution` | Maximum ballistic impulse velocity proxy attained during straight trajectory segments. |
| **2** | `kin_velocity_skewness` | Neuromotor Kinematics (ESTIMATED) | Plamondon Asymmetric Neuromuscular Impulse | Dimensionless (ESTIMATED) | `unstable_fragmentation` | Asymmetry between acceleration burst and deceleration glide. Sensitive to stroke fragmentation. |
| **3** | `kin_nvi_rate` | Neuromotor Kinematics (ESTIMATED) | Dysgraphia Kinematics (Drotár et al., 2016) | Inversions / s (ESTIMATED) | `unstable_fragmentation` | Temporal frequency of velocity reversals per nominal second. **WARNING (ISSUE 3): This feature is mathematically inversely proportional to mean stroke length** — larger handwriting → fewer inversions/s, independent of motor fluency. Synthetic boundary envelopes contribute ~1 inversion per fragment, so NVI rate ≈ K / mean_stroke_length. Do not use as a dysgraphia discriminator without correcting for stroke-length scaling. |
| **4** | `kin_nvi_per_stroke` | Neuromotor Kinematics (ESTIMATED) | Primary Scale-Invariant Fluency Marker | Inversions / stroke (ESTIMATED) | `stable` | Average number of speed reversals per recovered stroke; primary robust scale-invariant fluency marker. |
| **5** | `kin_nvi_per_h_med` | Neuromotor Kinematics (ESTIMATED) | Spatial Trajectory Smoothness Density | $1 / H_{\text{med}}$ (ESTIMATED) | `stable` | Spatial density of velocity reversals per unit median character height. |
| **6** | `kin_jerk_metric` | Neuromotor Kinematics (ESTIMATED) | Teulings et al. Movement Smoothness Theory | $H_{\text{med}}^2 / \text{s}^5$ (ESTIMATED) | `unstable_noise` | Third time derivative of position ($\int \dddot{s}^2 dt$). Highly sensitive to skeleton boundary noise. |
| **7** | `kin_dimensionless_jerk` | Neuromotor Kinematics (ESTIMATED) | Flash & Hogan (1985) Dimensionless Jerk | Dimensionless (ESTIMATED) | `stable` | Coordinate-free dimensionless neuromotor smoothness: $(T^5 / L^2) \int j^2 dt$. Invariant to scale and duration. |
| **8** | `kin_spatial_roughness_4_8hz` | Neuromotor Kinematics (ESTIMATED) | Welch PSD on estimated velocity profile; concept from Deuschl et al. 1998 | Ratio [0, 1] (ESTIMATED) | `unstable_low_stroke_count` | Relative power in 4–8 Hz band of the **estimated** velocity profile. Correlates with physiological micro-tremor but is **NOT** a measurement of actual tremor (no accelerometer). Requires ≥10 strokes for PSD stability. |
| **9** | `kin_pen_lift_count` | Neuromotor Kinematics (ESTIMATED) | Motor Program Continuity & Pen Lifts | Count (strokes) | `stable` | Count of continuous recovered on-surface trajectory paths; reflects motor fragmentation. |
| **10** | `kin_mean_stroke_length` | Neuromotor Kinematics (ESTIMATED) | Continuous Trajectory Efficiency | px (ESTIMATED) | `stable` | Average arc length of continuous recovered paths before a pen lift or directional break. |
| **11** | `kin_ink_width_ratio_mean` | Biophysical Pressure (ESTIMATED) | Optical Stroke Width (2·EDT) / H_med | W/H_med (ESTIMATED) | `unstable_under_pen_type` | Mean relative stroke width along recovered trajectory. **IMPORTANT: correlation with real tablet hardware pressure r ≈ 0.002 (effectively zero).** This is an optical ink-geometry feature, not a pressure proxy. Sensitive to pen nib type, paper texture, and DPI. |
| **12** | `kin_ink_width_ratio_std` | Biophysical Pressure (ESTIMATED) | Optical Down-force Variability | W/H_med (ESTIMATED) | `unstable_under_pen_type` | Std-dev of relative stroke width. Same r ≈ 0.002 limitation as above. Measures ink-width variation along strokes; has no demonstrable relationship with physical pen pressure. |

---

## 2. Per-Sample Export Schema Contract (JSON)

Every sample extraction outputs a standardized, JSON-serializable dictionary with the following schema:

```json
{
  "sample_id": "sample_001",
  "pipeline_version": "3.0.0",
  "feature_schema_version": "3.0.0",
  "feature_vector": [
    0.6988, 3.9805, 1.7749, 2.7999, 13.3810, 3.9995,
    13065.9008, 754.7428, 0.0388, 21.0, 3.3456, 0.1271, 0.0075
  ],
  "feature_names": [
    "kin_mean_velocity",
    "kin_peak_velocity",
    "kin_velocity_skewness",
    "kin_nvi_rate",
    "kin_nvi_per_stroke",
    "kin_nvi_per_h_med",
    "kin_jerk_metric",
    "kin_dimensionless_jerk",
    "kin_spatial_roughness_4_8hz",
    "kin_pen_lift_count",
    "kin_mean_stroke_length",
    "kin_ink_width_ratio_mean",
    "kin_ink_width_ratio_std"
  ],
  "feature_dict": {
    "kin_mean_velocity": 0.6988,
    "kin_peak_velocity": 3.9805,
    "kin_velocity_skewness": 1.7749,
    "kin_nvi_rate": 2.7999,
    "kin_nvi_per_stroke": 13.3810,
    "kin_nvi_per_h_med": 3.9995,
    "kin_jerk_metric": 13065.9008,
    "kin_dimensionless_jerk": 754.7428,
    "kin_spatial_roughness_4_8hz": 0.0388,
    "kin_pen_lift_count": 21.0,
    "kin_mean_stroke_length": 3.3456,
    "kin_ink_width_ratio_mean": 0.1271,
    "kin_ink_width_ratio_std": 0.0075
  },
  "per_feature_confidence": {
    "kin_mean_velocity": 0.3,
    "kin_peak_velocity": 0.3,
    "kin_velocity_skewness": 0.3,
    "kin_nvi_rate": 0.3,
    "kin_nvi_per_stroke": 0.3,
    "kin_nvi_per_h_med": 0.3,
    "kin_jerk_metric": 0.3,
    "kin_dimensionless_jerk": 0.3,
    "kin_spatial_roughness_4_8hz": 0.5,
    "kin_pen_lift_count": 0.3,
    "kin_mean_stroke_length": 0.5,
    "kin_ink_width_ratio_mean": 0.35,
    "kin_ink_width_ratio_std": 0.35
  },
  "quality_flags": {
    "low_ink": false,
    "ruled_residual": true,
    "low_resolution": true,
    "blur": false,
    "blur_score": 24209.89,
    "skew_corrected": false,
    "skew_angle_deg": 0.0,
    "multi_line_aggregated": false,
    "too_few_strokes": false,
    "no_text_lines_found": false,
    "abnormal_h_med": false,
    "line_removal_applied": true,
    "unreliable_extraction": false
  },
  "metadata": {
    "sample_id": "sample_001",
    "image_size_px": [506, 55],
    "h_med_px": 16.0,
    "ink_pixels": 2265,
    "skeleton_pixels": 971,
    "text_line_count": 1,
    "character_component_count": 24,
    "lines_detected": 2,
    "recovered_stroke_count": 21,
    "processing_steps": [
      "robust_load_binarize",
      "line_removal(2_lines)",
      "skeletonize",
      "segment",
      "stroke_recovery(21_strokes)",
      "kinematics_estimated"
    ],
    "elapsed_seconds": 0.168,
    "pipeline_version": "3.0.0",
    "feature_schema_version": "3.0.0",
    "kinematic_note": "All kin_* outputs are ESTIMATES from the Two-Thirds Power Law + Plamondon Sigma-Lognormal model. NOT measurements of actual pen timing."
  },
  "null_means_nan": true
}
```

---

## 4. Known Data-Quality Issues & Fixes

### Issue A — GT Timestamp Corruption (Step 1 Fix)

Wacom tablet firmware fires polling interrupts at ~250 kHz between genuine ~200 Hz
motion samples. This produces consecutive rows in SVC files with `dt < 1 ms` but real
coordinate displacement, causing velocity spikes of up to **2,887,128 px/s (~11 m/s)**
— physically impossible for handwriting.

**Fix applied in `src/loaders.py`**: `load_svc()` (and `_dedup_timestamps()`) drops any
consecutive pen-down row pair where `dt < MIN_DT_S = 1e-3 s`. Air rows are always kept.
The count of dropped rows is stored in `sample.metadata['dt_artifact_rows_dropped']`.

**Run the audit tool to verify**:
```bash
python scripts/gt_trajectory_audit.py
# Produces outputs/gt_audit/audit_report.csv and audit_summary.txt
```

### Issue B — Feature Renaming (Step 4)

| Old name (≤ v2.0.0) | New name (v2.1.0) | Reason |
|---|---|---|
| `kin_tremor_index_4_8hz` | `kin_spatial_roughness_4_8hz` | The pipeline estimates a PSD ratio over a reconstructed velocity profile, not an accelerometer tremor signal. The old name overstated clinical specificity. |
| `kin_pressure_proxy_mean` | `kin_ink_width_ratio_mean` | Rename to clarify it is an optical ink-width ratio, not a direct pressure sensor reading. |
| `kin_pressure_proxy_std` | `kin_ink_width_ratio_std` | Same rationale as above. |

---

### Issue C — NVI Rate Direction Inversion (ISSUE 3)

The reconstructed `kin_nvi_rate` shows the **wrong direction** relative to clinical expectation
in empirical group comparisons:
- Literature predicts: dysgraphic writers → **elevated** NVI (≥ 10 inv/s vs. 4–8 inv/s typical)
- Phase 3 observed: dysgraphic (PD) group had **lower** NVI (8.19 inv/s) than controls (9.93 inv/s)

**Root cause**: Every recovered stroke receives a synthetic boundary envelope
($\sin^{1/2}(\pi s / L)$) that forces velocity to 0 at stroke endpoints. Each endpoint
contributes ~1 velocity inversion. Therefore:
$$\text{nvi\_rate} \approx \frac{N_\text{strokes} \times K}{N_\text{strokes} \times \bar{L} \times \Delta t} = \frac{K}{\bar{L} \times \Delta t}$$
The metric is an **inverted proxy for mean stroke length**, not neuromotor fluency.
In the Malay cohort, dysgraphic (PD) samples had 38.8% longer pixel-strokes (larger
handwriting + closer camera crop), mechanically reducing their NVI rate.

> **Decision**: Do not use `kin_nvi_rate` as a dysgraphia discriminator.
> Use `kin_nvi_per_stroke` (scale-invariant, labelled `stable`) as the primary NVI metric.
> If `kin_nvi_rate` is retained in the feature vector, it must be accompanied by stroke-length
> normalization or explicitly labeled as an inverted stroke-length proxy.

---

## 5. Physical-Unit Calibration Path (Step 3)

All `kin_*` velocity features are expressed in **H_med / estimated_s** (a dimensionless
relative unit). To convert to physical **mm/s** for GT benchmarking:

```python
from src.render import render_trajectory_to_image_with_meta, WACOM_MM_PER_COORD

img, scale_meta = render_trajectory_to_image_with_meta(sample)
mm_per_px = scale_meta["mm_per_px"]   # e.g. ~0.000437 mm/px

# Convert a velocity in H_med/s to mm/s:
# v_mm_s = v_h_med_s * (h_med_px * mm_per_px)
```

**Wacom Intuos4 calibration**: 5080 lpi → 200 lines/mm → `1 raw coord unit = 0.005 mm`.
This is fixed hardware spec and does not require per-session calibration.

Note: because the pipeline velocity is expressed in *estimated* H_med/s units (where the
temporal axis is reconstructed, not measured), the mm/s conversion gives an *indicative*
magnitude for comparison, not a verified measurement. All comparisons must use the cleaned
GT files (Issue A above) as the reference.

---

## 6. Extraction Quality Flags Specification

| Quality Flag | Trigger Condition | Operational Meaning & Guidance |
| :--- | :--- | :--- |
| `low_ink` | Ink pixels $< 500$ or ink coverage $< 0.1\%$ | Image may be overexposed, blank, or extremely faint. |
| `ruled_residual` | Ruled paper or grid paper detected | Ruled lines were detected and removed; minor junction artifacts may exist. |
| `low_resolution` | $\min(W, H) < 400\text{ px}$ or $W \cdot H < 300,000\text{ px}$ | Insufficient pixel resolution for stable distance transform stroke-width calculation. |
| `blur` | Discrete 2D Laplacian variance $< 75.0$ | Soft camera focus or severe motion blur detected. |
| `skew_corrected` | Baseline skew angle $> 1.0^\circ$ detected and rotated | Image was automatically deskewed before feature calculation. |
| `multi_line_aggregated` | Number of segmented text lines $\ge 2$ | Line collisions and margin drift are computed across multiple valid lines. |
| `too_few_strokes` | Total recovered strokes $< 5$ | Trajectory graph has sparse coverage; kinematic statistics have reduced sample support. |
| `no_text_lines_found` | Segmentation found zero valid text lines | Spatial BHK metrics default to fallback bounding boxes. |
| `abnormal_h_med` | $H_{\text{med}} < 5.0\text{ px}$ or $H_{\text{med}} > 0.6 \cdot H_{\text{img}}$ | Character height estimation tripped bounds; scale normalization may be distorted. |
| `line_removal_applied` | $\ge 1$ horizontal/vertical ruled line removed | Trajectory-space line removal modified the raw ink mask. |
| `unreliable_extraction` | `low_ink` OR zero strokes OR `abnormal_h_med` | Feature vector contains NaN or invalid entries; must not be used downstream without review. |

---

## 7. Permanent Limitations — Features That Can Never Be Compared to Hardware GT

The following signals have **no optical equivalent** and cannot be inferred from any
image of handwriting, regardless of resolution or model complexity. These are
permanent architectural constraints, not implementation gaps.

### 7a. Pen Tilt (x_tilt, y_tilt)

SVC files record `x_tilt` and `y_tilt` (Wacom Intuos4 tilt in degrees, ±60° range).
No image-only method can recover pen tilt: it would require stereo vision, structured
light, or a dedicated tilt sensor. This limitation is **permanent and irrecoverable**.

> **Decision**: Do not include pen tilt in the canonical feature schema, the JSON export,
> or any GT-vs-predicted comparison table. State "not available from image" in reports.

### 7b. True Hardware Stylus Pressure

The Wacom digitizer reports true down-force via a Hall-effect sensor (0–1023 pressure
units). The optical ink-width features (`kin_ink_width_ratio_*`) were originally called
"pressure proxy" features. They are not:

| Metric | Value |
|--------|-------|
| Pearson r (optical width vs hardware pressure) | **≈0.002** |
| Interpretation | No demonstrable linear relationship |
| Root cause | Ink deposition is determined by paper fiber absorption and pen contact area, not linearly by force in the tactile digitizer range |

> **Decision**: Rename these features permanently to `kin_ink_width_ratio_*` (done in
> v2.1.0). Never use them as a pressure substitute in clinical reasoning or classifiers.
> Do not include hardware pressure in the GT-vs-predicted comparison. Report as
> "optically inaccessible" alongside pen tilt.

### 7c. BHK Spatial Features — Transferred Out of Scope

In earlier schema versions (v1.x–v2.x), 9 BHK-derived spatial features (`bhk_*`) were extracted. In v3.0.0, all static handwriting-scale (BHK) features and OCR modules were transferred to a dedicated teammate and repository. This repository is strictly **13-D kinematics-only**.

Historical reference: BHK features have no direct point-level trajectory GT in SVC files (as BHK scoring relies on visual character appearance assessed by raters, evaluated against published cohort norms rather than sensor trajectories). For users requiring BHK static scoring, refer to the teammate's dedicated BHK/OCR module or check git tag `pre-bhk-removal`.

---

## 8. Velocity Data Quality — Source-of-Corruption Record

### What was incorrect (now resolved)

An earlier comparison referenced `kinematics_1790656668.csv`, a pre-computed kinematics
file exported by the Gradio web app. That file recorded:
- Pixel-space coordinates (not raw Wacom tablet units)
- JavaScript `Date.now()` timestamps from browser pointer events
- Pre-computed `velocity = displacement / dt` column, with `dt` as low as 2.9 µs from
  USB HID burst packets, producing velocity values up to **410,799 px/s**

This CSV is NOT part of the validation dataset and was never a ground-truth file.

### Confirmed clean: raw SVC files

| Metric | Value |
|--------|-------|
| Dataset | Datasets/dataSciRep_public (121 files) |
| Timestamp format | Absolute wall-clock ms from Wacom driver |
| Sampling rate | 125 Hz (dt = 7–8 ms per sample) |
| Files with dt < 1ms in pen-down | **0 / 121** |
| Measured velocity range | 22–32 mm/s mean, 50–161 mm/s peak |
| Physical plausibility | Consistent with published handwriting kinematics |

All GT comparisons in this project use the raw SVC files exclusively.
The `kinematics_1790656668.csv` file is excluded from all validation pipelines.


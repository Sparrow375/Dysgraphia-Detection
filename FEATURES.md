# Dysgraphia Handwriting Analysis — Canonical Feature Schema (v2.1.0)

> **Architectural Role**: Pure mathematical and biophysical feature extraction module. This module **does not classify, score diagnostic risk, or output clinical verdicts**. Extracted feature vectors are exported for downstream multimodal ensembling alongside OCR-based features and supervised classifiers trained on validated clinical ground truth.

> **CRITICAL LIMITATION**:
> A static 2D image contains **no timing or physical velocity information**.
> All kinematic outputs (`kin_*`) are **ESTIMATES** derived from recovered 2D skeleton paths using the Two-Thirds Power Law ($v \propto \kappa^{-1/3}$) and the Plamondon Sigma-Lognormal model. They are **NOT measurements** of actual pen speed, timing, or physical acceleration. Every kinematic output carries an explicit confidence score in $[0, 1]$ reflecting the geometric quality of the recovered skeleton path. Absolute speed numbers must never be described as measured physical velocities.

---

## 1. Canonical Feature Vector Overview (20-Dimensional)

The extracted feature vector is an ordered numerical vector ($\mathbb{R}^{20}$). Every feature is scale-normalized with reference to the document's median character height ($H_{\text{med}}$) or formulated as a scale-free dimensionless ratio.

| Index | Feature Key | Category | Provenance / Standard Citation | Units | Stability & Reliability Label | Description |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `bhk_size_covariance` | BHK Static | BHK Item #1 — Letter Size Uniformity | Dimensionless (CV) | `stable` | Quantifies inconsistency in individual character bounding box area across text lines ($0.6 \cdot \text{CV}(H) + 0.4 \cdot \text{CV}(\text{Area})$). |
| **1** | `bhk_height_ratio_consistency` | BHK Static | BHK Item #4 — Relative Character Heights | Dimensionless (IQR/median) | `stable` | Evaluates relative character height dispersion between ascenders/descenders and body x-height. |
| **2** | `bhk_baseline_drift` | BHK Static | BHK Item #3 — Baseline Stability | $1 / H_{\text{med}}$ | `stable` | Quantifies baseline wandering, micro wobble, and line tilt along character contact points. |
| **3** | `bhk_spacing_entropy` | BHK Static | BHK Item #7 — Spatial Organization | Dimensionless (nats) | `stable` | Shannon entropy over normalized horizontal character gap distribution; measures spacing disorder. |
| **4** | `bhk_stroke_width_variance` | BHK Static | BHK Item #9 — Motor Down-Force Steadiness | Dimensionless (CV) | `unstable_under_pen_type` | Variability of stroke thickness via Euclidean distance transform. **Caution**: Sensitive to pen nib type (marker vs. ballpoint) and DPI. |
| **5** | `bhk_telescoping_overlap` | BHK Static | BHK Item #5 — Letter Crowding & Telescoping | Ratio [0, 1] | `stable` | Measures horizontal glyph collisions and bounding-box overlap depth relative to letter width. |
| **6** | `bhk_acute_turns` | BHK Static | BHK Item #8 — Motor Stiffness & Broken Turns | turns / $H_{\text{med}}$ | `stable` | Frequency of sharp directional reversals ($|\Delta\theta| \ge 110^\circ$) replacing smooth continuous curves. |
| **7** | `bhk_left_margin_drift` | BHK Static | BHK Item #2 — Left Margin Alignment | $1 / H_{\text{med}}$ | `stable` | Evaluates left margin alignment drift and indentation variance across successive text lines. |
| **8** | `bhk_line_collisions` | BHK Static | BHK Item #13 — Inter-Line Collisions | Ratio [0, 1] | `stable` | Measures collisions where descenders crash into ascenders of preceding lines, plus inter-line distance CV. |
| **9** | `kin_mean_velocity` | Neuromotor Kinematics (ESTIMATED) | Two-Thirds Power Law ($v \propto \kappa^{-1/3}$) | $H_{\text{med}} / \text{s}$ (ESTIMATED) | `unstable_low_resolution` | Reconstructed mean movement speed proxy along trajectory. Degrades below 300px image height. |
| **10** | `kin_peak_velocity` | Neuromotor Kinematics (ESTIMATED) | Plamondon Sigma-Lognormal Impulse Theory | $H_{\text{med}} / \text{s}$ (ESTIMATED) | `unstable_low_resolution` | Maximum ballistic impulse velocity proxy attained during straight trajectory segments. |
| **11** | `kin_velocity_skewness` | Neuromotor Kinematics (ESTIMATED) | Plamondon Asymmetric Neuromuscular Impulse | Dimensionless (ESTIMATED) | `unstable_fragmentation` | Asymmetry between acceleration burst and deceleration glide. Sensitive to stroke fragmentation. |
| **12** | `kin_nvi_rate` | Neuromotor Kinematics (ESTIMATED) | Dysgraphia Kinematics (Drotár et al., 2016) | Inversions / s (ESTIMATED) | `unstable_fragmentation` | Temporal frequency of velocity reversals per nominal second. **WARNING (ISSUE 3): This feature is mathematically inversely proportional to mean stroke length** — larger handwriting → fewer inversions/s, independent of motor fluency. Synthetic boundary envelopes contribute ~1 inversion per fragment, so NVI rate ≈ K / mean_stroke_length. Do not use as a dysgraphia discriminator without correcting for stroke-length scaling. |
| **13** | `kin_nvi_per_stroke` | Neuromotor Kinematics (ESTIMATED) | Primary Scale-Invariant Fluency Marker | Inversions / stroke (ESTIMATED) | `stable` | Average number of speed reversals per recovered stroke; primary robust scale-invariant fluency marker. |
| **14** | `kin_nvi_per_h_med` | Neuromotor Kinematics (ESTIMATED) | Spatial Trajectory Smoothness Density | $1 / H_{\text{med}}$ (ESTIMATED) | `stable` | Spatial density of velocity reversals per unit median character height. |
| **15** | `kin_jerk_metric` | Neuromotor Kinematics (ESTIMATED) | Teulings et al. Movement Smoothness Theory | $H_{\text{med}}^2 / \text{s}^5$ (ESTIMATED) | `unstable_noise` | Third time derivative of position ($\int \dddot{s}^2 dt$). Highly sensitive to skeleton boundary noise. |
| **16** | `kin_dimensionless_jerk` | Neuromotor Kinematics (ESTIMATED) | Flash & Hogan (1985) Dimensionless Jerk | Dimensionless (ESTIMATED) | `stable` | Coordinate-free dimensionless neuromotor smoothness: $(T^5 / L^2) \int j^2 dt$. Invariant to scale and duration. |
| **17** | `kin_spatial_roughness_4_8hz` | Neuromotor Kinematics (ESTIMATED) | Welch PSD on estimated velocity profile; concept from Deuschl et al. 1998 | Ratio [0, 1] (ESTIMATED) | `unstable_low_stroke_count` | Relative power in 4–8 Hz band of the **estimated** velocity profile. Correlates with physiological micro-tremor but is **NOT** a measurement of actual tremor (no accelerometer). Requires ≥10 strokes for PSD stability. |
| **18** | `kin_pen_lift_count` | Neuromotor Kinematics (ESTIMATED) | Motor Program Continuity & Pen Lifts | Count (strokes) | `stable` | Count of continuous recovered on-surface trajectory paths; reflects motor fragmentation. |
| **19** | `kin_mean_stroke_length` | Neuromotor Kinematics (ESTIMATED) | Continuous Trajectory Efficiency | px (ESTIMATED) | `stable` | Average arc length of continuous recovered paths before a pen lift or directional break. |
| **20** | `kin_ink_width_ratio_mean` | Biophysical Pressure (ESTIMATED) | Optical Stroke Width (2·EDT) / H_med | W/H_med (ESTIMATED) | `unstable_under_pen_type` | Mean relative stroke width along recovered trajectory. **IMPORTANT: correlation with real tablet hardware pressure r ≈ 0.002 (effectively zero).** This is an optical ink-geometry feature, not a pressure proxy. Sensitive to pen nib type, paper texture, and DPI. |
| **21** | `kin_ink_width_ratio_std` | Biophysical Pressure (ESTIMATED) | Optical Down-force Variability | W/H_med (ESTIMATED) | `unstable_under_pen_type` | Std-dev of relative stroke width. Same r ≈ 0.002 limitation as above. Measures ink-width variation along strokes; has no demonstrable relationship with physical pen pressure. |

---

## 2. Per-Sample Export Schema Contract (JSON)

Every sample extraction outputs a standardized, JSON-serializable dictionary with the following schema:

```json
{
  "sample_id": "sample_001",
  "pipeline_version": "2.0.0",
  "feature_schema_version": "2.0.0",
  "feature_vector": [
    0.284, 0.412, 0.089, 1.842, 0.195, 0.045, 0.320, 0.112, 0.000,
    0.705, 1.905, 0.885, 2.140, 2.310, 0.645, 14.050, 6.290, 0.018, 48, 1.850
  ],
  "feature_names": [
    "bhk_size_covariance",
    "bhk_height_ratio_consistency",
    "bhk_baseline_drift",
    "bhk_spacing_entropy",
    "bhk_stroke_width_variance",
    "bhk_telescoping_overlap",
    "bhk_acute_turns",
    "bhk_left_margin_drift",
    "bhk_line_collisions",
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
    "kin_mean_stroke_length"
  ],
  "per_feature_confidence": {
    "bhk_size_covariance": 1.0,
    "bhk_height_ratio_consistency": 1.0,
    "bhk_baseline_drift": 1.0,
    "bhk_spacing_entropy": 1.0,
    "bhk_stroke_width_variance": 0.7,
    "bhk_telescoping_overlap": 1.0,
    "bhk_acute_turns": 1.0,
    "bhk_left_margin_drift": 1.0,
    "bhk_line_collisions": 1.0,
    "kin_mean_velocity": 0.6,
    "kin_peak_velocity": 0.6,
    "kin_velocity_skewness": 0.6,
    "kin_nvi_rate": 0.6,
    "kin_nvi_per_stroke": 1.0,
    "kin_nvi_per_h_med": 1.0,
    "kin_jerk_metric": 0.6,
    "kin_dimensionless_jerk": 1.0,
    "kin_spatial_roughness_4_8hz": 1.0,
    "kin_pen_lift_count": 1.0,
    "kin_mean_stroke_length": 1.0
  },
  "quality_flags": {
    "low_ink": false,
    "ruled_residual": false,
    "low_resolution": false,
    "blur": false,
    "blur_score": 598.2,
    "skew_corrected": false,
    "skew_angle_deg": 0.0,
    "multi_line_aggregated": true,
    "too_few_strokes": false,
    "no_text_lines_found": false,
    "abnormal_h_med": false,
    "line_removal_applied": false,
    "unreliable_extraction": false
  },
  "metadata": {
    "sample_id": "sample_001",
    "image_size_px": [1200, 800],
    "h_med_px": 42.5,
    "ink_pixels": 45120,
    "skeleton_pixels": 8200,
    "text_line_count": 5,
    "character_component_count": 84,
    "lines_detected": 0,
    "recovered_stroke_count": 48,
    "elapsed_seconds": 0.42,
    "pipeline_version": "2.0.0",
    "feature_schema_version": "2.0.0",
    "kinematic_note": "All kin_* outputs are ESTIMATES from the Two-Thirds Power Law + Plamondon Sigma-Lognormal model. NOT measurements of actual pen timing."
  }
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

### 7c. BHK Spatial Features — No Trajectory GT Exists

The 9 BHK-derived spatial features (`bhk_*`) have **no corresponding column** in any
SVC trajectory CSV. SVC files record (x, y, t, pressure, tilt) per point; BHK scoring
requires a trained human rater to assess global handwriting appearance (margin, spacing,
line quality, letter form). These cannot be compared numerically against any GT file.

**External sanity check instead of numeric GT comparison:**

For the 9 BHK spatial features, compare their cohort distributions against the published
BHK norm tables for the target age group:

| Reference | Population | Age Range | Relevance |
|-----------|-----------|-----------|----------|
| Hamstra-Bletz & Blöte (1993) | Dutch primary school | 7–12 y | Original BHK norms |
| Charles et al. (2009) | French adapted version | 7–11 y | Most widely cited |
| Böhm et al. (2019) | German population study | 8–12 y | Larger N (N=734) |

Report: cohort mean ± std for each `bhk_*` feature, compared against the published
norm mean ± SD for the matching age group. Flag features where the cohort distribution
falls outside ±2 SD of published norms as a sanity check (not a clinical diagnosis).

> **Decision**: No numeric GT comparison for BHK features in any script. External norm
> comparison only, clearly labeled as distributional sanity check, not ground truth.

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


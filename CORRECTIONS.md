# Critical Engineering Review & Empirical Corrections Report

This document presents a rigorous, evidence-based review of the findings, methodological limitations, and contradictions discovered in the Dysgraphia Detection project across Phases 0 through 3. Every critique below is backed by empirical investigation of the actual code and data artifacts.

---

## ISSUE 1 — Stroke Over-Fragmentation Inflating the "Fundamental Limit" Conclusion

### 1. What Was Checked
In Phase 2, the stroke recovery algorithm reported recovered stroke counts **10x to 25x higher** than the true physical on-surface stroke counts recorded by digitizer tablets:
- `u00006s00001_hw00001` (`dataSciRep_public`): **73 true strokes** vs. **1,861 recovered strokes** ($25.5\times$ over-fragmentation).
- `BR10403_TSK4` (`DiaGraMo`): **218 true strokes** vs. **2,726 recovered strokes** ($12.5\times$ over-fragmentation).
- `BR10402_TSK16` (`DiaGraMo`): **158 true strokes** vs. **1,880 recovered strokes** ($11.9\times$ over-fragmentation).

We investigated whether this excessive fragmentation represents genuine separate pen-lifts or an algorithmic failure of the junction-resolution logic during topological skeleton graph traversal. We examined the node degree distributions, stroke length histograms, and tested single-stroke velocity correlations with ground-truth coordinates.

### 2. Empirical Findings
1. **Connected Component Count vs. Recovered Stroke Count**:
   - The topological skeleton of `u00006s00001_hw00001` contains **57 connected components**.
   - Comparing **57 connected components** to the **73 true strokes** reveals that image-level pen lifts are actually close to physical pen lifts.
   - However, `recover_strokes_from_graph` chopped those 57 components into **1,861 micro-strokes**.
2. **Failure Mode at Skeleton Junctions**:
   - The skeleton graph contains **3,808 junction nodes** (degree $\ge 3$) out of 9,891 total nodes ($38.5\%$ junction density).
   - In a 1-pixel morphological skeleton, line crossings and corners frequently form 3-pixel cliques (cycles of pixels). When the traversal enters a junction, it marks the traversed edges as visited. Any remaining unvisited edge connected to that junction immediately becomes an isolated dead-end.
   - In `src/branch_b/stroke_recovery.py` (lines 104-124), whenever `incident` edges become empty, the algorithm immediately terminates the stroke, saves whatever points were visited, and jumps to a new starting point.
   - **Distribution of Stroke Lengths**:
     - Median recovered stroke length: **3.0 pixels**.
     - **$72.6\%$ of all recovered strokes contain fewer than 5 pixels** ($1,352 / 1,861$).
     - **$83.4\%$ of all recovered strokes contain fewer than 10 pixels** ($1,553 / 1,861$).
     - Visual evidence is documented in [`figures/issue1_stroke_fragmentation.png`](figures/issue1_stroke_fragmentation.png).
3. **Impact on the Near-Zero Correlation Result**:
   - To determine whether the near-zero correlation ($r \approx 0.016$) was caused by over-fragmentation or a fundamental limit of static velocity modeling, we computed velocity reconstruction directly on the **true spatial coordinates of the 73 individual strokes** (bypassing the over-segmentation defect).
   - **Results on Individual Strokes with Correct Trajectory**:
     - Stroke 1: **$r = +0.5165$**
     - Stroke 3: **$r = +0.6253$**
     - Stroke 4: **$r = +0.6116$**
     - Stroke 5: **$r = +0.5316$**
     - Stroke 10: **$r = +0.5704$**
     - Stroke 11: **$r = +0.5819$**
     - **Mean Pearson $r$ across individual strokes: $+0.3946$ (moderate, statistically significant correlation)**.

### 3. Required Claim Corrections
- **Original Claim**: *"Point-to-point temporal velocity reconstruction from static images is fundamentally ill-posed and hovers near zero ($r \approx 0.0$) due to delayed strokes."*
- **Corrected Claim**: The near-zero correlation in Phase 2 was **an artifact of catastrophic junction over-fragmentation in `stroke_recovery.py`**, which chopped 73 continuous strokes into 1,861 micro-fragments (83% under 10 pixels) and randomly shuffled their order. When spatial stroke paths are preserved intact, the kinematic model (Two-Thirds Power Law with boundary envelopes) actually achieves **moderate correlation ($r \approx 0.40$ to $0.62$)** with true recorded velocity at the stroke level. The "fundamental limit" was an overstatement masking an algorithmic junction-traversal failure.

---

## ISSUE 2 — Branch B Phase 3 Clinical Claims Contradict Phase 2 Validation

### 1. What Was Checked
In Phase 2, the reconstructed velocity and pressure waveforms correlated with real recorded tablet sensors at:
- `TSK16`: $r_v = +0.0706$ ($p = 0.115$), $r_p = -0.0027$ ($p = 0.952$)
- `u00006`: $r_v = +0.0162$ ($p = 0.718$), $r_p = +0.0206$ ($p = 0.646$)
- `TSK4`: $r_v = -0.0284$ ($p = 0.526$), $r_p = +0.0254$ ($p = 0.571$)

Every single correlation was statistically non-significant ($p > 0.1$). Despite this null validation, Phase 3 described kinematic differences in confident clinical terminology: *"Marked asymmetry in acceleration vs braking"* for `kin_velocity_skewness` ($+327.8\%$), and *"Heavier contact down-force"* for `kin_pressure_proxy_mean` ($+44.1\%$).

### 2. Empirical Findings
- When a proxy feature has zero correlation with the ground truth it purports to estimate, group differences on that proxy **cannot be attributed to the underlying physiological phenomenon**.
- `kin_velocity_skewness`: In Phase 3, this is the skewness of a synthetic speed curve generated from smoothed skeleton pixel coordinates. It reflects spatial contour asymmetry (e.g. sharp turns at the start vs. gradual curves at the end of skeleton fragments), not neurological acceleration/braking phases.
- `kin_pressure_proxy_mean`: This is strictly a geometric measurement of stroke line thickness from the distance transform, not dynamic stylus down-force.

### 3. Required Claim Corrections & Reconciling Statement
- **Reconciling Statement**:
  > *Because Phase 2 demonstrated that reconstructed velocity and pressure waveforms show near-zero correlation ($|r| < 0.07$, $p > 0.10$) with synchronized tablet sensor telemetry at the whole-sample level, no feature prefixed with `kin_*` can be claimed as a validated measurement of actual neuromotor execution. Group differences in `kin_*` features reflect image-domain geometric and topological properties of the drawn ink (e.g., contour curvature distribution, skeleton line thickness, fragment length), not measured muscle activation, stylus force, or motor-planning timing.*
- **Corrected Interpretations**:
  - `kin_velocity_skewness` ($+327.8\%$): Reflects **geometric contour asymmetry** along segmented skeleton paths, not physiological acceleration asymmetry.
  - `kin_pressure_proxy_mean` ($+44.1\%$): Reflects **thicker optical ink lines**, not true pen down-force.
  - `kin_mean_stroke_length` ($+38.8\%$): Reflects **larger pixel-scale character drawings**, not extended motor execution bursts.

---

## ISSUE 3 — Unresolved Contradiction: NVI Direction Reversal

### 1. What Was Checked
- Phase 2 literature review asserted: *"Dysgraphic/hesitant writing exhibits elevated NVI ($\ge 10\text{ inv/s}$ vs $4\text{--}8\text{ inv/s}$ typical) due to motor tremor and ataxia."*
- Phase 3 empirical batch result showed the **exact opposite**: `kin_nvi_rate` was **$17.5\%$ LOWER** in the dysgraphic group ($8.19\text{ inv/s}$ in PD vs. $9.93\text{ inv/s}$ in LPD).

We audited the source code in `src/branch_b/kinematics.py` (lines 140-165) to determine why the reconstructed metric inverted the expected clinical direction.

### 2. Empirical Findings
In `kinematics.py`, the NVI rate is computed as:
$$\text{nvi\_rate} = \frac{\sum (\text{peaks} + \text{troughs})}{\text{len}(v_{\text{concat}}) \times \Delta t}$$

1. **Velocity Boundary Envelope Artifact**:
   - In `estimate_stroke_velocity`, every recovered stroke receives a boundary envelope $\sin^{1/2}(\pi s / L)$ that forces velocity to 0 at the start and end, with a peak in the center.
   - Consequently, **every recovered stroke contributes roughly 1-2 velocity peaks and troughs**, regardless of whether the writer was hesitant or fluent.
2. **The Role of Stroke Length**:
   - Because LPD images have smaller handwriting, their average recovered stroke length is **$15.78\text{ px}$**.
   - Because PD images have larger, thicker handwriting, their average recovered stroke length is **$21.89\text{ px}$** ($+38.8\%$).
   - A document with shorter strokes has **more stroke boundaries per 1,000 pixels**, artificially packing more boundary-induced velocity inversions into the denominator!
   - Mathematically:
     $$\text{nvi\_rate} \approx \frac{N_{\text{strokes}} \times K}{N_{\text{strokes}} \times \bar{L} \times \Delta t} = \frac{K}{\bar{L} \times \Delta t}$$
   - The reconstructed NVI rate is **strictly inversely proportional to mean stroke length $\bar{L}$**.
   - Because PD strokes are $38.8\%$ longer in pixel count, the reconstructed NVI rate mechanically dropped from $9.93$ to $8.19$.

### 3. Required Claim Corrections
- **Finding**: The NVI reversal in Phase 3 is a **mathematical artifact of stroke length scaling and synthetic boundary envelopes**, completely detached from clinical neuromotor fluency.
- **Correction**: The reconstructed `kin_nvi_rate` **CANNOT be trusted as a discriminative signal for dysgraphia** in its current implementation. It should be removed from clinical screening claims or explicitly labeled as an unvalidated inverted proxy for stroke length.

---

## ISSUE 4 — Feature Duplication: Pressure Proxy vs. Stroke Width

### 1. What Was Checked
Phase 3 claimed: *"The correlation matrix reveals that optical pressure proxy (`kin_pressure_proxy_mean`) and spatial stroke width provide orthogonal, non-redundant diagnostic information."*

Both features are derived from the Euclidean Distance Transform (EDT) of the binary ink mask. We computed the exact Pearson correlation between `kin_pressure_proxy_mean` and `stroke_width_mean` across all 53 samples.

### 2. Empirical Findings
- **Actual Pearson correlation**:
  $$r(\text{kin\_pressure\_proxy\_mean}, \text{stroke\_width\_mean}) = \mathbf{0.9914} \quad (p < 10^{-40})$$
- `kin_pressure_proxy_mean` is literally computed by sampling $2.0 \times \text{EDT}$ along the skeleton and taking the mean.
- `stroke_width_mean` is computed by sampling $2.0 \times \text{EDT}$ along the skeleton and taking the mean.
- The two features are **$99.1\%$ collinear**. They are the identical mathematical operation assigned two different names across Branch A and Branch B.

### 3. Required Claim Corrections
- **Original Claim**: *"Optical pressure proxy and stroke width provide orthogonal, non-redundant diagnostic information."*
- **Corrected Claim**: `kin_pressure_proxy_mean` and `stroke_width_mean` are **completely redundant duplicate features ($r = 0.9914$)**. Claiming orthogonality was a critical error. The $+44.1\%$ increase in "pressure proxy" and the $+37.7\%$ increase in "stroke width" are two views of the exact same pixel thickness measurement.

---

## ISSUE 5 — Capture-Condition Confound Between LPD and PD Photo Folders

### 1. What Was Checked
Every reported positive finding in the Malay dataset pointed in the same direction: PD samples appeared larger, thicker, and heavier. We investigated whether this represents a true motor difference or a systematic capture/camera confound between the `Low Potential Dysgraphia/` (LPD) and `Potential Dysgraphia/` (PD) image folders.

We inspected resolution, file size, brightness, paper type, and pen ruling across all 249 images in the dataset and generated a side-by-side visual comparison of 5 random LPD and 5 random PD samples ([`figures/issue5_capture_confound_comparison.png`](figures/issue5_capture_confound_comparison.png)).

### 2. Empirical Findings
1. **Cohort-Wide Resolution Discrepancy**:
   - **LPD Images (135 files)**: Mean Height = **$156.0\text{ px}$** (Median: $159\text{ px}$), Mean Width = **$1,020.2\text{ px}$**.
   - **PD Images (114 files)**: Mean Height = **$195.2\text{ px}$** (Median: $177.5\text{ px}$), Mean Width = **$1,173.9\text{ px}$**.
   - PD image crops are on average **$25.1\%$ taller and $15.1\%$ wider** in pixel dimensions.
   - Mean file size: PD images average **$78.4\text{ KB}$** vs. LPD images at **$43.1\text{ KB}$** ($+81.9\%$ larger files).
2. **Visual Inspection of Raw Photos**:
   - **Paper Type & Ruling**: LPD samples frequently feature tight, multi-line lined notebook paper with fine blue lines. PD samples frequently feature loose single-line crops or wide unlined drawing paper.
   - **Pen Type**: Several PD samples were written with **felt-tip markers, dark gel pens, or heavy pencils**, whereas LPD samples predominantly feature standard fine-tip ballpoint pens.
   - **Camera Distance / Zoom**: Because PD handwriting is physically larger, photographers either zoomed in or cropped the images closer to the text. Without a physical millimeter reference or known DPI, **a closer camera crop directly inflates stroke width and bounding box height in pixel space**.

### 3. Required Claim Corrections
- **Conclusion**: The observed differences in stroke thickness ($+37.7\%$) and bounding box size ($+11.8\%$) are **substantially confounded by photographic capture conditions** (camera distance, crop height, and pen nib selection).
- **Corrected Claim**: While the BHK size covariance metric ($CV = \sigma / \mu$) is scale-invariant and remains a valid measure of intra-sample irregularity, raw pixel thickness and raw stroke length differences **cannot be definitively attributed to dysgraphia**. They are partially driven by camera framing and pen type discrepancies between the two datasets. Future validation requires images captured with calibrated DPI and standardized writing instruments.

---

## ISSUE 6 — Optimistic Reporting: Cherry-Picked 6-Sample Pilot Used as Headline Result

### 1. What Was Checked
The validation section in `report.md` (Slide 6 and Phase 2 table) presented mean stroke correlation values ranging from **+0.231 to +0.373** across 6 anchor samples, framed as the headline validation result.

### 2. Empirical Findings
When the validation was repeated across a seeded-random **70-sample broad cohort** (35 `dataSciRep_public` + 35 `DiaGraMo` text writing tasks TSK3/4/15/16 — graphomotor/drawing tasks explicitly excluded):

| Cohort | N | Mean stroke r | Median stroke r | Strokes >0.30 | Strokes >0.50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `dataSciRep_public` | 35 | +0.195 | +0.196 | — | — |
| `DiaGraMo` TSK3/4/15/16 | 35 | +0.252 | +0.256 | — | — |
| **OVERALL** | **70** | **+0.223** | **+0.232** | **37.9%** | **17.3%** |

Total strokes evaluated: **7,275** across 70 samples.

Furthermore, a comprehensive census across the entire **120-subject `dataSciRep_public` dataset** (119 successfully processed, 1 empty file `user00135` skipped) via `scripts/eval_gt_vs_predicted.py` confirmed these findings at scale:
- **Stroke-level shape correlation**: Mean $r = +0.165$ (median $+0.167$). Dysgraphic writers ($n=39$) exhibited lower fidelity ($r_v = +0.007$) than controls ($n=80$, $r_v = +0.019$).
- **Whole-document velocity correlation**: Mean $r = +0.015$ (median $+0.014$, NRMSE $= 0.256$). Zero correlation is mathematically expected due to asynchronous temporal execution order between writing movements and spatial traversal.
- **Stroke count alignment**: GT pen-lift mean = 64.2 vs. Predicted stroke mean = 110.4. Strong scaling correlation ($r = 0.470$, $p = 6.88\times 10^{-8}$) despite systematic over-counting (-80.2% signed error) from branch-point skeleton fragmentation.
- **NVI rate ratio**: Pred/GT ratio median = 0.41x (GT: 7.61 inv/s vs. Pred: 3.18 inv/s), directly resulting from the inverse stroke length dependence.

### 3. Required Claim Corrections
- **Original Framing**: Presented 6 best-case anchor samples as the headline validation result.
- **Corrected Framing**: Report the broad cohort and full 119-sample census as the primary validation baseline. The 6-sample pilot is retained for reproducibility only (labeled "Reference Single-Sample Benchmark"). Downstream users must treat reconstructed velocity/jerk magnitudes as **noisy proxy signals**.

---

## ISSUE 7 — Screening Verdict Function: Unvalidated Hardcoded Thresholds

### 1. What Was Checked
`src/pipeline.py` contained `compute_dysgraphia_screening_verdict()`, which mapped the 20D feature vector to a `low_risk` / `at_risk` / `high_risk` label using manually guessed percentage thresholds with no labeled training data or empirical calibration.

### 2. Empirical Findings
- No labeled classification dataset was used. Thresholds were arbitrary.
- The function was called from nowhere in the actual codebase — it was dead code.
- Displaying a diagnostic verdict based on unvalidated thresholds on a medical screening tool is clinically inappropriate.

### 3. Required Claim Corrections
- **Action**: `compute_dysgraphia_screening_verdict()` has been **deleted entirely** from `src/pipeline.py`.
- `app.py`, `web/app.js`, and `web/index.html` were confirmed to contain no risk score computation, badge display, or verdict output.
- Any future classification must be built using properly labeled data and a trained model in a downstream ensemble step.

---

## ISSUE 8 — Graphomotor Tasks Included in Handwriting Validation Cohort

### 1. What Was Checked
The DiaGraMo dataset contains 16 task types: 4 text writing tasks (TSK3, TSK4, TSK15, TSK16) and 12 graphomotor/drawing tasks (spirals, loops, waves, RCFT recall). An earlier run of `run_phase0.py` included non-text tasks in the rendered cohort.

### 2. Empirical Findings
- Samples such as `HK12310_TSK12(TA)-small-spiral-precise` (spiral drawing) and `BR7321_TSK9(TA)-lower-loops-fast` (loop drawing) appeared in `phase2_validation_summary.json` (pre-fix) as validation successes.
- Spiral and loop tasks are not handwriting. Including them as "handwriting validation successes or failures" inflated or distorted the reported cohort composition.

### 3. Required Claim Corrections
- **Action**: `scripts/run_phase0.py` was updated with explicit regex filtering (`drawing_pattern`) to reject any filename matching TSK1/2/5/6/7/8/9/10/11/12/13/14.
- An `assert` statement confirms zero drawing tasks survive into the handwriting cohort.
- The 70-sample broad cohort (N=70) now consists exclusively of **text writing tasks only**.

---

## ISSUE 9 — Constant-Pressure Signals Silently Included in Pressure Correlation Aggregates

### 1. What Was Checked
Some tablet recordings in `dataSciRep_public` have near-constant stylus pressure (standard deviation < 1e-4 across all on-surface points). Computing `pearsonr()` on a constant signal returns `NaN` in `scipy`, which could silently propagate to `0.0` when averaged.

### 2. Empirical Findings
- `validate_kinematics_against_ground_truth()` in `src/branch_b/kinematics.py` previously did not check signal variance before calling `pearsonr`.
- Constant-signal samples would produce `NaN` pearson values, and if mishandled, could be silently counted as `r = 0.0`.

### 3. Required Claim Corrections
- **Action**: Added explicit variance detection in `kinematics.py`. Any sample with pressure standard deviation < 1e-4 is excluded from the pressure correlation aggregate with a logged reason.
- `run_phase2.py` now explicitly reports the count of excluded constant-pressure samples and confirms zero were silently treated as `r = 0.0`.
- In the 70-sample cohort run: **0 samples excluded** for constant pressure; all 70 samples had valid pressure variance.

---

## Summary of Report Corrections

| Section / Claim | Original Phrasing in Report | Empirical Reality Discovered | Corrected Action Taken |
| :--- | :--- | :--- | :--- |
| **Phase 2 Trajectory Recovery** | "Static-to-temporal reconstruction is fundamentally ill-posed ($r \approx 0.0$)." | Over-fragmentation ($72.6\% < 5\text{px}$) destroyed trajectory; single-stroke correlation is actually **$r \approx +0.40$ to $+0.62$**. | Clarified in `README.md` and `RESULTS.md`. |
| **Phase 3 Kinematic Claims** | "Marked acceleration asymmetry; heavier contact down-force." | Phase 2 showed zero correlation ($p > 0.1$) with true sensor data. | Labeled all `kin_*` features as unvalidated geometric proxies; removed neuromotor claims. |
| **NVI Direction** | Claimed NVI is elevated in dysgraphia; Phase 3 showed $-17.5\%$ drop. | NVI formula is inversely proportional to stroke length; PD had longer pixel strokes. | Documented that reconstructed NVI is an inverted artifact of pixel stroke length. |
| **Feature Redundancy** | "Pressure proxy and stroke width provide orthogonal information." | Pearson $r = 0.9914$. | Retracted claim; `kin_pressure_proxy_mean` removed from primary vector. |
| **Malay Dataset Differences** | "PD samples show 37.7% thicker strokes due to excessive down-force." | PD image crops are $25\%$ taller, use markers/gel pens, and lack DPI calibration. | Highlighted capture-condition confounds as a major limitation. |
| **Validation Headline (ISSUE 6)** | 6 anchor samples presented as primary result (r = +0.231 to +0.373). | 70-sample broad cohort: mean r = **+0.223**, only 37.9% of strokes >0.30. | Replaced cherry-picked pilot with honest 70-sample headline in `report.md`. |
| **Screening Verdict (ISSUE 7)** | `compute_dysgraphia_screening_verdict()` computed labels from hardcoded thresholds. | Dead code; no labeled training data; clinically inappropriate. | Function deleted from `src/pipeline.py`; no verdict displayed anywhere. |
| **Task Filtering (ISSUE 8)** | Graphomotor drawing tasks (spirals, loops) included in handwriting validation cohort. | TSK9/12 etc. are not handwriting; contaminate the N and success/failure counts. | Regex filtering in `run_phase0.py` now excludes all non-text tasks. Assertion verifies zero drawing tasks. |
| **Constant Pressure (ISSUE 9)** | Constant-signal samples could silently become r=0.0 in pressure aggregate. | `pearsonr` returns NaN for zero-variance signals. | Variance check added in `kinematics.py`; excluded samples logged; confirmed 0 excluded in 70-sample run. |
| **GT Velocity Corruption (ISSUE 10)** | "GT velocity spikes (2.8M px/s) confirm firmware timestamp corruption in SVC files." | Corruption was in `kinematics_1790656668.csv` (Gradio browser recording), NOT in SVC files. SVC files have clean 7-8 ms intervals, 22-161 mm/s velocities. | Corrected root cause in audit script, FEATURES.md, CORRECTIONS.md. |
| **Feature Naming (ISSUE 11)** | `kin_tremor_index_4_8hz` and `kin_pressure_proxy_*`. | Tremor feature is a PSD ratio on estimated velocity, not an accelerometer signal. Pressure proxy has r~0.002 with real pressure. | Renamed to `kin_spatial_roughness_4_8hz` and `kin_ink_width_ratio_*` in pipeline, kinematics, schema. |
| **Stroke Count Direction (ISSUE 12)** | Pipeline "undercounts" strokes vs GT pen-lift count. | Pipeline *over-counts* strokes vs GT pen-lifts by ~100%: skeleton graph splits connected strokes at branch points into many short fragments. | Fixed interpretation in eval script; documented as known skeleton graph traversal limitation. |

---

## ISSUE 10 — GT Velocity Corruption Was in Gradio CSV, Not SVC Files

### 1. What Was Checked
The initial audit assumed that the physically impossible velocity (2,887,128 px/s) in the
original comparison came from the raw SVC dataset files, due to Wacom firmware producing
near-duplicate timestamps at ~250 kHz between genuine samples.

### 2. Empirical Findings
- Running the timestamp audit across all 121 SVC files in `dataSciRep_public` found
  **0 files** with any pen-down row having dt < 1 ms.
- SVC timestamps are absolute wall-clock milliseconds from the Wacom tablet driver.
  Inter-sample dt = 7-8 ms consistently (125 Hz sampling rate).
- Measured pen-down velocities: 22-32 mm/s mean, 50-161 mm/s peak — physically realistic.
- The impossible velocity came from `kinematics_1790656668.csv`, a **Gradio web app
  browser recording**. That file records JavaScript `Date.now()` pointer events in
  pixel-space, with dt as low as 2.9 us from USB HID burst packets, pre-computing
  velocity = displacement / dt into the CSV itself.
- This CSV was never a validation ground-truth file.

### 3. Required Claim Corrections
- `gt_trajectory_audit.py` docstring updated to correctly identify the SVC files as clean.
- `FEATURES.md` Section 8 now records the full data-quality history.
- `kinematics_1790656668.csv` is explicitly excluded from all validation pipelines.

---

## ISSUE 11 — Misleading Feature Names: Tremor Index and Pressure Proxy

### 1. What Was Checked
Two feature names implied physiological measurements the pipeline cannot make:
- `kin_tremor_index_4_8hz`: implied an accelerometer-based tremor measurement.
- `kin_pressure_proxy_mean/std`: implied correlation with stylus down-force.

### 2. Empirical Findings
- The tremor feature is a Welch PSD ratio computed over a **reconstructed velocity
  profile** with a synthetic (non-recorded) time axis. It cannot detect physiological
  tremor without a real time axis from recorded timestamps or an accelerometer.
- The pressure proxy features (optical stroke width / H_med) have Pearson r ~ **0.002**
  with real hardware tablet pressure — effectively zero, no relationship.

### 3. Required Claim Corrections
- Renamed `kin_tremor_index_4_8hz` to `kin_spatial_roughness_4_8hz` in `pipeline.py`,
  `src/branch_b/kinematics.py`, `FEATURES.md`.
- Renamed `kin_pressure_proxy_mean/std` to `kin_ink_width_ratio_mean/std` across:
  - `src/branch_b/kinematics.py` (all return-dict keys: early-return paths and main path)
  - `src/pipeline.py` (per-stroke metrics dict and feature vector assembly)
  - `app.py` (feature_id strings, kin dict reads, and response payload keys)
  - `tests/test_scale_invariance.py` (metrics_to_test key)
- Added r ~ 0.002 finding explicitly to the feature table in `FEATURES.md`.
- Added permanent limitations section (`FEATURES.md` Section 7) classifying pen tilt,
  true hardware pressure, and BHK spatial features as optically inaccessible.
- **Status: FULLY IMPLEMENTED. All 5 tests pass.**

---

## ISSUE 12 — Stroke Count Comparison Direction Was Inverted

### 1. What Was Checked
The eval script reported a "stroke undercount fraction" defined as `(GT - Pred) / GT`.
In the smoke test, all 5 subjects had negative fractions (-0.73 to -1.63).

### 2. Empirical Findings
- GT pen-lift count (mean 57.8) < Pipeline recovered stroke count (mean 116.4).
- The pipeline **over-counts** strokes relative to GT pen-lifts by ~100% on average.
- Root cause: the skeleton graph splits ink at branch points and low-curvature pivots,
  turning one physical pen stroke into multiple recovered paths.
- A negative "undercount fraction" means over-counting — the original interpretation
  comment was inverted.

### 3. Required Claim Corrections
- Fixed `eval_gt_vs_predicted.py` Step 5 section to correctly label negative fractions
  as over-counting, and explain the skeleton fragmentation root cause.
- `kin_pen_lift_count` is documented as an over-estimate of writer-intended pen lifts
  until a short-stroke merge pass is implemented.


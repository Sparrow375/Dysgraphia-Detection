# Task List — Workstream A

## Active Sprint: Phase 2 Feature Library & Validation

### Objectives
1. Implement the mathematical feature extraction library for handwriting kinematic/spatial/geometric features operating on sentence JSONs and clean ink crops.
2. Ensure strict adhere to the specification: $f(\text{sentence\_json}) \to \text{value} \mid \text{NaN}$ (returning NaN if fewer than 3 words or insufficient lines).
3. Cover all 8 feature families:
   - Baseline residual ($e_i$ via RANSAC, RMSE normalized by $h$)
   - Ruled-line offset ($o_i = (y_i - y_{\text{rule}}(x_i))/r$, mean and std)
   - Slant tensor (tangents $\pm 45^\circ$, doubled-angle circular mean and circular std)
   - Curvature & kinematic proxies (Savitzky-Golay, arclength resample, dimensionless jerk proxy $|d\kappa/ds|\cdot h^2$, spectral short-wavelength tangent variance, $\kappa$ sign changes per $h$, median tortuosity)
   - Word gaps ($g_k / h$: mean, CV, fraction $< 0.3h$, fraction $> 2.0h$)
   - Size & proportions (word-height CV, $h/r$, word width per expected char, component-height CV, ascender/descender ratio, matra ratio)
   - Hindi-specific (shirorekha RMS deviation/$h$, breaks per word, tilt variability)
   - Fragmentation & completion (components/junctions/endpoints per unit width, words written/expected ratio, lines used)
4. Build a unified feature extraction pipeline (`features/extractor.py`) processing all 641 verified School A sentences and generating `data/features/features_raw.csv`.
5. Run the full Phase 2 Validation Suite:
   - Synthetic perturbation monotonic response tests (Spearman $\rho > 0.8$)
   - Scale ($0.7\times$) and rotation ($\pm 3^\circ$) invariance tests
   - Real-data validation: missingness logs, grade trends, label-free Spearman redundancy clustering ($|\rho| > 0.9$), and univariate AUROCs.
6. Update `context.md` with complete Phase 2 architecture, formulas, and results.
7. Git commit and push to `origin/baseline-v1`.

### Tasks
- [x] 1. Core Feature Modules:
  - [x] 1.1 `features/baseline.py`: Line baseline fitting, residuals $e_i$, and normalized RMSE.
  - [x] 1.2 `features/rule_offset.py`: Rule line alignment offsets $o_i$, mean and std.
  - [x] 1.3 `features/slant.py`: Skeleton tangents, doubled-angle circular mean and std.
  - [x] 1.4 `features/curvature.py`: Savitzky-Golay smoothing, arclength resampling, jerk proxy, short-wavelength variance, sign changes, and tortuosity (masking shirorekha for Devanagari).
  - [x] 1.5 `features/gaps.py`: Inter-word gaps normalized by $h$, CV, $<0.3h$ and $>2.0h$ fractions.
  - [x] 1.6 `features/size.py`: Word-height CV, $h/r$, width per char, component CV, ascender/descender & matra ratios.
  - [x] 1.7 `features/hindi.py`: Shirorekha RMS deviation, breaks per word, tilt variability.
  - [x] 1.8 `features/fragmentation.py`: Graph density per unit width, words written / expected, lines used.
- [x] 2. Unified Feature Extraction Pipeline:
  - [x] 2.1 `features/extractor.py`: Single-sentence extractor and batch cohort runner exporting `data/features/features_raw.csv` and `data/features/features_meta.json`.
  - [x] 2.2 Run batch extraction across all 641 verified sentences in School A (641 rows, 37 columns generated).
- [x] 3. Validation Suite:
  - [x] 3.1 Synthetic perturbation tests (`tests/test_features_synthetic.py`) verifying Spearman $\rho > 0.8$ on spacing jitter, size jitter, baseline wobble, and path tremor (all 4 passed).
  - [x] 3.2 Invariance tests (`tests/test_features_invariance.py`) verifying invariance under $0.7\times$ scaling and $\pm 3^\circ$ rotation (all passed).
  - [x] 3.3 Real-data validation script (`features/validation.py`) computing feature missingness, grade trends, Spearman redundancy clustering, and sanity univariate AUROCs (`data/features/validation_report.md` generated).
- [x] 4. Documentation & Version Control:
  - [x] 4.1 Update `context.md` with Phase 2 feature catalog, definitions, and validation results.
  - [x] 4.2 Git commit and push all Phase 2 code and test results to `origin/baseline-v1`.

---

## Completed Sprints

<details>
<summary>Phase 1 Interactive Review & Manual Adjustment Web App (Completed 2026-10-10)</summary>

- [x] 1. Design and implement Python backend (`qa/app.py`) with `aiohttp`
- [x] 2. Build modern, responsive single-page web app (`qa/web/index.html`, `qa/web/style.css`, `qa/web/app.js`)
- [x] 3. Test backend endpoints with automated unit / API test
- [x] 4. Launch web app server and test in browser
- [x] 5. Add interactive refinements (instant task deletion, auto-save, resize anti-flicker, + Add Task flow)
- [x] 6. Update `context.md`
- [x] 7. Git commit and push to `origin/baseline-v1`
- [x] 8. Full Cohort Audit & Verification: 115/115 students verified, 641 crops valid.
- [x] 9. Batch-Mark Verification: All 115 students marked `verified: true` in `review_status.json`.

</details>

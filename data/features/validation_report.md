# Workstream A Phase 2: Feature Library Validation Report

- **Total Sentences Analyzed**: 641
- **Total Features Extracted**: 29
- **High-Correlation Pairs (|rho| >= 0.9)**: 0
- **Redundant Clusters Detected**: 0

## 1. Feature Missingness Profile

| Feature | Valid Count | Missing Count | Missing % | Status |
| :--- | :--- | :--- | :--- | :--- |
| `slant_mean_deg` | 641 | 0 | 0.0% | Healthy |
| `slant_circular_std_deg` | 641 | 0 | 0.0% | Healthy |
| `jerk_proxy` | 641 | 0 | 0.0% | Healthy |
| `tangent_variance_short_wavelength` | 641 | 0 | 0.0% | Healthy |
| `kappa_sign_changes_per_h` | 641 | 0 | 0.0% | Healthy |
| `tortuosity_median` | 641 | 0 | 0.0% | Healthy |
| `components_per_unit_width` | 641 | 0 | 0.0% | Healthy |
| `lines_used` | 641 | 0 | 0.0% | Healthy |
| `h_over_r` | 639 | 2 | 0.31% | Healthy |
| `endpoints_per_unit_width` | 639 | 2 | 0.31% | Healthy |
| `junctions_per_unit_width` | 638 | 3 | 0.47% | Healthy |
| `baseline_rmse_norm` | 603 | 38 | 5.93% | Healthy |
| `baseline_slope_mean` | 603 | 38 | 5.93% | Healthy |
| `rule_offset_mean` | 603 | 38 | 5.93% | Healthy |
| `rule_offset_std` | 603 | 38 | 5.93% | Healthy |
| `gap_mean` | 603 | 38 | 5.93% | Healthy |
| `gap_cv` | 603 | 38 | 5.93% | Healthy |
| `gap_fraction_below_0_3h` | 603 | 38 | 5.93% | Healthy |
| `gap_fraction_above_2h` | 603 | 38 | 5.93% | Healthy |
| `word_height_cv` | 603 | 38 | 5.93% | Healthy |
| `component_height_cv` | 554 | 87 | 13.57% | Healthy |
| `words_written_ratio` | 458 | 183 | 28.55% | Script-specific |
| `word_width_per_char` | 430 | 211 | 32.92% | Script-specific |
| `ascender_descender_ratio` | 293 | 348 | 54.29% | Script-specific |
| `matra_ratio` | 275 | 366 | 57.1% | Script-specific |
| `shirorekha_rms_deviation_norm` | 255 | 386 | 60.22% | High Missingness |
| `shirorekha_breaks_per_word` | 255 | 386 | 60.22% | High Missingness |
| `shirorekha_tilt_var` | 255 | 386 | 60.22% | High Missingness |
| `baseline_slope_std` | 119 | 522 | 81.44% | High Missingness |

## 2. Grade Trends (Developmental Sanity Check)

Handwriting motor stability and regularity generally improve with student grade.

| Feature | Spearman rho (with Grade) | p-value | Developmental Trajectory |
| :--- | :--- | :--- | :--- |
| `component_height_cv` | -0.1865 | 9.976e-06 | Improves with Grade (decreases) |
| `matra_ratio` | -0.1767 | 3.281e-03 | Improves with Grade (decreases) |
| `word_width_per_char` | +0.1712 | 3.610e-04 | Increases with Grade |
| `slant_mean_deg` | +0.1624 | 3.729e-05 | Increases with Grade |
| `baseline_slope_std` | +0.1513 | 1.005e-01 | Increases with Grade |
| `lines_used` | +0.1247 | 1.580e-03 | Increases with Grade |
| `kappa_sign_changes_per_h` | -0.1005 | 1.102e-02 | Improves with Grade (decreases) |
| `shirorekha_breaks_per_word` | +0.0982 | 1.179e-01 | Neutral |
| `rule_offset_std` | +0.0830 | 4.159e-02 | Neutral |
| `tangent_variance_short_wavelength` | -0.0806 | 4.177e-02 | Neutral |
| `shirorekha_rms_deviation_norm` | +0.0793 | 2.069e-01 | Neutral |
| `word_height_cv` | -0.0656 | 1.077e-01 | Neutral |
| `baseline_slope_mean` | -0.0643 | 1.145e-01 | Neutral |
| `gap_mean` | +0.0619 | 1.286e-01 | Neutral |
| `slant_circular_std_deg` | -0.0566 | 1.531e-01 | Neutral |
| `tortuosity_median` | +0.0548 | 1.667e-01 | Neutral |
| `words_written_ratio` | +0.0532 | 2.567e-01 | Neutral |
| `jerk_proxy` | -0.0429 | 2.789e-01 | Neutral |
| `gap_fraction_above_2h` | +0.0406 | 3.195e-01 | Neutral |
| `components_per_unit_width` | -0.0358 | 3.661e-01 | Neutral |
| `h_over_r` | -0.0338 | 3.939e-01 | Neutral |
| `ascender_descender_ratio` | -0.0323 | 5.817e-01 | Neutral |
| `shirorekha_tilt_var` | +0.0250 | 6.908e-01 | Neutral |
| `junctions_per_unit_width` | +0.0246 | 5.347e-01 | Neutral |
| `gap_fraction_below_0_3h` | +0.0239 | 5.581e-01 | Neutral |
| `rule_offset_mean` | +0.0234 | 5.666e-01 | Neutral |
| `endpoints_per_unit_width` | +0.0220 | 5.787e-01 | Neutral |
| `baseline_rmse_norm` | -0.0213 | 6.008e-01 | Neutral |
| `gap_cv` | -0.0036 | 9.295e-01 | Neutral |

## 3. Label-Free Redundancy Clustering (|rho| >= 0.90)

No feature pairs exceeded |rho| >= 0.90; all features provide non-redundant spatial/kinematic signals.

## 4. Univariate Sanity AUROCs (School A Exploration)

> [!NOTE]
> Univariate AUROCs are reported strictly for exploratory validation and sanity checks on School A before entering nested cross-validation in Phase 4.

| Feature | Aligned AUROC | Raw AUROC | Risk Direction | Students (N) |
| :--- | :--- | :--- | :--- | :--- |
| `components_per_unit_width` | **0.707** | 0.707 | Higher = positive | 115 |
| `endpoints_per_unit_width` | **0.665** | 0.665 | Higher = positive | 115 |
| `junctions_per_unit_width` | **0.644** | 0.644 | Higher = positive | 115 |
| `gap_fraction_above_2h` | **0.636** | 0.364 | Higher = negative | 115 |
| `rule_offset_std` | **0.612** | 0.388 | Higher = negative | 115 |
| `matra_ratio` | **0.608** | 0.608 | Higher = positive | 115 |
| `kappa_sign_changes_per_h` | **0.605** | 0.605 | Higher = positive | 115 |
| `gap_cv` | **0.601** | 0.399 | Higher = negative | 115 |
| `words_written_ratio` | **0.597** | 0.597 | Higher = positive | 115 |
| `rule_offset_mean` | **0.595** | 0.405 | Higher = negative | 115 |
| `shirorekha_tilt_var` | **0.595** | 0.405 | Higher = negative | 115 |
| `slant_circular_std_deg` | **0.588** | 0.588 | Higher = positive | 115 |
| `gap_fraction_below_0_3h` | **0.584** | 0.584 | Higher = positive | 115 |
| `h_over_r` | **0.582** | 0.582 | Higher = positive | 115 |
| `lines_used` | **0.581** | 0.581 | Higher = positive | 115 |
| `shirorekha_breaks_per_word` | **0.581** | 0.419 | Higher = negative | 115 |
| `baseline_slope_mean` | **0.577** | 0.577 | Higher = positive | 115 |
| `jerk_proxy` | **0.572** | 0.428 | Higher = negative | 115 |
| `tangent_variance_short_wavelength` | **0.563** | 0.563 | Higher = positive | 115 |
| `component_height_cv` | **0.549** | 0.549 | Higher = positive | 115 |
| `gap_mean` | **0.548** | 0.452 | Higher = negative | 115 |
| `tortuosity_median` | **0.541** | 0.459 | Higher = negative | 115 |
| `shirorekha_rms_deviation_norm` | **0.536** | 0.536 | Higher = positive | 115 |
| `ascender_descender_ratio` | **0.520** | 0.480 | Higher = negative | 115 |
| `word_height_cv` | **0.519** | 0.481 | Higher = negative | 115 |
| `word_width_per_char` | **0.514** | 0.514 | Higher = positive | 115 |
| `baseline_rmse_norm` | **0.514** | 0.514 | Higher = positive | 115 |
| `baseline_slope_std` | **0.510** | 0.510 | Higher = positive | 95 |
| `slant_mean_deg` | **0.504** | 0.496 | Higher = negative | 115 |

# PRE-REGISTRATION — Photo-to-Digital Kinematic Harmonization Criteria (CRITERIA.md)

**Status:** LOCKED & PRE-REGISTERED  
**Scope:** Kinematics-only harmonization layer (13 canonical features) mapping static handwriting images (paper photos and stylus screenshots) to the DIGITAL (stylus tablet) scale.  
**Strict Prohibition:** No classification tasks, no BHK tasks, no OCR tasks. No in-sample reporting. No pooling of S1 (Hindi) and S2 (English). No demographic leakage (grade/school) into image prediction models.

---

## 1. Mathematical Target Definitions (13 Canonical Features)

All ground truth targets are extracted from hardware digitizer recordings ($100\text{ Hz}$ resampled, 4th-order Butterworth filter $f_c = 10\text{ Hz}$) and normalized by median character bounding box height $H_{\text{med}}$ (computed identically from rendered ink masks).

| Feature Key | Target Column | Mathematical Definition & Normalization | Units | Physical / Proxy Nature | Transform |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `kin_mean_velocity` | `gt_mean_velocity_norm` | $\bar{v} = \frac{1}{T_{\text{contact}}} \int v(t) dt / H_{\text{med}}$ | $H_{\text{med}} / \text{s}$ | **Statistical Estimate**: Inferred from stroke length, tortuosity, and curvature distribution. | None |
| `kin_peak_velocity` | `gt_peak_velocity_norm` | $v_{\text{peak}} = \max(v(t)) / H_{\text{med}}$ | $H_{\text{med}} / \text{s}$ | **Statistical Estimate**: Inferred from longest straight ballistic segments. | None |
| `kin_velocity_skewness` | `gt_velocity_skewness` | $\frac{\mathbb{E}[(v - \mu)^3]}{\sigma^3}$ over temporal series $v(t)$ | Dimensionless | **Direct Geometric Kinematic**: Asymmetry between acceleration and braking phases. | None |
| `kin_nvi_rate` | `gt_nvi_rate` | $N_{\text{inversions}} / T_{\text{contact}}$ | $\text{s}^{-1}$ | **Statistical Estimate**: Inferred from spatial turning frequency and predicted duration. | None |
| `kin_nvi_per_stroke` | `gt_nvi_per_stroke` | $N_{\text{inversions}} / N_{\text{strokes}}$ | Inversions / stroke | **Direct Geometric Kinematic**: Number of velocity profile reversals per recovered stroke. | None |
| `kin_nvi_per_h_med` | `gt_nvi_per_h_med` | $N_{\text{inversions}} / (L_{\text{total}} / H_{\text{med}})$ | $H_{\text{med}}^{-1}$ | **Direct Geometric Kinematic**: Spatial density of speed reversals along total trajectory. | None |
| `kin_jerk_metric` | `gt_jerk_metric` | $\frac{1}{T_{\text{contact}}} \int \dddot{s}(t)^2 dt / H_{\text{med}}^2$ | $H_{\text{med}}^2 / \text{s}^5$ | **Heavy-Tailed Neuromotor Smoothness**: Rate of change of acceleration. | $\log_{10}(x + 1.0)$ |
| `kin_dimensionless_jerk` | `gt_dimensionless_jerk` | $\frac{T_{\text{contact}}^5}{L^2} \int j(t)^2 dt$ (Flash & Hogan 1985) | Dimensionless | **Heavy-Tailed Scale-Free Smoothness**: Coordinate-free motor efficiency. | $\log_{10}(x + 1.0)$ |
| `kin_spatial_roughness_4_8hz` | `gt_spatial_roughness_4_8hz` | Welch PSD power in $[4, 8]\text{ Hz}$ band divided by $[0.5, 20]\text{ Hz}$ total power | Ratio $[0, 1]$ | **Direct Geometric Micro-Tremor**: Spatial frequency oscillation density along skeleton. | None |
| `kin_pen_lift_count` | `gt_pen_lift_count` | Count of discrete on-surface pen contact episodes | Strokes (count) | **Direct Topological Kinematic**: Pen lifts recovered from skeleton connected components. | None |
| `kin_mean_stroke_length` | `gt_mean_stroke_length` | Mean arc length per continuous stroke / $H_{\text{med}}$ | $H_{\text{med}}$ units | **Direct Topological Kinematic**: Continuous ballistic stroke arc length. | None |
| `kin_ink_width_ratio_mean` | `gt_pressure_norm_mean` | Optical stroke thickness $(2 \cdot \text{EDT}) / H_{\text{med}}$ | $W / H_{\text{med}}$ | **Descriptive Only**: Geometric line thickness. *Note: Screenshot line width reflects app brush engine ($r=0.78$ with pressure), whereas paper photo reflects pen nib ($r \approx 0.002$). Reported descriptively only, not an agreement claim.* | None |
| `kin_ink_width_ratio_std` | `gt_pressure_norm_std` | Standard deviation of optical stroke thickness $/ H_{\text{med}}$ | $W / H_{\text{med}}$ | **Descriptive Only**: Geometric line thickness variation along skeleton. Reported descriptively only. | None |

---

## 2. Heavy-Tailed Transforms

Two neuromotor smoothness targets (`kin_jerk_metric` and `kin_dimensionless_jerk`) span 5–8 orders of magnitude due to integration of squared 3rd-order differentials and $T^5$ power terms.
- Transform: $y_{\text{reg}} = \log_{10}(y + 1.0)$
- All regression losses, linear Pearson $r$, and prediction intervals for these two features are computed and reported in the $\log_{10}$ domain, with back-transformed medians for point estimates.

---

## 3. Confidence Tiers & Reporting Standards

Confidence tiers are defined **strictly by the lower bound of the 95% student-bootstrap confidence interval of out-of-fold Spearman $\rho$**:

$$\text{Tier}(\rho_{\text{lower}}) = \begin{cases}
\mathbf{A} & \text{if } \rho_{\text{lower}} \ge 0.60 \quad (\text{High Confidence / Clinical Screening Quality}) \\
\mathbf{B} & \text{if } 0.40 \le \rho_{\text{lower}} < 0.60 \quad (\text{Moderate Confidence / Research Quality}) \\
\mathbf{C} & \text{if } 0.20 \le \rho_{\text{lower}} < 0.40 \quad (\text{Weak Signal / Auxiliary Metric}) \\
\mathbf{D} & \text{if } \rho_{\text{lower}} < 0.20 \quad (\text{Low Confidence / High Uncertainty Estimate})
\end{cases}$$

### Explicit Rules for Tier D:
1. **Never Hidden:** Tier D estimates are always displayed with their full point estimate, wide prediction interval, and percentile rank.
2. **Explicit Uncertainty:** The UI and API contract explicitly flags Tier D with a warning: `"Low-confidence estimate; wide prediction interval."`
3. **Transparent Reporting:** If a feature fails to reach Tier C or B, the failure is reported honestly as a fundamental physical/algorithmic bottleneck.

---

## 4. Evaluation Protocol

### A. Nested Student-Grouped Cross-Validation (All 137 Students)
- The historic 25-student test split was consumed during earlier exploration.
- All evaluation uses **nested student-grouped cross-validation** across all 137 students ($N = 257$ valid sessions):
  - **Outer Loop:** 5 student-grouped folds ($K_{\text{outer}} = 5$). In each outer fold, ~27 students are completely held out as untouched test subjects.
  - **Inner Loop:** 4 student-grouped folds ($K_{\text{inner}} = 4$) inside the remaining ~110 students, used strictly for hyperparameter tuning, model selection, and conformal calibration.
  - **Out-of-Fold Reporting:** Every reported metric ($r, \rho$) comes solely from concatenating untouched outer-fold predictions. In-sample and train-included numbers are strictly forbidden.

### B. Statistical Rigor & Separate Script Evaluation
- **No Script Pooling:** S1 (Hindi, Devanagari script with Shirorekha headline) and S2 (English, Latin cursive/print) have distinct biomechanical rhythms and stroke topologies. They are evaluated and reported **strictly separately**, never pooled.
- **Student-Level Bootstrap CIs:** Standard errors and 95% CIs for Pearson $r$ and Spearman $\rho$ are computed via **2,000 student-clustered bootstrap resamples** (resampling entire students with replacement so paired S1/S2 sessions are not treated as independent).
- **Multiple Testing Correction:** Benjamini-Hochberg (FDR) correction is applied across the 13 feature tests at $\alpha = 0.05$.

### C. Trivial Baseline Benchmark
Every learned model is benchmarked against a **Trivial Metadata Baseline**:
- Trivial inputs: Task (S1/S2), Grade (4-7), School (Future Gen / Kanyashala), Stroke Count (from naive CCs), Total Stroke Length.
- **Gate for Genuine Improvement:** A kinematic feature model is declared an improvement if and only if its out-of-fold Spearman $\rho$ statistically exceeds the trivial baseline's $\rho$, with non-overlapping student-bootstrap 95% CIs ($\Delta \rho > 0, p_{\text{FDR}} < 0.05$).

### D. Prevention of Demographic Leakage
- Image-to-digital mapping models in Step 3 must predict **strictly from image geometry, topological skeleton properties, and optical cross-sections + script indicator (S1/S2)**.
- **Strict Prohibition:** Student Grade, School ID, Section, and Roll Number are **NEVER** input to the image feature estimator.
- Demographic variables are permitted **only** in:
  1. The trivial baseline comparison.
  2. The Step 5 normative reference percentile lookups (comparing a child's digital-equivalent estimate to peers of the same grade and script).

---

## 5. Non-Direct Quantities Declaration

Static 2D images contain no physical time sensor. The following kinematic targets are physically latent:
- `kin_mean_velocity`, `kin_peak_velocity`: Inferred statistically from path length, tortuosity, and curvature distributions under the Two-Thirds Power Law ($v \propto \kappa^{-1/3}$).
- `kin_nvi_rate`: Inferred from spatial inflection density and predicted contact duration.
- `kin_ink_width_ratio_mean/std`: On screenshots, stroke width reflects app software canvas rendering ($r=0.78$ with pressure). On real paper photos, stroke width reflects pen nib geometry and ink bleed ($r \approx 0.002$ with stylus pressure). They are declared as **descriptive optical proxies only**, not direct pressure measurements.

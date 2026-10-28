# InkSight vs Skeleton Trajectory-Recovery Evaluation Summary

**Status**: EVALUATION COMPLETED
**Pre-registration Reference**: `experiments/inksight/CRITERIA.md`
**Test Subjects Evaluated**: 20 (10 dysgraphic, 10 control)

## 1. Executive Verdict

### Final Decision: **NOT USEFUL** (InkSight fails pre-registered criteria; retain Skeleton)

| Gate | Pre-registered Criterion | Result | Status |
|---|---|---|---|
| **P1 — Chamfer Distance** | Paired bootstrap 95% CI of (SK - IS) strictly > 0 | Mean diff: -0.052 [95% CI: -0.072, -0.036] | **FAIL** |
| *P1 Tolerance Gate* | Mean IS Chamfer ≤ Mean SK Chamfer + 0.10 | IS: 0.131 vs SK: 0.079 | **PASS** |
| **P2 — Feature Spearman** | Beats Skeleton on ≥ 2 of 4 geometric features | 0/4 features with CI > 0 | **FAIL** |
| **P3 — Effect Size Retention** | Retains ≥ 80% GT Cohen's d on ≥ 2 of 4 features | 0/4 features with lower bound ≥ 0.80 | **FAIL** |

## 2. Spatial Metrics (Test Set Subject-Level)

- **Mean Chamfer Distance ($H_{med}$ units)**:  
  - InkSight: `0.1313`  
  - Skeleton: `0.0792`  
  - Difference (SK - IS): `-0.0521` (95% CI: `[-0.0721, -0.0360]`)  

- **Mean DTW Distance ($H_{med}$ units)**:  
  - InkSight: `0.2556`  
  - Skeleton: `1.5445`  

## 3. Geometric Feature Fidelity (Spearman $\rho$ to GT)

| Feature | InkSight $\rho$ | Skeleton $\rho$ | $\Delta\rho$ (IS - SK) | 95% Bootstrap CI | InkSight Beats? |
|---|---|---|---|---|---|
| Curvature | -0.177 | -0.114 | -0.063 | [-0.511, +0.269] | **NO** |
| Turn Density | 0.000 | 0.301 | -0.301 | [-0.911, +0.274] | **NO** |
| Tortuosity | 0.298 | 0.332 | -0.035 | [-0.519, +0.367] | **NO** |
| Path Length Ratio | -0.128 | 0.035 | -0.162 | [-0.476, +0.105] | **NO** |

## 4. Effect Size Retention (Dysgraphic vs Control)

| Feature | GT Cohen's $d$ | InkSight $d$ | Skeleton $d$ | IS Retention ($d_{IS}/d_{GT}$) | 95% Bootstrap CI | Retains $\ge 80\%$? |
|---|---|---|---|---|---|---|
| Curvature | +0.401 | -0.450 | +0.059 | -112.4% | [-988.0%, 1028.6%] | **NO** |
| Turn Density | +0.360 | -0.668 | -0.107 | -185.5% | [-946.1%, 1335.7%] | **NO** |
| Tortuosity | +0.341 | -0.394 | -0.059 | -115.7% | [-1307.1%, 1318.2%] | **NO** |
| Path Length Ratio | +0.499 | -0.353 | +0.004 | -70.7% | [-701.5%, 625.9%] | **NO** |

## 5. Structural Metrics

- **Mean Stroke Count Error**: InkSight = `12.93` vs. Skeleton = `10.91`
- **Per-stroke Direction Agreement**: InkSight = `87.9%` vs. Skeleton = `72.2%`
- **Stroke Order Accuracy**: InkSight = `92.1%` vs. Skeleton = `59.0%`

## 6. Constraints Adherence Verification

- [x] Fairness constraint: Both methods ran on identical 224×224 crops (§3)
- [x] Ground truth: Pen-on-paper contact strokes only (§4)
- [x] Coordinate round-trip error: Verified < 0.5px before evaluation (§5)
- [x] Forbidden metrics check: ZERO velocity, jerk, NVI, or Sigma-Lognormal metrics computed (§12)
- [x] Scope boundary: Evaluation strictly limited to Latin-script data; no Devanagari/Telugu deployment claimed (§11)
- [x] Core codebase protection: No modifications to `src/` kinematics pipeline

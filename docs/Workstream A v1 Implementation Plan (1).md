# Workstream A: v1 Implementation Plan

Dysgraphia Detection project. Baseline feature workstream, paper images to sentence-level features to per-language classifier to student-level risk score.

Scope: School A is for development and nested CV. School B is held out as an external test. All numeric parameters below are starting defaults to tune on A only.

## Ground rules

- **School B discipline.** The segmentation pipeline may be fixed using B images, because it is label-free. Feature definitions, model choice, hyperparameters and thresholds are fixed using A only. B labels are touched once, at the final evaluation.
- **Normalization assumption (flag).** School A has about 19 kids per grade, which is thin for robust per-grade statistics. Default: compute the grade x language x task medians/MAD from A and B pooled, using features only and no labels, giving about 39 per grade. The cost is that B's unlabeled features leak slightly into the reference. If B must stay fully pristine, use per-school norms and accept noisier z-scores.
- **Label discipline.** Student labels inherit to every sentence row. Splits are always grouped by student.
- **Prompt dependence.** Copy and dictated sentence prompts are used as a QA check and for completion and per-character features. The pipeline assumes a standardized prompt at deployment.

## Phase 0: Data foundation

1. Build `manifest.csv`: student\_id, school, grade, label, label\_source (paper+stylus or paper-only), image path, and a sentence table with expected text for copy/dictated.
2. Freeze School B images and labels in a separate read-only folder.
3. Generate the nested CV folds on A only: outer = stratified group 5-fold x 5 repeats, inner = stratified group 3-fold. Save to disk and share with other workstreams.
4. Fix the repo layout: `data/`, `pipeline/`, `features/`, `modeling/`, `qa/`, `configs/`, with seeds in config.

**Done when:** the manifest has no unmapped A images, the fold files are committed, and a unit test confirms no student appears across train/test.

## Phase 1: Preprocessing and segmentation

This is the critical path. Build in this order and keep an overlay render at each step.

1. **Page normalization.** Detect the page quadrilateral (Canny + contour, approxPolyDP), warp to a fixed width of about 2000 px, and fall back to manual corners for failures. Flatten illumination by dividing by a large-kernel background estimate, then apply Sauvola binarization (window about 31, k about 0.2) while keeping the grayscale.
2. **Ruled-line detection.** Open horizontally with a long kernel, find rule positions by projection peaks, and fit each as `y = a*x + b` (RANSAC). Record median spacing r, deskew by the median slope, and note any vertical margin line. Keep the rule parameters, since they feed the alignment features.
3. **Rule removal.** Remove rule pixels, then restore strokes that cross the rule with a vertical closing limited to the rule band. Save two layers: `ink_with_rules` and `ink_clean`. Check on crops that strokes crossing rules are not broken.
4. **Lines.** Assign ink components to the ruled band holding most of their mass (descenders will cross bands, so use the centroid).
5. **Script and sentence assignment.** Classify each line's script by the longest horizontal run as a fraction of word width (shirorekha at or above about 0.7 means Devanagari). Group consecutive same-script lines into blocks and check the sequence against the expected Hindi, English, Hindi, English order. Blocks after that are own-writing, with script from the classifier. Whether each sentence starts on a fresh line will be confirmed on the first overlays.
6. **Words.** Devanagari: connected components after a small horizontal closing, since the shirorekha already joins letters. English: cluster the gap lengths per line into intra-word and inter-word with a 2-component GMM, falling back to 0.5h. For copy/dictated, compare detected word count with the prompt and flag mismatches. The flag is for QA, not for tuning features.
7. **Baseline and x-height.** English: take the densest row band, with h as its height and the baseline at its lower edge. Hindi: the shirorekha row is the headline, the core band extends below it, and h is the headline-to-baseline distance.
8. **Skeleton and stroke graph.** Skeletonize (skimage), prune spurs under 0.15h, build the graph (skan), and trace paths between endpoints and junctions. Mask the shirorekha out of the curvature computations, because its straightness has its own features.
9. **Schema.** One JSON per sentence containing lines, words, components, baseline and rule params, skeleton graph and QA flags. OCR and kinematics will reuse this, so version it.

**Gate:** review overlays on about 30 sheets from both schools (ten per grade band where possible), plus word-count match on at least about 90% of copy and dictated sentences. Failures go to a log and are manually corrected where needed.

## Phase 2: Feature library

Each feature is a function `f(sentence_json) -> value | NaN`, with NaN if fewer than 3 words or 2 lines are available for a statistic. Here h is x-height (for Hindi, the core band below the shirorekha) and r is ruled-line spacing.

Key definitions:

- **Baseline residual:** fit `y = a*x + b` per line (RANSAC), e\_i = word-baseline residual, feature = sqrt(mean(e^2)) / h.
- **Rule offset:** o\_i = (y\_i - y\_rule(x\_i)) / r. Report mean (floating above or below the rule) and standard deviation.
- **Slant:** orientation tensor on skeleton tangents within 45 degrees of vertical, using the doubled-angle circular mean. Report the mean and the circular std within the sentence.
- **Curvature:** smooth with Savitzky-Golay (about 0.05h scale), resample by arclength at 0.05h, and compute theta(s) and kappa = d(theta)/ds. Features:
  - mean |d(kappa)/ds| \* h^2 (dimensionless jerk proxy);
  - fraction of tangent-angle variance at wavelengths under 0.5h;
  - kappa sign changes per h of path (small deadband);
  - median tortuosity (path length / chord) per stroke segment.
- **Gaps:** g\_k / h for consecutive words in the same line. Report mean, CV, fraction below 0.3h and above 2h.
- **Size:** word-height CV, h/r, word width per expected character (copy/dictated), component-height CV within words, ascender/descender ratios (English), matra length relative to the core band (Hindi).
- **Hindi-only:** shirorekha RMS deviation / h, breaks per word, tilt variability.
- **Fragmentation and completion:** components and junctions per unit width, words written / expected, lines used.

Left out of v1: stroke width and ink intensity (pencil/pen confound) and corrections.

**Validation of the library (before any modeling):**

- **Synthetic tests.** Render text with controlled perturbations (spacing jitter, size jitter, baseline wobble, tremor noise added to paths). Each feature must respond monotonically (Spearman rho above 0.8).
- **Invariance tests.** Rescale an image by 0.7x and rotate by +/-3 degrees. The features should move within tolerance.
- **Real-data checks.** Features should trend with grade (variability measures generally improve with grade), with missingness per feature logged.
- **Redundancy.** Label-free Spearman clustering, merging groups with |rho| above 0.9 to one representative. No label-based selection outside the CV.
- **Univariate AUCs** for sanity only.

## Phase 3: Dataset assembly

- Robust z-scores: z = (x - median\_g) / (1.4826 \* MAD\_g), per grade x language x task x feature, clipped to +/-5. Minimum cell size 15, otherwise fall back to grade x language with a task offset.
- Two tables, Hindi and English. Row = sentence. Columns: ids, grade, task (one-hot), z-features, label, school.
- Keep the raw feature table too, for later revisits (stroke width, Stage 2).
- Impute NaN inside folds with the training-fold median.

## Phase 4: Modeling and evaluation

**Candidates** (small tuning budget, about 20 random configs each, inner 3-fold):

- Elastic-net logistic regression (C 1e-3 to 10, l1\_ratio 0 to 1)
- SVM with an RBF kernel
- Random forest (depth 3 to 8, min leaf 3 to 10)
- Gradient boosting (small trees, strong regularization, early stopping in the inner loop)

**Training details:**

- Sample weight = 1 / (number of sentences from that student), combined with class weights.
- Platt calibration, fit on inner out-of-fold predictions.
- Student score = mean of calibrated sentence probabilities.
- Operating threshold chosen in the inner loop for a target specificity and applied to the outer test.

**Reporting:**

- AUROC, AUPRC, sensitivity at 90% and 95% specificity, Brier.
- Pooled outer-fold predictions per repeat, averaged over repeats, with a cluster bootstrap over students for confidence intervals. With roughly 24 positives, expect wide intervals.

**Baselines (same folds):**

- Majority class.
- Grade-only logistic model.
- Frozen-embedding baseline: sentence crops (rules removed) through a pretrained CNN/ViT encoder, mean-pooled, PCA to at most 20 dimensions inside the fold, logistic regression.

**Ablations and analysis:**

- Drop-one-group ablation.
- Per-task and per-grade breakdowns (the grade ones are small-n, so caution).
- Permutation importance on grouped outer-test data.
- Stability of selected features across folds.
- A gallery of misclassified students for visual review.

**External test (School B):**

1. Choose the final model and hyperparameters by nested CV on A only.
2. Refit on all of A.
3. Evaluate on B once, with metrics defined in advance.
4. Then report the reverse direction (train B, test A) and a pooled school-grouped CV for v1.1.

## Phase 5: Handoff

- Freeze the feature API and JSON schema, and version the dataset.
- Write up results as agreement with the team-derived labels, with intervals and the circularity caveat.
- Record the revisit list: stroke width/intensity (pencil/pen), handedness, Stage 2 cross-task comparison, own-writing missingness test, extra baselines.

## Dependencies

Phase 1 blocks features, so start it first. While it runs, the modeling scaffold (folds, CV loop, metrics, calibration) can be built and tested on synthetic features, and the encoder baseline needs only crops from Step 4 of Phase 1. The synthetic feature tests can also start before segmentation is final.

## Main risks

- About 24 positives in A limits every estimate, so the feature count and model grids stay small.
- Segmentation quality on messy B scans.
- Label and feature circularity: the labels were partly built from visual qualities the features capture.
- Thin grade cells.

# Multilingual, Stylus-Free Dysgraphia Detection
## Project Scope & Workstream Document — v0.1 (Initial Large Scope)

*This document is a planning baseline, not a locked spec. Expect it to be tuned as workstreams progress and we learn more.*

---

## 1. What we are actually building

Our college gave us a specific base project: **"CNN feature and classifier fusion on novel transformed image dataset for dysgraphia diagnosis in children."** That paper (Kunhoth et al., 2023) takes an existing *online* handwriting dataset — Slovak-speaking kids writing on a Wacom digitizing tablet — renders the pen-trajectory data into clean 400×400 images for four writing tasks (word, pseudoword, difficult word, sentence), and classifies dysgraphic vs. typical handwriting using DenseNet201 transfer learning, ensembles, and feature fusion into SVM/AdaBoost/Random Forest. Their best result: 97.3% accuracy using fused CNN features into an SVM.

**We're treating that paper as one validated component inside a larger system, not the whole project.**

The gap we see: existing dysgraphia detection research assumes access to a digitizing tablet and stylus, and almost all of it is built and tested on English/Latin-script, non-Indian populations. That's a real barrier for the population we actually want to help — children in rural and vernacular-medium Indian schools, where a stylus tablet is not something a school or family has, and where the child may be writing in Hindi, Marathi, Telugu, or another Indian language, not English.

**Our goal:** build a dysgraphia-detection system that works from a **plain photograph or scan of a child's normal schoolwork** — no stylus, no special hardware, no separate "diagnostic session" — and that works for **Indian-language handwriting**, starting with Hindi.

We are not trying to build a clinical diagnostic tool that replaces a doctor. We're building a **screening signal** — something that could plausibly sit inside an existing school workflow (notebook checking, answer sheet grading) and flag children who might benefit from a real evaluation by a teacher, OT, or specialist. This distinction matters for how we describe the project, what claims we make, and how we handle any data suggesting a specific child may have dysgraphia.

---

## 2. Why this matters (the honest version)

Dysgraphia is fundamentally a **motor/neurological transcription disorder** — a disconnect between knowing what to write and being able to physically execute the writing — not a language-specific reading/spelling disorder (that's closer to dyslexia, though the two frequently co-occur). Because it's rooted in motor control rather than language content, there's real reason to believe the core detectable signal — stroke consistency, spacing irregularity, size variance, baseline drift — should generalize across scripts, even though almost no existing research has tested this.

Diagnosis today is manual, subjective, slow, and depends on access to trained specialists (OTs, educational psychologists) that most rural schools simply don't have. An automated screening layer that works with zero extra hardware and in the languages kids actually write in could meaningfully lower the barrier to a child getting help at all — that's the actual stakes here, and why we're choosing to do this properly rather than just chasing an accuracy number.

---

## 3. System architecture — two layers

### Layer 1 — Baseline replication (safe, well-specified, must work)

We rebuild the conceptual approach from the given paper: task-specific feature extraction (from word/pseudoword/sentence-equivalent writing tasks) → CNN feature fusion → classical ML classifier (SVM / AdaBoost / Random Forest), with ensemble voting as a comparison point. This is our proof that the team can execute a known, published pipeline correctly, and it gives us a concrete accuracy benchmark to compare everything else against. We can start this immediately using the paper's own public Slovak dataset, without waiting on any of our own data collection.

### Layer 2 — Our extension (the real contribution)

This is where all our brainstorming lives, organized into distinct, mostly-parallel workstreams (below). The throughline connecting all of them: **can a version of this detection approach survive the move from clean, tablet-derived, single-language images to messy, stylus-free, multilingual, real-world images — and does it still work?**

That question — not a single accuracy number — is the actual research contribution of this project.

---

## 4. Workstreams

### Workstream A — Data collection app & school visit *(critical path)*

**What:** A lightweight Android app for a Samsung S-Pen–enabled device (we have access to an S26 Ultra) that captures raw stylus data — x/y position, pressure, tilt, timestamp, pen-up/pen-down — while a child writes a fixed set of prompts. The prompt set should mirror the *structure* of the base paper's tasks (a simple word, a pseudoword, a difficult word, a sentence) but in Hindi, so we have something structurally comparable to the baseline. The app should also support photographing/scanning the same child's handwriting on paper for the same or equivalent prompts.

**Why this is the most important single piece of data we can collect:** it gives us *paired* online (stylus) and offline (photo/scan) samples from the **same child, writing in the same language, at the same time**. Nothing else in our plan gives us that pairing, and it's the only way we can properly test whether stylus-derived signal can be estimated from an image alone.

**Status:** we don't currently have a doctor, OT, or psychologist who can formally evaluate the children we sample. Even if we did, the resulting labeled set would still be small. We are treating this openly as a **weak-label / exploratory dataset**, not a clinically validated one — see Section 5 for how we're handling this honestly.

**Dependency:** this app must be built and tested *before* the school visit — it cannot be built during the visit. The visit itself is likely a single day, so everything about the app, the prompt set, and the consent process needs to be locked in advance.

**Consent note:** because this data collection could informally surface signs that a specific child may have a learning difficulty, we need to talk to the school about parental/school consent *before* the visit, even though we're not sharing individual results with families. This is a different conversation than "we're collecting handwriting samples for a college project" and shouldn't be an afterthought.

### Workstream B — Baseline replication

Rebuild the paper's DenseNet201 fine-tuning + feature fusion + SVM/AdaBoost/RF pipeline. Runs on the public Slovak dataset first, independent of our own data collection. This is our benchmark and our proof-of-execution.

### Workstream C — Motor/geometric feature extraction

Research and build the language-agnostic, stylus-free feature set: stroke-width variance, letter/akshara size covariance, spacing entropy, baseline drift, pressure/speed *proxies* estimated from static ink geometry (not real sensor data — we should be precise about this distinction in every write-up). This workstream doesn't need labeled data to build, only to eventually validate — so it can start immediately and run in parallel with everything else.

**Indic-script note:** Devanagari (Hindi, Marathi) and Telugu are abugidas, not alphabets — a single visual unit (akshara/grapheme cluster) can combine a consonant, vowel matras, and conjuncts. Our segmentation and consistency metrics need to operate at the akshara/grapheme-cluster level, not the individual-character level the way Latin-script approaches do. This is real, non-trivial engineering work, not a config change — treat it as its own research task.

### Workstream D — Physics/kinematic reconstruction ("can we estimate stylus data from an image alone?")

The core hypothesis: can we predict online kinematic features (velocity, pressure proxy, stroke order) from a static handwriting image alone, using a model trained on paired online/offline data? This is explicitly its own testable research question, validated in two stages:

1. **General feasibility** — using existing public online handwriting datasets (flattened into static images), test whether a model can reconstruct kinematic features from the image with reasonable fidelity. This can start now, independent of the school visit.
2. **Task-relevant validation** — using our own paired online/offline data from the school visit (Workstream A), test not just reconstruction accuracy, but whether the *reconstructed* kinematic signal still separates typical from atypical writers as well as the *real* kinematic signal does. This is the more important question, and it's possible imperfect reconstruction still preserves the diagnostically useful part of the signal (e.g., relative pressure variance rather than absolute pressure).

We should be precise in language here: we are estimating physics-informed proxy features from ink-trace geometry, not literally simulating physics or recovering true kinematic data.

### Workstream E — Model/ensemble strategy

We are deliberately **not** locking in which ML/DL approach(es) we'll use yet. This should follow from in-depth research and from what Workstreams B, C, and D actually produce as feature sets — not be decided upfront. Ensemble voting across multiple techniques (as in the base paper) is a reasonable starting point/default, but the real decision — and any weighting between sectors (motor features, grammar/OCR signal, physics-proxy signal) — should be **learned from labeled data** (e.g., a small logistic regression or gradient-boosted model calibrated on our dataset) rather than manually assigned, since hand-picked weights on a claim this sensitive won't hold up to scrutiny.

### Workstream F — Multilingual scope

We're starting with **Hindi only** as the primary language target — highest impact given reach, and the language we have direct access to test in and volunteer connections around. Marathi and Telugu (and broader multilingual generalization) are explicit **stretch goals**, sequenced after Hindi is working, not parallel commitments from day one. The motor/geometric sector (Workstream C) is designed to be script-agnostic by construction, so extending to additional languages later should mainly mean adapting the akshara-segmentation layer and collecting new samples — not redesigning the core pipeline.

### Workstream G — Weak-label generation from the school visit ("jugaad" labeling)

Since we have no clinician access, we need another way to generate any labels at all from the school visit data. Two sources, used together rather than either alone:

1. A **statistical deviation score** — not a "dysgraphia label" — computed either from our own cohort's distribution (an anomaly/outlier score relative to grade-level norms among the kids we sample) or, as an exploratory secondary signal, from features extracted via a model trained on the Slovak dataset. We should be explicit and careful never to call this a diagnosis; it's a deviation signal only, and the Slovak-model-based version in particular carries a real domain-shift risk (different population, language, script, and tasks) that we should name openly rather than gloss over.
2. **Informal teacher input.** Teachers who've had a child in class for a year already notice writing struggles long before any formal diagnosis. Cross-referencing teacher-flagged children against our statistical outliers is a more credible weak-label source than either alone, and costs nothing extra during the school visit.

This entire dataset should be treated and described as **weakly-labeled / exploratory**, never as clinically validated ground truth, in any report, defense, or write-up.

---

## 5. What we're being careful about (worth the whole team internalizing)

- **We are building a screening aid, not a diagnostic tool.** No output from this system should be described or treated as a diagnosis. A flagged result should point toward "worth a closer look from a teacher/OT," not toward telling a child or parent they have dysgraphia.
- **Grammar/spelling issues and dysgraphia are not the same thing.** Dysgraphia is a motor/transcription disorder; persistent grammar/spelling problems that show up in typed or spoken output too point toward a different (often co-occurring) language-based disorder. If we build an OCR/grammar sector, it's a secondary co-indicator, not a primary dysgraphia signal, and should never be hand-assigned a large weight.
- **Our weak labels are weak.** Every report and every model evaluation needs to say clearly where labels came from and how uncertain they are. This protects the credibility of everything downstream far more than it costs us in polish.
- **Age/grade baseline matters.** A 7-year-old's normal handwriting looks nothing like a 12-year-old's, independent of dysgraphia. All our features should be evaluated relative to grade-level norms, not one global threshold.
- **Consent and framing at the school need to be sorted before the visit**, not during it.

---

## 6. Sequencing & rough timeline

This is a starting shape, not a locked schedule — expect it to shift once Workstream A's timeline (school availability) is confirmed.

| Phase | Focus | Key workstreams |
|---|---|---|
| **Phase 0 (now)** | Research, app build, baseline setup | A (app build), B (start on public data), C (research), D (start on public data), F (Hindi task-set design) |
| **Phase 1** | School visit + immediate baseline results | A (visit execution), B (baseline results on public data as comparison point) |
| **Phase 2** | Feature validation on real data | C (validate motor features on collected data), D (stage 2 validation with paired data), G (weak-label generation) |
| **Phase 3** | Model/ensemble convergence | E (converge on approach using real feature sets from C/D), integration of sectors into a combined pipeline |
| **Phase 4** | Consolidation & reporting | Full pipeline evaluation, honest reporting of what's validated vs. exploratory, stretch-goal scoping (Marathi/Telugu, if time allows) |

Workstreams A, B, C, D, and F's Hindi task design can genuinely run in parallel across team members starting now. The one hard dependency: **A's app must be functional before the school visit is scheduled to happen** — that's the one deadline that can't slip without cascading into everything else.

---

## 7. What we're explicitly deferring (not ignoring — sequencing)

- Full multilingual generalization beyond Hindi (Marathi, Telugu, others) — stretch goal after Hindi works.
- OCR/grammar sector — worth prototyping once AI4Bharat or similar Indic NLP tooling is evaluated, but secondary to the motor-feature sector.
- Locking in a final model/ensemble architecture — deliberately left open pending Workstreams B–D's outputs.
- Any claim of clinical validity — we are not positioning this as a diagnostic replacement at any stage of this project.

---

*End of v0.1. Next update should happen once the school visit date is confirmed and the app's task-set (Hindi word/pseudoword/sentence prompts) is finalized.*

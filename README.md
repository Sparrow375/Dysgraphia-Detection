# Dysgraphia Detection — Multilingual, Stylus-Free

Automated dysgraphia screening from ordinary handwriting images — no tablet or stylus required, built to work across Indian languages, starting with Hindi.

## Why

Dysgraphia diagnosis today is manual, subjective, and depends on access to specialists most rural schools don't have. Existing handwriting-based detection research assumes a digitizing tablet and is built almost entirely for English/Latin-script populations. We're building a screening layer that runs on a plain photo or scan of a child's normal schoolwork, in the language they actually write in.

This is a **screening aid, not a diagnostic tool**. Nothing this system outputs should be treated as a diagnosis — the goal is to flag children who may benefit from a real evaluation by a teacher or specialist.

## Status

🚧 Early scope & planning stage. See [`docs/project-scope.md`](./docs/project-scope.md) for the full v0.1 project scope, workstream breakdown, and timeline.

## Base reference

Our college-assigned baseline: Kunhoth et al., CNN feature and classifier fusion (DenseNet201 + SVM/AdaBoost/Random Forest) on a Slovak online-handwriting-derived image dataset. We're treating it as one validated component inside a larger system — see the scope doc for how our work extends beyond it.

## Workstreams

| Code | Workstream |
|---|---|
| A | Data collection app (S-Pen capture) & school visit |
| B | Baseline replication (DenseNet201 + feature fusion) |
| C | Motor/geometric feature extraction (script-agnostic) |
| D | Physics/kinematic reconstruction from static images |
| E | Model/ensemble strategy |
| F | Multilingual scope (Hindi first) |
| G | Weak-label generation from school visit data |

Full detail on each in the project scope doc.

## Repo structure

```
.
├── docs/           # planning docs, scope, research notes
├── app/            # S-Pen data collection app (Workstream A)
├── baseline/       # baseline replication pipeline (Workstream B)
├── features/       # motor/geometric feature extraction (Workstream C)
├── kinematics/     # physics/kinematic reconstruction (Workstream D)
├── models/         # ensemble & classifier experiments (Workstream E)
└── data/           # (gitignored) local datasets — never commit raw data
```

## Getting started

_Setup instructions to be added as the pipeline takes shape._

## A note on data

Any handwriting samples we collect from children are treated as sensitive. No raw data goes into this repo — see `docs/project-scope.md` §5 for how we're handling consent, labeling, and framing.

## Team

Avaneesh Devendra Verma (Team-Leader),
Sriram Gudlawar,
Embari Nitish Kumar,
Vaddem Srujani,
Kolloju Spoorthi

---

Here's to building something that actually reaches the kids it's for.

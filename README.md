# Crowd Intelligence — AI-Based Crowd Risk Estimation

BTech final-year project (IIT Patna, Electrical and Electronics Engineering) —
Author: Sagar Kumar. Supervisor: Prof. Mahesh Kumar Kolekar.

Vision-based estimation of physically-grounded crowd risk indicators —
density × motion-variance crowd pressure, per Helbing, Johansson &
Al-Abideen (2007) and independently confirmed by Gu et al. (Nature, 2025)
— from ordinary CCTV/handheld video, without requiring a labelled stampede
dataset. See `docs/ARCHITECTURE.md` for the full design and
`docs/ROADMAP.md` for the phase-by-phase plan.

## Start here

- **`PROJECT_STATE.md`** — where the project is *right now*. Read this first.
- **`docs/ARCHITECTURE.md`** — the system design and why each module exists.
- **`docs/ROADMAP.md`** — the phase-by-phase plan with experiment IDs.
- **`KAGGLE_PROTOCOL.md`** — how to run this on Kaggle with no personal GPU,
  without re-running everything each session. Read before opening any
  notebook.
- **`DAILY_LOG.md`** — per-day work trace. Check the last entry's "Next"
  line when returning after a gap.
- **`DECISION_LOG.md`** — why each non-obvious choice was made.
- **`EXPERIMENTS.md`** — flat index of experiment IDs, cross-referenced to
  GitHub Issues (use `.github/ISSUE_TEMPLATE/experiment.md` per experiment).

## Repo layout

```
data/            raw + processed data (git-ignored; lives in Kaggle Datasets)
notebooks/       Kaggle notebooks — compute only, no real logic (see KAGGLE_PROTOCOL.md)
src/
  data/          dataset loading, annotation tooling
  models/        density branch, motion branch, temporal model
  indicators/    the 7 per-cell risk indicators (density, speed, variance, pressure, entropy, divergence, growth rate)
  evaluation/    metrics, evaluation protocol
  visualisation/ plots, overlays
  utils/         kaggle_cache.py — cache_or_compute + checkpoint/resume helpers
configs/         experiment configs
checkpoints/     model checkpoints (git-ignored; lives in Kaggle Datasets)
outputs/         experiment results, per-experiment (git-ignored)
docs/            architecture, roadmap, paper survey, research map
.github/         Issue template for task-wise experiment tracking
```

## Working on Kaggle

Read `KAGGLE_PROTOCOL.md` in full before your first session. Short version:
clone this repo as the first cell of every notebook, write real code only
into `src/` and commit it, wrap every expensive computation in
`cache_or_compute` from `src/utils/kaggle_cache.py`, checkpoint every
training epoch, and publish anything expensive as a versioned Kaggle
Dataset at the end of a session.

# Decision Log

Append-only. One entry per non-obvious architectural or methodological
choice, with the reasoning, so you (and your supervisor) never have to
reconstruct "why did I do it this way" from memory. Never delete an entry —
if a decision is later reversed, add a new entry that supersedes it and
say why.

## D001 — Density-map regression, not detection, as the backbone
Detection (YOLOv8+SAHI) demoted to a baseline/cross-check module. Own
video showed count instability under occlusion at higher densities;
density-map regression is the established SOTA approach for dense-crowd
counting for exactly this reason (see `docs/ARCHITECTURE.md` §2,
`docs/PAPERS.md`).

## D002 — Farnebäck dense optical flow, not RAFT/learned flow, for motion
Farnebäck is classical, fast enough for the frame rates involved, needs no
training data, and its output granularity (per-pixel flow field) is what
the per-cell motion indicators (speed, speed variance, divergence) need.
Learned flow is a possible later upgrade, not a Phase-0/1 requirement.

## D003 — Crowd pressure P(z,t) = ρ(z,t)·Var(v(z,t)) as the core risk
indicator
Physically grounded (Helbing, Johansson & Al-Abideen 2007), independently
re-confirmed in dense-crowd turbulence contexts (Gu et al., Nature 2025).
Chosen over ad-hoc/learned risk scores specifically because it doesn't
require a labelled stampede dataset to define — see the Q1/Q2/Q3
validation chain in `docs/ARCHITECTURE.md`.

## D004 — GNNs excluded from scope
No graph-structured relational reasoning need identified yet that the
two-branch density+motion pipeline doesn't already cover; would add
complexity without a motivating failure. Revisit only if a specific
observed failure mode points at it.

## D005 — XGBoost before ConvLSTM for the temporal model
Start with the simpler, more interpretable model on the 7 per-cell
indicators; only justify the added complexity of ConvLSTM with a
documented experiment showing XGBoost's failure mode (see
`docs/ARCHITECTURE.md` §Training data strategy, `EXPERIMENTS.md` E0XX).

## D006 — GitHub clone-in-notebook-header + Kaggle Datasets for compute
continuity
See `KAGGLE_PROTOCOL.md`. Solves both "no persistent GPU machine" and
"per-day task tracking" with one mechanism: code lives in git (commit
history = task log), expensive artifacts live in versioned Kaggle
Datasets, notebooks are disposable compute only. (2026-09-24)

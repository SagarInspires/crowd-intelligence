# Project State (single source of truth for "where are we right now")

Update this file whenever a phase completes or a major decision is made.
Unlike `DAILY_LOG.md` (append-only trace), this file is *overwritten* to
always reflect the current snapshot — read this first to get oriented,
read `DAILY_LOG.md` for how you got here.

## Current phase
Phase 0 — Foundation (repo setup, ground-truth annotation, camera config,
density-map sanity check). See `docs/ROADMAP.md`.

## What's proven so far
- Baseline YOLOv8+SAHI detection count on your own video: mean ≈ 71.57
  (noisy under occlusion — this is the motivating failure for the
  density-regression branch, not a number to optimize further).

## What's NOT done yet
- Manual head-point annotation of your own video (`data/your_video/annotations/`)
- Density-map sanity check (Gaussian-kernel ground truth vs YOLO count)
- `camera_config.json` for your video
- Everything past Phase 0

## Active blockers
- None currently open. (Kaggle/GPU + tracking blockers resolved by this
  repo scaffold + `KAGGLE_PROTOCOL.md` — 2026-09-24.)

## Immediate next action
Day 1 tasks per `docs/ROADMAP.md` Phase 0: annotate head points on a
sample of frames from your own video, build the density map, compare
implied count to YOLO's 71.57.

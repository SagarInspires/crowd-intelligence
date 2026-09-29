# Experiments

One row per experiment ID (see `docs/ROADMAP.md` for the E001–E013 plan and
`.github/ISSUE_TEMPLATE/experiment.md` for the per-experiment write-up
template used in GitHub Issues). This file is the flat index; the Issue
(and `outputs/experiments/EXXX/`) holds the full detail.

| ID | Question | Status | Result (one line) | Issue |
|----|----------|--------|--------------------|-------|
| E001 | Does manual head-point density match/exceed YOLO's count reliability on own video? | not started | — | — |

Add a row per experiment as you start it — don't backfill in bulk, that's
how this file stops being trustworthy.

## E000: Ground truth, 15 frames (2026-09-29)
- Video: 14931663_1080_1920_60fps.mp4, portrait, moving camera. Frames every 93 (0..1302), 1080x1920.
- Annotation: VIA, head points + 1 ignore polygon per frame (frame 1023 has 3). 6,828 raw annotations, 12 near-duplicates (<6 px) merged -> 6,816.
- True count per frame: mean 454.4, std 34.1, min 401, max 535. Ignore area 4.9-9.8% of frame.
- Density maps: fixed (sigma=15) and geometry-adaptive (beta=0.3, k=3, sigma clipped 2-25 px), stride-8 sum-pooled targets, mask stored separately.
- Checks passed: mass = N (tol 1e-3) for all frames, pooling mass-preserving, no head inside an ignore polygon, 2 border heads clipped (frames 93, 1023).
- Observation: fixed sigma=15 merges heads in dense zones; adaptive resolves them. Adaptive will be the main GT.
- Open: annotation noise floor not yet measured (E001).

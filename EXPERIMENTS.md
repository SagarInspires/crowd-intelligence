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

   ## E001: Off-the-shelf person detectors vs clicked heads (2026-09-29)
   - Frames: 15 (5 tuning, 10 reporting), 1080x1920, imgsz=1920, ultralytics 8.4.165, ignore regions excluded.
   - Cutoff chosen on tuning frames (grid 0.001-0.5, F1 and count-error rules agree).
   - YOLOv8x (cutoff 0.005): MAE 18.7 (95% CI 11-28), bias -15, recall 66%, precision 68%, far-half recall 36%, near 79%.
   - YOLO26x (cutoff 0.02): MAE 44.3 (26-62), bias -43, recall 68%, precision 75%, far-half recall 42%, near 78%.
   - At default confidence 0.25, YOLOv8x finds ~14 and YOLO26x ~51 boxes per frame (true ~454).
   - Ignore regions hold only ~1-4% of raw boxes; the surplus boxes are not explained by them.
   - Reading: totals look good only because misses and false boxes cancel (recall/precision ~66-75%); the count swings hugely with the cutoff (YOLOv8x bias -149 at 0.01 vs -15 at 0.005). A detector needs per-scene tuning with labels.
   - Caveats: 10 reporting frames, one video, precision is a lower bound (true count is a lower bound), cutoff tuned on 5 labelled frames.

## E002 — Slicing (SAHI-style tiling) vs full-frame detection
**Date:** 2026-09-29 | **Notebook:** notebooks/03_sahi_E002.ipynb | **Data:** 15 annotated frames (5 tuning, 10 test)

**Question:** Does tiling with magnification find more of the distant heads that a full-frame detector at native 1920 misses?

**Setup:** YOLOv8x and YOLO26x; own tiler (ownership by head point, no NMS, zero duplicates verified);
settings: full frame @1920 (baseline), 640-tiles @640 (1x control), 640-tiles @1280 (2x), 320-tiles @1280 (4x).
Cutoff tuned on the 5 tuning frames (best F1), reported on the 10 test frames. Hungarian one-to-one matching.
Statistics: cluster bootstrap over frames, exact sign-flip test, Holm correction on the one confirmatory
comparison (2x vs baseline, recall). Everything else exploratory.

**Results (test frames, YOLO26x 2x vs baseline):**
- Own tuned cutoffs: recall +7.9 pp (95% CI +6.6 to +9.1), far recall +16.0 pp; better in 10/10 frames;
  precision fell 0.76 -> 0.69.
- Matched precision (0.759 vs 0.760): recall +1.4 pp (CI -0.05 to +2.9, Holm p=0.12, not shown);
  far recall +7.2 pp (CI +3.1 to +11.9, exploratory).
- YOLOv8x: no gain; 2x loses 3.2 pp recall (CI -4.9 to -1.1), worse in 7/10 frames.
- 1x control: no gain for either model, so the effect comes from magnification, not from tiling itself.
- 4x worse than 2x. Runtime cost: about 6x (2x) and 27x (4x) per frame.

**Conclusion:** With 2x magnification YOLO26x recovers distant heads that full frame misses (about +7 pp far
recall at equal precision). Most of the overall recall gain is a looser threshold, not better detection.
YOLOv8x gets nothing. Use tiled YOLO26x for person masks (registration, E003); use YOLOv8x full frame for counts.

**Caveats:** 10 test frames from one ~21 s clip with the same people in every frame; cutoff uncertainty is not
in the intervals; matched-precision rows for 1x/4x were not close to matched (coarse cutoff grid) and are not interpreted.

**Correction to E001:** the YOLO26x row in E001 used an explicit iou=0.7, which changed its results.
The E002 baseline row is authoritative (recall 71.08, far recall 47.38, MAE 31.3).

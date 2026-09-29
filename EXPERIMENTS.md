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
- Open: annotation noise floor not yet measured (planned: re-annotate frames 465, 279, 0 blind).

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
YOLOv8x gets nothing. Use tiled YOLO26x for person masks (registration, E003); MAE from box counts is driven by cutoff choice; compare detectors by recall at equal precision (see E003).


**Caveats:** 10 test frames from one ~21 s clip with the same people in every frame; cutoff uncertainty is not
in the intervals; matched-precision rows for 1x/4x were not close to matched (coarse cutoff grid) and are not interpreted.

**Correction to E001:** the YOLO26x row in E001 used an explicit iou=0.7, which changed its results.
The E002 baseline row is authoritative (recall 71.08, far recall 47.38, MAE 31.3).


## E003 — Full-video detection, count stability, and detector comparison at matched precision
**Date:** 2026-09-29 | **Notebook:** notebooks/04_detection_fullvideo_E003.ipynb
**Data:** 14931663_1080_1920_60fps.mp4 (1303 frames, 1080x1920, 59.94 fps); 15 labelled frames (5 tuning, 10 test)

**Question:** Which detector is best for (a) crowd masks and (b) a detection baseline, and how steady is each one's count over time?

**Setup:** four detectors on all 1303 frames at confidence floor 0.001: YOLOv8x full frame @1920; YOLO26x full frame with nms=True; YOLO26x full frame with nms=False; YOLO26x 2x tiles (640-px tiles @1280, overlap 0.2, head-point ownership). Speeds on a T4: 0.39 / 0.39 / 0.39 / 2.36 s per frame.
Video check: 1303 frames decoded = reported; annotated photos match video frames (mean abs diff 1.4-1.5, JPEG noise).

**Results:**
1. **Count stability** (at equal mean count ~450 boxes/frame): noise around a 1-s median is 5.0% for YOLOv8x and 2.4 / 2.4 / 2.3% for YOLO26x nms=True / nms=False / tiles. Frame pairs with count change >= 10: 42 / 25 / 26 / 22%. Steadiness comes from the model, not from tiling or the NMS mode. Every detector still changes by >= 5 in about half of frame pairs, so a 1-s median filter is needed. Both detectors show a ~15% count drop at 3-5 s (checked in E003b: YOLO26x count dips 5-8% during the camera tilt at 3-7 s, but a similar dip occurs at 18-21 s with no camera motion; tilt not established as the cause).
2. **Recall at equal precision** (0.65 / 0.70 / 0.76; far half in brackets):
   YOLOv8x 0.68 / 0.62 / 0.53 (0.42 / 0.34 / 0.24);
   YOLO26x nms=True 0.79 / 0.76 / 0.70 (0.61 / 0.55 / 0.47);
   YOLO26x nms=False 0.78 / 0.72 / 0.65 (0.60 / 0.51 / 0.39);
   YOLO26x tiles 0.81 / 0.78 / 0.70 (0.65 / 0.61 / 0.51).
3. **MAE from box counts is a cutoff artifact:** n_det ~ n_true x recall / precision. YOLOv8x's F1 cutoff has precision ~ recall (0.67 / 0.67), so its false alarms cancel its misses. The count bias changes sign with the cutoff (about +85 to +103 at precision 0.65, -43 to -143 at 0.76).
4. The far half (y < 960) holds 30% of the labelled test heads (1391 of 4582).

**Conclusions:** YOLO26x replaces YOLOv8x as the detection baseline (higher recall at equal precision, half the flicker, same speed). On this clip NMS-free gave no recall or stability advantage over NMS. Tiling adds about 4-5 pp far-half recall at equal precision for 6x the compute. The crowd-mask source (tiles vs full-frame nms=True) was decided in E003b: full-frame nms=True mask (0.003, 20 px); the tiled mask differs by at most 3.4 px.

**Caveats:** 10 labelled test frames from one clip; no confidence intervals in the matched-precision comparison; cutoffs tuned on 5 frames; absolute numbers differ slightly from E002 because E002 used JPEG stills and E003 uses decoded video frames.

**Correction to E002 (applied 2026-09-30):** replaced the sentence "use YOLOv8x full frame for counts" with "MAE from box counts is driven by cutoff choice; compare detectors by recall at equal precision (see E003)".

## E003b — Camera registration for a handheld video (notebook 05)

**Question.** Can we remove the camera motion from this handheld clip well enough for optical flow and crowd pressure, using only the background?

**Setup.** Crowd masked out with YOLO26x full-frame detections (nms=True, cutoff 0.003, boxes grown 20 px); head coverage 0.994 on labelled frames. Frame-to-frame motion at half resolution, chained to frame 0. Drift check: every 93rd frame registered directly to the previous checkpoint with ORB (3 px threshold).

**Results.**
- Tracker: ORB features under-read slow motion (whole-pixel positions): in a synthetic test with known motion, ORB chained 36 px of a true 63 px. Lucas–Kanade tracking (sub-pixel) gave 2–3 px error vs 18–30 px for ORB, and is faster (0.04 s/frame). → LK.
- The camera tilts up between ~3 s and ~7 s: ~230 px at the centre, ~305 px at the top; nearly still otherwise.
- Motion model: homography required. Similarity differs from homography by up to 35 / 16 / 49 px (top / centre / bottom).
- Robustness (max difference to the default over all frames): inlier threshold 1 vs 3 px ≤ 0.11 px; tiled mask vs full-frame mask ≤ 0.2 / 1.1 / 3.4 px. → cheaper full-frame mask.
- 0 failed frame pairs; ≥ 990 inliers per pair; largest one-frame jump 3 px.
- Drift check: no missing checkpoints; chained vs direct agree within 0.5–3.4 px outside the tilt. During the tilt the bottom of the frame disagrees by 5.4 / 11.9 / 8.0 px (frames 186 / 279 / 372); top and centre ≤ 4.6 px. Features come mostly from the upper background, so the bottom is extrapolated. Which of the two estimates is closer to the truth is not known.
- Apparent size after the tilt vs frame 0: top 0.88, bottom 1.08. A single fixed pixels-per-metre scale for the whole video would be wrong by up to 12%.
- Count vs tilt: count dips ~5–8% during the tilt (4–6 s) and recovers; a similar dip occurs at 18–21 s with no camera motion. Tilt not established as the cause.

**Decision.** Default registration: LK + homography, full-frame mask (0.003, 20 px), 1 px threshold. Saved as `registration_G.npy` (frame t → frame 0, full resolution).

**Consequences for Phase 3.**
- Crowd pressure uses the velocity variance over time at a fixed ground location (Helbing 2007). Registration error that changes over time adds to that variance, so the 5–12 px bottom-of-frame uncertainty during the tilt must be counted in the noise term (C1) or the tilt interval flagged as lower trust.
- One calibration of frame 0 (known object size or reference points) carries to every frame through G.
- The view is oblique and tilting, so the scale differs between directions: this clip is a test case for C5's error analysis, not for the simple scale-free argument.

**Files.** `E003b_mask_candidates.csv`, `E003b_mask_overlay.png`, `E003b_smoke_test.csv`, `E003b_rows_[A-D].csv`, `E003b_drift_check.csv`, `E003b_camera_vs_counts.csv/.png`, `registration_G.npy`.

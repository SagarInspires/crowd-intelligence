# E006 closeout (2 Oct 2026)

Status: E006 (notebook 08_flow_noise_E006) is CLOSED after Cell 25. Full detail is in `docs/E006_RESULTS.md`; its "Next" section is superseded by this file. Outputs: `/kaggle/working/E006_outputs.zip` (26 MB; CSVs, `fig1_farneback_gaps.png`, `fig2_degradation.png`, `e006_headline.txt`). Cell 25 tables match the numbers in E006_RESULTS findings 19-22.

## What goes into the report (with its limits)
1. Window dependence: true person-velocity variance falls with the averaging window (IM01 0.683 -> 0.448 from 1 to 16 frames), so any Var(v) or P value must state its window. (Findings 8.)
2. Person-level sampling beats pixel-level sampling: pixel proxy -23% to -49% at 240 px patches vs person estimator -1.6% to -12.8%. Mechanism (limb smoothing) is a hypothesis. (Finding 7.)
3. A short-gap (1-4 frames, 0.04-0.16 s) direct Farnebäck estimator read at TRUE person positions is within +-10% raw in four independent crowds (worst 9.6 / 8.8 / 6.4% at N = 1 / 2 / 4), needs no calibration gain, and is unchanged under Gaussian noise (std 4 and 8 of 255) and JPEG quality 30 (worst 9.1%). Confirmed on IM01+hDyn and IM03, which were new for Farnebäck; IM02 and IM05 were seen before. (Findings 20-21.)
4. SEA-RAFT, DIS and RAFT-large shrink the spread more than Farnebäck on every crowd tested; SEA-RAFT worsens under JPEG/noise in IM05 and IM01 and fails the +-10% check even on clean frames (IM03 -12.7% at N = 2). (Findings 19, 21.)
5. Negative results to state plainly: the original C1 (subtract a learned noise variance) fails even with the true error variance; a single gain does not transfer across crowds; start-point errors break long-window estimators; no estimator passes on the slow crowd IM04 (Farnebäck -25% to -31%, SEA-RAFT -40%, DIS -60%, RAFT-large -63% at one frame); local texture and contrast do not explain it (texture gap -0.097 vs declared +0.10). (Findings 3, 4, 12-18, 22.)
6. Lead only (not a finding): in IM04 people with neighbours within 20 px recover 0.66 of their motion vs 0.89 for the least crowded third (gap -0.231 vs -0.05 to +0.01 elsewhere); post-hoc, one crowd, 15 gaps looked at.

## Limits to state with every claim
- CrowdFlow is synthetic (clean rendered frames); five independent crowds from one engine; no real-video ground truth.
- All headline results use TRUE start positions; detector-like start errors broke the 0.64 s and 1.5 s estimators (Cells 14-17, SEA-RAFT only); Farnebäck from realistic starts is untested.
- Gates (+-10%, +-15%), displacement rule, 0.85 recovery and 0.10 texture gap are my own proposals. Many variants were scored (~48 in Cells 14-17, 6 in Cell 22, 8 in Cell 23, 15 in Cell 24); only pre-declared checks on crowds not used for choosing count.
- Degradations in Cell 23 are mild, synthetic and spatially independent.

## Next notebook (proposal, to confirm with the user)
- Start points from a real source: CSRNet density-map peaks and/or a head detector, with the Farnebäck short-gap estimator; compare against the true-start result; pre-declare the gate.
- Known-warp transfer-function test on real frames (does the estimator recover a known synthetic velocity field on real video frames).
- Real-video check with some independent reference (tracks from a detector+tracker, or a dataset with trajectories).
- Test the crowding lead on new crowds (declared beforehand) rather than IM04 again.
- Literature check and novelty check for the findings above (not done).

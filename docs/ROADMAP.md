# ROADMAP.md
## 6-Month Execution Plan — Crowd Intelligence & Risk Indicator Estimation

**Read this alongside ARCHITECTURE.md.** The roadmap says *when* and *in what order*. The architecture says *why*. They are separate documents because the order of experiments will change; the architectural reasons should not.

**Compressed-work assumption:** 5 normal workdays ≈ 1 actual workday. The 6-month calendar plan below should be read as approximately 6 "calendar months" at this pace. Adjust the phase boundaries if you know your actual availability.

**Rule for every phase:** each phase must produce a saved, reproducible artifact and a written entry in EXPERIMENTS.md before the next phase begins. Never start a new phase on a foundation that cannot be reloaded.

---

## Phase 0 — Foundation and orientation (Week 1 of Month 1)

**Goal:** the project exists as a real thing with ground truth, not just running YOLO on a video.

### 0.1 Repository setup (half a day)

Create the repo structure from ARCHITECTURE.md Section 6. Initialise with a README describing the research question. Add `.gitignore` for large files. Create the five Markdown documents (ARCHITECTURE, ROADMAP, RESEARCH_MAP, DECISION_LOG, EXPERIMENTS, PROJECT_STATE). They can be stubs — the important thing is that they exist and that you commit them.

Commit message: `init: project structure, research question, decision log seeds`

### 0.2 Ground truth annotation (one full day — non-negotiable)

This is the most important single task in the project. Everything else depends on it.

- Extract every 50th frame from your existing 1303-frame test video (~26 frames).
- Open each in an image viewer. Using a script or tool (LabelMe, or even matplotlib ginput), click on every visible head and record (x,y,frame_id) to a CSV.
- Do this for at least 30 frames. More is better. Do not sample only frames with few people.
- Save to `data/your_video/annotations/head_points.csv`.
- From these annotations, compute: true count per annotated frame, density range, mean count, std. These are your ground-truth numbers. The YOLO count you computed earlier was not ground truth.

**Expected output:** `head_points.csv` with columns [frame_id, x, y, annotator]. True mean count and range.

**Why before anything else:** you cannot evaluate MAE, AP, or any comparison between detector and density model without ground truth. Running experiments without this is measuring noise against noise.

### 0.3 Camera configuration (two hours)

Find two or more objects of known size or known distance in your test video (door frame, lane marking, car, person height). Estimate the ground-plane to image-plane homography, even approximately. If truly impossible, document why in DECISION_LOG entry D006. This determines whether any metric-density claim is possible.

**Output:** `data/your_video/camera_config.json` with best available homography estimate and stated uncertainty.

### 0.4 Concept verification (two hours)

Before training anything, verify you understand the density map:
- Take 5 annotated frames.
- Generate density maps using `make_density_map()` from `src/data/density_map.py`.
- Confirm ∑D̂ ≈ true count for each frame.
- Visualise side-by-side: frame, head points, density map.

Save the notebook as `notebooks/00_density_map_sanity.ipynb`. Commit.

**Phase 0 deliverable:** annotated ground truth, camera config, density map sanity check. One commit, one EXPERIMENTS.md entry.

---

## Phase 1 — Detection baseline + SAHI (Days 2–4 of Month 1)

**Goal:** reproduce the YOLO baseline properly, then apply SAHI, and produce the first real quantitative result of the project: does sliced inference recover small-person recall?

### 1.1 Re-run YOLO baseline with ground truth

Re-run YOLOv8-x at imgsz=1920 on all 1303 frames. Save detection counts to `outputs/detection_counts.csv`. For the 30 annotated frames, compute:

| Metric | Value |
|--------|-------|
| N_det mean | |
| N_true mean (from annotations) | |
| MAE: \|N_det − N_true\| | |
| AP_small (detections vs head-point annotations) | |

This is **Experiment E001**.

EXPERIMENTS.md entry format:
```
E001 | YOLO-x imgsz=1920 baseline
Q: What is the detection MAE against manual head-point annotations?
Config: YOLOv8-x, imgsz=1920, conf=0.10, class=person
Dataset: 30 annotated frames of test video
Result: MAE=?, AP_small=?, N_det_mean=?, N_true_mean=?
Conclusion: [fill after running]
```

### 1.2 SAHI experiment

Apply SAHI on top of the same YOLOv8-x. Config: `slice_height=640, slice_width=640, overlap=0.2`. Run on the same 30 annotated frames.

**Experiment E002:** Q: Does SAHI improve recall of small/far-away persons?
Measure: AP_small (SAHI) vs AP_small (no SAHI). Also N_det change. Also runtime.

**Decision point:** if AP_small improves >5 pp: SAHI is part of the detection baseline going forward. If AP_small improves <2 pp: the small-person problem requires density-based solution, not tiling. Either way is a result. Write the conclusion.

### 1.3 Temporal jitter analysis (already started, now formalise)

Plot N_det(t) over all 1303 frames. Also compute:
- Frame-to-frame absolute change |ΔN_det|
- For the annotated frames: plot N_det vs N_true on a scatter plot. What is the bias? What is the scatter?
- Compute the conservation residual: if N_det(t+1) − N_det(t) > 5 at 60fps, that is almost certainly noise.

**Experiment E003:** Q: What fraction of N_det variation is detector jitter vs real crowd change?
Claim quantitatively: "X% of frame transitions show |ΔN_det| > Y, which is physically implausible at 60fps."

**Phase 1 deliverable:** E001, E002, E003 written up. A quantitative answer to "does detection work for this video?" The answer is expected to be: "not reliably, and here is why, and here is where it fails first."

---

## Phase 2 — Density branch (Month 1, Days 5–15)

**Goal:** a working density estimator on a public benchmark, then applied to your own video.

### 2.1 ShanghaiTech B — dataset setup (half a day)

Download ShanghaiTech B. Write `data/shanghaitech/prepare.py` that:
- Loads every image and annotation
- Generates a density map for each image using `make_density_map(sigma=15)`
- Saves density maps as `.npy` files (one per image)
- Verifies ∑D̂ ≈ ground truth count for all training images

This is a data pipeline, not a model. It must be fully reproducible from a reload.

### 2.2 CSRNet training on ShanghaiTech B

Notebook `notebooks/03_csrnet_train_shb.ipynb`:
- Use a published CSRNet implementation (e.g. from the original authors' repo). Adapt, don't rewrite.
- Train for 400 epochs. Checkpoint every 50 epochs to `checkpoints/csrnet_shb_epoch_{N}.pt`.
- Log training MAE and val MAE to `outputs/experiments/E004/train_log.csv`.
- Stop when val MAE plateaus.

**Experiment E004:** Q: Does CSRNet reach published MAE on ShanghaiTech B?
Target: MAE ≤ 12 (published CSRNet: 10.6). If you get 11–14, that is a working reproduction. If you get >20, there is a bug.

### 2.3 Qualitative inspection

For 10 test images, plot: original image | predicted density map | ground truth density map | count comparison. Where does the model fail? High density? Perspective foreshortening? Occlusion? Document in EXPERIMENTS.md.

### 2.4 Apply CSRNet to your test video

Run the trained model on all 1303 frames of your video. Save N_dens(t) to `outputs/density_counts.csv`. For the 30 annotated frames, compute MAE(N_dens, N_true).

**Experiment E005:** Q: How does N_dens compare to N_det and N_true on the same frames?
Expected result: N_dens is smoother than N_det (less frame-to-frame jitter) but may be biased because the model was trained on ShanghaiTech, not on your video's domain.

Plot: N_det(t), N_dens(t), N_true (30 points) on the same axis over the 1303 frames.

**This plot is the project's first real result.** It directly shows the detection vs density tradeoff in practice.

### 2.5 Optional: ShanghaiTech A

Train on SHA only if SHB is working well. SHA is harder and takes longer. The research contribution does not require beating SHA SOTA. Skip if behind schedule.

**Phase 2 deliverable:** working CSRNet on SHB with logged MAE; N_dens(t) time series saved; E004/E005 written up; the N_det vs N_dens vs N_true comparison plot. Commit all.

---

## Phase 3 — Motion branch and indicators (Month 2)

**Goal:** produce v̂(x,y,t), derive all 7 indicators, compute crowd pressure P, and make the first physical claim of the project.

### 3.1 Dense optical flow on test video

Notebook `notebooks/05_optical_flow_indicators.ipynb`:
- Implement Farnebäck flow with `cv2.calcOpticalFlowFarneback()`.
- Run on consecutive grayscale frame pairs. Process all 1303 frames. Save magnitude and angle arrays as `processed/flow_{t:08d}.npy` (float32, 2 channels: magnitude, angle).
- Visualise: HSV overlay where hue = direction, value = speed. Inspect 10 frames visually.

**Runtime note:** Farnebäck at 512×512 is ~5–15ms per frame pair on CPU. 1303 frames ≈ 5–20 seconds. Save immediately; never recompute.

### 3.2 Per-cell indicator extraction

Implement `src/indicators/extract.py`. For each frame, compute the 7 indicators over a 32×32 grid. Save as `outputs/indicators/indicators_{t:08d}.npy`.

For the crowd pressure specifically: P(z,t) = ρ̂(z,t) × Var_v(z,t). Compute in image-space units. If homography is available, convert ρ̂ to ped/m² and note the conversion factor.

### 3.3 Crowd pressure visualisation

**Experiment E006:** Q: Does crowd pressure P rise before visually observable congestion onset in the test video?

Method: identify 3–5 frames where you can visually see congestion (slow, dense crowd). Record frame IDs manually. Plot P(t) averaged over the densest zone, with vertical lines at the congestion frames. Does P start rising 5–30 seconds before the congestion is visually obvious?

If yes: the indicator contains a signal. Proceed to temporal model.
If no: check whether the density estimate is accurate first. A wrong D̂ gives a wrong P.

**This is the project's scientific claim.** Be honest about what you observe. If P does not rise in advance, document that and investigate why. That is also a contribution.

### 3.4 Indicator time series analysis

For each of the 7 indicators, plot their time series over the full 1303 frames. Do they:
- Correlate with each other (if so, how?)
- Show clear transitions?
- Have different lead times relative to visual events?

This analysis determines which indicators are informative and which are redundant. Only informative indicators go into the temporal model.

**Phase 3 deliverable:** all indicator time series saved; crowd pressure visualisation on test video; E006 written up; honest assessment of whether P tracks congestion. Commit all.

---

## Phase 4 — Density-vs-detection comparison (Month 2 end / Month 3 start)

**Goal:** produce the core comparative result that answers the research question.

### 4.1 Density binning experiment

**Experiment E007:** Q: At what local density does detection-based counting diverge from density-based counting?

Method:
- Use the 30 annotated frames with N_true.
- Compute per-frame mean ρ̂ (density estimate from CSRNet).
- Bin frames by ρ̂: [0–1], [1–2], [2–4], [4–6], [>6] ped/cell.
- In each bin, compute: MAE(N_det, N_true) and MAE(N_dens, N_true).
- Plot both as a function of density bin.

**Expected finding:** at low density, both methods are roughly comparable. As density rises, detection MAE rises faster than density-model MAE. At high density, detection fails. If this is not what you observe, investigate why (maybe your video doesn't go above 3 ped/cell).

This is a figure you can put in a thesis or paper: "detection vs density as a function of crowd density."

### 4.2 Frame jitter vs conservation residual

**Experiment E008:** Q: How much of detection jitter is inconsistent with the conservation of people implied by optical flow?

Method:
- Compute the flow-implied count change between frames: ΔN_flow = total flow crossing frame boundary (people entering - people leaving, estimated from divergence at borders).
- Compare to ΔN_det.
- If |ΔN_det − ΔN_flow| >> expected noise, detection jitter is the culprit.

This is the quantitative version of "YOLO is noisy." It connects the flow measurement to the counting problem.

**Phase 4 deliverable:** the density-vs-detection comparison figure (E007); the jitter analysis (E008). These two results together constitute the minimum viable research contribution of the project. Even if nothing else works, these are honest, quantitative findings about when detection fails and why. Write them up clearly.

---

## Phase 5 — Temporal model (Month 3)

**Goal:** given the indicator time series, build a model that forecasts the risk state 10–60 seconds ahead.

### 5.1 Feature engineering and dataset construction

For the temporal model you need (input, label) pairs. Input: window of W=30 seconds of indicator vectors. Label: risk state at t+H.

Label options (choose one, document the choice in DECISION_LOG):
- **Option A:** Helbing threshold crossings: label = 1 if any cell has P > 0.01/s² at t+H.
- **Option B:** Visual congestion labels: you marked congestion frames manually in Phase 3.
- **Option C:** Use UCSD or ShanghaiTech Campus anomaly labels (requires running your indicators on those datasets).

For a 1303-frame video at 60fps: 30-second windows are 1800 frames. You have only 21 seconds of video total. You need more video. Options:
- Download FDST (100 videos with per-frame annotations).
- Download a subset of UCF-QNRF video sequences if available.
- Record more of your own video.

**This is the point where you need more data.** Document in PROJECT_STATE.md what video you have and what you need.

### 5.2 XGBoost baseline

Train an XGBoost regressor on (W=30s indicator window → risk at t+H) where H=10s, 30s, 60s. Use 80/20 train/test split on video sequences (not frames — splitting frames from the same video into train/test leaks temporal information).

**Experiment E009:** Q: Do the physical indicators contain predictive signal at H=10,30,60 seconds?
Metric: R² and MAPE on test sequences. Baseline: predict the mean (R²=0). If R² > 0.3 at H=10s: signal exists.

### 5.3 ConvLSTM (only if E009 shows signal)

If XGBoost finds signal: upgrade to ConvLSTM over density map sequences.
Input: stack of T=10 density maps (last 10 frames at 1fps or last 10 seconds).
Output: density map at t+H (H=10s,30s).
Loss: MSE + JS divergence.
Metric: JS divergence on held-out test sequences (following CrowdMAC, WACV 2025).

**Experiment E010:** Q: Does ConvLSTM outperform XGBoost at density forecasting?

### 5.4 Conformal prediction intervals

Wrap the final model (whichever performs better) with conformal prediction:
- Calibration set: 20% of sequences held out from training.
- Compute residuals on calibration set.
- Set α = 0.1 for 90% coverage target.
- On test set: verify empirical coverage ≥ 90%.

**Experiment E011:** Q: Are the model's uncertainty intervals calibrated?
If yes: the output is R̂(t+H) ± δ with guaranteed coverage. If not: investigate where calibration fails.

**Phase 5 deliverable:** trained temporal model (baseline + upgrade); forecast evaluation at multiple horizons; calibrated uncertainty intervals; E009–E011 written up.

---

## Phase 6 — Cross-domain evaluation (Month 4)

**Goal:** test whether indicators estimated from a model trained on ShanghaiTech transfer to a different environment.

### 6.1 JHU-CROWD++ weather subsets

JHU-CROWD++ has weather and illumination subset labels. Run your CSRNet (trained on ShanghaiTech B) on these subsets without fine-tuning. Compute MAE for each condition.

**Experiment E012:** Q: How does density estimation degrade under rain, night, haze conditions?
Expected: performance degrades. The question is by how much and under which condition most.

### 6.2 Your own video as a domain-shift test

Your test video was shot in an unknown environment (presumably in or around Bihtā/Patna). The CSRNet was trained on ShanghaiTech street scenes from China. Compute MAE on your 30 annotated frames. Compare to ShanghaiTech test MAE. The gap is the domain shift.

**Experiment E013:** Q: How large is the domain shift between ShanghaiTech and your target environment?

### 6.3 Optional: FDST temporal evaluation

If you obtained FDST (100 videos): run the full pipeline (density + flow + indicators) on 10 videos from FDST. Check that indicators behave consistently across different cameras and scenes.

**Phase 6 deliverable:** cross-domain MAE table (condition vs MAE); E012/E013 written up; domain shift quantified.

---

## Phase 7 — Ablation and hardening (Month 5)

**Goal:** prove that each component contributes, and characterise where the system fails.

### 7.1 Ablation table

Run the following configurations on the test set and record MAE, risk-score correlation with labels, and lead time:

| Configuration | What's removed | Why test this |
|---------------|---------------|---------------|
| Detection only | Density branch | Shows detection baseline |
| Density only | Motion branch | Shows density-only system |
| Density + mean speed | Speed variance removed from P | Does Var matter or just mean? |
| Density + Var_v (full P) | — | Full system |
| + entropy | Adding H indicator | Does directional disorder add signal? |
| + divergence | Adding div indicator | Does convergence measure add signal? |
| XGBoost only | ConvLSTM | Which model matters? |

This is the ablation study. Every component that does not improve MAE or risk correlation by >5% should be removed from the final system.

### 7.2 Failure mode analysis

For every experiment where the system failed (high MAE, missed event, wrong lead time), write a one-paragraph failure analysis:
- What happened
- Why it happened (wrong density, wrong flow, wrong temporal model, not enough video)
- What would fix it

This section often carries more intellectual value than the positive results in a thesis context.

### 7.3 Lead time measurement

On every sequence where you have a manually labelled onset (congestion visible at frame T):
- Record the first frame T_alert where the system's alert tier exceeded "Advisory."
- Lead time = T − T_alert (positive = warning came before onset; negative = warning came after).
- Plot a histogram of lead times across all such sequences.

**Target:** median lead time > 10 seconds on sequences where the system generates any warning at all. This is not guaranteed. If median lead time is negative, the system is reactive, not predictive. Document that honestly.

**Phase 7 deliverable:** ablation table; failure mode analysis; lead-time histogram; all experiments E001–E013 complete and in EXPERIMENTS.md.

---

## Phase 8 — Writeup and presentation (Month 6)

### 8.1 Thesis chapter structure (suggested)

```
1. Introduction
   1.1 Problem motivation (Kumbh, Hajj, railway stations)
   1.2 Why detection fails (your E001/E003 result)
   1.3 Research question (stated exactly as in ARCHITECTURE.md)
   1.4 Contributions (list 3–5 concrete things you measured or proved)

2. Related work
   2.1 Crowd counting evolution (MCNN → CSRNet → transformers → why we stopped here)
   2.2 Video crowd analysis (People-Flows, CrowdMAC)
   2.3 Crowd physics (Helbing 2007, Gu et al. 2025)
   2.4 Video anomaly detection (what it is, why it is not the same problem)
   2.5 Domain adaptation

3. Proposed system
   (ARCHITECTURE.md, condensed, with component diagrams)

4. Experiments and results
   4.1 Detection baseline and its failure (E001–E003)
   4.2 Density model performance (E004–E005)
   4.3 Indicator extraction and crowd pressure (E006)
   4.4 Detection vs density comparison (E007–E008)
   4.5 Temporal forecasting (E009–E011)
   4.6 Cross-domain evaluation (E012–E013)
   4.7 Ablation study

5. Discussion
   5.1 What the results mean
   5.2 Failure modes
   5.3 Limitations (no disaster ground truth, single camera, metric density uncertainty)
   5.4 Future work

6. Conclusion
```

### 8.2 What you are claiming (and what you are not)

**Claims you are allowed to make:**
- "Detection-based counting degrades to MAE X at crowd densities above Y ped/cell, while density-based estimation degrades more slowly (MAE Z), demonstrating the advantage of field-based approaches at high density."
- "The crowd pressure indicator P = ρ·Var(v) rises [N] seconds before visually-observable congestion onset in [K] of [M] test sequences, providing a lead time of [median] ± [σ] seconds."
- "The system's forecasting model achieves [MAPE] at [H]-second horizon with calibrated intervals showing [X]% empirical coverage."
- "Domain shift from ShanghaiTech to [target environment] causes MAE to degrade from [A] to [B]."

**Claims you are NOT allowed to make:**
- "This system predicts stampedes." (No validated stampede labels, no tested ground truth.)
- "This system prevents stampedes." (You have a research prototype, not a deployed system.)
- "Our approach outperforms state of the art." (You are not competing on a public leaderboard. You are comparing configurations of your own system.)
- Any number without a stated ground truth source and evaluation protocol.

### 8.3 Mid-term evaluation (end of Month 3 or 4 — check your institution's calendar)

By mid-term you should be able to show:
- Live demo: video with density overlay and crowd pressure heatmap updating frame by frame.
- The detection vs density comparison plot (E007) — this is visual and immediately comprehensible.
- At least one quantitative table with MAE and a comparison.
- The research question stated precisely, with the experimental design that will answer it.

Do not try to show everything. Show the detection failure, the density alternative, and the crowd pressure signal. Those three things together make a coherent story.

---

## Milestone summary

| Month | End-of-month deliverable | Must be saved to disk |
|-------|--------------------------|----------------------|
| 1 (Phase 0–2) | Ground truth annotations; E001–E005; detection vs density first plot | head_points.csv, detection_counts.csv, density_counts.csv, csrnet_shb_best.pt |
| 2 (Phase 3–4) | All 7 indicators extracted; E006–E008; the comparison figure | indicators/*.npy, indicator_timeseries.csv |
| 3 (Phase 5) | Temporal model trained; E009–E011; calibrated intervals | temporal_model.pkl or convlstm_best.pt |
| 4 (Phase 6) | Cross-domain evaluation; E012–E013 | cross_domain_results.csv |
| 5 (Phase 7) | Ablation table; failure analysis; lead-time histogram | ablation_results.csv, lead_times.csv |
| 6 (Phase 8) | Thesis draft; final demo | thesis_draft.pdf |

---

## Risk register and contingencies

| Risk | Trigger | Contingency |
|------|---------|-------------|
| Annotation takes too long | Still not done by end of Phase 0 | Annotate only 15 frames; use statistical estimates for the rest |
| CSRNet doesn't converge | Val MAE not improving after 200 epochs | Use a pre-trained CSRNet from a published repo and fine-tune for 50 epochs |
| No additional video for temporal model | Can't find enough video with known events | Shift scope: characterise indicators on FDST/UCF-QNRF frames; drop forecasting horizon |
| Domain shift too large for density model | MAE on own video > 3× ShanghaiTech MAE | Add fine-tuning experiment with 5–10 annotated frames; report domain shift honestly |
| Kaggle session lost mid-training | Training run lost | Every epoch saves a checkpoint. Re-start from last checkpoint. This is why the checkpointing rule exists. |
| XGBoost finds no signal | R² < 0.1 at all horizons | This is a result: indicators do not predict within the video duration available. Report it. Don't add a GNN to hide it. |
| Optical flow too noisy on low-texture frames | High variance in indicator time series | Temporal smoothing (Gaussian filter over time) on flow magnitude before indicator computation. Document in DECISION_LOG. |

---

## What "done" looks like

The project is done when you can answer the following questions with a number and a confidence interval:

1. At what density (in image-space cells or ped/m²) does YOLO detection MAE exceed the density model's MAE on your data?
2. Does crowd pressure P rise before visually-observable congestion onset? By how many seconds on average?
3. Does your temporal model forecast risk better than chance (R² > 0.0) at 10 seconds? At 30 seconds?
4. How much does cross-domain performance degrade compared to in-distribution performance?

If you can answer all four with honest numbers, the project is done. Everything else is commentary.

# RESEARCH_MAP.md — Crowd Intelligence & Crowd-Risk Estimation

**Phase 0 deliverable. Field map, not an implementation plan.**
Version 0.1 · Status: literature synthesis, no experiments run yet
Scope: what the field is, how it got here, what is solved, what is not, and which parts a BTech can genuinely contribute to.

Everything below is either (a) traceable to a cited primary source, or (b) explicitly labelled as my judgement. Where I am unsure, I say so. Do not cite this document in your thesis — cite the papers it points to.

---

## 0. How to read this

The single most important thing in this document is **Section 9 (research gaps)** and **Section 10 (recommended direction)**. Everything before it exists to justify those two sections. Everything after it is scheduling.

The second most important thing is Section 1, because it kills the assumption embedded in your current pipeline sketch. "Crowd counting → density → tracking → flow → anomaly → risk → warning" is not a pipeline. It is a list of **eight separate research problems with different inputs, different supervision, different metrics, and different literatures.** They are not stages of one system. Treating them as stages is the standard failure mode of crowd-safety student projects, and it is exactly what two of the three application papers in your project folder do.

---

## 1. Problem taxonomy

| # | Task | Input | Output | Supervision | Standard metric | Is it solved? |
|---|------|-------|--------|-------------|-----------------|---------------|
| T1 | Crowd counting | single image | scalar N | point annotations | MAE, RMSE | Largely mature on benchmarks |
| T2 | Density estimation | single image | D(x,y), ∑D = N | point annotations + kernel | MAE/RMSE + density quality | Mature; ground-truth generation still contested |
| T3 | Crowd localization | single image | set of head points | point annotations | F1 @ distance threshold, NAE | Active, harder than counting |
| T4 | Video counting / temporal consistency | frame sequence | N(t), D(x,y,t) | point annotations per frame | MAE + temporal error | **Under-served.** Few methods, few datasets |
| T5 | Multi-object tracking / head tracking | frame sequence | ID'd trajectories | box+ID annotations | HOTA, IDF1, MOTA | Solved for sparse; degrades hard in dense crowds |
| T6 | Crowd motion / flow field estimation | frame sequence | v(x,y,t) | usually unsupervised (optical flow) | flow EPE, or task-level | Mature as a tool, not as a task |
| T7 | Video anomaly detection | frame sequence | frame-level anomaly score | weak (video-level) or one-class | AUC, AP | Active; benchmark-saturating |
| T8 | Crowd-state / risk forecasting | past window | future density / risk over horizon H | varies, often none | JS divergence, or ad hoc | **Barely a field.** Biggest opportunity |

Two more that are not vision tasks but constrain everything:

- **T9 — Perspective calibration / metric density.** Converting image-space density to persons/m². Required for any claim about safety thresholds. Usually hand-waved.
- **T10 — Crowd physics.** Density–velocity relations, phase transitions, crowd pressure, turbulence. Not machine learning. This is where the actual safety knowledge lives.

**Note the asymmetry:** T1–T3 have huge literatures, standard benchmarks, and leaderboards. T8 has almost nothing. Your project's stated goal lives in T8. That is both the opportunity and the risk.

---

## 2. Historical evolution

### 2.1 Counting & density (T1–T3)

| Era | Dominant idea | Why it appeared | Why it broke |
|-----|---------------|-----------------|--------------|
| pre-2010 | Detect-then-count (HOG, part-based), or global regression from texture/edge features | Detectors existed; crowds were sparse | Detection fails past ~10 persons/frame; regression gives no spatial information |
| 2010–2015 | Regression to a **density map** (Lempitsky & Zisserman 2010) | Decoupled counting from detection. Count = ∑ D(x,y). Point annotation is 100× cheaper than boxes | Hand-crafted features couldn't handle scale |
| 2016–2018 | Multi-column CNNs (MCNN 2016), then single-column with dilated convs (CSRNet 2018), SANet | Perspective creates 10× head-scale variation in one image; different receptive fields for different scales | Multi-column columns turn out to be redundant; CSRNet showed one strong backbone + dilation beats three weak columns |
| 2019–2021 | Better **losses**, not architectures: Bayesian Loss, DM-Count (optimal transport), and point-based P2PNet | The Gaussian kernel used to make density ground truth is itself an unvalidated assumption; these methods regress to the point annotations directly | Localization quality still limited |
| 2021–2023 | Transformers: TransCrowd, CLTR, PET (2023), STEERER (2023) | Global context and long-range dependency; scale handled by attention instead of pyramids | Compute; data hunger |
| 2023–2026 | Prompt/foundation-model conditioning (mPrompt 2024), CLIP-based (CLIP-EBC), scale-decoupling (STEERER), better probabilistic heads (ZIP 2025, zero-inflated Poisson) | Counting is now a modelling-the-output-distribution problem more than a features problem | Gains are getting small. See 2.6 |

Reference point for how mature this is: on ShanghaiTech Part A, MCNN (2016) reported MAE ≈ 110; ZIP (2025) reports MAE 47.8 and RMSE 75.0, ahead of APGCC's 48.8/76.7 (arXiv 2506.19955). On NWPU-Crowd val, ZIP reports 28.2 MAE against CLIP-EBC's 36.6.

**Read that carefully.** Nine years of intense work moved SHA MAE from 110 to ~48, and the last three years moved it from ~55 to ~48. **You will not beat this. Do not try.** Reproducing a mid-tier method is a learning exercise, not a contribution.

### 2.2 Video counting (T4)

Small, distinct literature. The key idea worth your attention:

**Liu, Salzmann & Fua, "Estimating People Flows to Better Count Them in Crowded Scenes" (ECCV 2020; TPAMI 2021).** Instead of regressing density per frame, regress **people flows between locations across consecutive frames**, then infer density from flows. This lets you impose a hard **conservation-of-people constraint** rather than a weak smoothness penalty. Code: https://github.com/weizheliu/People-Flows · arXiv 1911.10782, 2012.00452.

This paper is the direct, correct, published answer to the failure you already observed empirically (79.88% of your frame transitions changed count; ±16 person jumps). Read it first.

Related: STANet / STNNet (temporal attention for video counting, DroneCrowd), and ConvLSTM-based counting (Xiong et al.).

### 2.3 Tracking (T5)

SORT → DeepSORT → ByteTrack → OC-SORT / BoT-SORT / StrongSORT. The important structural fact for you: all of these are **tracking-by-detection**, so they inherit your detector's recall. If YOLO misses 40% of small heads, no tracker recovers them. MOT20 exists because MOT17 was too easy; CroHD (Crowd of Heads) exists because MOT boxes are full-body and full bodies are invisible in dense crowds.

**Judgement:** tracking is a trap for this project. It is a large engineering commitment that gives you individual trajectories in exactly the regime where you don't need them (sparse), and fails in the regime where the safety problem lives (dense). Use it only if you can name the specific measurement it enables that flow fields cannot.

### 2.4 Video anomaly detection (T7)

Reconstruction/prediction autoencoders (ConvLSTM, future-frame prediction) → weakly-supervised MIL on video-level labels (UCF-Crime, Sultani 2018; XD-Violence 2020) → CLIP/VLM adaptation (VadCLIP, AnomalyCLIP, LAVAD) → MoE and prompting methods; GS-MoE reports 91.58% AUC on UCF-Crime (arXiv 2508.06318).

**Critical distinction the field itself blurs, and you must not:** UCF-Crime "anomalies" are *semantic events* (robbery, arson, road accident). Crowd danger is a *physical state*, not a semantic event. A model that is excellent at recognising "explosion" tells you nothing about whether a crowd is approaching a crush. **UCF-Crime is not a crowd-safety dataset.** Use it to learn the VAD paradigm, not as a target benchmark.

### 2.5 Crowd physics (T10) — the part AI projects skip

This is the most important section in the document.

**Helbing, Johansson & Al-Abideen, "Dynamics of Crowd Disasters: An Empirical Study" (Phys. Rev. E 75, 046109, 2007; arXiv physics/0701203).** Video analysis of the 2006 Mina/Jamarat disaster. Findings:

- Two sudden phase transitions: laminar flow → **stop-and-go waves** → **crowd turbulence**.
- Local density alone is *insufficient* to identify critical times and locations. Velocity field alone is also insufficient.
- The decisive quantity is **crowd pressure** = local density × variance of local speeds.
- The accident began ~10 minutes after crowd pressure exceeded ≈ 0.02 s⁻², and >30 minutes after stop-and-go waves set in.
- Explicitly: these warning signs "can be evaluated on-line by an automated video analysis system."

Follow-ups give thresholds: stop-and-go waves emerge around 2–4 persons/m², turbulence above roughly 4–7 persons/m² depending on body diameter (arXiv 1402.7011); turbulence at Jamarat occurred around 9 pilgrims/m².

**Gu, Guiselin, Bain, Zuriguel & Bartolo, "Emergence of collective oscillations in massive human crowds," Nature, 5 Feb 2025 (doi:10.1038/s41586-024-08514-6).** Dense crowds at the San Fermín festival self-organise into **macroscopic chiral oscillators** — hundreds of people in coordinated orbital motion, with no external driver — once density exceeds a critical threshold (~4 ped/m²). They build a mechanical model (odd friction → non-reciprocal phase transition) that reproduces this, show the **same chiral dynamics appeared at the onset of the 2010 Love Parade disaster**, and propose a monitoring protocol to anticipate it. Their measurement pipeline is computer vision: density and velocity *fields*, not individual tracking.

Also worth reading: the 2025 survey "Exploring Dense Crowd Dynamics" (arXiv 2505.05826), and the MADRAS dataset (Lyon Festival of Lights 2022, ~7000 microscopic trajectories, densities to 4 ped/m²).

**Why this matters to you.** Both results say the same thing in different vocabularies: **the early-warning signal is a low-dimensional statistic of the density and velocity fields — not an object-detection output.** You do not need to detect every person to compute ρ·Var(v). You need a reliable density field and a reliable flow field. This inverts the architecture you sketched.

### 2.6 Where the field is in 2026 — my read

- Single-image counting: **mature, saturating**. Improvements are fractions of an MAE point and increasingly come from output-distribution modelling, not perception.
- Localization: still improving, still useful.
- **Video** crowd analysis: much thinner than image-based. Temporal consistency is under-exploited.
- VAD: heavily active, benchmark-saturating, drifting toward VLMs, and **semantically misaligned with crowd safety**.
- Crowd forecasting: nearly empty. WACV 2025's CrowdMAC (arXiv 2407.14725) formalises "crowd density forecasting" — predict future density maps from past density maps, evaluated by Jensen-Shannon divergence on FDST/CroHD/VSCrowd/SDD/ETH-UCY/inD/JRDB — and explicitly notes that the field's prior work amounts to roughly one patch-based model (PDFN-ST). **A task with one or two prior methods is where a BTech can actually contribute.**
- Domain generalization for counting: active and unsolved (AAAI 2023 arXiv 2212.02573; MPCount; several 2025–26 follow-ups). Cross-dataset MAE degradation is severe and reported honestly by these papers.
- Crowd physics ↔ computer vision: **almost no crossover.** The Nature 2025 paper is physicists using CV. The CV community has largely not picked up crowd pressure or phase-transition detection as tasks. This gap is real.

---

## 3. Major model families (and whether you need them)

| Family | What it buys | Verdict for this project |
|--------|--------------|--------------------------|
| YOLO / RT-DETR person detection | boxes in sparse-to-moderate scenes | Keep as a **baseline to be beaten**, not as the system |
| Sliced/tiled inference (SAHI) | +6.8–14.5 AP on small objects for aerial detectors, model-agnostic, no retraining (arXiv 2202.06934, github.com/obss/sahi) | **Run this immediately.** Cheapest possible answer to your small-person failure |
| Density-map CNNs (CSRNet class) | robust count without detection | Core. Reproduce one |
| Point-based (P2PNet, CLTR, APGCC) | count + localization | Reproduce one if localization is needed |
| Optical flow (Farnebäck classical, RAFT deep) | velocity field without detection | **Core.** Farnebäck first — it's what the crowd-physics literature uses and it runs on CPU |
| Trackers | individual trajectories | Optional. Justify or omit |
| ConvLSTM / temporal transformers | temporal modelling of maps | Likely needed for T8 |
| GNNs | interaction modelling on person graphs | **Skip unless a specific measurement demands it.** You need individual nodes, which you cannot reliably get in dense crowds. Using a GNN here would be model-fashion, not method |
| Video foundation models (VideoMAE, InternVideo2) | strong video features | Feature-extraction only; no Kaggle-scale training |
| CLIP/VLM for anomaly | zero-shot semantic anomaly | Wrong semantic level for crowd physics. Interesting side-experiment at most |

---

## 4. Dataset landscape

### 4.1 Image counting/localization

| Dataset | Size | Annotation | Use to you |
|---|---|---|---|
| ShanghaiTech A/B | 482 + 716 images, 330k points | points | Standard entry benchmark. A: dense web images (33–3139/img); B: Shanghai streets (9–578) |
| UCF-CC-50 | 50 images | points | Extreme-density stress test only; too small to train on |
| UCF-QNRF | 1,535 images, 1.25M annotations, max 12,865 | points | Dense + diverse. Core |
| JHU-CROWD++ | 4,372 images, ~1.5M annotations | points + weather/illumination attributes | **Environmental robustness testing.** Directly relevant to your generalization claim |
| NWPU-Crowd | large-scale | points, held-out test server | Current standard large benchmark |
| WorldExpo'10 | video frames, per-scene | points + ROI | Cross-scene generalization, historical |

### 4.2 Video crowd (this is what your project actually needs)

| Dataset | Content | Annotation | Why it matters |
|---|---|---|---|
| **FDST** (Fudan-ShanghaiTech) | 100 videos, 15 locations, distinct camera poses | per-frame head points | Standard video counting benchmark |
| **VSCrowd** | 634 videos (479 train / 137 test) | per-head: **tracking ID + bbox + centre point**, per frame | Counting + localization + tracking in one annotation. https://github.com/HopLee6/VSCrowd-Dataset |
| **CroHD** | dense-crowd head tracking | head boxes + IDs | Head-level MOT, since bodies are occluded |
| **DroneCrowd** | 112 clips, 33,600 frames @1920×1080, 70 scenarios, 20,800 trajectories, 4.8M head annotations | points + trajectories + video attributes | Aerial, video, trajectories. arXiv 2105.02440 · github.com/VisDrone/DroneCrowd |
| MADRAS | Lyon 2022 festival, densities to 4 ped/m² | macroscopic flows, GPS, ~7000 trajectories | Real dense-crowd physics ground truth |

### 4.3 Anomaly / abnormal motion

| Dataset | Note |
|---|---|
| UMN | Classic escape/panic scenes. **Staged, tiny, and effectively saturated** — the paper in your project folder reports >99% accuracy on it. Do not use as a headline benchmark |
| PETS2009 | Staged crowd scenarios, multi-view |
| UCSD Ped1/Ped2, CUHK Avenue, ShanghaiTech Campus | Classic VAD. Anomalies are *objects in wrong places* (bikes on walkways), not crowd physics |
| UCF-Crime, XD-Violence | Weakly-supervised semantic anomaly. Not crowd safety |
| GBA Stampedes, GSMADC | Released with Cob-Parro et al. 2025 (in your project folder), "data available on request." Small/medium crowds, 5–50 people |

### 4.4 The dataset problem, stated plainly

**There is no benchmark for crowd-crush early warning.** There is no dataset of labelled crowd-disaster onsets with a defined event time, a prediction horizon, and a train/test split. Real crushes are rare, ethically impossible to stage, and recorded from bad angles.

Every project that claims "stampede prediction" resolves this in one of four ways, and you must choose consciously:

1. Use staged panic/dispersal (UMN) and call it a stampede. **Dishonest and saturated.**
2. Classify stills as "stampede"/"non-stampede" from self-labelled internet images. This is what `Presentation.pdf` in your project folder does, and its own Constraints slide admits the dataset was not available and had to be self-labelled. **This is not a crowd-safety system; it's an image classifier on a label with no operational definition.**
3. Report metrics from a deployment nobody can reproduce. `2606.05185v1.pdf` (Drishti) reports MAE 3.2 persons/m², anomaly F1 0.91, XGBoost congestion MAPE 8.3% at a 5-minute horizon on Kumbh Mela and RCB Parade data. There is no public dataset, no released code, no held-out split you can access. Treat as a systems paper, not a benchmark.
4. **Measure a physically-defined risk state that has published thresholds, and evaluate the estimator against ground truth you can actually get.** This is the honest option and it's the one I recommend. See Section 10.

---

## 5. Papers by tier

### Tier 1 — must read (understanding, no implementation)
1. Helbing, Johansson & Al-Abideen 2007, *Dynamics of Crowd Disasters* — arXiv physics/0701203. **Read first, before any ML paper.**
2. Gu et al. 2025, *Emergence of collective oscillations in massive human crowds*, Nature — doi:10.1038/s41586-024-08514-6.
3. Lempitsky & Zisserman 2010, *Learning to Count Objects in Images* — origin of the density map.
4. Zhang et al. 2016, MCNN (CVPR) — why multi-scale.
5. Li et al. 2018, CSRNet (CVPR) — why dilation beat multi-column.
6. Liu, Salzmann & Fua 2020/2021, *People Flows* — arXiv 1911.10782 / 2012.00452.
7. Fujii et al. 2025, CrowdMAC (WACV) — arXiv 2407.14725. Read for the **task definition and metric**, not the architecture.
8. Sultani et al. 2018, *Real-world Anomaly Detection in Surveillance Videos* — the MIL/weak-supervision paradigm.
9. Du, Deng & Shi 2023, *Domain-General Crowd Counting in Unseen Scenarios* (AAAI) — arXiv 2212.02573.
10. Cob-Parro, Losada-Gutiérrez & Marrón-Romera 2025, EAAI 142:109940 (in your project folder).

### Tier 2 — reproduce
1. CSRNet on ShanghaiTech B, then A. Your density baseline.
2. SAHI over your existing YOLO baseline. Cheap, decisive.
3. Farnebäck dense optical flow → velocity/entropy/divergence fields on your own video.
4. A point-based localizer (P2PNet or APGCC) if localization proves necessary.
5. People-Flows, if temporal counting becomes central.

### Tier 3 — skim for awareness
STEERER, PET, Gramformer, mPrompt, CLIP-EBC, ZIP (know the leaderboard, don't chase it) · ByteTrack/OC-SORT · RAFT · VadCLIP, GS-MoE · MPCount and 2025 single-domain-generalization follow-ups · DINOv2/SAM2 as frozen feature extractors.

---

## 6. Critique of the three application papers in this project folder

Because you will be tempted to imitate them.

**CrowdShield.** Proposes YOLO + CSRNet + threshold-based anomaly detection + heatmaps + SMS alerts within a 500 m/1 km radius. It contains **no quantitative evaluation** — no MAE, no dataset, no baseline comparison, no ablation. Its Results section describes intended behaviour in prose ("does an excellent job"). It also claims "real time stampede prediction" while describing threshold-based density anomaly detection. This is a system-description paper. **It is not a research contribution and it must not be your template.**

**Drishti (2606.05185v1).** Much more thorough — real deployments, seven evaluation dimensions, an XGBoost 5-minute congestion forecast, ablations of a sort. But it is a **systems/deployment paper**: no public data, no released code, no comparison against any published baseline on any public benchmark. Useful as a source of *feature engineering ideas* (zone density, adjacent-zone density, event-schedule phase, time-of-day encoding) and as evidence that a simple gradient-boosted model on physical features is a legitimate forecasting baseline. Not useful as a claim you can verify or beat.

**Presentation.pdf (MCNN stampede classification).** A binary image classifier on a self-labelled, admittedly-invalid dataset. Its own future-work slide asks for "a larger and a valid dataset." This is a clear demonstration of failure mode 2 from Section 4.4.

**Cob-Parro et al. 2025 (EAAI).** This is the only real research paper of the four, and it is your closest competitor. Farnebäck optical flow → three features (entropy of magnitude, TOV, KDE of flow angle) → three parallel 4-layer LSTM branches → frame-level stampede/no-stampede. Evaluated on UMN+PETS2009 and two new datasets (GBA Stampedes, GSMADC) with 10-fold CV, ablations over branch count and layer depth, and — importantly — an **explicit generalization experiment** where models are trained mostly on one dataset and evaluated on others without fine-tuning. Their headline: the full ETK model degrades from 99.4% to 91.7% accuracy on the hardest dataset under domain shift, while single-branch variants fall to ~70%.

Note what it does *not* do: no density estimation, no metric density (persons/m²), no crowd pressure, no lead-time/early-warning metric, no prediction horizon, and it is scoped to crowds of 5–50 people (their own limitation statement flags scaling to larger crowds as future work). **These are your openings.**

---

## 7. Unresolved problems in the field (2026)

1. Video temporal consistency in counting — few methods, few datasets.
2. Cross-domain generalization of counting/density — severe, honestly reported degradation.
3. Metric density (persons/m²) from an uncalibrated single camera — mostly avoided.
4. Crowd forecasting — a task with almost no methods.
5. Uncertainty and calibration in safety-critical crowd estimation — essentially absent from crowd counting.
6. The evaluation gap for early warning — no accepted protocol for lead time, false alarms per hour, or prediction horizon in this domain.
7. The physics/CV disconnect — Helbing's crowd pressure and Gu et al.'s chiral onset are not implemented, benchmarked, or stress-tested by any CV paper I found.
8. Small/far-person recall in dense high-resolution video — your own observed failure, still open.
9. Density-map ground truth generation (kernel choice) remains an unvalidated modelling assumption.
10. Semantic anomaly ≠ physical crowd risk, and no benchmark forces anyone to confront this.

---

## 8. What your existing YOLO experiment actually established

You have a real, interpretable result already. State it precisely:

- Detection count is **hyperparameter-dependent to the point of meaninglessness as an absolute quantity**: at imgsz=1280, count went 65 → 9 as conf went 0.05 → 0.30. That is a 7× swing from a threshold, on a fixed scene.
- Increasing resolution 1280 → 1920 raised detections 65 → 80 at low confidence: recall is resolution-limited, confirming a **scale** failure, not a detector-quality failure.
- Frame-to-frame count changed in 79.88% of transitions, mean |Δ| = 2.50, max +16/−14, on a scene whose true count cannot plausibly change that fast at 59.94 fps. This is **detector jitter, not crowd dynamics** — and it is exactly the failure People-Flows was designed to fix.
- σ = 5.42 on a mean of 71.57 is ~7.6% noise, before you've even asked whether the mean is correct.

**Conclusion you are entitled to draw:** a per-frame detection count is not a usable measurement primitive for a crowd-safety system. **Conclusion you are not entitled to draw:** anything about the true crowd size. You have no ground truth for this video. Get one — annotate head points on ~30 sampled frames — or stop quoting the number.

---

## 9. Research gaps, ranked

Scored on Novelty × Feasibility × Impact × Data availability × Compute feasibility (1–5 each; product shown).

| # | Gap | N | F | I | D | C | Score | Note |
|---|-----|---|---|---|---|---|-------|------|
| **G1** | **Vision-based estimation of physically-grounded crowd risk indicators (density, speed variance, crowd pressure ρ·Var(v), flow disorder) with quantified reliability — and characterisation of when detection-based vs density-based estimators fail** | 4 | 5 | 5 | 4 | 5 | **2000** | The physics/CV gap. Nobody has stress-tested Helbing's indicators against modern CV estimators |
| G2 | Temporal stabilization of crowd counts under detector jitter; count as a state-estimation rather than per-frame regression problem | 3 | 5 | 4 | 5 | 5 | 1500 | Partially done by People-Flows; your angle is the *hybrid detection/density* regime |
| G3 | Short-horizon crowd density forecasting with uncertainty, evaluated by lead time | 4 | 4 | 5 | 4 | 4 | 1280 | CrowdMAC opened the task; uncertainty and lead-time evaluation are open |
| G4 | Adaptive detection↔density switching driven by estimated local density | 3 | 5 | 4 | 4 | 5 | 1200 | Simple, testable, publishable as a component |
| G5 | Cross-domain robustness of *risk indicators* (not just counts) across camera geometry | 4 | 3 | 5 | 4 | 4 | 960 | Needs multi-domain video; harder |
| G6 | Metric density from single uncalibrated camera via approximate homography + head-size priors | 3 | 3 | 5 | 3 | 5 | 675 | Necessary for any safety claim; annoying to validate |
| G7 | Calibrated uncertainty for crowd counting (conformal prediction on count intervals) | 4 | 4 | 3 | 4 | 4 | 768 | Clean, small, self-contained. Good fallback |
| G8 | Automatic detection of the stop-and-go → turbulence phase transition from video | 5 | 2 | 5 | 2 | 4 | 400 | Highest novelty, worst data situation |
| G9 | Chiral-oscillation onset detection as an early-warning signal (operationalising Gu et al. 2025) | 5 | 2 | 5 | 1 | 4 | 200 | Beautiful, but you have no dense-crowd video with ground truth |
| G10 | GNN interaction modelling for crowd risk | 2 | 2 | 2 | 2 | 3 | 48 | **Listed to be explicitly rejected.** Requires reliable per-person nodes in exactly the regime where you can't get them |

---

## 10. Recommended research direction

### The question

> **When individual detection becomes unreliable, can physically-grounded crowd-risk indicators — local density, speed variance, and crowd pressure ρ·Var(v) — still be estimated reliably from video; how do detection-based and density-based estimators degrade differently as density rises; and how much lead time does the resulting risk signal provide relative to observable onset?**

This is G1 + G2 + parts of G4/G6.

### Why this and not "stampede prediction"

- The target quantity has a **published operational definition and a published threshold** (crowd pressure > 0.02 s⁻², Helbing et al. 2007), so you are not inventing your own label.
- It is measurable from **fields, not identities** — density map + optical flow — so it survives the exact regime where your YOLO baseline dies.
- It gives you a **defensible negative result** if things go badly: "here is precisely where and why estimation of the risk indicator breaks down" is a legitimate contribution. "Our stampede predictor got 94%" on a staged dataset is not.
- It engages a **real, checkable gap**: the crowd-physics literature specifies indicators and says they can be computed automatically; the CV literature has not verified this with modern estimators.
- It is **honest**. You never claim to predict a stampede. You claim to estimate a risk indicator and characterise its reliability.

### Architecture implied by this question (provisional, revisable)

```
Video
 ├─ Density branch:  density-map model → D(x,y,t) → metric density via approximate homography
 └─ Motion branch:   dense optical flow → v(x,y,t)
                          │
              per-cell risk indicators:
              ρ, |v|, Var(v), P = ρ·Var(v), divergence, directional entropy
                          │
              temporal model over indicator time series → short-horizon forecast + uncertainty
                          │
              risk state + explanation (which indicator drove it)
```

Detection (YOLO+SAHI) appears **only as a baseline and as a sparse-regime cross-check**, not as the backbone. Tracking, GNNs, and VLMs appear only if an experiment demands them.

### Falsifiable sub-questions (one experiment each)

- Q1: Does sliced inference (SAHI) recover small-person recall, and by how much, vs. raw imgsz scaling? Metric: recall / AP_small on annotated frames.
- Q2: At what local density does detection-based counting diverge from density-based counting on the same frames? Metric: MAE of each vs. point ground truth, binned by density.
- Q3: How much of the observed count jitter is detector noise vs. real change? Metric: |ΔN| vs. flow-implied conservation residual.
- Q4: Is crowd pressure computed from estimated fields stable under estimator noise? Metric: variance of P under detector/flow perturbation.
- Q5: Does P rise before observable congestion onset, and by how long? Metric: lead time, on the few real sequences you can obtain.

---

## 11. Learning & reproduction order

Just-enough theory, then implement, then read the theory the failure demands.

| Step | Build | Read alongside | Understanding to reach |
|------|-------|----------------|------------------------|
| 1 | Frame-accurate video I/O, ground-truth point annotation of ~30 frames of your own video | — | You cannot evaluate anything without ground truth |
| 2 | YOLO baseline, re-run with logged config; SAHI on top | SAHI paper | Detection is threshold- and scale-dependent |
| 3 | Density map ground-truth generation from points (Gaussian kernels), CSRNet forward pass | MCNN, CSRNet | ∑D = N; kernel choice is an assumption |
| 4 | Train CSRNet on ShanghaiTech B, then A | Bayesian Loss / DM-Count | Why the loss matters more than the backbone |
| 5 | Farnebäck flow on your video; compute |v|, Var(v), entropy, divergence | Helbing 2007 | Motion fields carry risk information without identity |
| 6 | Compute ρ·Var(v) per cell; visualise over time | Gu et al. 2025 | The actual safety quantity |
| 7 | Approximate homography / perspective map → persons/m² | Helbing thresholds | Why image-space density is not a safety statement |
| 8 | Temporal model over indicator series (start with gradient boosting, not an LSTM) | Cob-Parro 2025, Drishti feature set | Whether the features contain signal at all |
| 9 | Only then: ConvLSTM / transformer over density maps | CrowdMAC | Forecasting as a task with a metric |
| 10 | Cross-domain evaluation (JHU-CROWD++ attributes, FDST/VSCrowd, your own footage) | AAAI 2023 DG paper | Generalization as a measured property |

Rule for the whole project: **no model enters the pipeline without a failed experiment that motivates it.**

---

## 12. Scope tiers

**Minimum viable (must exist by month 5).** Reproduced density-map baseline with published-comparable MAE on ShanghaiTech; YOLO and YOLO+SAHI detection baselines; quantified comparison of detection- vs density-based counting binned by density on annotated frames; optical-flow indicator extraction; crowd-pressure computation with visualisation; honest failure analysis. No forecasting required.

**Strong.** All of the above plus: metric density via homography with error bars; short-horizon (10–30 s) forecasting of density/indicators with a real baseline comparison; lead-time evaluation on the sequences you have; cross-scene evaluation on at least two unseen environments; full ablation of indicator components.

**Research-level extension (only if months 1–5 go well).** Uncertainty-calibrated risk estimation (conformal intervals on count/density); domain-generalization study of *indicators* rather than counts; attempt at detecting the stop-and-go→turbulence transition or chiral onset on whatever dense footage is obtainable. Any of these could be a workshop paper. None of them is required for a strong BTech.

---

## 13. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| No dense-crowd video with usable ground truth | **High** | The project is designed so that estimator-reliability results stand without disaster footage |
| Homography/metric density unvalidatable | High | Report image-space indicators as primary, metric density as secondary with stated assumptions |
| Kaggle GPU quota / session loss | Certain | Checkpoint everything; expensive inference runs once and writes `.npy`/`.csv`; analysis notebooks never recompute |
| Temptation to add fashionable models | High | The "no model without a failed experiment" rule; DECISION_LOG.md entries required |
| Scope creep into 27 dashboard features | High | Dashboard is a month-6 presentation artifact, not a research output |
| Overclaiming in the writeup | High | Never write "stampede prediction." Write "crowd-risk indicator estimation" |

---

## 14. Open questions for the next session

1. Can you obtain any video with a known congestion/crush onset time? This determines whether Q5 (lead time) is answerable at all.
2. Is a fixed-camera dataset available to you locally (college gate, station) where you could record legally and with consent, giving a real domain-shift test set?
3. Do you have any way to estimate ground-plane geometry for your existing test video (known object dimensions, visible reference lengths)?

---

*Sources for every quantitative claim above are linked inline. Nothing in Sections 9–13 is from a paper — those are my judgements and should be argued with.*

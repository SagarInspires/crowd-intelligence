# PAPERS.md — Current & Trending Papers (2025–2026), Organized by System Pillar

**Purpose:** this supplements RESEARCH_MAP.md (which covers historical evolution and Tier 1 classics) with what is specifically new in 2025–2026. Every paper below is linked to a section of ARCHITECTURE.md so you know exactly why it's here.

**How to use this list:** don't read everything. Read the "closest competitor" and "read first" papers in each pillar carefully. Skim the rest for awareness of what direction the field is moving, so you can position your work honestly relative to it.

---

## 0. The single most important paper to read first

**Cob-Parro, A.C. et al., "Video-based abnormal crowd behavior recognition using topological data analysis and Farneback dense optical flow," Engineering Applications of Artificial Intelligence, 142:109940, 2025.** (In your project folder as `1s2.0S0952197624020992main.pdf`.)

This is not new to you, but re-flag it here because it is your **closest published competitor**, and two very recent 2025–2026 papers explicitly build on it — see below. Their method: Farnebäck optical flow → 3 features (flow-angle KDE, TOV, entropy) → 3 parallel LSTM branches → binary stampede classification, with honest cross-dataset generalization testing (94% in-domain, dropping to ~88-90% cross-domain). Scoped to 5–50 people, no density estimation, no metric density, no lead-time metric. These four absences remain your openings.

---

## 1. Crowd counting / density estimation — what's current

### Closest to production SOTA (2025)

**Wang et al. (or similar), "ZIP: A Zero-Inflated Poisson Model for Crowd Counting" (2025).** arXiv 2506.19955. Reports MAE 47.8/RMSE 75.0 on ShanghaiTech A (best published at time of writing) and MAE 28.2 on NWPU-Crowd val, ahead of APGCC (48.8/76.7 on SHA). **Tier 3 — skim.** The direction here (modelling the output as a zero-inflated count distribution rather than a smooth density map) is conceptually interesting but not something a BTech should try to beat on the leaderboard.

**APGCC** and **CLIP-EBC** — point/prompt-based localizers referenced as runners-up to ZIP above. Tier 3 — awareness only.

### Video crowd counting — directly relevant, thin literature

**Shu et al., "Adapting Lightweight Image-based Counting Models for Video Crowd Counting," CVPR 2026.** (openaccess.thecvf.com/content/CVPR2026) This is a **2026 paper doing exactly your T4 problem**: taking image-based counters and adapting them for video, addressing the same detect-then-count vs. density-map-then-count tension your YOLO experiment exposed. **Tier 1 — must read.** This is the most current, most directly relevant paper to your Phase 2 work; read it before finalizing your video-adaptation strategy for CSRNet.

**"Motion-guided Non-local Spatial-Temporal Network for Video Crowd Counting"** (AAAI, arXiv 2104.13946). Uses motion cues to guide non-local spatiotemporal attention for video counting — directly relevant to your Section 2.3 motion-branch-informs-density-branch idea. **Tier 2 — consider reproducing components** if your basic CSRNet+flow combination underperforms.

**"Fast Video Crowd Counting with a Temporal Aware Network"** and **"STANet"-class methods** (temporal attention for DroneCrowd-style video). Tier 3 — awareness.

Liu, Salzmann & Fua, **People-Flows** (arXiv 1911.10782, 2012.00452, TPAMI 2021) — already Tier 1 in RESEARCH_MAP.md. Still the correct answer to your observed frame-jitter problem; nothing since has replaced it as the cleanest formulation of flow-constrained counting.

### Domain generalization for counting — very active right now, directly relevant to Phase 6

**Peng & Chan, "Single Domain Generalization for Crowd Counting" (MPCount), CVPR 2024** (arXiv 2403.09124). The current standard baseline for single-source domain generalization: a memory bank of density values + content error mask + attention consistency loss, trained on one domain and generalized to unseen ones. **Tier 1 — must read** before your Phase 6 cross-domain experiments; this is the method your JHU-CROWD++ weather-subset test will effectively be compared against in spirit.

**"SinCount: Fourier transform-based single domain generalization for crowd counting," Scientific Reports, April 2026.** Improves on MPCount using frequency-domain decomposition (high-frequency cues → density regression, low-frequency cues → region classification) to separate texture-level domain shift from content. Explicitly benchmarks against MPCount as the strongest baseline, with results stability-tested across multiple seeds (2024/2025/2026). **Tier 2 — read the comparison table**, useful as the most current published number for what "good" cross-domain MAE degradation looks like on JHU-CROWD++ subsets — exactly your E012 experiment.

**"CSCC: Cross-Scene Crowd Counting via Learning to Diversify for Domain Generalization," IEEE (2025/26).** Unifies domain generalization and domain adaptation in one framework. Tier 3 — awareness.

**"Crowd counting in domain generalization based on multi-scale attention and hierarchy level enhancement," Scientific Reports, Jan 2025.** Notes explicitly that domain generalization for counting is "still in the early stages of exploration" — useful citation for justifying why this is a legitimate open gap, not a solved problem you're re-doing. Tier 3.

**Du, Deng & Shi, "Domain-General Crowd Counting in Unseen Scenarios," AAAI 2023** (arXiv 2212.02573). Already Tier 1 in RESEARCH_MAP.md — still the most-cited entry point to this subfield as of 2025-26 follow-ups.

**Practical note for your project:** you do not need to implement MPCount or SinCount. You need to know they exist, know their reported degradation numbers, and use them as the calibration point for whether your own cross-domain MAE gap (Phase 6) is "typical for the field" or "unusually large" (which would itself be a finding worth investigating).

---

## 2. Optical flow / motion-based crowd analysis — directly relevant, very current

**Balachandra, D.S. et al., "Density Estimation and Crowd Counting," arXiv 2511.09723 (Nov 2025).** This paper independently proposes **event-driven frame sampling using Farnebäck optical flow** to reduce computational load while preserving crowd dynamics — nearly identical motion-branch philosophy to your architecture, published one month before your session. **Tier 1 — must read.** Compare their sampling strategy against your indicator-extraction pipeline; if their approach reduces compute cost without losing signal, it may improve your Kaggle-runtime budget.

**"Stampede detection and crowd analysis using CNN-LSTM and Farneback optical flow," Scientific Reports, 2026** (nature.com/articles/s41598-026-45262-1; also on PMC as PMC13201541). **This is the single most relevant and most recent competing paper you should read.** Published 2026, it:
- Explicitly reviews "evolution of stampede detection technologies (2009–2025)"
- Directly cites and critiques Cob-Parro et al. 2025 (your project-folder paper), noting their focus was "primarily on binary abnormality detection" without "fine-grained risk stratification"
- Cites **Altowairqi et al. (2026)**, who combine C3D + LSTM + attention for spatiotemporal anomaly detection (STAD) — a third, even more recent competing architecture worth knowing about
- Uses CNN-LSTM on dense optical flow, similar backbone philosophy to your motion branch

**Tier 1 — must read before finalizing your temporal model design.** This paper's own stated gap — "fine-grained risk stratification" rather than binary detection — is very close to your G1 research gap. Read it carefully to make sure your contribution is differentiated from theirs, not a re-derivation of it.

---

## 3. Crowd physics / dense-crowd dynamics — the field's actual frontier, and thin on CV crossover

**Gu, Guiselin, Bain, Zuriguel & Bartolo, "Emergence of collective oscillations in massive human crowds," Nature 638:112-119, 5 Feb 2025.** Already Tier 1 in RESEARCH_MAP.md. Confirmed still the most important recent crowd-physics result: chiral oscillatory states in dense crowds (San Fermín data), same signature retrospectively found at the 2010 Love Parade disaster onset.

**"Exploring Dense Crowd Dynamics: State of the Art and Emerging Paradigms," arXiv 2505.05826, May 2025.** A comprehensive 2025 review explicitly stating that "physical interactions and contact forces fundamentally shape the dynamics of the crowd" and that psychological-physical interplay modeling is "rapidly evolving" but immature. **Tier 1 — must read.** This is your best single source for the current state of crowd-physics modelling and its (thin) intersection with computer vision — cites Bottinelli & Silverberg 2018 on forecasting high-density collective motion from spatiotemporal fluctuations, which is directly relevant to your Phase 5 forecasting task.

**Bottinelli, A. & Silverberg, J.L., "Can high-density human collective motion be forecasted by spatiotemporal fluctuations?" arXiv:1809.07875 (2018), cited extensively by the 2025 review above.** Older paper, but newly relevant because the 2025 review re-surfaces it as a foundational forecasting reference. **Tier 2 — read.** This is arguably the closest prior work to your exact Phase 5 research question — forecasting collective crowd state from fluctuation statistics — predating CrowdMAC by six years from the physics side rather than the CV side. Worth explicitly positioning your project relative to this.

**"Forward propagation of a push through a row of people," arXiv 2503.19104 (2025).** Studies how a physical push propagates through a queued crowd — mechanistic modelling of exactly the kind of localized force events that can trigger crowd crushes (cites the 2022 Seoul Halloween crowd crush and the Love Parade). Tier 3 — awareness; useful if you want a citation for why localized disturbances matter even in seemingly stable crowds.

**Fujii, Hachiuma & Saito, "CrowdMAC: Masked Crowd Density Completion for Robust Crowd Density Forecasting," WACV 2025** (arXiv 2407.14725). Already Tier 1 in RESEARCH_MAP.md — confirmed current. Notably explicit that crowd density forecasting "highly depends on the accuracy of the upstream module" (i.e., detection/density estimation errors propagate into forecasts) — this is a direct citation supporting your Phase 4 argument that detection-vs-density choice matters for everything downstream. Benchmarked on SDD, ETH-UCY, inD, JRDB, VSCrowd, FDST, CroHD — note VSCrowd and FDST and CroHD are the same video datasets recommended in your ARCHITECTURE.md Section 3.1.

---

## 4. Video anomaly detection — very active, but confirm the semantic-mismatch caution still holds

**Vision-language anomaly detection is the dominant 2025-2026 direction.** Representative recent results on UCF-Crime (AUC%): Sultani et al. baseline ~82.4 (2018) → RTFM 84.3 → MGFN 87.0 → CLIP-TSA 87.6 → VERA ~88+ (2024/25) → **ASK-HINT, WACV 2026, 89.83%** (arXiv 2510.02155) → a 2026 state-space temporal-reasoning VLM framework reporting 84.71% AUC on UCF-Crime and 97.58% on ShanghaiTech (sciencedirect.com, June 2026).

**ASK-HINT (WACV 2026)** — training-free, fine-grained VLM prompting, current best training-free UCF-Crime result. Tier 3 — awareness only; explicitly the wrong semantic level for your project (see below).

**Vision–language model with selective state-space temporal reasoning (ScienceDirect, June 2026)** — efficient VLM-based VAD, 94% fewer parameters in the temporal module than attention baselines. Tier 3 — interesting for compute-efficiency ideas if you ever add a semantic anomaly cross-check, but not a core component.

**Important caution, reconfirmed by this search:** every one of these current papers evaluates on UCF-Crime or XD-Violence, whose "anomalies" are semantic events — robbery, arson, explosion, fighting. None of them evaluate on a crowd-crush or crowd-turbulence construct. The field's VAD research direction in 2025-2026 (VLM prompting, state-space efficiency) is real and active, but it is **answering a different question than yours.** RESEARCH_MAP.md Section 2.4's caution stands unchanged by this newer literature — if anything, the 2025-2026 VAD papers make the semantic/physical distinction sharper, since they're increasingly about recognizing *what kind of crime* occurred, which is even further from "is this crowd becoming dangerously compressed."

**Use case for your project:** cite this VAD literature in your related-work section to explicitly explain why you did *not* adopt a VLM/semantic-anomaly approach, rather than to adopt it.

---

## 5. Small-object detection / SAHI — stable, no major 2025-2026 replacement found

No paper surfaced that supersedes SAHI (Akyon et al. 2022, arXiv 2202.06934) as the standard training-free tiled-inference technique for small-object recall. It remains current practice as of 2025-2026 aerial/small-object benchmarks. **Tier 2 — implement, unchanged from RESEARCH_MAP.md.** If you want to check for a 2025-26 successor, search "slicing aided hyper inference 2026" directly before Phase 1, since this is an active area and something newer may have appeared since this session.

---

## 6. Systems-level / applied crowd-safety papers — context, not benchmarks

**"Context-Aware Crowd Management in Smart Cities: A Scenario-Driven Systematic Review of Sensing, Prediction, and Intervention," Applied Sciences (MDPI), 2026** (doi:10.3390/app16147342). A PRISMA-protocol systematic review of 107 empirical studies (2020–2026), organizing the field into a Sensing–Prediction–Intervention–Feedback closed loop. **Tier 2 — read the framework, not for implementation.** Useful for your thesis's related-work section as a citation for "the field has moved toward closed-loop systems" and for positioning where your project's scope (sensing + prediction, explicitly not intervention) sits within that taxonomy.

**Shaheen & Sharma, "Smart Mobility and Planning Framework for Religious Tourism Management during Pitra Paksha Mela in Gaya, Bihar," Int. J. Traffic Mgmt. in Transportation Networks, 2026** (already in your project folder). This is an urban-planning/policy paper, not a CV paper — it proposes IoT/GPS/wearables/mobile-app infrastructure for pilgrim crowd management, qualitative in methodology, no CV model, no dataset, no evaluation metrics. **Use only for domain context** (scale of Indian religious gatherings, real infrastructure constraints, precedent technologies used at Kumbh/Hajj/Tirumala) in your introduction. Do not treat as a technical baseline or cite it as CV-related work.

**Altowairqi et al. (2026)**, cited within the Scientific Reports 2026 stampede paper above — C3D + LSTM + attention for spatiotemporal anomaly detection. I could not independently verify full bibliographic details in this session; if you cite it, verify the primary source directly before submission.

---

## 7. Updated priority reading order for 2025-2026 material specifically

If you only have time to read five *new* papers beyond what RESEARCH_MAP.md already flagged as Tier 1, read these in this order:

1. **Stampede detection and crowd analysis using CNN-LSTM and Farneback optical flow** (Sci Rep, 2026) — your most current direct competitor; explicitly names the "fine-grained risk stratification" gap you're targeting.
2. **Exploring Dense Crowd Dynamics: State of the Art and Emerging Paradigms** (arXiv 2505.05826, 2025) — the best current synthesis of the physics side, and points you to Bottinelli & Silverberg's forecasting work.
3. **Shu et al., Adapting Lightweight Image-based Counting Models for Video Crowd Counting** (CVPR 2026) — most current, most directly on-topic for your Phase 2 video-adaptation of CSRNet.
4. **Peng & Chan, MPCount** (CVPR 2024, still the standing domain-generalization baseline) — read before Phase 6, so you know what "good" cross-domain degradation looks like.
5. **Balachandra et al., Density Estimation and Crowd Counting** (arXiv 2511.09723, Nov 2025) — nearly identical motion-branch philosophy to yours; useful sanity check and possible compute optimization for your Farnebäck pipeline.

---

## 8. What did NOT change from RESEARCH_MAP.md

This search did not surface anything that overturns the core architectural conclusions already documented:
- No GNN-based crowd-safety paper appeared that solves the individual-identity problem at high density — D004 (exclude GNNs) stands.
- No public benchmark for validated crowd-crush/stampede onset labels appeared — Section 4.4's dataset-labelling problem is unchanged; if anything the 2026 stampede paper's explicit critique of prior binary-classification framing reinforces it.
- Crowd pressure P = ρ·Var(v) remains uncontested and unimplemented as a CV pipeline output — no 2025-2026 paper operationalizes Helbing's exact indicator end-to-end from video, confirming G1 is still open.
- CSRNet remains a legitimate, reproducible entry point; the field has moved past it in raw MAE but not in reproducibility or pedagogical clarity for a first implementation.

---

*This file should be re-searched roughly every 4–6 weeks during the active project period, since this is an area with monthly publication turnover (note several sources above are dated June–August 2026, i.e., within weeks of this session). Update DECISION_LOG.md if a new paper invalidates an existing architectural choice.*

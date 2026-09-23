# ARCHITECTURE.md
## Crowd Intelligence & Risk Indicator Estimation System

**Status:** Living document. Every component has a reason. If the reason becomes invalid (experiment fails, paper refutes it), update this file and log the change in DECISION_LOG.md.

**Core research question:**
> When individual detection becomes unreliable, can physically-grounded crowd-risk indicators — local density, speed variance, and crowd pressure ρ·Var(v) — still be estimated reliably from video? How do detection-based and density-based estimators degrade differently as density rises? How much lead time does the resulting risk signal provide?

---

## 0. Design principles

These are constraints that override feature requests.

1. **Every component has an experiment that motivated it.** Nothing enters this architecture because it sounded good. The DECISION_LOG records the failed experiment → hypothesis → added component chain for each module.

2. **Detection is a baseline, not the backbone.** Person detection fails at the density levels where the safety problem exists. The backbone of the system is the density branch and the motion branch, which operate on fields rather than on identities.

3. **Outputs are fields and indicators, not events.** The system does not output "stampede detected." It outputs: estimated local density D̂(x,y,t), estimated velocity field v̂(x,y,t), derived indicators (speed variance, divergence, crowd pressure P), and a calibrated risk score with confidence interval. All of these are continuously defined and measurable. An alert is a downstream threshold on these quantities, not a classification.

4. **Physical thresholds ground every claim.** Risk claims reference Helbing et al. (2007): stop-and-go waves emerge at ρ ≈ 2–4 ped/m², turbulence onset at ρ ≈ 4–7 ped/m², crowd pressure P = ρ·Var(v) > 0.02/s² preceded the 2006 Hajj disaster. We are not inventing our own risk labels.

5. **Kaggle persistence discipline.** Every expensive computation checkpoint to disk immediately after completion. No notebook depends on a live variable surviving a restart.

6. **Uncertainty is a first-class output.** A safety system with no confidence estimate is more dangerous than no system at all.

---

## 1. System overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         VIDEO INPUT                              │
│     (CCTV frame sequence, any resolution, ≥15fps)                │
└────────────────────────┬─────────────────────────────────────────┘
                         │
           ┌─────────────▼──────────────┐
           │     PREPROCESSING          │
           │  resize · grayscale copy   │
           │  face blur (privacy)       │
           │  perspective zone map      │
           └──────┬──────────┬──────────┘
                  │          │
     ┌────────────▼──┐   ┌───▼────────────────┐
     │  DENSITY      │   │  MOTION             │
     │  BRANCH       │   │  BRANCH             │
     │               │   │                     │
     │  CSRNet /     │   │  Farnebäck dense    │
     │  P2PNet       │   │  optical flow       │
     │  → D̂(x,y,t)  │   │  → v̂(x,y,t)        │
     └──────┬────────┘   └───┬─────────────────┘
            │               │
            └───────┬───────┘
                    │
     ┌──────────────▼────────────────────────┐
     │      INDICATOR EXTRACTION             │
     │                                       │
     │  per grid-cell (configurable size):   │
     │  ρ(z,t)      = mean D̂ in cell z      │
     │  |v|(z,t)    = mean flow magnitude    │
     │  Var_v(z,t)  = variance of speed      │
     │  P(z,t)      = ρ · Var_v (pressure)  │
     │  H(z,t)      = directional entropy    │
     │  div(z,t)    = flow divergence        │
     └──────────────┬────────────────────────┘
                    │
     ┌──────────────▼────────────────────────┐
     │      TEMPORAL MODEL                   │
     │                                       │
     │  gradient boosting (baseline)         │
     │  → ConvLSTM (if baseline signal ok)   │
     │                                       │
     │  input:  indicator time series        │
     │          window W seconds             │
     │  output: risk score R(z, t+H)         │
     │          uncertainty interval         │
     └──────────────┬────────────────────────┘
                    │
     ┌──────────────▼────────────────────────┐
     │      ALERT & EXPLANATION              │
     │                                       │
     │  advisory / warning / critical tiers  │
     │  feature attribution per alert        │
     │  lead-time logging                    │
     └───────────────────────────────────────┘
```

**Also present as a parallel path (baseline/cross-check only):**
```
VIDEO → YOLOv8 + SAHI → detection count N_det(t)
```
Detection count is logged alongside density-derived count N_dens(t). The gap between them, binned by ρ, is the core experimental result.

---

## 2. Module specifications

### 2.1 Preprocessing

**Purpose:** normalise inputs so downstream models are not surprised by resolution, aspect ratio, or identifiable faces.

**Operations, in order:**
1. Frame extraction at the original fps (do not drop frames).
2. Resize to a fixed inference resolution. For the density branch: 512×512 (configurable). For the motion branch: same as density, unless flow computation is too slow — then 320×240 with bilinear upscaling of the output.
3. Grayscale copy for motion branch (Farnebäck operates on grayscale).
4. Face blurring before any storage: OpenCV face detector + Gaussian blur on detected faces. Required for any footage of real people you did not yourself record under consent.
5. Perspective zone map (static, set up once per camera): manually annotate 4 ground-plane reference points with known metric distances → compute homography matrix H. Used downstream to convert image-space density to persons/m². If a homography is unavailable, all metric density claims are marked "image-space only" and no safety threshold claims are made.

**Outputs saved to disk:** `preprocessed/frame_{n:06d}.jpg`, `camera_config.json` (homography matrix, zone boundaries, fps).

---

### 2.2 Density branch

**Why this branch exists:** detection-based counting (YOLO) breaks above moderate density due to occlusion and scale. Density-map regression decouples counting from detection by asking "how dense is this region?" instead of "where is each person?"

**What it produces:** D̂(x,y,t), a 2D map where each cell value approximates the expected number of people in that cell. ∑ D̂ = N̂ (total count estimate).

**Baseline model: CSRNet**
- Backbone: VGG-16 frontend (conv1_1 through pool3), pretrained on ImageNet.
- Backend: 6 dilated convolutional layers (dilation rates 2,2,2,2,2,2). Dilation chosen because it expands the receptive field without losing spatial resolution — critical for density maps.
- Loss: MSE between predicted density map and ground-truth density map. Ground truth is generated by convolving head point annotations with a fixed Gaussian kernel (σ = 15 pixels on ShanghaiTech B; σ adaptive on QNRF).
- Training: ShanghaiTech B first (simpler), then A. Saved checkpoints after every epoch.
- Output resolution: same as input after nearest-neighbour upsampling if needed.
- Why CSRNet not something fancier: it is reproducible with published code, has a verified MAE on a public benchmark (68.2 on SHA, 10.6 on SHB), and gives a clean failure mode to measure against. If it doesn't work, the failure is informative. A foundation-model wrapper gives you a black box.

**Upgrade path (only if CSRNet fails or is insufficient):**
- P2PNet: if localization matters (you need head points, not just density), P2PNet predicts both.
- DM-Count / optimal-transport loss: if the MSE loss produces blurry densities that fail at high ρ, swap the loss, not the backbone.
- People-Flows (Liu et al. 2020): if temporal jitter in N̂(t) is still too large after smoothing, this replaces the per-frame prediction with a flow-constrained prediction that enforces conservation of people. This is the correct fix for the ±16 person frame-to-frame jump observed in the YOLO baseline.

**What this branch does NOT do:** it does not track individuals, it does not detect events, it does not predict the future. It produces a single quantity per frame: D̂(x,y,t).

---

### 2.3 Motion branch

**Why this branch exists:** density alone is insufficient to identify dangerous crowd states. Helbing et al. (2007) showed empirically that neither density nor velocity field alone predicts the onset of crowd turbulence. The decisive quantity requires both: P = ρ · Var(v). The motion branch produces v̂(x,y,t).

**Model: Farnebäck dense optical flow (classical)**
- Algorithm: `cv2.calcOpticalFlowFarneback()` with standard parameters (pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2).
- Input: consecutive grayscale frames (t, t+1).
- Output: flow field F(x,y) = (u,v) in pixels/frame. Magnitude |F| = sqrt(u²+v²) gives apparent speed.
- Why Farnebäck and not RAFT (deep flow): Farnebäck runs on CPU, requires no training, has no domain shift, and is what Helbing's group used on real crowd footage. RAFT is architecturally superior but requires GPU, and its advantage is on general optical flow benchmarks — we do not have a crowd-specific flow benchmark to verify whether RAFT is actually better here. Use Farnebäck first; if the resulting speed-variance maps are too noisy, try RAFT as an upgrade experiment.

**Metric conversion:** flow magnitude in pixels/frame × fps × (metres per pixel from homography) = speed in m/s. Without homography, speed is in image-space units. Mark clearly in outputs.

**What this branch does NOT do:** it does not identify which person is which. It does not track. It computes apparent motion of the image, which in a crowd scene is predominantly pedestrian motion, but also includes camera shake, shadows, and background clutter. Camera shake mitigation: subtract global frame mean before computing per-cell statistics if camera is not static.

---

### 2.4 Detection branch (baseline / cross-check only)

**Why it exists:** not as the system backbone, but as:
1. A baseline to measure against.
2. A sparse-regime cross-check: in image regions where ρ̂ < 1 ped/m² (low density), detection count should agree with density-derived count within ±20%. Large disagreement flags a density estimator failure.
3. The motivation for SAHI, which is the first experiment.

**Model: YOLOv8-x + SAHI**
- YOLOv8-x: largest variant, best AP_small.
- SAHI: slice input into overlapping 640×640 tiles, run inference on each, merge with NMS. No retraining required. Configured: slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2.
- Why SAHI: SAHI increases AP by 6.8–14.5 percentage points on small-object benchmarks (Akyon et al. 2022) with zero training cost. It is the cheapest possible answer to the small-person recall failure already observed.

**Output:** N_det(t), per-frame detection count. Saved to `outputs/detection_counts.csv`. Never used as ground truth for evaluation.

---

### 2.5 Indicator extraction

**Purpose:** convert the per-pixel fields D̂ and v̂ into per-zone (grid-cell) scalar indicators that can be modelled temporally and thresholded against published safety criteria.

**Grid:** default 32×32 cells over the image. Configurable. Each cell z has spatial extent ~16×16 pixels at 512×512 input resolution. Homography maps each cell to an approximate ground-plane area.

**Computed per cell per frame:**

| Indicator | Formula | Physical meaning | Threshold reference |
|-----------|---------|-----------------|---------------------|
| ρ̂(z,t) | mean(D̂) in cell z | local density (ped/m² after homography) | >4: elevated; >7: critical (Helbing 2007) |
| \|v\|(z,t) | mean(\|F\|) in cell z | mean crowd speed | Drops sharply at congestion onset |
| Var_v(z,t) | var(\|F\|) in cell z | speed variance | Rises before turbulence onset |
| P(z,t) | ρ̂ · Var_v | **crowd pressure** | >0.02/s² threshold in Helbing (2007) |
| H(z,t) | Shannon entropy of flow angle histogram | directional disorder | Higher → more turbulent motion |
| div(z,t) | ∂u/∂x + ∂v/∂y (finite difference) | convergence (<0) or divergence (>0) | Negative div = crowd compression |
| dρ/dt(z,t) | (ρ̂(z,t) − ρ̂(z,t−1)) / Δt | density growth rate | Sustained positive = accumulation |

**Output saved:** `outputs/indicators/indicators_{t:08d}.npy` (32×32×7 array, one per frame). Also a flattened CSV for the temporal model input.

**Metric density note:** ρ̂ is in image-space units (persons per cell) unless a homography is available. If no homography: report in image-space only, draw no safety-threshold conclusions. This is an honest limitation, not a bug.

---

### 2.6 Temporal model

**Purpose:** given a window of W seconds of indicator time series, estimate the risk state at t+H seconds. This is the forecasting task (T8 from the research map), where H is the prediction horizon.

**Stage 1 (Baseline): XGBoost regressor**
- Input: flattened indicator vector over W=30 seconds (30×fps frames × 7 indicators × n_cells), reduced by spatial mean and max across cells → ~50 input features.
- Output: risk score R̂(t+H) ∈ [0,1] for each zone.
- Why XGBoost first: Drishti (2606.05185v1.pdf) showed XGBoost on 8 crowd features achieves 8.3% MAPE at 5-minute horizon with no deep architecture. If the extracted indicators contain signal, XGBoost will find it. If XGBoost finds nothing, the problem is in the features, not the model — and adding a ConvLSTM will not fix bad features.
- Training: requires some sequences with labelled risk states (see Section 3 on training data).

**Stage 2 (if XGBoost shows signal): ConvLSTM over density maps**
- Input: stack of T=10 density maps D̂(x,y,t-T:t), each 32×32.
- Architecture: 2-layer ConvLSTM → 1×1 conv → sigmoid per cell.
- Output: predicted density map D̂(x,y,t+H) and risk map R̂(x,y,t+H).
- Why ConvLSTM: preserves spatial structure of density evolution. A flat LSTM loses spatial information. A transformer would be better but requires more data.
- Evaluation metric: Jensen-Shannon divergence between predicted and ground-truth density map (following CrowdMAC, WACV 2025).

**Uncertainty:** wrap the final risk score with conformal prediction. Using a calibration split, compute the residuals of Stage 1 predictions. Construct prediction intervals at 90% coverage. Report: R̂(t+H) ± δ. If δ is large (model uncertain), the alert tier is downgraded.

---

### 2.7 Alert and explanation

**Alert tiers (thresholds provisional, to be calibrated experimentally):**

| Tier | Condition | Meaning |
|------|-----------|---------|
| Normal | All indicators below baseline thresholds | No action |
| Advisory | ρ > 2 ped/m² sustained >30s, OR P rising | Monitor closely |
| Warning | ρ > 4 ped/m², OR P > 0.01/s², OR R̂ > 0.6 | Intervention recommended |
| Critical | ρ > 6 ped/m², OR P > 0.02/s², OR R̂ > 0.8 | Immediate action required |

**Explanation (per alert):** which indicator(s) triggered the tier, the trend direction (rising/falling), and the time the indicator first exceeded its threshold (this gives a lead-time estimate).

**Lead time logging:** whenever an alert is retrospectively validated (i.e., a ground-truth event occurred), record t_alert − t_event. This is the primary safety metric of the project.

**What is NOT in the alert:** no SMS, no public broadcast, no intervention recommendations. This is a research system. The output is a structured risk report that a human operator would act on. Anything beyond that is scope creep.

---

## 3. Training data strategy

### 3.1 What you need and where to get it

| Data need | Source | Used for |
|-----------|--------|---------|
| Head-point annotations for density training | ShanghaiTech B (400 train images) | CSRNet training |
| Denser / more diverse counting | UCF-QNRF or JHU-CROWD++ | CSRNet generalisation |
| Temporal video with per-frame annotations | FDST (100 videos) | Temporal consistency evaluation |
| Video with trajectories | DroneCrowd (112 clips, 4.8M head annotations) | Flow field validation |
| Cross-domain test | JHU-CROWD++ (weather/illumination subsets) | Robustness characterisation |
| Your own video | Self-recorded or the existing 21s test video | Domain-shift test; YOLO baseline already run |

**Annotate 30 frames of your own test video manually** (head-point count, ~5 minutes per frame with a labelling tool). This gives ground truth for every comparison against detection and density estimates.

### 3.2 Ground-truth density map generation

For a frame with head annotations {(xᵢ,yᵢ)}:

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def make_density_map(image_shape, points, sigma=15):
    dm = np.zeros(image_shape, dtype=np.float32)
    for (x, y) in points:
        if 0 <= int(y) < image_shape[0] and 0 <= int(x) < image_shape[1]:
            dm[int(y), int(x)] += 1.0
    dm = gaussian_filter(dm, sigma=sigma)
    return dm  # sum ≈ len(points)
```

σ=15 is a heuristic for ShanghaiTech B. Adaptive σ based on k-nearest-neighbour distance of head points is better (used in CSRNet for Part A). The choice of σ is an assumption; report it.

### 3.3 Risk labels — the honest version

For the temporal model you need some notion of "what was the risk at time t?" There are three honest approaches, in order of increasing strength:

1. **Indicator-threshold labels (zero annotation cost):** label a frame as "elevated" if any cell exceeds the published Helbing thresholds. This is a physical definition, not a human judgement. It can be applied to any video. The limitation: you are training to predict your own indicator, not an external ground truth.

2. **Retrospective congestion labels from visual inspection:** watch your own video. Mark the frames where you can visually observe congestion (slow movement, dense packing, visible stop-and-go). Label those as "elevated." Labelling 300 frames takes ~2 hours. This is the strategy used in GBA Stampedes.

3. **Public anomaly dataset labels:** UCSD Pedestrian, Avenue, and ShanghaiTech Campus have frame-level normal/abnormal labels. The "abnormal" events are not crowd crushes, but they give you a held-out evaluation dataset with external labels. Use them to check whether your indicators correlate with human-labelled anomaly events.

**Do not:** invent a "stampede" label by downloading random internet videos and calling some of them stampedes. This is what `Presentation.pdf` did and it is not a valid dataset.

---

## 4. Evaluation protocol

### 4.1 Counting evaluation

| Metric | Computed on | Ground truth source |
|--------|-------------|---------------------|
| MAE | ShanghaiTech B test set (316 images) | Published annotations |
| RMSE | Same | Same |
| N_det vs N_dens scatter | Your own 30 annotated frames | Manual head-point annotations |
| AP_small | Annotated frames, YOLO output | Manual bounding boxes |

Report N_det(t) and N_dens(t) plotted on the same axes over your 21s test video. The difference curve is the flickering artifact. Its magnitude is the first result.

### 4.2 Density evaluation

| Metric | What it measures |
|--------|-----------------|
| MAE / RMSE on count | Quality of ∑D̂ vs ground truth count |
| SSIM of density map | Spatial accuracy of density prediction |
| Density-by-density scatter | Does model under/overestimate at different densities? (bin by ρ_true) |

### 4.3 Flow and indicator evaluation

No external ground truth is available for flow fields in crowd scenes. Validate indirectly:
- Check that speed estimate drops at frames you visually identify as slow/stopped.
- Check that crowd pressure P rises before visually observable congestion in your video.
- Compare Farnebäck vs RAFT on a 100-frame subsequence (if GPU available): EPE on a standard optical flow benchmark (Sintel) is not the right measure here; correlation of P-values between the two methods is.

### 4.4 Forecasting evaluation

| Metric | What it measures |
|--------|-----------------|
| MAPE on density | Predictive accuracy of density forecast (follow CrowdMAC) |
| Jensen-Shannon divergence | Spatial accuracy of density forecast |
| Lead time distribution | Time between first exceeding R̂ threshold and observable onset |
| False alarm rate | Alerts per hour in clearly normal periods |
| Coverage at 90% | Does the conformal interval actually contain the true value 90% of the time? |

### 4.5 Cross-domain evaluation

Train: ShanghaiTech B. Evaluate: JHU-CROWD++ (rain subset, night subset). Report MAE separately for each condition. The degradation curve is the generalization result.

---

## 5. What is deliberately excluded

| Feature | Why excluded |
|---------|-------------|
| Multi-object tracking | Requires individual identity; fails in dense regime |
| GNN interaction modelling | Requires reliable per-person nodes; fails above moderate density |
| SMS / public broadcast | Engineering, not research; out of scope |
| Video foundation model fine-tuning | Kaggle compute insufficient for training; feature extraction acceptable |
| "Stampede classification" as a binary task | No valid labelled dataset exists |
| Multi-camera fusion | Single-camera baseline first; multi-camera is a later extension if time permits |

---

## 6. File and directory structure

```
crowd-intelligence/
│
├── README.md
├── ARCHITECTURE.md          ← this file
├── ROADMAP.md               ← phase-level milestones
├── RESEARCH_MAP.md          ← field map (Phase 0)
├── DECISION_LOG.md          ← every architectural decision + reason
├── EXPERIMENTS.md           ← one entry per experiment: Q / config / result / conclusion
├── PROJECT_STATE.md         ← current status in one page
│
├── data/
│   ├── shanghaitech/        ← do NOT commit; store on Kaggle dataset
│   ├── ucf_qnrf/
│   ├── jhu_crowd/
│   ├── your_video/
│   │   ├── frames/          ← extracted at full res
│   │   ├── annotations/     ← head_points_frame_XXXXXX.csv (30 frames)
│   │   └── camera_config.json
│   └── processed/           ← density maps, flow fields, indicators (npy files)
│
├── notebooks/
│   ├── 01_yolo_sahi_baseline.ipynb
│   ├── 02_density_map_generation.ipynb
│   ├── 03_csrnet_train_shb.ipynb
│   ├── 04_csrnet_eval.ipynb
│   ├── 05_optical_flow_indicators.ipynb
│   ├── 06_crowd_pressure_visualisation.ipynb
│   ├── 07_detection_vs_density_comparison.ipynb
│   ├── 08_temporal_model_baseline.ipynb
│   └── 09_cross_domain_eval.ipynb
│
├── src/
│   ├── data/
│   │   ├── density_map.py        ← make_density_map(), adaptive sigma
│   │   ├── video_reader.py       ← frame extraction, face blur
│   │   └── perspective.py        ← homography estimation, metric conversion
│   ├── models/
│   │   ├── csrnet.py             ← model definition + training loop
│   │   ├── optical_flow.py       ← Farnebäck wrapper + per-cell stats
│   │   └── temporal_model.py     ← XGBoost baseline + ConvLSTM
│   ├── indicators/
│   │   ├── extract.py            ← all 7 indicators from D̂ and v̂
│   │   └── crowd_pressure.py     ← P = rho * Var_v, thresholds
│   ├── evaluation/
│   │   ├── metrics.py            ← MAE, RMSE, JS divergence, AUC
│   │   ├── conformal.py          ← conformal prediction intervals
│   │   └── lead_time.py          ← lead-time logging and analysis
│   └── visualisation/
│       ├── density_overlay.py
│       ├── flow_hsv.py
│       └── risk_heatmap.py
│
├── configs/
│   ├── csrnet_shb.yaml
│   ├── csrnet_sha.yaml
│   ├── sahi_config.yaml
│   └── indicator_thresholds.yaml
│
├── checkpoints/             ← model weights; .gitignore large files
│   ├── csrnet_shb_best.pt
│   └── csrnet_sha_best.pt
│
└── outputs/
    ├── detection_counts.csv
    ├── density_counts.csv
    ├── indicators/           ← indicators_XXXXXXXX.npy
    └── experiments/          ← one folder per experiment ID
```

**Git policy:** commit source code and configs. Do NOT commit datasets, checkpoints >10MB, or raw video. Use Kaggle datasets for large files; store checkpoint paths in configs.

---

## 7. DECISION_LOG seed entries

These are the first entries for the log. Every future architectural decision gets an entry in the same format.

```
D001 | 2026-08-30 | Use density branch as primary backbone, not detection
Reason: YOLO detection count changed in 79.88% of frame transitions at 60fps
with max ΔN = ±16. This is detector jitter, not real crowd dynamics.
Density-map regression operates on fields and is not subject to
single-person detection failures.
Evidence: Phase-0 YOLO baseline experiment.
Alternatives rejected: tracking (inherits detection failures), detection+smoothing
(smoothing hides real dynamics and creates latency).

D002 | 2026-08-30 | Use Farnebäck optical flow, not a deep flow model
Reason: Farnebäck runs on CPU with no training and no domain shift.
Deep models (RAFT) are better on standard benchmarks but we have no
crowd-specific flow ground truth to verify they are better here.
Upgrade path: if Farnebäck indicators are too noisy, run RAFT on a
100-frame sample and compare P-value correlation.
Evidence: Cob-Parro et al. 2025 used Farnebäck successfully for stampede detection.

D003 | 2026-08-30 | Crowd pressure P = ρ·Var(v) as primary risk indicator
Reason: Helbing et al. (2007, Phys. Rev. E) showed empirically that
neither density alone nor velocity alone predicts turbulence onset.
Their P threshold (0.02/s²) preceded the 2006 Hajj disaster by ≥10 minutes.
Published threshold → falsifiable claim → honest evaluation.
Evidence: arXiv physics/0701203; Helbing "Dynamics of Crowd Disasters."

D004 | 2026-08-30 | Exclude GNNs from architecture
Reason: GNN nodes = persons. Person identity is unreliable above moderate density,
which is exactly when safety matters. The architecture cannot be built on
unreliable inputs.
Upgrade path: if a future experiment produces reliable head localisation above
6 ped/m², revisit.

D005 | 2026-08-30 | XGBoost before ConvLSTM for temporal modelling
Reason: Drishti showed XGBoost on 8 crowd features → 8.3% MAPE at 5-minute
horizon. If the extracted indicators contain no signal, a ConvLSTM will
also find nothing, and the problem is in features, not architecture.
Test: XGBoost on indicator features first. If R² > 0.4, proceed to ConvLSTM.
```

---

*This document is updated when experiments contradict its assumptions. Do not silently change a component; always add a DECISION_LOG entry explaining why.*

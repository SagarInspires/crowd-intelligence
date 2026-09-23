# Kaggle Protocol

Both blockers you raised — "no GPU, everything on Kaggle" and "don't want to
re-run everything after a long gap" — have one root cause: **Kaggle notebooks
have no persistent identity.** Every time you open a notebook to edit it, or
every time "Save & Run All" fires, you get a fresh container with nothing in
it except what the notebook cells recreate. If your code and your cached
results only live inside that one notebook, you're stuck re-running
everything, every time.

The fix is to stop treating the Kaggle notebook as storage. It is compute
only. Two things live outside it:

1. **Code → GitHub.** The notebook's first cell clones your repo. This also
   solves blocker (b): your commit history *is* your per-day, per-task work
   log — no separate tracking system needed.
2. **Anything expensive → a Kaggle Dataset.** Checkpoints, processed
   features, annotation CSVs, density maps — published as a versioned
   Kaggle Dataset, attached as input to future notebooks. Combined with the
   `cache_or_compute` pattern below, this means "Run All" *skips* anything
   already computed instead of redoing it.

---

## 1. The notebook header (paste this into cell 1 of every notebook)

```python
# ============================================================
# CELL 1 — always run first, always cheap, always safe to re-run
# ============================================================
import os, subprocess

REPO_URL = "https://github.com/<your-username>/crowd-intelligence.git"
REPO_DIR = "/kaggle/working/crowd-intelligence"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

import sys
sys.path.insert(0, REPO_DIR)

# If this notebook has a Kaggle Dataset attached with prior outputs
# (checkpoints, processed features), point CACHE_DIR at it; else use
# /kaggle/working so this session's outputs get published at the end.
PRIOR_DATASET_DIR = "/kaggle/input/crowd-intel-artifacts"  # attach as input, rename per your dataset slug
CACHE_DIR = PRIOR_DATASET_DIR if os.path.exists(PRIOR_DATASET_DIR) else "/kaggle/working/artifacts"
os.makedirs("/kaggle/working/artifacts", exist_ok=True)
print("Repo ready at", REPO_DIR)
print("Reading cache from", CACHE_DIR)
```

Why this works: cloning/pulling ~50MB of code takes seconds, so this cell is
always safe inside "Run All" — you never lose work by re-running it, and it
always gives you your latest committed code, not whatever was pasted into
the notebook three weeks ago.

**Rule: no real logic lives in notebook cells.** Cells only call functions
imported from `src/`. If you write a function directly in a notebook cell,
copy it into `src/` and commit it before you close the session — otherwise
it's gone from GitHub's history and blocker (b) breaks.

---

## 2. `cache_or_compute` — never recompute what's already on disk

This is the actual fix for "don't want to run everything after a long gap."
Every expensive cell (density map generation, feature extraction, model
inference over a video) gets wrapped like this instead of running directly:

```python
from src.utils.kaggle_cache import cache_or_compute

density_maps = cache_or_compute(
    output_path=f"{CACHE_DIR}/density_maps/shanghai_a_train.npy",
    compute_fn=lambda: generate_density_maps(train_image_paths, sigma=15),
    loader_fn=np.load,
)
```

First run: no file at that path → `compute_fn` runs, result is saved, then
returned. Every run after that, for weeks: file exists → loaded from disk in
under a second, `compute_fn` never executes. "Save & Run All" after a
month-long gap now takes as long as your *slowest still-unfinished* step,
not the sum of every step you've ever done.

See `src/utils/kaggle_cache.py` for the implementation — it's a real,
importable module, not a snippet to paste around.

---

## 3. Resumable training (for anything that trains, not just computes once)

Density-branch training won't fit in one Kaggle session once you're past a
few epochs. Checkpoint every epoch, and resume from the latest checkpoint
automatically:

```python
from src.utils.kaggle_cache import save_checkpoint, load_latest_checkpoint

start_epoch, model, optimizer = load_latest_checkpoint(
    checkpoint_dir=f"{CACHE_DIR}/checkpoints/csrnet",
    model=model, optimizer=optimizer, device=device,
)

for epoch in range(start_epoch, N_EPOCHS):
    train_one_epoch(model, optimizer, train_loader)
    save_checkpoint(
        checkpoint_dir="/kaggle/working/artifacts/checkpoints/csrnet",
        epoch=epoch, model=model, optimizer=optimizer,
    )
```

`load_latest_checkpoint` finds the highest-epoch file automatically and
returns `epoch=0` with the fresh model if none exists yet, so the same cell
works identically on the very first run and on the tenth resume.

---

## 4. Publishing artifacts so they survive past this session

`/kaggle/working` is wiped when a notebook session ends unless you commit
it — and even a commit only keeps it attached to *that* notebook, not
available to others. To make checkpoints/features reusable across future
notebooks (and this is what `PRIOR_DATASET_DIR` above reads from):

1. In the notebook, after your session's work: **File → Save Version →
   Save & Run All (Commit)**. This snapshots `/kaggle/working` as notebook
   output.
2. Go to that committed version's Output tab → **"New Dataset"** (or, once
   a dataset slug exists, use the Kaggle CLI: `kaggle datasets version -p
   /kaggle/working/artifacts -m "epoch 12 checkpoint + shanghai density
   maps"`).
3. In your *next* notebook, **Add Input → your dataset** — it appears at
   `/kaggle/input/<slug>/`, which is what `PRIOR_DATASET_DIR` points to.

Do this at the end of every session where you produced something expensive
(a trained checkpoint, a finished density-map cache, extracted features).
Skipping it is the single most common way people "lose" Kaggle work.

---

## 5. Weekly quota discipline (~30 GPU-hrs/week, resets Saturday midnight UTC)

- Default every new notebook to **CPU** while writing/debugging code
  (dataloaders, annotation tooling, plotting). Only switch the accelerator
  to GPU/T4×2 for the actual training or inference cell, and switch back
  after.
- Keep a **compute notebook** (GPU, does training/inference, publishes a
  Dataset) separate from an **analysis notebook** (CPU only, reads the
  published Dataset, does plots/ablation tables/error analysis). You will
  re-run analysis notebooks constantly while writing up results — never pay
  GPU time for that.
- If a session is approaching the 9–12hr wall or the 60–90min idle timeout,
  checkpoint *before* you plan to step away, not after you notice you got
  disconnected.

---

## 6. Session start/end checklist

**Start of every Kaggle session:**
- [ ] Run the header cell (clone/pull + attach input Dataset)
- [ ] Confirm `CACHE_DIR` resolved to the Dataset, not an empty
      `/kaggle/working/artifacts`
- [ ] Check `DAILY_LOG.md` (pulled with the repo) for what "today" is
      supposed to be, per `ROADMAP.md`

**End of every Kaggle session:**
- [ ] Any new function written in a cell → moved into `src/`, committed,
      pushed
- [ ] Save & Run All (Commit) if you produced anything worth keeping
- [ ] `kaggle datasets version` if checkpoints/features changed
- [ ] Append one line to `DAILY_LOG.md` and push (see that file for format)
- [ ] Close/comment the corresponding GitHub Issue if the task is done

Pushing from inside Kaggle needs a GitHub token (Kaggle → Settings → your
account → generate a fine-grained PAT with repo write access, add it as a
Kaggle Secret named `GITHUB_TOKEN`, then `git remote set-url origin
https://<token>@github.com/<user>/crowd-intelligence.git` once per session
before `git push`). Alternatively, edit `DAILY_LOG.md` and small `src/`
changes locally/on GitHub's web editor when you're not inside Kaggle —
either path keeps the same history.

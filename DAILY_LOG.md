# Daily Log

This file is the per-day, task-wise record the two blockers asked for. It is
deliberately boring and mechanical — one entry per day you touch the
project, written in under 2 minutes, so it never becomes a chore you skip.

**Rule:** every entry ends with a "Next" line. When you come back after a
gap of any length, read the last entry's "Next" line before doing anything
else — that's how you avoid re-deriving where you were.

Pair this with GitHub Issues (`.github/ISSUE_TEMPLATE/experiment.md`): one
Issue per experiment/roadmap step for the *what and why*, this file for the
*day-by-day trace*. Reference the Issue number in the entry.

Format — copy this block for each new day:

```
## YYYY-MM-DD
- Worked on: <1-2 lines>
- Issue: #<n> (if applicable)
- Result: <what happened — a number, a plot, a "still running", a blocker>
- Kaggle GPU used today: <hrs, rough>
- Next: <the single next concrete action>
```

---

## 2026-09-24
- Worked on: scaffolded the `crowd-intelligence` repo — repo structure,
  `KAGGLE_PROTOCOL.md`, `src/utils/kaggle_cache.py`, Issue template, this
  log. Ported over `ARCHITECTURE.md`, `ROADMAP.md`, `PAPERS.md`,
  `RESEARCH_MAP.md` from earlier planning into `docs/`.
- Issue: —
- Result: repo scaffold complete, not yet pushed to GitHub.
- Kaggle GPU used today: 0 (no GPU work yet — this was setup)
- Next: push this repo to your own GitHub, then open your first Kaggle
  notebook, paste the header cell from `KAGGLE_PROTOCOL.md`, and start
  Day 1 (head-point annotation + density-map sanity check, per
  `docs/ROADMAP.md` Phase 0). Report back true-count mean/std vs YOLO's
  71.57 once you have them.

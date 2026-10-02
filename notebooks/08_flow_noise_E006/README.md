# Notebook 08: flow noise / velocity-variance study ("E006" in the Kaggle notebook name)

Kaggle notebook `08_flow_noise_E006` (Tesla T4, Internet on). One file per notebook cell in `cells/` (Cells 1-26; cell numbers follow the notebook).
Results and decisions: `docs/E006_RESULTS.md` (all findings) and `docs/E006_CLOSEOUT.md` (what goes into the report, limits, next notebook).

NOTE on IDs: `docs/ROADMAP.md` already uses E006 for "does crowd pressure P rise before congestion onset" and E007 for detection-vs-density.
This notebook's "E006" is only the Kaggle notebook label; rename it (e.g. "F001") in the repo if you want unique experiment IDs.

Data: TUB CrowdFlow (synthetic, rendered; five independent crowds). Not included here. Kaggle outputs (CSVs, figures, videos) are not committed.
Cells needing earlier cells in the same Kaggle session: 18 (helpers) is required by 19-26; Cell 2 (loaders) by most; Cell 3 (SEA-RAFT model) by anything calling real_predict.

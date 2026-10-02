# ===== E006 · Cell 7 · Leave-scene-out test: fit the gain on IM01, test on the NEW scenes =====
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---- settings (needs Cells 1-6 run in this same session) ----
NEW_SEQS    = ["IM03", "IM05"]      # the two new scenes (second Kaggle dataset)
FRAME_STEP  = 2                     # use every 2nd frame pair to save time (set 1 for all 60 pairs)
GATE_BIAS   = 10.0                  # pre-registered: held-out |bias| must be <= 10 % ...
GATE_NMAE   = 0.15                  # ... and the 32 px nMAE must be >= 15 % lower than raw
DATA_ROOT_1 = DATA_ROOT             # remember the first dataset (IM01, IM01_hDyn)


def find_second_root(seqs):
    """Purpose: find the folder (under /kaggle/input) that holds images/ and gt_flow/ for the NEW scenes.
    Walks at most 6 levels deep and returns the first folder whose gt_flow contains all `seqs`.
    Returns None if nothing is found."""
    for root, dirs, _ in os.walk("/kaggle/input"):
        if root.count("/") - 2 > 6:
            dirs[:] = []
        if "gt_flow" in dirs and "images" in dirs:
            if all(os.path.isdir(os.path.join(root, "gt_flow", s)) for s in seqs):
                return root
    return None


def run_new_scenes(seqs, scales, step):
    """Purpose: run SEA-RAFT on the new scenes and build the same per-patch table as Cell 5 (var_gt,
    var_pred, var_err, cross at each scale). Uses scale_rows and real_predict from earlier cells. The loader
    functions read the global DATA_ROOT, so it is pointed at the new dataset for the duration of the run."""
    global DATA_ROOT
    rows = []
    DATA_ROOT = DATA_ROOT_2
    try:
        for s in seqs:
            ks = list_flow_indices(s)[::step]
            for k in tqdm(ks, desc=f"E006 new scene {s}"):
                img0, img1 = load_frame(s, k), load_frame(s, k + 1)
                gt = load_gt_flow(s, k)
                inside = (load_mask(s, k) > 0) & np.isfinite(gt).all(axis=2)
                pred, _ = real_predict(s, k, img0, img1)
                for sc in scales:
                    rows += scale_rows(gt, pred, inside, sc, s, k)
    finally:
        DATA_ROOT = DATA_ROOT_1          # always restore the first dataset
    return pd.DataFrame(rows)


def fit_gain(train):
    """Purpose: the single multiplier a (true = a * measured), fitted through the origin on the TRAIN patches."""
    vp, vg = train.var_pred.values, train.var_gt.values
    return float((vg * vp).sum() / (vp ** 2).sum())


def leave_scene_out(all_df, scales):
    """Purpose: for every scale and every (train scenes -> test scene) pair, fit the gain on the train scenes and
    report held-out bias, nMAE and the improvement of nMAE over doing nothing (raw). Uses `score` from Cell 6."""
    plans = [("IM01", ["IM01"]), ("IM01+hDyn", ["IM01", "IM01_hDyn"]),
             ("IM03", ["IM03"]), ("IM05", ["IM05"])]
    rows = []
    for sc in scales:
        d = all_df[all_df.scale == sc]
        for name, tr_seqs in plans:
            train = d[d.seq.isin(tr_seqs)]
            if len(train) < 20:
                continue
            a = fit_gain(train)
            for te in sorted(set(d.seq) - set(tr_seqs)):
                test = d[d.seq == te]
                raw, cor = score(test, test.var_pred.values), score(test, a * test.var_pred.values)
                rows.append({"scale_px": sc, "fit_on": name, "test_on": te, "a": round(a, 3), "patches": len(test),
                             "raw_bias_%": raw["bias_%"], "gain_bias_%": cor["bias_%"],
                             "raw_nMAE": raw["nMAE"], "gain_nMAE": cor["nMAE"],
                             "nMAE_gain_vs_raw_%": round(100 * (raw["nMAE"] - cor["nMAE"]) / raw["nMAE"], 1)})
    return pd.DataFrame(rows)


# ---------------- run ----------------
DATA_ROOT_2 = find_second_root(NEW_SEQS)
print("first dataset :", DATA_ROOT_1)
print("second dataset:", DATA_ROOT_2)
if DATA_ROOT_2 is None:
    raise RuntimeError("New scenes not found under /kaggle/input. Attach the second dataset and run `!ls /kaggle/input`.")

df7 = run_new_scenes(NEW_SEQS, SCALES, FRAME_STEP)
df7.to_csv(OUT_DIR / "e006_scale_table_new_scenes.csv", index=False)
all_df = pd.concat([df5, df7], ignore_index=True)

pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 200)
print("\n--- bias vs scale for the NEW scenes (same columns as Cell 5) ---")
print(summarise_scales(df7).to_string(index=False))

res7 = leave_scene_out(all_df, EVAL_SCALES)
res7.to_csv(OUT_DIR / "e006_leave_scene_out.csv", index=False)
print("\n--- leave-scene-out: gain fitted on one set of scenes, tested on a different scene ---")
print(res7.to_string(index=False))

# pre-registered gate, judged on the transfer IM01 -> each new scene
g = res7[(res7.fit_on == "IM01") & (res7.test_on.isin(NEW_SEQS))]
print("\n--- pre-registered gate (fit IM01 -> test new scene) ---")
for _, r in g.iterrows():
    ok_bias = abs(r["gain_bias_%"]) <= GATE_BIAS
    ok_mae = (r["scale_px"] != 32) or (r["nMAE_gain_vs_raw_%"] >= 100 * GATE_NMAE)
    print(f"{r['test_on']} @ {int(r['scale_px']):>2} px : bias {r['gain_bias_%']:+6.1f}%  "
          f"nMAE gain-vs-raw {r['nMAE_gain_vs_raw_%']:+5.1f}%  ->  {'PASS' if ok_bias and ok_mae else 'FAIL'}")

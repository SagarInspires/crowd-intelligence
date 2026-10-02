# ===== E006 · Cell 10 · One gain across crowds? And is the person-vs-pixel gap caused by sampling or by patch size? =====
import numpy as np
import pandas as pd

# ---- settings (needs Cell 9 run in this session: dfp, tracks, best_off, TRAJ_SCENES, pairs_at, sample_at ...) ----
GROUPS        = {"IM01": ["IM01", "IM01_hDyn"], "IM03": ["IM03"], "IM05": ["IM05"]}   # IM01 and its dynamic-camera copy show the SAME crowd,
                                                                                    # so they are held out together
PATCH_SIZES   = [80, 120, 160, 240]      # patch sizes (px) for the sensitivity test
GATE_BIAS10   = 10.0                     # pre-registered: held-out |bias| <= 10 % on EVERY held-out scene


def fit_gain_traj(train):
    """Purpose: the single multiplier a (reference = a * SEA-RAFT), fitted through the origin on the TRAIN patches."""
    return float((train.var_traj * train.var_pred).sum() / (train.var_pred ** 2).sum())


def nmae(est, true):
    """Purpose: mean absolute error divided by the mean reference value (per-patch error, scale free)."""
    return float(np.abs(est - true).mean() / true.mean())


def leave_group_out(df):
    """Purpose: for each smoothing option and each group of scenes, fit the gain on the OTHER groups and test it on
    the held-out group (every scene of the group separately). Reports raw bias, corrected bias and per-patch error."""
    rows = []
    for sm, d in df.groupby("smooth"):
        for g, members in GROUPS.items():
            train, test_all = d[~d.seq.isin(members)], d[d.seq.isin(members)]
            a = fit_gain_traj(train)
            for seq, t in test_all.groupby("seq"):
                tot = t.var_traj.sum()
                rows.append({"smooth_px": sm, "held_out": seq, "gain_a": round(a, 3), "patches": len(t),
                             "raw_bias_%": round(100 * (t.var_pred.sum() - tot) / tot, 1),
                             "gain_bias_%": round(100 * (a * t.var_pred.sum() - tot) / tot, 1),
                             "raw_nMAE": round(nmae(t.var_pred.values, t.var_traj.values), 3),
                             "gain_nMAE": round(nmae(a * t.var_pred.values, t.var_traj.values), 3)})
    return pd.DataFrame(rows)


def person_rows(seq, k, pred, gt, m, pos, d, ps):
    """Purpose: person-level patch rows for ONE frame pair and ONE patch size ps: for patches with >= MIN_PERSONS people,
    the spread of the people's real steps (var_traj) and of SEA-RAFT's flow at the people's pixels (var_pred)."""
    g, inside, xi, yi = sample_at(gt, pos)
    sel = inside & np.isfinite(g).all(1)
    sel[inside] &= m[yi[inside], xi[inside]]
    if sel.sum() < MIN_PERSONS:
        return []
    xi, yi, d = xi[sel], yi[sel], d[sel]
    H, W = pred.shape[:2]
    nbx = -(-W // ps)
    gid = (yi // ps) * nbx + (xi // ps)
    n_groups = nbx * (-(-H // ps))
    cnt, v_traj = grouped_var(d, gid, n_groups)
    _, v_pr = grouped_var(pred[yi, xi], gid, n_groups)
    return [{"seq": seq, "k": k, "patch_px": ps, "n": int(cnt[i]), "var_ref": v_traj[i], "var_pred": v_pr[i]}
            for i in np.nonzero(cnt >= MIN_PERSONS)[0]]


def sampling_vs_patch_test(sizes, step):
    """Purpose: the decisive test. For every patch size and every scene, with ONE SEA-RAFT run per frame pair, compute
    (a) the PERSON estimate: SEA-RAFT at the people's pixels against the people's real steps, and
    (b) the PIXEL estimate: SEA-RAFT over all crowd pixels against the pixel ground truth (the old proxy, scale 1),
    both in patches of the same size. If (b) keeps its large bias at big patches while (a) stays small, the way the
    variance is sampled is the cause; if they converge, the patch size was the cause."""
    global DATA_ROOT, BLOCK
    keep_root, keep_block = DATA_ROOT, BLOCK
    rows_p, rows_x = [], []
    try:
        for seq, root in TRAJ_SCENES.items():
            DATA_ROOT = root
            P, pr = tracks[seq]
            for k in tqdm(list_flow_indices(seq)[::step], desc=f"both estimators {seq}"):
                pos, d = pairs_at(P, pr, k, best_off[seq])
                img0, img1 = load_frame(seq, k), load_frame(seq, k + 1)
                gt = load_gt_flow(seq, k)
                m = load_mask(seq, k) > 0
                pred, _ = real_predict(seq, k, img0, img1)
                inside = m & np.isfinite(gt).all(axis=2)
                for ps in sizes:
                    if len(pos) >= MIN_PERSONS:
                        rows_p += person_rows(seq, k, pred, gt, m, pos, d, ps)
                    BLOCK = ps
                    for r in scale_rows(gt, pred, inside, 1, seq, k):
                        rows_x.append({"seq": seq, "k": k, "patch_px": ps, "n": 0, "var_ref": r["var_gt"], "var_pred": r["var_pred"]})
    finally:
        DATA_ROOT, BLOCK = keep_root, keep_block
    out = []
    for kind, rows in [("person (centre sample vs real steps)", rows_p), ("pixel (all crowd px vs pixel GT)", rows_x)]:
        df = pd.DataFrame(rows)
        for (ps, seq), t in df.groupby(["patch_px", "seq"]):
            if len(t) < 3:
                continue
            tot = t.var_ref.sum()
            out.append({"estimator": kind, "patch_px": ps, "seq": seq, "patches": len(t),
                        "bias_%": round(100 * (t.var_pred.sum() - tot) / tot, 1),
                        "corr": round(float(np.corrcoef(t.var_ref, t.var_pred)[0, 1]), 3),
                        "best_gain_a": round(float((t.var_ref * t.var_pred).sum() / (t.var_pred ** 2).sum()), 2)})
    return pd.DataFrame(out).sort_values(["estimator", "patch_px", "seq"]).reset_index(drop=True)


# ---------------- run ----------------
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 200)
loo = leave_group_out(dfp)
loo.to_csv(OUT_DIR / "e006_person_leave_group_out.csv", index=False)
print("--- ONE gain fitted on the other scenes, tested on the held-out scene (person-level reference, 160 px patches) ---")
print(loo.to_string(index=False))
print("\n--- pre-registered gate: held-out |bias| <= %.0f %% on every scene ---" % GATE_BIAS10)
for sm, d in loo.groupby("smooth_px"):
    worst = d["gain_bias_%"].abs().max()
    print(f"smoothing {sm} px: worst |bias| {worst:.1f}%  (raw worst {d['raw_bias_%'].abs().max():.1f}%)  ->  "
          f"{'PASS' if worst <= GATE_BIAS10 else 'FAIL'}")

cmp_tab = sampling_vs_patch_test(PATCH_SIZES, FRAME_STEP9)
cmp_tab.to_csv(OUT_DIR / "e006_sampling_vs_patch_test.csv", index=False)
print("\n--- person estimator vs pixel estimator, same patch sizes (bias of SEA-RAFT's Var(v) against its own reference) ---")
print(cmp_tab.to_string(index=False))

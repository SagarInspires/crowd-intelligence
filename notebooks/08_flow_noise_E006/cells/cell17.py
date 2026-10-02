# ===== E006 · Cell 17 · Round-trip check: drop points whose flow path does not come back to where it started =====
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---- settings (needs Cells 9, 12, 13, 14, 15 and 16 run in this session: tracks, best_off, TRAJ_SCENES, store_flows, bilinear,
#      advect, sample_at, grouped_var, load_mask, make_starts, blur_flows, crowd_of, leave_crowd_out, real_predict, load_frame) ----
GAPS17    = [16, 37]                  # window length in frames (37 frames = 1.5 s)
K17       = list(range(0, 60, 6))     # start frames
PATCH_17  = 160                       # patch size (px)
MIN_17    = 8                         # a patch needs at least this many reference people and this many kept start points
REPS17    = 2                         # repeats of the random start variants
TAU_MAIN  = 2.0                       # DECLARED BEFORE RUNNING: a point is kept if its round trip ends within 2 px of its start
TAUS17    = [1.0, 2.0, 4.0]           # 1 and 4 are only a sensitivity check; the gate is judged at TAU_MAIN
BASES17   = [("point", 0), ("box4", 4)]   # flow reading: single pixel, or averaged over a 9 x 9 square of person pixels
STARTS17  = ["true", "jitter2", "jitter5", "jitter10", "miss20_jit5", "mask_peaks"]
GATE17    = 10.0                      # pre-registered limit: worst held-out |bias| in percent


def store_backward_flows(seq, frames):
    """Purpose: run SEA-RAFT backwards on every consecutive pair: bflows[j] is the flow from frame j+1 to frame j (half precision).
    Following a point forward with flows[j] and then backwards with bflows[j] should bring it home if the flow is consistent."""
    out = {}
    for j in tqdm(frames, desc=f"store backward flows {seq}"):
        pred, _ = real_predict(seq, j, load_frame(seq, j + 1), load_frame(seq, j))
        out[j] = pred.astype(np.float16)
    return out


def advect_back(bflows, k, N, p_end):
    """Purpose: follow points backwards from their positions in frame k+N to frame k, using the backward flows of frames
    k+N-1, ..., k. Returns the positions reached in frame k (NaN if a point leaves the image)."""
    q = p_end.copy()
    for i in range(N - 1, -1, -1):
        q = q + bilinear(bflows[k + i], q)
    return q


def start_rows17(seq, P, present, off, fl, bfl, base, ks, gaps, rng):
    """Purpose: per start frame, window and start variant: follow the start points forward with the flow, then back with the
    backward flow; the round-trip error is the distance between the start and where the return trip ends. For every tau
    (no filter, 1, 2, 4 px) keep only points with error <= tau and compare, patch by patch, the spread of the kept points'
    velocities with the spread of the people's real velocities (same reference as Cells 14-16: all real people, no filtering).
    The same random start points are used for all tau, so the comparison is paired. Also stores how many points were kept."""
    rows = []
    for k in tqdm(ks, desc=f"{seq} {base}", leave=False):
        a = k + off
        m = load_mask(seq, k) > 0
        nbx = -(-m.shape[1] // PATCH_17)
        n_groups = nbx * (-(-m.shape[0] // PATCH_17))
        for N in gaps:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a:b + 1].all(1)
            if ok.sum() < MIN_17:
                continue
            P_ok = P[ok]
            pos0 = P_ok[:, a]
            ref = (P_ok[:, b] - P_ok[:, a]) / N
            _, inside, xi, yi = sample_at(np.zeros((m.shape[0], m.shape[1], 2)), pos0)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            if sel.sum() < MIN_17:
                continue
            pos_true = pos0[sel]
            gid_r = (yi[sel] // PATCH_17) * nbx + (xi[sel] // PATCH_17)
            cnt_r, v_ref = grouped_var(ref[sel], gid_r, n_groups)
            for var in STARTS17:
                for rep in range(REPS17 if var not in ("true", "mask_peaks") else 1):
                    pts = make_starts(var, pos_true, m, rng)
                    if len(pts) == 0:
                        continue
                    v = advect(fl, k, N, pts)
                    back = advect_back(bfl, k, N, pts + N * v)
                    err = np.linalg.norm(back - pts, axis=1)
                    inb = (pts[:, 0] >= 0) & (pts[:, 0] < m.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < m.shape[0])
                    fin = np.isfinite(v).all(1) & np.isfinite(err) & inb
                    for tau in [None] + TAUS17:
                        keep = fin & (err <= tau) if tau is not None else fin
                        if keep.sum() < MIN_17:
                            continue
                        gid_p = (pts[keep, 1].astype(int) // PATCH_17) * nbx + (pts[keep, 0].astype(int) // PATCH_17)
                        cnt_p, v_pred = grouped_var(v[keep], gid_p, n_groups)
                        cnt_all = np.bincount((pts[fin, 1].astype(int) // PATCH_17) * nbx + (pts[fin, 0].astype(int) // PATCH_17), minlength=n_groups)
                        name = base if tau is None else f"{base}_fb{tau:g}"
                        for i in np.nonzero((cnt_r >= MIN_17) & (cnt_p >= MIN_17))[0]:
                            rows.append({"seq": seq, "k": k, "N": N, "estimator": name, "variant": var, "rep": rep,
                                         "var_ref": v_ref[i], "var_pred": v_pred[i], "n_kept": int(cnt_p[i]), "n_all": int(cnt_all[i])})
    return rows


# ---------------- run ----------------
rng17 = np.random.default_rng(17)
keep_root = DATA_ROOT
rows17 = []
try:
    for seq, root in TRAJ_SCENES.items():
        DATA_ROOT = root
        P, pr = tracks[seq]
        frames = list(range(0, max(K17) + max(GAPS17)))
        flows = store_flows(seq, frames)
        bflows = store_backward_flows(seq, frames)
        masks = {j: load_mask(seq, j) > 0 for j in range(0, max(frames) + 2)}
        for base, r in BASES17:
            if r == 0:
                fl, bfl = flows, bflows
            else:
                fl = blur_flows(flows, masks, r)                                    # forward flow of frame j starts in frame j
                bfl = blur_flows(bflows, {j: masks[j + 1] for j in bflows}, r)      # backward flow of frame j starts in frame j+1
            rows17 += start_rows17(seq, P, pr, best_off[seq], fl, bfl, base, K17, GAPS17, rng17)
            if r != 0:
                del fl, bfl
        del flows, bflows, masks
finally:
    DATA_ROOT = keep_root
df17 = pd.DataFrame(rows17)
df17.to_csv(OUT_DIR / "e006_roundtrip_table.csv", index=False)
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 500)

# per scene: bias at 37 frames and the share of points kept
def per_scene(df, N):
    """Purpose: pooled bias (percent) per scene and the share of start points kept by the filter, for window N."""
    out = []
    for (seq, est, var), t in df[df.N == N].groupby(["seq", "estimator", "variant"]):
        if len(t) < 3:
            continue
        out.append({"seq": seq, "estimator": est, "variant": var,
                    "bias_%": round(100 * (t.var_pred.sum() - t.var_ref.sum()) / t.var_ref.sum(), 1),
                    "kept_%": round(100 * t.n_kept.sum() / t.n_all.sum(), 0)})
    return pd.DataFrame(out)

for N in GAPS17:
    ps = per_scene(df17, N)
    b = ps.pivot(index=["estimator", "variant"], columns="seq", values="bias_%")
    kp = ps.pivot(index=["estimator", "variant"], columns="seq", values="kept_%").add_prefix("kept_")
    print(f"\n--- N = {N} frames: bias % per scene and share of points kept ---")
    print(b.join(kp).to_string())

lco17 = leave_crowd_out(df17)
lco17.to_csv(OUT_DIR / "e006_roundtrip_leave_crowd_out.csv", index=False)
summ = (lco17.assign(raw=lco17["raw_bias_%"].abs(), gained=lco17["gained_bias_%"].abs())
             .groupby(["N", "estimator", "variant"]).agg(worst_raw=("raw", "max"), worst_with_gain=("gained", "max")).round(1))
summ["gate_with_gain"] = np.where(summ.worst_with_gain <= GATE17, "PASS", "FAIL")
for N in GAPS17:
    print(f"\n--- N = {N} frames, leave-crowd-out: worst held-out |bias| % without and with a gain from the OTHER crowds (gate {GATE17:.0f}%; main filter = fb{TAU_MAIN:g}) ---")
    print(summ.loc[N].sort_values("worst_with_gain").to_string())

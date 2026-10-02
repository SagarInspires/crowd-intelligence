# ===== E006 · Cell 15 · Make the estimate less sensitive to the exact start point =====
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from tqdm.auto import tqdm

# ---- settings (needs Cells 9, 12, 13 and 14 run in this session: tracks, best_off, TRAJ_SCENES, store_flows, bilinear,
#      advect, sample_at, grouped_var, load_mask, make_starts, mask_peaks) ----
GAPS15    = [16, 37]               # window length in frames (37 frames = 1.5 s)
K15       = list(range(0, 60, 6))  # start frames
PATCH_15  = 160                    # patch size (px)
MIN_15    = 8                      # a patch needs at least this many reference people and this many start points
REPS15    = 2                      # repeats of the random start variants
RADII15   = [0, 4, 8, 12]          # half-size of the square window the flow is averaged over (0 = single pixel, as before)
CHUNK15   = 4                      # re-seeding interval in frames for the "reseed" estimators
GATE15    = 10.0                   # pre-registered limit: worst scene |bias| at 1.5 s, percent
STARTS15  = ["true", "jitter2", "jitter5", "jitter10", "miss20_jit5", "mask_peaks"]
JITTER15  = {"true", "jitter2", "jitter5", "jitter10"}   # variants where every person keeps an identity (reseed needs that)
SIGMA15   = {"true": 0.0, "jitter2": 2.0, "jitter5": 5.0, "jitter10": 10.0}


def blur_flows(flows, masks, r):
    """Purpose: replace every stored flow field by its average over a (2r+1) x (2r+1) square, counting only pixels the person
    mask marks as people (so that background pixels with zero flow do not pull the value towards zero). Where fewer than 5 percent
    of the pixels in the square are people the original flow is kept. Reading such a field at a slightly wrong position
    gives almost the same velocity, which is the whole idea. Returns a new {frame: field} dictionary."""
    out = {}
    size = 2 * r + 1
    for j, f in flows.items():
        f32 = f.astype(np.float32)
        m = masks[j].astype(np.float32)
        num = uniform_filter(f32 * m[..., None], size=(size, size, 1), mode="nearest")
        den = uniform_filter(m, size=size, mode="nearest")[..., None]
        good = den > 0.05
        out[j] = np.where(good, num / np.maximum(den, 1e-6), f32).astype(np.float16)
    return out


def body_size_table(tracks, best_off, ks):
    """Purpose: a rough size of a person in pixels for every scene: the person-mask area divided by the number of people present,
    as an equivalent diameter sqrt(area / people). Overlapping people make it a little too small. Used only to express a
    start-point error in body sizes."""
    rows = []
    for seq, root in TRAJ_SCENES.items():
        P, pr = tracks[seq]
        d = []
        for k in ks:
            a = k + best_off[seq]
            if 0 <= a < P.shape[1] and pr[:, a].sum() > 0:
                global DATA_ROOT
                DATA_ROOT = root
                d.append(np.sqrt((load_mask(seq, k) > 0).sum() / pr[:, a].sum()))
        rows.append({"seq": seq, "approx_body_size_px": round(float(np.median(d)), 1)})
    return pd.DataFrame(rows)


def window_velocity(flows, k, N, pts):
    """Purpose: the one-shot estimate: follow the start points with the flow for the whole window and return the average velocity."""
    return advect(flows, k, N, pts)


def reseed_velocity(flows, k, N, P_sub, a, sigma, rng, chunk):
    """Purpose: the re-seeding estimate: every `chunk` frames each person gets a new start point (true position at that time plus
    Gaussian error of sigma px, like a detector run again), is followed with the flow for `chunk` frames, and the displacements
    of all chunks are added and divided by N. Needs person identity between chunks (here taken from the trajectories), so it
    shows what a good tracker could buy, not what a detector alone gives. Returns (n, 2)."""
    total = np.zeros((len(P_sub), 2))
    i = 0
    while i < N:
        L = min(chunk, N - i)
        start = P_sub[:, a + i] + rng.normal(0, sigma, (len(P_sub), 2))
        end = start + L * advect(flows, k + i, L, start)
        total += end - start
        i += L
    return total / N


def start_rows15(seq, P, present, off, fl, est, ks, gaps, rng, do_reseed):
    """Purpose: per start frame, window and start variant, compare in patches with enough people the spread of the estimated
    velocity (flow field `fl`, estimator name `est`) with the spread of the people's real average velocity. The one-shot
    estimator is used for all start variants; the re-seeding estimator (identity from trajectories) only for the jitter variants.
    Everything is summed over x and y."""
    rows = []
    for k in tqdm(ks, desc=f"{seq} {est}", leave=False):
        a = k + off
        m = load_mask(seq, k) > 0
        nbx = -(-m.shape[1] // PATCH_15)
        n_groups = nbx * (-(-m.shape[0] // PATCH_15))
        for N in gaps:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a:b + 1].all(1)
            if ok.sum() < MIN_15:
                continue
            P_ok = P[ok]
            pos0 = P_ok[:, a]
            ref = (P_ok[:, b] - P_ok[:, a]) / N
            _, inside, xi, yi = sample_at(np.zeros((m.shape[0], m.shape[1], 2)), pos0)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            if sel.sum() < MIN_15:
                continue
            P_sub, pos_true = P_ok[sel], pos0[sel]
            gid_r = (yi[sel] // PATCH_15) * nbx + (xi[sel] // PATCH_15)
            cnt_r, v_ref = grouped_var(ref[sel], gid_r, n_groups)
            for var in STARTS15:
                for rep in range(REPS15 if var not in ("true", "mask_peaks") else 1):
                    pts = make_starts(var, pos_true, m, rng)
                    if len(pts) > 0:
                        v = window_velocity(fl, k, N, pts)
                        inb = (pts[:, 0] >= 0) & (pts[:, 0] < m.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < m.shape[0])
                        fin = np.isfinite(v).all(1) & inb
                        gid_p = (pts[fin, 1].astype(int) // PATCH_15) * nbx + (pts[fin, 0].astype(int) // PATCH_15)
                        cnt_p, v_pred = grouped_var(v[fin], gid_p, n_groups)
                        for i in np.nonzero((cnt_r >= MIN_15) & (cnt_p >= MIN_15))[0]:
                            rows.append({"seq": seq, "k": k, "N": N, "estimator": est, "variant": var, "rep": rep,
                                         "var_ref": v_ref[i], "var_pred": v_pred[i]})
                    if do_reseed and var in JITTER15:
                        v = reseed_velocity(fl, k, N, P_sub, a, SIGMA15[var], rng, CHUNK15)
                        fin = np.isfinite(v).all(1)
                        cnt_p, v_pred = grouped_var(v[fin], gid_r[fin], n_groups)
                        for i in np.nonzero((cnt_r >= MIN_15) & (cnt_p >= MIN_15))[0]:
                            rows.append({"seq": seq, "k": k, "N": N, "estimator": "reseed_" + est, "variant": var, "rep": rep,
                                         "var_ref": v_ref[i], "var_pred": v_pred[i]})
    return rows


def summarise15(df):
    """Purpose: pooled results per scene, window, estimator and start variant: bias (percent) and correlation over patches."""
    out = []
    for (seq, N, est, var), t in df.groupby(["seq", "N", "estimator", "variant"]):
        if len(t) < 3:
            continue
        tot = t.var_ref.sum()
        out.append({"seq": seq, "N_frames": N, "estimator": est, "variant": var, "patches": len(t),
                    "bias_%": round(100 * (t.var_pred.sum() - tot) / tot, 1),
                    "corr": round(float(np.corrcoef(t.var_ref, t.var_pred)[0, 1]), 3)})
    return pd.DataFrame(out)


# ---------------- run ----------------
rng15 = np.random.default_rng(15)
keep_root = DATA_ROOT
rows15 = []
try:
    for seq, root in TRAJ_SCENES.items():
        DATA_ROOT = root
        P, pr = tracks[seq]
        frames = list(range(0, max(K15) + max(GAPS15)))
        flows = store_flows(seq, frames)
        masks = {j: load_mask(seq, j) > 0 for j in frames}
        for r in RADII15:
            fl = flows if r == 0 else blur_flows(flows, masks, r)
            name = "point" if r == 0 else f"box{r}"
            rows15 += start_rows15(seq, P, pr, best_off[seq], fl, name, K15, GAPS15, rng15, do_reseed=(r in (0, 8)))
            if r != 0:
                del fl
        del flows, masks
finally:
    DATA_ROOT = keep_root
df15 = pd.DataFrame(rows15)
df15.to_csv(OUT_DIR / "e006_robust_start_table.csv", index=False)
tab15 = summarise15(df15)
tab15.to_csv(OUT_DIR / "e006_robust_start_summary.csv", index=False)
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 500)
print("--- approximate body size per scene (px), to read the start-point error in body sizes ---")
print(body_size_table(tracks, best_off, K15).to_string(index=False))

g = tab15[tab15.N_frames == max(GAPS15)].pivot(index=["estimator", "variant"], columns="seq", values="bias_%")
g["worst_abs"] = g.abs().max(axis=1)
g["gate"] = np.where(g.worst_abs <= GATE15, "PASS", "FAIL")
print(f"\n--- bias % at {max(GAPS15)} frames (1.5 s) per scene, worst scene, gate = worst |bias| <= {GATE15:.0f}% ---")
print(g.round(1).to_string())

print("\n--- worst-scene |bias| % per estimator and start variant, at 16 and 37 frames ---")
w = tab15.assign(a=tab15["bias_%"].abs()).groupby(["N_frames", "estimator", "variant"]).a.max().unstack("variant")
print(w.round(1).to_string())

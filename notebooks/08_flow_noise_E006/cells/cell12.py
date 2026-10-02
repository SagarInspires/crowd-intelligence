# ===== E006 · Cell 12 · Without the oracle: follow people with the flow itself, up to ~1.5 s =====
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from tqdm.auto import tqdm

# ---- settings (needs Cell 9 run in this session: tracks, best_off, TRAJ_SCENES, grouped_var, sample_at ...) ----
GAPS12   = [4, 8, 16, 24, 37]    # window length N in frames (37 frames = 1.5 s at 25 fps)
K12      = list(range(0, 60, 6)) # start frames (10 starts; needs k + N <= 119, so N up to 37 is fine)
PATCH_12 = 160                   # patch size (px)
MIN_P12  = 8                     # a patch needs at least this many people


def store_flows(seq, frames):
    """Purpose: run SEA-RAFT on every consecutive pair j -> j+1 for the given frame numbers and keep the whole flow fields
    (half precision to save memory, about 3.7 MB each). Returns {j: array (H, W, 2)}. Needed here because the people are
    followed by the flow itself, so we must read it at any position, not only at the true positions."""
    out = {}
    for j in tqdm(frames, desc=f"store flows {seq}"):
        pred, _ = real_predict(seq, j, load_frame(seq, j), load_frame(seq, j + 1))
        out[j] = pred.astype(np.float16)
    return out


def bilinear(field, pos):
    """Purpose: read a (H, W, 2) field at float positions (x, y) with bilinear interpolation. Positions outside the
    image give NaN."""
    H, W = field.shape[:2]
    out = np.stack([map_coordinates(field[..., c].astype(np.float32), [pos[:, 1], pos[:, 0]], order=1,
                                    mode="constant", cval=np.nan) for c in range(2)], axis=1)
    out[(pos[:, 0] < 0) | (pos[:, 0] > W - 1) | (pos[:, 1] < 0) | (pos[:, 1] > H - 1)] = np.nan
    return out


def velocity_estimates(flows, k, N, pos0, true_pos_by_step):
    """Purpose: three ways to turn the N one-step flows k..k+N-1 into one velocity per person over the window:
      oracle   : average of the flows read at the person's TRUE position at every step (needs real tracks);
      advected : follow the person with the flow itself (start at the true position, move by the flow, repeat), the
                 velocity is the total displacement / N (what a real system without tracks could do);
      fixed    : average of the flows read at the START position, no following (the Eulerian way, like PIV).
    Returns three arrays (persons, 2)."""
    orc = np.mean([bilinear(flows[k + i], true_pos_by_step[i]) for i in range(N)], axis=0)
    fix = np.mean([bilinear(flows[k + i], pos0) for i in range(N)], axis=0)
    p = pos0.copy()
    for i in range(N):
        p = p + bilinear(flows[k + i], p)
    adv = (p - pos0) / N
    return orc, adv, fix


def window_rows(seq, P, present, off, flows, ks, gaps):
    """Purpose: for every start frame and window, build per-patch rows (>= MIN_P12 people): var_ref (the people's real average
    velocity over the window, from the trajectories) and the spread of each of the three SEA-RAFT estimates. Only people
    present during the whole window and standing on a person pixel at the start are used, the same people for all three."""
    rows = []
    for k in tqdm(ks, desc=f"windows {seq}"):
        a = k + off
        m = load_mask(seq, k) > 0
        for N in gaps:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a:b + 1].all(1)
            if ok.sum() < MIN_P12:
                continue
            pos0 = P[ok, a]
            ref = (P[ok, b] - P[ok, a]) / N
            steps = [P[ok, a + i] for i in range(N)]
            _, inside, xi, yi = sample_at(np.zeros((m.shape[0], m.shape[1], 2)), pos0)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            orc, adv, fix = velocity_estimates(flows, k, N, pos0, steps)
            good = sel & np.isfinite(orc).all(1) & np.isfinite(adv).all(1) & np.isfinite(fix).all(1)
            if good.sum() < MIN_P12:
                continue
            xi, yi = xi[good], yi[good]
            nbx = -(-m.shape[1] // PATCH_12)
            gid = (yi // PATCH_12) * nbx + (xi // PATCH_12)
            n_groups = nbx * (-(-m.shape[0] // PATCH_12))
            cnt, v_ref = grouped_var(ref[good], gid, n_groups)
            vs = {n: grouped_var(arr[good], gid, n_groups)[1] for n, arr in [("oracle", orc), ("advected", adv), ("fixed", fix)]}
            for i in np.nonzero(cnt >= MIN_P12)[0]:
                for n, v in vs.items():
                    rows.append({"seq": seq, "k": k, "N": N, "estimator": n, "var_ref": v_ref[i], "var_pred": v[i]})
    return rows


def summarise_windows(df):
    """Purpose: pooled results per scene, window and estimator: mean true spread (it falls with the window), the bias of
    SEA-RAFT's spread, its correlation with the true spread and the best single multiplier."""
    out = []
    for (seq, est, N), t in df.groupby(["seq", "estimator", "N"]):
        if len(t) < 3:
            continue
        tot = t.var_ref.sum()
        out.append({"seq": seq, "estimator": est, "N_frames": N, "patches": len(t), "mean_var_ref": round(t.var_ref.mean(), 3),
                    "bias_%": round(100 * (t.var_pred.sum() - tot) / tot, 1),
                    "corr": round(float(np.corrcoef(t.var_ref, t.var_pred)[0, 1]), 3),
                    "best_gain_a": round(float((t.var_ref * t.var_pred).sum() / (t.var_pred ** 2).sum()), 2)})
    return pd.DataFrame(out)


# ---------------- run ----------------
keep_root = DATA_ROOT
rows12 = []
try:
    for seq, root in TRAJ_SCENES.items():
        DATA_ROOT = root
        P, pr = tracks[seq]
        frames = list(range(0, max(K12) + max(GAPS12)))
        flows = store_flows(seq, frames)
        rows12 += window_rows(seq, P, pr, best_off[seq], flows, K12, GAPS12)
        del flows                                   # free the memory before the next scene
finally:
    DATA_ROOT = keep_root
df12 = pd.DataFrame(rows12)
df12.to_csv(OUT_DIR / "e006_window_table.csv", index=False)
tab12 = summarise_windows(df12)
tab12.to_csv(OUT_DIR / "e006_window_summary.csv", index=False)
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 200)
print("--- Var(v) over windows up to 1.5 s: three ways to follow the people, against the real velocity over the same window ---")
print(tab12.to_string(index=False))

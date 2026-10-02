# ===== E006 · Cell 11 · Longer time windows: does the bias change when velocity is measured over N frames? =====
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---- settings (needs Cell 9 and Cell 10 run in this session: tracks, best_off, TRAJ_SCENES, grouped_var, sample_at ...) ----
GAPS     = [1, 2, 4, 8, 16]      # window length N in frames (25 fps: 16 frames = 0.64 s; the literature uses ~1.5 s = 37 frames)
K_STEP   = 4                     # use every 4th starting frame
PATCH_G  = 160                   # patch size (px) for the spread of the people's velocities
MIN_P    = 8                     # a patch needs at least this many people


def flows_at_people(seq, P, present, off, frames):
    """Purpose: run SEA-RAFT once on every consecutive pair (j -> j+1) for the frame numbers in `frames` and store only
    its flow at the people's positions (NaN where the person is absent or outside the image). Returns {j: array (persons, 2)}.
    These one-step flows are reused to build the 'chain' estimate (average of N one-step flows)."""
    out = {}
    for j in tqdm(frames, desc=f"one-step flows {seq}"):
        c = j + off
        cache = np.full((P.shape[0], 2), np.nan)
        if 0 <= c < P.shape[1] and present[:, c].any():
            pred, _ = real_predict(seq, j, load_frame(seq, j), load_frame(seq, j + 1))
            v, inside, _, _ = sample_at(pred, P[present[:, c], c])
            tmp = np.full((int(present[:, c].sum()), 2), np.nan)
            tmp[inside] = v[inside]
            cache[present[:, c]] = tmp
        out[j] = cache
    return out


def gap_rows(seq, P, present, off, ks, gaps, one_step):
    """Purpose: for every start frame k and window N build, for the people present during the whole window and standing
    on a person pixel at frame k:
      reference : the person's real average velocity over the window, (x_{k+N} - x_k) / N, from the trajectories;
      direct    : SEA-RAFT run directly on frame k and frame k+N, sampled at the person's start position, divided by N;
      chain     : the average of the N one-step SEA-RAFT flows, each sampled at the person's true position at that frame
                  (an oracle tracker, so this isolates what temporal averaging does to the flow errors).
    Returns one row per patch (>= MIN_P people) and estimator with the spread (variance, summed over x and y) of each."""
    rows = []
    for k in tqdm(ks, desc=f"gap windows {seq}"):
        a = k + off
        m = load_mask(seq, k) > 0
        img0 = load_frame(seq, k)
        for N in gaps:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a:b + 1].all(1)
            if ok.sum() < MIN_P:
                continue
            pos = P[ok, a]
            ref = (P[ok, b] - P[ok, a]) / N
            _, inside, xi, yi = sample_at(np.zeros((m.shape[0], m.shape[1], 2)), pos)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            if N == 1:
                direct = one_step[k][ok]
                chain = direct
            else:
                pred, _ = real_predict(seq, k, img0, load_frame(seq, k + N))
                direct, ins2, _, _ = sample_at(pred, pos)
                direct = direct / N
                direct[~ins2] = np.nan
                chain = np.stack([one_step[k + i][ok] for i in range(N)]).mean(axis=0)   # NaN if any step is missing
            good = sel & np.isfinite(direct).all(1) & np.isfinite(chain).all(1)
            if good.sum() < MIN_P:
                continue
            xi, yi, ref, direct, chain = xi[good], yi[good], ref[good], direct[good], chain[good]
            nbx = -(-m.shape[1] // PATCH_G)
            gid = (yi // PATCH_G) * nbx + (xi // PATCH_G)
            n_groups = nbx * (-(-m.shape[0] // PATCH_G))
            cnt, v_ref = grouped_var(ref, gid, n_groups)
            _, v_dir = grouped_var(direct, gid, n_groups)
            _, v_chn = grouped_var(chain, gid, n_groups)
            for i in np.nonzero(cnt >= MIN_P)[0]:
                for est, v in [("direct", v_dir[i]), ("chain", v_chn[i])]:
                    rows.append({"seq": seq, "k": k, "N": N, "estimator": est, "var_ref": v_ref[i], "var_pred": v})
    return rows


def summarise_gaps(df):
    """Purpose: pooled results per scene, window and estimator: how much the people's real velocity spread falls as the window
    grows (mean_var_ref), and the bias, correlation and best single multiplier of SEA-RAFT's spread against it."""
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
rows11 = []
try:
    for seq, root in TRAJ_SCENES.items():
        DATA_ROOT = root
        P, pr = tracks[seq]
        ks = [k for k in list_flow_indices(seq) if k % K_STEP == 0]
        frames = list(range(0, max(ks) + max(GAPS)))
        one_step = flows_at_people(seq, P, pr, best_off[seq], frames)
        rows11 += gap_rows(seq, P, pr, best_off[seq], ks, GAPS, one_step)
finally:
    DATA_ROOT = keep_root
df11 = pd.DataFrame(rows11)
df11.to_csv(OUT_DIR / "e006_gap_table.csv", index=False)
tab11 = summarise_gaps(df11)
tab11.to_csv(OUT_DIR / "e006_gap_summary.csv", index=False)
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 200)
print("--- person-level Var(v) over windows of N frames: SEA-RAFT vs the people's real velocity over the same window ---")
print(tab11.to_string(index=False))

# ===== E006 · Cell 9 · Person-level reference: read the trajectories, check them, compare with SEA-RAFT =====
import os
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

# ---- settings (needs Cell 7's DATA_ROOT_1 / DATA_ROOT_2, or set them by hand) ----
TRAJ_SCENES = {"IM01": DATA_ROOT_1, "IM01_hDyn": DATA_ROOT_1, "IM03": DATA_ROOT_2, "IM05": DATA_ROOT_2}
PATCH_P     = 160        # patch size (px) for the person-level variance: bigger than before, because a patch needs several people
MIN_PERSONS = 8          # a patch needs at least this many people to give a variance
SMOOTH_PX   = [1, 5]     # SEA-RAFT flow sampled at the person's pixel (1) or averaged over 5 x 5 pixels (5)
OFFSETS     = [-1, 0, 1] # possible shift between the trajectory frame index and the image frame index
FRAME_STEP9 = 2          # every 2nd frame pair


def find_traj_file(seq):
    """Purpose: find <seq>/personTrajectories.txt anywhere under /kaggle/input (the small trajectory dataset).
    Returns the path or None."""
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.basename(root) == seq and "personTrajectories.txt" in files:
            return os.path.join(root, "personTrajectories.txt")
    return None


def load_person_tracks(path):
    """Purpose: read personTrajectories.txt. Each line is ONE person: x1,y1,x2,y2,... one pair per frame, and a
    pair (0,0) means 'not present in that frame'. Each pair is stored as (y, x) (found in Cell 9b), so the two
    numbers are swapped here. Returns an array (persons, frames, 2) and a boolean array
    (persons, frames) saying where the person is present. Lines of different length are padded with 'absent'."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                v = np.array(line.split(","), dtype=np.float64)
                rows.append(v[: (len(v) // 2) * 2].reshape(-1, 2))
    T = max(len(r) for r in rows)
    P = np.zeros((len(rows), T, 2))
    for i, r in enumerate(rows):
        P[i, : len(r)] = r
    present = (P[..., 0] != 0) | (P[..., 1] != 0)
    P = P[..., ::-1].copy()          # the file stores each point as (y, x): swap so that P[..., 0] = x and P[..., 1] = y
    return P, present


def pairs_at(P, present, k, off):
    """Purpose: people present in BOTH trajectory frames k+off and k+off+1: returns their position at the first
    frame (x, y) and their displacement to the second frame (dx, dy), in pixels per frame."""
    a, b = k + off, k + off + 1
    if a < 0 or b >= P.shape[1]:
        return np.zeros((0, 2)), np.zeros((0, 2))
    ok = present[:, a] & present[:, b]
    return P[ok, a], P[ok, b] - P[ok, a]


def sample_at(field, pos):
    """Purpose: read a (H, W, 2) field at float pixel positions (x, y) (nearest pixel). Returns the values and a
    boolean saying which positions fall inside the image."""
    H, W = field.shape[:2]
    xi, yi = np.rint(pos[:, 0]).astype(int), np.rint(pos[:, 1]).astype(int)
    inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    out = np.full((len(pos), 2), np.nan)
    out[inside] = field[yi[inside], xi[inside]]
    return out, inside, xi, yi


def alignment_check(seq, P, present, ks):
    """Purpose: do the trajectories agree with the ground-truth flow, and with which frame shift? For every shift
    in OFFSETS compare the person's step (x_{k+1} - x_k) with the true flow at the person's pixel, only where the
    mask says 'person'. A good shift gives a small median difference and most positions inside the mask."""
    out = []
    for off in OFFSETS:
        diffs, inmask, total = [], 0, 0
        for k in ks:
            pos, d = pairs_at(P, present, k, off)
            if len(pos) == 0:
                continue
            gt = load_gt_flow(seq, k)
            m = load_mask(seq, k) > 0
            g, inside, xi, yi = sample_at(np.nan_to_num(gt, nan=np.nan), pos)
            good = inside & np.isfinite(g).all(1)
            total += len(pos)
            msk = np.zeros(len(pos), bool)
            msk[inside] = m[yi[inside], xi[inside]]
            inmask += int((msk & good).sum())
            sel = msk & good
            diffs.append(np.sqrt(((g[sel] - d[sel]) ** 2).sum(1)))
        dd = np.concatenate(diffs) if diffs else np.array([np.nan])
        out.append({"seq": seq, "offset": off, "persons_checked": total,
                    "in_mask_%": round(100 * inmask / max(total, 1), 1),
                    "median_|traj_step - gt_flow|_px": round(float(np.nanmedian(dd)), 3),
                    "mean_|traj_step - gt_flow|_px": round(float(np.nanmean(dd)), 3)})
    return out


def run_scene(seq, P, present, off, step):
    """Purpose: for one scene run SEA-RAFT on every `step`-th pair and build, for every patch with enough people:
    var_traj (spread of the people's real steps), var_gt (spread of the ground-truth flow at their pixels),
    var_pred (spread of SEA-RAFT's flow at their pixels, for each smoothing in SMOOTH_PX). Also accumulates the
    within-patch slope of SEA-RAFT's flow on the real step (shrinkage factor beta)."""
    rows, beta_num, beta_den = [], {s: 0.0 for s in SMOOTH_PX}, {s: 0.0 for s in SMOOTH_PX}
    epe = {s: [] for s in SMOOTH_PX}
    epe_gt = []
    for k in tqdm(list_flow_indices(seq)[::step], desc=f"persons {seq}"):
        pos, d = pairs_at(P, present, k, off)
        if len(pos) < MIN_PERSONS:
            continue
        img0, img1 = load_frame(seq, k), load_frame(seq, k + 1)
        gt = load_gt_flow(seq, k)
        m = load_mask(seq, k) > 0
        pred, _ = real_predict(seq, k, img0, img1)
        g, inside, xi, yi = sample_at(gt, pos)
        sel = inside & np.isfinite(g).all(1)
        sel[inside] &= m[yi[inside], xi[inside]]
        if sel.sum() < MIN_PERSONS:
            continue
        xi, yi, d, g = xi[sel], yi[sel], d[sel], g[sel]
        epe_gt.append(np.sqrt(((g - d) ** 2).sum(1)))
        nbx = -(-pred.shape[1] // PATCH_P)
        gid = (yi // PATCH_P) * nbx + (xi // PATCH_P)
        n_groups = nbx * (-(-pred.shape[0] // PATCH_P))
        cnt, v_traj = grouped_var(d, gid, n_groups)
        _, v_gt = grouped_var(g, gid, n_groups)
        use = cnt >= MIN_PERSONS
        for s in SMOOTH_PX:
            f = pred if s == 1 else cv2.blur(pred, (s, s))
            pr = f[yi, xi]
            epe[s].append(np.sqrt(((pr - d) ** 2).sum(1)))
            _, v_pr = grouped_var(pr, gid, n_groups)
            # within-patch centred slope of SEA-RAFT on the real step
            for c in range(2):
                dm = np.bincount(gid, weights=d[:, c], minlength=n_groups) / np.maximum(cnt, 1)
                pm = np.bincount(gid, weights=pr[:, c], minlength=n_groups) / np.maximum(cnt, 1)
                ok = use[gid]
                beta_num[s] += float(((pr[ok, c] - pm[gid][ok]) * (d[ok, c] - dm[gid][ok])).sum())
                beta_den[s] += float(((d[ok, c] - dm[gid][ok]) ** 2).sum())
            for i in np.nonzero(use)[0]:
                rows.append({"seq": seq, "k": k, "smooth": s, "persons": int(cnt[i]), "var_traj": v_traj[i],
                             "var_gt": v_gt[i], "var_pred": v_pr[i]})
    summ = []
    for s in SMOOTH_PX:
        e = np.concatenate(epe[s]) if epe[s] else np.array([np.nan])
        summ.append({"seq": seq, "smooth_px": s, "persons_used": len(e),
                     "median_|gt-traj|_px": round(float(np.median(np.concatenate(epe_gt))), 3) if epe_gt else np.nan,
                     "median_|pred-traj|_px": round(float(np.median(e)), 3),
                     "beta_pred_on_traj": round(beta_num[s] / max(beta_den[s], 1e-9), 3)})
    return pd.DataFrame(rows), pd.DataFrame(summ)


def summarise_person(df):
    """Purpose: pooled results per scene and smoothing, as percent of the TRAJECTORY-based Var(v): bias of the
    SEA-RAFT spread, bias of the pixel ground-truth spread (does the old reference agree with the people's
    real steps?), correlation, and the best single multiplier (traj = a * pred)."""
    out = []
    for (s, sm), d in df.groupby(["seq", "smooth"]):
        tot = d.var_traj.sum()
        out.append({"seq": s, "smooth_px": sm, "patches": len(d), "mean_var_traj": round(d.var_traj.mean(), 3),
                    "pred_bias_%": round(100 * (d.var_pred.sum() - tot) / tot, 1),
                    "gt_bias_%": round(100 * (d.var_gt.sum() - tot) / tot, 1),
                    "corr(traj,pred)": round(float(np.corrcoef(d.var_traj, d.var_pred)[0, 1]), 3) if len(d) > 2 else np.nan,
                    "best_gain_a": round(float((d.var_traj * d.var_pred).sum() / (d.var_pred ** 2).sum()), 2)})
    return pd.DataFrame(out)


# ---------------- run ----------------
tracks, missing = {}, []
for seq in TRAJ_SCENES:
    p = find_traj_file(seq)
    if p is None:
        missing.append(seq)
    else:
        tracks[seq] = load_person_tracks(p)
if missing:
    raise RuntimeError(f"personTrajectories.txt not found for {missing}. Attach the trajectory dataset "
                       "(tub-crowdflow-persontraj) and run `!ls /kaggle/input`.")
pd.set_option("display.width", 240, "display.max_columns", 30)
print("--- trajectory files: people, frames per line, how many people are present per frame ---")
for seq, (P, pr) in tracks.items():
    print(f"{seq}: {P.shape[0]} people, {P.shape[1]} frames per line, present per frame: "
          f"min {pr.sum(0).min()} / median {int(np.median(pr.sum(0)))} / max {pr.sum(0).max()}")

DATA_ROOT_KEEP = DATA_ROOT
align, best_off = [], {}
try:
    for seq, root in TRAJ_SCENES.items():
        DATA_ROOT = root
        P, pr = tracks[seq]
        rows_a = alignment_check(seq, P, pr, list_flow_indices(seq)[:40:4])
        align += rows_a
        best = min(rows_a, key=lambda r: r["median_|traj_step - gt_flow|_px"])
        best_off[seq] = best["offset"]
    print("\n--- do the trajectories match the ground-truth flow? (smaller difference = better; pick the best offset) ---")
    print(pd.DataFrame(align).to_string(index=False))
    print("chosen offsets:", best_off)

    all_rows, all_summ = [], []
    for seq, root in TRAJ_SCENES.items():
        DATA_ROOT = root
        P, pr = tracks[seq]
        r, s_ = run_scene(seq, P, pr, best_off[seq], FRAME_STEP9)
        all_rows.append(r)
        all_summ.append(s_)
finally:
    DATA_ROOT = DATA_ROOT_KEEP
dfp = pd.concat(all_rows, ignore_index=True)
dfp.to_csv(OUT_DIR / "e006_person_patch_table.csv", index=False)
print("\n--- person level: how close are SEA-RAFT's and the ground-truth flow to the people's real steps? ---")
print(pd.concat(all_summ).to_string(index=False))
print("\n--- person-level Var(v) per patch: SEA-RAFT and pixel ground truth vs the trajectory reference ---")
tab9 = summarise_person(dfp)
print(tab9.to_string(index=False))
tab9.to_csv(OUT_DIR / "e006_person_summary.csv", index=False)

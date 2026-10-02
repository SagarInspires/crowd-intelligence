# ===== E006 · Cell 13 · No detections needed? Follow crowd PIXELS with the flow and compare with the people's real velocity =====
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---- settings (needs Cells 9 and 12 run in this session: tracks, best_off, TRAJ_SCENES, store_flows, bilinear, sample_at, grouped_var) ----
GAPS13    = [1, 4, 8, 16, 24, 37]   # window length in frames (37 frames = 1.5 s)
K13       = list(range(0, 60, 6))   # start frames
PATCH_13  = 160                     # patch size (px)
SEED_STEP = 6                       # seed one point every 6 pixels inside the crowd mask
MIN_PEOPLE13 = 8                    # a patch needs at least this many people (for the reference)
MIN_SEEDS13  = 30                   # and at least this many seed points


def advect(flows, k, N, pos0):
    """Purpose: follow points with the flow itself: start at pos0 (x, y) in frame k, move by the flow of frame k, then by the
    flow of frame k+1 at the new position, and so on for N steps. Returns the average velocity (total displacement / N)
    and the final positions. Points that leave the image become NaN."""
    p = pos0.copy()
    for i in range(N):
        p = p + bilinear(flows[k + i], p)
    return (p - pos0) / N


def seed_points(mask, step):
    """Purpose: a regular grid of points (x, y) on the pixels the mask marks as people: no detector and no tracks are needed
    to get them, only the person/background mask (in real video this would come from a segmentation or a density map)."""
    ys, xs = np.mgrid[step // 2:mask.shape[0]:step, step // 2:mask.shape[1]:step]
    keep = mask[ys, xs]
    return np.column_stack([xs[keep], ys[keep]]).astype(np.float64)


def seed_rows(seq, P, present, off, flows, ks, gaps):
    """Purpose: for every start frame k and window N build per-patch rows that compare, in patches that contain enough people:
      var_ref    : spread of the people's real average velocity over the window (from the trajectories),
      var_people : spread of the flow-followed velocity of the people themselves (start at their true position),
      var_pixels : spread of the flow-followed velocity of grid points on ALL crowd pixels (no detections needed).
    Everything is summed over x and y. The pixel seeds also cover limbs, so over a short window they carry limb motion;
    the question is whether a long window averages that out."""
    rows = []
    for k in tqdm(ks, desc=f"pixel seeds {seq}"):
        a = k + off
        m = load_mask(seq, k) > 0
        seeds = seed_points(m, SEED_STEP)
        nbx = -(-m.shape[1] // PATCH_13)
        n_groups = nbx * (-(-m.shape[0] // PATCH_13))
        gid_s = (seeds[:, 1].astype(int) // PATCH_13) * nbx + (seeds[:, 0].astype(int) // PATCH_13)
        for N in gaps:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a:b + 1].all(1)
            if ok.sum() < MIN_PEOPLE13:
                continue
            pos0 = P[ok, a]
            ref = (P[ok, b] - P[ok, a]) / N
            _, inside, xi, yi = sample_at(np.zeros((m.shape[0], m.shape[1], 2)), pos0)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            v_pe = advect(flows, k, N, pos0)
            good = sel & np.isfinite(v_pe).all(1)
            if good.sum() < MIN_PEOPLE13:
                continue
            gid_p = (yi[good] // PATCH_13) * nbx + (xi[good] // PATCH_13)
            cnt_p, v_ref = grouped_var(ref[good], gid_p, n_groups)
            _, v_people = grouped_var(v_pe[good], gid_p, n_groups)
            vs = advect(flows, k, N, seeds)
            fin = np.isfinite(vs).all(1)
            cnt_s, v_pix = grouped_var(vs[fin], gid_s[fin], n_groups)
            for i in np.nonzero((cnt_p >= MIN_PEOPLE13) & (cnt_s >= MIN_SEEDS13))[0]:
                rows.append({"seq": seq, "k": k, "N": N, "var_ref": v_ref[i], "var_people": v_people[i], "var_pixels": v_pix[i]})
    return rows


def summarise_seeds(df):
    """Purpose: pooled results per scene and window: how the spread from the flow-followed PIXELS and from the flow-followed
    PEOPLE compares with the people's real velocity spread (bias in percent and correlation over patches)."""
    out = []
    for (seq, N), t in df.groupby(["seq", "N"]):
        if len(t) < 3:
            continue
        tot = t.var_ref.sum()
        out.append({"seq": seq, "N_frames": N, "patches": len(t), "mean_var_ref": round(t.var_ref.mean(), 3),
                    "people_bias_%": round(100 * (t.var_people.sum() - tot) / tot, 1),
                    "pixels_bias_%": round(100 * (t.var_pixels.sum() - tot) / tot, 1),
                    "corr_people": round(float(np.corrcoef(t.var_ref, t.var_people)[0, 1]), 3),
                    "corr_pixels": round(float(np.corrcoef(t.var_ref, t.var_pixels)[0, 1]), 3)})
    return pd.DataFrame(out)


# ---------------- run ----------------
keep_root = DATA_ROOT
rows13 = []
try:
    for seq, root in TRAJ_SCENES.items():
        DATA_ROOT = root
        P, pr = tracks[seq]
        flows = store_flows(seq, list(range(0, max(K13) + max(GAPS13))))
        rows13 += seed_rows(seq, P, pr, best_off[seq], flows, K13, GAPS13)
        del flows
finally:
    DATA_ROOT = keep_root
df13 = pd.DataFrame(rows13)
df13.to_csv(OUT_DIR / "e006_pixel_seed_table.csv", index=False)
tab13 = summarise_seeds(df13)
tab13.to_csv(OUT_DIR / "e006_pixel_seed_summary.csv", index=False)
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 200)
print("--- Var(v) from flow-followed crowd PIXELS vs flow-followed PEOPLE, against the people's real velocity spread ---")
print(tab13.to_string(index=False))

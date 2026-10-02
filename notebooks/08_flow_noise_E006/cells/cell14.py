# ===== E006 · Cell 14 · Realistic start points: does the flow-followed estimator survive a detector's mistakes? =====
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
from tqdm.auto import tqdm

# ---- settings (needs Cells 9, 12 and 13 run in this session: tracks, best_off, TRAJ_SCENES, store_flows, bilinear,
#      advect, sample_at, grouped_var, load_mask) ----
GAPS14     = [4, 16, 37]            # window length in frames (37 frames = 1.5 s)
K14        = list(range(0, 60, 6))  # start frames
PATCH_14   = 160                    # patch size (px)
MIN_PTS14  = 8                      # a patch needs at least this many reference people and this many start points
REPS14     = 3                      # repeats of the random variants (different random errors each time)
GATE14     = 10.0                   # pre-registered limit: worst scene |bias| at 1.5 s, in percent
VARIANTS14 = ["true", "jitter2", "jitter5", "jitter10", "miss20_jit5", "miss30_jit5_fp10", "blob_centroids", "mask_peaks"]
RANDOM14   = {"jitter2", "jitter5", "jitter10", "miss20_jit5", "miss30_jit5_fp10"}   # variants that use random numbers


def blob_centroids(m, min_area=30):
    """Purpose: the simplest detector stand-in: one point at the centre of every connected blob of the person mask
    (blobs smaller than min_area pixels are ignored). Touching people merge into one blob, so dense crowds get too few points.
    Returns an array (n, 2) of (x, y)."""
    lab, n = label(m)
    if n == 0:
        return np.zeros((0, 2))
    ids = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    cm = np.array(center_of_mass(m, lab, ids))            # (n, 2) in (y, x)
    keep = area >= min_area
    return cm[keep][:, ::-1]


def mask_peaks(m, sigma=5, size=21, thr=0.3):
    """Purpose: a density-map stand-in: blur the person mask (so every person becomes a smooth bump) and take the local
    maxima as people. Separates touching people better than blob centroids but still makes mistakes. Returns (n, 2) of (x, y)."""
    s = gaussian_filter(m.astype(np.float32), sigma)
    pk = (s == maximum_filter(s, size=size)) & (s > thr)
    ys, xs = np.nonzero(pk)
    return np.column_stack([xs, ys]).astype(np.float64)


def make_starts(variant, pos_true, m, rng):
    """Purpose: build the start points a real system could have, from the true positions of the people (pos_true, (n, 2) in x, y)
    and the person mask m:
      true / jitter<s>      : true positions, optionally with Gaussian position error of s pixels;
      miss<p>_jit5          : additionally p percent of the people are missed;
      miss30_jit5_fp10      : 30 percent missed and false points added (10 percent of the true count, placed on random mask pixels);
      blob_centroids / mask_peaks : points taken from the mask only (no trajectories at all)."""
    if variant == "true":
        return pos_true.copy()
    if variant == "blob_centroids":
        return blob_centroids(m)
    if variant == "mask_peaks":
        return mask_peaks(m)
    sigma = {"jitter2": 2, "jitter5": 5, "jitter10": 10, "miss20_jit5": 5, "miss30_jit5_fp10": 5}[variant]
    miss = {"miss20_jit5": 0.2, "miss30_jit5_fp10": 0.3}.get(variant, 0.0)
    pts = pos_true[rng.random(len(pos_true)) >= miss]
    pts = pts + rng.normal(0, sigma, pts.shape)
    if variant == "miss30_jit5_fp10":
        flat = np.flatnonzero(m.ravel())
        if len(flat) > 0:
            pick = rng.choice(flat, size=max(1, int(0.1 * len(pos_true))), replace=False)
            pts = np.vstack([pts, np.column_stack([pick % m.shape[1], pick // m.shape[1]]).astype(np.float64)])
    return pts


def start_rows(seq, P, present, off, flows, ks, gaps, rng):
    """Purpose: for every start frame k, window N and start-point variant, compare in patches with enough people:
      var_ref  : spread of the people's REAL average velocity over the window (trajectories, only people who stay present
                 during the whole window and stand on a person pixel at the start);
      var_pred : spread of the flow-followed velocity of the variant's start points (each point is followed with the flow).
    Also stores how many start points there were compared with real people in the patch (n_pts, n_ref). Everything is
    summed over x and y."""
    rows = []
    for k in tqdm(ks, desc=f"start points {seq}"):
        a = k + off
        m = load_mask(seq, k) > 0
        nbx = -(-m.shape[1] // PATCH_14)
        n_groups = nbx * (-(-m.shape[0] // PATCH_14))
        for N in gaps:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a:b + 1].all(1)
            if ok.sum() < MIN_PTS14:
                continue
            pos0 = P[ok, a]
            ref = (P[ok, b] - P[ok, a]) / N
            _, inside, xi, yi = sample_at(np.zeros((m.shape[0], m.shape[1], 2)), pos0)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            if sel.sum() < MIN_PTS14:
                continue
            pos_true = pos0[sel]
            gid_r = (yi[sel] // PATCH_14) * nbx + (xi[sel] // PATCH_14)
            cnt_r, v_ref = grouped_var(ref[sel], gid_r, n_groups)
            for var in VARIANTS14:
                for rep in range(REPS14 if var in RANDOM14 else 1):
                    pts = make_starts(var, pos_true, m, rng)
                    if len(pts) == 0:
                        continue
                    v = advect(flows, k, N, pts)
                    # a point is assigned to the patch where it STARTS; points that start outside the image are dropped
                    inb = (pts[:, 0] >= 0) & (pts[:, 0] < m.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < m.shape[0])
                    fin = np.isfinite(v).all(1) & inb
                    gid_p = (pts[fin, 1].astype(int) // PATCH_14) * nbx + (pts[fin, 0].astype(int) // PATCH_14)
                    cnt_p, v_pred = grouped_var(v[fin], gid_p, n_groups)
                    for i in np.nonzero((cnt_r >= MIN_PTS14) & (cnt_p >= MIN_PTS14))[0]:
                        rows.append({"seq": seq, "k": k, "N": N, "variant": var, "rep": rep, "var_ref": v_ref[i],
                                     "var_pred": v_pred[i], "n_ref": int(cnt_r[i]), "n_pts": int(cnt_p[i])})
    return rows


def summarise_starts(df):
    """Purpose: pooled results per scene, window and variant: bias of the estimated spread against the people's real spread
    (percent), correlation over patches, and the ratio of start points to real people (1.0 = a perfect count)."""
    out = []
    for (seq, N, var), t in df.groupby(["seq", "N", "variant"]):
        if len(t) < 3:
            continue
        tot = t.var_ref.sum()
        out.append({"seq": seq, "N_frames": N, "variant": var, "patches": len(t),
                    "bias_%": round(100 * (t.var_pred.sum() - tot) / tot, 1),
                    "corr": round(float(np.corrcoef(t.var_ref, t.var_pred)[0, 1]), 3),
                    "points/people": round(float(t.n_pts.sum() / t.n_ref.sum()), 2)})
    return pd.DataFrame(out)


# ---------------- run ----------------
rng14 = np.random.default_rng(14)
keep_root = DATA_ROOT
rows14 = []
try:
    for seq, root in TRAJ_SCENES.items():
        DATA_ROOT = root
        P, pr = tracks[seq]
        flows = store_flows(seq, list(range(0, max(K14) + max(GAPS14))))
        rows14 += start_rows(seq, P, pr, best_off[seq], flows, K14, GAPS14, rng14)
        del flows
finally:
    DATA_ROOT = keep_root
df14 = pd.DataFrame(rows14)
df14.to_csv(OUT_DIR / "e006_start_point_table.csv", index=False)
tab14 = summarise_starts(df14)
tab14.to_csv(OUT_DIR / "e006_start_point_summary.csv", index=False)
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 300)
print("--- Var(v) of flow-followed start points vs the people's real velocity spread, per start-point variant ---")
print(tab14.to_string(index=False))

# gate view at 1.5 s: bias per scene for every variant, worst scene, and PASS/FAIL against the +-10 percent limit
g = tab14[tab14.N_frames == max(GAPS14)].pivot(index="variant", columns="seq", values="bias_%")
g = g.reindex([v for v in VARIANTS14 if v in g.index])
g["worst_abs"] = g.abs().max(axis=1)
g["gate"] = np.where(g.worst_abs <= GATE14, "PASS", "FAIL")
print(f"\n--- bias % at {max(GAPS14)} frames (1.5 s), worst scene, gate = worst |bias| <= {GATE14:.0f}% ---")
print(g.round(1).to_string())

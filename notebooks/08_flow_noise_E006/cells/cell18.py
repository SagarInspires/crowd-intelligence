# ===== E006 · Cell 18 · Confirmatory test on NEW crowds (IM02, IM04): candidates declared BEFORE running =====
import os
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates, uniform_filter
from tqdm.auto import tqdm

# ---- needs only: Recovery cell A (loaders, DATA_ROOT, OUT_DIR, find_root_with), Cell 3 (model) and the
#      saved table e006_roundtrip_table.csv from Cell 17. Everything else is defined here. ----
NEW18     = ["IM02", "IM04"]            # new crowds (dataset tub-crowdflow-small3); trajectories are in crowdflow_persontraj
GAPS18    = [16, 37]                    # window length in frames (16 = 0.64 s is the PRIMARY test, 37 = 1.5 s is reported next to it)
K18       = list(range(0, 60, 6))       # start frames
PATCH_18  = 160                         # patch size (px)
MIN_18    = 8                           # a patch needs at least this many reference people and kept points
REPS18    = 2                           # repeats of the random start variants
TAU18     = 2.0                         # round-trip tolerance in px (same as Cell 17, declared there before running)
GATE18    = 10.0                        # PRE-DECLARED: a candidate passes if |held-out bias| <= 10 % in BOTH new crowds
# PRE-DECLARED candidates (estimator, start variant); nothing else is judged:
CANDIDATES = [("box4", "true"), ("box4", "jitter2"), ("box4_fb2", "jitter2"), ("box4_fb2", "jitter5")]
# body sizes (px) measured in Cell 15 for the old crowds (IM01 and hDyn are the same crowd):
OLD_BODY = {"IM01+hDyn": 26.5, "IM03": 11.3, "IM05": 17.4}
OLD_GROUP = {"IM01": "IM01+hDyn", "IM01_hDyn": "IM01+hDyn", "IM03": "IM03", "IM05": "IM05"}


# ----- small helpers (same logic as Cells 9, 12, 13, 14, 15, 17) -----
def find_traj_file(seq):
    """Purpose: find <seq>/personTrajectories.txt anywhere under /kaggle/input. Returns the path or None."""
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.basename(root) == seq and "personTrajectories.txt" in files:
            return os.path.join(root, "personTrajectories.txt")
    return None


def load_person_tracks(path):
    """Purpose: read personTrajectories.txt (one line per person, pairs stored as (y, x), (0,0) = absent). Returns P (persons, frames, 2)
    with P[..., 0] = x and P[..., 1] = y, and a boolean array 'present'."""
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
    return P[..., ::-1].copy(), present


def sample_at(field, pos):
    """Purpose: read a (H, W, 2) field at float pixel positions (x, y) with the nearest pixel; returns values, an 'inside' flag, and the pixel indices."""
    H, W = field.shape[:2]
    xi, yi = np.rint(pos[:, 0]).astype(int), np.rint(pos[:, 1]).astype(int)
    inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    out = np.full((len(pos), 2), np.nan)
    out[inside] = field[yi[inside], xi[inside]]
    return out, inside, xi, yi


def grouped_var(vals, gid, n_groups):
    """Purpose: per group (patch) the number of points and the variance summed over the two axes (bincount, fast)."""
    cnt = np.bincount(gid, minlength=n_groups)
    out = np.zeros(n_groups)
    for c in range(vals.shape[1]):
        s1 = np.bincount(gid, weights=vals[:, c], minlength=n_groups)
        s2 = np.bincount(gid, weights=vals[:, c] ** 2, minlength=n_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            out += s2 / cnt - (s1 / cnt) ** 2
    return cnt, out


def bilinear(field, pos):
    """Purpose: read a (H, W, 2) field at float positions (x, y) with bilinear interpolation; positions outside the image give NaN."""
    H, W = field.shape[:2]
    out = np.stack([map_coordinates(field[..., c].astype(np.float32), [pos[:, 1], pos[:, 0]], order=1,
                                    mode="constant", cval=np.nan) for c in range(2)], axis=1)
    out[(pos[:, 0] < 0) | (pos[:, 0] > W - 1) | (pos[:, 1] < 0) | (pos[:, 1] > H - 1)] = np.nan
    return out


def advect(flows, k, N, pos0):
    """Purpose: follow points with the flow itself for N frames starting at frame k; returns the average velocity (displacement / N)."""
    p = pos0.copy()
    for i in range(N):
        p = p + bilinear(flows[k + i], p)
    return (p - pos0) / N


def advect_back(bflows, k, N, p_end):
    """Purpose: follow points BACKWARDS from frame k+N to frame k with the backward flows; returns where they end up in frame k."""
    q = p_end.copy()
    for i in range(N - 1, -1, -1):
        q = q + bilinear(bflows[k + i], q)
    return q


def blur_flows(flows, masks, r):
    """Purpose: average every flow field over a (2r+1)-square counting only person pixels (keep the raw flow where fewer than 5 percent are people)."""
    out, size = {}, 2 * r + 1
    for j, f in flows.items():
        f32 = f.astype(np.float32)
        m = masks[j].astype(np.float32)
        num = uniform_filter(f32 * m[..., None], size=(size, size, 1), mode="nearest")
        den = uniform_filter(m, size=size, mode="nearest")[..., None]
        out[j] = np.where(den > 0.05, num / np.maximum(den, 1e-6), f32).astype(np.float16)
    return out


def store_flows(seq, frames, backward=False):
    """Purpose: run SEA-RAFT on every consecutive pair j -> j+1 (or j+1 -> j if backward) and keep the full fields in half precision."""
    out = {}
    for j in tqdm(frames, desc=f"{'backward' if backward else 'forward'} flows {seq}"):
        a, b = (load_frame(seq, j + 1), load_frame(seq, j)) if backward else (load_frame(seq, j), load_frame(seq, j + 1))
        pred, _ = real_predict(seq, j, a, b)
        out[j] = pred.astype(np.float16)
    return out


def best_offset(seq, P, present):
    """Purpose: find the shift between the trajectory index and the image index (-1, 0 or 1): the one for which a person's one-frame step best
    matches the ground-truth flow at the person's pixel (median difference, only on person pixels). Printed, because the old scenes all gave -1."""
    best, res = None, {}
    for off in (-1, 0, 1):
        d = []
        for k in range(0, 60, 6):
            a, b = k + off, k + off + 1
            if a < 0 or b >= P.shape[1]:
                continue
            ok = present[:, a] & present[:, b]
            if ok.sum() == 0:
                continue
            gt, m = load_gt_flow(seq, k), load_mask(seq, k) > 0
            v, inside, xi, yi = sample_at(gt, P[ok, a])
            use = inside.copy()
            use[inside] &= m[yi[inside], xi[inside]]
            d += list(np.linalg.norm((P[ok, b] - P[ok, a])[use] - v[use], axis=1))
        res[off] = float(np.nanmedian(d)) if d else np.inf
    best = min(res, key=res.get)
    print(f"  {seq}: median |step - GT flow| per offset", {k: round(v, 3) for k, v in res.items()}, "-> offset", best)
    return best


def body_size(seq, P, present, off):
    """Purpose: rough person size in px: sqrt(person-mask area / number of people present), median over some frames."""
    d = []
    for k in range(0, 60, 6):
        a = k + off
        if 0 <= a < P.shape[1] and present[:, a].sum() > 0:
            d.append(np.sqrt((load_mask(seq, k) > 0).sum() / present[:, a].sum()))
    return float(np.median(d))


def make_starts(variant, pos_true, rng):
    """Purpose: start points for the candidate variants: 'true' = true positions; 'jitter2' / 'jitter5' = true positions plus Gaussian error of 2 / 5 px."""
    sigma = {"true": 0.0, "jitter2": 2.0, "jitter5": 5.0}[variant]
    return pos_true + rng.normal(0, sigma, pos_true.shape) if sigma > 0 else pos_true.copy()


def start_rows18(seq, P, present, off, flows_by_base, bflows_by_base, ks, gaps, rng):
    """Purpose: per start frame, window, start variant and estimator (point / box4, each without and with the tau = 2 px round-trip filter):
    spread of the estimated velocities of the (kept) points per patch against the spread of the people's real velocity over the window.
    Same reference and patch rules as Cells 14-17. Returns a list of rows."""
    rows = []
    for k in tqdm(ks, desc=f"{seq} start points", leave=False):
        a = k + off
        m = load_mask(seq, k) > 0
        nbx = -(-m.shape[1] // PATCH_18)
        n_groups = nbx * (-(-m.shape[0] // PATCH_18))
        for N in gaps:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a:b + 1].all(1)
            if ok.sum() < MIN_18:
                continue
            pos0 = P[ok][:, a]
            ref = (P[ok][:, b] - P[ok][:, a]) / N
            _, inside, xi, yi = sample_at(np.zeros((m.shape[0], m.shape[1], 2)), pos0)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            if sel.sum() < MIN_18:
                continue
            pos_true = pos0[sel]
            gid_r = (yi[sel] // PATCH_18) * nbx + (xi[sel] // PATCH_18)
            cnt_r, v_ref = grouped_var(ref[sel], gid_r, n_groups)
            for var in ["true", "jitter2", "jitter5"]:
                for rep in range(1 if var == "true" else REPS18):
                    pts = make_starts(var, pos_true, rng)
                    for base in flows_by_base:
                        fl, bfl = flows_by_base[base], bflows_by_base[base]
                        v = advect(fl, k, N, pts)
                        err = np.linalg.norm(advect_back(bfl, k, N, pts + N * v) - pts, axis=1)
                        inb = (pts[:, 0] >= 0) & (pts[:, 0] < m.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < m.shape[0])
                        fin = np.isfinite(v).all(1) & np.isfinite(err) & inb
                        for tau in (None, TAU18):
                            keep = fin if tau is None else fin & (err <= tau)
                            if keep.sum() < MIN_18:
                                continue
                            gid_p = (pts[keep, 1].astype(int) // PATCH_18) * nbx + (pts[keep, 0].astype(int) // PATCH_18)
                            cnt_p, v_pred = grouped_var(v[keep], gid_p, n_groups)
                            cnt_all = np.bincount((pts[fin, 1].astype(int) // PATCH_18) * nbx + (pts[fin, 0].astype(int) // PATCH_18), minlength=n_groups)
                            name = base if tau is None else f"{base}_fb{tau:g}"
                            for i in np.nonzero((cnt_r >= MIN_18) & (cnt_p >= MIN_18))[0]:
                                rows.append({"seq": seq, "k": k, "N": N, "estimator": name, "variant": var, "rep": rep,
                                             "var_ref": v_ref[i], "var_pred": v_pred[i], "n_kept": int(cnt_p[i]), "n_all": int(cnt_all[i])})
    return rows


def pooled_bias(t, a=1.0):
    """Purpose: pooled bias in percent of the summed reference variance after multiplying the prediction by a."""
    return 100 * (a * t.var_pred.sum() - t.var_ref.sum()) / t.var_ref.sum()


def gain(t):
    """Purpose: single multiplier a (reference = a * prediction, through the origin) for a set of patch rows."""
    return float((t.var_ref * t.var_pred).sum() / (t.var_pred ** 2).sum())


# ---------------- run ----------------
DATA_ROOT_3 = find_root_with(NEW18[0])
assert DATA_ROOT_3 is not None, "IM02 not found: attach the dataset tub-crowdflow-small3 (restart, then run Recovery A, Cell 3 and Recovery B first)."
if "real_predict" not in globals():
    real_predict = lambda seq, k, a, b: predict(model, args, a, b)        # model call from Cell 3
keep_root = DATA_ROOT
rows18, info = [], {}
try:
    DATA_ROOT = DATA_ROOT_3
    for seq in NEW18:
        path = find_traj_file(seq)
        assert path is not None, f"trajectories of {seq} not found: attach the dataset crowdflow_persontraj"
        P, pr = load_person_tracks(path)
        off = best_offset(seq, P, pr)
        info[seq] = {"people": len(P), "offset": off, "body_px": round(body_size(seq, P, pr, off), 1)}
        frames = list(range(0, max(K18) + max(GAPS18)))
        fwd, bwd = store_flows(seq, frames), store_flows(seq, frames, backward=True)
        masks = {j: load_mask(seq, j) > 0 for j in range(0, max(frames) + 2)}
        fb = {"point": fwd, "box4": blur_flows(fwd, masks, 4)}
        bb = {"point": bwd, "box4": blur_flows(bwd, {j: masks[j + 1] for j in bwd}, 4)}
        rows18 += start_rows18(seq, P, pr, off, fb, bb, K18, GAPS18, np.random.default_rng(18))
        del fwd, bwd, masks, fb, bb
finally:
    DATA_ROOT = keep_root
new = pd.DataFrame(rows18)
new.to_csv(OUT_DIR / "e006_confirm_new_crowds.csv", index=False)
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 300)
print("\n--- new crowds ---")
print(pd.DataFrame(info).T.to_string())

old = pd.read_csv(OUT_DIR / "e006_roundtrip_table.csv")                      # Cell 17 table: old crowds IM01, IM01_hDyn, IM03, IM05
old = old.assign(group=old.seq.map(OLD_GROUP))

# 1) raw bias per new crowd for every estimator / start variant (candidates marked)
raw = (new.groupby(["N", "estimator", "variant", "seq"]).apply(lambda t: pd.Series({"bias_%": round(pooled_bias(t), 1), "patches": len(t),
        "kept_%": round(100 * t.n_kept.sum() / t.n_all.sum())})).reset_index())
raw["candidate"] = [(e, v) in CANDIDATES for e, v in zip(raw.estimator, raw.variant)]
print("\n--- raw bias % on the NEW crowds (candidates marked; no gain applied) ---")
print(raw[raw.candidate].pivot_table(index=["N", "estimator", "variant"], columns="seq", values=["bias_%", "patches", "kept_%"]).to_string())

# 2) the confirmatory gate: one gain fitted on the OLD crowds only, applied to each new crowd
gate = []
for N in GAPS18:
    for est, var in CANDIDATES:
        tr = old[(old.N == N) & (old.estimator == est) & (old.variant == var)]
        a = gain(tr) if len(tr) >= 3 else np.nan
        row = {"N": N, "estimator": est, "variant": var, "gain_from_old": round(a, 2)}
        for seq in NEW18:
            te = new[(new.N == N) & (new.estimator == est) & (new.variant == var) & (new.seq == seq)]
            row[f"{seq}_raw_%"] = round(pooled_bias(te), 1) if len(te) >= 3 else np.nan
            row[f"{seq}_gained_%"] = round(pooled_bias(te, a), 1) if len(te) >= 3 else np.nan
        g = [row[f"{s}_gained_%"] for s in NEW18]
        row["gate"] = "PASS" if all(np.isfinite(g)) and max(abs(x) for x in g) <= GATE18 else "FAIL"
        gate.append(row)
gate = pd.DataFrame(gate)
print(f"\n--- CONFIRMATORY GATE: gain fitted on the old crowds, applied to the new crowds; PASS = both |bias| <= {GATE18:.0f}% ---")
print(gate.to_string(index=False))

# 3) body size against bias, box4 with true starts (5 independent crowds), at 16 frames
sizes = dict(OLD_BODY, **{s: info[s]["body_px"] for s in NEW18})
pts = []
for grp, t in old[(old.N == 16) & (old.estimator == "box4") & (old.variant == "true")].groupby("group"):
    pts.append({"crowd": grp, "body_px": sizes[grp], "bias_%": round(pooled_bias(t), 1)})
for seq in NEW18:
    t = new[(new.N == 16) & (new.estimator == "box4") & (new.variant == "true") & (new.seq == seq)]
    if len(t) >= 3:
        pts.append({"crowd": seq, "body_px": sizes[seq], "bias_%": round(pooled_bias(t), 1)})
bs = pd.DataFrame(pts).sort_values("body_px")
print("\n--- body size vs bias (box4, true start, 16 frames; negative = spread too small) ---")
print(bs.to_string(index=False))
if len(bs) >= 4:
    print("Spearman rank correlation (n =", len(bs), "):", round(float(bs.body_px.rank().corr(bs["bias_%"].rank())), 2),
          " (pre-declared reading: >= +0.8 supports 'smaller people are shrunk more'; with 5 crowds this is only indicative)")

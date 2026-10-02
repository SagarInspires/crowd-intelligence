# ===== E006 · Cell 22 · Confirmatory test of Farnebäck (fixed settings) on all six crowds, with leave-crowd-out gain =====
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

# ---- needs Cell 18 run in this session (find_traj_file, load_person_tracks, sample_at, grouped_var, find_root_with, load_mask, load_frame) ----
SEQS22   = ["IM01", "IM01_hDyn", "IM03", "IM05", "IM02", "IM04"]
K22      = list(range(0, 60, 6))        # frames
GAPS22   = [1, 2, 4]                    # direct flow over N frames, divided by N (declared before running)
PATCH_22 = 160                          # patch size (px)
MIN_22   = 8                            # people per patch
GATE22   = 10.0                         # +-10 % gate (declared before running)
FARNE22  = dict(pyr_scale=0.5, levels=5, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)   # FIXED, same as Cell 21, not tuned
# PRE-DECLARED PASS RULE for a gap N: (a) raw: all five independent crowds within +-10 %, or (b) one gain fitted on the other crowds
# (leave-crowd-out, IM01 and IM01_hDyn held out together) gives all five held-out biases within +-10 %.
# IM04 is expected to fail; it is reported separately and counts. 6 cells (3 gaps x raw/gain) are looked at, so a single pass is only suggestive.


def label_of(seq):
    """Purpose: crowd label for leave-crowd-out; IM01 and IM01_hDyn show the same crowd and are merged."""
    return "IM01+hDyn" if seq in ("IM01", "IM01_hDyn") else seq


def farneback_flow(img0, img1):
    """Purpose: Farnebäck dense optical flow (OpenCV) between two RGB images with the fixed settings above -> flow (H, W, 2) in px."""
    g0, g1 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, **FARNE22)


def collect(seq, P, present, off=-1):
    """Purpose: for one crowd, run Farnebäck between frame k and k+N for every gap N, read it at each person's true start position
    (divided by N) and return, per gap, the per-patch spread of the real velocity (x_{k+N}-x_k)/N and of the predicted velocity
    (only patches with >= MIN_22 people), plus the share of the real motion recovered."""
    store = {N: ([], [], []) for N in GAPS22}
    for k in tqdm(K22, desc=seq, leave=False):
        m = load_mask(seq, k) > 0
        a = k + off
        nbx = -(-m.shape[1] // PATCH_22)
        img0 = load_frame(seq, k)
        for N in GAPS22:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a] & present[:, b]
            if ok.sum() < MIN_22:
                continue
            flow = farneback_flow(img0, load_frame(seq, k + N))
            pos = P[ok, a]
            step = (P[ok, b] - P[ok, a]) / N
            v, inside, xi, yi = sample_at(flow, pos)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            gid = ((yi[sel] // PATCH_22) * nbx + (xi[sel] // PATCH_22)) + 100000 * len(store[N][0])
            store[N][0].append(step[sel]); store[N][1].append(v[sel] / N); store[N][2].append(gid)
    res = {}
    for N in GAPS22:
        st, pr, gd = np.vstack(store[N][0]), np.vstack(store[N][1]), np.concatenate(store[N][2])
        u, inv = np.unique(gd, return_inverse=True)
        cnt, vr = grouped_var(st, inv, len(u))
        _, vp = grouped_var(pr, inv, len(u))
        good = cnt >= MIN_22
        res[N] = dict(vr=vr[good], vp=vp[good], rec=float((st * pr).sum() / (st ** 2).sum()), n_patch=int(good.sum()))
    return res


def bias(vr, vp, a=1.0):
    """Purpose: pooled bias (percent) of the summed predicted spread (times gain a) relative to the summed real spread."""
    return 100.0 * (a * vp.sum() - vr.sum()) / vr.sum()


def gain_of(parts):
    """Purpose: best single gain through the origin, a = sum(ref * pred) / sum(pred^2), over the per-patch spreads of the listed crowds."""
    vr = np.concatenate([p["vr"] for p in parts]); vp = np.concatenate([p["vp"] for p in parts])
    return float((vr * vp).sum() / (vp ** 2).sum())


# ---------------- run ----------------
keep_root = DATA_ROOT
R = {}                                   # R[(label, seq)][N]
try:
    for seq in SEQS22:
        DATA_ROOT = find_root_with(seq)
        P, pr = load_person_tracks(find_traj_file(seq))
        R[seq] = collect(seq, P, pr)
finally:
    DATA_ROOT = keep_root

# merge IM01 and hDyn into one crowd
crowds = {}
for seq in SEQS22:
    crowds.setdefault(label_of(seq), []).append(R[seq])
merged = {c: {N: dict(vr=np.concatenate([r[N]["vr"] for r in rs]), vp=np.concatenate([r[N]["vp"] for r in rs]),
                      rec=float(np.mean([r[N]["rec"] for r in rs])), n_patch=sum(r[N]["n_patch"] for r in rs)) for N in GAPS22}
          for c, rs in crowds.items()}

rows = []
for N in GAPS22:
    for c, d in merged.items():
        others = [merged[o][N] for o in merged if o != c]
        a = gain_of(others)
        rows.append({"gap_N": N, "crowd": c, "patches": d[N]["n_patch"], "recovered": round(d[N]["rec"], 2),
                     "raw_bias_%": round(bias(d[N]["vr"], d[N]["vp"]), 1), "gain_from_others": round(a, 2),
                     "heldout_bias_%": round(bias(d[N]["vr"], d[N]["vp"], a), 1)})
d22 = pd.DataFrame(rows)
d22.to_csv(OUT_DIR / "e006_farneback_confirm.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 30)
print("--- Farnebäck, true starts, direct flow over N frames (divided by N); bias of the velocity spread per crowd ---")
print(d22.to_string(index=False))

print(f"\n--- PRE-DECLARED PASS RULE (all five crowds within +-{GATE22:.0f}%) ---")
for N in GAPS22:
    t = d22[d22.gap_N == N]
    wr, wg = t["raw_bias_%"].abs().max(), t["heldout_bias_%"].abs().max()
    wr4, wg4 = t[t.crowd != "IM04"]["raw_bias_%"].abs().max(), t[t.crowd != "IM04"]["heldout_bias_%"].abs().max()
    print(f"N={N}: raw worst {wr:.1f}% -> {'PASS' if wr <= GATE22 else 'FAIL'} | with gain worst {wg:.1f}% -> {'PASS' if wg <= GATE22 else 'FAIL'}"
          f"   (without IM04: raw {wr4:.1f}%, gain {wg4:.1f}%)")

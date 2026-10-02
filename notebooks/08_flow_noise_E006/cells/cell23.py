# ===== E006 · Cell 23 · Do the results survive noise and compression? (SEA-RAFT vs Farnebäck on degraded frames) =====
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

# ---- needs Cell 18 run in this session (find_traj_file, load_person_tracks, sample_at, grouped_var, find_root_with, load_mask, load_frame, real_predict) ----
CROWDS23 = ["IM01", "IM03", "IM05", "IM02"]      # the four crowds that worked; IM04 is left out (it fails for every method)
K23      = list(range(0, 60, 6))                 # frames
N23      = 2                                     # one gap, direct flow over 2 frames divided by 2 (declared before running)
PATCH_23 = 160
MIN_23   = 8
GATE23   = 10.0                                  # +-10 % gate on all four crowds (declared before running)
CONDS23  = ["clean", "noise_sd4", "noise_sd8", "jpeg_q30"]   # declared before running: Gaussian noise (std 4 and 8 grey levels of 255, independent in the two frames), JPEG quality 30
# Reference values from earlier cells at N = 2, for a reproduction check of the clean condition (tolerance 2 points; K may differ slightly):
REF23 = {("farneback", "IM03"): -6.6, ("farneback", "IM05"): -4.7, ("farneback", "IM02"): -8.8,
         ("sea_raft", "IM03"): -12.7, ("sea_raft", "IM05"): -6.3, ("sea_raft", "IM02"): -3.4}
# PRE-DECLARED: a method "holds" under a degradation if all four crowds stay within +-10 % raw bias at N = 2.


def degrade(img, cond, seed):
    """Purpose: return a degraded copy of an RGB uint8 image: 'clean' (unchanged), 'noise_sd4' / 'noise_sd8' (Gaussian noise, same seed -> same noise),
    or 'jpeg_q30' (JPEG compression and decompression at quality 30)."""
    if cond == "clean":
        return img
    if cond.startswith("noise_sd"):
        sd = float(cond.replace("noise_sd", ""))
        rng = np.random.default_rng(seed)
        return np.clip(img.astype(np.float32) + rng.normal(0, sd, img.shape), 0, 255).astype(np.uint8)
    if cond == "jpeg_q30":
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    raise ValueError(cond)


def build_methods23():
    """Purpose: the two flow methods as functions (RGB image 0, RGB image 1, seq, k) -> flow (H, W, 2) in px:
    sea_raft (the model used so far, via real_predict) and farneback (OpenCV, same fixed settings as Cells 21-22)."""
    def farne(a, b, seq, k):
        g0, g1 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 5, 15, 3, 7, 1.5, 0)
    return {"sea_raft": lambda a, b, seq, k: real_predict(seq, k, a, b)[0], "farneback": farne}


def evaluate23(seq, P, present, fn, cond, off=-1):
    """Purpose: for one crowd, one flow method and one degradation: degrade both frames, run the method between frame k and k+N23, read it at
    each person's TRUE start position (divided by N23) and compare with the real velocity (x_{k+N}-x_k)/N. Returns the pooled bias (percent)
    of the per-patch velocity spread and the share of the real motion recovered."""
    st_l, pr_l, gd_l = [], [], []
    for k in K23:
        m = load_mask(seq, k) > 0
        a = k + off
        b = a + N23
        if a < 0 or b >= P.shape[1] or k + N23 > 119:
            continue
        ok = present[:, a] & present[:, b]
        if ok.sum() < MIN_23:
            continue
        nbx = -(-m.shape[1] // PATCH_23)
        i0 = degrade(load_frame(seq, k), cond, 2 * k)
        i1 = degrade(load_frame(seq, k + N23), cond, 2 * k + 1)
        flow = fn(i0, i1, seq, k)
        step = (P[ok, b] - P[ok, a]) / N23
        v, inside, xi, yi = sample_at(flow, P[ok, a])
        sel = inside.copy()
        sel[inside] &= m[yi[inside], xi[inside]]
        st_l.append(step[sel]); pr_l.append(v[sel] / N23)
        gd_l.append(((yi[sel] // PATCH_23) * nbx + (xi[sel] // PATCH_23)) + 100000 * len(st_l))
    st, pr, gd = np.vstack(st_l), np.vstack(pr_l), np.concatenate(gd_l)
    u, inv = np.unique(gd, return_inverse=True)
    cnt, vr = grouped_var(st, inv, len(u))
    _, vp = grouped_var(pr, inv, len(u))
    good = cnt >= MIN_23
    return round(100 * (vp[good].sum() - vr[good].sum()) / vr[good].sum(), 1), round(float((st * pr).sum() / (st ** 2).sum()), 2)


# ---------------- run ----------------
methods = build_methods23()
keep_root = DATA_ROOT
rows23 = []
try:
    for seq in tqdm(CROWDS23, desc="crowds"):
        DATA_ROOT = find_root_with(seq)
        P, pr = load_person_tracks(find_traj_file(seq))
        for name, fn in methods.items():
            for cond in tqdm(CONDS23, desc=f"{seq} {name}", leave=False):
                b_, r_ = evaluate23(seq, P, pr, fn, cond)
                rows23.append({"method": name, "condition": cond, "crowd": seq, "bias_%": b_, "recovered": r_})
finally:
    DATA_ROOT = keep_root
d23 = pd.DataFrame(rows23)
d23.to_csv(OUT_DIR / "e006_degradation_check.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(f"--- bias % of the velocity spread, direct flow over {N23} frames, true starts ---")
print(d23.pivot_table(index=["method", "condition"], columns="crowd", values="bias_%").round(1).to_string())
print("\n--- share of real motion recovered ---")
print(d23.pivot_table(index=["method", "condition"], columns="crowd", values="recovered").round(2).to_string())

print("\n--- reproduction check of the clean condition against earlier cells (tolerance 2 points) ---")
for (m_, c_), ref in REF23.items():
    got = d23[(d23.method == m_) & (d23.condition == "clean") & (d23.crowd == c_)]["bias_%"].iloc[0]
    print(f"{m_:10s} {c_}: now {got:+.1f}% earlier {ref:+.1f}%  ->", "ok" if abs(got - ref) <= 2 else "differs")

print(f"\n--- PRE-DECLARED CHECK: all four crowds within +-{GATE23:.0f}% ---")
for m_ in methods:
    for c_ in CONDS23:
        t = d23[(d23.method == m_) & (d23.condition == c_)]
        w = t["bias_%"].abs().max()
        print(f"{m_:10s} {c_:10s} worst {w:5.1f}%  mean recovery {t['recovered'].mean():.2f}  ->", "HOLDS" if w <= GATE23 else "fails")

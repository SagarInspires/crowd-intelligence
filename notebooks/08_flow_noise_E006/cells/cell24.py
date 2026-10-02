# ===== E006 · Cell 24 · What is special about the slow crowd IM04? Does motion recovery depend on local image texture or crowding? =====
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

# ---- needs Cell 18 run in this session (find_traj_file, load_person_tracks, sample_at, find_root_with, load_mask, load_frame) ----
CROWDS24  = ["IM01", "IM03", "IM05", "IM02", "IM04"]    # five independent crowds
K24       = list(range(0, 60, 6))                       # frames
WIN24     = 15                                          # window (px) for local texture and contrast
R_CROWD   = 20                                          # radius (px) for counting neighbours
SPEED24   = [0, 1, 2, 1e9]                              # speed strata in px/frame (the features are compared within a speed stratum)
GAP24     = 0.10                                        # declared: recovery(top feature tertile) - recovery(bottom tertile) >= 0.10
# PRE-DECLARED. Primary feature: local TEXTURE (mean Sobel gradient magnitude of the grey image in a 15x15 window at the person's start).
# "Texture explains IM04" is supported only if BOTH hold: (a) the speed-stratified gap (top minus bottom texture tertile) is >= +0.10 in IM04 AND
# positive in at least 4 of the 5 crowds, and (b) IM04 has the lowest median texture of the five crowds.
# Secondary (reported, less weight): local CONTRAST (std of grey) and CROWDING (neighbours within 20 px; expected negative gap).


def farneback_flow(img0, img1):
    """Purpose: Farnebäck dense optical flow (OpenCV, same fixed settings as Cells 21-23) between two RGB images -> flow (H, W, 2) in px."""
    g0, g1 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 5, 15, 3, 7, 1.5, 0)


def feature_maps(img):
    """Purpose: two per-pixel maps from an RGB image: local texture (mean Sobel gradient magnitude in a WIN24 x WIN24 window) and
    local contrast (standard deviation of grey level in the same window)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    tex = cv2.blur(mag, (WIN24, WIN24))
    mean = cv2.blur(g, (WIN24, WIN24))
    con = np.sqrt(np.maximum(cv2.blur(g * g, (WIN24, WIN24)) - mean ** 2, 0))
    return tex, con


def collect24(seq, P, present, off=-1):
    """Purpose: for one crowd, run one-frame Farnebäck at every kept frame, read it at each person's true start, and return one table row per
    person-frame with the real step, the predicted step, the speed, the local texture and contrast at the start, and the number of neighbours."""
    rows = []
    for k in tqdm(K24, desc=seq, leave=False):
        a = k + off
        if a < 0 or a + 1 >= P.shape[1]:
            continue
        ok = present[:, a] & present[:, a + 1]
        if ok.sum() < 8:
            continue
        m = load_mask(seq, k) > 0
        img0 = load_frame(seq, k)
        flow = farneback_flow(img0, load_frame(seq, k + 1))
        tex, con = feature_maps(img0)
        pos = P[ok, a]
        step = P[ok, a + 1] - P[ok, a]
        v, inside, xi, yi = sample_at(flow, pos)
        sel = inside.copy()
        sel[inside] &= m[yi[inside], xi[inside]]
        nb = cKDTree(pos).query_ball_point(pos, r=R_CROWD, return_length=True) - 1
        rows.append(pd.DataFrame({"crowd": seq, "sx": step[sel, 0], "sy": step[sel, 1], "px": v[sel, 0], "py": v[sel, 1],
                                  "texture": tex[yi[sel], xi[sel]], "contrast": con[yi[sel], xi[sel]], "neighbours": nb[sel]}))
    d = pd.concat(rows, ignore_index=True)
    d["speed"] = np.hypot(d.sx, d.sy)
    return d


def recovery(d):
    """Purpose: share of the real motion recovered, sum(step * predicted) / sum(step^2) over both components, for a set of person-frames."""
    return float((d.sx * d.px + d.sy * d.py).sum() / max((d.sx ** 2 + d.sy ** 2).sum(), 1e-9))


def tertile_gap(d, feat):
    """Purpose: speed-stratified effect of a feature on recovery: within each speed stratum split the persons into tertiles of the feature,
    take recovery(top tertile) - recovery(bottom tertile), and average over strata weighted by the real motion (sum of step^2).
    Returns (gap, recovery bottom tertile, recovery top tertile, persons used)."""
    gaps, w, lo_l, hi_l, n = [], [], [], [], 0
    for lo, hi in zip(SPEED24[:-1], SPEED24[1:]):
        s = d[(d.speed >= lo) & (d.speed < hi)]
        if len(s) < 60:
            continue
        q1, q2 = s[feat].quantile([1 / 3, 2 / 3])
        bot, top = s[s[feat] <= q1], s[s[feat] > q2]
        rb, rt = recovery(bot), recovery(top)
        wt = float((s.sx ** 2 + s.sy ** 2).sum())
        gaps.append(rt - rb); w.append(wt); lo_l.append(rb); hi_l.append(rt); n += len(s)
    if not gaps:
        return np.nan, np.nan, np.nan, 0
    w = np.array(w) / np.sum(w)
    return float(np.dot(gaps, w)), float(np.dot(lo_l, w)), float(np.dot(hi_l, w)), n


# ---------------- run ----------------
keep_root = DATA_ROOT
parts = []
try:
    for seq in CROWDS24:
        DATA_ROOT = find_root_with(seq)
        P, pr = load_person_tracks(find_traj_file(seq))
        parts.append(collect24(seq, P, pr))
finally:
    DATA_ROOT = keep_root
D = pd.concat(parts, ignore_index=True)
D.to_csv(OUT_DIR / "e006_im04_features.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", 30)

print("--- crowd-level view: how IM04 compares on the image features (person start positions) ---")
cl = D.groupby("crowd").agg(person_frames=("speed", "size"), mean_speed=("speed", "mean"), median_texture=("texture", "median"),
                            median_contrast=("contrast", "median"), median_neighbours=("neighbours", "median")).round(2)
cl["recovery"] = [round(recovery(D[D.crowd == c]), 2) for c in cl.index]
print(cl.to_string())

print("\n--- within-crowd effect on recovery (speed-stratified, top minus bottom tertile of the feature; recovery in the bottom and top tertile) ---")
res = []
for c in CROWDS24:
    for feat in ["texture", "contrast", "neighbours"]:
        g, rb, rt, n = tertile_gap(D[D.crowd == c], feat)
        res.append({"crowd": c, "feature": feat, "gap": round(g, 3), "rec_bottom": round(rb, 2), "rec_top": round(rt, 2), "n": n})
R24 = pd.DataFrame(res)
R24.to_csv(OUT_DIR / "e006_im04_feature_effects.csv", index=False)
print(R24.pivot(index="crowd", columns="feature", values="gap").to_string())
print()
print(R24.to_string(index=False))

print("\n--- PRE-DECLARED CHECK (texture) ---")
tg = R24[R24.feature == "texture"].set_index("crowd")["gap"]
a_ok = (tg["IM04"] >= GAP24) and ((tg > 0).sum() >= 4)
b_ok = cl["median_texture"].idxmin() == "IM04"
print(f"(a) texture gap IM04 = {tg['IM04']:+.3f} (need >= +{GAP24}), crowds with positive gap = {(tg > 0).sum()}/5 (need >= 4) -> {'ok' if a_ok else 'NOT met'}")
print(f"(b) lowest median texture = {cl['median_texture'].idxmin()} (need IM04) -> {'ok' if b_ok else 'NOT met'}")
print("TEXTURE EXPLAINS IM04:", "SUPPORTED" if (a_ok and b_ok) else "NOT SUPPORTED")

# ===== E006 · Cell 5 · Bias of Var(v) as a function of the averaging scale =====
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ---- settings (BLOCK, SEQS_RUN, real_predict and OUT_DIR come from Cell 4 / Cell 3) ----
SCALES   = [1, 4, 8, 16, 32]     # averaging window width in pixels (1 = pixel level, same as Cell 4)
MIN_PTS  = 8                      # a patch needs at least this many sample points to be used
MIN_FILL = 0.25                   # a sample point must have at least this share of crowd pixels in its window


def coarse_field(flow, m, s):
    """Purpose: average a flow field over s x s windows, counting ONLY crowd pixels (m = 1 inside the
    crowd, 0 outside). Returns the averaged flow (H, W, 2) and the share of crowd pixels in each window
    (H, W). For s = 1 the flow is returned unchanged."""
    if s == 1:
        return flow, m.astype(np.float32)
    fm = flow * m[..., None]
    num = cv2.boxFilter(fm.astype(np.float32), -1, (s, s), normalize=True)
    den = cv2.boxFilter(m.astype(np.float32), -1, (s, s), normalize=True)
    return num / np.maximum(den[..., None], 1e-6), den


def grouped_var(vals, gid, n_groups):
    """Purpose: for every group (patch) compute the number of points and the variance of the values
    summed over the two flow axes. Done with bincount so it is fast."""
    cnt = np.bincount(gid, minlength=n_groups)
    out = np.zeros(n_groups)
    for c in range(vals.shape[1]):
        s1 = np.bincount(gid, weights=vals[:, c], minlength=n_groups)
        s2 = np.bincount(gid, weights=vals[:, c] ** 2, minlength=n_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            out += s2 / cnt - (s1 / cnt) ** 2
    return cnt, out


def scale_rows(gt, pred, inside, s, seq, k):
    """Purpose: for one frame pair and one averaging scale: average true and predicted flow, take sample points on
    a grid inside the crowd (spacing = half the window, so windows overlap and every patch gets enough points),
    group them into BLOCK x BLOCK patches, and return per-patch var_gt, var_pred, var_err and the cross term,
    exactly as in Cell 4 but at the coarser scale."""
    m = inside.astype(np.float32)
    cg, den = coarse_field(np.nan_to_num(gt), m, s)
    cp, _ = coarse_field(pred, m, s)
    H, W = inside.shape
    sp = max(1, s // 2)
    ys, xs = np.meshgrid(np.arange(sp // 2, H, sp), np.arange(sp // 2, W, sp), indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()
    keep = den[ys, xs] >= MIN_FILL
    ys, xs = ys[keep], xs[keep]
    if len(ys) == 0:
        return []
    g, p = cg[ys, xs], cp[ys, xs]
    nbx = W // BLOCK
    ok = (ys < (H // BLOCK) * BLOCK) & (xs < nbx * BLOCK)
    g, p, ys, xs = g[ok], p[ok], ys[ok], xs[ok]
    gid = (ys // BLOCK) * nbx + (xs // BLOCK)
    n_groups = (H // BLOCK) * nbx
    cnt, vg = grouped_var(g, gid, n_groups)
    _, vp = grouped_var(p, gid, n_groups)
    _, ve = grouped_var(p - g, gid, n_groups)
    return [{"seq": seq, "k": k, "scale": s, "var_gt": vg[i], "var_pred": vp[i], "var_err": ve[i],
             "cross": vp[i] - vg[i] - ve[i]} for i in np.nonzero(cnt >= MIN_PTS)[0]]


def summarise_scales(df):
    """Purpose: pooled percentages per sequence and scale (relative to the true spread): net bias, noise
    (error spread), cross term, the correlation between measured and true spread over patches, and the
    best single multiplier a (true = a * measured) fitted through the origin. Rows with few patches are
    less reliable, so the number of patches is shown."""
    out = []
    for (s, sc), d in df.groupby(["seq", "scale"]):
        tot = d.var_gt.sum()
        out.append({"seq": s, "scale_px": sc, "patches": len(d), "mean_var_gt": round(d.var_gt.mean(), 3),
                    "net_bias_%": round(100 * (d.var_pred.sum() - tot) / tot, 1),
                    "noise_%": round(100 * d.var_err.sum() / tot, 1),
                    "cross_%": round(100 * d.cross.sum() / tot, 1),
                    "corr(true,measured)": round(float(np.corrcoef(d.var_gt, d.var_pred)[0, 1]), 3) if len(d) > 2 else np.nan,
                    "best_gain_a": round(float((d.var_gt * d.var_pred).sum() / (d.var_pred ** 2).sum()), 2)})
    return pd.DataFrame(out)


# ---------------- run ----------------
rows5 = []
for s in SEQS_RUN:
    for k in tqdm(list_flow_indices(s), desc=f"E006 scales {s}"):
        img0, img1 = load_frame(s, k), load_frame(s, k + 1)
        gt = load_gt_flow(s, k)
        inside = (load_mask(s, k) > 0) & np.isfinite(gt).all(axis=2)
        pred, _ = real_predict(s, k, img0, img1)               # the model call from Cell 4
        for sc in SCALES:
            rows5 += scale_rows(gt, pred, inside, sc, s, k)
df5 = pd.DataFrame(rows5)
df5.to_csv(OUT_DIR / "e006_scale_table.csv", index=False)
tab = summarise_scales(df5)
pd.set_option("display.width", 240, "display.max_columns", 30)
print("\n--- bias of Var(v) vs averaging scale (percent of the true Var(v), pooled) ---")
print(tab.to_string(index=False))
print("(rows with only a few patches, e.g. under 30, are unreliable)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for s in SEQS_RUN:
    t = tab[tab.seq == s]
    ax[0].plot(t.scale_px, t["net_bias_%"], "o-", label=f"{s} net")
    ax[0].plot(t.scale_px, t["noise_%"], "s--", label=f"{s} noise")
    ax[0].plot(t.scale_px, t["cross_%"], "^:", label=f"{s} cross")
    ax[1].plot(t.scale_px, t["best_gain_a"], "o-", label=s)
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xscale("log", base=2); ax[0].set_xlabel("averaging window (px)")
ax[0].set_ylabel("% of true Var(v)"); ax[0].legend(fontsize=7)
ax[1].axhline(1, color="k", lw=0.8); ax[1].set_xscale("log", base=2); ax[1].set_xlabel("averaging window (px)")
ax[1].set_ylabel("best multiplier a"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "e006_scale_sweep.png", dpi=90); plt.show()

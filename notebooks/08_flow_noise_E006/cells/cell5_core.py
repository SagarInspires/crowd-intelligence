import cv2, numpy as np, pandas as pd
BLOCK=80; MIN_PTS=8; MIN_FILL=0.5
def coarse_field(flow, m, s):
    if s == 1:
        return flow, m.astype(np.float32)
    fm = flow * m[..., None]
    num = cv2.boxFilter(fm.astype(np.float32), -1, (s, s), normalize=True)
    den = cv2.boxFilter(m.astype(np.float32), -1, (s, s), normalize=True)
    return num / np.maximum(den[..., None], 1e-6), den
def grouped_var(vals, gid, n_groups):
    cnt = np.bincount(gid, minlength=n_groups)
    out = np.zeros(n_groups)
    for c in range(vals.shape[1]):
        s1 = np.bincount(gid, vals[:, c], n_groups)
        s2 = np.bincount(gid, vals[:, c] ** 2, n_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            out += s2 / cnt - (s1 / cnt) ** 2
    return cnt, out
def scale_rows(gt, pred, inside, s, seq, k, spacing_fn):
    m = inside.astype(np.float32)
    cg, den = coarse_field(np.nan_to_num(gt), m, s)
    cp, _ = coarse_field(pred, m, s)
    H, W = inside.shape
    sp = spacing_fn(s)
    ys, xs = np.meshgrid(np.arange(sp // 2, H, sp), np.arange(sp // 2, W, sp), indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()
    keep = den[ys, xs] >= MIN_FILL
    ys, xs = ys[keep], xs[keep]
    if len(ys) == 0: return []
    g, p = cg[ys, xs], cp[ys, xs]
    nbx = W // BLOCK
    ok = (ys < (H // BLOCK) * BLOCK) & (xs < nbx * BLOCK)
    g, p, ys, xs = g[ok], p[ok], ys[ok], xs[ok]
    gid = (ys // BLOCK) * nbx + (xs // BLOCK)
    n_groups = (H // BLOCK) * nbx
    cnt, vg = grouped_var(g, gid, n_groups); _, vp = grouped_var(p, gid, n_groups); _, ve = grouped_var(p - g, gid, n_groups)
    return [{"scale": s, "var_gt": vg[i], "var_pred": vp[i], "var_err": ve[i], "cross": vp[i]-vg[i]-ve[i], "pts": int(cnt[i])} for i in np.nonzero(cnt >= MIN_PTS)[0]]

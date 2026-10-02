# ===== E006 · Cell 8 · Can the gain be predicted from things we can SEE without ground truth? =====
import os
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from sklearn.linear_model import Ridge

# ---- settings (needs Cells 1-7 or the recovery cells A, B plus Cell 7 run in this session) ----
ALL_SCENES  = {"IM01": DATA_ROOT_1, "IM01_hDyn": DATA_ROOT_1, "IM03": DATA_ROOT_2, "IM05": DATA_ROOT_2}
FEAT_SCALES = [8, 16, 32]
FRAME_STEP8 = 2                       # every 2nd frame pair
FEATS       = ["log_vp", "speed", "fill", "texture", "flip_dis"]   # all computable WITHOUT ground truth
GATE_BIAS8  = 10.0                    # pre-registered: held-out |bias| <= 10 % on EVERY scene at 16 and 32 px
GATE_GAIN8  = 0.05                    # and mean nMAE at least 5 % (relative) below the constant-gain baseline


def flip_predict(seq, k, img0, img1):
    """Purpose: run SEA-RAFT on the horizontally mirrored pair and mirror the answer back (dx changes sign).
    A trustworthy flow should not change when the picture is mirrored, so the difference to the normal
    prediction is an uncertainty signal that needs no ground truth."""
    f, _ = real_predict(seq, k, img0[:, ::-1].copy(), img1[:, ::-1].copy())
    f = f[:, ::-1].copy()
    f[..., 0] *= -1.0
    return f


def feature_rows(gt, pred, pred_f, gray_grad, inside, s, seq, k):
    """Purpose: like scale_rows (Cell 5) but ALSO returns, per patch, features that need no ground truth:
    log of the measured spread, mean speed, share of crowd pixels, mean image texture, and how much the
    normal and mirrored predictions disagree. The ground-truth columns are kept only to score the result."""
    m = inside.astype(np.float32)
    cg, den = coarse_field(np.nan_to_num(gt), m, s)
    cp, _ = coarse_field(pred, m, s)
    cf, _ = coarse_field(pred_f, m, s)
    tex = cv2.boxFilter(gray_grad, -1, (s, s), normalize=True)
    H, W = inside.shape
    sp = max(1, s // 2)
    ys, xs = np.meshgrid(np.arange(sp // 2, H, sp), np.arange(sp // 2, W, sp), indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()
    keep = den[ys, xs] >= MIN_FILL
    ys, xs = ys[keep], xs[keep]
    if len(ys) == 0:
        return []
    nbx = W // BLOCK
    ok = (ys < (H // BLOCK) * BLOCK) & (xs < nbx * BLOCK)
    ys, xs = ys[ok], xs[ok]
    g, p, f = cg[ys, xs], cp[ys, xs], cf[ys, xs]
    gid = (ys // BLOCK) * nbx + (xs // BLOCK)
    n_groups = (H // BLOCK) * nbx
    cnt, vg = grouped_var(g, gid, n_groups)
    _, vp = grouped_var(p, gid, n_groups)
    _, vf = grouped_var(f, gid, n_groups)
    den_safe = np.maximum(cnt, 1)
    mean_of = lambda a: np.bincount(gid, weights=a, minlength=n_groups) / den_safe
    speed = mean_of(np.sqrt((p ** 2).sum(1)))
    fill = mean_of(den[ys, xs])
    texture = mean_of(tex[ys, xs])
    flip_dis = mean_of(((p - f) ** 2).sum(1))
    return [{"seq": seq, "k": k, "scale": s, "var_gt": vg[i], "var_pred": vp[i], "log_vp": np.log(vp[i] + 1e-4),
             "speed": speed[i], "fill": fill[i], "texture": texture[i], "flip_dis": flip_dis[i]}
            for i in np.nonzero(cnt >= MIN_PTS)[0]]


def build_feature_table():
    """Purpose: for all four scenes run SEA-RAFT twice per frame pair (normal + mirrored) and collect the
    per-patch table with features. Points DATA_ROOT at the right dataset for each scene and always restores it."""
    global DATA_ROOT
    rows, keep_root = [], DATA_ROOT
    try:
        for seq, root in ALL_SCENES.items():
            DATA_ROOT = root
            for k in tqdm(list_flow_indices(seq)[::FRAME_STEP8], desc=f"features {seq}"):
                img0, img1 = load_frame(seq, k), load_frame(seq, k + 1)
                gt = load_gt_flow(seq, k)
                inside = (load_mask(seq, k) > 0) & np.isfinite(gt).all(axis=2)
                pred, _ = real_predict(seq, k, img0, img1)
                pred_f = flip_predict(seq, k, img0, img1)
                gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY).astype(np.float32)
                grad = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1) ** 2)
                for sc in FEAT_SCALES:
                    rows += feature_rows(gt, pred, pred_f, grad, inside, sc, seq, k)
    finally:
        DATA_ROOT = keep_root
    return pd.DataFrame(rows)


def fit_log_gain(train):
    """Purpose: learn log(true / measured) from the ground-truth-free features with a small ridge regression
    (features standardised on the train scenes). Returns a function that gives the corrected spread."""
    X = train[FEATS].values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    y = np.log((train.var_gt.values + 1e-4) / (train.var_pred.values + 1e-4))
    mdl = Ridge(alpha=10.0).fit((X - mu) / sd, y)
    return lambda d: d.var_pred.values * np.exp(mdl.predict((d[FEATS].values - mu) / sd))


def fit_const_gain(train):
    """Purpose: baseline: the single multiplier a fitted through the origin on the train scenes."""
    a = float((train.var_gt * train.var_pred).sum() / (train.var_pred ** 2).sum())
    return lambda d: a * d.var_pred.values


def leave_one_scene_out(tab):
    """Purpose: for each scale and each scene, fit on the OTHER three scenes and test on the held-out one.
    Compares raw, a constant gain and the feature-based gain (uses `score` from Cell 6)."""
    rows = []
    for sc in FEAT_SCALES:
        d = tab[tab.scale == sc]
        for te in sorted(d.seq.unique()):
            train, test = d[d.seq != te], d[d.seq == te]
            for name, est in [("raw", test.var_pred.values), ("const_gain", fit_const_gain(train)(test)),
                              ("feature_gain", fit_log_gain(train)(test))]:
                rows.append({"scale_px": sc, "test_on": te, "method": name, **score(test, est)})
    return pd.DataFrame(rows)


# ---------------- run ----------------
tab8 = build_feature_table()
tab8.to_csv(OUT_DIR / "e006_feature_table.csv", index=False)
res8 = leave_one_scene_out(tab8)
res8.to_csv(OUT_DIR / "e006_leave_one_scene_out.csv", index=False)
pd.set_option("display.width", 220, "display.max_rows", 200)
print("\n--- how the features relate to the true log-gain (correlation over patches, per scale) ---")
for sc in FEAT_SCALES:
    d = tab8[tab8.scale == sc]
    y = np.log((d.var_gt + 1e-4) / (d.var_pred + 1e-4))
    print(sc, "px:", {f: round(float(np.corrcoef(d[f], y)[0, 1]), 2) for f in FEATS})
for sc in FEAT_SCALES:
    print(f"\n--- scale {sc}px, leave-one-scene-out (bias_% and nMAE per held-out scene) ---")
    print(res8[res8.scale_px == sc].pivot_table(index="test_on", columns="method", values=["bias_%", "nMAE"]).round(3).to_string())

print("\n--- pre-registered gate for the feature-based gain ---")
for sc in [16, 32]:
    r = res8[res8.scale_px == sc]
    fg, cg_ = r[r.method == "feature_gain"], r[r.method == "const_gain"]
    worst = fg["bias_%"].abs().max()
    rel = 1 - fg.nMAE.mean() / cg_.nMAE.mean()
    ok = worst <= GATE_BIAS8 and rel >= GATE_GAIN8
    print(f"{sc} px: worst |bias| {worst:.1f}%  | mean nMAE {fg.nMAE.mean():.3f} vs constant gain {cg_.nMAE.mean():.3f} "
          f"({100*rel:+.1f}%)  ->  {'PASS' if ok else 'FAIL'}")

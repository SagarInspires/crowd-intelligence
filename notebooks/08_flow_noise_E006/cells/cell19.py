# ===== E006 · Cell 19 · Why is IM04 so different? One-step diagnosis of IM04 against IM02 and IM05 =====
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---- needs Cell 18 run in this session (find_traj_file, load_person_tracks, sample_at, grouped_var, bilinear, find_root_with,
#      real_predict and the loaders). One-frame steps only, so it takes about 1 minute per crowd. ----
DIAG19    = ["IM04", "IM02", "IM05"]        # IM04 = the surprise, IM02 = new crowd that behaved, IM05 = old crowd of similar body size
K19       = list(range(0, 60, 6))           # frames
PATCH_19  = 160                             # patch size (px)
MIN_19    = 8                               # people per patch
SPEED_BINS = [0, 1, 2, 4, 1e9]              # true person speed bins in px/frame


def diagnose(seq, P, present, off):
    """Purpose: for one crowd, on one-frame steps (no advection, no windows), measure where SEA-RAFT loses the person velocity:
      (1) pixel level: end-point error and the share of the true flow SEA-RAFT recovers on person pixels, and the true flow vs the
          SEA-RAFT flow on BACKGROUND pixels (a moving camera would show up here);
      (2) person level: the share of the real step SEA-RAFT recovers at the person's position, split by how fast the person moves,
          the share of people who move faster than 4 px/frame, and the pooled bias of the spread (Var) of velocities per patch;
      (3) how much of the total real velocity spread sits in the 3 patches with the most spread.
    Returns a one-row dict."""
    epe, sp_num, sp_den, bg_gt, bg_pr = [], 0.0, 0.0, [], []
    steps, preds, gids = [], [], []
    for k in tqdm(K19, desc=f"diagnose {seq}", leave=False):
        gt = load_gt_flow(seq, k)
        m = load_mask(seq, k) > 0
        pred, _ = real_predict(seq, k, load_frame(seq, k), load_frame(seq, k + 1))
        use = m & np.isfinite(gt).all(2)
        epe.append(np.linalg.norm(pred[use] - gt[use], axis=1))
        sp_num += float((pred[use] * gt[use]).sum()); sp_den += float((gt[use] ** 2).sum())
        bgm = (~m) & np.isfinite(gt).all(2)
        bg_gt.append(np.median(np.linalg.norm(gt[bgm], axis=1))); bg_pr.append(np.median(np.linalg.norm(pred[bgm], axis=1)))
        a = k + off
        if a < 0 or a + 1 >= P.shape[1]:
            continue
        ok = present[:, a] & present[:, a + 1]
        pos = P[ok, a]
        step = P[ok, a + 1] - P[ok, a]
        v, inside, xi, yi = sample_at(pred, pos)
        sel = inside.copy()
        sel[inside] &= m[yi[inside], xi[inside]]
        nbx = -(-m.shape[1] // PATCH_19)
        steps.append(step[sel]); preds.append(v[sel])
        gids.append(((yi[sel] // PATCH_19) * nbx + (xi[sel] // PATCH_19)) + 100000 * len(gids))   # patch ids unique per frame
    steps, preds, gids = np.vstack(steps), np.vstack(preds), np.concatenate(gids)
    speed = np.linalg.norm(steps, axis=1)
    row = {"seq": seq, "people": len(P), "person_steps": len(steps), "mean_speed_px": round(float(speed.mean()), 2),
           "p99_speed_px": round(float(np.percentile(speed, 99)), 2), "frac_speed>4px_%": round(100 * float((speed > 4).mean()), 1),
           "pixel_EPE": round(float(np.concatenate(epe).mean()), 2), "pixel_recovered": round(sp_num / sp_den, 2),
           "bg_median_GT_px": round(float(np.mean(bg_gt)), 3), "bg_median_pred_px": round(float(np.mean(bg_pr)), 3)}
    for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        b = (speed >= lo) & (speed < hi)
        row[f"recovered_{lo:g}-{hi if hi < 1e8 else 'inf'}px"] = round(float((steps[b] * preds[b]).sum() / max((steps[b] ** 2).sum(), 1e-9)), 2) if b.sum() > 20 else np.nan
    u, inv = np.unique(gids, return_inverse=True)
    cnt, v_ref = grouped_var(steps, inv, len(u))
    _, v_pr = grouped_var(preds, inv, len(u))
    good = cnt >= MIN_19
    row["patches"] = int(good.sum())
    row["var_bias_%"] = round(100 * (v_pr[good].sum() - v_ref[good].sum()) / v_ref[good].sum(), 1)
    row["top3_share_of_true_var_%"] = round(100 * float(np.sort(v_ref[good])[-3:].sum() / v_ref[good].sum()), 1)
    return row


# ---------------- run ----------------
keep_root = DATA_ROOT
rows19 = []
try:
    for seq in DIAG19:
        DATA_ROOT = find_root_with(seq)
        P, pr = load_person_tracks(find_traj_file(seq))
        rows19.append(diagnose(seq, P, pr, -1))
finally:
    DATA_ROOT = keep_root
d19 = pd.DataFrame(rows19).set_index("seq")
d19.to_csv(OUT_DIR / "e006_im04_diagnosis.csv")
pd.set_option("display.width", 250, "display.max_columns", 40)
print("--- one-frame diagnosis (rows = crowds) ---")
print(d19.T.to_string())

# ===== E006 · Cell 21 · Is the slow-crowd failure specific to SEA-RAFT? Same one-frame test with other flow methods =====
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

# ---- needs Cell 18 run in this session (find_traj_file, load_person_tracks, sample_at, grouped_var, find_root_with, real_predict and the loaders) ----
CROWDS21  = ["IM04", "IM02", "IM05"]       # IM04 = the slow crowd that failed; IM02 and IM05 = crowds that behaved
K21       = list(range(0, 60, 6))          # frames
PATCH_21  = 160                            # patch size (px)
MIN_21    = 8                              # people per patch
GAPS21    = [1, 4]                         # one-frame flow, and direct flow over 4 frames (divided by 4)
SPEED_BINS21 = [0, 1, 2, 4, 1e9]           # true person speed bins in px/frame
# PRE-DECLARED: a method "helps" if on IM04 the one-frame Var bias is within +-15 % AND it recovers >= 0.85 of the person motion below 1 px/frame,
# AND the one-frame Var bias stays within +-15 % on IM02 and IM05.
LIM21     = 15.0


def build_methods():
    """Purpose: the flow methods to compare, each as a function (RGB image 0, RGB image 1, seq, k) -> flow (H, W, 2) in px:
      sea_raft    : the model used so far (via real_predict);
      farneback   : classical polynomial-expansion flow (OpenCV), good at small smooth motion;
      dis_medium  : OpenCV Dense Inverse Search flow, medium preset;
      raft_large  : torchvision RAFT-large (12 updates), only if the weights can be downloaded (Internet must be on).
    A method that cannot be built is skipped with a message."""
    m = {"sea_raft": lambda a, b, seq, k: real_predict(seq, k, a, b)[0]}

    def farne(a, b, seq, k):
        g0, g1 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 5, 15, 3, 7, 1.5, 0)
    m["farneback"] = farne
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    m["dis_medium"] = lambda a, b, seq, k: dis.calc(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)
    try:
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        w = Raft_Large_Weights.DEFAULT
        net = raft_large(weights=w).eval().cuda()
        prep = w.transforms()

        def raft(a, b, seq, k):
            t0 = torch.from_numpy(a).permute(2, 0, 1)[None].cuda()
            t1 = torch.from_numpy(b).permute(2, 0, 1)[None].cuda()
            t0, t1 = prep(t0, t1)
            with torch.no_grad():
                return net(t0, t1, num_flow_updates=12)[-1][0].permute(1, 2, 0).cpu().numpy()
        m["raft_large"] = raft
    except Exception as e:
        print("raft_large skipped:", repr(e)[:150])
    return m


def evaluate(seq, P, present, fn, off=-1):
    """Purpose: for one crowd and one flow method: for gap N in GAPS21, run the method directly between frame k and k+N, read it at each
    person's true start position (divided by N) and compare with the real velocity (x_{k+N} - x_k) / N. Reports the pooled bias of the
    per-patch velocity spread (percent) and the share of the real motion recovered (sum of step*pred / sum of step^2). For N = 1 it also
    reports the share of the ground-truth flow recovered on person pixels and the recovery split by true speed."""
    out = {}
    pix_num = pix_den = 0.0
    store = {N: ([], [], []) for N in GAPS21}
    for k in tqdm(K21, desc=f"{seq}", leave=False):
        m = load_mask(seq, k) > 0
        a = k + off
        nbx = -(-m.shape[1] // PATCH_21)
        img0 = load_frame(seq, k)
        for N in GAPS21:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a] & present[:, b]
            if ok.sum() < MIN_21:
                continue
            flow = fn(img0, load_frame(seq, k + N), seq, k)
            if N == 1:
                gt = load_gt_flow(seq, k)
                use = m & np.isfinite(gt).all(2)
                pix_num += float((flow[use] * gt[use]).sum()); pix_den += float((gt[use] ** 2).sum())
            pos = P[ok, a]
            step = (P[ok, b] - P[ok, a]) / N
            v, inside, xi, yi = sample_at(flow, pos)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            gid = ((yi[sel] // PATCH_21) * nbx + (xi[sel] // PATCH_21)) + 100000 * len(store[N][0])
            store[N][0].append(step[sel]); store[N][1].append(v[sel] / N); store[N][2].append(gid)
    for N in GAPS21:
        st, pr, gd = np.vstack(store[N][0]), np.vstack(store[N][1]), np.concatenate(store[N][2])
        u, inv = np.unique(gd, return_inverse=True)
        cnt, vr = grouped_var(st, inv, len(u))
        _, vp = grouped_var(pr, inv, len(u))
        good = cnt >= MIN_21
        out[f"bias_N{N}_%"] = round(100 * (vp[good].sum() - vr[good].sum()) / vr[good].sum(), 1)
        out[f"recovered_N{N}"] = round(float((st * pr).sum() / (st ** 2).sum()), 2)
        if N == 1:
            speed = np.linalg.norm(st, axis=1)
            for lo, hi in zip(SPEED_BINS21[:-1], SPEED_BINS21[1:]):
                bsel = (speed >= lo) & (speed < hi)
                out[f"rec_{lo:g}-{hi if hi < 1e8 else 'inf'}px"] = round(float((st[bsel] * pr[bsel]).sum() / max((st[bsel] ** 2).sum(), 1e-9)), 2) if bsel.sum() > 20 else np.nan
    out["pixel_recovered"] = round(pix_num / pix_den, 2)
    return out


# ---------------- run ----------------
methods = build_methods()
keep_root = DATA_ROOT
rows21 = []
try:
    for seq in CROWDS21:
        DATA_ROOT = find_root_with(seq)
        P, pr = load_person_tracks(find_traj_file(seq))
        for name, fn in methods.items():
            rows21.append({"method": name, "crowd": seq, **evaluate(seq, P, pr, fn)})
finally:
    DATA_ROOT = keep_root
d21 = pd.DataFrame(rows21)
d21.to_csv(OUT_DIR / "e006_other_flow_methods.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
print("--- one-frame (N=1) and 4-frame direct (N=4) person-level results per method and crowd ---")
print(d21.to_string(index=False))

print(f"\n--- PRE-DECLARED CHECK: IM04 one-frame bias within +-{LIM21:.0f}% and rec_0-1px >= 0.85, and IM02 / IM05 within +-{LIM21:.0f}% ---")
for name in methods:
    t = d21[d21.method == name].set_index("crowd")
    ok = (abs(t.loc["IM04", "bias_N1_%"]) <= LIM21 and t.loc["IM04", "rec_0-1px"] >= 0.85
          and abs(t.loc["IM02", "bias_N1_%"]) <= LIM21 and abs(t.loc["IM05", "bias_N1_%"]) <= LIM21)
    print(f"{name:12s}", "HELPS" if ok else "does not help")

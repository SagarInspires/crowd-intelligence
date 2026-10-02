# ===== E006 · Cell 20 · Slow crowds: choose the frame gap so that people move a few pixels between the two frames =====
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---- needs Cell 18 run in this session (find_traj_file, load_person_tracks, sample_at, grouped_var, bilinear, find_root_with,
#      real_predict and the loaders). About 40 SEA-RAFT runs per crowd, a few minutes in total. ----
CROWDS20  = ["IM01", "IM01_hDyn", "IM03", "IM05", "IM02", "IM04"]
GROUP20   = {"IM01": "IM01+hDyn", "IM01_hDyn": "IM01+hDyn", "IM03": "IM03", "IM05": "IM05", "IM02": "IM02", "IM04": "IM04"}
GAPS20    = [1, 2, 4, 8]                  # frame gaps tried (direct flow between frame k and frame k+N, divided by N)
K20       = list(range(0, 60, 6))         # start frames
PATCH_20  = 160                           # patch size (px)
MIN_20    = 8                             # people per patch
OFF20     = -1                            # trajectory shift found in all six crowds
TARGET_DISP = 4.0                         # PRE-DECLARED rule: choose N so that the typical person moves about 4 px between the two frames
N_MAX     = 8                             # PRE-DECLARED: never use a gap above 8 frames (0.32 s): long direct gaps failed in dense crowds (Cell 11)
GATE20    = 10.0                          # PRE-DECLARED: with the rule, raw |bias| <= 10 % in every crowd = pass; the leave-crowd-out version is also printed


def direct_rows(seq, P, present):
    """Purpose: for every start frame k and gap N: run SEA-RAFT directly between frame k and k+N, read it at each person's true start
    position, divide by N, and compare per patch (>= MIN_20 people) with the people's real average velocity (x_{k+N} - x_k) / N.
    Also records the typical predicted displacement (mean |flow| at people, px) so the rule 'about 4 px' can be applied.
    Returns a list of per-patch rows; the one-frame rows (N = 1) also give the mean predicted speed per frame."""
    rows = []
    for k in tqdm(K20, desc=f"direct flows {seq}", leave=False):
        m = load_mask(seq, k) > 0
        a = k + OFF20
        nbx = -(-m.shape[1] // PATCH_20)
        n_groups = nbx * (-(-m.shape[0] // PATCH_20))
        img0 = load_frame(seq, k)
        for N in GAPS20:
            b = a + N
            if a < 0 or b >= P.shape[1] or k + N > 119:
                continue
            ok = present[:, a] & present[:, b]
            if ok.sum() < MIN_20:
                continue
            pos = P[ok, a]
            ref = (P[ok, b] - P[ok, a]) / N
            pred, _ = real_predict(seq, k, img0, load_frame(seq, k + N))
            v, inside, xi, yi = sample_at(pred, pos)
            sel = inside.copy()
            sel[inside] &= m[yi[inside], xi[inside]]
            if sel.sum() < MIN_20:
                continue
            gid = (yi[sel] // PATCH_20) * nbx + (xi[sel] // PATCH_20)
            cnt, v_ref = grouped_var(ref[sel], gid, n_groups)
            _, v_pr = grouped_var(v[sel] / N, gid, n_groups)
            disp = float(np.linalg.norm(v[sel], axis=1).mean())                      # predicted displacement over N frames (px)
            for i in np.nonzero(cnt >= MIN_20)[0]:
                rows.append({"seq": seq, "k": k, "N": N, "var_ref": v_ref[i], "var_pred": v_pr[i], "disp_px": disp})
    return rows


def pooled_bias(t, a=1.0):
    """Purpose: pooled bias in percent of the summed reference variance after multiplying the prediction by a."""
    return 100 * (a * t.var_pred.sum() - t.var_ref.sum()) / t.var_ref.sum()


def gain(t):
    """Purpose: single multiplier a (reference = a * prediction, through the origin)."""
    return float((t.var_ref * t.var_pred).sum() / (t.var_pred ** 2).sum())


# ---------------- run ----------------
keep_root = DATA_ROOT
rows20 = []
try:
    for seq in CROWDS20:
        DATA_ROOT = find_root_with(seq)
        P, pr = load_person_tracks(find_traj_file(seq))
        rows20 += direct_rows(seq, P, pr)
finally:
    DATA_ROOT = keep_root
df20 = pd.DataFrame(rows20)
df20["group"] = df20.seq.map(GROUP20)
df20.to_csv(OUT_DIR / "e006_adaptive_gap_table.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)

# 1) bias per crowd and gap, with the typical predicted displacement
tab = (df20.groupby(["group", "N"]).apply(lambda t: pd.Series({"patches": len(t), "disp_px": round(t.disp_px.mean(), 2),
        "bias_%": round(pooled_bias(t), 1), "corr": round(float(np.corrcoef(t.var_ref, t.var_pred)[0, 1]), 3)}), include_groups=False).reset_index())
print("--- direct flow over gap N: bias of the velocity spread per crowd (disp_px = typical predicted displacement over the gap) ---")
print(tab.pivot(index="group", columns="N", values=["disp_px", "bias_%"]).to_string())

# 2) the declared rule: N* = ceil(TARGET_DISP / one-frame predicted displacement), clipped to [1, N_MAX]
one = df20[df20.N == 1].groupby("group").disp_px.mean()
nstar = {g: int(np.clip(np.ceil(TARGET_DISP / max(d, 1e-6)), 1, N_MAX)) for g, d in one.items()}
nstar = {g: min(n, N_MAX) for g, n in nstar.items()}
nearest = {g: min(GAPS20, key=lambda n: abs(n - nstar[g])) for g in nstar}          # use the gap from the tried list that is closest
rule = []
for g in nstar:
    t = df20[(df20.group == g) & (df20.N == nearest[g])]
    rule.append({"crowd": g, "one_frame_disp_px": round(one[g], 2), "N_star": nstar[g], "N_used": nearest[g], "patches": len(t),
                 "bias_%": round(pooled_bias(t), 1)})
rule = pd.DataFrame(rule).set_index("crowd")
print("\n--- DECLARED RULE: gap chosen from the one-frame predicted displacement (target", TARGET_DISP, "px, max", N_MAX, "frames) ---")
print(rule.to_string())
print(f"all crowds within +-{GATE20:.0f}% without any gain:", "PASS" if (rule["bias_%"].abs() <= GATE20).all() else "FAIL")

# 3) leave-crowd-out gain on top of the rule (one gain from the other crowds' rule rows)
chosen = pd.concat([df20[(df20.group == g) & (df20.N == nearest[g])] for g in nstar])
lco = []
for g in nstar:
    tr, te = chosen[chosen.group != g], chosen[chosen.group == g]
    a = gain(tr)
    lco.append({"held_out": g, "gain_from_others": round(a, 2), "raw_bias_%": round(pooled_bias(te), 1), "gained_bias_%": round(pooled_bias(te, a), 1)})
lco = pd.DataFrame(lco).set_index("held_out")
print("\n--- leave-crowd-out gain on top of the rule ---")
print(lco.to_string())
print(f"worst held-out |bias| with the gain: {lco['gained_bias_%'].abs().max():.1f}%  ->", "PASS" if lco["gained_bias_%"].abs().max() <= GATE20 else "FAIL")

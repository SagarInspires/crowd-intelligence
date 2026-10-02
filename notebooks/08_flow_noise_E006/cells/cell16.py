# ===== E006 · Cell 16 · Analysis only (no GPU, no flow): is a gain learned on other crowds enough, and are the pooled numbers outlier-driven? =====
import numpy as np
import pandas as pd

# ---- settings (reads the table saved by Cell 15; nothing else is needed) ----
GROUPS16 = {"IM01": ["IM01", "IM01_hDyn"], "IM03": ["IM03"], "IM05": ["IM05"]}   # IM01 and hDyn are the same crowd: held out together
GATE16   = 10.0                                                                  # pre-registered limit, percent
NS16     = [16, 37]                                                              # windows to report


def crowd_of(seq):
    """Purpose: map a scene name to its crowd group (IM01 and IM01_hDyn show the same crowd)."""
    return next(g for g, members in GROUPS16.items() if seq in members)


def gain(t):
    """Purpose: best single multiplier a (reference = a * prediction, through the origin) for a set of patch rows."""
    return float((t.var_ref * t.var_pred).sum() / (t.var_pred ** 2).sum())


def pooled_bias(t, a=1.0):
    """Purpose: pooled bias in percent of the summed reference variance after multiplying the prediction by a."""
    return 100 * (a * t.var_pred.sum() - t.var_ref.sum()) / t.var_ref.sum()


def leave_crowd_out(df):
    """Purpose: for every estimator, start variant and window: for each crowd group, fit ONE gain on the rows of the other groups
    and report the held-out bias before (raw) and after (with that gain). Also the robust numbers on the held-out group: the
    median of the per-patch ratio prediction / reference and the share of the total predicted variance that the 3 largest
    patches carry (a high share means the pooled bias is driven by a few patches). Returns one row per case and group."""
    out = []
    for (est, var, N), t in df.groupby(["estimator", "variant", "N"]):
        t = t.assign(group=t.seq.map(crowd_of))
        for g in GROUPS16:
            test, train = t[t.group == g], t[t.group != g]
            if len(test) < 3 or len(train) < 3:
                continue
            a = gain(train)
            ratio = (test.var_pred / test.var_ref.replace(0, np.nan)).median()
            top3 = test.var_pred.nlargest(3).sum() / test.var_pred.sum()
            out.append({"estimator": est, "variant": var, "N": N, "held_out": g, "patches": len(test), "gain_from_others": round(a, 2),
                        "raw_bias_%": round(pooled_bias(test), 1), "gained_bias_%": round(pooled_bias(test, a), 1),
                        "median_patch_ratio": round(float(ratio), 2), "top3_share": round(float(top3), 2)})
    return pd.DataFrame(out)


# ---------------- run ----------------
df16 = pd.read_csv(OUT_DIR / "e006_robust_start_table.csv")
lco = leave_crowd_out(df16)
lco.to_csv(OUT_DIR / "e006_leave_crowd_out_start.csv", index=False)
pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 500)

summ = (lco.assign(raw=lco["raw_bias_%"].abs(), gained=lco["gained_bias_%"].abs())
           .groupby(["N", "estimator", "variant"]).agg(worst_raw=("raw", "max"), worst_with_gain=("gained", "max")).round(1))
summ["gate_with_gain"] = np.where(summ.worst_with_gain <= GATE16, "PASS", "FAIL")
for N in NS16:
    print(f"\n--- N = {N} frames: worst held-out |bias| % without and with a gain fitted on the OTHER crowds (gate {GATE16:.0f}%) ---")
    print(summ.loc[N].sort_values("worst_with_gain").to_string())

print("\n--- mask_peaks, 37 frames: are the pooled numbers driven by a few patches? (per held-out crowd) ---")
print(lco[(lco.N == 37) & (lco.variant == "mask_peaks")].to_string(index=False))
print("\n--- IM01 only, all variants, estimator 'point' and 'box4', 37 frames ---")
print(lco[(lco.N == 37) & (lco.held_out == "IM01") & lco.estimator.isin(["point", "box4"])].to_string(index=False))

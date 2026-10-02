# ===== E006 · Cell 25 · Closing cell: final summary tables, two figures, headline statements, and one zip of all E006 outputs =====
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ---- reads only the CSV files written by Cells 21-24 in OUT_DIR; runs no models, loads no images ----
GATE25 = 10.0        # the +-10 % gate used throughout E006
FILES25 = {"cell21": "e006_other_flow_methods.csv", "cell22": "e006_farneback_confirm.csv",
           "cell23": "e006_degradation_check.csv", "cell24": "e006_im04_feature_effects.csv"}
BLUE, ORANGE, GREY = "#1f77b4", "#d95f02", "#888888"       # colour-blind-safe pair plus neutral grey


def load_tables():
    """Purpose: read the four result CSVs from OUT_DIR; stop with a clear message if one is missing (run the earlier cell first)."""
    out, missing = {}, []
    for key, name in tqdm(FILES25.items(), desc="reading tables"):
        p = OUT_DIR / name
        if p.exists():
            out[key] = pd.read_csv(p)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError("missing in OUT_DIR: " + ", ".join(missing) + " -> run the cell that writes it first")
    return out


def fig_farneback_gaps(t22, path):
    """Purpose: Figure 1. Raw bias of the velocity spread for Farnebäck (true starts) per crowd and per gap N = 1, 2, 4 frames, with the +-10 % band.
    IM04 (slow crowd) is marked in orange."""
    crowds = list(dict.fromkeys(t22["crowd"]))
    gaps = sorted(t22["gap_N"].unique())
    x = np.arange(len(crowds)); w = 0.8 / len(gaps)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhspan(-GATE25, GATE25, color="#dddddd", alpha=0.6, label=f"+-{GATE25:.0f}% gate")
    shades = [0.45, 0.7, 1.0]
    for i, g in enumerate(gaps):
        v = [t22[(t22.gap_N == g) & (t22.crowd == c)]["raw_bias_%"].iloc[0] for c in crowds]
        cols = [ORANGE if c == "IM04" else BLUE for c in crowds]
        ax.bar(x + (i - (len(gaps) - 1) / 2) * w, v, w * 0.92, color=cols, alpha=shades[i % 3], label=f"gap {g} frame(s)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(crowds)
    ax.set_ylabel("bias of velocity spread (%)"); ax.set_title("Farnebäck, true starts: bias by crowd and gap (darker = longer gap)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_degradation(t23, path):
    """Purpose: Figure 2. Bias of the velocity spread for SEA-RAFT and Farnebäck under clean / noise / JPEG frames, one panel per method,
    one bar group per crowd, with the +-10 % band."""
    conds = list(dict.fromkeys(t23["condition"])); crowds = list(dict.fromkeys(t23["crowd"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, m in zip(axes, ["farneback", "sea_raft"]):
        ax.axhspan(-GATE25, GATE25, color="#dddddd", alpha=0.6)
        x = np.arange(len(crowds)); w = 0.8 / len(conds)
        for i, c in enumerate(conds):
            v = [t23[(t23.method == m) & (t23.condition == c) & (t23.crowd == cr)]["bias_%"].iloc[0] for cr in crowds]
            ax.bar(x + (i - (len(conds) - 1) / 2) * w, v, w * 0.92, color=BLUE if m == "farneback" else ORANGE, alpha=0.35 + 0.2 * i, label=c)
        ax.axhline(0, color="black", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(crowds); ax.set_title(m)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("bias of velocity spread (%)")
    fig.suptitle("Effect of noise and JPEG on the 2-frame estimate (true starts, four crowds)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def headline(t):
    """Purpose: build the headline statements from the tables themselves (no typed numbers) so the report text always matches the data."""
    t21, t22, t23, t24 = t["cell21"], t["cell22"], t["cell23"], t["cell24"]
    lines = []
    for g in sorted(t22.gap_N.unique()):
        s = t22[t22.gap_N == g]; s4 = s[s.crowd != "IM04"]
        lines.append(f"Farnebäck, gap {g}: worst raw |bias| without IM04 = {s4['raw_bias_%'].abs().max():.1f}% ; IM04 = {s[s.crowd == 'IM04']['raw_bias_%'].iloc[0]:+.1f}%")
    s1 = t21[t21.crowd == "IM04"].set_index("method")["bias_N1_%"]
    lines.append("IM04 one-frame bias by method: " + ", ".join(f"{m} {v:+.1f}%" for m, v in s1.items()))
    for m in ["farneback", "sea_raft"]:
        for c in dict.fromkeys(t23.condition):
            s = t23[(t23.method == m) & (t23.condition == c)]
            lines.append(f"Degradation {m:9s} {c:10s}: worst |bias| over four crowds = {s['bias_%'].abs().max():.1f}% -> {'within' if s['bias_%'].abs().max() <= GATE25 else 'outside'} +-{GATE25:.0f}%")
    tx = t24[t24.feature == "texture"].set_index("crowd")["gap"]
    nb = t24[t24.feature == "neighbours"].set_index("crowd")["gap"]
    lines.append(f"IM04 texture gap {tx['IM04']:+.3f} (declared threshold +0.10) ; IM04 neighbours gap {nb['IM04']:+.3f} (other crowds {nb.drop('IM04').min():+.3f} to {nb.drop('IM04').max():+.3f})")
    return lines


# ---------------- run ----------------
T = load_tables()
pd.set_option("display.width", 220, "display.max_columns", 30)
print("--- TABLE 1: Farnebäck, true starts, direct flow over N frames (Cell 22) ---")
print(T["cell22"].pivot(index="crowd", columns="gap_N", values="raw_bias_%").round(1).to_string())
print("\n--- TABLE 2: one-frame bias by flow method, three crowds (Cell 21) ---")
print(T["cell21"].pivot(index="method", columns="crowd", values="bias_N1_%").round(1).to_string())
print("\n--- TABLE 3: 2-frame bias under degraded frames (Cell 23) ---")
print(T["cell23"].pivot_table(index=["method", "condition"], columns="crowd", values="bias_%").round(1).to_string())
print("\n--- TABLE 4: image-feature effect on recovery inside each crowd (Cell 24; top minus bottom tertile) ---")
print(T["cell24"].pivot(index="crowd", columns="feature", values="gap").round(3).to_string())

fig_farneback_gaps(T["cell22"], OUT_DIR / "fig1_farneback_gaps.png")
fig_degradation(T["cell23"], OUT_DIR / "fig2_degradation.png")
H = headline(T)
(OUT_DIR / "e006_headline.txt").write_text("\n".join(H))
print("\n--- HEADLINE STATEMENTS (computed from the tables) ---")
print("\n".join(H))
print("\nLIMITS: CrowdFlow is synthetic (clean rendered frames); start points are the TRUE person positions; five independent crowds;"
      " no real-video ground truth; IM04 fails for every flow method and its cause is unexplained.")

# one zip with every CSV, figure and text file
zip_path = shutil.make_archive("/kaggle/working/E006_outputs", "zip", str(OUT_DIR))
print("\nzip with all E006 outputs:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.1f} MB)")
from IPython.display import Image, display
for f in ["fig1_farneback_gaps.png", "fig2_degradation.png"]:
    display(Image(filename=str(OUT_DIR / f)))

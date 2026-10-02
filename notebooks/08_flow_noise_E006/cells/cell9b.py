# ===== E006 · Cell 9b · Find the coordinate convention of the trajectories (they do not land on the people) =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- settings (needs Cell 9 run: tracks, TRAJ_SCENES, DATA_ROOT_1 / DATA_ROOT_2) ----
CHECK_SCENES = ["IM01", "IM03", "IM05"]
KS           = [0, 15, 30, 45, 59]                    # image frames used for the test
COORD_SCALES = [1.0, 2 / 3, 1.5, 0.5, 2.0]            # trajectory pixels vs image pixels (e.g. 1920x1080 -> 1280x720 is 2/3)
MAX_SHIFT    = 60                                     # try trajectory columns k-60 ... k+60 for image frame k
FLIPS        = {"identity": (False, False, False), "swap x,y": (True, False, False),
                "flip y": (False, False, True), "flip x": (False, True, False),
                "flip x and y": (False, True, True), "swap + flip y": (True, False, True)}   # (swap, flip_x, flip_y)


def to_pixels(pos, scale, flags, W, H):
    """Purpose: turn trajectory coordinates into image pixel coordinates under one hypothesis: multiply by
    `scale`, optionally swap x and y, optionally mirror x (W - x) and/or y (H - y)."""
    swap, fx, fy = flags
    x, y = pos[:, 0] * scale, pos[:, 1] * scale
    if swap:
        x, y = y, x
    if fx:
        x = W - x
    if fy:
        y = H - y
    return np.column_stack([x, y])


def share_on_people(P, present, masks, shift, scale, flags):
    """Purpose: for one hypothesis (frame shift, scale, flip) return the share of trajectory points that land on a
    person pixel (mask > 0) over the test frames, and the number of points. By chance this equals the share of
    the image covered by people (about 5-8 percent)."""
    hit, tot = 0, 0
    for k, m in masks.items():
        c = k + shift
        if c < 0 or c >= P.shape[1]:
            continue
        pos = P[present[:, c], c]
        if len(pos) == 0:
            continue
        H, W = m.shape
        px = np.rint(to_pixels(pos, scale, flags, W, H)).astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        hit += int((m[px[ok, 1], px[ok, 0]] > 0).sum())
        tot += len(pos)
    return (hit / tot if tot else np.nan), tot


def search_scene(seq, P, present):
    """Purpose: try every combination of flip, scale and frame shift for one scene and return a table sorted by the
    share of points that land on people, together with the chance level (share of image covered by people)."""
    masks = {k: load_mask(seq, k) for k in KS}
    chance = float(np.mean([(m > 0).mean() for m in masks.values()]))
    rows = []
    for name, flags in FLIPS.items():
        for sc in COORD_SCALES:
            for sh in range(-MAX_SHIFT, MAX_SHIFT + 1):
                frac, n = share_on_people(P, present, masks, sh, sc, flags)
                rows.append({"seq": seq, "transform": name, "scale": round(sc, 3), "shift": sh,
                             "on_people_%": round(100 * frac, 1) if frac == frac else np.nan, "points": n})
    t = pd.DataFrame(rows).sort_values("on_people_%", ascending=False).reset_index(drop=True)
    return t, 100 * chance


def overlay(seq, P, present, best, fname):
    """Purpose: draw the trajectory points of image frame 30 on top of the image under the best hypothesis and
    under the identity, so the result can be judged by eye (green = on a person, red = not)."""
    k = 30
    img, m = load_frame(seq, k), load_mask(seq, k) > 0
    H, W = m.shape
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
    for a, (title, sh, sc, flags) in zip(ax, [("identity, shift 0, scale 1", 0, 1.0, FLIPS["identity"]),
                                              (f"best: {best['transform']}, scale {best['scale']}, shift {best['shift']}",
                                               int(best["shift"]), float(best["scale"]), FLIPS[best["transform"]])]):
        c = k + sh
        pos = P[present[:, c], c] if 0 <= c < P.shape[1] else np.zeros((0, 2))
        px = np.rint(to_pixels(pos, sc, flags, W, H)).astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        px = px[ok]
        on = m[px[:, 1], px[:, 0]]
        a.imshow(img)
        a.scatter(px[on, 0], px[on, 1], s=8, c="lime")
        a.scatter(px[~on, 0], px[~on, 1], s=8, c="red")
        a.set_title(f"{seq} frame {k}: {title}  ({100 * on.mean() if len(on) else 0:.0f}% on people)")
        a.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=80)
    plt.show()


# ---------------- run ----------------
KEEP = DATA_ROOT
best_by_scene = {}
try:
    for seq in CHECK_SCENES:
        DATA_ROOT = TRAJ_SCENES[seq]
        P, pr = tracks[seq]
        t, chance = search_scene(seq, P, pr)
        print(f"\n=== {seq}: chance level (share of image covered by people) = {chance:.1f}% ===")
        print(t.head(8).to_string(index=False))
        best_by_scene[seq] = t.iloc[0]
        overlay(seq, P, pr, t.iloc[0], f"e006_traj_overlay_{seq}.png")
finally:
    DATA_ROOT = KEEP

# ===== E006 · Cell 2 · Load frames + ground-truth flow + masks, and CHECK the flow convention =====
import re
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ---- settings (DATA_ROOT and read_flo come from Cell 1; run Cell 1 first) ----
SEQS         = ["IM01", "IM01_hDyn"]     # the two sequences in our small dataset
CHECK_IDX    = [0, 15, 30, 45, 59]       # which frame pairs to test (must have a .flo file)
MOVING_THR   = 0.3                       # a pixel counts as "moving" if its true |flow| > this (pixels)
OUT_DIR      = Path("/kaggle/working/E006")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_flow_indices(seq):
    """Purpose: find which frame numbers have a ground-truth .flo file for this sequence
    (names look like frameGT_0007.flo). Returns a sorted list of integers."""
    folder = Path(DATA_ROOT) / "gt_flow" / seq
    idx = [int(re.search(r"(\d+)", p.stem).group(1)) for p in folder.glob("*.flo")]
    return sorted(idx)


def load_frame(seq, k):
    """Purpose: read frame number k of a sequence as an RGB uint8 image (H, W, 3)."""
    p = Path(DATA_ROOT) / "images" / seq / f"frame_{k:04d}.png"
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_gt_flow(seq, k):
    """Purpose: read the ground-truth flow file for frame k (float32, shape H x W x 2,
    pixels per frame). Unknown pixels (huge values) are set to NaN so they are never used."""
    fl = read_flo(Path(DATA_ROOT) / "gt_flow" / seq / f"frameGT_{k:04d}.flo").copy()
    fl[np.abs(fl) > 1e8] = np.nan
    return fl


def load_mask(seq, k):
    """Purpose: read the mask image for frame k as a 2-D integer array (we do not yet know
    what its values mean, so we only load it and describe it below)."""
    p = Path(DATA_ROOT) / "masks" / seq / f"maskGT_{k:04d}.png"
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(p)
    return m if m.ndim == 2 else m[..., 0]


def warp_back(img1, flow):
    """Purpose: build a prediction of frame k by sampling frame k+1 at (x + flow). If the
    flow has the convention 'motion from frame k to frame k+1', the result should look
    like frame k. Uses bilinear sampling."""
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = xs + np.nan_to_num(flow[..., 0])
    my = ys + np.nan_to_num(flow[..., 1])
    return cv2.remap(img1, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def photometric_error(img0, pred, sel):
    """Purpose: mean absolute colour difference (0-255 scale) between the real frame and
    the warped prediction, counted only on the pixels in `sel` (the moving ones)."""
    d = np.abs(img0.astype(np.float32) - pred.astype(np.float32)).mean(axis=2)
    return float(d[sel].mean()) if sel.any() else np.nan


def convention_check(seq, k):
    """Purpose: test how the ground-truth flow relates to the images. Compares four
    candidate readings on the moving pixels: flow as given (frame k -> k+1), flow with
    the sign flipped, no motion at all, and the flow applied to the wrong pair (k-1 -> k).
    The reading with the SMALLEST error is the right convention."""
    img0, img1 = load_frame(seq, k), load_frame(seq, k + 1)
    fl = load_gt_flow(seq, k)
    mag = np.sqrt(np.nansum(fl ** 2, axis=2))
    sel = (mag > MOVING_THR) & np.isfinite(fl).all(axis=2)
    row = {"seq": seq, "k": k, "moving_px_%": round(100 * sel.mean(), 2),
           "err_as_given": photometric_error(img0, warp_back(img1, fl), sel),
           "err_sign_flipped": photometric_error(img0, warp_back(img1, -fl), sel),
           "err_no_motion": photometric_error(img0, img1, sel)}
    if k >= 1:
        row["err_wrong_pair"] = photometric_error(load_frame(seq, k - 1), warp_back(img0, fl), sel)
    return {a: (round(b, 2) if isinstance(b, float) else b) for a, b in row.items()}


def describe_mask(seq, k):
    """Purpose: report what the mask contains (its distinct values, share of non-zero
    pixels) and how many of the truly moving pixels fall inside it. This tells us
    whether the mask marks people, so we know if we can use it to select the crowd."""
    m = load_mask(seq, k)
    fl = load_gt_flow(seq, k)
    mag = np.sqrt(np.nansum(fl ** 2, axis=2))
    moving = mag > MOVING_THR
    inside = m > 0
    return {"seq": seq, "k": k, "mask_values": str(np.unique(m)[:8].tolist()),
            "mask_nonzero_%": round(100 * inside.mean(), 2),
            "moving_px_%": round(100 * moving.mean(), 2),
            "moving_inside_mask_%": round(100 * (moving & inside).sum() / max(moving.sum(), 1), 1),
            "mask_inside_moving_%": round(100 * (moving & inside).sum() / max(inside.sum(), 1), 1)}


def make_overview(seq, k):
    """Purpose: save one picture per sequence: the frame, the true flow speed, and the mask,
    so we can see with our own eyes that the three files describe the same scene."""
    img = load_frame(seq, k)
    fl = load_gt_flow(seq, k)
    mag = np.sqrt(np.nansum(fl ** 2, axis=2))
    m = load_mask(seq, k)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    ax[0].imshow(img); ax[0].set_title(f"{seq}  frame {k}")
    im = ax[1].imshow(mag, vmin=0, vmax=max(float(np.nanpercentile(mag, 99.5)), 1e-3), cmap="magma")
    ax[1].set_title("true flow speed (px/frame)"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(m, cmap="nipy_spectral"); ax[2].set_title("mask (raw values)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    out = OUT_DIR / f"overview_{seq}.png"
    plt.savefig(out, dpi=90)
    plt.show()
    return out


# ---------------- run ----------------
print("DATA_ROOT =", DATA_ROOT)
avail = {s: list_flow_indices(s) for s in SEQS}
for s in SEQS:
    a = avail[s]
    print(f"{s}: {len(a)} ground-truth flow files, frame numbers {a[0]}..{a[-1]}" if a else f"{s}: no .flo files")

rows = []
for s in SEQS:
    for k in tqdm([k for k in CHECK_IDX if k in avail[s]], desc=f"convention check {s}"):
        rows.append(convention_check(s, k))
conv = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n--- which reading of the flow matches the images? (lower error = better) ---")
print(conv.to_string(index=False))

masks = pd.DataFrame([describe_mask(s, avail[s][len(avail[s]) // 2]) for s in SEQS])
print("\n--- what do the masks contain? ---")
print(masks.to_string(index=False))

for s in SEQS:
    make_overview(s, avail[s][len(avail[s]) // 2])

conv.to_csv(OUT_DIR / "flow_convention_check.csv", index=False)
masks.to_csv(OUT_DIR / "mask_check.csv", index=False)
print("\nsaved tables and pictures ->", OUT_DIR)

# ===== E006 · Cell 3 · Load SEA-RAFT and try it on ONE frame pair =====
import sys, os, json, glob, subprocess, importlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# ---- settings ----
REPO_DIR   = Path("/kaggle/working/SEA-RAFT")
HF_MODEL   = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"   # released SEA-RAFT checkpoint on HuggingFace
CFG_NAME   = "spring-M.json"                                    # config file that matches this checkpoint
TEST_PAIRS = [("IM01", 30), ("IM01_hDyn", 30)]                  # (sequence, k): frame k -> k+1
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# packages that are safe to install automatically if SEA-RAFT asks for them (anything else is reported, never installed)
SAFE_PIP   = {"cv2": "opencv-python-headless", "yaml": "pyyaml", "skimage": "scikit-image",
              "einops": "einops", "timm": "timm", "imageio": "imageio", "scipy": "scipy",
              "huggingface_hub": "huggingface_hub"}


def remove_wrong_update_package():
    """Purpose: remove the unrelated PyPI package called 'update' that an earlier version of this cell
    installed by mistake. It would hide SEA-RAFT's own file update.py. Harmless if it is not installed."""
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "update"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_repo():
    """Purpose: make sure the SEA-RAFT code is on disk and that BOTH the repo folder and its 'core' sub-folder
    are the FIRST places Python looks. The repo's files import each other by short names
    (update, corr, utils, extractor, layer), so 'core' must be on the path. Also removes any earlier cached
    copies of those short names so the repo's own files win."""
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/princeton-vl/SEA-RAFT.git", str(REPO_DIR)], check=True)
    for p in (str(REPO_DIR / "core"), str(REPO_DIR)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    clash = {"update", "corr", "utils", "extractor", "layer", "raft", "core"}
    for name in list(sys.modules):
        if name.split(".")[0] in clash:
            del sys.modules[name]
    importlib.invalidate_caches()
    os.chdir(REPO_DIR)


def import_raft():
    """Purpose: import the SEA-RAFT model class. Only packages from the SAFE_PIP list are installed
    automatically; any other missing name is reported (never installed), because names like 'update' belong
    to the repo's own files and a same-named PyPI package would be the wrong thing.
    Returns (RAFT class, load_ckpt function or None)."""
    for _ in range(6):
        try:
            from core.raft import RAFT
            try:
                from core.utils.utils import load_ckpt
            except Exception:
                load_ckpt = None
            return RAFT, load_ckpt
        except ModuleNotFoundError as e:
            missing = (e.name or "").split(".")[0]
            if missing not in SAFE_PIP:
                raise RuntimeError(f"SEA-RAFT needs '{missing}', which is not on the safe-install list. "
                                   "Paste this message to Claude.") from e
            print("installing missing package:", SAFE_PIP[missing])
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", SAFE_PIP[missing]], check=True)
            importlib.invalidate_caches()
    raise RuntimeError("could not import SEA-RAFT; paste the error to Claude")


def load_args():
    """Purpose: read the model's settings from the repo's JSON config file and wrap them as an
    'args' object, which the SEA-RAFT code expects. Prints which config files exist so we can
    see the real layout if the name is wrong."""
    found = sorted(glob.glob(str(REPO_DIR / "config" / "**" / "*.json"), recursive=True))
    print("config files in repo:", [os.path.relpath(p, REPO_DIR) for p in found])
    cfg_path = [p for p in found if p.endswith(CFG_NAME)]
    if not cfg_path:
        raise FileNotFoundError(f"{CFG_NAME} not found; pick one from the list above and set CFG_NAME")
    with open(cfg_path[0]) as f:
        cfg = json.load(f)
    print("config used:", os.path.relpath(cfg_path[0], REPO_DIR), "| keys:", sorted(cfg.keys()))
    return argparse.Namespace(**cfg)


def build_model(RAFT, args):
    """Purpose: create the SEA-RAFT network, download its trained weights from HuggingFace, move it to
    the GPU and switch it to evaluation mode (no training)."""
    model = RAFT.from_pretrained(HF_MODEL, args=args)
    return model.to(DEVICE).eval()


def to_tensor(rgb):
    """Purpose: turn an RGB picture (H, W, 3, values 0-255) into the tensor the model expects:
    shape (1, 3, H, W), floating point, still on the 0-255 scale (the model rescales it itself)."""
    return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(DEVICE)


@torch.no_grad()
def predict(model, args, rgb0, rgb1):
    """Purpose: run SEA-RAFT on a frame pair. Returns the estimated flow (H, W, 2) in pixels/frame
    and the raw 'info' map (channels, H, W) that carries the model's uncertainty."""
    out = model(to_tensor(rgb0), to_tensor(rgb1), iters=args.iters, test_mode=True)
    flow = out["flow"][-1][0].permute(1, 2, 0).float().cpu().numpy()
    info = out["info"][-1][0].float().cpu().numpy()
    return flow, info


def mixture_variance(info):
    """Purpose: turn the raw 'info' map into one variance value per pixel (pixels squared).
    ASSUMPTION to be checked in E007: channels 0-1 are the two mixture weights (before softmax) and
    channels 2-3 are the log-scales of two Laplace curves; a Laplace curve with scale b has variance
    2*b^2, so the mixture variance is sum(weight_i * 2 * exp(2*log_b_i)). These values are NOT
    calibrated; E007 tests how close they are to the real errors."""
    w = np.exp(info[:2] - info[:2].max(axis=0, keepdims=True))
    w = w / w.sum(axis=0, keepdims=True)
    logb = np.clip(info[2:4], -10.0, 10.0)
    return (w * 2.0 * np.exp(2.0 * logb)).sum(axis=0)


def evaluate_pair(model, args, seq, k):
    """Purpose: compare SEA-RAFT with the ground truth on one frame pair, inside the crowd mask:
    mean end-point error (EPE, pixels), the mean error vector (a lean = bias), the root-mean-square
    error per axis, and the model's own predicted spread. Also reports EPE of a 'zero flow' guess so
    we can tell if the model is doing anything useful."""
    img0, img1 = load_frame(seq, k), load_frame(seq, k + 1)
    gt = load_gt_flow(seq, k)
    inside = (load_mask(seq, k) > 0) & np.isfinite(gt).all(axis=2)
    pred, info = predict(model, args, img0, img1)
    err = (pred - gt)[inside]                                  # (N, 2) error of each crowd pixel
    var_raw = mixture_variance(info)[inside]
    return {"seq": seq, "k": k, "crowd_px": int(inside.sum()),
            "EPE_px": round(float(np.sqrt((err ** 2).sum(1)).mean()), 3),
            "EPE_zero_flow_px": round(float(np.sqrt((gt[inside] ** 2).sum(1)).mean()), 3),
            "mean_err_dx": round(float(err[:, 0].mean()), 3), "mean_err_dy": round(float(err[:, 1].mean()), 3),
            "rms_err_dx": round(float(np.sqrt((err[:, 0] ** 2).mean())), 3),
            "rms_err_dy": round(float(np.sqrt((err[:, 1] ** 2).mean())), 3),
            "pred_sigma_px": round(float(np.sqrt(var_raw).mean()), 3)}


# ---------------- run ----------------
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "NO GPU - switch the accelerator on")
remove_wrong_update_package()
ensure_repo()
RAFT, load_ckpt = import_raft()
args = load_args()
model = build_model(RAFT, args)
print("model loaded | iters =", getattr(args, "iters", "?"))

# one quick look at the raw output shapes
f0, i0 = predict(model, args, load_frame("IM01", 30), load_frame("IM01", 31))
print("flow shape:", f0.shape, "| info shape:", i0.shape,
      "| info channel ranges:", [(round(float(c.min()), 2), round(float(c.max()), 2)) for c in i0])

rows = [evaluate_pair(model, args, s, k) for s, k in TEST_PAIRS]
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\n--- SEA-RAFT vs ground truth, inside the crowd mask, one pair per sequence ---")
print(res.to_string(index=False))
res.to_csv(OUT_DIR / "searaft_one_pair.csv", index=False)

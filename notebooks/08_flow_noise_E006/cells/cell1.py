# ===== E006 · Cell 1 · Find CrowdFlow, unpack it if needed, and look inside =====
import os, zipfile, shutil, subprocess, struct
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---- settings (the only things you may need to change) ----
SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working"]   # where we look for the dataset
EXTRACT_TO   = Path("/kaggle/working/crowdflow")       # where an archive gets unpacked
OUT_DIR      = Path("/kaggle/working/E006")            # all E006 outputs go here
MAX_DEPTH    = 5                                       # how deep we search each root
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_dirs_and_files(root, max_depth):
    """Purpose: walk a folder tree only `max_depth` levels deep (so a huge /kaggle/input
    does not take forever) and yield (folder_path, subfolder_names, file_names)."""
    root = str(root)
    base_depth = root.rstrip("/").count("/")
    for cur, dirs, files in os.walk(root):
        depth = cur.rstrip("/").count("/") - base_depth
        if depth >= max_depth:
            dirs[:] = []          # stop going deeper
        yield cur, dirs, files


def find_crowdflow(roots, max_depth=MAX_DEPTH):
    """Purpose: look for CrowdFlow in the given roots. Returns ('folder', path) if the
    unpacked dataset is found, ('archive', path) if only a .rar/.zip/.7z is found,
    or (None, None) if nothing is found."""
    archive = None
    for root in roots:
        if not os.path.isdir(root):
            continue
        for cur, dirs, files in list_dirs_and_files(root, max_depth):
            low = {d.lower() for d in dirs}
            # the unpacked dataset has an 'images' folder and a 'gt_flow' folder side by side
            if "images" in low and "gt_flow" in low:
                return "folder", cur
            for f in files:
                if "crowdflow" in f.lower() and f.lower().endswith((".rar", ".zip", ".7z")):
                    archive = archive or os.path.join(cur, f)
    return ("archive", archive) if archive else (None, None)


def extract_archive(archive, dest):
    """Purpose: unpack a .zip / .rar / .7z file into `dest`. Tries Python's zipfile
    first, then command-line tools (unrar, 7zz, 7z, bsdtar) ONE AFTER ANOTHER: if a
    tool exists but fails (for example an old 7z cannot read RAR5), the next tool is
    tried. Raises one clear error listing what went wrong if all of them fail."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if archive.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for name in tqdm(z.namelist(), desc="unzipping"):
                z.extract(name, dest)
        return dest
    tools = [("unrar",  ["unrar", "x", "-o+", archive, str(dest) + "/"]),
             ("7zz",    ["7zz", "x", "-y", f"-o{dest}", archive]),
             ("7z",     ["7z", "x", "-y", f"-o{dest}", archive]),
             ("bsdtar", ["bsdtar", "-xf", archive, "-C", str(dest)])]
    problems = []
    for name, cmd in tools:
        if not shutil.which(name):
            problems.append(f"{name}: not installed")
            continue
        print(f"unpacking with {name} ... (large archive, please wait)")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return dest
        problems.append(f"{name}: failed ({(res.stderr or res.stdout).strip()[-150:]})")
    raise RuntimeError("Could not unpack the archive.\n  " + "\n  ".join(problems) +
                       "\nTry in a Kaggle cell:  !apt-get update -qq && apt-get install -y -qq unrar p7zip-full"
                       "\n(if 'unrar' is not found, try the package '7zip' instead), then run this cell again."
                       "\nOr unpack the .rar on your PC and upload the unpacked folder as a Kaggle dataset.")


def read_flo(path):
    """Purpose: read a Middlebury .flo optical-flow file. Returns an array of shape
    (height, width, 2) holding the (dx, dy) motion of every pixel in pixels/frame."""
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        if abs(magic - 202021.25) > 1e-3:
            raise ValueError(f"{path} is not a valid .flo file (magic number {magic})")
        w, h = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(w * h * 2 * 4), dtype="<f4")
    return data.reshape(h, w, 2)


def inventory(root):
    """Purpose: describe every folder under the dataset root: how many files it holds
    and which file types. Returns a DataFrame so we can see the real layout instead
    of guessing it."""
    rows = []
    walk = list(list_dirs_and_files(root, 6))
    for cur, dirs, files in tqdm(walk, desc="scanning folders"):
        if not files:
            continue
        exts = pd.Series([os.path.splitext(f)[1].lower() or "(none)" for f in files]).value_counts()
        rows.append({"folder": os.path.relpath(cur, root), "n_files": len(files),
                     "types": ", ".join(f"{k}:{v}" for k, v in exts.items()),
                     "first_file": sorted(files)[0]})
    return pd.DataFrame(rows).sort_values("folder").reset_index(drop=True)


def check_flow_files(root, inv):
    """Purpose: open ONE .flo file from every folder that contains them and report
    its size, its biggest motion, and how many pixels are marked 'unknown'
    (value >= 1e9). This tells us how to read the ground truth safely. It picks the
    first real .flo file by extension, so stray files like .DS_Store or notes.txt
    in the same folder cannot crash it."""
    out = []
    for _, r in inv[inv["types"].str.contains(r"\.flo")].iterrows():
        folder_path = Path(root) / r["folder"]
        flo_candidates = sorted(p for p in folder_path.iterdir() if p.suffix.lower() == ".flo")
        if not flo_candidates:
            continue
        p = flo_candidates[0]
        fl = read_flo(p)
        bad = ~np.isfinite(fl) | (np.abs(fl) > 1e8)
        good = fl[~bad.any(axis=2)]
        out.append({"folder": r["folder"], "file": p.name,
                    "height": fl.shape[0], "width": fl.shape[1],
                    "unknown_pixels_%": round(100 * bad.any(axis=2).mean(), 2),
                    "max_|flow|_px": round(float(np.abs(good).max()), 2) if good.size else np.nan,
                    "mean_|flow|_px": round(float(np.abs(good).mean()), 3) if good.size else np.nan})
    return pd.DataFrame(out)


# ---------------- run ----------------
kind, where = find_crowdflow(SEARCH_ROOTS)
print("search result:", kind, where)

if kind == "archive":
    where = str(extract_archive(where, EXTRACT_TO))
    kind, where = find_crowdflow([where])
    print("after unpacking:", kind, where)

if kind != "folder":
    print("\nCrowdFlow was NOT found. Do this once:\n"
          " 1. On your PC open  https://hidrive.ionos.com/lnk/LUiCHfYG  (password: CrowdFlow, case-sensitive).\n"
          " 2. Download the archive (check the size first) and, if it is a .rar, unpack it.\n"
          " 3. In Kaggle: Add Input -> Upload -> create a dataset (name it e.g. 'tub-crowdflow')\n"
          "    containing the folders 'images' and 'gt_flow'.\n"
          " 4. Attach it to this notebook and run this cell again.")
else:
    DATA_ROOT = where
    print("\nDATA_ROOT =", DATA_ROOT)
    inv = inventory(DATA_ROOT)
    pd.set_option("display.width", 200, "display.max_colwidth", 70, "display.max_rows", 200)
    print("\n--- every folder that contains files ---")
    print(inv.to_string())
    flow_info = check_flow_files(DATA_ROOT, inv)
    print("\n--- one ground-truth flow file per folder ---")
    print(flow_info.to_string() if len(flow_info) else "no .flo files found")
    inv.to_csv(OUT_DIR / "crowdflow_inventory.csv", index=False)
    flow_info.to_csv(OUT_DIR / "crowdflow_flow_check.csv", index=False)
    print("\nsaved inventory ->", OUT_DIR)

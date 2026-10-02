# ===== E006 · Cell 26 · One short video per experiment (13 videos) =====
import os
import shutil
import subprocess
import textwrap
import numpy as np
import cv2
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

# ---- needs Cell 18 run in this session (find_traj_file, load_person_tracks, sample_at, grouped_var, bilinear, advect_back, blur_flows,
#      store_flows, find_root_with, real_predict) and Cell 2 (load_frame, load_gt_flow, load_mask). About 10-15 minutes on the T4. ----
K_VID     = list(range(0, 56, 4))       # start frames shown in the one-gap videos (14 frames)
FPS_ARROW = 2                           # frames per second of the one-gap videos
FPS_STEP  = 6                           # frames per second of the 37-step window videos
OFF26     = -1                          # trajectory shift found in all crowds
PW, PH    = 640, 360                    # one panel = a 640 x 360 px crop of the 1280 x 720 frame
TITLE_H, FOOT_H = 24, 26                # bars above and below a panel
MAXA      = 120                         # at most this many people are drawn per panel
SCALE     = 8                           # arrows show motion in px per frame times this number
PATCH_26, MIN_26 = 160, 8               # same patches as the experiments
WIN_K0, WIN_N = 10, 37                  # the window shown in the step videos: starts at frame 10, 37 frames (1.5 s)
VID_DIR   = OUT_DIR / "videos"
VID_DIR.mkdir(parents=True, exist_ok=True)
GREEN, RED, ORANGE, WHITE, GREY = (0, 200, 0), (0, 0, 255), (0, 140, 255), (255, 255, 255), (170, 170, 170)   # BGR
FARNE26 = dict(pyr_scale=0.5, levels=5, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
TRACKS26, WINDOWS26 = {}, {}


# ------------------------------------------------------------------ small helpers
def use(seq):
    """Purpose: point the loaders (DATA_ROOT) at the dataset folder that holds this crowd."""
    global DATA_ROOT
    DATA_ROOT = find_root_with(seq)


def tracks(seq):
    """Purpose: person trajectories of a crowd (loaded once): P (persons, frames, 2) with x, y and a 'present' array."""
    if seq not in TRACKS26:
        TRACKS26[seq] = load_person_tracks(find_traj_file(seq))
    return TRACKS26[seq]


def degrade(img, cond, seed):
    """Purpose: copy of an RGB uint8 image with 'clean' (unchanged), 'noise_sd8' (Gaussian noise, 8 grey levels) or 'jpeg_q30' (JPEG quality 30)."""
    if cond == "clean":
        return img
    if cond.startswith("noise_sd"):
        rng = np.random.default_rng(seed)
        return np.clip(img.astype(np.float32) + rng.normal(0, float(cond[8:]), img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 30])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def build_methods26():
    """Purpose: the flow methods as functions (RGB image 0, RGB image 1, seq, k) -> flow (H, W, 2): sea_raft, farneback, dis, and raft_large
    (the last one only if its weights can be downloaded)."""
    m = {"sea_raft": lambda a, b, seq, k: real_predict(seq, k, a, b)[0]}
    m["farneback"] = lambda a, b, seq, k: cv2.calcOpticalFlowFarneback(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None, **FARNE26)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    m["dis"] = lambda a, b, seq, k: dis.calc(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)
    try:
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        w = Raft_Large_Weights.DEFAULT
        net = raft_large(weights=w).eval().cuda()
        prep = w.transforms()

        def raft(a, b, seq, k):
            t0 = torch.from_numpy(a).permute(2, 0, 1)[None].cuda(); t1 = torch.from_numpy(b).permute(2, 0, 1)[None].cuda()
            t0, t1 = prep(t0, t1)
            with torch.no_grad():
                return net(t0, t1, num_flow_updates=12)[-1][0].permute(1, 2, 0).cpu().numpy()
        m["raft_large"] = raft
    except Exception as e:
        print("raft_large skipped:", repr(e)[:120])
    return m


def panel_stats(pos, ref, est, shape=(720, 1280)):
    """Purpose: two numbers for a set of people: the bias (percent) of the per-patch velocity spread of the estimate against the real motion
    (patches of 160 px with >= 8 people, pooled) and the share of the real motion recovered. NaN if too few people."""
    good = np.isfinite(est).all(1) & np.isfinite(ref).all(1)
    pos, ref, est = pos[good], ref[good], est[good]
    if len(pos) < MIN_26:
        return np.nan, np.nan
    pos = np.clip(pos, 0, [shape[1] - 1, shape[0] - 1])
    nbx = -(-shape[1] // PATCH_26)
    gid = (pos[:, 1] // PATCH_26).astype(int) * nbx + (pos[:, 0] // PATCH_26).astype(int)
    u, inv = np.unique(gid, return_inverse=True)
    cnt, vr = grouped_var(ref, inv, len(u)); _, vp = grouped_var(est, inv, len(u))
    g = cnt >= MIN_26
    bias = 100 * (vp[g].sum() - vr[g].sum()) / vr[g].sum() if g.any() and vr[g].sum() > 0 else np.nan
    return float(bias), float((ref * est).sum() / max((ref ** 2).sum(), 1e-9))


def crop_box(seq, shape=(720, 1280)):
    """Purpose: fixed crop (x0, y0) of PW x PH px centred on where the people are in this crowd, so the video does not jump."""
    P, pr = tracks(seq)
    pts = np.vstack([P[pr[:, a], a] for a in range(0, min(P.shape[1], 60)) if pr[:, a].any()])
    cx, cy = np.median(pts, axis=0)
    return int(np.clip(cx - PW / 2, 0, shape[1] - PW)), int(np.clip(cy - PH / 2, 0, shape[0] - PH))


def pick(pos, box):
    """Purpose: indices of at most MAXA people whose start lies inside the crop (evenly spread), used to keep the same people in every item."""
    x0, y0 = box
    inb = np.nonzero(np.isfinite(pos).all(1) & (pos[:, 0] >= x0) & (pos[:, 0] < x0 + PW) & (pos[:, 1] >= y0) & (pos[:, 1] < y0 + PH))[0]
    return inb[:: max(1, len(inb) // MAXA)][:MAXA]


def make_panel(img_rgb, box, idx, items, title, foot):
    """Purpose: one panel: the crop of the frame (dimmed), the drawn items for the chosen people (arrows, dots, lines), a title bar and a footer
    bar with numbers. Items: ('arrows', pos, vec, colour, thickness), ('dots', pos, colour, radius), ('lines', pos_a, pos_b, colour)."""
    x0, y0 = box
    crop = (cv2.cvtColor(np.ascontiguousarray(img_rgb[y0:y0 + PH, x0:x0 + PW]), cv2.COLOR_RGB2BGR) * 0.7).astype(np.uint8)
    for it in items:
        if it[0] == "arrows":
            _, pos, vec, col, th = it
            for p, v in zip(pos[idx], vec[idx]):
                if np.isfinite(p).all() and np.isfinite(v).all():
                    a = (int(p[0] - x0), int(p[1] - y0)); b = (int(p[0] - x0 + SCALE * v[0]), int(p[1] - y0 + SCALE * v[1]))
                    cv2.arrowedLine(crop, a, b, col, th, tipLength=0.35)
        elif it[0] == "dots":
            _, pos, col, r = it
            for p in pos[idx]:
                if np.isfinite(p).all():
                    cv2.circle(crop, (int(p[0] - x0), int(p[1] - y0)), r, col, -1)
        else:
            _, pa, pb, col = it
            for p, q in zip(pa[idx], pb[idx]):
                if np.isfinite(p).all() and np.isfinite(q).all():
                    cv2.line(crop, (int(p[0] - x0), int(p[1] - y0)), (int(q[0] - x0), int(q[1] - y0)), col, 1)
    out = np.zeros((TITLE_H + PH + FOOT_H, PW, 3), np.uint8)
    out[TITLE_H:TITLE_H + PH] = crop
    cv2.putText(out, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(out, foot, (6, TITLE_H + PH + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 255, 255), 1, cv2.LINE_AA)
    return out


def compose(panels, head, ncols=2):
    """Purpose: put the panels in a grid under a header (title, description, legend; long lines are wrapped) and return one BGR frame
    with even width and height (needed for H.264)."""
    cols = min(ncols, len(panels)); rows = -(-len(panels) // cols)
    ph, pw = panels[0].shape[:2]
    W = pw * cols
    lines = []
    for txt, sc, col in head:
        for w in textwrap.wrap(txt, int(W / (22 * sc))) or [""]:
            lines.append((w, sc, col))
    hh = 10 + sum(int(26 * sc / 0.6) for _, sc, _ in lines)
    H = hh + ph * rows
    canvas = np.zeros((H + (H % 2), W, 3), np.uint8)
    y = 6
    for w, sc, col in lines:
        y += int(26 * sc / 0.6)
        cv2.putText(canvas, w, (8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        canvas[hh + r * ph: hh + (r + 1) * ph, c * pw:(c + 1) * pw] = p
    return canvas


class Sink:
    """Purpose: collects frames of one video, writes an mp4 (OpenCV), then re-encodes it to H.264 with ffmpeg (if available) so that it plays
    in browsers and players."""
    def __init__(self, name, fps):
        self.name, self.fps, self.w = name, fps, None
        self.tmp, self.path = VID_DIR / f"{name}_raw.mp4", VID_DIR / f"{name}.mp4"

    def add(self, frame, hold=1):
        """Purpose: append a frame (held for 'hold' video frames)."""
        if self.w is None:
            h, w = frame.shape[:2]
            self.w = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for _ in range(hold):
            self.w.write(frame)

    def close(self):
        """Purpose: finish the file and re-encode it to H.264 (falls back to the raw OpenCV file if ffmpeg is missing)."""
        self.w.release()
        exe = shutil.which("ffmpeg")
        if exe is None:
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                exe = None
        if exe:
            r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(self.tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(self.path)])
            if r.returncode == 0:
                os.remove(self.tmp)
                return self.path
        os.replace(self.tmp, self.path)
        return self.path


# ------------------------------------------------------------------ one-gap (arrow) videos
def person_motion(seq, k, N, fn, deg="clean", subset=None):
    """Purpose: for one crowd, start frame k and gap N: the people present in both frames with their start position, real motion per frame
    and the estimate per frame (flow of 'fn' between frame k and k+N read at the true start, divided by N), keeping people on the person mask.
    'subset' = ('few' | 'many') keeps people with <= 1 or >= 3 neighbours within 20 px (used by the crowding video)."""
    use(seq)
    P, pr = tracks(seq)
    a = k + OFF26; b = a + N
    ok = pr[:, a] & pr[:, b]
    m = load_mask(seq, k) > 0
    img0 = load_frame(seq, k)
    flow = fn(degrade(img0, deg, 2 * k), degrade(load_frame(seq, k + N), deg, 2 * k + 1), seq, k)
    pos = P[ok, a]; ref = (P[ok, b] - P[ok, a]) / N
    v, inside, xi, yi = sample_at(flow, pos)
    sel = inside.copy(); sel[inside] &= m[yi[inside], xi[inside]]
    pos, ref, est = pos[sel], ref[sel], v[sel] / N
    if subset is not None and len(pos) > 3:
        nb = cKDTree(pos).query_ball_point(pos, r=20, return_length=True) - 1
        keep = nb <= 1 if subset == "few" else nb >= 3
        pos, ref, est = pos[keep], ref[keep], est[keep]
    return img0, pos, ref, est


def arrow_video(name, title, desc, specs, ncols=2):
    """Purpose: write one video: for every start frame in K_VID, one panel per spec (crowd, gap N, method, optional degradation or subset):
    green arrows = real motion of each person, red arrows = estimate from the flow, both from the same true start and magnified by SCALE;
    the footer shows the bias of the velocity spread and the share of motion recovered for ALL people of that frame."""
    sink = Sink(name, FPS_ARROW)
    boxes = {}
    for k in tqdm(K_VID, desc=name, leave=False):
        panels = []
        for sp in specs:
            seq = sp["seq"]
            if seq not in boxes:
                use(seq); boxes[seq] = crop_box(seq)
            img0, pos, ref, est = person_motion(seq, k, sp["N"], sp["fn"], sp.get("deg", "clean"), sp.get("subset"))
            b_, r_ = panel_stats(pos, ref, est)
            idx = pick(pos, boxes[seq])
            bt = "n/a" if not np.isfinite(b_) else f"{b_:+.0f}%"
            panels.append(make_panel(img0, boxes[seq], idx, [("arrows", pos, ref, GREEN, 2), ("arrows", pos, est, RED, 1)], sp["label"],
                                     f"bias {bt}  recovered {r_:.2f}  people {len(pos)}"))
        head = [(title, 0.75, WHITE), (desc, 0.5, GREY), (f"start frame {k}   green = real motion, red = estimate (px per frame x{SCALE})", 0.5, (120, 255, 255))]
        sink.add(compose(panels, head, ncols))
    return sink.close()


def flow_color(img_rgb, flow, mask):
    """Purpose: colour picture of a flow field on the person pixels (hue = direction, brightness = speed, 4 px per frame = full brightness)
    over a dimmed copy of the frame."""
    f = np.nan_to_num(flow); mag = np.linalg.norm(f, axis=2); ang = np.arctan2(f[..., 1], f[..., 0])
    hsv = np.zeros(img_rgb.shape, np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8); hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / 4 * 255, 60, 255).astype(np.uint8)
    out = (img_rgb * 0.35).astype(np.uint8)
    out[mask] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[mask]
    return out


def flowmap_video(name, title, desc, seq, fn):
    """Purpose: video of the pixel-level experiment: ground-truth flow and SEA-RAFT flow as colour pictures side by side, with the spread (std)
    of the flow over person pixels in the footer."""
    sink = Sink(name, FPS_ARROW)
    use(seq); box = crop_box(seq)
    for k in tqdm(K_VID, desc=name, leave=False):
        use(seq)
        img0 = load_frame(seq, k); m = load_mask(seq, k) > 0
        gt = load_gt_flow(seq, k); pred = fn(img0, load_frame(seq, k + 1), seq, k)
        use_px = m & np.isfinite(gt).all(2)
        sg, sp_ = np.nanstd(gt[use_px], axis=0).sum(), np.nanstd(pred[use_px], axis=0).sum()
        pa = make_panel(flow_color(img0, gt, use_px), box, np.array([], int), [], "ground-truth flow", f"spread (std x + y) {sg:.2f} px")
        pb = make_panel(flow_color(img0, pred, use_px), box, np.array([], int), [], "SEA-RAFT flow", f"spread (std x + y) {sp_:.2f} px")
        head = [(title, 0.75, WHITE), (desc, 0.5, GREY), (f"start frame {k}   hue = direction, brightness = speed", 0.5, (120, 255, 255))]
        sink.add(compose([pa, pb], head, 2))
    return sink.close()


# ------------------------------------------------------------------ 37-step window videos
def get_window(seq, k0, N, backward=False):
    """Purpose: everything the window videos need for one crowd (computed once and kept): people present in all N + 1 frames and on the mask
    at the start, their true path, the forward SEA-RAFT flows (and the backward flows if asked) and the person masks."""
    key = (seq, k0, N)
    if key not in WINDOWS26:
        use(seq)
        P, pr = tracks(seq); a = k0 + OFF26
        ok = pr[:, a:a + N + 1].all(axis=1)
        m0 = load_mask(seq, k0) > 0
        idx = np.nonzero(ok)[0]
        xi, yi = np.rint(P[idx, a, 0]).astype(int), np.rint(P[idx, a, 1]).astype(int)
        inb = (xi >= 0) & (xi < m0.shape[1]) & (yi >= 0) & (yi < m0.shape[0])
        keep = idx[inb][m0[yi[inb], xi[inb]]]
        WINDOWS26[key] = dict(path=P[keep][:, a:a + N + 1].copy(), flows=store_flows(seq, list(range(k0, k0 + N))),
                              masks={j: load_mask(seq, j) > 0 for j in range(k0, k0 + N)}, bfl=None)
    w = WINDOWS26[key]
    if backward and w["bfl"] is None:
        use(seq); w["bfl"] = store_flows(seq, list(range(k0, k0 + N)), backward=True)
    return w


def path_advect(flows, k0, start, N):
    """Purpose: follow points with the flow itself: positions after 0, 1, ..., N frames (list of N + 1 arrays)."""
    out, p = [start.copy()], start.copy()
    for i in range(N):
        p = p + bilinear(flows[k0 + i], p); out.append(p.copy())
    return out


def path_fixed(flows, k0, start, N):
    """Purpose: read the flow at the FIXED start pixel in every frame and add the readings up (no following): 'positions' start + running sum."""
    out, d = [start.copy()], np.zeros_like(start)
    for i in range(N):
        d = d + bilinear(flows[k0 + i], start); out.append(start + d)
    return out


def path_reseed(flows, k0, true_path, jit, N, chunk=4):
    """Purpose: advect only for 'chunk' frames, then restart from the true position (plus the start error 'jit'); the displacements of the
    chunks are added up. Returns the virtual tracked positions (true start + summed displacement) after 0..N frames."""
    n = len(true_path); out = [true_path[:, 0].copy()]; total = np.zeros((n, 2))
    for c in range(0, N, chunk):
        s = true_path[:, c] + jit; p = s.copy()
        for t in range(min(chunk, N - c)):
            p = p + bilinear(flows[k0 + c + t], p)
            out.append(true_path[:, 0] + total + (p - s))
        total = total + (p - s)
    return out


def step_stats(true_path, tracked, i):
    """Purpose: numbers at step i for a tracked path: the bias (percent) of the velocity spread and the mean position error (px)."""
    ref = (true_path[:, i] - true_path[:, 0]) / i
    est = (tracked[i] - tracked[0]) / i
    b, _ = panel_stats(true_path[:, 0], ref, est)
    return b, float(np.nanmean(np.linalg.norm(tracked[i] - true_path[:, i], axis=1)))


def window_video(name, title, desc, seq, variants):
    """Purpose: write a window video: for every step i = 1..37 (1.5 s) one panel per variant, green dot = real position of a person, red dot =
    where the estimator says the person is, a line joins them. 'variants' = list of (label, tracked path list, optional keep mask)."""
    w = get_window(seq, WIN_K0, WIN_N)
    use(seq); box = crop_box(seq)
    img = load_frame(seq, WIN_K0)
    tp = w["path"]; idx = pick(tp[:, 0], box)
    sink = Sink(name, FPS_STEP)
    for i in range(1, WIN_N + 1):
        panels = []
        for label, tr, keep in variants:
            b_, e_ = step_stats(tp, tr, i)
            items = [("lines", tp[:, i], tr[i], GREY), ("dots", tp[:, i], GREEN, 3)]
            if keep is None:
                items.append(("dots", tr[i], RED, 3))
                foot = f"bias {b_:+.0f}%  mean position error {e_:.1f} px"
            else:
                kp = keep(i)
                items.append(("dots", np.where(kp[:, None], tr[i], np.nan), (255, 200, 0), 3))
                items.append(("dots", np.where(~kp[:, None], tr[i], np.nan), RED, 3))
                bk, _ = panel_stats(tp[kp, 0], ((tp[:, i] - tp[:, 0]) / i)[kp], ((tr[i] - tr[0]) / i)[kp])
                foot = f"all {b_:+.0f}%  kept {bk:+.0f}% ({100 * kp.mean():.0f}% kept)"
            panels.append(make_panel(img, box, idx, items, label, foot))
        head = [(title, 0.75, WHITE), (desc, 0.5, GREY), (f"step {i} of {WIN_N} frames ({i / 25:.2f} s)   green = real position, red = estimated position", 0.5, (120, 255, 255))]
        sink.add(compose(panels, head, 2), hold=4 if i == 1 else 1)
    return sink.close()


# ------------------------------------------------------------------ the 13 experiments
def build_all(M):
    """Purpose: the list of experiments: (file name, function that writes the video), in the order of the notebook cells."""
    S, F, D, R = M["sea_raft"], M["farneback"], M.get("dis"), M.get("raft_large")
    rng = lambda s: np.random.default_rng(s)
    ex = []
    ex.append(("E01_pixel_flow_maps", lambda: flowmap_video("E01_pixel_flow_maps", "E01  Pixel level (early cells): ground truth against SEA-RAFT",
               "Same crowd (IM01), same frame. Ground-truth flow against SEA-RAFT flow on the people. The learned flow tends to be smoother, one reason the pixel-level spread came out too low.", "IM01", S)))
    ex.append(("E02_person_one_frame", lambda: arrow_video("E02_person_one_frame", "E02  Person level, one frame (person-level cells)",
               "Read the flow at each person's true position. Arrows are real step (green) and SEA-RAFT estimate (red). They agree well in these two crowds.",
               [dict(seq="IM01", N=1, fn=S, label="IM01  SEA-RAFT, 1 frame"), dict(seq="IM05", N=1, fn=S, label="IM05  SEA-RAFT, 1 frame")])))
    ex.append(("E03_window_length", lambda: arrow_video("E03_window_length", "E03  Window length (Cell 11), crowd IM05",
               "Direct flow between frame k and k+N divided by N. Short gaps agree; at long gaps (up to 16 frames) the estimate falls apart in this dense crowd.",
               [dict(seq="IM05", N=n, fn=S, label=f"IM05  gap N = {n}") for n in (1, 4, 8, 16)])))

    def e04():
        w = get_window("IM05", WIN_K0, WIN_N); st = w["path"][:, 0]
        return window_video("E04_follow_vs_fixed", "E04  Following the flow vs reading a fixed pixel (Cell 12), crowd IM05",
                            "Two ways to turn flow into 1.5 s of motion from the true start. Following the flow tracks the people; reading the same pixel again and again fails.",
                            "IM05", [("fixed pixel (Eulerian)", path_fixed(w["flows"], WIN_K0, st, WIN_N), None),
                                     ("follow the flow (advected)", path_advect(w["flows"], WIN_K0, st, WIN_N), None)])
    ex.append(("E04_follow_vs_fixed", e04))

    def e05():
        w = get_window("IM05", WIN_K0, WIN_N); tp = w["path"]; r = rng(5); vs = []
        for sg in (0, 2, 5, 10):
            s = tp[:, 0] + (r.normal(0, sg, tp[:, 0].shape) if sg else 0)
            vs.append((f"start error {sg} px" if sg else "exact start", path_advect(w["flows"], WIN_K0, s, WIN_N), None))
        return window_video("E05_start_errors", "E05  Wrong start points (Cell 14), crowd IM05",
                            "Start the same flow-following from slightly wrong positions (as a detector would). The tracked points drift away from the people and the spread is typically overestimated.", "IM05", vs)
    ex.append(("E05_start_errors", e05))

    def e06():
        w = get_window("IM05", WIN_K0, WIN_N); tp = w["path"]; r = rng(6)
        jit = r.normal(0, 2, tp[:, 0].shape); s = tp[:, 0] + jit
        fb = blur_flows(w["flows"], w["masks"], 4)
        return window_video("E06_box_and_reseed", "E06  Box averaging and re-seeding (Cell 15), crowd IM05, 2 px start error",
                            "Two repairs for start errors: average the flow over a 9 x 9 px person box, and restart from the person's position every 4 frames. In Cell 15 they helped at 0.64 s but not reliably at 1.5 s.",
                            "IM05", [("plain following", path_advect(w["flows"], WIN_K0, s, WIN_N), None),
                                     ("box 4 (9 x 9 px average)", path_advect(fb, WIN_K0, s, WIN_N), None),
                                     ("re-seed every 4 frames", path_reseed(w["flows"], WIN_K0, tp, jit, WIN_N), None),
                                     ("box 4 + re-seed", path_reseed(fb, WIN_K0, tp, jit, WIN_N), None)])
    ex.append(("E06_box_and_reseed", e06))

    def e07():
        w = get_window("IM05", WIN_K0, WIN_N, backward=True); tp = w["path"]; r = rng(7)
        s = tp[:, 0] + r.normal(0, 5, tp[:, 0].shape)
        fb = blur_flows(w["flows"], w["masks"], 4)
        bb = blur_flows(w["bfl"], w["masks"], 4)
        out = []
        for label, fl, bl in (("plain, 5 px start error", w["flows"], w["bfl"]), ("box 4, 5 px start error", fb, bb)):
            tr = path_advect(fl, WIN_K0, s, WIN_N)
            keep = (lambda tr_, bl_: (lambda i: np.linalg.norm(advect_back(bl_, WIN_K0, i, tr_[i]) - s, axis=1) <= 2.0))(tr, bl)
            out.append((label + "  (blue = kept, red = rejected)", tr, keep))
        return window_video("E07_round_trip_filter", "E07  Round-trip filter (Cell 17), crowd IM05",
                            "Follow each point forward, then backward again. If it does not come back within 2 px, reject it. Blue points are kept, red ones are rejected.", "IM05", out)
    ex.append(("E07_round_trip_filter", e07))
    ex.append(("E08_slow_crowd", lambda: arrow_video("E08_slow_crowd", "E08  A slow crowd (Cells 18-19): IM04 against IM02",
               "Same SEA-RAFT, same one-frame test. In IM04 (slow crowd) the red arrows are visibly shorter than the green ones.",
               [dict(seq="IM04", N=1, fn=S, label="IM04 (slow)  SEA-RAFT, 1 frame"), dict(seq="IM02", N=1, fn=S, label="IM02  SEA-RAFT, 1 frame")])))
    ex.append(("E09_gap_on_slow_crowd", lambda: arrow_video("E09_gap_on_slow_crowd", "E09  Longer gaps on the slow crowd (Cell 20), crowd IM04",
               "Waiting longer between the two frames does not repair IM04: the red arrows stay short at every gap.",
               [dict(seq="IM04", N=n, fn=S, label=f"IM04  SEA-RAFT, gap N = {n}") for n in (1, 2, 4, 8)])))
    if D is not None and R is not None:
        ex.append(("E10_other_flow_methods", lambda: arrow_video("E10_other_flow_methods", "E10  Other flow methods (Cell 21), crowd IM04, one frame",
                   "Four flow methods on the slow crowd. All of them lose motion; the classical Farnebäck loses least.",
                   [dict(seq="IM04", N=1, fn=S, label="SEA-RAFT"), dict(seq="IM04", N=1, fn=F, label="Farneback"),
                    dict(seq="IM04", N=1, fn=D, label="DIS"), dict(seq="IM04", N=1, fn=R, label="RAFT-large")])))
    ex.append(("E11_farneback_confirm", lambda: arrow_video("E11_farneback_confirm", "E11  Farneback on four crowds (Cell 22), gap 2 frames",
               "Classical Farneback flow read at the true starts. Red and green arrows agree in IM03, IM05 and IM02; IM04 (bottom right) is again too short.",
               [dict(seq=s_, N=2, fn=F, label=f"{s_}  Farneback, gap 2") for s_ in ("IM03", "IM05", "IM02", "IM04")])))
    ex.append(("E12_noise_and_jpeg", lambda: arrow_video("E12_noise_and_jpeg", "E12  Noise and compression (Cell 23), crowd IM05, gap 2 frames",
               "Top: Farneback on clean frames and on JPEG-compressed frames. Bottom: the same for SEA-RAFT. Farneback does not change; SEA-RAFT loses some spread.",
               [dict(seq="IM05", N=2, fn=F, label="Farneback  clean"), dict(seq="IM05", N=2, fn=F, deg="jpeg_q30", label="Farneback  JPEG q30"),
                dict(seq="IM05", N=2, fn=S, label="SEA-RAFT  clean"), dict(seq="IM05", N=2, fn=S, deg="jpeg_q30", label="SEA-RAFT  JPEG q30")])))
    ex.append(("E13_crowding_in_IM04", lambda: arrow_video("E13_crowding_in_IM04", "E13  Neighbours in the slow crowd (Cell 24), IM04, Farneback, one frame",
               "Left: people with at most one neighbour within 20 px. Right: people with three or more neighbours. The right panel recovers less motion (a lead, not a proven cause). Numbers in the footer use only the people shown.",
               [dict(seq="IM04", N=1, fn=F, subset="few", label="few neighbours"), dict(seq="IM04", N=1, fn=F, subset="many", label="many neighbours")])))
    return ex


# ---------------- run ----------------
keep_root = DATA_ROOT
M26 = build_methods26()
INDEX = []
try:
    for fname, fn_ in tqdm(build_all(M26), desc="videos"):
        try:
            p = fn_()
            INDEX.append(f"{fname}.mp4  ({os.path.getsize(p) / 1e6:.1f} MB)")
            print("written:", p.name)
        except Exception as e:
            print(f"FAILED {fname}: {repr(e)[:200]}")
finally:
    DATA_ROOT = keep_root
(VID_DIR / "INDEX.txt").write_text("\n".join(INDEX))
zip_path = shutil.make_archive("/kaggle/working/E006_videos", "zip", str(VID_DIR))
print("\n".join(INDEX))
print("\nvideos in", VID_DIR, "| zip:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.1f} MB)")

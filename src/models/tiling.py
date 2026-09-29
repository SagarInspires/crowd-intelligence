"""Sliced (tiled) person detection in the style of SAHI (Akyon et al., ICIP 2022), with explicit control of magnification.

Why our own version: SAHI's slices only help if each tile is enlarged before the detector sees it. Here the tile size and
the detector input size (imgsz) are separate settings, so 'tile 320 at imgsz 1280' means every far-away person is 4x larger.
Duplicates at tile seams are avoided by ownership, not by NMS: each detection is kept only by the tile whose CORE region
contains its head point, so two neighbouring people are never merged by mistake. A person is found intact if their box is
no taller than the overlap between tiles (tile x overlap); taller (near-field) people are the full-frame detector's job."""
import numpy as np

def tile_starts(length, tile, overlap):
    """Start positions of overlapping tiles along one axis; the last tile is shifted back so it ends exactly at `length`."""
    if length <= tile:
        return [0]
    step = max(int(tile * (1 - overlap)), 1)
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts

def core_bounds(starts, tile, length, bias=0.5):
    """Split [0, length] into one 'core' interval per tile. Each boundary lies `bias` of the way through the overlap
    between neighbouring tiles (0.5 = middle). Returns a list of (lo, hi); together they cover the axis exactly once.
    Use 0.5 across (left-right) and the head-point ratio (0.15) down the image: a body extends DOWNWARD from its
    head point, so the upper tile should hand over early and the lower tile owns the person."""
    cuts = [0.0] + [starts[i + 1] + bias * (starts[i] + tile - starts[i + 1]) for i in range(len(starts) - 1)] + [float(length)]
    return [(cuts[i], cuts[i + 1]) for i in range(len(starts))]

def head_point(boxes):
    """Head position of each person box: horizontal centre, 15% down from the top edge."""
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1] + 0.15 * (boxes[:, 3] - boxes[:, 1])], axis=1)

def detect_tiled(model, img, tile, imgsz, overlap=0.2, conf_floor=0.001, max_det=3000):
    """Run `model` (an Ultralytics YOLO) on overlapping tiles of `img` (BGR array) and return (boxes xyxy, scores) in
    full-image coordinates. Each tile is shown to the detector at `imgsz` pixels, so magnification = imgsz / tile."""
    H, W = img.shape[:2]
    ys, xs = tile_starts(H, tile, overlap), tile_starts(W, tile, overlap)
    cy, cx = core_bounds(ys, min(tile, H), H, bias=0.15), core_bounds(xs, min(tile, W), W, bias=0.5)
    all_b, all_s = [], []
    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            crop = img[y0:y0 + tile, x0:x0 + tile]
            r = model.predict(crop, imgsz=imgsz, classes=[0], conf=conf_floor, max_det=max_det, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            s = r.boxes.conf.cpu().numpy().astype(np.float32)
            if len(b) == 0:
                continue
            b[:, [0, 2]] += x0; b[:, [1, 3]] += y0                    # tile coordinates -> full-image coordinates
            hp = head_point(b)
            own = (hp[:, 0] >= cx[ix][0]) & (hp[:, 0] < cx[ix][1]) & (hp[:, 1] >= cy[iy][0]) & (hp[:, 1] < cy[iy][1])
            all_b.append(b[own]); all_s.append(s[own])                # keep only detections this tile owns
    if not all_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.concatenate(all_b), np.concatenate(all_s)

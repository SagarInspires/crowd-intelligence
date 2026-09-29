"""Score person detectors against head-point ground truth (matching, per-frame metrics, bootstrap CI)."""
import json
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def head_anchor(boxes):
    """Where a head should be for each person box: horizontal centre, 15% down from the top edge."""
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1] + 0.15 * (boxes[:, 3] - boxes[:, 1])], axis=1)

def inside_any(xy, polys):
    """Boolean array: is each point strictly inside any ignore polygon (exact geometric test)."""
    out = np.zeros(len(xy), bool)
    for v in polys:
        c = np.asarray(v, np.float32).reshape(-1, 1, 2)
        out |= np.array([cv2.pointPolygonTest(c, (float(x), float(y)), False) > 0 for x, y in xy], bool)
    return out

def match(points, boxes, gate_frac=0.5, gate_min=6.0, gate_max=40.0):
    """One-to-one matching of ground-truth heads to box head-anchors (Hungarian).
    A pair is a true positive if the distance is below a per-box gate = 0.5 x box width (clipped), so big near-field
    people get a loose gate and tiny far-field people a tight one. Returns tp mask over points and over boxes."""
    if len(points) == 0 or len(boxes) == 0:
        return np.zeros(len(points), bool), np.zeros(len(boxes), bool)
    a = head_anchor(boxes)
    gate = np.clip(gate_frac * (boxes[:, 2] - boxes[:, 0]), gate_min, gate_max)
    dist = np.linalg.norm(points[:, None] - a[None], axis=2)
    cost = np.where(dist <= gate[None], dist, 1e6)           # forbid matches outside the gate
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < 1e6
    tp_pts, tp_box = np.zeros(len(points), bool), np.zeros(len(boxes), bool)
    tp_pts[r[ok]], tp_box[c[ok]] = True, True
    return tp_pts, tp_box

def eval_frame(points, boxes, conf, polys, conf_thr, far_y):
    """Metrics for one frame at one confidence threshold, all outside the ignore regions.
    Boxes whose head anchor falls in an ignore polygon are dropped so every method is scored on the same area."""
    keep = conf >= conf_thr
    boxes = boxes[keep]
    if len(boxes) and polys:
        boxes = boxes[~inside_any(head_anchor(boxes), polys)]
    tp_p, tp_b = match(points, boxes)
    far = points[:, 1] < far_y                                # top of a portrait frame = far field
    return dict(n_true=len(points), n_det=len(boxes), tp=int(tp_p.sum()), fp=int((~tp_b).sum()), fn=int((~tp_p).sum()),
                n_far=int(far.sum()), tp_far=int(tp_p[far].sum()), n_near=int((~far).sum()), tp_near=int(tp_p[~far].sum()))

def summarise(df):
    """Aggregate per-frame rows: count MAE / RMSE / bias, and pooled precision / recall / F1 (near and far recall too)."""
    e = df.n_det - df.n_true
    p = df.tp.sum() / max(df.tp.sum() + df.fp.sum(), 1); r = df.tp.sum() / max(df.tp.sum() + df.fn.sum(), 1)
    return dict(MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(), precision=p, recall=r,
                F1=2 * p * r / max(p + r, 1e-9), recall_far=df.tp_far.sum() / max(df.n_far.sum(), 1),
                recall_near=df.tp_near.sum() / max(df.n_near.sum(), 1))

def cluster_bootstrap_ci(values, n=2000, seed=0, agg=np.mean):
    """95% CI of a statistic over frames: frames are the independent unit (heads within a frame are not)."""
    rng = np.random.default_rng(seed); v = np.asarray(values, float)
    s = [agg(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return np.percentile(s, [2.5, 97.5])

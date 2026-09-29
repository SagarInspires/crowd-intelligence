"""Read VIA annotation projects (head points + ignore regions) into clean tables.
Used by every notebook that needs the ground truth, so the parsing lives in one place."""
import json, math
from pathlib import Path
import pandas as pd

def load_via_project(path):
    """Read a VIA project .json and return {image filename: list of shape dicts}
    (each shape dict is a point, rect or polygon exactly as VIA stored it)."""
    with open(path) as f:
        d = json.load(f)
    return {v["filename"]: [r["shape_attributes"] for r in v["regions"]]
            for v in d["_via_img_metadata"].values()}

def frame_id_from_name(name):
    """'frame_00279.jpg' -> 279 (the frame number inside the video)."""
    return int(Path(name).stem.split("_")[1])

def merge_close_points(points, min_dist=6.0):
    """Treat two points closer than min_dist pixels as one accidental double-click:
    keep the first, drop the later one. Returns (kept points, number dropped)."""
    kept = []
    for p in points:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_dist for q in kept):
            kept.append(p)
    return kept, len(points) - len(kept)

def to_vertices(shape):
    """Return an ignore region as a list of [x, y] corners (rectangles become 4 corners),
    or None if the shape is not an ignore region."""
    if shape["name"] == "polygon":
        return [[x, y] for x, y in zip(shape["all_points_x"], shape["all_points_y"])]
    if shape["name"] == "rect":
        x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return None

def polygon_area(verts):
    """Area in pixels of a polygon given as [[x, y], ...] (shoelace formula)."""
    n = len(verts)
    s = sum(verts[i][0] * verts[(i + 1) % n][1] - verts[(i + 1) % n][0] * verts[i][1]
            for i in range(n))
    return abs(s) / 2


def load_ground_truth(data_dir, frame_wh=(1080, 1920), annotator="sagar",
                      json_rel="annotations/via_project_phase0.json"):
    """Parse the committed VIA project and return (points, ignore_regions, summary) DataFrames.
    points: [frame_id, x, y, annotator]; ignore_regions: [frame_id, region_id, vertices (list of [x, y])].
    Near-duplicate points (< 6 px) are merged. Nothing is written to disk: the JSON is the single source of truth."""
    W, H = frame_wh
    shapes = load_via_project(Path(data_dir) / json_rel)
    point_rows, ignore_rows, summary = [], [], []
    for fname in sorted(shapes):
        fid = frame_id_from_name(fname)
        kept, dropped = merge_close_points([(s["cx"], s["cy"]) for s in shapes[fname] if s["name"] == "point"])
        point_rows += [{"frame_id": fid, "x": x, "y": y, "annotator": annotator} for x, y in kept]
        area, rid = 0.0, 0
        for s in shapes[fname]:
            verts = to_vertices(s)
            if verts is None:
                continue
            ignore_rows.append({"frame_id": fid, "region_id": rid, "vertices": verts})
            area += polygon_area(verts); rid += 1
        summary.append({"frame_id": fid, "n_points": len(kept), "dup_dropped": dropped, "n_ignore_regions": rid,
                        "ignore_pct_of_frame": round(100 * area / (W * H), 1)})
    return pd.DataFrame(point_rows), pd.DataFrame(ignore_rows), pd.DataFrame(summary)

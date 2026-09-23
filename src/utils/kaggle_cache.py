"""
Reusable caching + checkpointing helpers for running this project on Kaggle
without a persistent machine.

Two problems, two functions:
  - "I don't want to recompute things I already computed"      -> cache_or_compute
  - "I don't want to restart training from epoch 0 every time"  -> save_checkpoint / load_latest_checkpoint

See KAGGLE_PROTOCOL.md for the reasoning and the notebook-header pattern
these are meant to be used inside.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Callable, Optional


def cache_or_compute(
    output_path: str,
    compute_fn: Callable[[], Any],
    loader_fn: Callable[[str], Any],
    saver_fn: Optional[Callable[[str, Any], None]] = None,
    force: bool = False,
) -> Any:
    """
    If `output_path` already exists (and force=False), load and return it
    via `loader_fn`. Otherwise call `compute_fn()`, save the result to
    `output_path`, and return it.

    Parameters
    ----------
    output_path : where the cached result lives (any extension you like —
        this function is format-agnostic, you supply the load/save logic).
    compute_fn : zero-arg function that does the expensive work and
        returns the result.
    loader_fn : function(path) -> result, used when the cache hits.
        e.g. np.load, pd.read_csv, or a lambda wrapping torch.load.
    saver_fn : function(path, result) -> None, used when the cache misses.
        If omitted, we try to infer it from the extension for the common
        cases (.npy via numpy, .csv via pandas' .to_csv). For anything
        else, pass this explicitly.
    force : if True, ignore any existing cache and recompute (use this
        once, deliberately, if you changed the logic in compute_fn and
        need to invalidate a stale cache — don't leave it True).

    Example
    -------
    >>> density_maps = cache_or_compute(
    ...     output_path=f"{CACHE_DIR}/density_maps/shanghai_a_train.npy",
    ...     compute_fn=lambda: generate_density_maps(paths, sigma=15),
    ...     loader_fn=np.load,
    ... )
    """
    if os.path.exists(output_path) and not force:
        print(f"[cache hit]  {output_path}")
        return loader_fn(output_path)

    print(f"[cache miss] {output_path} — computing...")
    result = compute_fn()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if saver_fn is not None:
        saver_fn(output_path, result)
    else:
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".npy":
            import numpy as np
            np.save(output_path, result)
        elif ext == ".csv":
            result.to_csv(output_path, index=False)
        elif ext in (".pt", ".pth"):
            import torch
            torch.save(result, output_path)
        else:
            raise ValueError(
                f"Don't know how to save '{ext}' automatically — "
                f"pass saver_fn explicitly for {output_path}"
            )
    print(f"[cached]     {output_path}")
    return result


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model,
    optimizer,
    extra: Optional[dict] = None,
) -> str:
    """
    Save model + optimizer + epoch (+ any extra scalars, e.g. best_val_mae)
    to `<checkpoint_dir>/epoch_<N>.pt`. Returns the path written.
    """
    import torch

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[checkpoint] saved {path}")
    return path


def load_latest_checkpoint(
    checkpoint_dir: str,
    model,
    optimizer=None,
    device: str = "cpu",
):
    """
    Find the highest-epoch checkpoint in `checkpoint_dir` and load it into
    `model` (and `optimizer`, if given) in place.

    Returns (start_epoch, model, optimizer):
      - if a checkpoint was found, start_epoch = checkpoint's epoch + 1
        (so a `for epoch in range(start_epoch, N_EPOCHS)` loop resumes
        correctly without repeating a finished epoch)
      - if none was found, start_epoch = 0 and model/optimizer are
        returned untouched (fresh run)

    Safe to call identically on the very first run and on every resume —
    that's the point.
    """
    import torch

    if not os.path.isdir(checkpoint_dir):
        print(f"[resume] no checkpoint dir at {checkpoint_dir} — starting from epoch 0")
        return 0, model, optimizer

    candidates = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))
    if not candidates:
        print(f"[resume] no checkpoints in {checkpoint_dir} — starting from epoch 0")
        return 0, model, optimizer

    def _epoch_num(p: str) -> int:
        m = re.search(r"epoch_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    latest_path = max(candidates, key=_epoch_num)
    payload = torch.load(latest_path, map_location=device)

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    start_epoch = payload["epoch"] + 1
    print(f"[resume] loaded {latest_path} — resuming at epoch {start_epoch}")
    return start_epoch, model, optimizer

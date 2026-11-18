"""Raw-data readers.

Files per trip directory <trip_id>/:
  .track  lon lat t   (whitespace separated, no header)
  .nodes  lon lat     (0-based row index = node id)
  .arcs   from to     (0-based row index = arc id)
  .route  arc_id      (chronological reference route; ground truth)

The reference route is loaded by a *separate* function on purpose so the
prediction pipeline never touches it.
"""
from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np

from . import PROJECT_ROOT, load_config


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray   # (N, 3) lon, lat, t
    nodes: np.ndarray   # (M, 2) lon, lat
    arcs: np.ndarray    # (K, 2) from, to


def partition_dir(partition, cfg=None):
    cfg = cfg or load_config()
    root = Path(cfg["data_root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / cfg["partitions"].get(partition, partition)


def list_trips(partition, cfg=None):
    d = partition_dir(partition, cfg)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def _read(path, dtype=float, ncols=None):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and arr.size == 0:
        arr = arr.reshape(0, ncols)
    return arr


def clean_track(track):
    """Drop non-finite rows, implausible coordinates, sort by time, drop duplicate timestamps."""
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track = track[ok]
    track = track[np.argsort(track[:, 2], kind="stable")]
    if len(track) > 1:
        keep = np.r_[True, np.diff(track[:, 2]) > 0]
        track = track[keep]
    return track


def load_trip(partition, trip_id, cfg=None, clean=True):
    d = partition_dir(partition, cfg) / trip_id
    track = _read(d / f"{trip_id}.track", float, 3)
    if clean:
        track = clean_track(track)
    nodes = _read(d / f"{trip_id}.nodes", float, 2)
    arcs = _read(d / f"{trip_id}.arcs", np.int64, 2)
    return Trip(trip_id, track, nodes, arcs)


def load_route(partition, trip_id, cfg=None):
    """Reference (ground-truth) route. Only for labelling / evaluation, never for prediction."""
    p = partition_dir(partition, cfg) / trip_id / f"{trip_id}.route"
    if not p.exists() or os.path.getsize(p) == 0:
        return None
    return np.loadtxt(p, dtype=np.int64, ndmin=1)

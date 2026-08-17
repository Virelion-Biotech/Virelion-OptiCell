"""Lightweight cell tracking for 2-D time-lapse segmentations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TrackingConfig:
    max_distance_px: float = 30.0
    max_gap: int = 0

def _centroids(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    out = {}
    for label in np.unique(labels):
        if label <= 0: continue
        y, x = np.nonzero(labels == label)
        if len(x): out[int(label)] = (float(x.mean()), float(y.mean()))
    return out

def link_frames(labels_by_time: list[np.ndarray], config: TrackingConfig = TrackingConfig()) -> pd.DataFrame:
    """Greedy nearest-neighbour tracker with optional short gaps."""
    if config.max_distance_px <= 0 or config.max_gap < 0: raise ValueError("invalid tracking configuration")
    tracks = []; next_track = 1; active = {}
    for t, labels in enumerate(labels_by_time):
        centers = _centroids(np.asarray(labels))
        unused = set(centers)
        assignments = []
        candidates = []
        for prev_label, state in active.items():
            last_x, last_y, last_t, track_id = state
            if t - last_t > config.max_gap + 1: continue
            for label, (x, y) in centers.items():
                d = float(np.hypot(x - last_x, y - last_y))
                if d <= config.max_distance_px * max(1.0, t - last_t): candidates.append((d, prev_label, label, track_id))
        used_prev = set(); used_new = set()
        for d, prev_label, label, track_id in sorted(candidates):
            if prev_label in used_prev or label in used_new: continue
            used_prev.add(prev_label); used_new.add(label); unused.discard(label)
            px, py, _, _ = active[prev_label]
            x, y = centers[label]
            tracks.append({"frame": t, "label": label, "track_id": track_id, "x": x, "y": y, "dx": x-px, "dy": y-py, "distance_px": d, "gap": t-active[prev_label][2]-1})
            active.pop(prev_label, None); active[label] = (x, y, t, track_id)
        for label in sorted(unused):
            x, y = centers[label]
            tracks.append({"frame": t, "label": label, "track_id": next_track, "x": x, "y": y, "dx": np.nan, "dy": np.nan, "distance_px": np.nan, "gap": 0})
            active[label] = (x, y, t, next_track); next_track += 1
        active = {label: state for label, state in active.items() if t-state[2] <= config.max_gap}
    return pd.DataFrame(tracks, columns=["frame","label","track_id","x","y","dx","dy","distance_px","gap"])

def summarize_tracks(tracks: pd.DataFrame, pixel_size: float = 1.0, frame_interval: float = 1.0) -> pd.DataFrame:
    """Per-track displacement, path length and average speed."""
    if pixel_size <= 0 or frame_interval <= 0: raise ValueError("pixel_size and frame_interval must be positive")
    required = {"track_id","frame","x","y"}
    missing = required - set(tracks.columns)
    if missing: raise ValueError(f"tracks missing required columns: {sorted(missing)}")
    rows = []
    for tid, group in tracks.sort_values("frame").groupby("track_id"):
        g = group.reset_index(drop=True); n = len(g)
        if n == 0: continue
        dx = np.diff(g["x"].to_numpy()); dy = np.diff(g["y"].to_numpy())
        step = np.hypot(dx, dy); duration = max(1, int(g["frame"].iloc[-1] - g["frame"].iloc[0])) * frame_interval
        rows.append({"track_id": int(tid), "frames": n, "start_frame": int(g["frame"].iloc[0]), "end_frame": int(g["frame"].iloc[-1]), "path_length": float(step.sum()*pixel_size), "net_displacement": float(np.hypot(g["x"].iloc[-1]-g["x"].iloc[0], g["y"].iloc[-1]-g["y"].iloc[0])*pixel_size), "mean_speed": float(step.mean()*pixel_size/frame_interval) if len(step) else 0.0, "net_speed": float(np.hypot(g["x"].iloc[-1]-g["x"].iloc[0], g["y"].iloc[-1]-g["y"].iloc[0])*pixel_size/duration)})
    return pd.DataFrame(rows)

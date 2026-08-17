"""3-D time-series tracking for labelled microscopy volumes.

The tracker uses deterministic centroid matching in physical units and a
constant-velocity prediction when track history is available. It is a
transparent baseline rather than a claim of universal cell-lineage accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Tracking3DConfig:
    max_distance_um: float = 30.0
    max_gap: int = 0
    use_velocity_prediction: bool = True
    velocity_smoothing: float = 0.5

    def validate(self) -> None:
        if self.max_distance_um <= 0:
            raise ValueError("max_distance_um must be > 0")
        if self.max_gap < 0:
            raise ValueError("max_gap must be >= 0")
        if not 0 <= self.velocity_smoothing <= 1:
            raise ValueError("velocity_smoothing must be in [0, 1]")


def _centroids_3d(labels: np.ndarray, voxel_size: Sequence[float]) -> dict[int, np.ndarray]:
    arr = np.asarray(labels)
    if arr.ndim != 3:
        raise ValueError("labels must be a 3-D array")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("labels must contain integer instance IDs")
    spacing = np.asarray(tuple(float(x) for x in voxel_size), dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("voxel_size must contain three positive values")
    out: dict[int, np.ndarray] = {}
    for label in np.unique(arr):
        if label <= 0:
            continue
        coords = np.argwhere(arr == label)
        if len(coords):
            out[int(label)] = coords.mean(axis=0) * spacing
    return out


def link_frames_3d(
    labels_by_time: Sequence[np.ndarray],
    *,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    frame_interval: float = 1.0,
    config: Tracking3DConfig | None = None,
) -> pd.DataFrame:
    """Link 3-D instances across time and return one row per observed object."""
    config = config or Tracking3DConfig()
    config.validate()
    if frame_interval <= 0:
        raise ValueError("frame_interval must be > 0")

    active: dict[int, dict[str, object]] = {}
    next_track = 1
    rows: list[dict[str, object]] = []

    for frame, labels in enumerate(labels_by_time):
        centers = _centroids_3d(labels, voxel_size)
        if not centers:
            active = {
                label: state for label, state in active.items()
                if frame - int(state["last_frame"]) <= config.max_gap
            }
            continue

        labels_now = sorted(centers)
        points = np.vstack([centers[label] for label in labels_now])
        tree = cKDTree(points)
        candidates: list[tuple[float, int, int, np.ndarray]] = []

        for prev_label, state in list(active.items()):
            last = np.asarray(state["position"], dtype=float)
            last_frame = int(state["last_frame"])
            delta_frames = frame - last_frame
            if delta_frames > config.max_gap + 1:
                continue
            if config.use_velocity_prediction and state.get("velocity") is not None and delta_frames > 0:
                velocity = np.asarray(state["velocity"], dtype=float)
                predicted = last + velocity * frame_interval * delta_frames
            else:
                predicted = last
            nearest = tree.query_ball_point(predicted, config.max_distance_um * max(1, delta_frames))
            for idx in nearest:
                new_label = labels_now[idx]
                distance = float(np.linalg.norm(points[idx] - predicted))
                candidates.append((distance, int(prev_label), int(new_label), points[idx]))

        used_prev: set[int] = set()
        used_new: set[int] = set()
        matched_new: set[int] = set()

        for distance, prev_label, new_label, position in sorted(candidates, key=lambda x: x[0]):
            if prev_label in used_prev or new_label in used_new:
                continue
            state = active.get(prev_label)
            if state is None:
                continue
            previous_position = np.asarray(state["position"], dtype=float)
            delta_frames = frame - int(state["last_frame"])
            displacement = position - previous_position
            velocity_obs = displacement / (frame_interval * max(1, delta_frames))
            old_velocity = state.get("velocity")
            if old_velocity is None:
                velocity = velocity_obs
            else:
                alpha = config.velocity_smoothing
                velocity = alpha * velocity_obs + (1 - alpha) * np.asarray(old_velocity, dtype=float)
            track_id = int(state["track_id"])
            gap = delta_frames - 1
            rows.append({
                "frame": frame,
                "label": new_label,
                "track_id": track_id,
                "z_um": float(position[0]),
                "y_um": float(position[1]),
                "x_um": float(position[2]),
                "dz_um": float(displacement[0]),
                "dy_um": float(displacement[1]),
                "dx_um": float(displacement[2]),
                "distance_um": distance,
                "gap": gap,
                "match_confidence": float(max(0.0, 1.0 - distance / (config.max_distance_um * max(1, delta_frames)))),
            })
            active.pop(prev_label, None)
            active[new_label] = {
                "position": position,
                "velocity": velocity,
                "last_frame": frame,
                "track_id": track_id,
            }
            used_prev.add(prev_label)
            used_new.add(new_label)
            matched_new.add(new_label)

        for label in labels_now:
            if label in matched_new:
                continue
            position = centers[label]
            rows.append({
                "frame": frame,
                "label": label,
                "track_id": next_track,
                "z_um": float(position[0]),
                "y_um": float(position[1]),
                "x_um": float(position[2]),
                "dz_um": np.nan,
                "dy_um": np.nan,
                "dx_um": np.nan,
                "distance_um": np.nan,
                "gap": 0,
                "match_confidence": np.nan,
            })
            active[label] = {
                "position": position,
                "velocity": np.zeros(3, dtype=float),
                "last_frame": frame,
                "track_id": next_track,
            }
            next_track += 1

        active = {
            label: state for label, state in active.items()
            if frame - int(state["last_frame"]) <= config.max_gap
        }

    columns = [
        "frame", "label", "track_id", "z_um", "y_um", "x_um",
        "dz_um", "dy_um", "dx_um", "distance_um", "gap", "match_confidence",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_tracks_3d(tracks: pd.DataFrame, *, frame_interval: float = 1.0) -> pd.DataFrame:
    """Summarize 3-D trajectories, including path, net displacement, and straightness."""
    if frame_interval <= 0:
        raise ValueError("frame_interval must be > 0")
    required = {"track_id", "frame", "x_um", "y_um", "z_um"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | int]] = []
    for track_id, group in tracks.sort_values("frame").groupby("track_id"):
        g = group.reset_index(drop=True)
        positions = g[["z_um", "y_um", "x_um"]].to_numpy(dtype=float)
        step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(g) > 1 else np.array([], dtype=float)
        path = float(step_distances.sum())
        net = float(np.linalg.norm(positions[-1] - positions[0]))
        duration_frames = int(g["frame"].iloc[-1] - g["frame"].iloc[0])
        duration = max(duration_frames, 1) * frame_interval
        straightness = float(net / path) if path > 0 else (1.0 if net == 0 else 0.0)
        rows.append({
            "track_id": int(track_id),
            "frames": int(len(g)),
            "start_frame": int(g["frame"].iloc[0]),
            "end_frame": int(g["frame"].iloc[-1]),
            "path_length_um": path,
            "net_displacement_um": net,
            "mean_speed_um_per_frame": float(step_distances.mean() / frame_interval) if len(step_distances) else 0.0,
            "net_speed_um_per_frame": float(net / duration),
            "straightness": straightness,
            "max_step_um": float(step_distances.max()) if len(step_distances) else 0.0,
            "mean_match_confidence": float(g["match_confidence"].dropna().mean()) if "match_confidence" in g and g["match_confidence"].notna().any() else np.nan,
            "max_gap": int(g["gap"].max()) if "gap" in g else 0,
        })
    return pd.DataFrame(rows)


__all__ = ["Tracking3DConfig", "link_frames_3d", "summarize_tracks_3d"]

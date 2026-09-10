"""Assignment-aware 3-D time-series tracking for labelled microscopy volumes."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Tracking3DConfig:
    max_distance_um: float = 30.0
    max_gap: int = 0
    use_velocity_prediction: bool = True
    velocity_smoothing: float = 0.5

    def validate(self) -> None:
        if not np.isfinite(self.max_distance_um) or self.max_distance_um <= 0:
            raise ValueError("max_distance_um must be a finite positive value")
        if self.max_gap < 0:
            raise ValueError("max_gap must be >= 0")
        if not 0 <= self.velocity_smoothing <= 1:
            raise ValueError("velocity_smoothing must be in [0, 1]")


def _centroids_3d(labels: np.ndarray, voxel_size: Sequence[float]) -> dict[int, np.ndarray]:
    arr = np.asarray(labels)
    if arr.ndim != 3 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("labels must be a 3-D integer array")
    spacing = np.asarray(tuple(float(value) for value in voxel_size), dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("voxel_size must contain three finite positive values")
    return {int(label): np.argwhere(arr == label).mean(axis=0) * spacing for label in np.unique(arr) if label > 0}


def _assignment_3d(distance_matrix: np.ndarray, row_limits: np.ndarray) -> list[tuple[int, int, float]]:
    """Return minimum-distance assignments while maximizing valid gated matches."""
    distances = np.asarray(distance_matrix, dtype=float)
    limits = np.asarray(row_limits, dtype=float).reshape(-1)
    if distances.ndim != 2 or limits.shape != (distances.shape[0],):
        raise ValueError("distance_matrix must be 2-D and row_limits must match its rows")
    valid = np.isfinite(distances) & (distances <= limits[:, None])
    if not valid.any():
        return []
    finite_distances = distances[valid]
    penalty = float(finite_distances.max() * min(distances.shape) + 1.0)
    cost = np.where(valid, distances, penalty)
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(row), int(col), float(distances[row, col]))
        for row, col in zip(rows, cols)
        if valid[row, col]
    ]


def link_frames_3d(
    labels_by_time: Sequence[np.ndarray],
    *,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    frame_interval: float = 1.0,
    config: Tracking3DConfig | None = None,
) -> pd.DataFrame:
    """Link 3-D instances using gated one-to-one Hungarian assignment in physical units."""
    cfg = config or Tracking3DConfig()
    cfg.validate()
    if not np.isfinite(frame_interval) or frame_interval <= 0:
        raise ValueError("frame_interval must be a finite positive value")
    spacing = tuple(float(value) for value in voxel_size)
    if len(spacing) != 3 or not np.all(np.isfinite(spacing)) or np.any(np.asarray(spacing) <= 0):
        raise ValueError("voxel_size must contain three finite positive values")
    if not labels_by_time:
        return pd.DataFrame(columns=[
            "frame", "label", "track_id", "z_um", "y_um", "x_um",
            "dz_um", "dy_um", "dx_um", "distance_um", "gap", "match_confidence",
        ])
    reference_shape = np.asarray(labels_by_time[0]).shape
    for labels in labels_by_time:
        arr = np.asarray(labels)
        if arr.shape != reference_shape:
            raise ValueError("all label volumes must have identical shapes")

    rows: list[dict[str, object]] = []
    active: dict[int, dict[str, object]] = {}
    next_track = 1

    for frame, labels in enumerate(labels_by_time):
        centers = _centroids_3d(labels, spacing)
        current_labels = sorted(centers)
        current_points = np.asarray([centers[label] for label in current_labels], dtype=float)
        previous = []
        for track_id, state in list(active.items()):
            gap = frame - int(state["last_frame"])
            if gap <= cfg.max_gap + 1:
                position = np.asarray(state["position"], dtype=float)
                velocity = np.asarray(state["velocity"], dtype=float)
                predicted = position + velocity * frame_interval * gap if cfg.use_velocity_prediction and gap > 0 else position
                previous.append((track_id, state, predicted, gap))

        used_current: set[int] = set()
        if previous and len(current_points):
            previous_points = np.asarray([item[2] for item in previous], dtype=float)
            distance_matrix = np.linalg.norm(previous_points[:, None, :] - current_points[None, :, :], axis=2)
            row_limits = np.asarray([cfg.max_distance_um * max(1, item[3]) for item in previous], dtype=float)
            for previous_index, current_index, distance in _assignment_3d(distance_matrix, row_limits):
                previous_track_id, state, _, gap = previous[previous_index]
                new_label = current_labels[current_index]
                position = current_points[current_index]
                old_position = np.asarray(state["position"], dtype=float)
                displacement = position - old_position
                observed_velocity = displacement / (frame_interval * max(1, gap))
                old_velocity = np.asarray(state["velocity"], dtype=float)
                alpha = cfg.velocity_smoothing
                velocity = alpha * observed_velocity + (1 - alpha) * old_velocity
                track_id = int(state["track_id"])
                rows.append({
                    "frame": frame, "label": new_label, "track_id": track_id,
                    "z_um": float(position[0]), "y_um": float(position[1]), "x_um": float(position[2]),
                    "dz_um": float(displacement[0]), "dy_um": float(displacement[1]), "dx_um": float(displacement[2]),
                    "distance_um": distance, "gap": gap - 1,
                    "match_confidence": max(0.0, 1.0 - distance / row_limits[previous_index]),
                })
                active.pop(previous_track_id, None)
                active[track_id] = {"position": position, "velocity": velocity, "last_frame": frame, "track_id": track_id}
                used_current.add(new_label)

        for label in current_labels:
            if label in used_current:
                continue
            position = centers[label]
            rows.append({
                "frame": frame, "label": label, "track_id": next_track,
                "z_um": float(position[0]), "y_um": float(position[1]), "x_um": float(position[2]),
                "dz_um": np.nan, "dy_um": np.nan, "dx_um": np.nan,
                "distance_um": np.nan, "gap": 0, "match_confidence": np.nan,
            })
            active[next_track] = {"position": position, "velocity": np.zeros(3), "last_frame": frame, "track_id": next_track}
            next_track += 1
        active = {track_id: state for track_id, state in active.items() if frame - int(state["last_frame"]) <= cfg.max_gap}

    columns = ["frame", "label", "track_id", "z_um", "y_um", "x_um", "dz_um", "dy_um", "dx_um", "distance_um", "gap", "match_confidence"]
    return pd.DataFrame(rows, columns=columns)


def summarize_tracks_3d(tracks: pd.DataFrame, *, frame_interval: float = 1.0) -> pd.DataFrame:
    """Summarize 3-D trajectories, including path length, speed, and straightness."""
    if not np.isfinite(frame_interval) or frame_interval <= 0:
        raise ValueError("frame_interval must be a finite positive value")
    required = {"track_id", "frame", "x_um", "y_um", "z_um"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks missing required columns: {sorted(missing)}")
    if tracks.duplicated(["track_id", "frame"]).any():
        raise ValueError("tracks must contain at most one observation per track_id and frame")
    numeric = tracks[["frame", "x_um", "y_um", "z_um"]].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("frame, x_um, y_um, and z_um must contain only finite numeric values")
    if not np.equal(numeric["frame"], np.floor(numeric["frame"])).all():
        raise ValueError("frame values must be integers")

    rows = []
    for track_id, group in tracks.sort_values("frame").groupby("track_id"):
        g = group.reset_index(drop=True)
        positions = g[["z_um", "y_um", "x_um"]].to_numpy(float)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(g) > 1 else np.array([], dtype=float)
        frame_delta = np.diff(g["frame"].to_numpy(float)) if len(g) > 1 else np.array([], dtype=float)
        if np.any(frame_delta <= 0):
            raise ValueError("frame values must increase strictly within each track")
        elapsed = frame_delta * frame_interval
        path = float(steps.sum())
        net = float(np.linalg.norm(positions[-1] - positions[0]))
        duration = max(1, int(g["frame"].iloc[-1] - g["frame"].iloc[0])) * frame_interval
        mean_speed = float(path / elapsed.sum()) if elapsed.size and elapsed.sum() > 0 else 0.0
        rows.append({
            "track_id": int(track_id), "frames": len(g),
            "start_frame": int(g["frame"].iloc[0]), "end_frame": int(g["frame"].iloc[-1]),
            "path_length_um": path, "net_displacement_um": net,
            "mean_speed_um_per_frame": mean_speed,
            "net_speed_um_per_frame": float(net / duration),
            "straightness": float(net / path) if path > 0 else (1.0 if net == 0 else 0.0),
            "max_step_um": float(steps.max()) if len(steps) else 0.0,
            "mean_match_confidence": float(g["match_confidence"].dropna().mean()) if "match_confidence" in g and g["match_confidence"].notna().any() else np.nan,
            "max_gap": int(g["gap"].max()) if "gap" in g else 0,
        })
    return pd.DataFrame(rows)


__all__ = ["Tracking3DConfig", "link_frames_3d", "summarize_tracks_3d"]

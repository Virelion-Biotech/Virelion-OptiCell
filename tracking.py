"""Assignment-aware cell tracking for 2-D time-lapse segmentations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackingConfig:
    max_distance_px: float = 30.0
    max_gap: int = 0
    use_velocity_prediction: bool = True
    velocity_smoothing: float = 0.5
    ambiguity_margin_fraction: float = 0.1

    def validate(self) -> None:
        if not np.isfinite(self.max_distance_px) or self.max_distance_px <= 0:
            raise ValueError("max_distance_px must be a finite positive value")
        if self.max_gap < 0:
            raise ValueError("max_gap must be >= 0")
        if not 0 <= self.velocity_smoothing <= 1:
            raise ValueError("velocity_smoothing must be in [0, 1]")
        if not 0 <= self.ambiguity_margin_fraction <= 1:
            raise ValueError("ambiguity_margin_fraction must be in [0, 1]")


def _centroids(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError("labels must be a 2-D array")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("labels must contain integer instance IDs")
    out: dict[int, tuple[float, float]] = {}
    for label in np.unique(arr):
        if label <= 0:
            continue
        y, x = np.nonzero(arr == label)
        if len(x):
            out[int(label)] = (float(x.mean()), float(y.mean()))
    return out


def _assignment(cost: np.ndarray, row_limits: np.ndarray) -> list[tuple[int, int, float]]:
    """Hungarian assignment with distance gates."""
    if cost.size == 0:
        return []
    gated = np.asarray(cost, dtype=float).copy()
    if gated.ndim != 2:
        raise ValueError("cost must be 2-D")
    limits = np.asarray(row_limits, dtype=float).reshape(-1)
    if limits.shape[0] != gated.shape[0]:
        raise ValueError("row_limits length must match cost rows")
    gated[gated > limits[:, None]] = np.inf
    finite_mask = np.isfinite(gated)
    if not finite_mask.any():
        return []
    row_idx = np.flatnonzero(finite_mask.any(axis=1))
    col_idx = np.flatnonzero(finite_mask.any(axis=0))
    sub = gated[np.ix_(row_idx, col_idx)].copy()
    finite_vals = sub[np.isfinite(sub)]
    big = float(np.max(finite_vals)) if finite_vals.size else 1e6
    if not np.isfinite(big) or big <= 0:
        big = 1e6
    sub[~np.isfinite(sub)] = big * 10.0 + 1.0
    rows, cols = linear_sum_assignment(sub)
    out: list[tuple[int, int, float]] = []
    for r, c in zip(rows, cols):
        dist = float(gated[int(row_idx[r]), int(col_idx[c])])
        if np.isfinite(dist):
            out.append((int(row_idx[r]), int(col_idx[c]), dist))
    return out


def link_frames(labels_by_time: list[np.ndarray], config: TrackingConfig | None = None) -> pd.DataFrame:
    """Link 2-D instances using one-to-one assignment and optional short gaps.

    Assignment IDs are independent of per-frame segmentation labels. For matched
    observations, local assignment ambiguity is reported by comparing the chosen
    distance with the second-best gated candidate for the same previous track.
    This is a diagnostic, not a calibrated identity-swap probability.
    """
    cfg = config or TrackingConfig()
    cfg.validate()
    rows: list[dict[str, object]] = []
    active: dict[int, dict[str, object]] = {}
    next_track = 1

    for frame, labels in enumerate(labels_by_time):
        centers = _centroids(labels)
        current_labels = sorted(centers)
        current_points = np.asarray([centers[label] for label in current_labels], dtype=float)
        previous = []
        for track_id, state in list(active.items()):
            gap = frame - int(state["last_frame"])
            if gap <= cfg.max_gap + 1:
                position = np.asarray(state["position"], dtype=float)
                velocity = np.asarray(state["velocity"], dtype=float)
                predicted = position + velocity * gap if cfg.use_velocity_prediction and gap > 0 else position
                previous.append((track_id, state, predicted, gap))

        used_current: set[int] = set()
        if previous and len(current_points):
            previous_points = np.asarray([p[2] for p in previous], dtype=float)
            distances = np.linalg.norm(previous_points[:, None, :] - current_points[None, :, :], axis=2)
            row_limits = np.asarray([cfg.max_distance_px * max(1, p[3]) for p in previous], dtype=float)
            for prev_idx, new_idx, distance in _assignment(distances, row_limits):
                previous_track_id, state, _, gap = previous[prev_idx]
                new_label = current_labels[new_idx]
                position = current_points[new_idx]
                old_position = np.asarray(state["position"], dtype=float)
                displacement = position - old_position
                observed_velocity = displacement / max(1, gap)
                old_velocity = np.asarray(state["velocity"], dtype=float)
                alpha = cfg.velocity_smoothing
                velocity = alpha * observed_velocity + (1 - alpha) * old_velocity
                track_id = int(state["track_id"])

                candidate_distances = distances[prev_idx]
                valid = np.isfinite(candidate_distances) & (candidate_distances <= row_limits[prev_idx])
                valid[new_idx] = False
                alternative_distance = float(np.min(candidate_distances[valid])) if np.any(valid) else np.nan
                assignment_margin = alternative_distance - distance if np.isfinite(alternative_distance) else np.nan
                assignment_margin_fraction = (
                    assignment_margin / max(row_limits[prev_idx], np.finfo(float).eps)
                    if np.isfinite(assignment_margin)
                    else np.nan
                )
                assignment_ambiguous = (
                    bool(assignment_margin_fraction < cfg.ambiguity_margin_fraction)
                    if np.isfinite(assignment_margin_fraction)
                    else np.nan
                )

                rows.append({
                    "frame": frame, "label": new_label, "track_id": track_id,
                    "x": float(position[0]), "y": float(position[1]),
                    "dx": float(displacement[0]), "dy": float(displacement[1]),
                    "distance_px": distance, "gap": gap - 1,
                    "match_confidence": max(0.0, 1.0 - distance / row_limits[prev_idx]),
                    "alternative_distance_px": alternative_distance,
                    "assignment_margin_px": assignment_margin,
                    "assignment_margin_fraction": assignment_margin_fraction,
                    "assignment_ambiguous": assignment_ambiguous,
                })
                active.pop(previous_track_id, None)
                active[track_id] = {
                    "position": position,
                    "velocity": velocity,
                    "last_frame": frame,
                    "track_id": track_id,
                }
                used_current.add(new_label)

        for label in current_labels:
            if label in used_current:
                continue
            position = centers[label]
            rows.append({
                "frame": frame, "label": label, "track_id": next_track,
                "x": position[0], "y": position[1], "dx": np.nan, "dy": np.nan,
                "distance_px": np.nan, "gap": 0, "match_confidence": np.nan,
                "alternative_distance_px": np.nan,
                "assignment_margin_px": np.nan,
                "assignment_margin_fraction": np.nan,
                "assignment_ambiguous": np.nan,
            })
            active[next_track] = {
                "position": np.asarray(position),
                "velocity": np.zeros(2),
                "last_frame": frame,
                "track_id": next_track,
            }
            next_track += 1

        active = {
            track_id: state
            for track_id, state in active.items()
            if frame - int(state["last_frame"]) <= cfg.max_gap
        }

    columns = [
        "frame", "label", "track_id", "x", "y",
        "dx", "dy", "distance_px", "gap", "match_confidence",
        "alternative_distance_px", "assignment_margin_px",
        "assignment_margin_fraction", "assignment_ambiguous",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_tracks(
    tracks: pd.DataFrame,
    pixel_size: float = 1.0,
    frame_interval: float = 1.0,
) -> pd.DataFrame:
    """Summarize trajectory length, displacement, speed, straightness, and confidence."""
    if not np.isfinite(pixel_size) or pixel_size <= 0 or not np.isfinite(frame_interval) or frame_interval <= 0:
        raise ValueError("pixel_size and frame_interval must be finite positive values")
    required = {"track_id", "frame", "x", "y"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks missing required columns: {sorted(missing)}")
    if tracks.duplicated(["track_id", "frame"]).any():
        raise ValueError("tracks must contain at most one observation per track_id and frame")
    numeric = tracks[["frame", "x", "y"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("frame, x, and y must contain only finite numeric values")
    if not np.equal(numeric["frame"], np.floor(numeric["frame"])).all():
        raise ValueError("frame values must be integers")

    rows = []
    for track_id, group in tracks.sort_values("frame").groupby("track_id"):
        g = group.reset_index(drop=True)
        positions = g[["x", "y"]].to_numpy(float)
        step = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(g) > 1 else np.array([], dtype=float)
        frame_delta = np.diff(g["frame"].to_numpy(float)) if len(g) > 1 else np.array([], dtype=float)
        if np.any(frame_delta <= 0):
            raise ValueError("frame values must increase strictly within each track")
        duration = max(1, int(g["frame"].iloc[-1] - g["frame"].iloc[0])) * frame_interval
        path = float(step.sum() * pixel_size)
        net = float(np.linalg.norm(positions[-1] - positions[0]) * pixel_size)
        elapsed = frame_delta * frame_interval
        mean_speed = float(np.sum(step * pixel_size) / np.sum(elapsed)) if np.all(elapsed > 0) else 0.0
        rows.append({
            "track_id": int(track_id), "frames": len(g),
            "start_frame": int(g["frame"].iloc[0]), "end_frame": int(g["frame"].iloc[-1]),
            "path_length": path, "net_displacement": net,
            "mean_speed": mean_speed, "net_speed": float(net / duration),
            "straightness": float(net / path) if path > 0 else (1.0 if net == 0 else 0.0),
            "mean_match_confidence": (
                float(g["match_confidence"].dropna().mean())
                if "match_confidence" in g and g["match_confidence"].notna().any()
                else np.nan
            ),
            "max_gap": int(g["gap"].max()) if "gap" in g else 0,
        })
    return pd.DataFrame(rows)


__all__ = ["TrackingConfig", "link_frames", "summarize_tracks"]

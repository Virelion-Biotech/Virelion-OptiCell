"""Validation and benchmarking utilities for OptiCell segmentation.

These metrics are intentionally model-agnostic and can be used with manually
annotated masks or synthetic ground truth. They are not a substitute for a
proper domain-specific validation study.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def binary_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Pixel-level intersection-over-union for binary masks."""
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape:
        raise ValueError("pred and truth must have identical shapes")
    union = np.logical_or(p, t).sum()
    return float(np.logical_and(p, t).sum() / union) if union else 1.0


def binary_dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Pixel-level Sørensen-Dice coefficient."""
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape:
        raise ValueError("pred and truth must have identical shapes")
    denom = p.sum() + t.sum()
    return float(2 * np.logical_and(p, t).sum() / denom) if denom else 1.0


def segmentation_pixel_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Return IoU, Dice, precision and recall for foreground segmentation."""
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape:
        raise ValueError("pred and truth must have identical shapes")
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "iou": binary_iou(p, t),
        "dice": binary_dice(p, t),
        "precision": float(precision),
        "recall": float(recall),
    }


def count_error(pred_count: int, truth_count: int) -> dict[str, float]:
    """Return absolute and relative cell-count error."""
    absolute = abs(int(pred_count) - int(truth_count))
    relative = absolute / abs(truth_count) if truth_count else (0.0 if pred_count == 0 else float("inf"))
    return {"absolute_count_error": float(absolute), "relative_count_error": float(relative)}


def _centroids_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    points = []
    for label_id in ids:
        y, x = np.nonzero(labels == label_id)
        if len(x):
            points.append((float(x.mean()), float(y.mean())))
    return np.asarray(points, dtype=float)


def match_instance_centroids(
    predicted_labels: np.ndarray,
    truth_labels: np.ndarray,
    max_distance_px: float = 20.0,
) -> tuple[int, int, int]:
    """Greedy one-to-one centroid matching for instance-level validation."""
    pred = _centroids_from_labels(predicted_labels)
    truth = _centroids_from_labels(truth_labels)
    if pred.size == 0 and truth.size == 0:
        return 0, 0, 0
    if pred.size == 0:
        return 0, 0, len(truth)
    if truth.size == 0:
        return 0, len(pred), 0

    distances = np.sqrt(((pred[:, None, :] - truth[None, :, :]) ** 2).sum(axis=2))
    candidates = np.argwhere(distances <= max_distance_px)
    order = sorted(candidates.tolist(), key=lambda ij: distances[ij[0], ij[1]])
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    tp = 0
    for p_idx, t_idx in order:
        if p_idx in used_pred or t_idx in used_truth:
            continue
        used_pred.add(p_idx)
        used_truth.add(t_idx)
        tp += 1
    fp = len(pred) - tp
    fn = len(truth) - tp
    return tp, fp, fn


def instance_metrics(predicted_labels: np.ndarray, truth_labels: np.ndarray, max_distance_px: float = 20.0) -> dict[str, float]:
    """Calculate object-level precision, recall, F1 and count error."""
    tp, fp, fn = match_instance_centroids(predicted_labels, truth_labels, max_distance_px)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pred_count = int(np.max(predicted_labels)) if np.asarray(predicted_labels).size else 0
    truth_count = int(np.max(truth_labels)) if np.asarray(truth_labels).size else 0
    return {
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        **count_error(pred_count, truth_count),
    }


def benchmark_segmentation(
    predicted_labels: Sequence[np.ndarray],
    truth_labels: Sequence[np.ndarray],
    max_distance_px: float = 20.0,
) -> dict[str, Any]:
    """Aggregate pixel and instance metrics over paired validation images."""
    if len(predicted_labels) != len(truth_labels):
        raise ValueError("predicted_labels and truth_labels must have the same length")
    if not predicted_labels:
        raise ValueError("At least one validation image is required")

    pixel_rows = []
    instance_rows = []
    for pred, truth in zip(predicted_labels, truth_labels):
        pixel_rows.append(segmentation_pixel_metrics(np.asarray(pred) > 0, np.asarray(truth) > 0))
        instance_rows.append(instance_metrics(pred, truth, max_distance_px=max_distance_px))

    def mean(key: str, rows: list[dict[str, float]]) -> float:
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_images": len(pixel_rows),
        "pixel_iou_mean": mean("iou", pixel_rows),
        "pixel_dice_mean": mean("dice", pixel_rows),
        "pixel_precision_mean": mean("precision", pixel_rows),
        "pixel_recall_mean": mean("recall", pixel_rows),
        "instance_precision_mean": mean("precision", instance_rows),
        "instance_recall_mean": mean("recall", instance_rows),
        "instance_f1_mean": mean("f1", instance_rows),
        "absolute_count_error_mean": mean("absolute_count_error", instance_rows),
        "relative_count_error_mean": mean("relative_count_error", instance_rows),
    }

"""Validation and benchmarking utilities for OptiCell segmentation.

These metrics are model-agnostic and support both 2-D and 3-D labelled masks.
They are not a substitute for a domain-specific validation study.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment


def binary_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool); t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape: raise ValueError("pred and truth must have identical shapes")
    union = np.logical_or(p, t).sum()
    return float(np.logical_and(p, t).sum() / union) if union else 1.0


def binary_dice(pred: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool); t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape: raise ValueError("pred and truth must have identical shapes")
    denom = p.sum() + t.sum()
    return float(2 * np.logical_and(p, t).sum() / denom) if denom else 1.0


def segmentation_pixel_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=bool); t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape: raise ValueError("pred and truth must have identical shapes")
    tp = int(np.logical_and(p, t).sum()); fp = int(np.logical_and(p, ~t).sum()); fn = int(np.logical_and(~p, t).sum())
    return {"iou": binary_iou(p, t), "dice": binary_dice(p, t), "precision": float(tp / (tp + fp) if tp + fp else 1.0), "recall": float(tp / (tp + fn) if tp + fn else 1.0)}


def count_error(pred_count: int, truth_count: int) -> dict[str, float]:
    absolute = abs(int(pred_count) - int(truth_count))
    relative = absolute / abs(truth_count) if truth_count else (0.0 if pred_count == 0 else float("inf"))
    return {"absolute_count_error": float(absolute), "relative_count_error": float(relative)}


def _label_ids(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim not in {2, 3}: raise ValueError("label arrays must be 2-D or 3-D")
    return np.unique(arr[arr > 0])


def _centroids_from_labels(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    ids = _label_ids(arr)
    return np.asarray([np.argwhere(arr == label).mean(axis=0) for label in ids], dtype=float)


def _instance_count(labels: np.ndarray) -> int:
    return int(_label_ids(np.asarray(labels)).size)


def match_instance_centroids(predicted_labels: np.ndarray, truth_labels: np.ndarray, max_distance_px: float = 20.0) -> tuple[int, int, int]:
    """Perform globally optimal one-to-one centroid matching within a distance gate."""
    if max_distance_px <= 0: raise ValueError("max_distance_px must be positive")
    pred = _centroids_from_labels(predicted_labels); truth = _centroids_from_labels(truth_labels)
    if pred.size == 0 and truth.size == 0: return 0, 0, 0
    if pred.size == 0: return 0, 0, len(truth)
    if truth.size == 0: return 0, len(pred), 0
    distances = np.sqrt(((pred[:, None, :] - truth[None, :, :]) ** 2).sum(axis=2))
    cost = distances.copy()
    cost[cost > max_distance_px] = max_distance_px + 1.0
    rows, cols = linear_sum_assignment(cost)
    tp = int(sum(distances[r, c] <= max_distance_px for r, c in zip(rows, cols)))
    return tp, len(pred) - tp, len(truth) - tp


def instance_metrics(predicted_labels: np.ndarray, truth_labels: np.ndarray, max_distance_px: float = 20.0) -> dict[str, float]:
    tp, fp, fn = match_instance_centroids(predicted_labels, truth_labels, max_distance_px)
    precision = tp / (tp + fp) if tp + fp else 1.0; recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pred_count = _instance_count(predicted_labels); truth_count = _instance_count(truth_labels)
    return {"true_positives": float(tp), "false_positives": float(fp), "false_negatives": float(fn), "precision": float(precision), "recall": float(recall), "f1": float(f1), **count_error(pred_count, truth_count)}


def paired_segmentation_metrics(predicted: np.ndarray, truth: np.ndarray, max_distance_px: float = 20.0) -> dict[str, float]:
    """Return pixel/voxel and instance metrics for one paired prediction."""
    return {**segmentation_pixel_metrics(np.asarray(predicted) > 0, np.asarray(truth) > 0), **instance_metrics(predicted, truth, max_distance_px)}


def benchmark_segmentation(predicted_labels: Sequence[np.ndarray], truth_labels: Sequence[np.ndarray], max_distance_px: float = 20.0) -> dict[str, float]:
    """Aggregate metrics and expose both current prefixed keys and legacy aliases."""
    if len(predicted_labels) != len(truth_labels): raise ValueError("predicted_labels and truth_labels must have the same length")
    if not predicted_labels: raise ValueError("At least one validation image is required")
    rows = [paired_segmentation_metrics(pred, truth, max_distance_px) for pred, truth in zip(predicted_labels, truth_labels)]
    mean = lambda key: float(np.nanmean([row[key] for row in rows]))
    return {
        "n_images": float(len(rows)),
        "pixel_iou_mean": mean("iou"), "pixel_dice_mean": mean("dice"),
        "pixel_precision_mean": mean("precision"), "pixel_recall_mean": mean("recall"),
        "instance_precision_mean": mean("precision"), "instance_recall_mean": mean("recall"),
        "instance_f1_mean": mean("f1"), "absolute_count_error_mean": mean("absolute_count_error"),
        "relative_count_error_mean": mean("relative_count_error"),
        "iou": mean("iou"), "dice": mean("dice"), "precision": mean("precision"), "recall": mean("recall"),
        "f1": mean("f1"), "absolute_count_error": mean("absolute_count_error"), "relative_count_error": mean("relative_count_error"),
    }

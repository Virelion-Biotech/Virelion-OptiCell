"""Reproducible segmentation backend benchmarking."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
import pandas as pd

from validation import paired_segmentation_metrics


def _positive_label_count(labels: np.ndarray) -> int:
    values = np.asarray(labels)
    if values.ndim not in {2, 3} or 0 in values.shape or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("segmentation labels must be non-empty 2-D or 3-D integer arrays")
    positive = values[values > 0]
    return int(np.unique(positive).size) if positive.size else 0


def benchmark_backends(
    image: np.ndarray,
    reference_labels: np.ndarray,
    backends: Mapping[str, object],
    *,
    max_distance: float = 3.0,
    metadata: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Run several segmenters against ground truth with explicit timing and metadata."""
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be a finite positive value")
    if not isinstance(backends, Mapping):
        raise ValueError("backends must be a mapping of names to segmenters")
    image_array = np.asarray(image)
    reference = np.asarray(reference_labels)
    if image_array.size == 0 or reference.size == 0:
        raise ValueError("image and reference_labels must be non-empty")
    if reference.ndim not in {2, 3} or not np.issubdtype(reference.dtype, np.integer):
        raise ValueError("reference_labels must be a 2-D or 3-D integer array")
    if not np.issubdtype(image_array.dtype, np.number) or not np.isfinite(image_array).all():
        raise ValueError("image must contain finite numeric values")
    reference_instances = _positive_label_count(reference)
    supplied_metadata = dict(metadata or {})
    if any(key in supplied_metadata for key in ("backend", "instances_reference")):
        raise ValueError("metadata cannot override benchmark identity fields")
    rows: list[dict[str, object]] = []
    for name, backend in backends.items():
        if not isinstance(name, str) or not name:
            raise ValueError("backend names must be non-empty strings")
        if not hasattr(backend, "segment") or not callable(backend.segment):
            raise ValueError(f"backend {name!r} must provide a callable segment method")
        started = perf_counter()
        base = {"backend": name, "instances_reference": reference_instances, **supplied_metadata}
        try:
            result = backend.segment(image_array)
            if not hasattr(result, "labels"):
                raise ValueError("backend result must provide a labels attribute")
            labels = np.asarray(result.labels)
            if labels.shape != reference.shape:
                raise ValueError("predicted labels and reference labels must have identical shapes")
            metrics = paired_segmentation_metrics(labels, reference, max_distance_px=max_distance)
            elapsed = perf_counter() - started
            rows.append({
                **base,
                "elapsed_seconds": elapsed,
                "instances_predicted": _positive_label_count(labels),
                "error": None,
                **metrics,
            })
        except Exception as exc:
            elapsed = perf_counter() - started
            rows.append({
                **base,
                "elapsed_seconds": elapsed,
                "instances_predicted": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(rows)


def aggregate_backend_benchmarks(results: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-image benchmark tables while preserving failure counts."""
    if not results:
        return pd.DataFrame()
    if any(not isinstance(result, pd.DataFrame) for result in results):
        raise ValueError("results must contain pandas DataFrames")
    frame = pd.concat(results, ignore_index=True)
    if "backend" not in frame.columns:
        raise ValueError("benchmark results must contain a backend column")
    numeric = [c for c in ["elapsed_seconds", "instances_predicted", "instances_reference", "iou", "dice", "precision", "recall", "f1", "absolute_count_error", "relative_count_error"] if c in frame.columns]
    failures = (
        frame.assign(_failed=frame["error"].notna() if "error" in frame.columns else False)
        .groupby("backend", dropna=False)["_failed"]
        .sum()
        .astype(int)
        .rename("failed_runs")
        .reset_index()
    )
    if not numeric:
        return failures
    grouped = frame.groupby("backend", dropna=False)[numeric].agg(["mean", "median", "std", "count"]).reset_index()
    grouped.columns = [str(column[0]) if isinstance(column, tuple) and column[1] == "" else (f"{column[0]}_{column[1]}" if isinstance(column, tuple) else str(column)) for column in grouped.columns]
    return grouped.merge(failures, on="backend", how="left")


__all__ = ["aggregate_backend_benchmarks", "benchmark_backends"]

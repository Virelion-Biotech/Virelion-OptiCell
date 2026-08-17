"""Reproducible segmentation backend benchmarking."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
import pandas as pd

from validation import paired_segmentation_metrics


def benchmark_backends(
    image: np.ndarray,
    reference_labels: np.ndarray,
    backends: Mapping[str, object],
    *,
    max_distance: float = 3.0,
    metadata: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Run several segmenters against ground truth with explicit timing and metadata."""
    if max_distance <= 0:
        raise ValueError("max_distance must be > 0")
    rows: list[dict[str, object]] = []
    reference = np.asarray(reference_labels)
    reference_instances = int(np.max(reference)) if reference.size else 0
    for name, backend in backends.items():
        started = perf_counter()
        base = {"backend": name, "instances_reference": reference_instances, **dict(metadata or {})}
        try:
            result = backend.segment(image)
            labels = np.asarray(result.labels)
            metrics = paired_segmentation_metrics(labels, reference, max_distance_px=max_distance)
            rows.append({**base, "elapsed_seconds": perf_counter() - started, "instances_predicted": int(np.max(labels)) if labels.size else 0, "error": None, **metrics})
        except Exception as exc:
            rows.append({**base, "elapsed_seconds": perf_counter() - started, "instances_predicted": 0, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def aggregate_backend_benchmarks(results: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-image benchmark tables while preserving failure counts."""
    if not results:
        return pd.DataFrame()
    frame = pd.concat(results, ignore_index=True)
    if "backend" not in frame.columns:
        raise ValueError("benchmark results must contain a backend column")
    numeric = [c for c in ["elapsed_seconds", "instances_predicted", "instances_reference", "iou", "dice", "precision", "recall", "f1", "absolute_count_error", "relative_count_error"] if c in frame.columns]
    grouped = frame.groupby("backend", dropna=False)[numeric].agg(["mean", "median", "std", "count"]).reset_index()
    grouped.columns = [str(column[0]) if isinstance(column, tuple) and column[1] == "" else (f"{column[0]}_{column[1]}" if isinstance(column, tuple) else str(column)) for column in grouped.columns]
    failures = frame.groupby("backend", dropna=False)["error"].apply(lambda values: int(values.notna().sum())).rename("failed_runs").reset_index()
    return grouped.merge(failures, on="backend", how="left")


__all__ = ["aggregate_backend_benchmarks", "benchmark_backends"]

"""Reproducible segmentation backend benchmarking."""
from __future__ import annotations

from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from validation import paired_segmentation_metrics


def benchmark_backends(
    image: np.ndarray,
    reference_labels: np.ndarray,
    backends: Mapping[str, object],
    *,
    max_distance: float = 3.0,
) -> pd.DataFrame:
    """Run several segmenters against one ground-truth label image.

    Every backend must expose ``segment(image)`` and return an object with a
    ``labels`` array. Failures are retained as rows for auditability.
    """
    rows: list[dict[str, object]] = []
    reference_instances = int(np.max(reference_labels)) if np.asarray(reference_labels).size else 0
    for name, backend in backends.items():
        started = perf_counter()
        try:
            result = backend.segment(image)
            labels = np.asarray(result.labels)
            metrics = paired_segmentation_metrics(labels, reference_labels, max_distance_px=max_distance)
            rows.append({"backend": name, "elapsed_seconds": perf_counter() - started,
                         "instances_predicted": int(np.max(labels)) if labels.size else 0,
                         "instances_reference": reference_instances, "error": None, **metrics})
        except Exception as exc:
            rows.append({"backend": name, "elapsed_seconds": perf_counter() - started,
                         "instances_predicted": 0, "instances_reference": reference_instances,
                         "error": repr(exc)})
    return pd.DataFrame(rows)


def aggregate_backend_benchmarks(results: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-image benchmark tables while preserving failure counts."""
    if not results:
        return pd.DataFrame()
    frame = pd.concat(results, ignore_index=True)
    numeric = [c for c in ["elapsed_seconds", "instances_predicted", "instances_reference",
                           "iou", "dice", "precision", "recall", "f1",
                           "absolute_count_error", "relative_count_error"] if c in frame.columns]
    grouped = frame.groupby("backend", dropna=False)[numeric].agg(["mean", "median", "std", "count"])
    grouped["failed_runs"] = frame.groupby("backend", dropna=False)["error"].apply(lambda s: int(s.notna().sum())).to_numpy()
    grouped.columns = [f"{a}_{b}" if b else str(a) for a, b in grouped.columns]
    return grouped.reset_index()


__all__ = ["benchmark_backends", "aggregate_backend_benchmarks"]

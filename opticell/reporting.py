"""Machine-readable experiment and performance reports for OptiCell."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuntimeStats:
    operation: str
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    items: int
    items_per_second: float


def runtime_stats(operation: str, started: datetime, finished: datetime, items: int) -> RuntimeStats:
    if started.tzinfo is None or started.utcoffset() is None or finished.tzinfo is None or finished.utcoffset() is None:
        raise ValueError("started and finished datetimes must be timezone-aware")
    if finished < started:
        raise ValueError("finished datetime must be at or after started datetime")
    if not isinstance(items, (int, np.integer)) or isinstance(items, bool) or items < 0:
        raise ValueError("items must be a non-negative integer")
    elapsed = (finished - started).total_seconds()
    return RuntimeStats(
        operation=str(operation),
        started_utc=started.astimezone(timezone.utc).isoformat(),
        finished_utc=finished.astimezone(timezone.utc).isoformat(),
        elapsed_seconds=float(elapsed),
        items=int(items),
        items_per_second=float(items / elapsed) if elapsed > 0 else 0.0,
    )


def dataframe_summary(df: pd.DataFrame, numeric_columns: list[str] | None = None) -> dict[str, Any]:
    """Summarize a result table without serializing the full dataset into the report."""
    columns = numeric_columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric: dict[str, dict[str, float | int]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        numeric[column] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {"rows": int(len(df)), "columns": [str(c) for c in df.columns], "numeric": numeric}


def build_report(
    *,
    operation: str,
    runtime: RuntimeStats | None = None,
    results: pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "metadata": dict(metadata or {}),
        "results": dataframe_summary(results) if results is not None else {},
        "validation": dict(validation or {}),
    }
    if runtime is not None:
        report["runtime"] = asdict(runtime)
    return report


def write_report(report: Mapping[str, Any], path: str) -> str:
    """Write a JSON report with NaN/NumPy values normalized for portability."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, np.ndarray):
            return normalize(value.tolist())
        if isinstance(value, np.generic):
            return normalize(value.item())
        if value is pd.NA:
            return None
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    target.write_text(json.dumps(normalize(dict(report)), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(target)


__all__ = ["RuntimeStats", "runtime_stats", "dataframe_summary", "build_report", "write_report"]

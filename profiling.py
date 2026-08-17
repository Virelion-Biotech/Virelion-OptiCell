"""Lightweight reproducible profiling helpers for OptiCell pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class ProfileRecord:
    operation: str
    elapsed_seconds: float
    items: int = 1
    status: str = "success"
    error: str | None = None

    @property
    def items_per_second(self) -> float:
        return self.items / self.elapsed_seconds if self.elapsed_seconds > 0 else float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "elapsed_seconds": self.elapsed_seconds,
            "items": self.items,
            "items_per_second": self.items_per_second,
            "status": self.status,
            "error": self.error,
        }


def profile_call(operation: str, callable_: Callable[..., Any], *args: Any, items: int = 1, **kwargs: Any) -> tuple[Any, ProfileRecord]:
    """Time one callable and return its result plus a structured profile record."""
    started = perf_counter()
    try:
        result = callable_(*args, **kwargs)
    except Exception as exc:
        elapsed = perf_counter() - started
        raise RuntimeError(f"profiled operation {operation!r} failed after {elapsed:.6f}s") from exc
    elapsed = perf_counter() - started
    return result, ProfileRecord(operation, elapsed, int(items))


def profile_records(records: list[ProfileRecord]) -> pd.DataFrame:
    """Convert profiling records to a stable, analysis-friendly table."""
    return pd.DataFrame([record.as_dict() for record in records], columns=["operation", "elapsed_seconds", "items", "items_per_second", "status", "error"])


def summarize_profile(records: list[ProfileRecord]) -> dict[str, float]:
    """Summarize total time, items, and aggregate throughput."""
    total_time = float(sum(record.elapsed_seconds for record in records))
    total_items = int(sum(record.items for record in records))
    return {
        "operations": float(len(records)),
        "total_elapsed_seconds": total_time,
        "total_items": float(total_items),
        "overall_items_per_second": float(total_items / total_time) if total_time > 0 else float("inf"),
        "failed_operations": float(sum(record.status != "success" for record in records)),
    }


__all__ = ["ProfileRecord", "profile_call", "profile_records", "summarize_profile"]

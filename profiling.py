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


class ProfiledOperationError(RuntimeError):
    """Raised for a failed profiled operation while retaining its profile record."""

    def __init__(self, message: str, record: ProfileRecord, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.record = record
        self.__cause__ = cause


def profile_call(operation: str, callable_: Callable[..., Any], *args: Any, items: int = 1, **kwargs: Any) -> tuple[Any, ProfileRecord]:
    """Time one callable and return its result plus a structured profile record.

    Failures still propagate, but the raised exception exposes the failed
    ``ProfileRecord`` so callers can retain structured profiling evidence.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    if not isinstance(items, int) or isinstance(items, bool) or items < 0:
        raise ValueError("items must be a non-negative integer")
    started = perf_counter()
    try:
        result = callable_(*args, **kwargs)
    except Exception as exc:
        elapsed = perf_counter() - started
        record = ProfileRecord(operation, elapsed, items=int(items), status="error", error=f"{type(exc).__name__}: {exc}")
        raise ProfiledOperationError(
            f"profiled operation {operation!r} failed after {elapsed:.6f}s",
            record,
            cause=exc,
        )
    elapsed = perf_counter() - started
    return result, ProfileRecord(operation, elapsed, int(items), status="success", error=None)


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


__all__ = ["ProfileRecord", "ProfiledOperationError", "profile_call", "profile_records", "summarize_profile"]

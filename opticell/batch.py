"""High-throughput, failure-isolating batch execution for OptiCell."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from typing import Callable, Optional, Sequence

import pandas as pd

from qc_pipeline import QCThresholds, adaptive_dataset_qc, analyze_image


@dataclass(frozen=True)
class BatchConfig:
    """Controls for parallel image analysis."""

    workers: int = 1
    cell_method: str = "threshold"
    adaptive_qc: bool = True
    adaptive_threshold: bool = False
    fail_fast: bool = False

    def validate(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.cell_method not in {"threshold", "cellpose"}:
            raise ValueError("cell_method must be 'threshold' or 'cellpose'")


def analyze_paths_parallel(
    paths: Sequence[str],
    *,
    thresholds: Optional[QCThresholds] = None,
    config: Optional[BatchConfig] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Analyze independent paths concurrently while preserving deterministic output order."""
    config = config or BatchConfig()
    config.validate()
    thresholds = thresholds or QCThresholds()
    thresholds.validate()
    normalized = [os.fspath(path) for path in paths]
    total = len(normalized)
    if not normalized:
        return adaptive_dataset_qc(pd.DataFrame()) if config.adaptive_qc else pd.DataFrame()

    def analyze_one(path: str):
        return analyze_image(
            path,
            thresholds=thresholds,
            cell_method=config.cell_method,
            adaptive_threshold=config.adaptive_threshold,
        )

    rows: list[dict] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(analyze_one, path): path for path in normalized}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                if isinstance(result, tuple):
                    result = result[0]
                rows.append(result.to_row())
            except Exception as exc:
                if config.fail_fast:
                    for pending in futures:
                        pending.cancel()
                    raise
                rows.append(
                    {
                        "filename": os.path.basename(path),
                        "path": os.path.abspath(path),
                        "flags": "BATCH_WORKER_ERROR",
                        "error": str(exc),
                    }
                )
            completed += 1
            if progress_callback:
                progress_callback(completed, total, os.path.basename(path))

    frame = pd.DataFrame(rows)
    if not frame.empty and "path" in frame.columns:
        frame = frame.sort_values("path", kind="stable").reset_index(drop=True)
    return adaptive_dataset_qc(frame) if config.adaptive_qc else frame


__all__ = ["BatchConfig", "analyze_paths_parallel"]

"""Interoperable tabular export helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def dataframe_to_long_form(frame: pd.DataFrame, *, id_columns: list[str]) -> pd.DataFrame:
    """Convert wide cell/object features into deterministic feature/value rows."""
    missing = set(id_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"id_columns missing from dataframe: {sorted(missing)}")
    value_columns = [column for column in frame.columns if column not in id_columns]
    long = frame.melt(id_vars=id_columns, value_vars=value_columns, var_name="feature", value_name="value")
    return long.sort_values(id_columns + ["feature"], kind="stable").reset_index(drop=True)


def write_dataframe(frame: pd.DataFrame, path: str) -> str:
    """Write CSV or Parquet, requiring pyarrow/fastparquet only for Parquet."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(target, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(target, index=False)
    else:
        raise ValueError("supported export formats are .csv and .parquet")
    return str(target)


__all__ = ["dataframe_to_long_form", "write_dataframe"]

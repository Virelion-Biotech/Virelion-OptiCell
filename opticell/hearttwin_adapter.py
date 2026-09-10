"""HeartTwin local-command adapter for OptiCell imaging QC."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qc_pipeline import QCThresholds, analyze_folder, analyze_paths


def _find_input(payload: dict) -> tuple[str, dict]:
    for obs in payload.get("observations", []):
        if obs.get("modality") == "imaging" and "input_path" in obs.get("values", {}):
            return str(obs["values"]["input_path"]), obs["values"]
    raise ValueError("No 'imaging' observation with values.input_path was provided")


def _parse_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError("boolean parameters must be bool, 0/1, or a recognized boolean string")


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        records = value.astype(object).where(pd.notna(value), None).to_dict(orient="records")
        return [_jsonable(record) for record in records]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def main() -> int:
    raw = os.environ.get("HEARTTWIN_PAYLOAD")
    if not raw:
        print("HEARTTWIN_PAYLOAD environment variable not set", file=sys.stderr)
        return 1
    try:
        payload = json.loads(raw)
        input_path, params = _find_input(payload)
        thresholds = QCThresholds(
            focus_min=float(params.get("focus_min", 100.0)),
            brightness_min=float(params.get("brightness_min", 25.0)),
            brightness_max=float(params.get("brightness_max", 230.0)),
            saturation_max_fraction=float(params.get("saturation_max_fraction", 0.02)),
            min_cell_area=int(params.get("min_cell_area", 15)),
            max_cell_area_frac=float(params.get("max_cell_area_frac", 0.25)),
            cell_count_low=int(params.get("cell_count_low", 1)),
            cell_count_high=(int(params["cell_count_high"]) if params.get("cell_count_high") is not None else None),
        )
        thresholds.validate()
        method = str(params.get("cell_method", "threshold"))
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        adaptive_qc = _parse_bool(params.get("adaptive_qc"), default=True)
        if path.is_dir():
            result = analyze_folder(
                str(path), thresholds=thresholds, cell_method=method, adaptive_qc=adaptive_qc
            )
        else:
            result = analyze_paths(
                [str(path)], thresholds=thresholds, cell_method=method, adaptive_qc=adaptive_qc
            )
        output = {
            "entity_id": payload.get("entity_id"),
            "input_path": str(path),
            "cell_method": method,
            "n_rows": int(len(result)) if hasattr(result, "__len__") else None,
            "results": _jsonable(result),
        }
        print(json.dumps(output, allow_nan=False, default=str))
        if hasattr(result, "columns") and "error" in result.columns and result["error"].notna().any():
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001 - adapter boundary
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

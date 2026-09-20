#!/usr/bin/env python3
"""Validate committed LIVECell benchmark artifacts without running inference."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "livecell_validation"
SPECS = {"cellpose": True, "hybrid": True, "auto": True, "threshold": False}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _load_csv(backend: str) -> list[dict[str, str]]:
    path = OUT / f"livecell_val_{backend}_n20.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise AssertionError(f"{path}: expected 20 rows, found {len(rows)}")
    return rows


def _load_json(backend: str) -> dict:
    path = OUT / f"livecell_val_{backend}_n20.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate() -> None:
    required = [
        OUT / "LIVECELL_CURRENT_N20_REPORT.md",
        *(OUT / f"livecell_val_{backend}_n20.{ext}" for backend in SPECS for ext in ("json", "csv")),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError("Missing committed LIVECell artifacts: " + ", ".join(missing))

    name_sets: dict[str, list[str]] = {}

    for backend, gpu_expected in SPECS.items():
        rows = _load_csv(backend)
        data = _load_json(backend)

        _require(data["dataset"] == "LIVECell", f"{backend}: incorrect dataset")
        _require(data["split"] == "val", f"{backend}: incorrect split")
        _require(data["n_requested"] == 20, f"{backend}: incorrect requested count")
        _require(data["n_scored"] == 20, f"{backend}: incorrect scored count")
        _require(data["n_missing_files"] == 0, f"{backend}: missing files recorded")
        _require(data["gpu"] is gpu_expected, f"{backend}: GPU flag mismatch")

        names = [row["file_name"] for row in rows]
        _require(len(set(names)) == 20, f"{backend}: duplicate FOV names")
        name_sets[backend] = names

        summary = data["summary"]
        expected = {
            "iou": _mean(rows, "iou"),
            "dice": _mean(rows, "dice"),
            "precision": _mean(rows, "precision"),
            "recall": _mean(rows, "recall"),
            "f1": _mean(rows, "f1"),
            "absolute_count_error": _mean(rows, "absolute_count_error"),
            "relative_count_error": _mean(rows, "relative_count_error"),
        }

        for key, value in expected.items():
            observed = float(summary[key])
            if not math.isclose(observed, value, rel_tol=1e-10, abs_tol=1e-10):
                raise AssertionError(f"{backend}: summary.{key}={observed} != CSV mean {value}")

        if "pixel_iou_mean" in summary:
            assert math.isclose(float(summary["pixel_iou_mean"]), expected["iou"], rel_tol=1e-10, abs_tol=1e-10)
        if "pixel_dice_mean" in summary:
            assert math.isclose(float(summary["pixel_dice_mean"]), expected["dice"], rel_tol=1e-10, abs_tol=1e-10)

    reference = name_sets["cellpose"]
    for backend, names in name_sets.items():
        _require(names == reference, f"{backend}: FOV ordering differs from Cellpose baseline")

    report = (OUT / "LIVECELL_CURRENT_N20_REPORT.md").read_text(encoding="utf-8")
    for token in ("Cellpose", "Hybrid", "Auto", "Threshold", "0.930", "0.054"):
        _require(token in report, f"current report missing expected token: {token}")

    print("LIVECell committed artifacts: OK")
    print("20 FOVs verified across:", ", ".join(SPECS))
    print("CSV/JSON summary consistency: OK")


if __name__ == "__main__":
    validate()

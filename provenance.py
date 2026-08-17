"""Reproducibility and provenance manifests for OptiCell analyses."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_input_manifest(paths: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        resolved = Path(path).resolve()
        record: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
        if resolved.is_file():
            stat = resolved.stat()
            record.update({
                "size_bytes": int(stat.st_size),
                "modified_ns": int(stat.st_mtime_ns),
                "sha256": file_sha256(str(resolved)),
            })
        records.append(record)
    return records


def build_manifest(
    *,
    opticell_version: str,
    inputs: Iterable[str] = (),
    parameters: Mapping[str, Any] | None = None,
    operation: str = "analysis",
) -> dict[str, Any]:
    """Create a portable analysis manifest with software/runtime/input provenance."""
    return {
        "manifest_version": "1.0",
        "operation": operation,
        "opticell_version": opticell_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "working_directory": os.getcwd(),
        "parameters": dict(parameters or {}),
        "inputs": collect_input_manifest(inputs),
    }


def write_manifest(manifest: Mapping[str, Any], path: str) -> str:
    """Write a human-readable JSON provenance file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(target)


__all__ = ["file_sha256", "collect_input_manifest", "build_manifest", "write_manifest"]

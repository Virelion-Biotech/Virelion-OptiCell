"""Reproducibility and provenance manifests for OptiCell analyses."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
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


def _git_commit() -> str | None:
    env_sha = os.getenv("GITHUB_SHA") or os.getenv("OPTICELL_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


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


def build_immutable_manifest(
    *,
    opticell_version: str,
    inputs: Iterable[str] = (),
    parameters: Mapping[str, Any] | None = None,
    operation: str = "analysis",
    models: Iterable[Mapping[str, Any]] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a content-addressed manifest with inputs, environment, models and parameters."""
    input_records = collect_input_manifest(inputs)
    model_records: list[dict[str, Any]] = []
    for item in models:
        record = dict(item)
        model_path = record.get("path")
        if model_path and Path(str(model_path)).is_file() and not record.get("sha256"):
            record["sha256"] = file_sha256(str(model_path))
        model_records.append(record)
    payload: dict[str, Any] = {
        "manifest_version": "2.0",
        "operation": operation,
        "opticell_version": opticell_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"git_commit": _git_commit()},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": _package_versions(
            ("opticell", "numpy", "pandas", "scipy", "opencv-python-headless", "tifffile", "streamlit", "cellpose", "torch")
        ),
        "working_directory": os.getcwd(),
        "parameters": dict(parameters or {}),
        "models": model_records,
        "inputs": input_records,
        "extra": dict(extra or {}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def verify_manifest(manifest: Mapping[str, Any]) -> bool:
    """Verify a v2 content-addressed manifest has not been modified."""
    expected = manifest.get("manifest_sha256")
    if not expected:
        return False
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return actual == expected


def write_manifest(manifest: Mapping[str, Any], path: str) -> str:
    """Write a human-readable JSON provenance file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(target)


__all__ = ["file_sha256", "collect_input_manifest", "build_manifest", "build_immutable_manifest", "verify_manifest", "write_manifest"]

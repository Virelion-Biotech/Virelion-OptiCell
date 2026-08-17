"""Reproducibility helpers for comparing OptiCell analyses."""
from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def analysis_fingerprint(parameters: Mapping[str, Any], *, input_hashes: Mapping[str, str] | None = None) -> str:
    """Return a SHA-256 fingerprint of analysis parameters and input hashes."""
    payload = {"parameters": dict(parameters), "input_hashes": dict(input_hashes or {})}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def environment_fingerprint() -> dict[str, str]:
    """Return stable runtime identifiers useful for provenance comparisons."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
    }


def compare_manifests(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two manifests without treating timestamps as scientific differences."""
    ref_inputs = reference.get("inputs", {})
    cand_inputs = candidate.get("inputs", {})
    ref_hashes = {str(k): str(v.get("sha256", v)) if isinstance(v, Mapping) else str(v) for k, v in ref_inputs.items()}
    cand_hashes = {str(k): str(v.get("sha256", v)) if isinstance(v, Mapping) else str(v) for k, v in cand_inputs.items()}
    changed = sorted(set(ref_hashes) | set(cand_hashes))
    changed = [key for key in changed if ref_hashes.get(key) != cand_hashes.get(key)]
    ref_params = reference.get("parameters", {})
    cand_params = candidate.get("parameters", {})
    parameter_keys = sorted(set(ref_params) | set(cand_params))
    parameter_changes = [key for key in parameter_keys if ref_params.get(key) != cand_params.get(key)]
    return {
        "inputs_match": not changed,
        "changed_inputs": changed,
        "parameters_match": not parameter_changes,
        "changed_parameters": parameter_changes,
        "environment_match": reference.get("environment") == candidate.get("environment"),
    }


__all__ = ["analysis_fingerprint", "canonical_json", "compare_manifests", "environment_fingerprint"]

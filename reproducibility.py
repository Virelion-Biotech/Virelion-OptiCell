"""Reproducibility helpers for comparing OptiCell analyses."""
from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
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


def _manifest_input_hashes(inputs: Any) -> tuple[dict[str, str], set[str]]:
    """Normalize input manifests and record entries whose SHA-256 is absent."""
    if inputs is None:
        return {}, set()
    if isinstance(inputs, Mapping):
        items = inputs.items()
    elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes, bytearray)):
        items = []
        for index, item in enumerate(inputs):
            if isinstance(item, Mapping):
                key = item.get("relative_path") or item.get("path") or item.get("filename") or str(index)
                value = item.get("sha256", "")
            else:
                key, value = str(index), item
            items.append((key, value))
    else:
        raise TypeError("manifest 'inputs' must be a mapping or sequence")

    hashes: dict[str, str] = {}
    missing_hashes: set[str] = set()
    for key, value in items:
        key_str = str(key)
        digest = value.get("sha256", "") if isinstance(value, Mapping) else value
        digest_str = str(digest) if digest is not None else ""
        hashes[key_str] = digest_str
        if not digest_str:
            missing_hashes.add(key_str)
    return hashes, missing_hashes


def _manifest_environment(manifest: Mapping[str, Any]) -> Any:
    """Read either the current runtime field or the legacy environment field."""
    return manifest.get("environment", manifest.get("runtime"))


def compare_manifests(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare manifests without treating missing evidence as a match."""
    ref_hashes, ref_missing = _manifest_input_hashes(reference.get("inputs", {}))
    cand_hashes, cand_missing = _manifest_input_hashes(candidate.get("inputs", {}))
    changed = sorted(set(ref_hashes) | set(cand_hashes))
    changed = [key for key in changed if ref_hashes.get(key) != cand_hashes.get(key)]
    unverifiable_inputs = sorted(ref_missing | cand_missing)

    ref_params = reference.get("parameters", {})
    cand_params = candidate.get("parameters", {})
    parameter_keys = sorted(set(ref_params) | set(cand_params))
    parameter_changes = [key for key in parameter_keys if ref_params.get(key) != cand_params.get(key)]

    ref_environment = _manifest_environment(reference)
    cand_environment = _manifest_environment(candidate)
    environment_evaluated = ref_environment is not None and cand_environment is not None
    environment_match = bool(ref_environment == cand_environment) if environment_evaluated else None
    return {
        "inputs_match": not changed and not unverifiable_inputs,
        "changed_inputs": changed,
        "unverifiable_inputs": unverifiable_inputs,
        "parameters_match": not parameter_changes,
        "changed_parameters": parameter_changes,
        "environment_match": environment_match,
        "environment_evaluated": environment_evaluated,
    }


__all__ = ["analysis_fingerprint", "canonical_json", "compare_manifests", "environment_fingerprint"]

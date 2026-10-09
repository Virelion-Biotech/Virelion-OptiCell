"""Fail-closed scientific-domain qualification; no installed backend is qualified by default."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path


def require_domain_qualification(*, domain, backend, report_path, report_sha256, expected_model_sha256=None):
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("Scientific segmentation requires an explicit acquisition domain")
    if backend == "auto":
        raise ValueError("Scientific segmentation requires an explicit qualified backend")
    if not report_path or not report_sha256:
        raise ValueError("Scientific segmentation requires a locked domain qualification report")
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != report_sha256:
        raise ValueError("Domain qualification report digest mismatch")
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get("domain") != domain or report.get("backend") != backend:
        raise ValueError("Qualification report does not match domain/backend")
    # Accept evidence only after all prespecified domain gates have passed.
    gates = report.get("gates")
    if not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
        raise ValueError("Domain qualification gates failed or are absent")
    for field in ["model_id", "model_sha256", "dataset_id", "dataset_version", "protocol_sha256"]:
        if not isinstance(report.get(field), str) or not report[field].strip():
            raise ValueError(f"Domain qualification missing {field}")
    for field in ["model_sha256", "protocol_sha256"]:
        if len(report[field]) != 64 or any(c not in "0123456789abcdef" for c in report[field]):
            raise ValueError(f"Invalid {field}")
    if not expected_model_sha256 or report["model_sha256"] != expected_model_sha256:
        raise ValueError("Qualification must match the executing model/code fingerprint")
    return report

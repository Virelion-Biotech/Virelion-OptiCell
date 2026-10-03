#!/usr/bin/env python3
"""Validate that the repository is internally consistent for the v2.18.0 release."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.18.0"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def validate() -> None:
    init = read("opticell/__init__.py")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init)
    require(match is not None, "opticell.__version__ not found")
    require(match.group(1) == VERSION, f"package version is {match.group(1)!r}, expected {VERSION!r}")

    pyproject = read("pyproject.toml")
    require('dynamic = ["version"]' in pyproject, "pyproject must use dynamic versioning")
    require("AGPL-3.0-or-later" in pyproject, "pyproject license missing")
    require("defusedxml>=0.7.1,<0.8" in pyproject, "defusedxml dependency missing")

    citation = read("CITATION.cff")
    require(f'version: "{VERSION}"' in citation, "CITATION.cff version mismatch")
    require('date-released: "2026-09-20"' in citation, "CITATION.cff release date mismatch")

    release = read("docs/RELEASE_NOTES_v2.18.0.md")
    require("# OptiCell v2.18.0 release notes" in release, "release notes heading missing")
    require("Release date: 2026-09-20" in release, "release notes date mismatch")
    require("LIVECell T4 n=20 measured results" in release, "LIVECell release evidence missing")

    changelog = read("CHANGELOG.md")
    require("## 2.18.0 — 2026-09-20" in changelog, "CHANGELOG v2.18.0 heading missing")

    roadmap = read("docs/PRODUCT_ROADMAP.md")
    require("| Cellpose (GPU) | 0.930 | 0.904 | 25.7 |" in roadmap, "roadmap Cellpose row missing")
    require("| Hybrid (GPU) | 0.930 | 0.904 | 25.7 |" in roadmap, "roadmap Hybrid row missing")
    require("| Auto (GPU) | 0.930 | 0.904 | 25.7 |" in roadmap, "roadmap Auto row missing")
    require("| Threshold (CPU) | 0.054 | 0.435 | 87.5 |" in roadmap, "roadmap threshold row missing")

    readme = read("README.md")
    require("Auto (GPU)" in readme and "0.930" in readme, "README missing current LIVECell measurements")

    report = read("outputs/livecell_validation/LIVECELL_CURRENT_N20_REPORT.md")
    for token in ("Cellpose-SAM (GPU)", "Hybrid (GPU)", "Auto (GPU)", "Threshold (CPU)", "0.930", "0.054"):
        require(token in report, f"LIVECell current report missing {token!r}")

    required = [
        "outputs/livecell_validation/livecell_val_cellpose_n20.json",
        "outputs/livecell_validation/livecell_val_cellpose_n20.csv",
        "outputs/livecell_validation/livecell_val_hybrid_n20.json",
        "outputs/livecell_validation/livecell_val_hybrid_n20.csv",
        "outputs/livecell_validation/livecell_val_auto_n20.json",
        "outputs/livecell_validation/livecell_val_auto_n20.csv",
        "outputs/livecell_validation/livecell_val_threshold_n20.json",
        "outputs/livecell_validation/livecell_val_threshold_n20.csv",
    ]
    for path in required:
        require((ROOT / path).is_file(), f"missing release artifact: {path}")

    require(not (ROOT / "livecell_validation_results.zip").exists(), "temporary LIVECell ZIP must not be committed")

    print(f"OptiCell v{VERSION} release metadata: OK")
    print("LIVECell evidence set: OK")
    print("Temporary ZIP absent: OK")


if __name__ == "__main__":
    validate()

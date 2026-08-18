import importlib

import opticell


def test_all_public_exports_resolve():
    missing = [name for name in opticell.__all__ if not hasattr(opticell, name)]
    assert missing == []


def test_public_package_wrappers_import():
    modules = [
        "acceptance",
        "artifact_quality",
        "batch",
        "benchmarking",
        "experiment_audit",
        "experiment_qc",
        "experiment_stats",
        "exports",
        "lineage",
        "lineage_events",
        "lineage_quality",
        "ome_io",
        "power",
        "profiling",
        "quality_gate",
        "reproducibility",
        "robustness",
        "runtime",
        "screening",
        "screening_advanced",
        "screening_qc",
        "segmentation",
        "sensitivity",
        "statistics",
        "stream_io",
        "tracking3d",
        "tracking_events",
        "tracking_validation",
        "volumetric_segmentation",
    ]
    for module in modules:
        imported = importlib.import_module(f"opticell.{module}")
        assert imported.__name__ == f"opticell.{module}"


def test_version_is_semver_like():
    parts = opticell.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)

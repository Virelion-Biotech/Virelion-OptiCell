import copy
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from opticell.calibration import calibration_curve, choose_threshold, threshold_performance
from opticell.dataset_qc import annotate_outliers, dataset_qc_summary, replicate_qc
from opticell.large_data import iter_ome_planes, plan_ome_iteration
from opticell.model_config import compare_configs, config_for_checkpoint, config_from_preset
from opticell.observability import health_snapshot
from opticell.validation_registry import coverage_summary, known_validation_evidence
from opticell.workbench_store import WorkbenchStore
from provenance import build_immutable_manifest, verify_manifest


def test_workbench_store_projects_runs_reviews_quota_and_backup(tmp_path):
    db = tmp_path / "workbench.sqlite3"
    store = WorkbenchStore(db)
    store.create_user("researcher", "long-password-123", quota_runs=1)
    assert store.authenticate("researcher", "long-password-123")
    assert not store.authenticate("researcher", "bad")
    project = store.create_project("researcher", "Study A", description="test")
    sample = store.add_sample(project, "sample-1", condition="control", replicate="R1", actor="researcher")
    run = store.save_run(
        project,
        owner="researcher",
        backend="threshold",
        config={"backend": "threshold"},
        manifest={"manifest_version": "2.0"},
        summary={"per_image": [{"filename": "a.png", "gate": "PASS"}]},
        sample_id=sample,
    )
    assert store.get_run(run)["summary"]["per_image"][0]["gate"] == "PASS"
    review = store.add_review(run, "a.png", "accept", reviewer="researcher", notes="looks good")
    assert store.list_reviews(run)[0]["review_id"] == review
    assert store.audit_events()
    try:
        store.save_run(project, owner="researcher", backend="threshold", config={}, manifest={}, summary={})
    except RuntimeError as exc:
        assert "quota" in str(exc)
    else:
        raise AssertionError("quota should prevent a second run")
    backup = tmp_path / "backup.sqlite3"
    assert Path(store.backup(backup)).is_file()
    assert WorkbenchStore(backup).get_project(project)["name"] == "Study A"


def test_model_configs_hash_checkpoints_and_compare(tmp_path):
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"model-weights")
    custom = config_for_checkpoint(checkpoint, gpu=False)
    assert custom.checkpoint_sha256
    threshold = config_from_preset("cpu-threshold")
    assert "backend" in compare_configs(custom, threshold)


def test_calibration_requires_observed_labels_and_selects_explicit_threshold():
    scores = [10, 20, 40, 60, 80, 90, 95, 99]
    outcomes = [0, 0, 0, 0, 1, 1, 1, 1]
    curve = calibration_curve(scores, outcomes, bins=4)
    table = threshold_performance(scores, outcomes)
    choice = choose_threshold(scores, outcomes, minimum_sensitivity=0.75, minimum_specificity=0.75)
    assert not curve.empty
    assert {"sensitivity", "specificity", "precision"}.issubset(table.columns)
    assert 0 <= choice["threshold"] <= 100


def test_dataset_qc_outliers_and_replicates():
    frame = pd.DataFrame({
        "condition": ["c", "c", "c", "t", "t", "t"],
        "replicate": ["1", "1", "2", "1", "1", "2"],
        "objects": [10, 11, 9, 12, 13, 120],
        "quality": [90, 91, 89, 88, 87, 20],
    })
    annotated = annotate_outliers(frame, ["objects", "quality"], z_threshold=2.0)
    assert annotated["dataset_qc_outlier"].any()
    summary = dataset_qc_summary(frame, ["objects", "quality"])
    assert set(summary["metric"]) == {"objects", "quality"}
    reps = replicate_qc(frame, metrics=["objects", "quality"])
    assert "objects_mean" in reps


def test_v2_manifest_is_content_addressed(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"abc")
    manifest = build_immutable_manifest(
        opticell_version="test",
        inputs=[str(source)],
        parameters={"backend": "threshold"},
        models=[{"name": "threshold"}],
    )
    assert verify_manifest(manifest)
    changed = copy.deepcopy(manifest)
    changed["parameters"]["backend"] = "cellpose"
    assert not verify_manifest(changed)


def test_large_tiff_plane_iteration_is_bounded(tmp_path):
    path = tmp_path / "stack.tif"
    array = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    tifffile.imwrite(path, array, photometric="minisblack", metadata={"axes": "ZYX"})
    plan = plan_ome_iteration(str(path))
    assert plan.plane_shape == (8, 9)
    planes = list(iter_ome_planes(str(path)))
    assert len(planes) == 3
    assert np.array_equal(planes[1][1], array[1])


def test_validation_registry_does_not_invent_independent_lab_evidence():
    coverage = coverage_summary(known_validation_evidence())
    assert coverage["n_evidence_sets"] >= 3
    assert coverage["n_independent_lab_sets"] == 0
    assert coverage["has_independent_lab_validation"] is False


def test_health_snapshot_reports_workspace(tmp_path):
    health = health_snapshot(str(tmp_path))
    assert health["workspace_writable"]
    assert "runtime" in health

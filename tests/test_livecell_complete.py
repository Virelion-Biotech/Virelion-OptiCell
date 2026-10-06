"""End-to-end coverage and stale-image cache rejection for checkpointed studies."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def test_checkpointed_complete_runner_covers_splits_and_invalidates_changed_image(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data, output = tmp_path / "data", tmp_path / "out"
    (data / "raw").mkdir(parents=True)
    (data / "images").mkdir()
    image = np.zeros((32, 32), np.uint8)
    image[8:24, 8:24] = 200
    path = data / "images" / "test_cell.png"
    cv2.imwrite(str(path), image)
    coco = dict(
        images=[dict(id=i, file_name=path.name, height=32, width=32) for i in [42, 43]],
        annotations=[
            dict(id=i, image_id=i, segmentation=[[8.0, 8.0, 24.0, 8.0, 24.0, 24.0, 8.0, 24.0]]) for i in [42, 43]
        ],
        categories=[],
    )
    for split in ["val", "test"]:
        (data / "raw" / f"livecell_coco_{split}.json").write_text(json.dumps(coco))
    command = [
        sys.executable,
        str(root / "scripts/run_livecell_complete.py"),
        "--data-dir",
        str(data),
        "--out-dir",
        str(output),
    ]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run(command, check=True, capture_output=True, env=env, timeout=120)
    result = output / "val/livecell_val_threshold_n2.json"
    payload = json.loads(result.read_text())
    assert payload["complete"] and payload["coverage_verified_against_full_official_split"]
    assert payload["n_scored"] == 2
    assert payload["n_unique_source_images"] == 1
    assert {r["image_id"] for r in payload["per_image"]} == {42, 43}
    assert (output / "test/livecell_test_threshold_n2.json").exists()
    manifest = output / "_chunks/val_0000/manifest.json"
    prior = json.loads(manifest.read_text())
    cv2.imwrite(str(path), np.zeros_like(image))
    subprocess.run(command, check=True, capture_output=True, env=env, timeout=120)
    assert prior["image_sha256"] != json.loads(manifest.read_text())["image_sha256"]
    assert json.loads(result.read_text())["per_image"][0]["pred_count"] == 0

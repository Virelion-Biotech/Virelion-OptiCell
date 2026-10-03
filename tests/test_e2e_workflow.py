"""End-to-end CLI tests for the production workflow."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_synthetic(path: Path, shift: int = 0, blur: bool = False) -> None:
    image = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(image, (35 + shift, 45), 10, 220, -1)
    cv2.circle(image, (88 + shift, 82), 12, 220, -1)
    cv2.rectangle(image, (55 + shift, 20), (72 + shift, 38), 220, -1)
    if blur:
        image = cv2.GaussianBlur(image, (21, 21), 0)
    assert cv2.imwrite(str(path), image)


def test_killer_workflow_runs_end_to_end_and_writes_review_queue(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    _write_synthetic(images / "frame_01.png")
    _write_synthetic(images / "frame_02.png", shift=1, blur=True)

    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_killer_workflow.py",
            str(images),
            "--backend",
            "threshold",
            "--write-review-queue",
            "-o",
            str(out),
        ],
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "workflow_summary.json").is_file()
    assert (out / "workflow_summary.csv").is_file()
    assert (out / "cell_features_phenotype.csv").is_file()
    assert (out / "phenotype_summary.csv").is_file()
    assert (out / "review_queue.csv").is_file()
    assert list((out / "masks").glob("*_labels.png"))


def test_killer_workflow_empty_directory_fails_cleanly(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_killer_workflow.py",
            str(empty),
            "--backend",
            "threshold",
            "-o",
            str(out),
        ],
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 3
    assert "no images found" in proc.stderr.lower()


def test_killer_workflow_review_decisions_round_trip(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "frame_01.png"
    _write_synthetic(image_path, blur=True)

    first = tmp_path / "first"
    p1 = subprocess.run(
        [
            sys.executable,
            "scripts/run_killer_workflow.py",
            str(images),
            "--backend",
            "threshold",
            "--write-review-queue",
            "-o",
            str(first),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert p1.returncode == 0, p1.stdout + p1.stderr
    assert (first / "review_queue.csv").is_file()

    reviewed = tmp_path / "reviewed.csv"
    reviewed.write_text(
        "image,decision,reviewer,notes\n"
        + f"{image_path},accept,tester,synthetic smoke review\n",
        encoding="utf-8",
    )
    second = tmp_path / "second"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_killer_workflow.py",
            str(images),
            "--backend",
            "threshold",
            "--review-decisions",
            str(reviewed),
            "-o",
            str(second),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = (second / "review_queue.csv").read_text(encoding="utf-8")
    assert "COMPLETED" in rows
    assert "accept" in rows

"""Unit tests for the HeartTwin local-command adapter."""
from __future__ import annotations

import io
import json
import re

import cv2
import numpy as np
import pandas as pd
import pytest

from opticell import hearttwin_adapter as adapter


def _image(path, value: int = 180) -> None:
    image = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(image, (24, 24), 7, value, -1)
    assert cv2.imwrite(str(path), image)


def _payload(input_path, **values) -> dict:
    return {
        "entity_id": "patient-1",
        "observations": [
            {"modality": "ecg", "values": {"input_path": "ignored"}},
            {"modality": "imaging", "values": {"input_path": str(input_path), **values}},
        ],
    }


def _run(monkeypatch, capsys, payload, *, stdin: bool = False):
    raw = json.dumps(payload)
    if stdin:
        monkeypatch.setenv("HEARTTWIN_PAYLOAD_STDIN", "1")
        monkeypatch.delenv("HEARTTWIN_PAYLOAD", raising=False)
        monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    else:
        monkeypatch.delenv("HEARTTWIN_PAYLOAD_STDIN", raising=False)
        monkeypatch.setenv("HEARTTWIN_PAYLOAD", raw)
    code = adapter.main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_find_input_picks_imaging_observation():
    path, values = adapter._find_input(_payload("img.png", focus_min=5))
    assert path == "img.png" and values["focus_min"] == 5
    with pytest.raises(ValueError, match="imaging"):
        adapter._find_input({"observations": [{"modality": "ecg", "values": {}}]})


@pytest.mark.parametrize(
    "value, expected",
    [(None, True), (False, False), (1, True), (np.int64(0), False), (" Yes ", True), ("n", False)],
)
def test_parse_bool_accepts_known_values(value, expected):
    assert adapter._parse_bool(value, default=True) is expected


@pytest.mark.parametrize("value", [2, "maybe", 1.0])
def test_parse_bool_rejects_unknown_values(value):
    with pytest.raises(ValueError):
        adapter._parse_bool(value, default=False)


def test_parse_float_and_int():
    assert adapter._parse_float({}, "k", 1.5) == 1.5
    assert adapter._parse_float({"k": "2"}, "k", 0.0) == 2.0
    for bad in (True, float("inf")):
        with pytest.raises(ValueError):
            adapter._parse_float({"k": bad}, "k", 0.0)
    assert adapter._parse_int({}, "k", 3) == 3
    assert adapter._parse_int({"k": None}, "k", 3) is None
    assert adapter._parse_int({"k": " 4.0 "}, "k", 0) == 4
    assert adapter._parse_int({"k": np.int32(7)}, "k", 0) == 7
    for bad in (True, "", "abc", "4.5", 4.5):
        with pytest.raises(ValueError):
            adapter._parse_int({"k": bad}, "k", 0)


def test_jsonable_handles_frames_numpy_and_non_finite():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]})
    assert adapter._jsonable(frame) == [{"a": 1.0, "b": "x"}, {"a": None, "b": None}]
    assert adapter._jsonable({1: (np.float32(2.5), float("nan"), pd.NA)}) == {"1": [2.5, None, None]}
    assert adapter._jsonable(pd.Series({"k": np.int64(3)})) == {"k": 3}
    assert adapter._jsonable("plain") == "plain"


def test_main_requires_payload(monkeypatch, capsys):
    monkeypatch.delenv("HEARTTWIN_PAYLOAD", raising=False)
    monkeypatch.delenv("HEARTTWIN_PAYLOAD_STDIN", raising=False)
    assert adapter.main() == 1
    assert "not set" in capsys.readouterr().err


@pytest.mark.parametrize("stdin", [False, True])
def test_main_analyzes_single_image(tmp_path, monkeypatch, capsys, stdin):
    image = tmp_path / "cell.png"
    _image(image)
    code, out, err = _run(monkeypatch, capsys, _payload(image, adaptive_qc="false", focus_min=0), stdin=stdin)
    result = json.loads(out)
    assert code == 0, err
    assert result["entity_id"] == "patient-1"
    assert result["cell_method"] == "threshold"
    assert result["scientific_mode"] is False
    assert result["scientific_claim"] == "exploratory image QC"
    assert result["optogenetic_stimulation"] is None
    assert result["n_rows"] == 1 and len(result["results"]) == 1


def test_main_analyzes_folder(tmp_path, monkeypatch, capsys):
    for name in ("a.png", "b.png"):
        _image(tmp_path / name)
    code, out, err = _run(monkeypatch, capsys, _payload(tmp_path, adaptive_qc=False, focus_min=0))
    assert code == 0, err
    assert json.loads(out)["n_rows"] == 2


@pytest.mark.parametrize(
    "values, message",
    [
        ({"cell_method": "magic"}, "cell_method"),
        ({"scientific_mode": True, "cell_method": "cellpose", "domain": "x"}, "weight binding"),
        ({"scientific_mode": True}, "acquisition domain"),
        ({"min_cell_area": 1.5}, "lossy"),
        ({"adaptive_qc": "sometimes"}, "boolean"),
    ],
)
def test_main_reports_parameter_errors(tmp_path, monkeypatch, capsys, values, message):
    image = tmp_path / "cell.png"
    _image(image)
    code, out, err = _run(monkeypatch, capsys, _payload(image, **values))
    assert code == 1 and out == ""
    assert re.search(message, err)


def test_main_reports_missing_input(tmp_path, monkeypatch, capsys):
    code, _, err = _run(monkeypatch, capsys, _payload(tmp_path / "missing.png"))
    assert code == 1 and "does not exist" in err


def test_main_rejects_invalid_json(monkeypatch, capsys):
    monkeypatch.delenv("HEARTTWIN_PAYLOAD_STDIN", raising=False)
    monkeypatch.setenv("HEARTTWIN_PAYLOAD", "{not json")
    assert adapter.main() == 1
    assert capsys.readouterr().err

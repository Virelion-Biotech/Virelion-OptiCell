import pytest

from scripts.run_stage3_orchestrate import select_backend


def _row(backend, *, cv=0.2, mean_count=10.0, zero_frac=0.0):
    return {
        "backend": backend,
        "mean_confidence": 100.0,
        "count_cv": cv,
        "zero_object_fraction": zero_frac,
        "mean_object_count": mean_count,
        "n_images": 10,
        "n_requested_images": 10,
        "n_successful_images": 10,
        "n_failed_images": 0,
        "complete": True,
        "n_low_confidence_lt_50": 0,
        "error": None,
    }


def test_rejected_backend_is_not_reintroduced():
    rows = [_row("cellpose", cv=2.0), _row("threshold", cv=0.2)]
    decision = select_backend(rows, cv_reject=1.0)
    assert decision["selected_backend"] == "threshold"
    assert {item["backend"] for item in decision["rejected"]} == {"cellpose"}


def test_no_viable_backend_is_a_hard_failure():
    rows = [_row("cellpose", cv=2.0), _row("threshold", cv=1.5)]
    with pytest.raises(ValueError, match="no viable backends"):
        select_backend(rows, cv_reject=1.0)


def test_incomplete_backend_is_rejected():
    rows = [_row("cellpose"), _row("threshold")]
    rows[0]["complete"] = False
    rows[0]["n_failed_images"] = 1
    rows[0]["n_successful_images"] = 9
    decision = select_backend(rows)
    assert decision["selected_backend"] == "threshold"
    assert decision["rejected"] == [{"backend": "cellpose", "reason": "incomplete_run"}]

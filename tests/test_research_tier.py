import numpy as np
import pandas as pd

from benchmarking import aggregate_backend_benchmarks, benchmark_backends
from experiment_stats import bootstrap_ci, replicate_effect_summary, summarize_experiment
from runtime import capabilities, preferred_accelerator
from tracking_events import classify_divisions, detect_time_series_events, detect_transition_events
from validation import paired_segmentation_metrics


class DummyResult:
    def __init__(self, labels):
        self.labels = labels


class DummyBackend:
    def __init__(self, labels):
        self.labels = labels
    def segment(self, image):
        return DummyResult(self.labels)


def test_backend_benchmark_and_aggregation():
    truth = np.zeros((20, 20), dtype=np.int32); truth[2:6, 2:6] = 1
    image = truth.astype(np.uint8) * 255
    table = benchmark_backends(image, truth, {"good": DummyBackend(truth), "bad": DummyBackend(np.zeros_like(truth))})
    assert set(table["backend"]) == {"good", "bad"}
    assert table.loc[table["backend"] == "good", "f1"].iloc[0] == 1.0
    aggregate = aggregate_backend_benchmarks([table, table])
    assert "elapsed_seconds_mean" in aggregate.columns
    assert int(aggregate.loc[aggregate["backend"] == "bad", "failed_runs"].iloc[0]) == 0


def test_validation_metrics_are_explicit():
    truth = np.zeros((10, 10), dtype=np.int32); truth[1:4, 1:4] = 1
    metrics = paired_segmentation_metrics(truth, truth)
    assert metrics["iou"] == 1.0 and metrics["f1"] == 1.0


def test_event_detection_finds_split_merge_appearance_disappearance():
    prev = np.zeros((12, 12), dtype=np.int32); prev[2:8, 2:8] = 1
    curr = np.zeros_like(prev); curr[2:5, 2:8] = 2; curr[5:8, 2:8] = 3
    split = detect_transition_events(prev, curr, frame=1, min_overlap=0.2)
    assert "split" in set(split["event"])
    assert bool(classify_divisions(split).iloc[0]["is_division"])
    later = np.zeros_like(curr); later[2:8, 2:8] = 4
    merge = detect_transition_events(curr, later, frame=2, min_overlap=0.2)
    assert "merge" in set(merge["event"])
    sequence = detect_time_series_events([prev, curr, later])
    assert set(sequence["event"]) >= {"split", "merge"}


def test_experiment_statistics_are_replicate_level():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "replicate": np.repeat([1, 2, 3, 4], 5),
        "condition": np.repeat(["control", "control", "MI", "MI"], 5),
        "value": np.concatenate([rng.normal(10, 1, 10), rng.normal(15, 1, 10)]),
    })
    low, high = bootstrap_ci(frame["value"], n_bootstrap=500, seed=1)
    assert low < high
    summary = summarize_experiment(frame, replicate_column="replicate", group_column="condition", value_columns=["value"])
    assert set(summary["condition"]) == {"control", "MI"}
    effect = replicate_effect_summary(frame, "value", "condition", "control", "MI", n_bootstrap=500)
    assert effect["n_a"] == 2 and effect["n_b"] == 2


def test_runtime_capabilities_are_safe():
    info = capabilities()
    assert info.cpu_count >= 1
    assert preferred_accelerator() in {"cpu", "cuda"}

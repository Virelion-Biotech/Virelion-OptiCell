"""Tests for ensemble.hybrid_threshold_cellpose threshold-collapse handling.

Regression for LIVECell validation: original count-gated hybrid fell back to
threshold when counts disagreed sharply. Backwards once threshold collapsed
(fg_frac < 0.005 or zero objects). See ensemble.hybrid_threshold_cellpose.
"""
import numpy as np

from ensemble import hybrid_threshold_cellpose
from qc_pipeline import SegmentationResult, segment_threshold


class _FakeCellposeSegmenter:
    """Returns a pre-built result; no real Cellpose install required."""

    def __init__(self, result: SegmentationResult):
        self._result = result

    def segment(self, gray, min_area=15, max_area_frac=0.25):
        return self._result


def _labels_with_n_blobs(shape, n, radius=4, seed=0):
    labels = np.zeros(shape, dtype=np.int32)
    h, w = shape
    rng = np.random.default_rng(seed)
    next_id = 1
    tries = 0
    while next_id <= n and tries < 4000:
        tries += 1
        cy = rng.integers(radius + 2, h - radius - 2)
        cx = rng.integers(radius + 2, w - radius - 2)
        yy, xx = np.ogrid[:h, :w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
        if (labels[blob] != 0).any():
            continue
        labels[blob] = next_id
        next_id += 1
    return labels


def _fake_cellpose_result(shape, n_objects, seed=0):
    labels = _labels_with_n_blobs(shape, n_objects, seed=seed)
    fg = labels > 0
    return SegmentationResult(
        count=int(labels.max()),
        labels=labels,
        method="cellpose:fake",
        foreground_fraction=float(fg.mean()),
        median_area=30.0,
        area_cv=0.2,
        border_fraction=0.0,
        tiny_object_fraction=0.0,
        merged_object_fraction=0.0,
        quality_score=90.0,
        error=None,
    )


def _low_contrast_image(shape=(96, 96), seed=1):
    """Objects with <8 gray levels contrast — segment_threshold collapses."""
    rng = np.random.default_rng(seed)
    img = rng.normal(120, 2, shape).clip(0, 255).astype(np.uint8)
    h, w = shape
    for _ in range(8):
        cy, cx = rng.integers(15, h - 15), rng.integers(15, w - 15)
        yy, xx = np.ogrid[:h, :w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= 25
        img[blob] = np.clip(img[blob].astype(int) + 4, 0, 255).astype(np.uint8)
    return img


def _high_contrast_image(shape=(96, 96), seed=2):
    """Easily thresholdable blobs — threshold must NOT collapse."""
    rng = np.random.default_rng(seed)
    img = rng.normal(30, 5, shape).clip(0, 255).astype(np.uint8)
    occupied = np.zeros(shape, dtype=bool)
    h, w = shape
    placed, tries = 0, 0
    while placed < 6 and tries < 300:
        tries += 1
        cy, cx = rng.integers(15, h - 15), rng.integers(15, w - 15)
        yy, xx = np.ogrid[:h, :w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= 36
        if (blob & occupied).any():
            continue
        img[blob] = 220
        occupied |= blob
        placed += 1
    return img


def test_hybrid_prefers_cellpose_when_threshold_collapses():
    img = _low_contrast_image()
    fake_cp_result = _fake_cellpose_result(img.shape, n_objects=40, seed=5)
    result = hybrid_threshold_cellpose(
        img, cellpose_segmenter=_FakeCellposeSegmenter(fake_cp_result)
    )
    assert result.count == 40
    assert "threshold_collapsed" in result.method
    assert np.array_equal(result.labels, fake_cp_result.labels)


def test_hybrid_still_prefers_threshold_when_not_collapsed_and_counts_disagree():
    img = _high_contrast_image()
    fake_cp_result = _fake_cellpose_result(img.shape, n_objects=200, seed=6)
    result = hybrid_threshold_cellpose(
        img, cellpose_segmenter=_FakeCellposeSegmenter(fake_cp_result)
    )
    assert result.method.startswith("hybrid:threshold(")
    assert result.count != 200


def test_hybrid_prefers_cellpose_when_counts_agree_normally():
    img = _high_contrast_image()
    thr = segment_threshold(img)
    fake_cp_result = _fake_cellpose_result(img.shape, n_objects=max(thr.count, 1), seed=7)
    result = hybrid_threshold_cellpose(
        img, cellpose_segmenter=_FakeCellposeSegmenter(fake_cp_result)
    )
    assert result.method.startswith("hybrid:cellpose(")


def test_hybrid_falls_back_to_threshold_only_without_a_segmenter():
    img = _high_contrast_image()
    result = hybrid_threshold_cellpose(img, cellpose_segmenter=None)
    assert result.method == "hybrid:threshold_only"

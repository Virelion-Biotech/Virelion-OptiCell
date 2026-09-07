"""Segmentation ensemble and hybrid backends for disagreement-aware analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qc_pipeline import SegmentationResult, segment_threshold, CellposeSegmenter, _build_segmentation_result


@dataclass(frozen=True)
class EnsembleResult:
    labels: np.ndarray
    method: str
    member_counts: dict[str, int]
    count_median: float
    count_mad: float
    agreement_fraction: float
    warning: str | None = None


def _iou_by_mask(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a > 0, b > 0
    union = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / union) if union else 1.0


def threshold_ensemble(
    gray: np.ndarray, min_area: int = 15, max_area_frac: float = 0.25
) -> EnsembleResult:
    """Run Otsu and adaptive thresholding and expose their disagreement."""
    otsu = segment_threshold(gray, min_area=min_area, max_area_frac=max_area_frac, adaptive=False)
    adaptive = segment_threshold(gray, min_area=min_area, max_area_frac=max_area_frac, adaptive=True)
    members = {"otsu": otsu.count, "adaptive": adaptive.count}
    counts = np.asarray(list(members.values()), dtype=float)
    median = float(np.median(counts))
    mad = float(np.median(np.abs(counts - median)))
    iou = _iou_by_mask(otsu.labels, adaptive.labels)
    chosen = otsu if abs(otsu.count - median) <= abs(adaptive.count - median) else adaptive
    warning = None if iou >= 0.6 else "SEGMENTATION_DISAGREEMENT"
    return EnsembleResult(
        chosen.labels, "threshold_ensemble", members, median, mad, iou, warning
    )


def ensemble_from_results(
    results: list[SegmentationResult],
    names: list[str] | None = None,
    min_agreement: float = 0.6,
) -> EnsembleResult:
    """Fuse precomputed masks by selecting the member closest to consensus count."""
    if not results:
        raise ValueError("at least one segmentation result is required")
    if names and len(names) != len(results):
        raise ValueError("names length must match results")
    labels = [r.labels for r in results]
    shapes = {x.shape for x in labels}
    if len(shapes) != 1:
        raise ValueError("all masks must have identical shapes")
    names = names or [f"model_{i+1}" for i in range(len(results))]
    counts = np.asarray([r.count for r in results], dtype=float)
    median = float(np.median(counts))
    mad = float(np.median(np.abs(counts - median)))
    pairwise = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairwise.append(_iou_by_mask(labels[i], labels[j]))
    agreement = float(np.mean(pairwise)) if pairwise else 1.0
    idx = int(np.argmin(np.abs(counts - median)))
    warning = None if agreement >= min_agreement else "SEGMENTATION_DISAGREEMENT"
    return EnsembleResult(
        labels[idx],
        "ensemble_consensus",
        dict(zip(names, counts.astype(int))),
        median,
        mad,
        agreement,
        warning,
    )


def hybrid_threshold_cellpose(
    gray: np.ndarray,
    cellpose_segmenter: Optional[CellposeSegmenter] = None,
    min_area: int = 15,
    max_area_frac: float = 0.25,
    count_tol_frac: float = 0.12,
    count_tol_abs: int = 8,
) -> SegmentationResult:
    """Hybrid backend measured to balance Cellpose boundaries with threshold counts.

    Measured on BBBC039 n=50 (2026-09):
      - threshold: Dice≈0.95, instance F1≈0.96, |count err|≈6
      - cellpose cpsam: Dice≈0.97, instance F1≈0.91, |count err|≈15

    Rule (count-safe):
      - Run both backends.
      - If Cellpose count is within tol of threshold count → prefer **Cellpose**
        (better pixel overlap when instance topology agrees).
      - Otherwise prefer **threshold** (more reliable object counts on this modality).

    Always attaches diagnostics on the chosen SegmentationResult.method string.
    """
    thr = segment_threshold(
        gray, min_area=min_area, max_area_frac=max_area_frac, adaptive=False
    )

    if cellpose_segmenter is None:
        # classical-only fallback
        return SegmentationResult(
            count=thr.count,
            labels=thr.labels,
            method="hybrid:threshold_only",
            foreground_fraction=thr.foreground_fraction,
            median_area=thr.median_area,
            area_cv=thr.area_cv,
            border_fraction=thr.border_fraction,
            tiny_object_fraction=thr.tiny_object_fraction,
            merged_object_fraction=thr.merged_object_fraction,
            quality_score=thr.quality_score,
            error=thr.error,
        )

    cp = cellpose_segmenter.segment(
        gray, min_area=min_area, max_area_frac=max_area_frac
    )
    tol = max(count_tol_abs, int(round(count_tol_frac * max(thr.count, 1))))
    count_delta = abs(int(cp.count) - int(thr.count))
    agree_iou = _iou_by_mask(cp.labels, thr.labels)

    if count_delta <= tol:
        chosen = cp
        tag = f"hybrid:cellpose(delta={count_delta},iou={agree_iou:.2f})"
    else:
        chosen = thr
        tag = f"hybrid:threshold(delta={count_delta},iou={agree_iou:.2f})"

    return SegmentationResult(
        count=chosen.count,
        labels=chosen.labels,
        method=tag,
        foreground_fraction=chosen.foreground_fraction,
        median_area=chosen.median_area,
        area_cv=chosen.area_cv,
        border_fraction=chosen.border_fraction,
        tiny_object_fraction=chosen.tiny_object_fraction,
        merged_object_fraction=chosen.merged_object_fraction,
        quality_score=chosen.quality_score,
        error=chosen.error,
    )

"""Segmentation ensemble and hybrid backends for disagreement-aware analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qc_pipeline import CellposeSegmenter, SegmentationResult, segment_threshold


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


def _positive_label_count(labels: np.ndarray) -> int:
    """Count distinct foreground labels; label IDs need not be contiguous."""
    positive = np.asarray(labels)[np.asarray(labels) > 0]
    return int(np.unique(positive).size) if positive.size else 0


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


def fov_confidence(
    gray: np.ndarray,
    labels: np.ndarray,
    *,
    focus_score: float | None = None,
    agreement_iou: float | None = None,
    count_delta: int | None = None,
) -> dict[str, float | str]:
    """Cheap per-FOV confidence features for orchestration / QC dashboards.

    Not a calibrated probability. Higher `score` (0–100) means fewer red flags.
    """
    import cv2

    if focus_score is None:
        focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    fg = labels > 0
    fg_frac = float(fg.mean()) if labels.size else 0.0
    n = _positive_label_count(labels)

    score = 100.0
    flags: list[str] = []
    if focus_score < 50:
        score -= 25
        flags.append("LOW_FOCUS")
    if fg_frac < 0.005:
        score -= 20
        flags.append("SPARSE_FG")
    if fg_frac > 0.55:
        score -= 15
        flags.append("DENSE_FG")
    if n == 0:
        score -= 30
        flags.append("ZERO_OBJECTS")
    if agreement_iou is not None and agreement_iou < 0.5:
        score -= 20
        flags.append("BACKEND_DISAGREE")
    if count_delta is not None and count_delta > 20:
        score -= 15
        flags.append("COUNT_DISAGREE")

    return {
        "confidence_score": float(np.clip(score, 0, 100)),
        "focus_score": float(focus_score),
        "foreground_fraction": fg_frac,
        "object_count": float(n),
        "agreement_iou": float(agreement_iou) if agreement_iou is not None else float("nan"),
        "count_delta": float(count_delta) if count_delta is not None else float("nan"),
        "flags": ";".join(flags) if flags else "",
    }


def hybrid_threshold_cellpose(
    gray: np.ndarray,
    cellpose_segmenter: Optional[CellposeSegmenter] = None,
    min_area: int = 15,
    max_area_frac: float = 0.25,
    count_tol_frac: float = 0.12,
    count_tol_abs: int = 8,
) -> SegmentationResult:
    """Count-gated hybrid: Cellpose when counts agree with threshold, else threshold.

    Measured BBBC039 n=50: threshold F1≈0.96 / count≈6; cellpose Dice≈0.97 / count≈15.
    n=200: hybrid Dice 0.929, F1 0.924, |count|≈9 — count-safe default, not dual-axis SOTA.
    """
    thr = segment_threshold(
        gray, min_area=min_area, max_area_frac=max_area_frac, adaptive=False
    )

    if cellpose_segmenter is None:
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

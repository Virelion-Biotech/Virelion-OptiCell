"""Segmentation ensemble utilities for disagreement-aware analysis."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from qc_pipeline import SegmentationResult, segment_threshold

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

def threshold_ensemble(gray: np.ndarray, min_area: int = 15, max_area_frac: float = 0.25) -> EnsembleResult:
    """Run Otsu and adaptive thresholding and expose their disagreement."""
    otsu = segment_threshold(gray, min_area=min_area, max_area_frac=max_area_frac, adaptive=False)
    adaptive = segment_threshold(gray, min_area=min_area, max_area_frac=max_area_frac, adaptive=True)
    members = {"otsu": otsu.count, "adaptive": adaptive.count}
    counts = np.asarray(list(members.values()), dtype=float)
    median = float(np.median(counts)); mad = float(np.median(np.abs(counts - median)))
    iou = _iou_by_mask(otsu.labels, adaptive.labels)
    chosen = otsu if abs(otsu.count - median) <= abs(adaptive.count - median) else adaptive
    warning = None if iou >= 0.6 else "SEGMENTATION_DISAGREEMENT"
    return EnsembleResult(chosen.labels, "threshold_ensemble", members, median, mad, iou, warning)

def ensemble_from_results(results: list[SegmentationResult], names: list[str] | None = None, min_agreement: float = 0.6) -> EnsembleResult:
    """Fuse precomputed masks by selecting the member closest to consensus count."""
    if not results: raise ValueError("at least one segmentation result is required")
    if names and len(names) != len(results): raise ValueError("names length must match results")
    labels = [r.labels for r in results]
    shapes = {x.shape for x in labels}
    if len(shapes) != 1: raise ValueError("all masks must have identical shapes")
    names = names or [f"model_{i+1}" for i in range(len(results))]
    counts = np.asarray([r.count for r in results], dtype=float)
    median = float(np.median(counts)); mad = float(np.median(np.abs(counts - median)))
    pairwise = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)): pairwise.append(_iou_by_mask(labels[i], labels[j]))
    agreement = float(np.mean(pairwise)) if pairwise else 1.0
    idx = int(np.argmin(np.abs(counts - median)))
    warning = None if agreement >= min_agreement else "SEGMENTATION_DISAGREEMENT"
    return EnsembleResult(labels[idx], "ensemble_consensus", dict(zip(names, counts.astype(int))), median, mad, agreement, warning)

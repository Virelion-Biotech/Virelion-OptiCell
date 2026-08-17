"""Stable, model-agnostic segmentation API for OptiCell."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from qc_pipeline import CellposeSegmenter, SegmentationResult, segment_threshold
from ensemble import EnsembleResult, ensemble_from_results, threshold_ensemble


class SegmenterBackend(Protocol):
    """Minimal runtime protocol implemented by segmentation backends."""

    name: str

    def segment(self, image: np.ndarray, **kwargs) -> SegmentationResult:
        ...


class BaseSegmenter(ABC):
    """Convenience base class for third-party OptiCell segmentation plugins."""

    name: str = "base"

    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> SegmentationResult:
        raise NotImplementedError


class ThresholdSegmenter(BaseSegmenter):
    name = "threshold"

    def __init__(self, adaptive: bool = False, min_area: int = 15, max_area_frac: float = 0.25) -> None:
        self.adaptive = bool(adaptive)
        self.min_area = int(min_area)
        self.max_area_frac = float(max_area_frac)

    def segment(self, image: np.ndarray, **kwargs) -> SegmentationResult:
        return segment_threshold(
            image,
            min_area=kwargs.get("min_area", self.min_area),
            max_area_frac=kwargs.get("max_area_frac", self.max_area_frac),
            adaptive=kwargs.get("adaptive", self.adaptive),
        )


class CellposeBackend(BaseSegmenter):
    """Persistent Cellpose backend exposed through the common interface."""

    def __init__(self, model_type: str = "cyto3") -> None:
        self._segmenter = CellposeSegmenter(model_type=model_type)
        self.name = f"cellpose:{model_type}"

    def segment(self, image: np.ndarray, **kwargs) -> SegmentationResult:
        return self._segmenter.segment(image, **kwargs)


def get_backend(name: str, **kwargs) -> SegmenterBackend:
    """Construct a standard backend by name."""
    key = name.strip().lower()
    if key in {"threshold", "otsu"}:
        return ThresholdSegmenter(adaptive=False, **kwargs)
    if key in {"adaptive", "adaptive_threshold"}:
        return ThresholdSegmenter(adaptive=True, **kwargs)
    if key == "cellpose":
        return CellposeBackend(**kwargs)
    raise ValueError(f"Unknown segmentation backend: {name!r}")


def compare_backends(
    image: np.ndarray,
    backends: dict[str, SegmenterBackend],
    min_agreement: float = 0.6,
) -> EnsembleResult:
    """Run registered backends and summarize their consensus/disagreement."""
    if not backends:
        raise ValueError("at least one backend is required")
    results = []
    names = []
    for name, backend in backends.items():
        results.append(backend.segment(image))
        names.append(name)
    return ensemble_from_results(results, names=names, min_agreement=min_agreement)


__all__ = [
    "BaseSegmenter",
    "SegmenterBackend",
    "ThresholdSegmenter",
    "CellposeBackend",
    "CellposeSegmenter",
    "SegmentationResult",
    "segment_threshold",
    "EnsembleResult",
    "ensemble_from_results",
    "threshold_ensemble",
    "get_backend",
    "compare_backends",
]

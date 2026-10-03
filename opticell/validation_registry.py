"""Evidence registry for documenting validation breadth without overclaiming coverage."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class ValidationEvidence:
    dataset: str
    modality: str
    task: str
    source: str
    independent_lab: bool
    n_images: int | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def coverage_summary(entries: Iterable[ValidationEvidence]) -> dict:
    rows = list(entries)
    return {
        "datasets": sorted({row.dataset for row in rows}),
        "modalities": sorted({row.modality for row in rows}),
        "tasks": sorted({row.task for row in rows}),
        "n_evidence_sets": len(rows),
        "n_independent_lab_sets": sum(bool(row.independent_lab) for row in rows),
        "has_independent_lab_validation": any(row.independent_lab for row in rows),
    }


def known_validation_evidence() -> tuple[ValidationEvidence, ...]:
    """Evidence currently documented in this repository; not a universal-validity claim."""
    return (
        ValidationEvidence("BBBC039", "fluorescence nuclei", "segmentation", "repository validation outputs", False, 197),
        ValidationEvidence("CTC TRA", "time-lapse cell tracking challenge", "tracking/segmentation", "repository validation outputs", False, None),
        ValidationEvidence("LIVECell", "phase contrast", "segmentation", "repository validation outputs", False, 20, "Current locked subset evidence"),
    )


__all__ = ["ValidationEvidence", "coverage_summary", "known_validation_evidence"]

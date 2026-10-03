"""Versioned segmentation configuration and model/checkpoint provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SegmentationConfig:
    backend: str = "auto"
    gpu: bool = False
    cellpose_model: str = "cpsam"
    diameter: float | None = None
    min_area: int = 15
    max_area_frac: float = 0.25
    adaptive_threshold: bool = False
    minimum_quality: float = 70.0
    maximum_border_fraction: float = 0.35
    maximum_tiny_fraction: float = 0.50
    maximum_merged_fraction: float = 0.25
    checkpoint_sha256: str | None = None

    def validate(self) -> None:
        if self.backend not in {"auto", "threshold", "adaptive", "cellpose", "hybrid"}:
            raise ValueError("unsupported segmentation backend")
        if self.diameter is not None and self.diameter <= 0:
            raise ValueError("diameter must be positive")
        if self.min_area < 1:
            raise ValueError("min_area must be positive")
        if not 0 < self.max_area_frac <= 1:
            raise ValueError("max_area_frac must be in (0,1]")
        if not self.cellpose_model.strip():
            raise ValueError("cellpose_model must not be empty")
        if not 0 <= self.minimum_quality <= 100:
            raise ValueError("minimum_quality must be in [0,100]")
        for name in ("maximum_border_fraction", "maximum_tiny_fraction", "maximum_merged_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


PRESETS: dict[str, SegmentationConfig] = {
    "auto-balanced": SegmentationConfig(),
    "cpu-threshold": SegmentationConfig(backend="threshold", gpu=False),
    "cpu-adaptive": SegmentationConfig(backend="adaptive", gpu=False, adaptive_threshold=True),
    "cellpose-cpsam": SegmentationConfig(backend="cellpose", gpu=True, cellpose_model="cpsam"),
    "hybrid-cpsam": SegmentationConfig(backend="hybrid", gpu=True, cellpose_model="cpsam"),
}


def checkpoint_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_from_preset(name: str, overrides: Mapping[str, Any] | None = None) -> SegmentationConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}")
    values = PRESETS[name].to_dict()
    values.update(dict(overrides or {}))
    config = SegmentationConfig(**values)
    config.validate()
    return config


def config_for_checkpoint(path: str | Path, *, backend: str = "cellpose", gpu: bool = True, **kwargs: Any) -> SegmentationConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return SegmentationConfig(
        backend=backend,
        gpu=gpu,
        cellpose_model=str(source),
        checkpoint_sha256=checkpoint_sha256(source),
        **kwargs,
    )


def compare_configs(left: SegmentationConfig, right: SegmentationConfig) -> dict[str, tuple[Any, Any]]:
    l, r = left.to_dict(), right.to_dict()
    return {key: (l.get(key), r.get(key)) for key in sorted(set(l) | set(r)) if l.get(key) != r.get(key)}


__all__ = ["PRESETS", "SegmentationConfig", "checkpoint_sha256", "compare_configs", "config_for_checkpoint", "config_from_preset"]

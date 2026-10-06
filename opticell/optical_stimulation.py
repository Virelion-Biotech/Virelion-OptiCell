"""Shared optical-stimulation metadata helpers for OptiCell.

OptiCell records the perturbation associated with an imaging experiment; it does
not control light hardware or infer opsin biology.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Mapping

OPTICAL_STIMULATION_SCHEMA_VERSION = "virelion.optical-stimulation/1.0.0"


def validate_optical_stimulation_metadata(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimal fields OptiCell needs and return a detached copy."""
    if protocol.get("schema_version") != OPTICAL_STIMULATION_SCHEMA_VERSION:
        raise ValueError("unsupported optical stimulation schema_version")
    protocol_id = str(protocol.get("protocol_id", "")).strip()
    if not protocol_id:
        raise ValueError("protocol_id must be non-empty")
    if protocol.get("modality") not in {"optogenetic", "optoelectronic"}:
        raise ValueError("unsupported optical stimulation modality")

    light = protocol.get("light")
    timing = protocol.get("timing")
    control = protocol.get("control")
    if not isinstance(light, Mapping) or not isinstance(timing, Mapping) or not isinstance(control, Mapping):
        raise ValueError("light, timing and control objects are required")

    required_light = ("wavelength_nm", "irradiance_mw_mm2", "pulse_width_ms", "frequency_hz")
    values = []
    for key in required_light:
        if key not in light:
            raise ValueError(f"light.{key} is required")
        values.append(float(light[key]))
    for key in ("start_ms", "duration_ms"):
        if key not in timing:
            raise ValueError(f"timing.{key} is required")
        values.append(float(timing[key]))
    if not all(isfinite(value) for value in values):
        raise ValueError("optical stimulation numeric values must be finite")
    if float(light["wavelength_nm"]) <= 0 or float(light["pulse_width_ms"]) <= 0:
        raise ValueError("wavelength and pulse width must be positive")
    if float(light["irradiance_mw_mm2"]) < 0 or float(light["frequency_hz"]) < 0:
        raise ValueError("irradiance and frequency must be non-negative")
    if float(timing["start_ms"]) < 0 or float(timing["duration_ms"]) <= 0:
        raise ValueError("start must be non-negative and duration positive")
    if control.get("mode") not in {"open_loop", "closed_loop"}:
        raise ValueError("control.mode must be open_loop or closed_loop")
    return deepcopy(dict(protocol))


def extract_optical_stimulation(payload: Mapping[str, Any], observation_values: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Find one shared protocol object at the HeartTwin or imaging-observation boundary."""
    candidate = payload.get("optogenetic_stimulation")
    if candidate is None and observation_values is not None:
        candidate = observation_values.get("optogenetic_stimulation")
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        raise ValueError("optogenetic_stimulation must be an object")
    return validate_optical_stimulation_metadata(candidate)

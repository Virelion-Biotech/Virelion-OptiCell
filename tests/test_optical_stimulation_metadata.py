import pytest

from opticell.optical_stimulation import (
    OPTICAL_STIMULATION_SCHEMA_VERSION,
    extract_optical_stimulation,
    validate_optical_stimulation_metadata,
)


def _protocol():
    return {
        "schema_version": OPTICAL_STIMULATION_SCHEMA_VERSION,
        "protocol_id": "fixture-optical",
        "modality": "optogenetic",
        "target": {"cell_type": "hiPSC-CM", "spatial_pattern": "global"},
        "actuator": {"name": "CheRiff2.0", "class": "opsin", "expression_method": "viral"},
        "light": {
            "wavelength_nm": 470.0,
            "irradiance_mw_mm2": 0.8,
            "pulse_width_ms": 5.0,
            "frequency_hz": 2.0,
        },
        "timing": {"start_ms": 0.0, "duration_ms": 1000.0},
        "control": {"mode": "open_loop"},
        "provenance": {"source": "experimental", "source_id": "fixture"},
    }


def test_valid_protocol_round_trips_without_mutating_input():
    protocol = _protocol()
    validated = validate_optical_stimulation_metadata(protocol)
    assert validated == protocol
    assert validated is not protocol


def test_extract_prefers_top_level_hearttwin_protocol():
    top = _protocol()
    nested = _protocol()
    nested["protocol_id"] = "nested"
    found = extract_optical_stimulation(
        {"optogenetic_stimulation": top},
        {"optogenetic_stimulation": nested},
    )
    assert found["protocol_id"] == "fixture-optical"


def test_missing_protocol_is_backward_compatible():
    assert extract_optical_stimulation({}, {}) is None


def test_invalid_protocol_fails_closed():
    protocol = _protocol()
    protocol["light"]["wavelength_nm"] = -1
    with pytest.raises(ValueError):
        validate_optical_stimulation_metadata(protocol)

from __future__ import annotations

import json

import pytest

from experiments.historical_recovery.recovery_first_campaign import (
    DEFAULT_PROTOCOL,
    EXPECTED_CASES,
    EXPECTED_MAPPING,
    EXPECTED_SEEDS,
    b0_provenance,
    b1_fixed_action,
    b2_selector_parity,
    b3_end_to_end,
    load_protocol,
)


def test_recovery_first_protocol_freezes_scope_and_disables_new_mechanisms() -> None:
    protocol = load_protocol()
    assert tuple(protocol["cases"]) == EXPECTED_CASES
    assert tuple(protocol["seeds"]) == EXPECTED_SEEDS
    assert protocol["historical_action_mapping"] == EXPECTED_MAPPING
    assert protocol["phase1_fes"] == 180_000
    assert protocol["phase2_fes"] == 2_820_000
    assert protocol["terminal_fes"] == 3_000_000
    assert protocol["patch_enabled"] is False
    assert protocol["soft_routing_enabled"] is False
    assert protocol["new_selector_enabled"] is False


@pytest.mark.parametrize("field", ("cases", "seeds", "historical_action_mapping"))
def test_recovery_first_protocol_rejects_scope_drift(tmp_path, field: str) -> None:
    payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    if field == "cases":
        payload[field] = payload[field][:-1]
    elif field == "seeds":
        payload[field] = payload[field][1:]
    else:
        payload[field]["A1"] = "ctp"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol drifted|mapping drifted"):
        load_protocol(path)


def test_b0_preflight_provenance_is_read_only_and_complete() -> None:
    protocol = load_protocol()
    result = b0_provenance(
        protocol,
        cases=protocol["preflight_cases"],
        seeds=protocol["preflight_seeds"],
    )
    assert result["gate_passed"] is True
    assert result["context_count"] == 4
    assert all(row["valid"] for row in result["rows"])


def test_b1_does_not_call_incomplete_matrix_a_recovery() -> None:
    protocol = load_protocol()
    result = b1_fixed_action(protocol, mode="full", execute=False)
    assert result["gate_passed"] is False
    assert result["expected_arm_count"] == 2400
    assert result["observed_arm_count"] == 0
    assert result["historical_comparison"]["status"] == "not_bitwise_comparable"


def test_b2_selector_parity_never_evaluates_an_action() -> None:
    protocol = load_protocol()
    result = b2_selector_parity(
        protocol,
        cases=protocol["preflight_cases"],
        seeds=protocol["preflight_seeds"],
    )
    assert result["gate_passed"] is True
    assert result["action_evaluation_performed"] is False
    assert all(row["parity"] for row in result["rows"])


def test_b3_e2e_keeps_selector_route_separate_from_historical_fixed_expert() -> None:
    protocol = load_protocol()
    result = b3_end_to_end(
        protocol,
        cases=protocol["preflight_cases"],
        seeds=protocol["preflight_seeds"],
    )
    assert result["gate_passed"] is False
    assert set(result["unrecovered_cases"]) == {"A1", "R1", "S1"}
    assert all(row["terminal_contract"] for row in result["case_summaries"].values())

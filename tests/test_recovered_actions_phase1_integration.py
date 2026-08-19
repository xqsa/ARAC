from __future__ import annotations

from arac.core import select_core_action
from arac.runtime.contracts import ACTION_NAMES
from experiments.historical_recovery.recovered_actions_phase1_integration import (
    REPOSITORY_ROOT,
    _checkpoint,
    _load_json,
    load_core_protocol,
    load_protocol,
)


def test_fixed_action_protocol_freezes_one_lane_per_action() -> None:
    protocol = load_protocol()

    assert protocol["phase1_fes"] == 180_000
    assert protocol["phase2_fes"] == 2_820_000
    assert protocol["total_budget_fes"] == 3_000_000
    assert protocol["allow_out_of_bounds"] is True
    assert {lane["action"] for lane in protocol["lanes"]} == set(ACTION_NAMES)
    assert all(lane["historical_p90"] >= 0.0 for lane in protocol["lanes"])


def test_core_protocol_is_authorized_by_fixed_action_gate() -> None:
    protocol = load_core_protocol()

    assert protocol["selector_execution_allowed"] is True
    assert protocol["selector_fallback_allowed"] is False
    assert protocol["allow_out_of_bounds"] is True
    assert {lane["expected_action"] for lane in protocol["lanes"]} == set(ACTION_NAMES)


def test_current_selector_prediction_is_recorded_before_core_execution() -> None:
    fixed = load_protocol()
    observed = {}
    for lane in fixed["lanes"]:
        wrapper = _load_json(REPOSITORY_ROOT / lane["checkpoint"])
        checkpoint = _checkpoint(wrapper["checkpoint"])
        observed[lane["case_id_audit_metadata"]] = select_core_action(checkpoint).action_name

    assert observed == {"A1": "ctp", "E1": "smp", "R1": "smp", "S1": "smp"}

from __future__ import annotations

from experiments.overlap_shared_patch_matched_host_gate import (
    DEFAULT_PROTOCOL,
    HOSTS,
    MODES,
    load_protocol,
    run_gate,
)


def test_matched_host_protocol_freezes_conflicting_forced_hosts() -> None:
    protocol = load_protocol()
    assert protocol["conflict_mode"] == "conflicting"
    assert tuple(protocol["hosts"]) == HOSTS
    assert tuple(protocol["modes"]) == MODES
    assert protocol["patch_lane_fes"] == 8
    assert protocol["selector_participates"] is False
    assert protocol["production_planner_modified"] is False


def test_m0_reachability_has_real_patch_receipts_and_traces() -> None:
    result = run_gate(DEFAULT_PROTOCOL, stage="m0")
    assert result["gate_passed"] is True
    assert result["performance_comparison_authorized"] is False
    assert result["checks"]["not_arbitration_only"] is True
    assert result["checks"]["patch_receipts_nonzero"] is True
    assert result["checks"]["a3_state_trace"] is True
    assert result["checks"]["a4_radius_trace"] is True


def test_m1_declares_nested_ablation_without_claiming_superiority() -> None:
    result = run_gate(DEFAULT_PROTOCOL, stage="m1")
    assert result["gate_passed"] is True
    assert result["checks"]["nested_ablation_declared"] is True
    assert result["performance_comparison_authorized"] is False


def test_matched_host_output_is_auditable() -> None:
    result = run_gate(DEFAULT_PROTOCOL, stage="m0")
    for context in result["contexts"]:
        assert context["route_is_forced_host"] is True
        assert context["matched_inputs"] is True
        assert {arm["mode"] for arm in context["arms"]} == set(MODES)
        assert all(arm["patch_lane_fes"] == 0 for arm in context["arms"] if arm["mode"] == "a0")
        assert all(arm["patch_lane_fes"] > 0 for arm in context["arms"] if arm["mode"] != "a0")

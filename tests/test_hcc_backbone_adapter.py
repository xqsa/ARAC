from __future__ import annotations

import pytest

from arac.action_space import ActionFamily
from arac.backends.hcc import (
    HccBackboneSnapshot,
    HccGroupSignal,
    build_hcc_evidence_profile,
    hcc_backend_semantics_for,
)
from arac.evidence import validate_runtime_payload
from arac.policy import ActionDecision


def test_hcc_snapshot_builds_reference_blind_evidence_profile() -> None:
    snapshot = HccBackboneSnapshot(
        run_id="hcc-smoke",
        problem_id="S4",
        seed=7,
        dimension=1000,
        group_count=4,
        overlap_group_count=2,
        overlapping_element_count=50,
        budget_remaining_ratio=0.75,
        groups=(
            HccGroupSignal(group_id="g0", fitness_delta=12.0, rank=1, shared_variable_count=20),
            HccGroupSignal(group_id="g1", fitness_delta=4.0, rank=2, shared_variable_count=15),
            HccGroupSignal(group_id="g2", fitness_delta=1.0, rank=3, shared_variable_count=0),
            HccGroupSignal(group_id="g3", fitness_delta=0.5, rank=4, shared_variable_count=0),
        ),
    )

    evidence = build_hcc_evidence_profile(snapshot)

    assert evidence.run_id == "hcc-smoke"
    assert evidence.problem_id == "S4"
    assert evidence.unit_type == "problem"
    assert evidence.overlap_degree == pytest.approx(0.5)
    assert evidence.shared_var_support_ratio == pytest.approx(0.05)
    assert evidence.group_gain_asymmetry > 0
    assert evidence.priority_spread == pytest.approx(0.75)
    assert evidence.budget_remaining_ratio == pytest.approx(0.75)
    validate_runtime_payload(evidence.__dict__)


def test_hcc_backend_semantics_maps_action_families_to_hcc_effects() -> None:
    decisions = {
        "isolate": ActionDecision(
            ActionFamily.ISOLATE,
            "isolate_conflicting_relation",
            "allow",
            "test",
            0.4,
        ),
        "protect": ActionDecision(
            ActionFamily.PROTECT,
            "protect_high_margin_group",
            "allow",
            "test",
            0.4,
        ),
        "repair": ActionDecision(
            ActionFamily.REASSIGN_REPAIR,
            "repair_shared_variable_binding",
            "allow",
            "test",
            0.4,
        ),
        "coordinate": ActionDecision(
            ActionFamily.COORDINATE,
            "allow_beneficial_coordination",
            "allow",
            "test",
            0.4,
        ),
    }

    assert hcc_backend_semantics_for(decisions["isolate"]).relation_handling_changed
    assert hcc_backend_semantics_for(decisions["protect"]).budget_allocation_changed
    assert hcc_backend_semantics_for(decisions["repair"]).variable_owner_changed
    assert hcc_backend_semantics_for(decisions["coordinate"]).coordination_mode_changed


def test_hcc_snapshot_rejects_forbidden_outcome_fields() -> None:
    snapshot = HccBackboneSnapshot(
        run_id="bad",
        problem_id="S4",
        seed=7,
        dimension=1000,
        group_count=1,
        overlap_group_count=0,
        overlapping_element_count=0,
        budget_remaining_ratio=0.5,
        groups=(HccGroupSignal(group_id="g0", fitness_delta=1.0, rank=1),),
        runtime_payload_extra={"final_error": 1.23},
    )

    with pytest.raises(ValueError, match="final_error"):
        build_hcc_evidence_profile(snapshot)

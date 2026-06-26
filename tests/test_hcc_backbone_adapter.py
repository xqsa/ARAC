from __future__ import annotations

import pytest

from arac.action_space import ActionFamily
from arac.backends.hcc import (
    HccBackboneSnapshot,
    HccGroupSignal,
    build_hcc_evidence_profile,
    hcc_backend_semantics_for,
    load_hcc_aob_topology,
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


def test_load_hcc_aob_topology_reads_source_metadata_without_optimizer_run() -> None:
    topology = load_hcc_aob_topology("S6")

    assert topology.problem_id == "S6"
    assert topology.function_name == "schwefel"
    assert topology.function_id == 6
    assert topology.dimension == 1000
    assert topology.dimension_real == 1190
    assert topology.overlap_gamma == 10
    assert topology.group_count == 20
    assert topology.overlap_group_count == 19
    assert topology.overlapping_element_count == 190
    assert topology.degree_of_overlap == pytest.approx(0.19)
    assert topology.global_fes == 1_056_000
    assert topology.source_level == "hcc_source_topology"
    assert topology.fresh_optimizer_execution is False
    assert topology.groups[0].shared_variable_count == 10


def test_hcc_aob_topology_preserves_aob_overlap_gradient() -> None:
    topologies = [load_hcc_aob_topology(f"E{idx}") for idx in range(1, 7)]

    assert [topology.overlap_gamma for topology in topologies] == [0, 1, 3, 5, 7, 10]
    assert [topology.dimension for topology in topologies] == [1000] * 6
    assert [topology.dimension_real for topology in topologies] == [1000, 1019, 1057, 1095, 1133, 1190]

from dataclasses import asdict

import pytest

from arac.backends.binary_lsgo import (
    BinaryLsgoExecutionRequest,
    BinaryLsgoGroupStats,
    BinaryLsgoSnapshot,
    build_binary_lsgo_evidence_profile,
    run_binary_lsgo,
)
from arac.action_space import ActionFamily
from arac.benchmarks.binary_lsgo import BinaryLsgoSpec, generate_binary_lsgo
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS
from arac.policy import ActionDecision, decide_action


def small_problem():
    return generate_binary_lsgo(
        BinaryLsgoSpec("small", 40, 8, 2, 5, True, 0.5, 0.5, 0.5, 0.5, 11)
    )


def test_request_rejects_invalid_budget_or_seed():
    problem = small_problem()
    with pytest.raises(ValueError, match="total_fes"):
        BinaryLsgoExecutionRequest(problem, optimizer_seed=1, total_fes=1)
    with pytest.raises(ValueError, match="phase_one_fraction"):
        BinaryLsgoExecutionRequest(problem, optimizer_seed=1, phase_one_fraction=1.0)
    with pytest.raises(ValueError, match="optimizer_seed"):
        BinaryLsgoExecutionRequest(problem, optimizer_seed=-1)


def test_snapshot_converts_to_runtime_legal_evidence():
    problem = small_problem()
    stats = tuple(
        BinaryLsgoGroupStats(group_index=index, proposed=2, accepted=index % 2, gain=float(index))
        for index in range(len(problem.topology.groups))
    )
    snapshot = BinaryLsgoSnapshot(
        run_id="test",
        lane_id="arac_policy",
        problem_id=problem.spec.problem_id,
        optimizer_seed=9,
        consumed_fes=8,
        total_fes=40,
        group_stats=stats,
        shared_proposals=4,
        rejected_shared_proposals=1,
        conflicting_shared_variables=2,
        rank_stability=0.75,
        topology=problem.topology,
    )
    evidence = build_binary_lsgo_evidence_profile(snapshot)
    assert evidence.problem_id == "small"
    assert evidence.budget_remaining_ratio == pytest.approx(0.8)
    assert evidence.harmful_coord_score == pytest.approx(0.25)
    assert set(asdict(evidence)).isdisjoint(FORBIDDEN_RUNTIME_FIELDS)


def test_reliable_high_conflict_evidence_can_trigger_isolation():
    problem = small_problem()
    stats = tuple(
        BinaryLsgoGroupStats(group_index=index, proposed=2, accepted=1, gain=1.0)
        for index in range(len(problem.topology.groups))
    )
    evidence = build_binary_lsgo_evidence_profile(
        BinaryLsgoSnapshot(
            run_id="test",
            lane_id="arac_policy",
            problem_id=problem.spec.problem_id,
            optimizer_seed=9,
            consumed_fes=16,
            total_fes=80,
            group_stats=stats,
            shared_proposals=10,
            rejected_shared_proposals=8,
            conflicting_shared_variables=0,
            rank_stability=0.8,
            topology=problem.topology,
        )
    )
    decision = decide_action(evidence)
    assert evidence.fallback_margin_proxy == pytest.approx(0.9)
    assert decision.action_name == "isolate_conflicting_relation"
    assert decision.decision == "allow"


def decision(family: ActionFamily, name: str) -> ActionDecision:
    return ActionDecision(family, name, "allow", "test_override", 1.0)


def test_execution_is_reproducible_and_exact_budget():
    request = BinaryLsgoExecutionRequest(small_problem(), optimizer_seed=17, total_fes=80)
    first = run_binary_lsgo(request)
    second = run_binary_lsgo(request)
    assert first == second
    assert first.ledger.phase_i_fe == 16
    assert first.ledger.phase_ii_fe == 64
    assert first.ledger.total_fe == 80
    assert not first.ledger.violation


@pytest.mark.parametrize(
    ("action", "field"),
    [
        (
            decision(ActionFamily.COORDINATE, "allow_beneficial_coordination"),
            "coordination_mode_changed",
        ),
        (
            decision(ActionFamily.ISOLATE, "isolate_conflicting_relation"),
            "relation_handling_changed",
        ),
        (
            decision(ActionFamily.REASSIGN_REPAIR, "repair_shared_variable_binding"),
            "variable_owner_changed",
        ),
        (
            decision(ActionFamily.PROTECT, "protect_high_margin_group"),
            "budget_allocation_changed",
        ),
    ],
)
def test_actions_change_optimizer_consumed_semantics(action: ActionDecision, field: str):
    result = run_binary_lsgo(
        BinaryLsgoExecutionRequest(small_problem(), optimizer_seed=17, total_fes=80),
        decision_override=action,
    )
    assert result.optimizer_consumed
    assert getattr(result.semantics, field)
    assert result.action_trace.consumed_fe == result.ledger.phase_ii_fe


def test_unsupported_action_fails_loudly():
    unsupported = decision(ActionFamily.TRAJECTORY, "budget_shift_mean_blend")
    with pytest.raises(ValueError, match="unsupported binary LSGO action"):
        run_binary_lsgo(
            BinaryLsgoExecutionRequest(small_problem(), optimizer_seed=17, total_fes=80),
            decision_override=unsupported,
        )

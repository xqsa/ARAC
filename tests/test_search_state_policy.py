from __future__ import annotations

from dataclasses import fields, replace

import pytest

from arac.actions import ActionFamily, action_by_name
import arac.policy.search_state_policy as policy


@pytest.fixture
def eligible_evidence() -> policy.SearchStateEvidence:
    return policy.SearchStateEvidence(
        complete_sweep=True,
        overlap_degree=0.10,
        phase_rescue_enabled=True,
        repair_lock_active=False,
        phase_i_tail_utility=2.0e-6,
        non_coordinate_fraction=0.60,
        conflict_fraction=0.50,
        writeback_unstable=False,
        recent_cc_utilities=(1.0e-7,),
        remaining_fes=900_000,
        max_fes=3_000_000,
        population_size=24,
    )


def _successful_outcome(
    state: policy.SearchStateSchedulerState,
    stage: str,
    *,
    utility: float = 2.0e-7,
    cc_utility: float = 1.0e-7,
    used_fes: int = 30_000,
) -> policy.SearchStateSchedulerState:
    return policy.record_search_state_outcome(
        state,
        stage=stage,
        accepted=True,
        utility=utility,
        required_utility_ratio=1.5,
        cc_utility=cc_utility,
        used_fes=used_fes,
    )


def test_initial_probe_is_rounded_and_reserves_cc(eligible_evidence) -> None:
    plan = policy.plan_search_state_action(
        eligible_evidence,
        policy.SearchStateSchedulerState(),
    )

    assert plan.action_name == policy.RESUME_PHASE_I_SEARCH_STATE
    assert plan.stage == policy.SEARCH_STATE_PROBE
    assert plan.requested_fes == 30_000
    assert plan.requested_fes % 24 == 0
    assert plan.cc_reserve_fes == 300_000
    assert plan.required_utility_ratio == 1.50


def test_ineligible_evidence_abstains(eligible_evidence) -> None:
    for evidence in (
        replace(eligible_evidence, complete_sweep=False),
        replace(eligible_evidence, repair_lock_active=True),
        replace(eligible_evidence, overlap_degree=0.0),
        replace(eligible_evidence, phase_i_tail_utility=0.0),
        replace(
            eligible_evidence,
            non_coordinate_fraction=0.0,
            conflict_fraction=0.0,
            writeback_unstable=False,
        ),
        replace(eligible_evidence, remaining_fes=300_000),
    ):
        plan = policy.plan_search_state_action(
            evidence,
            policy.SearchStateSchedulerState(),
        )
        assert plan.action_name == policy.CONTINUE_CANONICAL_CC
        assert plan.requested_fes == 0


def test_each_structural_signal_can_qualify(eligible_evidence) -> None:
    for evidence in (
        replace(eligible_evidence, non_coordinate_fraction=0.50, conflict_fraction=0.0),
        replace(eligible_evidence, non_coordinate_fraction=0.0, conflict_fraction=0.50),
        replace(
            eligible_evidence,
            non_coordinate_fraction=0.0,
            conflict_fraction=0.0,
            writeback_unstable=True,
        ),
    ):
        plan = policy.plan_search_state_action(
            evidence,
            policy.SearchStateSchedulerState(),
        )
        assert plan.action_name == policy.RESUME_PHASE_I_SEARCH_STATE


def test_accelerating_cc_uses_two_x_gate(eligible_evidence) -> None:
    evidence = replace(
        eligible_evidence,
        recent_cc_utilities=(1.0e-7, 2.0e-7),
    )
    plan = policy.plan_search_state_action(
        evidence,
        policy.SearchStateSchedulerState(),
    )

    assert plan.required_utility_ratio == 2.0


def test_failed_probe_blocks_future_state_actions(eligible_evidence) -> None:
    state = policy.record_search_state_outcome(
        policy.SearchStateSchedulerState(),
        stage=policy.SEARCH_STATE_PROBE,
        accepted=False,
        utility=0.0,
        required_utility_ratio=1.5,
        cc_utility=1.0e-7,
        used_fes=30_000,
    )

    assert state.phase == policy.SEARCH_STATE_BLOCKED
    assert state.intervention_fe == 30_000
    plan = policy.plan_search_state_action(eligible_evidence, state)
    assert plan.action_name == policy.CONTINUE_CANONICAL_CC


def test_normalized_gain_utility_clamps_non_improving_candidates() -> None:
    assert policy.normalized_gain_utility(100.0, 90.0, 10) == pytest.approx(0.01)
    assert policy.normalized_gain_utility(100.0, 100.0, 10) == 0.0
    assert policy.normalized_gain_utility(100.0, 110.0, 10) == 0.0
    assert policy.normalized_gain_utility(0.0, -1.0, 0) == pytest.approx(1.0)


def test_non_improving_candidate_fails_strict_acceptance() -> None:
    state = policy.record_search_state_outcome(
        policy.SearchStateSchedulerState(),
        stage=policy.SEARCH_STATE_PROBE,
        accepted=False,
        utility=0.0,
        required_utility_ratio=1.5,
        cc_utility=0.0,
        used_fes=24,
    )

    assert state.phase == policy.SEARCH_STATE_BLOCKED


def test_zero_cc_utility_requires_positive_state_utility() -> None:
    successful = policy.record_search_state_outcome(
        policy.SearchStateSchedulerState(),
        stage=policy.SEARCH_STATE_PROBE,
        accepted=True,
        utility=0.01,
        required_utility_ratio=1.5,
        cc_utility=0.0,
        used_fes=24,
    )
    failed = policy.record_search_state_outcome(
        policy.SearchStateSchedulerState(),
        stage=policy.SEARCH_STATE_PROBE,
        accepted=True,
        utility=0.0,
        required_utility_ratio=1.5,
        cc_utility=0.0,
        used_fes=24,
    )

    assert successful.phase == policy.SEARCH_STATE_AWAITING_CONFIRMATION_CC
    assert failed.phase == policy.SEARCH_STATE_BLOCKED


def test_population_rounded_budget_rounds_down_to_complete_populations() -> None:
    assert policy.population_rounded_budget(30_001, 24) == 30_000
    assert policy.population_rounded_budget(23, 24) == 0
    assert policy.population_rounded_budget(0, 24) == 0


def test_policy_dataclasses_exclude_forbidden_dispatch_fields() -> None:
    forbidden = {
        "case_id",
        "function_label",
        "paper_value",
        "historical_outcome",
        "final_error",
        "relative_gain",
        "problem_family",
    }
    for dataclass_type in (
        policy.SearchStateEvidence,
        policy.SearchStateSchedulerState,
        policy.SearchStateActionPlan,
    ):
        assert not forbidden.intersection(field.name for field in fields(dataclass_type))


def test_awaiting_confirmation_requires_a_new_cc_sweep(eligible_evidence) -> None:
    state = _successful_outcome(
        policy.SearchStateSchedulerState(),
        policy.SEARCH_STATE_PROBE,
    )

    same_sweep = policy.plan_search_state_action(
        eligible_evidence,
        state,
        new_complete_cc_sweep=False,
    )
    next_sweep = policy.plan_search_state_action(
        eligible_evidence,
        state,
        new_complete_cc_sweep=True,
    )

    assert same_sweep.action_name == policy.CONTINUE_CANONICAL_CC
    assert next_sweep.action_name == policy.RESUME_PHASE_I_SEARCH_STATE
    assert next_sweep.stage == policy.SEARCH_STATE_CONFIRMATION


def test_two_qualified_blocks_are_required_before_expansion(eligible_evidence) -> None:
    initial = policy.plan_search_state_action(
        eligible_evidence,
        policy.SearchStateSchedulerState(),
    )
    assert initial.stage == policy.SEARCH_STATE_PROBE

    after_probe = _successful_outcome(
        policy.SearchStateSchedulerState(),
        policy.SEARCH_STATE_PROBE,
    )
    confirmation = policy.plan_search_state_action(
        eligible_evidence,
        after_probe,
        new_complete_cc_sweep=True,
    )
    assert confirmation.stage == policy.SEARCH_STATE_CONFIRMATION

    after_confirmation = _successful_outcome(
        after_probe,
        policy.SEARCH_STATE_CONFIRMATION,
    )
    expansion = policy.plan_search_state_action(eligible_evidence, after_confirmation)
    assert after_confirmation.phase == policy.SEARCH_STATE_EXPANSION
    assert expansion.stage == policy.SEARCH_STATE_EXPANSION
    assert expansion.requested_fes == 30_000


def test_failed_utility_gate_permanently_blocks() -> None:
    state = policy.record_search_state_outcome(
        policy.SearchStateSchedulerState(),
        stage=policy.SEARCH_STATE_PROBE,
        accepted=True,
        utility=0.14,
        required_utility_ratio=1.5,
        cc_utility=0.10,
        used_fes=30_000,
    )

    assert state.phase == policy.SEARCH_STATE_BLOCKED


def test_cumulative_intervention_cap_and_reserve_are_hard_limits(eligible_evidence) -> None:
    near_cap = replace(
        policy.SearchStateSchedulerState(),
        intervention_fe=420_000,
    )
    plan = policy.plan_search_state_action(eligible_evidence, near_cap)
    assert plan.requested_fes == 30_000
    assert near_cap.intervention_fe + plan.requested_fes <= 450_000

    at_cap = replace(
        policy.SearchStateSchedulerState(),
        intervention_fe=450_000,
    )
    capped = policy.plan_search_state_action(eligible_evidence, at_cap)
    assert capped.action_name == policy.CONTINUE_CANONICAL_CC
    assert capped.requested_fes == 0

    reserve_limited = replace(eligible_evidence, remaining_fes=300_023)
    reserve_plan = policy.plan_search_state_action(
        reserve_limited,
        policy.SearchStateSchedulerState(),
    )
    assert reserve_plan.requested_fes == 0
    assert reserve_plan.requested_fes <= reserve_plan.cc_reserve_fes


def test_resume_action_is_registered_as_trajectory_core_intervention() -> None:
    action = action_by_name(policy.RESUME_PHASE_I_SEARCH_STATE)

    assert action.family == ActionFamily.TRAJECTORY
    assert action.backend_role == "core_intervention"
    assert action.requires_semantic_effect is True


def test_policy_can_emit_configured_diagonal_trajectory_action(
    eligible_evidence,
) -> None:
    plan = policy.plan_search_state_action(
        eligible_evidence,
        policy.SearchStateSchedulerState(),
        trajectory_action_name=policy.CONTINUE_DIAGONAL_SEARCH_STATE,
    )

    assert plan.action_name == policy.CONTINUE_DIAGONAL_SEARCH_STATE
    action = action_by_name(plan.action_name)
    assert action.family == ActionFamily.TRAJECTORY
    assert action.backend_role == "core_intervention"


def test_policy_rejects_unknown_trajectory_action(eligible_evidence) -> None:
    with pytest.raises(ValueError, match="unsupported trajectory action"):
        policy.plan_search_state_action(
            eligible_evidence,
            policy.SearchStateSchedulerState(),
            trajectory_action_name="unknown_search_backend",
        )


def test_diagonal_trajectory_can_probe_under_repair_lock_with_conflict_evidence(
    eligible_evidence,
) -> None:
    evidence = replace(eligible_evidence, repair_lock_active=True)

    diagonal_plan = policy.plan_search_state_action(
        evidence,
        policy.SearchStateSchedulerState(),
        trajectory_action_name=policy.CONTINUE_DIAGONAL_SEARCH_STATE,
    )
    mmes_plan = policy.plan_search_state_action(
        evidence,
        policy.SearchStateSchedulerState(),
        trajectory_action_name=policy.RESUME_PHASE_I_SEARCH_STATE,
    )

    assert diagonal_plan.action_name == policy.CONTINUE_DIAGONAL_SEARCH_STATE
    assert mmes_plan.action_name == policy.CONTINUE_CANONICAL_CC

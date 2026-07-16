from __future__ import annotations

import inspect
import math

import pytest

from arac.policy.overlap_hypergraph import (
    DELAYED_OVERWRITE_PENALTY,
    FINAL_OWNER_PROPOSAL_WATERMARK,
    GroupCycleObservation,
    HyperedgeCycleState,
    HyperedgeScore,
    SharedProposal,
    build_delayed_hyperedge_credit,
    build_group_cycle_observation,
    build_hyperedge_cycle_states,
    build_overlap_hypergraph,
    directional_survival,
    midrank_percentiles,
    plan_sweep_coordination,
    project_owner_weights,
    score_hyperedge_states,
    unit_fe_contribution,
)


def _observation(
    *,
    sweep: int,
    group: int,
    contribution: float,
    values: tuple[tuple[int, float], ...],
) -> GroupCycleObservation:
    return GroupCycleObservation(
        sweep_index=sweep,
        group_index=group,
        primary_requested_fe=8,
        primary_actual_fe=8,
        full_interval_actual_fe=10,
        full_interval_start_fe=(sweep * 3 + group) * 10,
        full_interval_end_fe=(sweep * 3 + group + 1) * 10,
        unit_fe_contribution=contribution,
        successful=contribution > 0.0,
        shared_proposal=SharedProposal(
            group_index=group,
            anchor_values=tuple((variable, 0.0) for variable, _ in values),
            proposed_values=values,
        ),
    )


def _three_sweep_fixture():
    topology = build_overlap_hypergraph([[0, 1], [1, 2], [2, 3]])
    histories = {
        0: ((0.0, ((1, 0.0),)), (0.0, ((1, 1.0),)), (3.0, ((1, 0.0),))),
        1: (
            (1.0, ((1, 0.0), (2, 1.0))),
            (2.0, ((1, 1.0), (2, 1.0))),
            (0.0, ((1, 2.0), (2, 2.0))),
        ),
        2: ((0.0, ((2, 0.0),)), (0.0, ((2, 1.0),)), (0.0, ((2, 6.0),))),
    }
    observations = tuple(
        _observation(sweep=sweep, group=group, contribution=gain, values=values)
        for group, history in histories.items()
        for sweep, (gain, values) in enumerate(history)
    )
    states = build_hyperedge_cycle_states(
        topology,
        observations,
        prior_next_sweep_survival_by_group={0: 0.8, 1: 0.5, 2: 0.2},
        prior_next_sweep_overwrite_by_group={0: 0.2, 1: 0.5, 2: 0.8},
        lower_bound=-5.0,
        upper_bound=5.0,
    )
    current = {
        observation.group_index: observation
        for observation in observations
        if observation.sweep_index == 2
    }
    return topology, observations, states, current


def test_raw_hypergraph_preserves_stars_without_transitive_scope() -> None:
    topology = build_overlap_hypergraph([[0, 1], [1, 2], [2, 3], [4]])

    assert topology.hyperedges == ((0, 1), (1, 2), (2, 3), (4,))
    assert topology.variable_owner_groups == (
        (0, (0,)),
        (1, (0, 1)),
        (2, (1, 2)),
        (3, (2,)),
        (4, (3,)),
    )
    assert topology.overlap_variables == (1, 2)
    assert topology.membership_histogram == ((1, 3), (2, 2))
    assert topology.focal_scope(0).direct_owner_group_indices == (0, 1)
    assert topology.focal_scope(0).neighbor_group_indices == (1,)
    assert 2 not in topology.focal_scope(0).direct_owner_group_indices
    assert topology.focal_scope(1).direct_owner_group_indices == (0, 1, 2)


def test_group_observation_keeps_derived_contribution_and_shared_values_only() -> None:
    topology = build_overlap_hypergraph([[0, 1], [1, 2]])
    before = [4.0, 1.0, 9.0]
    proposal = [8.0, 3.0, 7.0]

    observation = build_group_cycle_observation(
        topology,
        sweep_index=2,
        group_index=0,
        pre_error=100.0,
        best_error=80.0,
        primary_requested_fe=20,
        primary_actual_fe=20,
        full_interval_actual_fe=25,
        full_interval_start_fe=10,
        full_interval_end_fe=35,
        pre_block_candidate=before,
        final_owner_candidate=proposal,
    )
    before[1] = -99.0
    proposal[1] = -99.0

    assert observation.successful is True
    assert observation.shared_proposal.anchor_values == ((1, 1.0),)
    assert observation.shared_proposal.proposed_values == ((1, 3.0),)
    assert observation.shared_proposal.capture_watermark == (
        FINAL_OWNER_PROPOSAL_WATERMARK
    )
    assert observation.full_interval_actual_fe > observation.primary_requested_fe
    assert math.isclose(observation.unit_fe_contribution, 40.0 * math.log(1.25))
    assert "error" not in observation.__dataclass_fields__
    assert math.isclose(
        unit_fe_contribution(pre_error=100.0, best_error=120.0, actual_fe=20),
        0.0,
    )


def test_group_observation_rejects_incomplete_full_interval_accounting() -> None:
    proposal = SharedProposal(
        group_index=0,
        anchor_values=((1, 0.0),),
        proposed_values=((1, 1.0),),
    )

    with pytest.raises(ValueError, match="must cover primary_actual_fe"):
        GroupCycleObservation(
            sweep_index=0,
            group_index=0,
            primary_requested_fe=20,
            primary_actual_fe=20,
            full_interval_actual_fe=19,
            full_interval_start_fe=0,
            full_interval_end_fe=19,
            unit_fe_contribution=1.0,
            successful=True,
            shared_proposal=proposal,
        )
    with pytest.raises(ValueError, match="final owner watermark"):
        SharedProposal(
            group_index=0,
            anchor_values=((1, 0.0),),
            proposed_values=((1, 1.0),),
            capture_watermark="before_recovery",
        )


def test_three_sweep_state_uses_fixed_ewma_difficulty_and_stagnation() -> None:
    topology, _, states, _ = _three_sweep_fixture()
    by_group = dict(zip(topology.eligible_group_indices, states, strict=True))

    assert math.isclose(by_group[0].ewma_unit_fe_contribution_3, 1.5)
    assert math.isclose(by_group[0].success_ratio_3, 1.0 / 3.0)
    assert math.isclose(by_group[0].zero_gain_difficulty, 2.0 / 3.0)
    assert by_group[0].stagnation_ratio_3 == 0.0
    assert math.isclose(by_group[0].direct_owner_proposal_disagreement, 0.2)
    assert math.isclose(by_group[1].ewma_unit_fe_contribution_3, 0.75)
    assert math.isclose(by_group[1].stagnation_ratio_3, 1.0 / 3.0)
    assert math.isclose(by_group[1].direct_owner_proposal_disagreement, 0.3)
    assert by_group[2].zero_gain_difficulty == 1.0
    assert by_group[2].stagnation_ratio_3 == 1.0
    assert math.isclose(by_group[2].direct_owner_proposal_disagreement, 0.4)


def test_incomplete_history_or_prior_credit_fails_closed() -> None:
    topology, observations, _, _ = _three_sweep_fixture()

    with pytest.raises(ValueError, match="three consecutive complete sweeps"):
        build_hyperedge_cycle_states(
            topology,
            [row for row in observations if row.sweep_index != 0],
            prior_next_sweep_survival_by_group={0: 0.8, 1: 0.5, 2: 0.2},
            prior_next_sweep_overwrite_by_group={0: 0.2, 1: 0.5, 2: 0.8},
            lower_bound=-5.0,
            upper_bound=5.0,
        )
    with pytest.raises(ValueError, match="missing closed prior"):
        build_hyperedge_cycle_states(
            topology,
            observations,
            prior_next_sweep_survival_by_group={0: 0.8, 1: 0.5},
            prior_next_sweep_overwrite_by_group={0: 0.2, 1: 0.5, 2: 0.8},
            lower_bound=-5.0,
            upper_bound=5.0,
        )


def test_midrank_scores_have_unique_focal_without_index_tiebreak() -> None:
    topology, _, states, _ = _three_sweep_fixture()
    scores = score_hyperedge_states(states)
    by_group = dict(zip(topology.eligible_group_indices, scores, strict=True))

    assert midrank_percentiles([4.0]) == (0.5,)
    assert midrank_percentiles([1.0, 1.0, 3.0]) == (1.0 / 3.0, 1.0 / 3.0, 5.0 / 6.0)
    assert by_group[0].focal_priority > by_group[1].focal_priority
    assert by_group[0].focal_priority > by_group[2].focal_priority
    assert math.isclose(by_group[0].owner_reliability, 5.0 / 6.0)

    tied_states = tuple(
        HyperedgeCycleState(
            current_unit_fe_contribution=1.0,
            ewma_unit_fe_contribution_3=1.0,
            success_ratio_3=0.5,
            zero_gain_difficulty=0.5,
            stagnation_ratio_3=0.5,
            direct_owner_proposal_disagreement=0.5,
            prior_next_sweep_survival=0.5,
            prior_next_sweep_overwrite=0.5,
        )
        for _ in range(3)
    )
    assert len({score.focal_priority for score in score_hyperedge_states(tied_states)}) == 1
    assert "group_index" not in HyperedgeCycleState.__dataclass_fields__
    assert "group_index" not in HyperedgeScore.__dataclass_fields__


def test_candidate_plan_uses_capped_direct_owner_weights_and_fixed_risk_step() -> None:
    topology, _, states, current = _three_sweep_fixture()
    scores = score_hyperedge_states(states)
    plan = plan_sweep_coordination(
        topology,
        scores=scores,
        current_observations=current,
        sweep_end_anchor=(0.0, -5.0, 0.0, 0.0),
        lower_bound=-5.0,
        upper_bound=5.0,
    )

    assert project_owner_weights((1.0, 0.0)) == pytest.approx((0.65, 0.35))
    assert plan.selected is True
    assert plan.focal_group_index == 0
    assert plan.shared_variables == (1,)
    assert plan.direct_owner_group_indices == (0, 1)
    assert math.isclose(plan.structural_risk or -1.0, 0.5)
    assert math.isclose(plan.proposal_range_norm or -1.0, 0.2)
    assert plan.candidate[0] == 0.0
    assert math.isclose(plan.candidate[1], -4.0)
    assert plan.candidate[2:] == (0.0, 0.0)


def test_complete_priority_tie_abstains_without_group_index_tiebreak() -> None:
    topology, _, _, current = _three_sweep_fixture()
    tied = tuple(
        HyperedgeScore(
            contribution_score=0.5,
            need_score=0.5,
            focal_priority=0.5,
            owner_reliability=0.5,
        )
        for _ in topology.eligible_group_indices
    )

    plan = plan_sweep_coordination(
        topology,
        scores=tied,
        current_observations=current,
        sweep_end_anchor=(0.0, 0.0, 0.0, 0.0),
        lower_bound=-5.0,
        upper_bound=5.0,
    )

    assert plan.selected is False
    assert plan.reason == "abstain_focal_priority_tie"
    assert plan.focal_group_index is None


def test_directional_survival_and_overwrite_penalty_close_at_next_sweep_end() -> None:
    retained = directional_survival(
        anchor_values=(0.0, 10.0),
        candidate_values=(2.0, 6.0),
        next_sweep_values=(1.0, 8.0),
    )
    credit = build_delayed_hyperedge_credit(
        action_sweep_index=3,
        resolution_sweep_index=4,
        all_groups_completed=True,
        native_sweep_end_completed=True,
        anchor_error=100.0,
        next_sweep_error=90.0,
        anchor_shared_values=(0.0, 10.0),
        candidate_shared_values=(2.0, 6.0),
        next_sweep_shared_values=(1.0, 8.0),
    )

    assert retained.survival == 0.5
    assert retained.overwrite == 0.5
    assert credit.survival == 0.5
    assert math.isclose(credit.next_sweep_log_improvement, math.log(100.0 / 90.0))
    assert math.isclose(
        credit.penalized_credit,
        math.log(100.0 / 90.0) - 0.5 * DELAYED_OVERWRITE_PENALTY,
    )
    neutral = directional_survival(
        anchor_values=(1.0, 2.0),
        candidate_values=(1.0, 2.0),
        next_sweep_values=(4.0, -3.0),
    )
    assert neutral.survival == 0.5
    assert neutral.overwrite == 0.5
    assert neutral.changed_variable_count == 0

    with pytest.raises(ValueError, match="complete native sweep end"):
        build_delayed_hyperedge_credit(
            action_sweep_index=3,
            resolution_sweep_index=4,
            all_groups_completed=True,
            native_sweep_end_completed=False,
            anchor_error=100.0,
            next_sweep_error=90.0,
            anchor_shared_values=(0.0,),
            candidate_shared_values=(1.0,),
            next_sweep_shared_values=(1.0,),
        )
    with pytest.raises(ValueError, match="wait for the next sweep"):
        build_delayed_hyperedge_credit(
            action_sweep_index=3,
            resolution_sweep_index=3,
            all_groups_completed=True,
            native_sweep_end_completed=True,
            anchor_error=100.0,
            next_sweep_error=90.0,
            anchor_shared_values=(0.0,),
            candidate_shared_values=(1.0,),
            next_sweep_shared_values=(1.0,),
        )


def test_plan_call_graph_excludes_identity_and_outcome_dispatch_inputs() -> None:
    signature = inspect.signature(plan_sweep_coordination)
    source = inspect.getsource(plan_sweep_coordination).lower()
    forbidden = ("case", "seed", "family", "fingerprint", "paper", "error", "outcome")

    assert not any(fragment in name.lower() for name in signature.parameters for fragment in forbidden)
    assert not any(fragment in source for fragment in ("paper_best", "final_outcome", "case_label"))

from __future__ import annotations

import math
from dataclasses import replace

import pytest

import arac.policy as policy_package
import arac.policy.overlap_hypergraph as hypergraph_module
from arac.policy.overlap_hypergraph import (
    DELAYED_OVERWRITE_PENALTY,
    FINAL_OWNER_PROPOSAL_WATERMARK,
    ClosedOwnerCredit,
    CompletedSweepSnapshot,
    GroupCycleObservation,
    HyperedgeCycleState,
    HyperedgeScore,
    OverlapHypergraphTopology,
    SharedProposal,
    SharedVariableStar,
    build_closed_owner_credit,
    build_delayed_hyperedge_credit,
    build_group_cycle_observation,
    build_hyperedge_cycle_states,
    build_overlap_hypergraph,
    directional_survival,
    midrank_percentiles,
    score_hyperedge_states,
    unit_fe_contribution,
)


def _observation(
    topology: OverlapHypergraphTopology,
    *,
    sweep: int,
    group: int,
    contribution: float,
    values: tuple[tuple[int, float], ...],
) -> GroupCycleObservation:
    group_count = len(topology.hyperedges)
    start_fe = (sweep * group_count + group) * 10
    end_fe = start_fe + 10
    return GroupCycleObservation(
        sweep_index=sweep,
        group_index=group,
        primary_requested_fe=8,
        primary_actual_fe=8,
        full_interval_actual_fe=10,
        full_interval_start_fe=start_fe,
        full_interval_end_fe=end_fe,
        unit_fe_contribution=contribution,
        successful=contribution > 0.0,
        shared_proposal=SharedProposal(
            group_index=group,
            anchor_values=tuple((variable, 0.0) for variable, _ in values),
            proposed_values=values,
            capture_stage=FINAL_OWNER_PROPOSAL_WATERMARK,
            capture_fe=end_fe,
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
    endpoints = (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.8, 0.2, 0.0),
    )
    snapshots: list[CompletedSweepSnapshot] = []
    for sweep in range(3):
        observations = tuple(
            _observation(
                topology,
                sweep=sweep,
                group=group,
                contribution=histories[group][sweep][0],
                values=histories[group][sweep][1],
            )
            for group in range(3)
        )
        snapshots.append(
            CompletedSweepSnapshot(
                topology=topology,
                sweep_index=sweep,
                observations=observations,
                sweep_end_candidate=endpoints[sweep],
                native_sweep_end_completed=True,
                sweep_end_fe=(sweep + 1) * 30,
            )
        )
    credits = tuple(
        build_closed_owner_credit(
            proposal_observation=snapshots[1].observation_for_group(group),
            resolution_snapshot=snapshots[2],
        )
        for group in topology.eligible_group_indices
    )
    states = build_hyperedge_cycle_states(
        topology,
        snapshots,
        closed_owner_credits=credits,
        decision_fe=snapshots[-1].sweep_end_fe,
        lower_bound=-5.0,
        upper_bound=5.0,
    )
    return topology, tuple(snapshots), credits, states


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
    assert 2 not in topology.focal_scope(0).direct_owner_group_indices
    assert topology.focal_scope(1).direct_owner_group_indices == (0, 1, 2)


def test_hand_built_topology_must_match_raw_hyperedges() -> None:
    with pytest.raises(ValueError, match="owner stars"):
        OverlapHypergraphTopology(
            hyperedges=((0, 1), (1, 2)),
            stars=(
                SharedVariableStar(0, (0,)),
                SharedVariableStar(1, (0,)),
                SharedVariableStar(2, (1,)),
            ),
            group_shared_variables=((), ()),
        )


def test_group_observation_uses_full_interval_and_explicit_capture_evidence() -> None:
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
        capture_stage=FINAL_OWNER_PROPOSAL_WATERMARK,
        capture_fe=35,
    )
    before[1] = -99.0
    proposal[1] = -99.0

    assert observation.shared_proposal.anchor_values == ((1, 1.0),)
    assert observation.shared_proposal.proposed_values == ((1, 3.0),)
    assert observation.shared_proposal.capture_stage == FINAL_OWNER_PROPOSAL_WATERMARK
    assert observation.shared_proposal.capture_fe == 35
    assert observation.full_interval_actual_fe > observation.primary_requested_fe
    assert math.isclose(observation.unit_fe_contribution, 40.0 * math.log(1.25))
    assert "error" not in observation.__dataclass_fields__
    assert math.isclose(
        unit_fe_contribution(pre_error=100.0, best_error=120.0, actual_fe=20),
        0.0,
    )


@pytest.mark.parametrize(
    ("capture_stage", "capture_fe", "match"),
    [
        ("before_recovery", 35, "final owner watermark"),
        (FINAL_OWNER_PROPOSAL_WATERMARK, 34, "capture_fe"),
    ],
)
def test_group_observation_rejects_invalid_capture_evidence(
    capture_stage: str,
    capture_fe: int,
    match: str,
) -> None:
    topology = build_overlap_hypergraph([[0, 1], [1, 2]])

    with pytest.raises(ValueError, match=match):
        build_group_cycle_observation(
            topology,
            sweep_index=0,
            group_index=0,
            pre_error=10.0,
            best_error=9.0,
            primary_requested_fe=10,
            primary_actual_fe=8,
            full_interval_actual_fe=10,
            full_interval_start_fe=0,
            full_interval_end_fe=10,
            pre_block_candidate=(0.0, 0.0, 0.0),
            final_owner_candidate=(0.0, 1.0, 0.0),
            capture_stage=capture_stage,
            capture_fe=capture_fe,
        )


def test_completed_snapshot_requires_every_raw_group_and_true_handler_closure() -> None:
    topology, snapshots, _, _ = _three_sweep_fixture()
    snapshot = snapshots[0]

    with pytest.raises(ValueError, match="every raw group exactly once"):
        replace(snapshot, observations=snapshot.observations[:-1])
    with pytest.raises(ValueError, match="canonical group order"):
        replace(snapshot, observations=tuple(reversed(snapshot.observations)))
    with pytest.raises(ValueError, match="must be boolean"):
        replace(snapshot, native_sweep_end_completed="1")
    with pytest.raises(ValueError, match="native sweep-end handlers"):
        replace(snapshot, native_sweep_end_completed=False)
    with pytest.raises(ValueError, match="follow every group"):
        replace(snapshot, sweep_end_fe=snapshot.observations[-1].full_interval_end_fe - 1)

    disjoint_topology = build_overlap_hypergraph([[0, 1], [1, 2], [3]])
    only_overlap_groups = tuple(
        _observation(
            disjoint_topology,
            sweep=0,
            group=group,
            contribution=1.0,
            values=((1, float(group)),),
        )
        for group in (0, 1)
    )
    with pytest.raises(ValueError, match="every raw group exactly once"):
        CompletedSweepSnapshot(
            topology=disjoint_topology,
            sweep_index=0,
            observations=only_overlap_groups,
            sweep_end_candidate=(0.0, 0.0, 0.0, 0.0),
            native_sweep_end_completed=True,
            sweep_end_fe=20,
        )


def test_closed_owner_credit_uses_previous_proposal_and_current_sweep_endpoint() -> None:
    topology, snapshots, credits, _ = _three_sweep_fixture()
    by_group = dict(zip(topology.eligible_group_indices, credits, strict=True))

    assert by_group[0].proposal_sweep_index == 1
    assert by_group[0].resolution_sweep_index == 2
    assert by_group[0].proposal_source_fe == 40
    assert by_group[0].resolution_fe == 90
    assert by_group[0].survival == pytest.approx(0.8)
    assert by_group[1].survival == pytest.approx(0.5)
    assert by_group[2].survival == pytest.approx(0.2)

    with pytest.raises(ValueError, match="must be consecutive"):
        build_closed_owner_credit(
            proposal_observation=snapshots[2].observation_for_group(0),
            resolution_snapshot=snapshots[2],
        )


def test_state_recomputes_credit_and_rejects_future_or_forged_values() -> None:
    topology, snapshots, credits, _ = _three_sweep_fixture()
    forged = replace(credits[0], survival=0.9, overwrite=0.1)

    with pytest.raises(ValueError, match="sealed sweep evidence"):
        build_hyperedge_cycle_states(
            topology,
            snapshots,
            closed_owner_credits=(forged, *credits[1:]),
            decision_fe=90,
            lower_bound=-5.0,
            upper_bound=5.0,
        )
    with pytest.raises(ValueError, match="equal the current completed sweep end"):
        build_hyperedge_cycle_states(
            topology,
            snapshots,
            closed_owner_credits=credits,
            decision_fe=89,
            lower_bound=-5.0,
            upper_bound=5.0,
        )
    with pytest.raises(ValueError, match="equal the current completed sweep end"):
        build_hyperedge_cycle_states(
            topology,
            snapshots,
            closed_owner_credits=credits,
            decision_fe=91,
            lower_bound=-5.0,
            upper_bound=5.0,
        )
    with pytest.raises(ValueError, match="every eligible group"):
        build_hyperedge_cycle_states(
            topology,
            snapshots,
            closed_owner_credits=credits[:-1],
            decision_fe=90,
            lower_bound=-5.0,
            upper_bound=5.0,
        )
    future_route = replace(
        credits[0],
        proposal_sweep_index=2,
        resolution_sweep_index=3,
        proposal_source_fe=80,
        resolution_fe=120,
    )
    with pytest.raises(ValueError, match="proposal sweep"):
        build_hyperedge_cycle_states(
            topology,
            snapshots,
            closed_owner_credits=(future_route, *credits[1:]),
            decision_fe=90,
            lower_bound=-5.0,
            upper_bound=5.0,
        )


def test_three_sweep_state_uses_fixed_ewma_difficulty_and_stagnation() -> None:
    topology, _, _, states = _three_sweep_fixture()
    by_group = dict(zip(topology.eligible_group_indices, states, strict=True))

    assert math.isclose(by_group[0].ewma_unit_fe_contribution_3, 1.5)
    assert math.isclose(by_group[0].zero_gain_difficulty, 2.0 / 3.0)
    assert by_group[0].stagnation_ratio_3 == 0.0
    assert math.isclose(by_group[0].direct_owner_proposal_disagreement, 0.2)
    assert math.isclose(by_group[1].ewma_unit_fe_contribution_3, 0.75)
    assert math.isclose(by_group[1].stagnation_ratio_3, 1.0 / 3.0)
    assert math.isclose(by_group[1].direct_owner_proposal_disagreement, 0.3)
    assert by_group[2].zero_gain_difficulty == 1.0
    assert by_group[2].stagnation_ratio_3 == 1.0
    assert math.isclose(by_group[2].direct_owner_proposal_disagreement, 0.4)
    assert by_group[0].prior_next_sweep_overwrite == pytest.approx(0.2)

    assert tuple(HyperedgeCycleState.__dataclass_fields__) == (
        "current_unit_fe_contribution",
        "ewma_unit_fe_contribution_3",
        "zero_gain_difficulty",
        "stagnation_ratio_3",
        "direct_owner_proposal_disagreement",
        "prior_next_sweep_overwrite",
    )


def test_state_requires_exactly_three_consecutive_completed_snapshots() -> None:
    topology, snapshots, credits, _ = _three_sweep_fixture()

    with pytest.raises(ValueError, match="exactly three"):
        build_hyperedge_cycle_states(
            topology,
            snapshots[1:],
            closed_owner_credits=credits,
            decision_fe=90,
            lower_bound=-5.0,
            upper_bound=5.0,
        )
    shifted = replace(
        snapshots[0],
        sweep_index=4,
        observations=tuple(
            replace(observation, sweep_index=4) for observation in snapshots[0].observations
        ),
    )
    with pytest.raises(ValueError, match="three consecutive"):
        build_hyperedge_cycle_states(
            topology,
            (shifted, snapshots[1], snapshots[2]),
            closed_owner_credits=credits,
            decision_fe=90,
            lower_bound=-5.0,
            upper_bound=5.0,
        )


def test_midrank_scores_are_identity_free_and_fixed() -> None:
    topology, _, _, states = _three_sweep_fixture()
    scores = score_hyperedge_states(states)
    by_group = dict(zip(topology.eligible_group_indices, scores, strict=True))

    assert midrank_percentiles([4.0]) == (0.5,)
    assert midrank_percentiles([1.0, 1.0, 3.0]) == (
        1.0 / 3.0,
        1.0 / 3.0,
        5.0 / 6.0,
    )
    assert by_group[0].focal_priority > by_group[1].focal_priority
    assert by_group[0].focal_priority > by_group[2].focal_priority
    assert math.isclose(by_group[0].owner_reliability, 5.0 / 6.0)
    assert "group_index" not in HyperedgeCycleState.__dataclass_fields__
    assert "group_index" not in HyperedgeScore.__dataclass_fields__


def test_task_two_exports_no_candidate_or_runtime_action_surface() -> None:
    forbidden = {
        "project_owner_weights",
        "SweepCoordinationPlan",
        "plan_sweep_coordination",
    }

    assert forbidden.isdisjoint(policy_package.__all__)
    assert all(not hasattr(hypergraph_module, name) for name in forbidden)


def test_directional_survival_and_delayed_credit_require_strict_closure() -> None:
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
    assert credit.survival == 0.5
    assert math.isclose(
        credit.penalized_credit,
        math.log(100.0 / 90.0) - 0.5 * DELAYED_OVERWRITE_PENALTY,
    )
    neutral = directional_survival(
        anchor_values=(1.0, 2.0),
        candidate_values=(1.0, 2.0),
        next_sweep_values=(4.0, -3.0),
    )
    assert (neutral.survival, neutral.overwrite, neutral.changed_variable_count) == (
        0.5,
        0.5,
        0,
    )

    with pytest.raises(ValueError, match="must be boolean"):
        build_delayed_hyperedge_credit(
            action_sweep_index=3,
            resolution_sweep_index=4,
            all_groups_completed="1",
            native_sweep_end_completed=True,
            anchor_error=100.0,
            next_sweep_error=90.0,
            anchor_shared_values=(0.0,),
            candidate_shared_values=(1.0,),
            next_sweep_shared_values=(1.0,),
        )
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


def test_closed_owner_credit_flags_are_strict_booleans() -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        ClosedOwnerCredit(
            group_index=0,
            proposal_sweep_index=1,
            resolution_sweep_index=2,
            proposal_source_fe=40,
            resolution_fe=90,
            all_groups_completed="1",
            native_sweep_end_completed=True,
            survival=0.5,
            overwrite=0.5,
        )
    with pytest.raises(ValueError, match="must follow its proposal"):
        ClosedOwnerCredit(
            group_index=0,
            proposal_sweep_index=1,
            resolution_sweep_index=2,
            proposal_source_fe=40,
            resolution_fe=40,
            all_groups_completed=True,
            native_sweep_end_completed=True,
            survival=0.5,
            overwrite=0.5,
        )

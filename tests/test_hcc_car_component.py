from __future__ import annotations

import inspect

import numpy as np
import pytest

from arac.backends.hcc_car import (
    CARRelationProposal,
    CARWritebackPlan,
    GroupOptimizationResult,
    allocate_component_horizon_budgets,
    apply_candidate_writeback,
    freeze_component_writeback_plan,
    run_component_horizon,
    shuffled_component_writeback_plan,
)
from arac.policy.counterfactual_action_racing import (
    BranchState,
    derive_probe_seed,
    fingerprint_branch_state,
)


def make_plan(*, max_delta_norm: float = 0.5) -> CARWritebackPlan:
    return CARWritebackPlan(
        graph_fingerprint="graph-a",
        component_fingerprint="component-a",
        action_name="isolate_conflicting_relation",
        action_family="isolate",
        group_indices=(0, 1),
        group_dims=((0, 1), (2, 3)),
        group_population_sizes=(4, 6),
        shared_indices=(0,),
        target_values=(0.0,),
        lower=-5.0,
        upper=5.0,
        alpha=0.20,
        max_delta_norm=max_delta_norm,
    )


def test_component_horizon_budget_is_population_complete_and_exact() -> None:
    allocation = allocate_component_horizon_budgets(
        max_arm_fes=25,
        population_sizes=(4, 6),
    )

    assert allocation == (12, 12)
    assert 1 + sum(allocation) == 25
    assert all(
        budget % population == 0
        for budget, population in zip(allocation, (4, 6), strict=True)
    )
    assert allocate_component_horizon_budgets(
        max_arm_fes=10,
        population_sizes=(4, 6),
    ) == ()


def test_candidate_writeback_uses_fixed_alpha_and_v33_norm_guard() -> None:
    plan = make_plan(max_delta_norm=0.5)
    incumbent = np.array([4.0, 1.0, 2.0, 3.0])

    adjusted, delta_norm = apply_candidate_writeback(incumbent, plan)

    assert delta_norm == pytest.approx(0.5)
    np.testing.assert_allclose(adjusted, [3.5, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(incumbent, [4.0, 1.0, 2.0, 3.0])


def test_shuffled_component_plan_preserves_targets_but_breaks_pairing() -> None:
    plan = CARWritebackPlan(
        graph_fingerprint="graph-a",
        component_fingerprint="component-a",
        action_name="allow_beneficial_coordination",
        action_family="coordinate",
        group_indices=(0, 1, 2),
        group_dims=((0,), (1,), (2,)),
        group_population_sizes=(4, 4, 4),
        shared_indices=(0, 1, 2),
        target_values=(0.1, 0.2, 0.3),
        lower=-5.0,
        upper=5.0,
    )

    shuffled = shuffled_component_writeback_plan(plan)

    assert shuffled == shuffled_component_writeback_plan(plan)
    assert shuffled.shared_indices == plan.shared_indices
    assert sorted(shuffled.target_values) == sorted(plan.target_values)
    assert shuffled.target_values != plan.target_values


class SphereEvaluator:
    def __init__(self) -> None:
        self.fitness_record: list[float] = []

    def __call__(self, x_batch):
        values = np.asarray(x_batch, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        result = np.sum(np.square(values), axis=1)
        self.fitness_record.extend(result.tolist())
        return result


def no_change_group_optimizer(
    *,
    evaluator,
    background,
    dims,
    requested_fes,
    population_size,
    seed,
) -> GroupOptimizationResult:
    assert requested_fes % population_size == 0
    batch = np.tile(np.asarray(background)[list(dims)], (requested_fes, 1))
    values = evaluator(
        np.tile(background, (requested_fes, 1))
    )
    return GroupOptimizationResult(
        best_x=batch[0],
        best_y=float(np.min(values)),
        actual_fes=requested_fes,
    )


def test_component_horizon_is_exact_and_preserves_checkpoint_payload() -> None:
    plan = make_plan(max_delta_norm=0.5)
    checkpoint = BranchState(
        incumbent=(4.0, 1.0, 2.0, 3.0),
        committed_fitness=30.0,
        evaluator_record=[],
        state_fingerprint="",
        state_payload={"rng": {"counter": 7}, "cache": {"0": 1.5}},
    )
    checkpoint.state_fingerprint = fingerprint_branch_state(checkpoint)
    seed = derive_probe_seed(
        base_seed=19,
        sweep_index=2,
        component_fingerprint=plan.component_fingerprint,
        pair_index=0,
    )
    evaluator = SphereEvaluator()

    state = run_component_horizon(
        checkpoint=checkpoint,
        evaluator=evaluator,
        seed_descriptor=seed,
        requested_fes=25,
        plan=plan,
        apply_candidate=True,
        optimize_group=no_change_group_optimizer,
    )

    assert len(evaluator.fitness_record) == 25
    assert state.committed_fitness == pytest.approx(26.25)
    np.testing.assert_allclose(state.incumbent, [3.5, 1.0, 2.0, 3.0])
    assert state.state_payload["rng"] == {"counter": 7}
    assert state.state_payload["car_component_fingerprint"] == "component-a"
    assert state.state_fingerprint == fingerprint_branch_state(state)


def test_component_horizon_rejects_non_population_complete_request() -> None:
    plan = make_plan()
    checkpoint = BranchState(
        incumbent=(4.0, 1.0, 2.0, 3.0),
        committed_fitness=30.0,
        evaluator_record=[],
        state_fingerprint="",
    )
    checkpoint.state_fingerprint = fingerprint_branch_state(checkpoint)

    with pytest.raises(ValueError, match="complete component horizon"):
        run_component_horizon(
            checkpoint=checkpoint,
            evaluator=SphereEvaluator(),
            seed_descriptor=derive_probe_seed(
                base_seed=19,
                sweep_index=2,
                component_fingerprint=plan.component_fingerprint,
                pair_index=0,
            ),
            requested_fes=24,
            plan=plan,
            apply_candidate=False,
            optimize_group=no_change_group_optimizer,
        )


def proposal(
    *,
    sweep_index: int,
    group_left: int,
    group_right: int,
    shared_indices: tuple[int, ...],
    target_values: tuple[float, ...],
    action_name: str = "allow_beneficial_coordination",
    action_family: str = "coordinate",
) -> CARRelationProposal:
    return CARRelationProposal(
        sweep_index=sweep_index,
        group_left=group_left,
        group_right=group_right,
        shared_indices=shared_indices,
        target_values=target_values,
        action_name=action_name,
        action_family=action_family,
        overlap_strength=1.0,
        feature_coverage=1.0,
        writeback_norm=0.25,
    )


def test_component_barrier_freezes_order_invariant_identity_free_plan() -> None:
    grouping = ((0, 1), (1, 2), (2, 3))
    overlaps = ((1,), (2,))
    sweep_zero = (
        proposal(
            sweep_index=0,
            group_left=0,
            group_right=1,
            shared_indices=(1,),
            target_values=(0.5,),
        ),
        proposal(
            sweep_index=0,
            group_left=1,
            group_right=2,
            shared_indices=(2,),
            target_values=(0.25,),
        ),
    )
    sweep_one = tuple(
        reversed(
            (
                proposal(
                    sweep_index=1,
                    group_left=0,
                    group_right=1,
                    shared_indices=(1,),
                    target_values=(0.4,),
                ),
                proposal(
                    sweep_index=1,
                    group_left=1,
                    group_right=2,
                    shared_indices=(2,),
                    target_values=(0.2,),
                ),
            )
        )
    )

    decision = freeze_component_writeback_plan(
        grouping_result=grouping,
        overlapping_elements=overlaps,
        group_population_sizes=(4, 4, 4),
        proposal_sweeps=(sweep_zero, sweep_one),
        lower=-5.0,
        upper=5.0,
    )

    assert decision.abstain_reason == ""
    assert decision.plan is not None
    assert decision.plan.group_indices == (0, 1, 2)
    assert decision.plan.shared_indices == (1, 2)
    assert decision.plan.target_values == pytest.approx((0.4, 0.2))
    assert decision.evidence.candidate_action_family == "coordinate"
    assert set(decision.evidence.runtime_field_names()).isdisjoint(
        {"problem_id", "seed", "case_label", "final_outcome"}
    )


def test_component_barrier_uses_stable_support_inside_full_component() -> None:
    decision = freeze_component_writeback_plan(
        grouping_result=((0, 1), (1, 2), (2, 3)),
        overlapping_elements=((1,), (2,)),
        group_population_sizes=(4, 4, 4),
        proposal_sweeps=(
            (
                proposal(
                    sweep_index=0,
                    group_left=0,
                    group_right=1,
                    shared_indices=(1,),
                    target_values=(0.5,),
                ),
                proposal(
                    sweep_index=0,
                    group_left=1,
                    group_right=2,
                    shared_indices=(2,),
                    target_values=(0.25,),
                ),
            ),
            (
                proposal(
                    sweep_index=1,
                    group_left=0,
                    group_right=1,
                    shared_indices=(1,),
                    target_values=(0.4,),
                ),
                proposal(
                    sweep_index=1,
                    group_left=1,
                    group_right=2,
                    shared_indices=(2,),
                    target_values=(0.2,),
                    action_name="conservative_no_action",
                    action_family="fallback",
                ),
            ),
        ),
        lower=-5.0,
        upper=5.0,
    )

    assert decision.plan is not None
    assert decision.plan.group_indices == (0, 1, 2)
    assert decision.plan.shared_indices == (1,)
    assert decision.plan.target_values == pytest.approx((0.4,))
    assert decision.evidence.shared_variable_count == 1


def test_component_plan_callable_has_no_identity_or_outcome_parameters() -> None:
    forbidden = {
        "case",
        "case_label",
        "problem_id",
        "seed",
        "function_family",
        "paper_best",
        "historical_best",
        "final_outcome",
        "final_error",
    }

    assert set(inspect.signature(freeze_component_writeback_plan).parameters).isdisjoint(
        forbidden
    )


@pytest.mark.parametrize(
    ("second_action_family", "expected_reason"),
    [
        ("fallback", "no_stable_non_fallback_action"),
        ("reassign_repair", "no_stable_non_fallback_action"),
    ],
)
def test_component_barrier_abstains_on_unstable_or_fallback_family(
    second_action_family: str,
    expected_reason: str,
) -> None:
    decision = freeze_component_writeback_plan(
        grouping_result=((0, 1), (1, 2)),
        overlapping_elements=((1,),),
        group_population_sizes=(4, 4),
        proposal_sweeps=(
            (
                proposal(
                    sweep_index=0,
                    group_left=0,
                    group_right=1,
                    shared_indices=(1,),
                    target_values=(0.5,),
                ),
            ),
            (
                proposal(
                    sweep_index=1,
                    group_left=0,
                    group_right=1,
                    shared_indices=(1,),
                    target_values=(0.4,),
                    action_name=(
                        "conservative_no_action"
                        if second_action_family == "fallback"
                        else "repair_shared_variable_binding"
                    ),
                    action_family=second_action_family,
                ),
            ),
        ),
        lower=-5.0,
        upper=5.0,
    )

    assert decision.plan is None
    assert decision.abstain_reason == expected_reason


def test_component_barrier_requires_two_complete_sweeps() -> None:
    decision = freeze_component_writeback_plan(
        grouping_result=((0, 1), (1, 2)),
        overlapping_elements=((1,),),
        group_population_sizes=(4, 4),
        proposal_sweeps=(
            (
                proposal(
                    sweep_index=0,
                    group_left=0,
                    group_right=1,
                    shared_indices=(1,),
                    target_values=(0.5,),
                ),
            ),
        ),
        lower=-5.0,
        upper=5.0,
    )

    assert decision.plan is None
    assert decision.abstain_reason == "insufficient_complete_evidence_sweeps"

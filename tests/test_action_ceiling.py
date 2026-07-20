from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import pytest
import numpy as np

from arac.actions.full_space_sep_cma import FULL_SPACE_SEP_CMA_ACTION
from arac.backends.hcc_action_ceiling import (
    ActionExecutionRequest,
    ContinuationResult,
    NativeContinuationState,
    OptimizationResult,
    allocate_efficiency_budgets,
    branch_horizon_errors,
    delta_priority_order,
    execute_action_ceiling_arm,
    native_eq8_values,
    run_native_group_cycle,
    run_native_continuation,
    selector_arm_for_context,
    unique_group_positions,
    update_efficiency_ewma,
)
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    CATASTROPHIC_DELTA,
    GUARDED_EQ8_PROBE_FES,
    MATERIAL_POSITIVE_DELTA,
    ActionCeilingObservation,
    FrozenProbeCandidate,
    RelationActionSet,
    actionability_delta,
    summarize_action_ceiling,
)
from arac.policy.evidence_overlay import (
    BridgeWeights,
    ProbeUtilities,
    RelationKey,
    runtime_probe_anchor_hash,
    runtime_probe_shared_values_hash,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _action_set() -> RelationActionSet:
    relation = RelationKey((0, 1), (4, 5))

    def candidate(name: str, values: tuple[float, ...], fitness: float) -> FrozenProbeCandidate:
        values_hash = (
            runtime_probe_anchor_hash(relation, values)
            if name == "anchor"
            else runtime_probe_shared_values_hash(relation, values)
        )
        return FrozenProbeCandidate(
            name=name,
            shared_values=values,
            shared_values_hash=values_hash,
            candidate_hash=_hash(name),
            fitness=fitness,
            utility=math.log(10.0 / fitness),
        )

    return RelationActionSet(
        relation=relation,
        anchor=candidate("anchor", (0.0, 0.0), 10.0),
        left_owner=candidate("left_owner", (1.0, 2.0), 8.0),
        right_owner=candidate("right_owner", (3.0, 4.0), 9.0),
        bridge=candidate("bridge", (2.0, 3.0), 7.0),
        bridge_weights=BridgeWeights(0.5, 0.5),
        probe_utilities=ProbeUtilities(math.log(10 / 8), math.log(10 / 9), math.log(10 / 7)),
        selector_winner="bridge",
        selector_utility=math.log(10 / 7),
        selector_reason="unique_probe_winner_above_one_percent",
        checkpoint_fe=100,
        checkpoint_hash=_hash("checkpoint"),
        issued_sweep=2,
        target_sweep=3,
    )


def test_relation_action_set_preserves_all_exact_candidates() -> None:
    actions = _action_set()

    assert actions.candidate_for_arm("exact_left").shared_values == (1.0, 2.0)
    assert actions.candidate_for_arm("exact_right").shared_values == (3.0, 4.0)
    assert actions.candidate_for_arm("exact_bridge").shared_values == (2.0, 3.0)
    assert len(actions.action_set_hash) == 64


def test_relation_action_set_rejects_non_local_anchor_hash() -> None:
    actions = _action_set()
    bad_anchor = FrozenProbeCandidate(
        name="anchor",
        shared_values=actions.anchor.shared_values,
        shared_values_hash=runtime_probe_shared_values_hash(
            actions.relation,
            actions.anchor.shared_values,
        ),
        candidate_hash=actions.anchor.candidate_hash,
        fitness=actions.anchor.fitness,
        utility=actions.anchor.utility,
    )

    with pytest.raises(ValueError, match="anchor hash"):
        RelationActionSet(**{**actions.__dict__, "anchor": bad_anchor})


def test_actionability_delta_uses_native_over_arm() -> None:
    assert actionability_delta(100.0, 50.0) == pytest.approx(math.log(2.0))
    assert actionability_delta(50.0, 100.0) == pytest.approx(-math.log(2.0))


def _observations(
    per_context: list[dict[str, float]],
    *,
    selector_arm: str,
) -> list[ActionCeilingObservation]:
    rows: list[ActionCeilingObservation] = []
    for index, overrides in enumerate(per_context):
        deltas = {arm: -0.01 for arm in ACTION_CEILING_ARMS}
        deltas["native_eq8"] = 0.0
        deltas.update(overrides)
        for arm in ACTION_CEILING_ARMS:
            rows.append(
                ActionCeilingObservation(
                    context_id=f"context-{index}",
                    cohort="real_aob",
                    problem_id=f"E{index + 1}",
                    seed=117,
                    arm=arm,
                    horizon="sweep_1",
                    delta=deltas[arm],
                    selector_arm=selector_arm,
                )
            )
    return rows


def test_native_wins_vbs_ties_and_is_sbs() -> None:
    summary = summarize_action_ceiling(
        _observations([{"true_no_writeback": 0.0}], selector_arm="true_no_writeback"),
        cohort="real_aob",
        bootstrap_replicates=20,
    )

    assert summary["sbs_arm"] == "native_eq8"
    assert summary["vbs_mean_delta"] == 0.0
    assert summary["recommendation"] == "redesign_actions"


def test_positive_vbs_but_negative_selector_upgrades_evidence() -> None:
    rows = _observations(
        [
            {"exact_left": MATERIAL_POSITIVE_DELTA + 0.1},
            {"exact_right": MATERIAL_POSITIVE_DELTA + 0.1},
            {"exact_bridge": MATERIAL_POSITIVE_DELTA + 0.1},
            {"efficiency_budget_reallocation": MATERIAL_POSITIVE_DELTA + 0.1},
        ],
        selector_arm="true_no_writeback",
    )
    for row_index, row in enumerate(rows):
        if row.arm == "true_no_writeback":
            rows[row_index] = ActionCeilingObservation(
                **{**row.__dict__, "delta": -0.01}
            )
    summary = summarize_action_ceiling(
        rows,
        cohort="real_aob",
        bootstrap_replicates=100,
    )

    assert summary["vbs_lcb"] > 0.0
    assert summary["selector_lcb"] < 0.0
    assert summary["recommendation"] == "upgrade_evidence"


def test_catastrophic_selector_blocks_runtime_validation() -> None:
    rows = _observations(
        [
            {"exact_left": 0.1},
            {"exact_left": 0.1},
            {"exact_left": 0.1},
            {"exact_left": 0.1},
        ],
        selector_arm="exact_left",
    )
    first_context = "context-0"
    for index, row in enumerate(rows):
        if row.context_id == first_context and row.arm == "exact_left":
            rows[index] = ActionCeilingObservation(
                **{**row.__dict__, "delta": CATASTROPHIC_DELTA - 0.01}
            )
    summary = summarize_action_ceiling(
        rows,
        cohort="real_aob",
        bootstrap_replicates=100,
    )

    assert summary["selector_catastrophic_count"] == 1
    assert summary["selector_catastrophic_rate"] == pytest.approx(0.25)
    assert summary["recommendation"] == "upgrade_evidence"


OWNER_GROUP_DIMENSIONS = ((0, 1, 4, 5), (2, 3, 4, 5))


def _owner_optimizer_means(
    incumbent: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return tuple(
        tuple(float(incumbent[index]) for index in dimensions)
        for dimensions in OWNER_GROUP_DIMENSIONS
    )


def _request(
    arm: str,
    *,
    dimension: int = 8,
    incumbent_shared: tuple[float, float] = (8.0, 9.0),
    owner_optimizer_means: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> ActionExecutionRequest:
    actions = _action_set()
    incumbent = np.arange(dimension, dtype=float)
    incumbent[4:6] = incumbent_shared
    return ActionExecutionRequest(
        arm=arm,
        context_hash=_hash("context"),
        action_set=actions,
        incumbent=tuple(incumbent),
        incumbent_fitness=100.0,
        previous_values=(6.0, 7.0),
        current_values=incumbent_shared,
        previous_delta=1.0,
        current_delta=3.0,
        owner_group_dimensions=OWNER_GROUP_DIMENSIONS,
        owner_optimizer_means=(
            _owner_optimizer_means(incumbent)
            if owner_optimizer_means is None
            else owner_optimizer_means
        ),
    )


def test_native_eq8_and_true_no_writeback_are_distinct() -> None:
    no_writeback = execute_action_ceiling_arm(
        _request("true_no_writeback"),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )
    native = execute_action_ceiling_arm(
        _request("native_eq8"),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    assert no_writeback.incumbent[4:6] == (8.0, 9.0)
    np.testing.assert_allclose(native.incumbent[4:6], native_eq8_values((6, 7), (8, 9), 1, 3))
    assert no_writeback.counterfactual_applied is False
    assert no_writeback.mutation_norm == 0.0
    assert no_writeback.optimizer_mean_mutation_norm == 0.0
    assert no_writeback.owner_optimizer_means == ((0.0, 1.0, 8.0, 9.0), (2.0, 3.0, 8.0, 9.0))
    assert native.counterfactual_applied is True
    assert native.optimizer_mean_mutation_norm == 0.0
    assert native.owner_optimizer_means == no_writeback.owner_optimizer_means


def test_runner_legacy_no_action_is_native_and_explicit_noop_is_none() -> None:
    from scripts import hcc_smoke_runner as runner

    previous = np.asarray([1.0, 2.0])
    current = np.asarray([3.0, 4.0])
    legacy = runner.apply_arac_overlap_action(
        "conservative_no_action", previous, current, 1.0, 3.0
    )
    explicit = runner.apply_arac_overlap_action(
        runner.NATIVE_EQ8_ACTION, previous, current, 1.0, 3.0
    )
    no_writeback = runner.apply_arac_overlap_action(
        runner.TRUE_NO_WRITEBACK_ACTION, previous, current, 1.0, 3.0
    )

    np.testing.assert_allclose(legacy, explicit)
    assert no_writeback is None


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("exact_left", (1.0, 2.0)),
        ("exact_right", (3.0, 4.0)),
        ("exact_bridge", (2.0, 3.0)),
    ],
)
def test_exact_arms_ignore_phase2_delta(arm: str, expected: tuple[float, ...]) -> None:
    result = execute_action_ceiling_arm(
        _request(arm),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    assert result.incumbent[4:6] == expected
    assert result.owner_optimizer_means == (
        (0.0, 1.0, *expected),
        (2.0, 3.0, *expected),
    )


def test_shared_writeback_repairs_stale_owner_means_even_without_incumbent_delta() -> None:
    result = execute_action_ceiling_arm(
        _request(
            "exact_right",
            incumbent_shared=(3.0, 4.0),
            owner_optimizer_means=(
                (10.0, 11.0, -1.0, -2.0),
                (20.0, 21.0, -3.0, -4.0),
            ),
        ),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    assert result.mutation_norm == 0.0
    assert result.optimizer_mean_mutation_norm > 0.0
    assert result.counterfactual_applied is True
    assert result.owner_optimizer_means == (
        (10.0, 11.0, 3.0, 4.0),
        (20.0, 21.0, 3.0, 4.0),
    )


@pytest.mark.parametrize(
    "arm",
    [
        "efficiency_budget_reallocation",
        "delta_priority_scan",
        "stagnation_cross_group_warm_start",
        FULL_SPACE_SEP_CMA_ACTION,
    ],
)
def test_continuation_arms_reuse_native_eq8_at_dispatch(arm: str) -> None:
    result = execute_action_ceiling_arm(
        _request(arm),
        evaluate=lambda batch: np.zeros(len(batch)),
    )

    assert result.selected_candidate == arm
    assert result.action_actual_fes == 0
    np.testing.assert_allclose(
        result.incumbent[4:6],
        native_eq8_values((6.0, 7.0), (8.0, 9.0), 1.0, 3.0),
    )
    assert result.counterfactual_applied is True
    assert result.mutation_norm > 0.0
    assert result.optimizer_mean_mutation_norm == 0.0


def test_efficiency_ewma_and_budget_allocation_preserve_frozen_total() -> None:
    ewma = update_efficiency_ewma(
        (0.0, 0.0, 0.0),
        (90.0, 9.0, 0.0),
        (10, 10, 10),
    )
    budgets = allocate_efficiency_budgets(
        ewma,
        uniform_budgets=(10, 10, 10),
        population_sizes=(4, 4, 4),
    )

    assert ewma == pytest.approx((2.7, 0.27, 0.0))
    assert sum(budgets) == 30
    assert budgets[0] > budgets[1] > budgets[2]
    assert all(4 <= budget <= 30 for budget in budgets)
    assert allocate_efficiency_budgets(
        (0.0, 0.0, 0.0),
        (10, 10, 10),
        (4, 4, 4),
    ) == (10, 10, 10)


def test_delta_priority_order_is_stable_by_original_group_index() -> None:
    assert delta_priority_order((0.1, 2.0, 2.0, 0.0)) == (1, 2, 0, 3)


def test_unique_group_positions_exclude_both_neighbor_overlaps() -> None:
    assert unique_group_positions(
        ((0, 3), (1, 3, 4), (2, 4)),
        ((3,), (4,)),
    ) == ((0,), (0,), (0,))


def _completed_continuation_state(
    *,
    deltas: tuple[float, float, float] = (90.0, 9.0, 0.0),
    efficiency_ewma: tuple[float, float, float] = (0.0, 0.0, 0.0),
    stagnation_streaks: tuple[int, int, int] = (0, 0, 0),
    stagnation_cooldowns: tuple[int, int, int] = (0, 0, 0),
) -> NativeContinuationState:
    return NativeContinuationState(
        incumbent=(1.0, 2.0, 3.0, 5.0, 6.0),
        sweep_index=3,
        next_group_index=3,
        completed_group_deltas=deltas,
        completed_group_actual_fes=(7, 7, 7),
        group_dims=((0, 3), (1, 3, 4), (2, 4)),
        overlapping_elements=((3,), (4,)),
        population_sizes=(2, 2, 2),
        optimizer_budgets=(6, 6, 6),
        efficiency_ewma=efficiency_ewma,
        completed_efficiency_sweeps=3,
        stagnation_streaks=stagnation_streaks,
        stagnation_cooldowns=stagnation_cooldowns,
        lower_bound=-100.0,
        upper_bound=100.0,
        sigma=0.5,
    )


def _run_continuation_arm(
    arm: str,
    state: NativeContinuationState,
    *,
    target_relative_fe: int | None = None,
) -> tuple[ContinuationResult, list[tuple[int, tuple[float, ...]]]]:
    record: list[float] = []
    observed_means: list[tuple[int, tuple[float, ...]]] = []

    def evaluate(values: np.ndarray) -> np.ndarray:
        rows = np.asarray(values, dtype=float)
        if rows.ndim == 1:
            rows = rows[None, :]
        result = np.sum(np.square(rows), axis=1)
        record.extend(float(value) for value in result)
        return result

    def optimize(**kwargs) -> OptimizationResult:
        background = np.asarray(kwargs["background"], dtype=float)
        dims = tuple(int(value) for value in kwargs["dims"])
        mean = np.asarray(kwargs["mean"], dtype=float)
        observed_means.append((int(kwargs["group_index"]), tuple(mean)))
        batch = np.repeat(background[None, :], kwargs["requested_fes"], axis=0)
        batch[:, np.asarray(dims)] = mean
        values = evaluate(batch)
        return OptimizationResult(tuple(mean), float(np.min(values)), len(values))

    result = run_native_continuation(
        state,
        evaluate=evaluate,
        fitness_record=record,
        optimize_group=optimize,
        group_seed=lambda sweep, group: sweep * 100 + group,
        target_relative_fe=(
            state.sweep_horizon_fe
            if target_relative_fe is None
            else target_relative_fe
        ),
        continuation_arm=arm,
    )
    return result, observed_means


@pytest.mark.parametrize(
    "arm",
    [
        "efficiency_budget_reallocation",
        "delta_priority_scan",
        "stagnation_cross_group_warm_start",
    ],
)
def test_continuation_actions_wait_for_next_complete_sweep(arm: str) -> None:
    state = replace(
        _completed_continuation_state(stagnation_streaks=(3, 3, 3)),
        next_group_index=1,
        completed_group_deltas=(90.0,),
        completed_group_actual_fes=(7,),
    )
    result, _ = _run_continuation_arm(
        arm,
        state,
        target_relative_fe=7,
    )

    assert result.execution_order_trace == (1,)
    assert result.group_budget_trace == (6,)
    assert result.continuation_policy_applied is False
    assert result.warm_start_trigger_count == 0


def test_efficiency_budget_arm_changes_budgets_only_after_sweep_boundary() -> None:
    state = _completed_continuation_state()
    result, _ = _run_continuation_arm("efficiency_budget_reallocation", state)

    assert result.continuation_policy_applied is True
    assert result.execution_order_trace == (0, 1, 2)
    assert sum(result.group_budget_trace) == sum(state.optimizer_budgets)
    assert result.group_budget_trace[0] > result.group_budget_trace[1]
    assert result.group_budget_trace[2] == state.population_sizes[2]


def test_delta_priority_scan_changes_call_order_but_keeps_original_indices() -> None:
    state = _completed_continuation_state(deltas=(1.0, 5.0, 2.0))
    result, observed = _run_continuation_arm("delta_priority_scan", state)

    assert result.continuation_policy_applied is True
    assert result.execution_order_trace == (1, 2, 0)
    assert tuple(group for group, _mean in observed) == (1, 2, 0)
    assert result.group_budget_trace == (6, 6, 6)


def test_stagnation_warm_start_perturbs_only_unique_mean_positions() -> None:
    state = _completed_continuation_state(stagnation_streaks=(3, 0, 0))
    result, observed = _run_continuation_arm(
        "stagnation_cross_group_warm_start",
        state,
    )

    first_group, first_mean = observed[0]
    assert first_group == 0
    assert first_mean[0] != state.incumbent[0]
    assert first_mean[1] == state.incumbent[3]
    assert result.continuation_policy_applied is True
    assert result.warm_start_trigger_count == 1
    assert result.warm_start_mean_shift_norm > 0.0
    assert result.stagnation_cooldowns[0] == 3


def test_stagnation_warm_start_respects_group_cooldown() -> None:
    state = _completed_continuation_state(
        stagnation_streaks=(3, 0, 0),
        stagnation_cooldowns=(1, 0, 0),
    )
    result, observed = _run_continuation_arm(
        "stagnation_cross_group_warm_start",
        state,
    )

    assert observed[0][1] == (state.incumbent[0], state.incumbent[3])
    assert result.warm_start_trigger_count == 0
    assert result.stagnation_cooldowns[0] == 0


def test_selector_mismatch_maps_to_true_no_writeback() -> None:
    actions = _action_set()
    assert selector_arm_for_context(
        actions,
        relation=actions.relation,
        current_sweep=actions.target_sweep,
        checkpoint_hash=actions.checkpoint_hash,
        current_shared_values=actions.anchor.shared_values,
    ) == "exact_bridge"
    assert selector_arm_for_context(
        actions,
        relation=actions.relation,
        current_sweep=actions.target_sweep,
        checkpoint_hash=actions.checkpoint_hash,
        current_shared_values=(9.0, 9.0),
    ) == "true_no_writeback"


def test_native_continuation_uses_frozen_budgets_and_eq8() -> None:
    record: list[float] = []

    def evaluate(values: np.ndarray) -> np.ndarray:
        rows = np.asarray(values, dtype=float)
        if rows.ndim == 1:
            rows = rows[None, :]
        result = np.sum(np.square(rows), axis=1)
        record.extend(float(value) for value in result)
        return result

    def optimize(**kwargs) -> OptimizationResult:
        background = kwargs["background"]
        dims = kwargs["dims"]
        candidate = np.zeros(len(dims))
        batch = np.repeat(background[None, :], kwargs["requested_fes"], axis=0)
        batch[:, np.asarray(dims)] = candidate
        values = evaluate(batch)
        return OptimizationResult(tuple(candidate), float(np.min(values)), len(values))

    state = NativeContinuationState(
        incumbent=(2.0, 3.0, 4.0),
        sweep_index=1,
        next_group_index=1,
        completed_group_deltas=(1.0,),
        completed_group_actual_fes=(3,),
        group_dims=((0, 1), (1, 2)),
        overlapping_elements=((1,),),
        population_sizes=(2, 2),
        optimizer_budgets=(2, 2),
        efficiency_ewma=(0.0, 0.0),
        completed_efficiency_sweeps=1,
        stagnation_streaks=(0, 0),
        stagnation_cooldowns=(0, 0),
        lower_bound=-100.0,
        upper_bound=100.0,
        sigma=0.5,
    )
    result = run_native_continuation(
        state,
        evaluate=evaluate,
        fitness_record=record,
        optimize_group=optimize,
        group_seed=lambda sweep, group: sweep * 10 + group,
        target_relative_fe=state.sweep_horizon_fe,
    )

    assert len(result.fitness_record) >= state.sweep_horizon_fe
    assert result.sweep_index >= 2
    assert len(record) == len(result.fitness_record)


def test_native_group_cycle_uses_actual_fe_when_optimizer_stops_early() -> None:
    record: list[float] = []

    def evaluate(values: np.ndarray) -> np.ndarray:
        rows = np.asarray(values, dtype=float)
        if rows.ndim == 1:
            rows = rows[None, :]
        result = np.sum(np.square(rows), axis=1)
        record.extend(float(value) for value in result)
        return result

    def optimize(**kwargs) -> OptimizationResult:
        background = kwargs["background"]
        dims = kwargs["dims"]
        actual_fes = kwargs["population_size"]
        batch = np.repeat(background[None, :], actual_fes, axis=0)
        values = evaluate(batch)
        return OptimizationResult(
            tuple(float(background[index]) for index in dims),
            float(np.min(values)),
            actual_fes,
        )

    state = NativeContinuationState(
        incumbent=(2.0, 3.0, 4.0),
        sweep_index=1,
        next_group_index=1,
        completed_group_deltas=(1.0,),
        completed_group_actual_fes=(3,),
        group_dims=((0, 1), (1, 2)),
        overlapping_elements=((1,),),
        population_sizes=(2, 2),
        optimizer_budgets=(4, 4),
        efficiency_ewma=(0.0, 0.0),
        completed_efficiency_sweeps=1,
        stagnation_streaks=(0, 0),
        stagnation_cooldowns=(0, 0),
        lower_bound=-100.0,
        upper_bound=100.0,
        sigma=0.5,
    )
    result = run_native_group_cycle(
        state,
        evaluate=evaluate,
        fitness_record=record,
        optimize_group=optimize,
        group_seed=lambda sweep, group: sweep * 10 + group,
    )

    assert len(result.fitness_record) == 6
    assert result.sweep_index == 2
    assert result.next_group_index == 1
    assert len(result.completed_group_deltas) == 1


def test_horizon_errors_charge_action_fes_before_native_continuation() -> None:
    errors = branch_horizon_errors(
        prefix_best_error=100.0,
        post_checkpoint_record=[90.0, 80.0, 70.0, 60.0, 50.0, 40.0],
        sweep_horizon_fe=2,
    )

    assert errors == {"immediate": 90.0, "sweep_1": 80.0, "sweep_3": 40.0}


def test_guarded_eq8_writeback_probes_candidates_and_charges_same_horizon() -> None:
    result = execute_action_ceiling_arm(
        _request("guarded_eq8_writeback"),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    # sum-of-squares probes: previous = 184.0, eq8_blend = 227.5; current = 100.0
    assert result.selected_candidate == "current"
    assert result.action_accepted is False
    assert result.action_budget_fes == GUARDED_EQ8_PROBE_FES
    assert result.action_actual_fes == GUARDED_EQ8_PROBE_FES
    assert result.incumbent[4:6] == (8.0, 9.0)
    assert result.mutation_norm == 0.0
    assert result.optimizer_mean_mutation_norm == 0.0
    assert result.counterfactual_applied is True
    assert len(result.action_instance_hash) == 64
    assert len(result.action_lifecycle_hash) == 64
    payload = json.loads(result.action_lifecycle_payload)
    assert payload["action_actual_fes"] == GUARDED_EQ8_PROBE_FES
    assert payload["selected_candidate"] == "current"
    assert payload["instance_hash"] == result.action_instance_hash
    assert payload["instance"]["parameter_hash"] == _hash_payload(
        payload["instance"]["parameters"]
    )
    assert [candidate["name"] for candidate in payload["probe_outcomes"]] == [
        "current",
        "previous",
        "eq8_blend",
    ]
    assert result.action_candidate_fitness == pytest.approx(100.0)
    assert len(result.action_candidate_hash) == 64
    assert len(result.action_post_incumbent_hash) == 64
    assert result.optimizer_scope == "relation_writeback"


def test_guarded_eq8_writeback_accepts_better_previous_and_syncs_owner_means() -> None:
    request = replace(_request("guarded_eq8_writeback"), incumbent_fitness=1000.0)
    result = execute_action_ceiling_arm(
        request,
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    # current = 1000.0 (known), previous probe = 184.0, blend probe = 227.5
    assert result.selected_candidate == "previous"
    assert result.action_accepted is True
    assert result.incumbent[4:6] == (6.0, 7.0)
    assert result.owner_optimizer_means == ((0.0, 1.0, 6.0, 7.0), (2.0, 3.0, 6.0, 7.0))
    assert result.action_candidate_fitness == pytest.approx(184.0)
    assert result.mutation_norm > 0.0


def test_stagnation_guard_writeback_abstains_on_zero_delta_sum() -> None:
    request = replace(
        _request("stagnation_guard_writeback"),
        previous_delta=0.0,
        current_delta=0.0,
    )
    result = execute_action_ceiling_arm(
        request,
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    # native Eq.8 would have written the unaudited mean (7.0, 8.0) here
    assert result.selected_candidate == "current"
    assert result.incumbent[4:6] == (8.0, 9.0)
    assert result.action_actual_fes == 0
    assert len(result.action_instance_hash) == 64
    assert len(result.action_lifecycle_hash) == 64
    assert len(result.action_candidate_hash) == 64
    assert len(result.action_post_incumbent_hash) == 64
    assert result.counterfactual_applied is False
    assert result.mutation_norm == 0.0


def test_stagnation_guard_writeback_matches_native_outside_stagnation() -> None:
    result = execute_action_ceiling_arm(
        _request("stagnation_guard_writeback"),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )
    native = execute_action_ceiling_arm(
        _request("native_eq8"),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    assert result.selected_candidate == "native_eq8"
    assert result.incumbent == native.incumbent
    assert result.owner_optimizer_means == native.owner_optimizer_means
    assert result.action_actual_fes == 0
    assert result.action_accepted is True
    assert json.loads(result.action_lifecycle_payload)["selected_candidate"] == (
        "native_eq8"
    )


def test_stagnation_guard_requires_both_owner_deltas_to_be_zero() -> None:
    request = replace(
        _request("stagnation_guard_writeback"),
        previous_delta=1.0,
        current_delta=-1.0,
    )

    result = execute_action_ceiling_arm(
        request,
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    assert result.selected_candidate == "native_eq8"
    assert result.action_accepted is True
    assert result.incumbent[4:6] == (7.0, 8.0)


def test_contribution_owner_writeback_follows_larger_delta_and_syncs_means() -> None:
    result = execute_action_ceiling_arm(
        _request(
            "contribution_owner_writeback",
            owner_optimizer_means=(
                (10.0, 11.0, -1.0, -2.0),
                (20.0, 21.0, -3.0, -4.0),
            ),
        ),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    # current_delta 3.0 > previous_delta 1.0: the larger-delta (right) owner wins
    assert result.selected_candidate == "right_owner"
    assert result.incumbent[4:6] == (8.0, 9.0)
    assert result.mutation_norm == 0.0
    assert result.owner_optimizer_means == (
        (10.0, 11.0, 8.0, 9.0),
        (20.0, 21.0, 8.0, 9.0),
    )
    assert result.optimizer_mean_mutation_norm > 0.0
    assert result.counterfactual_applied is True
    assert result.action_actual_fes == 0
    assert result.action_accepted is True
    assert len(result.action_instance_hash) == 64
    assert len(result.action_lifecycle_hash) == 64


def test_contribution_owner_writeback_left_win_and_tie_abstain() -> None:
    left_win = execute_action_ceiling_arm(
        replace(_request("contribution_owner_writeback"), previous_delta=5.0),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )
    assert left_win.selected_candidate == "left_owner"
    assert left_win.incumbent[4:6] == (6.0, 7.0)

    tie = execute_action_ceiling_arm(
        replace(_request("contribution_owner_writeback"), previous_delta=3.0),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )
    assert tie.selected_candidate == "current"
    assert tie.incumbent[4:6] == (8.0, 9.0)
    assert tie.counterfactual_applied is False


def test_contribution_owner_reverse_writeback_is_directional_control() -> None:
    result = execute_action_ceiling_arm(
        _request("contribution_owner_reverse_writeback"),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )

    # the reverse arm takes the smaller-delta owner (previous, 1.0 < 3.0)
    assert result.selected_candidate == "left_owner"
    assert result.incumbent[4:6] == (6.0, 7.0)

    tie = execute_action_ceiling_arm(
        replace(_request("contribution_owner_reverse_writeback"), previous_delta=3.0),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
    )
    assert tie.selected_candidate == "current"
    assert tie.counterfactual_applied is False

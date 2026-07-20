from __future__ import annotations

import hashlib
import math

import pytest
import numpy as np

from arac.backends.hcc_action_ceiling import (
    ActionExecutionRequest,
    NativeContinuationState,
    OptimizationResult,
    branch_horizon_errors,
    execute_action_ceiling_arm,
    native_eq8_values,
    run_native_group_cycle,
    run_native_continuation,
    selector_arm_for_context,
)
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    CATASTROPHIC_DELTA,
    MATERIAL_POSITIVE_DELTA,
    ActionCeilingObservation,
    FrozenProbeCandidate,
    RelationActionSet,
    RelationCredit,
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
            {"multi_context_winner": MATERIAL_POSITIVE_DELTA + 0.1},
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


def test_multi_context_winner_picks_best_and_writes_back() -> None:
    actions = _action_set()
    dimension = 8
    incumbent = np.arange(dimension, dtype=float)
    incumbent[4:6] = (8.0, 9.0)
    left_bg = incumbent.copy()
    right_bg = incumbent.copy()
    request = ActionExecutionRequest(
        arm="multi_context_winner",
        context_hash=_hash("ctx"),
        action_set=actions,
        incumbent=tuple(incumbent),
        incumbent_fitness=100.0,
        previous_values=(6.0, 7.0),
        current_values=(8.0, 9.0),
        previous_delta=1.0,
        current_delta=3.0,
        owner_group_dimensions=OWNER_GROUP_DIMENSIONS,
        owner_optimizer_means=_owner_optimizer_means(incumbent),
        left_background=tuple(left_bg),
        right_background=tuple(right_bg),
    )

    eval_calls: list[int] = []

    def evaluate(batch: np.ndarray) -> np.ndarray:
        eval_calls.append(len(batch))
        # exact_left (index 1) has lowest fitness in both contexts
        return np.asarray([100.0, 60.0, 80.0, 75.0])

    result = execute_action_ceiling_arm(request, evaluate=evaluate)

    assert len(eval_calls) == 2
    assert result.extra_fes == 8
    assert result.selected_candidate == "exact_left"
    assert result.incumbent[4:6] == (1.0, 2.0)
    assert result.owner_optimizer_means == (
        (0.0, 1.0, 1.0, 2.0),
        (2.0, 3.0, 1.0, 2.0),
    )
    assert result.counterfactual_applied is True


def test_multi_context_winner_falls_back_to_current_when_no_improvement() -> None:
    actions = _action_set()
    dimension = 8
    incumbent = np.arange(dimension, dtype=float)
    incumbent[4:6] = (8.0, 9.0)
    request = ActionExecutionRequest(
        arm="multi_context_winner",
        context_hash=_hash("ctx"),
        action_set=actions,
        incumbent=tuple(incumbent),
        incumbent_fitness=100.0,
        previous_values=(6.0, 7.0),
        current_values=(8.0, 9.0),
        previous_delta=1.0,
        current_delta=3.0,
        owner_group_dimensions=OWNER_GROUP_DIMENSIONS,
        owner_optimizer_means=_owner_optimizer_means(incumbent),
        left_background=tuple(incumbent),
        right_background=tuple(incumbent),
    )

    # current (index 0) wins in both contexts — no improvement over itself
    result = execute_action_ceiling_arm(
        request,
        evaluate=lambda batch: np.asarray([50.0, 80.0, 90.0, 70.0]),
    )

    assert result.selected_candidate == "current"
    assert result.counterfactual_applied is False


def test_initialization_bias_charges_no_fes_and_names_winner() -> None:
    eval_calls: list[int] = []
    result = execute_action_ceiling_arm(
        _request("initialization_bias"),
        evaluate=lambda batch: (eval_calls.append(len(batch)), None)[1],
    )

    assert eval_calls == []
    assert result.extra_fes == 0
    assert result.selected_candidate == "bias_bridge"
    assert result.counterfactual_applied is True
    assert result.mutation_norm == 0.0
    assert result.optimizer_mean_mutation_norm > 0.0
    assert result.owner_optimizer_means == (
        (0.0, 1.0, 2.0, 3.0),
        (2.0, 3.0, 2.0, 3.0),
    )


def test_delayed_sweep_reconciliation_cold_start_no_writeback() -> None:
    result = execute_action_ceiling_arm(
        _request("delayed_sweep_reconciliation"),
        evaluate=lambda batch: np.zeros(len(batch)),
    )

    assert result.selected_candidate == "cold_start_no_writeback"
    assert result.counterfactual_applied is False


def test_delayed_sweep_reconciliation_warm_applies_last_winner() -> None:
    credit = RelationCredit(
        ewma_credit=0.05,
        last_winner="left_owner",
        n_sweeps=5,
        last_updated_sweep=4,
    )
    dimension = 8
    incumbent = np.arange(dimension, dtype=float)
    incumbent[4:6] = (8.0, 9.0)
    actions = _action_set()
    request = ActionExecutionRequest(
        arm="delayed_sweep_reconciliation",
        context_hash=_hash("ctx"),
        action_set=actions,
        incumbent=tuple(incumbent),
        incumbent_fitness=100.0,
        previous_values=(6.0, 7.0),
        current_values=(8.0, 9.0),
        previous_delta=1.0,
        current_delta=3.0,
        owner_group_dimensions=OWNER_GROUP_DIMENSIONS,
        owner_optimizer_means=_owner_optimizer_means(incumbent),
        relation_credit=credit,
    )

    result = execute_action_ceiling_arm(
        request,
        evaluate=lambda batch: np.zeros(len(batch)),
    )

    assert result.selected_candidate == "delayed_left_owner"
    assert result.incumbent[4:6] == (1.0, 2.0)
    assert result.owner_optimizer_means == (
        (0.0, 1.0, 1.0, 2.0),
        (2.0, 3.0, 1.0, 2.0),
    )
    assert result.counterfactual_applied is True


def test_delayed_sweep_reconciliation_negative_credit_no_writeback() -> None:
    credit = RelationCredit(
        ewma_credit=-0.01,
        last_winner="left_owner",
        n_sweeps=5,
        last_updated_sweep=4,
    )
    result = execute_action_ceiling_arm(
        ActionExecutionRequest(
            arm="delayed_sweep_reconciliation",
            context_hash=_hash("ctx"),
            action_set=_action_set(),
            incumbent=tuple(np.arange(8, dtype=float)),
            incumbent_fitness=100.0,
            previous_values=(6.0, 7.0),
            current_values=(8.0, 9.0),
            previous_delta=1.0,
            current_delta=3.0,
            owner_group_dimensions=OWNER_GROUP_DIMENSIONS,
            owner_optimizer_means=_owner_optimizer_means(np.arange(8, dtype=float)),
            relation_credit=credit,
        ),
        evaluate=lambda batch: np.zeros(len(batch)),
    )

    assert result.selected_candidate == "credit_negative_no_writeback"
    assert result.counterfactual_applied is False


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
        group_dims=((0, 1), (1, 2)),
        overlapping_elements=((1,),),
        population_sizes=(2, 2),
        optimizer_budgets=(2, 2),
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
        group_dims=((0, 1), (1, 2)),
        overlapping_elements=((1,),),
        population_sizes=(2, 2),
        optimizer_budgets=(4, 4),
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


def test_horizon_errors_charge_extra_fes_before_native_continuation() -> None:
    errors = branch_horizon_errors(
        prefix_best_error=100.0,
        post_checkpoint_record=[90.0, 80.0, 70.0, 60.0, 50.0, 40.0],
        sweep_horizon_fe=2,
    )

    assert errors == {"immediate": 90.0, "sweep_1": 80.0, "sweep_3": 40.0}

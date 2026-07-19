from __future__ import annotations

import hashlib
import math

import pytest
import numpy as np

from arac.backends.hcc_action_ceiling import (
    FULL_SPACE_RESCUE_FE,
    ActionExecutionRequest,
    NativeContinuationState,
    OptimizationResult,
    branch_horizon_errors,
    execute_action_ceiling_arm,
    native_eq8_values,
    run_native_group_cycle,
    run_native_continuation,
    selector_arm_for_context,
    shared_population_size,
)
from arac.policy.action_ceiling import (
    ACTION_CEILING_ARMS,
    CATASTROPHIC_DELTA,
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
            {"shared_trust_region": MATERIAL_POSITIVE_DELTA + 0.1},
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


def _request(arm: str, *, dimension: int = 8) -> ActionExecutionRequest:
    actions = _action_set()
    incumbent = np.arange(dimension, dtype=float)
    incumbent[4:6] = (8.0, 9.0)
    return ActionExecutionRequest(
        arm=arm,
        context_hash=_hash("context"),
        action_set=actions,
        incumbent=tuple(incumbent),
        incumbent_fitness=100.0,
        previous_values=(6.0, 7.0),
        current_values=(8.0, 9.0),
        previous_delta=1.0,
        current_delta=3.0,
        lower=-100.0,
        upper=100.0,
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
    assert no_writeback.counterfactual_applied is True
    assert no_writeback.mutation_norm == 0.0
    assert native.counterfactual_applied is True


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


def test_reprobe_consumes_four_fes_and_uses_exact_best() -> None:
    calls: list[int] = []

    def evaluate(values: np.ndarray) -> np.ndarray:
        calls.append(len(values))
        return np.asarray([10.0, 5.0, 7.0, 6.0])

    result = execute_action_ceiling_arm(_request("reprobe_then_exact"), evaluate=evaluate)

    assert calls == [4]
    assert result.extra_fes == 4
    assert result.selected_candidate == "exact_left"
    assert result.incumbent[4:6] == (1.0, 2.0)


def test_shared_search_changes_only_shared_coordinates() -> None:
    def optimize(**kwargs) -> OptimizationResult:
        assert kwargs["requested_fes"] == 4 * shared_population_size(2)
        return OptimizationResult((0.5, 0.75), 50.0, kwargs["requested_fes"])

    request = _request("shared_trust_region")
    result = execute_action_ceiling_arm(
        request,
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
        shared_optimizer=optimize,
    )

    before = np.asarray(request.incumbent)
    after = np.asarray(result.incumbent)
    np.testing.assert_array_equal(np.delete(after, (4, 5)), np.delete(before, (4, 5)))
    np.testing.assert_allclose(after[4:6], (0.5, 0.75))
    assert result.extra_fes == 4 * shared_population_size(2)


def test_full_space_rescue_is_1000d_and_charges_97_fes() -> None:
    observed: dict[str, int] = {}

    def optimize(**kwargs) -> OptimizationResult:
        observed["dimension"] = len(kwargs["mean"])
        observed["population"] = kwargs["population_size"]
        best = np.asarray(kwargs["mean"]) * 0.5
        return OptimizationResult(tuple(best), 50.0, kwargs["requested_fes"])

    result = execute_action_ceiling_arm(
        _request("non_decomposition_rescue", dimension=1000),
        evaluate=lambda values: np.sum(np.square(values), axis=-1),
        full_optimizer=optimize,
    )

    assert observed == {"dimension": 1000, "population": 24}
    assert result.extra_fes == FULL_SPACE_RESCUE_FE == 97
    assert len(result.incumbent) == 1000


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

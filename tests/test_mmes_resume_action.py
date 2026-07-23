from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


HCC_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "hcc"
if str(HCC_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(HCC_VENDOR_ROOT))

from HCC.NDAs.MMES.mmes import MMES  # noqa: E402
from HCC.NDAs.MMES.state import MMESState  # noqa: E402

from arac.actions.mmes_resume import (  # noqa: E402
    MMES_RUN_BLOCK_REFERENCE_VERSION,
    PHASE1_MMES_RESUME_ACTION_SPEC,
    PHASE1_MMES_RESUME_SCHEMA_VERSION,
    FrozenMmesState,
    MmesResumeExecutionResult,
    MmesResumeExecutionContext,
    MmesResumeRejectionResult,
    MmesResumeExecutionState,
    Phase1MmesResumeAction,
    canonical_mmes_parameters,
    canonical_mmes_state_hash,
    execute_phase1_mmes_resume_action,
    mmes_resume_anchor_hash,
)


CHECKPOINT_HASH = "a" * 64
SEED_NAMESPACE = "phase1_mmes_resume:test"


def _rng_state(seed: int) -> dict[str, object]:
    return copy.deepcopy(np.random.default_rng(seed).bit_generator.state)


def _state(ndim: int = 4, population: int = 4) -> MMESState:
    mean = np.linspace(-0.5, 0.5, ndim)
    return MMESState(
        x=np.arange(population * ndim, dtype=float).reshape(population, ndim),
        mean=mean.copy(),
        p=np.zeros((1, ndim), dtype=float),
        w=0.25,
        q=np.zeros((4, ndim), dtype=float),
        t=np.arange(4, dtype=float),
        v=np.arange(4, dtype=int),
        y=np.linspace(10.0, 5.0, population),
        sigma=0.75,
        sigma_bak=1.0,
        initial_mean=mean.reshape(1, -1),
        n_individuals=population,
        n_parents=population // 2,
        n_mirror_sampling=(population + 1) // 2,
        n_generations=3,
        n_restart=1,
        list_generations=[2],
        list_fitness=[float("inf"), 8.0, 5.0],
        list_initial_mean=[np.ones(ndim)],
        best_so_far_x=np.zeros(ndim),
        best_so_far_y=2.0,
        n_function_evaluations=13,
        termination_signal=1,
        fitness=[10.0, 8.0, 5.0],
        recent_best=[(1, 10.0), (7, 5.0), (13, 2.0)],
        rng_initialization_state=_rng_state(11),
        rng_optimization_state=_rng_state(17),
        counter_early_stopping=2,
        base_early_stopping=2.0,
        printed_evaluations=13,
        time_function_evaluations=0.25,
        runtime=1.5,
    )


class StructuralMmesState(MMESState):
    """Behavior-compatible state with deliberately invalid provenance."""


class CountingSphere:
    def __init__(self) -> None:
        self.evaluations = 0

    def __call__(self, x_batch):
        values = np.asarray(x_batch, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        self.evaluations += len(values)
        return np.sum(np.square(values), axis=1)


def _optimizer(*, max_fes: int, seed: int = 23) -> MMES:
    ndim = 4
    return MMES(
        {
            "fitness_function": CountingSphere(),
            "ndim_problem": ndim,
            "lower_boundary": -5.0 * np.ones(ndim),
            "upper_boundary": 5.0 * np.ones(ndim),
        },
        {
            "max_function_evaluations": max_fes,
            "mean": (np.ones(ndim),),
            "sigma": 0.5,
            "n_individuals": 4,
            "n_parents": 2,
            "seed_rng": seed,
            "is_restart": False,
            "verbose": 0,
        },
    )


def _resumable_state() -> MMESState:
    _result, state = _optimizer(max_fes=9).optimize_with_state()
    return state


def _action(state: MMESState, *, budget_fes: int = 8) -> Phase1MmesResumeAction:
    incumbent = tuple(float(value) for value in state.best_so_far_x)
    parameters = canonical_mmes_parameters(state)
    return Phase1MmesResumeAction(
        problem_id="A4",
        run_seed=117,
        checkpoint_fe=100,
        dispatch_checkpoint_hash=CHECKPOINT_HASH,
        anchor_hash=mmes_resume_anchor_hash("A4", incumbent, state.best_so_far_y),
        state_snapshot=FrozenMmesState.capture(state),
        state_dimension=len(incumbent),
        population_size=state.n_individuals,
        budget_fes=budget_fes,
        optimizer_parameters=parameters,
        optimizer_parameter_hash=parameters.parameter_hash,
        seed_namespace=SEED_NAMESPACE,
        acceptance_fitness=state.best_so_far_y,
        issued_sweep=0,
        target_sweep=0,
        ttl_sweeps=0,
        expires_sweep=0,
    )


def _context(
    state: MMESState,
    *,
    objective=None,
    mmes_factory=MMES,
) -> MmesResumeExecutionContext:
    return MmesResumeExecutionContext(
        current_fe=100,
        current_sweep=0,
        dispatch_checkpoint_hash=CHECKPOINT_HASH,
        incumbent=tuple(float(value) for value in state.best_so_far_x),
        incumbent_fitness=state.best_so_far_y,
        required_seed_namespace=SEED_NAMESPACE,
        objective=CountingSphere() if objective is None else objective,
        mmes_factory=mmes_factory,
    )


def test_canonical_state_hash_excludes_only_wall_clock_counters() -> None:
    baseline = _state()
    timing_changed = baseline.clone()
    timing_changed.runtime += 100.0
    timing_changed.time_function_evaluations += 10.0

    assert canonical_mmes_state_hash(timing_changed) == canonical_mmes_state_hash(baseline)
    baseline_action = _action(baseline)
    timing_action = _action(timing_changed)
    assert baseline_action.state_hash == timing_action.state_hash
    assert baseline_action.state_snapshot.payload_hash != timing_action.state_snapshot.payload_hash
    assert baseline_action.action_hash != timing_action.action_hash
    assert "state_payload_hash" in PHASE1_MMES_RESUME_ACTION_SPEC.parameter_names

    rng_changed = baseline.clone()
    rng_changed.rng_optimization_state["state"]["state"] += 1
    assert canonical_mmes_state_hash(rng_changed) != canonical_mmes_state_hash(baseline)

    fe_changed = baseline.clone()
    fe_changed.n_function_evaluations += 1
    assert canonical_mmes_state_hash(fe_changed) != canonical_mmes_state_hash(baseline)


def test_frozen_snapshot_is_independent_of_the_captured_state() -> None:
    state = _state()
    expected_hash = canonical_mmes_state_hash(state)
    snapshot = FrozenMmesState.capture(state)

    state.mean[0] += 100.0
    state.rng_optimization_state["state"]["state"] += 1
    restored = snapshot.clone_state()

    assert canonical_mmes_state_hash(restored) == expected_hash
    assert snapshot.canonical_hash == expected_hash
    assert not np.shares_memory(restored.mean, state.mean)


def test_frozen_snapshot_rejects_non_vendor_state_provenance() -> None:
    structural_state = StructuralMmesState(**copy.deepcopy(vars(_state())))

    with pytest.raises(ValueError, match="unsupported MMES state type"):
        FrozenMmesState.capture(structural_state)


def test_action_requires_population_aligned_budget() -> None:
    state = _resumable_state()

    with pytest.raises(ValueError, match="whole number of MMES populations"):
        _action(state, budget_fes=6)


def test_action_rejects_state_unrelated_to_its_frozen_boundary_anchor() -> None:
    state = _resumable_state()
    action = _action(state)
    unrelated = state.clone()
    unrelated.best_so_far_x[0] += 1.0

    with pytest.raises(ValueError, match="anchor_hash does not match"):
        replace(action, state_snapshot=FrozenMmesState.capture(unrelated))

    with pytest.raises(ValueError, match="acceptance_fitness must equal"):
        replace(action, acceptance_fitness=action.acceptance_fitness + 1.0)


def test_action_freezes_schema_reference_parameters_and_phase_boundary() -> None:
    state = _resumable_state()
    action = _action(state)
    payload = action.optimizer_parameters.audit_payload()

    assert action.schema_version == PHASE1_MMES_RESUME_SCHEMA_VERSION
    assert action.run_block_reference_version == MMES_RUN_BLOCK_REFERENCE_VERSION
    assert action.trigger_scope == "phase_boundary"
    assert action.issued_sweep == action.target_sweep == action.expires_sweep == 0
    assert action.ttl_sweeps == 0
    assert action.optimizer_parameter_hash == action.optimizer_parameters.parameter_hash
    assert {
        "m",
        "c_c",
        "ms",
        "c_s",
        "a_z",
        "distance",
        "c_a",
        "gamma",
        "population_size",
        "n_parents",
        "max_runtime",
        "fitness_threshold",
        "early_stopping_evaluations",
    } <= payload.keys()

    with pytest.raises(ValueError, match="schema version"):
        replace(action, schema_version=PHASE1_MMES_RESUME_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="reference version"):
        replace(action, run_block_reference_version="unknown-run-block")
    with pytest.raises(ValueError, match="ttl_sweeps=0"):
        replace(action, ttl_sweeps=1)
    with pytest.raises(ValueError, match="target the issued sweep"):
        replace(action, target_sweep=1, expires_sweep=1)
    with pytest.raises(ValueError, match="parameter_hash"):
        replace(action, optimizer_parameter_hash="c" * 64)


def test_executor_is_deterministic_and_does_not_mutate_the_frozen_state() -> None:
    state = _resumable_state()
    action = _action(state)
    before_hash = action.state_hash
    left_objective = CountingSphere()
    right_objective = CountingSphere()

    left = execute_phase1_mmes_resume_action(
        action,
        _context(state, objective=left_objective),
        MmesResumeExecutionState.for_action(action),
    )
    right = execute_phase1_mmes_resume_action(
        action,
        _context(state, objective=right_objective),
        MmesResumeExecutionState.for_action(action),
    )

    assert isinstance(left, MmesResumeExecutionResult)
    assert isinstance(right, MmesResumeExecutionResult)
    assert left.consumed_fes == right.consumed_fes == 8
    assert left.unused_fes == right.unused_fes == 0
    assert left.final_state_hash == right.final_state_hash
    assert left.candidate == right.candidate
    assert left.accepted is (left.candidate_fitness < action.acceptance_fitness)
    assert right.accepted is (right.candidate_fitness < action.acceptance_fitness)
    assert left.lifecycle.status == right.lifecycle.status == "completed"
    assert left.lifecycle.completed_fe == right.lifecycle.completed_fe == 108
    assert left_objective.evaluations == right_objective.evaluations == 8
    assert canonical_mmes_state_hash(action.state_snapshot.clone_state()) == before_hash


def test_strict_acceptance_rejects_an_equal_fitness_candidate() -> None:
    state = _resumable_state()
    action = _action(state)

    def constant_objective(values):
        array = np.asarray(values)
        count = 1 if array.ndim == 1 else array.shape[0]
        return np.full(count, action.acceptance_fitness)

    result = execute_phase1_mmes_resume_action(
        action,
        _context(state, objective=constant_objective),
        MmesResumeExecutionState.for_action(action),
    )

    assert isinstance(result, MmesResumeExecutionResult)
    assert result.candidate_fitness == action.acceptance_fitness
    assert result.accepted is False
    assert result.incumbent == tuple(float(value) for value in state.best_so_far_x)


@pytest.mark.parametrize(
    ("context_change", "message"),
    [
        ({"current_sweep": 1}, "phase boundary expired"),
        ({"current_fe": 101}, "current_fe"),
        ({"dispatch_checkpoint_hash": "c" * 64}, "checkpoint_hash"),
        ({"required_seed_namespace": "wrong:namespace"}, "seed namespace"),
    ],
)
def test_preflight_mismatch_records_explicit_abstention(
    context_change: dict[str, object],
    message: str,
) -> None:
    state = _resumable_state()
    action = _action(state)
    objective = CountingSphere()
    lifecycle = MmesResumeExecutionState.for_action(action)
    factory_calls = 0

    def counted_factory(problem, options):
        nonlocal factory_calls
        factory_calls += 1
        return MMES(problem, options)

    context = replace(
        _context(state, objective=objective, mmes_factory=counted_factory),
        **context_change,
    )
    result = execute_phase1_mmes_resume_action(action, context, lifecycle)

    assert isinstance(result, MmesResumeRejectionResult)
    assert result.disposition == "abstained"
    assert message in result.reason
    assert lifecycle.status == "abstained"
    assert lifecycle.invalidation_reason == result.reason
    assert lifecycle.consumed_fes == 0
    assert factory_calls == 0
    assert objective.evaluations == 0


def test_repeated_invocation_returns_rejection_without_mutating_terminal_state() -> None:
    state = _resumable_state()
    action = _action(state, budget_fes=4)
    objective = CountingSphere()
    lifecycle = MmesResumeExecutionState.for_action(action)
    context = _context(state, objective=objective)

    first = execute_phase1_mmes_resume_action(action, context, lifecycle)
    before_repeat = lifecycle.audit_payload(action)
    second = execute_phase1_mmes_resume_action(action, context, lifecycle)

    assert isinstance(first, MmesResumeExecutionResult)
    assert isinstance(second, MmesResumeRejectionResult)
    assert second.disposition == "rejected"
    assert second.reason == "execution_state_already_completed"
    assert lifecycle.audit_payload(action) == before_repeat
    assert objective.evaluations == 4


def test_actual_optimizer_parameter_drift_abstains_before_objective_evaluation() -> None:
    state = _resumable_state()
    action = _action(state)
    objective = CountingSphere()
    lifecycle = MmesResumeExecutionState.for_action(action)

    def drifting_factory(problem, options):
        optimizer = MMES(problem, options)
        optimizer.ms = 1
        optimizer._z_2 = np.sqrt(optimizer.gamma / optimizer.ms)
        return optimizer

    result = execute_phase1_mmes_resume_action(
        action,
        _context(state, objective=objective, mmes_factory=drifting_factory),
        lifecycle,
    )

    assert isinstance(result, MmesResumeRejectionResult)
    assert result.disposition == "abstained"
    assert "parameter receipt mismatch" in result.reason
    assert lifecycle.status == "abstained"
    assert objective.evaluations == 0


def test_partial_vendor_block_is_failed_without_fallback_and_records_fes() -> None:
    state = _resumable_state()
    action = _action(state, budget_fes=8)
    objective = CountingSphere()
    lifecycle = MmesResumeExecutionState.for_action(action)

    def partial_factory(problem, options):
        optimizer = MMES(problem, options)
        original_run_block = optimizer.run_block

        def partial_run_block(frozen_state, _requested_fes):
            return original_run_block(frozen_state, action.population_size)

        optimizer.run_block = partial_run_block
        return optimizer

    with pytest.raises(RuntimeError, match="requested_fes drifted"):
        execute_phase1_mmes_resume_action(
            action,
            _context(state, objective=objective, mmes_factory=partial_factory),
            lifecycle,
        )

    assert lifecycle.status == "failed"
    assert lifecycle.consumed_fes == action.population_size
    assert lifecycle.unused_fes == action.budget_fes - action.population_size
    assert lifecycle.started_fe == 100
    assert lifecycle.completed_fe == 100 + action.population_size
    assert "execution:RuntimeError" in lifecycle.invalidation_reason
    assert objective.evaluations == action.population_size


def test_post_start_exception_records_failed_lifecycle_and_actual_fes() -> None:
    state = _resumable_state()
    action = _action(state, budget_fes=8)
    objective = CountingSphere()
    lifecycle = MmesResumeExecutionState.for_action(action)

    def failing_factory(problem, options):
        optimizer = MMES(problem, options)
        original_run_block = optimizer.run_block

        def failing_run_block(frozen_state, _requested_fes):
            original_run_block(frozen_state, action.population_size)
            raise RuntimeError("forced post-population failure")

        optimizer.run_block = failing_run_block
        return optimizer

    with pytest.raises(RuntimeError, match="forced post-population failure"):
        execute_phase1_mmes_resume_action(
            action,
            _context(state, objective=objective, mmes_factory=failing_factory),
            lifecycle,
        )

    assert lifecycle.status == "failed"
    assert lifecycle.consumed_fes == action.population_size
    assert lifecycle.unused_fes == action.budget_fes - action.population_size
    assert lifecycle.completed_fe == 100 + action.population_size
    assert "forced post-population failure" in lifecycle.invalidation_reason
    assert objective.evaluations == action.population_size


def test_post_validation_failure_records_the_full_objective_receipt() -> None:
    state = _resumable_state()
    action = _action(state, budget_fes=4)
    objective = CountingSphere()
    lifecycle = MmesResumeExecutionState.for_action(action)

    def corrupting_factory(problem, options):
        optimizer = MMES(problem, options)
        original_run_block = optimizer.run_block

        def corrupting_run_block(frozen_state, requested_fes):
            block = original_run_block(frozen_state, requested_fes)
            return replace(block, state_fingerprint_after="c" * 64)

        optimizer.run_block = corrupting_run_block
        return optimizer

    with pytest.raises(RuntimeError, match="after fingerprint"):
        execute_phase1_mmes_resume_action(
            action,
            _context(state, objective=objective, mmes_factory=corrupting_factory),
            lifecycle,
        )

    assert lifecycle.status == "failed"
    assert lifecycle.consumed_fes == action.budget_fes
    assert lifecycle.unused_fes == 0
    assert lifecycle.completed_fe == 100 + action.budget_fes
    assert objective.evaluations == action.budget_fes

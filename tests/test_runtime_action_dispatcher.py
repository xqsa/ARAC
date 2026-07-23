from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arac.actions.full_space_sep_cma import (
    CANONICAL_SEP_CMA_PARAMETERIZATION,
    CANONICAL_SEP_CMA_PARAMETERS_HASH,
    CANONICAL_SEP_CMA_POPULATION_SIZE,
    CANONICAL_SEP_CMA_REFERENCE_VERSION,
    FULL_SPACE_DIMENSION,
    FULL_SPACE_SEP_CMA_ACTION,
    FullSpaceSepCmaAction,
    FullSpaceSepCmaExecutionContext,
    FullSpaceSepCmaExecutionResult,
    execute_full_space_sep_cma_action,
    full_space_sep_cma_anchor_hash,
    full_space_vector_hash,
)
from arac.actions.runtime_dispatcher import (
    DEFAULT_RUNTIME_ACTION_DISPATCHER,
    RuntimeActionDispatcher,
    UnsupportedRuntimeActionError,
)


def _hash(character: str) -> str:
    return character * 64


def _action(*, acceptance_fitness: float) -> FullSpaceSepCmaAction:
    mean = tuple(0.0 for _ in range(FULL_SPACE_DIMENSION))
    return FullSpaceSepCmaAction(
        problem_id="R4",
        run_seed=117,
        checkpoint_fe=300_000,
        dispatch_checkpoint_hash=_hash("a"),
        trigger_relation_hash=_hash("e"),
        anchor_hash=full_space_sep_cma_anchor_hash("R4", mean),
        initial_mean=mean,
        initial_mean_hash=full_space_vector_hash(mean),
        initial_state_hash=_hash("b"),
        initial_sigma=0.5,
        lower_bound=-5.12,
        upper_bound=5.12,
        acceptance_fitness=acceptance_fitness,
        population_size=CANONICAL_SEP_CMA_POPULATION_SIZE,
        budget_fes=CANONICAL_SEP_CMA_POPULATION_SIZE,
        parameterization=CANONICAL_SEP_CMA_PARAMETERIZATION,
        canonical_reference_version=CANONICAL_SEP_CMA_REFERENCE_VERSION,
        canonical_parameters_hash=CANONICAL_SEP_CMA_PARAMETERS_HASH,
        optimizer_seed=2026071901,
        seed_namespace=FULL_SPACE_SEP_CMA_ACTION,
        restart_policy="none",
        issued_sweep=3,
        target_sweep=4,
        ttl_sweeps=1,
        expires_sweep=4,
    )


@dataclass(frozen=True)
class _OptimizerState:
    state_hash: str
    generation: int = 1


class _FakeSepCma:
    def __init__(
        self,
        objective: Any,
        candidate: tuple[float, ...],
        candidate_fitness: float,
    ) -> None:
        self._objective = objective
        self._candidate = candidate
        self._candidate_fitness = candidate_fitness

    def initialize_state(self) -> _OptimizerState:
        return _OptimizerState(_hash("b"))

    def advance(self, budget_fes: int) -> dict[str, object]:
        self._objective(np.zeros((budget_fes, FULL_SPACE_DIMENSION)))
        return {
            "advanced_function_evaluations": budget_fes,
            "n_function_evaluations": budget_fes,
            "parameter_hash": CANONICAL_SEP_CMA_PARAMETERS_HASH,
            "optimizer_state": _OptimizerState(_hash("d")),
            "best_so_far_x": self._candidate,
            "best_so_far_y": self._candidate_fitness,
        }


def _execution_context(
    action: FullSpaceSepCmaAction,
    *,
    candidate_fitness: float,
) -> tuple[FullSpaceSepCmaExecutionContext, list[int]]:
    objective_call_sizes: list[int] = []
    candidate = tuple(0.25 for _ in range(FULL_SPACE_DIMENSION))

    def objective(batch: np.ndarray) -> np.ndarray:
        array = np.asarray(batch, dtype=float)
        objective_call_sizes.append(len(array))
        return np.sum(np.square(array), axis=1)

    def factory(problem: dict[str, object], options: dict[str, object]) -> _FakeSepCma:
        assert problem["ndim_problem"] == FULL_SPACE_DIMENSION
        assert options["max_function_evaluations"] == action.budget_fes
        assert options["n_individuals"] == CANONICAL_SEP_CMA_POPULATION_SIZE
        return _FakeSepCma(
            problem["fitness_function"],
            candidate,
            candidate_fitness,
        )

    context = FullSpaceSepCmaExecutionContext(
        objective=objective,
        sepcmaes_factory=factory,
        current_fe=action.checkpoint_fe,
        current_sweep=action.target_sweep,
        dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
        trigger_context_hash=action.trigger_relation_hash,
        trigger_scope=action.trigger_scope,
        incumbent=action.initial_mean,
        required_seed_namespace=FULL_SPACE_SEP_CMA_ACTION,
    )
    return context, objective_call_sizes


@pytest.mark.parametrize(
    ("acceptance_fitness", "candidate_fitness", "expected_accepted"),
    ((100.0, 10.0, True), (5.0, 10.0, False)),
)
def test_default_dispatcher_routes_full_space_action_and_preserves_hash_contract(
    acceptance_fitness: float,
    candidate_fitness: float,
    expected_accepted: bool,
) -> None:
    action = _action(acceptance_fitness=acceptance_fitness)
    action_hash = action.action_hash
    context, objective_call_sizes = _execution_context(
        action,
        candidate_fitness=candidate_fitness,
    )

    result = DEFAULT_RUNTIME_ACTION_DISPATCHER.execute(action, context)

    assert isinstance(DEFAULT_RUNTIME_ACTION_DISPATCHER, RuntimeActionDispatcher)
    assert isinstance(result, FullSpaceSepCmaExecutionResult)
    assert sum(objective_call_sizes) == action.budget_fes
    assert result.consumed_fes == action.budget_fes
    assert result.lifecycle.started_fe == action.checkpoint_fe
    assert result.lifecycle.completed_fe == action.checkpoint_fe + action.budget_fes
    assert result.action_hash == action_hash == action.action_hash
    assert result.accepted is expected_accepted
    assert result.candidate_hash == full_space_vector_hash(result.candidate)
    assert result.post_incumbent_hash == full_space_vector_hash(result.incumbent)
    assert result.post_incumbent_hash == (
        result.candidate_hash if expected_accepted else action.initial_mean_hash
    )
    assert result.incumbent_fitness == (
        candidate_fitness if expected_accepted else acceptance_fitness
    )
    assert result.resume_native is True


@pytest.mark.parametrize(
    "executor",
    (
        execute_full_space_sep_cma_action,
        DEFAULT_RUNTIME_ACTION_DISPATCHER.execute,
    ),
)
def test_full_space_executor_rejects_the_wrong_context_type(executor: Any) -> None:
    action = _action(acceptance_fitness=100.0)

    with pytest.raises(TypeError):
        executor(action, object())


@dataclass(frozen=True)
class _UnknownAction:
    name: str = "unknown_action"


def test_dispatcher_fails_explicitly_for_an_unregistered_action() -> None:
    full_space_action = _action(acceptance_fitness=100.0)
    context, _ = _execution_context(full_space_action, candidate_fitness=10.0)

    with pytest.raises(UnsupportedRuntimeActionError):
        DEFAULT_RUNTIME_ACTION_DISPATCHER.execute(_UnknownAction(), context)


def test_full_space_numerical_execution_has_one_action_layer_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    runner_source = (root / "scripts" / "hcc_smoke_runner.py").read_text(
        encoding="utf-8"
    )
    ceiling_source = (
        root / "src" / "arac" / "backends" / "hcc_action_ceiling_runtime.py"
    ).read_text(encoding="utf-8")
    compatibility_source = (
        root / "src" / "arac" / "backends" / "hcc_persistent_phase2.py"
    ).read_text(encoding="utf-8")

    assert "execute_full_space_sep_cma_burst_action(" not in runner_source
    assert "sep_optimizer.advance(" not in ceiling_source
    assert "def _execute_full_space_sep_cma_action(" not in compatibility_source
    assert runner_source.count("DEFAULT_RUNTIME_ACTION_DISPATCHER.execute(") == 2
    assert ceiling_source.count("DEFAULT_RUNTIME_ACTION_DISPATCHER.execute(") == 1

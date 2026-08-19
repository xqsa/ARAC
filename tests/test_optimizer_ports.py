from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import (
    PypopOptimizerPort,
    ResumableOptimizerSession,
    _BatchedMMES,
)


@pytest.mark.parametrize("algorithm", ["cmaes", "sepcmaes", "mmes"])
def test_upstream_optimizer_port_consumes_exact_budget(algorithm: str) -> None:
    seen_shapes = []

    def objective(x):
        seen_shapes.append(np.asarray(x).shape)
        return np.sum(np.asarray(x, dtype=float) ** 2, axis=-1)

    problem = OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )
    ledger = EvaluationLedger(problem, 37)

    result = PypopOptimizerPort().run(
        algorithm,
        problem=problem,
        ledger=ledger,
        initial_mean=np.ones(4),
        sigma=0.5,
        seed=7,
        budget_fes=37,
        population_size=6,
    )

    assert result.consumed_fes == 37
    assert ledger.count == 37
    assert result.package == "pypop7"
    assert np.isfinite(result.best_error)
    assert any(len(shape) == 2 and shape[0] > 1 for shape in seen_shapes)


def test_full_cma_is_rejected_for_large_dimension() -> None:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=1000,
        lower_bounds=(-5.0,) * 1000,
        upper_bounds=(5.0,) * 1000,
    )
    ledger = EvaluationLedger(problem, 10)

    with pytest.raises(ValueError, match="disabled above 256"):
        PypopOptimizerPort().run(
            "cmaes",
            problem=problem,
            ledger=ledger,
            initial_mean=np.zeros(1000),
            sigma=0.5,
            seed=1,
            budget_fes=10,
        )


@pytest.mark.parametrize("algorithm", ["cmaes", "sepcmaes", "mmes"])
def test_optimizer_port_repairs_candidates_to_public_bounds(algorithm: str) -> None:
    def bounded_objective(x):
        candidates = np.asarray(x, dtype=float)
        assert np.all(candidates >= -1.0)
        assert np.all(candidates <= 1.0)
        return np.sum(candidates**2, axis=-1)

    problem = OptimizationProblem(
        objective=bounded_objective,
        dimension=4,
        lower_bounds=(-1.0,) * 4,
        upper_bounds=(1.0,) * 4,
    )
    ledger = EvaluationLedger(problem, 12)

    PypopOptimizerPort().run(
        algorithm,
        problem=problem,
        ledger=ledger,
        initial_mean=np.full(4, 0.99),
        sigma=50.0,
        seed=7,
        budget_fes=12,
        population_size=6,
    )

    assert ledger.count == 12


def test_batched_mmes_repairs_sigma_underflow_before_distribution_update() -> None:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )
    ledger = EvaluationLedger(problem, 16)
    optimizer = _BatchedMMES(
        {
            "fitness_function": ledger.evaluate,
            "ndim_problem": problem.dimension,
            "lower_boundary": problem.lower_array,
            "upper_boundary": problem.upper_array,
        },
        {
            "max_function_evaluations": 16,
            "mean": np.zeros(4),
            "sigma": 0.5,
            "seed_rng": 7,
            "n_individuals": 4,
            "is_restart": False,
            "verbose": 0,
        },
    )
    x, mean, p, w, q, t, v, y = optimizer.initialize()
    optimizer.sigma = 0.0
    optimizer._n_generations = 0

    x, y = optimizer.iterate(x, mean, q, v, y)
    state = optimizer._update_distribution(x, mean, p, w, q, t, v, y, np.copy(y))

    assert optimizer.sigma >= optimizer.sigma_floor > 0.0
    assert optimizer.sigma_floor_repair_count >= 1
    assert all(np.all(np.isfinite(value)) for value in state)


@pytest.mark.parametrize("algorithm", ["cmaes", "sepcmaes", "mmes"])
def test_resumable_optimizer_session_preserves_split_prefix(algorithm: str) -> None:
    full_events: list[tuple[float, ...]] = []

    def full_objective(values):
        rows = np.asarray(values, dtype=float)
        full_events.extend(tuple(float(value) for value in row) for row in rows.reshape(-1, 4))
        return np.sum(rows**2, axis=-1)

    problem = OptimizationProblem(
        objective=full_objective,
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )
    full_ledger = EvaluationLedger(
        problem,
        42,
        initial_incumbent=(1.0,) * 4,
        initial_error=4.0,
    )
    uninterrupted = ResumableOptimizerSession(
        algorithm,
        problem=problem,
        ledger=full_ledger,
        initial_mean=(1.0,) * 4,
        sigma=0.5,
        seed=7,
        budget_fes=42,
        population_size=6,
    )
    uninterrupted.step(42)

    split_events: list[tuple[float, ...]] = []

    def split_objective(values):
        rows = np.asarray(values, dtype=float)
        split_events.extend(tuple(float(value) for value in row) for row in rows.reshape(-1, 4))
        return np.sum(rows**2, axis=-1)

    split_problem = OptimizationProblem(
        objective=split_objective,
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )
    split_ledger = EvaluationLedger(
        split_problem,
        42,
        initial_incumbent=(1.0,) * 4,
        initial_error=4.0,
    )
    split = ResumableOptimizerSession(
        algorithm,
        problem=split_problem,
        ledger=split_ledger,
        initial_mean=(1.0,) * 4,
        sigma=0.5,
        seed=7,
        budget_fes=42,
        population_size=6,
    )
    split.step(13)
    payload = split.json_payload()
    resumed_ledger = EvaluationLedger(
        split_problem,
        42,
        initial_count=13,
        initial_incumbent=tuple(float(value) for value in split_ledger.best_x),
        initial_error=split_ledger.best_error,
    )
    resumed = ResumableOptimizerSession(
        algorithm,
        problem=split_problem,
        ledger=resumed_ledger,
        initial_mean=(1.0,) * 4,
        sigma=0.5,
        seed=7,
        budget_fes=42,
        population_size=6,
        initial_consumed=13,
    )
    resumed.restore_json_payload(payload)
    resumed.step(29)

    assert split_events == full_events
    assert resumed_ledger.count == full_ledger.count == 42
    assert resumed_ledger.best_error == uninterrupted.ledger.best_error


def test_resumable_optimizer_session_can_optimize_a_fixed_coordinate_subset() -> None:
    seen: list[np.ndarray] = []

    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        seen.extend(row.copy() for row in batch)
        return np.sum(batch**2, axis=-1)

    problem = OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )
    ledger = EvaluationLedger(
        problem,
        20,
        initial_incumbent=(1.0,) * 4,
        initial_error=4.0,
    )
    session = ResumableOptimizerSession(
        "cmaes",
        problem=problem,
        ledger=ledger,
        initial_mean=(1.0, 1.0),
        sigma=0.5,
        seed=3,
        budget_fes=20,
        population_size=4,
        dimensions=(0, 2),
        anchor=(1.0,) * 4,
    )
    session.step(20)

    assert ledger.count == 20
    assert all(np.array_equal(row[[1, 3]], np.ones(2)) for row in seen)


def test_resumable_optimizer_session_repairs_sigma_underflow_before_sampling() -> None:
    seen: list[np.ndarray] = []

    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        seen.extend(row.copy() for row in batch)
        result = np.sum(batch**2, axis=1)
        return float(result[0]) if rows.ndim == 1 else result

    problem = OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )
    ledger = EvaluationLedger(
        problem,
        10,
        initial_incumbent=(1.0,) * 4,
        initial_error=4.0,
    )
    session = ResumableOptimizerSession(
        "mmes",
        problem=problem,
        ledger=ledger,
        initial_mean=(1.0,) * 4,
        sigma=0.5,
        seed=5,
        budget_fes=10,
        population_size=4,
    )
    session.optimizer.sigma = 0.0

    session.step(1)

    assert session.optimizer.sigma == session.sigma_floor
    assert session.sigma_floor_repair_count == 1
    assert np.all(np.isfinite(seen[0]))

from __future__ import annotations

import numpy as np
import pytest

from arac.backends.diagonal_cma import (
    initialize_diagonal_cma_state,
    run_diagonal_cma_block,
)


def _state(*, seed: int = 7, incumbent_fitness: float = 10.0):
    return initialize_diagonal_cma_state(
        initial_mean=np.array([1.0, -1.0, 0.5]),
        sigma=0.5,
        lower=np.full(3, -5.0),
        upper=np.full(3, 5.0),
        seed=seed,
        population_size=4,
        incumbent_fitness=incumbent_fitness,
    )


def _sphere(batch: np.ndarray) -> np.ndarray:
    values = np.asarray(batch, dtype=float)
    return np.sum(values * values, axis=1)


def test_same_seed_blocks_are_deterministic_and_resumable() -> None:
    left = _state(seed=11)
    right = _state(seed=11)

    first_left = run_diagonal_cma_block(left, _sphere, requested_fes=8)
    first_right = run_diagonal_cma_block(right, _sphere, requested_fes=8)

    assert first_left.actual_fes == first_right.actual_fes == 8
    assert first_left.best_after == pytest.approx(first_right.best_after)
    np.testing.assert_allclose(first_left.state.best_x, first_right.state.best_x)
    assert first_left.state_fingerprint_after == first_right.state_fingerprint_after

    second_left = run_diagonal_cma_block(left, _sphere, requested_fes=4)
    second_right = run_diagonal_cma_block(right, _sphere, requested_fes=4)
    assert second_left.state.total_fes == second_right.state.total_fes == 12
    assert second_left.state_fingerprint_after == second_right.state_fingerprint_after


def test_block_rounds_down_to_complete_populations() -> None:
    state = _state()

    result = run_diagonal_cma_block(state, _sphere, requested_fes=10)

    assert result.requested_fes == 10
    assert result.actual_fes == 8
    assert result.unused_fes == 2
    assert result.actual_fes % state.population_size == 0
    assert state.total_fes == 8


def test_zero_budget_is_a_true_noop() -> None:
    state = _state()
    fingerprint = state.fingerprint()
    calls = 0

    def objective(batch: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += len(batch)
        return _sphere(batch)

    result = run_diagonal_cma_block(state, objective, requested_fes=3)

    assert result.actual_fes == 0
    assert result.unused_fes == 3
    assert calls == 0
    assert state.fingerprint() == fingerprint
    assert result.state_fingerprint_before == result.state_fingerprint_after


def test_worse_candidates_cannot_replace_protected_incumbent() -> None:
    state = _state(incumbent_fitness=1.0)
    incumbent = state.best_x.copy()

    result = run_diagonal_cma_block(
        state,
        lambda batch: np.full(len(batch), 100.0),
        requested_fes=8,
    )

    assert result.accepted is False
    assert result.candidate_best == 100.0
    assert result.best_before == result.best_after == 1.0
    np.testing.assert_allclose(state.best_x, incumbent)


def test_initialization_rejects_invalid_shapes_and_values() -> None:
    with pytest.raises(ValueError, match="boundary shape"):
        initialize_diagonal_cma_state(
            initial_mean=np.zeros(3),
            sigma=0.5,
            lower=np.full(2, -5.0),
            upper=np.full(3, 5.0),
            seed=1,
            population_size=4,
            incumbent_fitness=1.0,
        )
    with pytest.raises(ValueError, match="sigma"):
        initialize_diagonal_cma_state(
            initial_mean=np.zeros(3),
            sigma=0.0,
            lower=np.full(3, -5.0),
            upper=np.full(3, 5.0),
            seed=1,
            population_size=4,
            incumbent_fitness=1.0,
        )


def test_nonfinite_objective_output_fails_explicitly() -> None:
    state = _state()

    with pytest.raises(RuntimeError, match="non-finite"):
        run_diagonal_cma_block(
            state,
            lambda batch: np.full(len(batch), np.nan),
            requested_fes=4,
        )


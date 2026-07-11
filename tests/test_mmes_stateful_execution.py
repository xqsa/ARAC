from __future__ import annotations

import copy
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest


HCC_SRC = Path(__file__).resolve().parents[1] / "HCC_SRC"
if str(HCC_SRC) not in sys.path:
    sys.path.insert(0, str(HCC_SRC))

from HCC.NDAs.MMES.state import MMESBlockResult, MMESState


def _rng_state(seed: int) -> dict[str, object]:
    return copy.deepcopy(np.random.default_rng(seed).bit_generator.state)


def make_state(ndim: int = 4, population: int = 6) -> MMESState:
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


def test_state_validation_accepts_real_mmes_shapes() -> None:
    state = make_state()

    state.validate()


def test_state_validation_rejects_wrong_shapes() -> None:
    state = make_state(ndim=4, population=6)
    state.x = np.zeros((5, 4))

    with pytest.raises(ValueError, match="x shape"):
        state.validate()


def test_state_validation_rejects_more_than_three_recent_checkpoints() -> None:
    state = make_state()
    state.recent_best.append((14, 1.0))

    with pytest.raises(ValueError, match="recent_best"):
        state.validate()


def test_state_clone_is_deep_and_fingerprint_changes() -> None:
    state = make_state(ndim=4, population=6)
    clone = state.clone()

    assert clone.fingerprint() == state.fingerprint()
    clone.mean[0] += 1.0
    clone.list_initial_mean[0][0] += 1.0
    clone.rng_initialization_state["state"]["state"] += 1

    assert not np.shares_memory(state.mean, clone.mean)
    assert not np.shares_memory(state.list_initial_mean[0], clone.list_initial_mean[0])
    assert state.rng_initialization_state != clone.rng_initialization_state
    assert state.fingerprint() != clone.fingerprint()


def test_rng_state_is_part_of_fingerprint() -> None:
    left = make_state(ndim=4, population=6)
    right = left.clone()
    right.rng_optimization_state["state"]["state"] += 1

    assert left.fingerprint() != right.fingerprint()


def test_clone_round_trips_all_continuation_fields() -> None:
    state = make_state()
    clone = state.clone()

    state.validate()
    clone.validate()
    assert clone.fingerprint() == state.fingerprint()
    assert clone.sigma == state.sigma
    assert clone.sigma_bak == state.sigma_bak
    assert clone.n_individuals == state.n_individuals
    assert clone.n_parents == state.n_parents
    assert clone.n_mirror_sampling == state.n_mirror_sampling
    assert clone.n_restart == state.n_restart
    assert clone.list_generations == state.list_generations
    assert clone.list_fitness == state.list_fitness
    assert np.array_equal(clone.best_so_far_x, state.best_so_far_x)
    assert clone.best_so_far_y == state.best_so_far_y
    assert clone.n_function_evaluations == state.n_function_evaluations
    assert clone.termination_signal == state.termination_signal
    assert clone.fitness == state.fitness
    assert clone.recent_best == state.recent_best
    assert clone.rng_initialization_state == state.rng_initialization_state
    assert clone.rng_optimization_state == state.rng_optimization_state


def test_block_result_is_an_immutable_audit_record() -> None:
    state = make_state()
    result = MMESBlockResult(
        state=state,
        best_before=3.0,
        best_after=2.0,
        actual_fes=24,
        requested_fes=24,
        unused_fes=0,
        normalized_utility=1.0 / 72.0,
        termination_reason="block_complete",
        state_fingerprint_before="before",
        state_fingerprint_after="after",
    )

    with pytest.raises(FrozenInstanceError):
        result.actual_fes = 25

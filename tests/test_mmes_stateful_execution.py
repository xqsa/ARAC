from __future__ import annotations

import copy
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest


HCC_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "hcc"
if str(HCC_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(HCC_VENDOR_ROOT))

from HCC.NDAs.MMES.state import MMESBlockResult, MMESState
from HCC.NDAs.MMES.mmes import MMES


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


class CountingSphere:
    def __init__(self) -> None:
        self.evaluations = 0

    def __call__(self, x_batch):
        values = np.asarray(x_batch, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        self.evaluations += len(values)
        return np.sum(np.square(values), axis=1)


def make_optimizer(
    *,
    max_fes: int,
    objective: CountingSphere | None = None,
    seed: int = 23,
    restart: bool = False,
) -> tuple[MMES, CountingSphere]:
    sphere = objective or CountingSphere()
    ndim = 4
    optimizer = MMES(
        {
            "fitness_function": sphere,
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
            "is_restart": restart,
            "verbose": 0,
        },
    )
    return optimizer, sphere


def test_initialize_state_captures_the_single_initial_evaluation() -> None:
    optimizer, objective = make_optimizer(max_fes=25)

    state = optimizer.initialize_state()

    state.validate()
    assert state.n_function_evaluations == 1
    assert objective.evaluations == 1
    assert state.recent_best == [(1, state.best_so_far_y)]
    assert state.pending_distribution_update is False


def test_optimize_and_optimize_with_state_are_equivalent() -> None:
    legacy_optimizer, _ = make_optimizer(max_fes=25)
    stateful_optimizer, _ = make_optimizer(max_fes=25)

    legacy = legacy_optimizer.optimize()
    stateful, state = stateful_optimizer.optimize_with_state()

    assert stateful["best_so_far_y"] == pytest.approx(legacy["best_so_far_y"])
    assert np.array_equal(stateful["best_so_far_x"], legacy["best_so_far_x"])
    assert stateful["n_function_evaluations"] == legacy["n_function_evaluations"]
    assert stateful["termination_signal"] == legacy["termination_signal"]
    assert np.array_equal(stateful["mean"], legacy["mean"])
    assert np.array_equal(stateful["p"], legacy["p"])
    assert stateful["w"] == pytest.approx(legacy["w"])
    assert state.n_function_evaluations == 25
    assert state.pending_distribution_update is True


def test_run_block_never_exceeds_request_and_uses_complete_populations() -> None:
    optimizer, objective = make_optimizer(max_fes=9)
    _results, state = optimizer.optimize_with_state()
    before_objective_fe = objective.evaluations

    block = optimizer.run_block(state, additional_function_evaluations=10)

    assert block.requested_fes == 10
    assert block.actual_fes == 8
    assert block.unused_fes == 2
    assert block.actual_fes <= block.requested_fes
    assert block.actual_fes % state.n_individuals == 0
    assert objective.evaluations - before_objective_fe == block.actual_fes


def test_run_block_with_less_than_population_is_a_true_noop() -> None:
    optimizer, objective = make_optimizer(max_fes=9)
    _results, state = optimizer.optimize_with_state()
    before_objective_fe = objective.evaluations
    before_fingerprint = state.fingerprint()

    block = optimizer.run_block(state, additional_function_evaluations=3)

    assert block.actual_fes == 0
    assert block.termination_reason == "insufficient_population_budget"
    assert objective.evaluations == before_objective_fe
    assert block.state.fingerprint() == before_fingerprint


def test_sequential_block_resumption_matches_one_continuous_run() -> None:
    full_optimizer, _ = make_optimizer(max_fes=25)
    full_results, full_state = full_optimizer.optimize_with_state()

    phase_optimizer, _ = make_optimizer(max_fes=9)
    _phase_results, phase_state = phase_optimizer.optimize_with_state()
    assert phase_state.pending_distribution_update is True

    resumed = phase_optimizer.run_block(
        phase_state,
        additional_function_evaluations=16,
    )

    assert resumed.actual_fes == 16
    assert resumed.state.n_function_evaluations == 25
    assert resumed.best_after == pytest.approx(full_results["best_so_far_y"])
    assert np.array_equal(resumed.state.best_so_far_x, full_state.best_so_far_x)
    assert np.allclose(resumed.state.mean, full_state.mean)
    assert np.allclose(resumed.state.p, full_state.p)
    assert np.allclose(resumed.state.q, full_state.q)
    assert np.array_equal(resumed.state.v, full_state.v)
    assert resumed.state.sigma == pytest.approx(full_state.sigma)
    assert resumed.state.rng_optimization_state == full_state.rng_optimization_state


def test_malformed_state_fails_before_objective_evaluation() -> None:
    optimizer, objective = make_optimizer(max_fes=9)
    _results, state = optimizer.optimize_with_state()
    malformed = state.clone()
    malformed.x = np.zeros((1, 4))
    before_objective_fe = objective.evaluations

    with pytest.raises(ValueError, match="x shape"):
        optimizer.run_block(malformed, additional_function_evaluations=8)

    assert objective.evaluations == before_objective_fe


def test_state_to_result_does_not_mutate_the_checkpoint() -> None:
    optimizer, _ = make_optimizer(max_fes=9)
    expected, state = optimizer.optimize_with_state()
    before = state.fingerprint()

    result = optimizer.state_to_result(state)

    assert state.fingerprint() == before
    assert result["best_so_far_y"] == pytest.approx(expected["best_so_far_y"])
    assert np.array_equal(result["best_so_far_x"], expected["best_so_far_x"])
    assert result["n_function_evaluations"] == expected["n_function_evaluations"]

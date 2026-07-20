from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "hcc"
sys.path.insert(0, str(VENDOR_ROOT))

from HCC.OPT.CMAES.sepcmaes import (  # noqa: E402
    CANONICAL_PARAMETERIZATION,
    CANONICAL_REFERENCE_VERSION,
    SEPCMAES,
    SepCMAESState,
    canonical_sep_cma_parameters,
)


def _optimizer(
    *,
    dimension: int = 6,
    population_size: int = 6,
    max_evaluations: int = 18,
    seed: int = 117,
    calls: list[int] | None = None,
) -> SEPCMAES:
    def sphere(values: np.ndarray) -> np.ndarray:
        if calls is not None:
            calls.append(len(values))
        return np.sum(np.square(values), axis=1)

    return SEPCMAES(
        {
            "fitness_function": sphere,
            "ndim_problem": dimension,
            "lower_boundary": -10.0 * np.ones(dimension),
            "upper_boundary": 10.0 * np.ones(dimension),
        },
        {
            "max_function_evaluations": max_evaluations,
            "mean": np.linspace(0.25, 1.0, dimension),
            "sigma": 0.2,
            "n_individuals": population_size,
            "is_restart": False,
            "seed_rng": seed,
            "verbose": 0,
        },
    )


def test_canonical_parameters_match_pypop7_sepcmaes_formula() -> None:
    dimension = 8
    population_size = 10
    parents = population_size // 2
    parameters = canonical_sep_cma_parameters(dimension, population_size)

    raw_weights = np.asarray(
        [
            math.log((population_size + 1.0) / 2.0) - math.log(index + 1.0)
            for index in range(parents)
        ]
    )
    weights = raw_weights / np.sum(raw_weights)
    mu_eff = 1.0 / np.sum(np.square(weights))
    base_c_cov = (1.0 / mu_eff) * 2.0 / (dimension + math.sqrt(2.0)) ** 2
    base_c_cov += (1.0 - 1.0 / mu_eff) * min(
        1.0,
        (2.0 * mu_eff - 1.0) / ((dimension + 2.0) ** 2 + mu_eff),
    )
    expected_c_cov = base_c_cov * (dimension + 2.0) / 3.0

    assert parameters.parameterization == CANONICAL_PARAMETERIZATION
    assert parameters.reference_version == CANONICAL_REFERENCE_VERSION
    assert parameters.parent_count == parents
    assert parameters.recombination_weights == pytest.approx(weights)
    assert parameters.mu_eff == pytest.approx(mu_eff)
    assert parameters.c_sigma == pytest.approx(
        (mu_eff + 2.0) / (dimension + mu_eff + 3.0)
    )
    assert parameters.d_sigma == pytest.approx(
        1.0
        + parameters.c_sigma
        + 2.0
        * math.sqrt(max((mu_eff - 1.0) / (dimension + 1.0) - 1.0, 0.0))
    )
    assert parameters.c_c == pytest.approx(4.0 / (dimension + 4.0))
    assert parameters.c_cov == pytest.approx(expected_c_cov)
    assert parameters.separable_covariance_multiplier == pytest.approx(
        (dimension + 2.0) / 3.0
    )
    assert parameters.rank_one_rate == pytest.approx(expected_c_cov / mu_eff)
    assert parameters.rank_mu_rate == pytest.approx(
        expected_c_cov * (1.0 - 1.0 / mu_eff)
    )
    assert len(parameters.parameter_hash) == 64
    assert canonical_sep_cma_parameters(1000).population_size == 24


def test_first_generation_uses_combined_pypop_covariance_update() -> None:
    optimizer = _optimizer(dimension=5, population_size=6, max_evaluations=6)
    initial = optimizer.initialize_state()
    parameters = optimizer.parameters

    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(initial.rng_state_json)
    z = rng.standard_normal((parameters.population_size, parameters.dimension))
    candidates = initial.mean + initial.sigma * z
    fitness = np.sum(np.square(candidates), axis=1)
    selected = np.argsort(fitness)[: parameters.parent_count]
    weights = np.asarray(parameters.recombination_weights)
    weighted_z = weights @ z[selected]
    expected_path_sigma = math.sqrt(
        parameters.mu_eff
        * parameters.c_sigma
        * (2.0 - parameters.c_sigma)
    ) * weighted_z
    normalized_path = np.linalg.norm(expected_path_sigma) / math.sqrt(
        1.0 - (1.0 - parameters.c_sigma) ** 2
    )
    h_sigma = float(
        normalized_path
        < (1.4 + 2.0 / (parameters.dimension + 1.0)) * parameters.chi_n
    )
    expected_path_covariance = (
        h_sigma
        * math.sqrt(
            parameters.c_c * (2.0 - parameters.c_c) * parameters.mu_eff
        )
        * weighted_z
    )
    rank_mu = weights @ np.square(z[selected])
    expected_variances = (
        (1.0 - parameters.c_cov) * np.ones(parameters.dimension)
        + parameters.rank_one_rate * np.square(expected_path_covariance)
        + parameters.rank_mu_rate * rank_mu
    )
    expected_sigma = initial.sigma * math.exp(
        parameters.c_sigma
        / parameters.d_sigma
        * (np.linalg.norm(expected_path_sigma) / parameters.chi_n - 1.0)
    )

    final = optimizer.advance(6, state=initial)["optimizer_state"]

    np.testing.assert_allclose(final.mean, weights @ candidates[selected])
    np.testing.assert_allclose(final.path_sigma, expected_path_sigma)
    np.testing.assert_allclose(final.path_covariance, expected_path_covariance)
    np.testing.assert_allclose(final.variances, expected_variances)
    assert final.sigma == pytest.approx(expected_sigma)
    assert final.generation == 1


def test_state_is_read_only_cloneable_and_linear_in_dimension(monkeypatch) -> None:
    optimizer = _optimizer(
        dimension=1000,
        population_size=24,
        max_evaluations=24,
    )
    initial = optimizer.initialize_state()

    state_vectors = (
        initial.mean,
        initial.path_sigma,
        initial.path_covariance,
        initial.variances,
    )
    assert all(vector.shape == (1000,) for vector in state_vectors)
    assert sum(vector.nbytes for vector in state_vectors) == 4 * 1000 * 8
    with pytest.raises(ValueError):
        initial.mean[0] = 1.0

    cloned = initial.clone()
    assert cloned.state_hash == initial.state_hash
    assert not np.shares_memory(cloned.mean, initial.mean)

    def fail_quadratic_operation(*_args, **_kwargs):
        raise AssertionError("Sep-CMA-ES must not construct a dense covariance matrix")

    monkeypatch.setattr(np, "outer", fail_quadratic_operation)
    monkeypatch.setattr(np.linalg, "eigh", fail_quadratic_operation)
    result = optimizer.advance(24, state=cloned)
    assert result["optimizer_state"].variances.shape == (1000,)


def test_partial_population_consumes_exact_fes_without_distribution_update() -> None:
    full_calls: list[int] = []
    partial_calls: list[int] = []
    full = _optimizer(
        dimension=5,
        population_size=4,
        max_evaluations=8,
        calls=full_calls,
    ).optimize()
    partial = _optimizer(
        dimension=5,
        population_size=4,
        max_evaluations=10,
        calls=partial_calls,
    ).optimize()

    assert full_calls == [4, 4]
    assert partial_calls == [4, 4, 2]
    assert partial["n_function_evaluations"] == 10
    assert partial["advanced_function_evaluations"] == 10
    assert partial["partial_population_evaluations"] == 2
    assert (
        partial["optimizer_state"].terminal_partial_population_evaluations
        == 2
    )
    assert partial["_n_generations"] == 2
    assert np.isfinite(partial["best_so_far_y"])
    assert np.all(np.isfinite(partial["best_so_far_x"]))

    full_state = full["optimizer_state"]
    partial_state = partial["optimizer_state"]
    np.testing.assert_array_equal(partial_state.mean, full_state.mean)
    np.testing.assert_array_equal(partial_state.path_sigma, full_state.path_sigma)
    np.testing.assert_array_equal(
        partial_state.path_covariance,
        full_state.path_covariance,
    )
    np.testing.assert_array_equal(partial_state.variances, full_state.variances)
    assert partial_state.sigma == full_state.sigma


def test_terminal_partial_population_cannot_be_continued_or_restored() -> None:
    optimizer = _optimizer(
        dimension=5,
        population_size=6,
        max_evaluations=12,
    )
    terminal = optimizer.advance(5)["optimizer_state"]

    assert terminal.terminal_partial_population_evaluations == 5
    with pytest.raises(RuntimeError, match="terminal partial-population"):
        optimizer.advance(1)

    fresh = _optimizer(
        dimension=5,
        population_size=6,
        max_evaluations=12,
    )
    with pytest.raises(ValueError, match="terminal partial-population"):
        fresh.advance(1, state=terminal)


def test_full_generation_segments_restore_to_identical_state() -> None:
    one_shot = _optimizer(max_evaluations=18, seed=117).optimize()[
        "optimizer_state"
    ]

    first_optimizer = _optimizer(max_evaluations=18, seed=117)
    checkpoint = first_optimizer.advance(6)["optimizer_state"].clone()
    checkpoint_hash = checkpoint.state_hash

    resumed_optimizer = _optimizer(max_evaluations=18, seed=999)
    resumed = resumed_optimizer.advance(12, state=checkpoint)["optimizer_state"]

    assert checkpoint.state_hash == checkpoint_hash
    assert resumed.state_hash == one_shot.state_hash
    assert resumed.generation == 3
    assert resumed.n_function_evaluations == 18


def test_state_restore_fails_closed_on_shape_finiteness_and_parameters() -> None:
    optimizer = _optimizer()
    state = optimizer.initialize_state()

    with pytest.raises(ValueError, match="mean must have shape"):
        replace(state, mean=np.zeros(state.dimension + 1))
    with pytest.raises(ValueError, match="variances must contain only finite"):
        replace(state, variances=np.full(state.dimension, np.nan))

    other_population = _optimizer(population_size=8)
    with pytest.raises(ValueError, match="population_size"):
        other_population.restore_state(state)

    mismatched_hash = "0" * 64 if state.parameter_hash != "0" * 64 else "1" * 64
    incompatible_state = replace(state, parameter_hash=mismatched_hash)
    with pytest.raises(ValueError, match="parameter_hash"):
        optimizer.restore_state(incompatible_state)

    state.mean.setflags(write=True)
    state.mean[0] = np.nan
    with pytest.raises(ValueError, match="mean must contain only finite"):
        optimizer.restore_state(state)


def test_state_type_contains_no_dense_numerical_state() -> None:
    state = _optimizer().initialize_state()

    for field_name in state.__slots__:
        value = getattr(state, field_name)
        if isinstance(value, np.ndarray):
            assert value.ndim == 1, field_name
    assert isinstance(state, SepCMAESState)

from __future__ import annotations

import time

import numpy as np

from scripts import hcc_smoke_runner as _runner  # noqa: F401
from HCC.OPT.CMAES.cmaes import CMAES


def _optimizer(*, diagonal_only: bool) -> CMAES:
    dimension = 4

    def sphere(values: np.ndarray) -> np.ndarray:
        return np.sum(np.square(values), axis=1)

    return CMAES(
        {
            "fitness_function": sphere,
            "ndim_problem": dimension,
            "lower_boundary": -5.0 * np.ones(dimension),
            "upper_boundary": 5.0 * np.ones(dimension),
        },
        {
            "max_function_evaluations": 20,
            "mean": (np.ones(dimension),),
            "sigma": 0.5,
            "n_individuals": 4,
            "is_restart": False,
            "verbose": 0,
            "early_stopping_evaluations": 1000,
            "seed_rng": 117,
            "diagonal_only": diagonal_only,
        },
    )


def test_default_cma_keeps_dense_covariance_representation() -> None:
    optimizer = _optimizer(diagonal_only=False)

    _, _, _, _, covariance, eigenvectors, _, _, _ = optimizer.initialize()

    assert covariance.shape == (4, 4)
    assert eigenvectors.shape == (4, 4)


def test_diagonal_cma_uses_vector_covariance_representation() -> None:
    optimizer = _optimizer(diagonal_only=True)

    _, _, _, _, covariance, eigenvectors, deviations, _, _ = optimizer.initialize()

    assert covariance.shape == (4,)
    assert eigenvectors is None
    assert deviations.shape == (4,)


def test_diagonal_update_does_not_construct_outer_products(monkeypatch) -> None:
    optimizer = _optimizer(diagonal_only=True)
    x, mean, p_s, p_c, covariance, eigenvectors, deviations, y, d = (
        optimizer.initialize()
    )
    optimizer.start_time = time.time()
    x, y, d = optimizer.iterate(x, mean, eigenvectors, deviations, y, d)

    def fail_outer(*_args, **_kwargs):
        raise AssertionError("diagonal CMA must not construct dense outer products")

    monkeypatch.setattr(np, "outer", fail_outer)
    result = optimizer.update_distribution(
        x,
        p_s,
        p_c,
        covariance,
        eigenvectors,
        deviations,
        y,
        d,
    )

    updated_covariance = result[3]
    assert updated_covariance.shape == (4,)
    assert result[4] is None
    assert np.all(np.isfinite(updated_covariance))


def test_diagonal_optimizer_completes_finite_sphere_run() -> None:
    result = _optimizer(diagonal_only=True).optimize()

    assert np.isfinite(result["best_so_far_y"])
    assert len(result["best_so_far_x"]) == 4

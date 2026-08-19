from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.overlap_objective import build_overlap_problem
from experiments.overlap_identifiability_gate12 import (
    INTERACTION_STRENGTH,
    _context,
    _mixed_difference,
    _quadratic_fit_residual,
)


def test_interaction_strength_is_default_off_and_opt_in() -> None:
    common = dict(
        dimension=24,
        overlap_budget=6,
        min_group_size=3,
        max_group_size=5,
        num_groups=6,
        base_function="sphere",
        conflict_mode="conflicting",
        rotation=False,
        transforms=False,
        seed=31001,
    )
    _, baseline = build_overlap_problem(**common)
    _, interactive = build_overlap_problem(**common, interaction_strength=INTERACTION_STRENGTH)
    point = np.random.default_rng(3).uniform(-2.0, 2.0, size=24)

    assert baseline.config.interaction_strength == 0.0
    assert interactive.config.interaction_strength == INTERACTION_STRENGTH
    assert float(interactive.evaluate(point)) != float(baseline.evaluate(point))


def test_interaction_objective_scalar_and_batch_match() -> None:
    problem, _objective = build_overlap_problem(
        24,
        overlap_budget=6,
        min_group_size=3,
        max_group_size=5,
        num_groups=6,
        base_function="sphere",
        conflict_mode="conflicting",
        rotation=False,
        transforms=False,
        interaction_strength=INTERACTION_STRENGTH,
        seed=31001,
    )
    points = np.random.default_rng(8).uniform(-2.0, 2.0, size=(4, 24))

    batch = np.asarray(problem.objective(points), dtype=float)
    scalar = np.asarray([problem.objective(point) for point in points], dtype=float)
    np.testing.assert_allclose(batch, scalar, rtol=1.0e-12, atol=1.0e-9)


@pytest.mark.parametrize("strength", [-0.1, float("nan"), float("inf")])
def test_invalid_interaction_strength_is_rejected(strength: float) -> None:
    with pytest.raises(ValueError, match="interaction_strength"):
        build_overlap_problem(
            24,
            overlap_budget=6,
            min_group_size=3,
            max_group_size=5,
            num_groups=6,
            base_function="sphere",
            conflict_mode="conflicting",
            interaction_strength=strength,
        )


def test_conforming_and_conflicting_share_interaction_topology() -> None:
    conforming = _context("conforming", "chain", 6, 31001)
    conflicting = _context("conflicting", "chain", 6, 31001)

    assert conforming["groups"] == conflicting["groups"]
    assert conforming["interaction_pairs"] == conflicting["interaction_pairs"]
    assert conforming["shared_variables"] == conflicting["shared_variables"]
    assert conforming["parameter_parity"] is True


def test_mixed_difference_is_nonzero_for_shared_interaction_pair() -> None:
    context = _context("conflicting", "random", 6, 31001)

    assert context["max_abs_mixed_difference"] > 1.0e-6
    assert context["interaction_pair_count"] > 0


def test_quadratic_fit_has_positive_heldout_residual() -> None:
    context = _context("conflicting", "star", 12, 31002)

    assert context["quadratic_fit"]["heldout_rmse"] > 1.0e-6
    assert np.isfinite(context["quadratic_fit"]["heldout_rmse"])


def test_quadratic_fit_helper_rejects_invalid_probe_shape() -> None:
    with np.testing.assert_raises(ValueError):
        _quadratic_fit_residual(lambda x: np.sum(x**2, axis=1), np.zeros((2, 3)), np.zeros((1, 3)))


def test_mixed_difference_helper_rejects_invalid_coordinates() -> None:
    with np.testing.assert_raises(ValueError):
        _mixed_difference(lambda x: float(np.sum(x**2)), np.zeros(3), 0, 0, 0.5)

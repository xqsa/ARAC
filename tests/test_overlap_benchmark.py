from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.overlap_groups import generate_overlap_groups
from arac.benchmarks.overlap_objective import build_overlap_problem


def _edges(structure) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for owners in structure.membership:
        for left in owners:
            for right in owners:
                if left < right:
                    result.add((left, right))
    return result


def test_generated_groups_are_unique_and_budget_exact_across_seeds() -> None:
    for budget in (0, 1, 5, 10, 20, 50):
        for seed in range(20):
            structure = generate_overlap_groups(
                100,
                overlap_budget=budget,
                min_group_size=2,
                max_group_size=8,
                contiguous=False,
                seed=seed,
            )
            assert all(len(group) == len(set(group)) for group in structure.groups)
            assert sum(structure.group_sizes) - 100 == budget
            assert sum(structure.overlap_shares) == budget
            assert all(len(owners) >= 1 for owners in structure.membership)


def test_chain_and_star_topologies_have_only_requested_edges() -> None:
    chain = generate_overlap_groups(
        100,
        overlap_budget=12,
        min_group_size=2,
        max_group_size=8,
        contiguous=False,
        seed=3,
        topology="chain",
    )
    star = generate_overlap_groups(
        100,
        overlap_budget=12,
        min_group_size=2,
        max_group_size=8,
        contiguous=False,
        seed=3,
        topology="star",
    )
    chain_edges = _edges(chain)
    star_edges = _edges(star)
    assert chain_edges
    assert all(right == left + 1 for left, right in chain_edges)
    assert star_edges
    assert all(left == 0 for left, _ in star_edges)


def test_invalid_topology_is_rejected() -> None:
    with pytest.raises(ValueError, match="topology"):
        generate_overlap_groups(
            20,
            overlap_budget=4,
            min_group_size=2,
            max_group_size=5,
            topology="ring",
        )


@pytest.mark.parametrize("base_function", ["sphere", "ackley", "elliptic", "rastrigin", "schwefel"])
def test_objective_scalar_and_batch_evaluations_match(base_function: str) -> None:
    problem, _ = build_overlap_problem(
        30,
        overlap_budget=6,
        min_group_size=2,
        max_group_size=6,
        base_function=base_function,
        conflict_mode="conforming",
        rotation=True,
        transforms=True,
        seed=3,
    )
    points = np.random.default_rng(9).uniform(-10.0, 10.0, size=(4, 30))
    batch = np.asarray(problem.objective(points), dtype=float)
    scalar = np.asarray([problem.objective(point) for point in points], dtype=float)
    assert batch.shape == (4,)
    np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=1e-8)


def test_schwefel_uses_cumulative_square_definition() -> None:
    problem, objective = build_overlap_problem(
        12,
        overlap_budget=0,
        min_group_size=3,
        max_group_size=4,
        base_function="schwefel",
        conflict_mode="conforming",
        bounds=10.0,
        rotation=False,
        transforms=False,
        seed=4,
    )
    point = objective.optimum_point().copy()
    group = objective.structure.groups[0]
    shifted = np.zeros(len(group))
    shifted[:] = 1.0
    for variable, value in zip(group, shifted, strict=True):
        point[variable] = objective._optima[0][list(group).index(variable)] + value
    contributions = objective.per_group_contribution(point)
    expected = float(np.sum(np.cumsum(shifted) ** 2) * objective._weights[0])
    assert contributions[0] == pytest.approx(expected)


def test_sphere_and_elliptic_are_distinct_and_conflict_bound_is_marked() -> None:
    common = dict(
        dimension=12,
        overlap_budget=3,
        min_group_size=3,
        max_group_size=4,
        bounds=10.0,
        rotation=False,
        transforms=False,
        seed=4,
    )
    sphere, conforming = build_overlap_problem(
        base_function="sphere", conflict_mode="conforming", **common
    )
    elliptic, _ = build_overlap_problem(
        base_function="elliptic", conflict_mode="conforming", **common
    )
    _, conflicting = build_overlap_problem(
        base_function="sphere", conflict_mode="conflicting", **common
    )
    point = np.full(12, 1.0)
    assert sphere.objective(point) != elliptic.objective(point)
    assert conforming.optimum_is_attainable is True
    assert conflicting.optimum == 0.0
    assert conflicting.optimum_is_attainable is False


def test_conforming_and_conflicting_instances_share_non_conflict_parameters() -> None:
    common = dict(
        dimension=24,
        overlap_budget=6,
        min_group_size=3,
        max_group_size=5,
        num_groups=6,
        base_function="rastrigin",
        bounds=10.0,
        contiguous=False,
        topology="chain",
        rotation=True,
        transforms=True,
        seed=17,
    )
    _, conforming = build_overlap_problem(conflict_mode="conforming", **common)
    _, conflicting = build_overlap_problem(conflict_mode="conflicting", **common)

    assert conforming.structure.groups == conflicting.structure.groups
    np.testing.assert_array_equal(conforming._weights, conflicting._weights)
    for left, right in zip(conforming._rotations, conflicting._rotations, strict=True):
        np.testing.assert_array_equal(left, right)

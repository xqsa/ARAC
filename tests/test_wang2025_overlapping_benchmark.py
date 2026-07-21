from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from arac.benchmarks import (
    WANG2025_MAX_SHARED_MEMBERSHIPS,
    Wang2025OverlappingProblem,
    Wang2025OverlappingSpec,
)


def test_manual_overlap_problem_matches_wang_matlab_values() -> None:
    problem = Wang2025OverlappingProblem(
        spec=Wang2025OverlappingSpec(
            dimension=4,
            min_group_size=1,
            max_group_size=3,
            alpha=0.5,
            overlap_count=1,
            beta=1.0,
            gamma=1.0,
        ),
        template=(0, 0, 0, 0),
        groups=((0, 1), (1, 2), (3,)),
        base_owner_by_variable=(0, 0, 1, 2),
    )
    candidates = np.asarray(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
        ]
    )

    np.testing.assert_allclose(problem.legacy_objective(candidates), [-5.0, -4.0, -1.12])
    np.testing.assert_allclose(problem(candidates), [0.0, 1.0, 3.88])
    assert problem.shared_variables == (1,)
    assert problem.overlap_relations == ((1, 0, 1),)
    assert problem.global_optimum == (1, 1, 1, 1)


@pytest.mark.parametrize("overlap_count", [10, 20, 30])
def test_generated_overlap_degree_is_exact(overlap_count: int) -> None:
    spec = Wang2025OverlappingSpec(
        dimension=100,
        min_group_size=2,
        max_group_size=5,
        alpha=0.8,
        overlap_count=overlap_count,
        beta=0.4,
        gamma=0.4,
        permuted=True,
        seed=20260721 + overlap_count,
    )
    first = Wang2025OverlappingProblem.generate(spec)
    second = Wang2025OverlappingProblem.generate(spec)

    assert first == second
    assert first.instance_hash == second.instance_hash
    assert len(first.shared_variables) == overlap_count
    assert len(first.overlap_relations) == overlap_count
    assert first.spec.overlap_ratio == pytest.approx(overlap_count / 100)
    assert sum(first.group_sizes) == 100 + overlap_count
    assert min(first.group_sizes) >= 2
    assert max(first.group_sizes) <= 5

    memberships = np.zeros(100, dtype=int)
    for group in first.groups:
        memberships[np.asarray(group)] += 1
    assert np.count_nonzero(memberships == 2) == overlap_count
    assert np.count_nonzero(memberships == 1) == 100 - overlap_count
    assert memberships.max() == WANG2025_MAX_SHARED_MEMBERSHIPS
    np.testing.assert_allclose(first(first.global_optimum), [0.0], atol=1e-12)

    sources_by_target: dict[int, set[int]] = {}
    load_by_target: dict[int, int] = {}
    for _, source, target in first.overlap_relations:
        sources_by_target.setdefault(target, set()).add(source)
        load_by_target[target] = load_by_target.get(target, 0) + 1
    for target, sources in sources_by_target.items():
        load = load_by_target[target]
        expected = max(1, min(load, len(first.groups) - 1, math.floor(load * 0.4 + 0.5)))
        assert len(sources) == expected

    concentrated_load = max(1, math.floor(overlap_count * 0.4 + 0.5))
    desired_target_count = math.ceil(overlap_count / concentrated_load)
    minimum_target_count = 0
    capacity_total = 0
    for capacity in sorted((size - 1 for size in first.group_sizes), reverse=True):
        capacity_total += capacity
        minimum_target_count += 1
        if capacity_total >= overlap_count:
            break
    assert len(sources_by_target) == max(desired_target_count, minimum_target_count)


def test_aob_scale_overlap_instance_is_1000_dimensional() -> None:
    problem = Wang2025OverlappingProblem.generate(
        Wang2025OverlappingSpec(
            dimension=1000,
            min_group_size=2,
            max_group_size=5,
            alpha=0.8,
            overlap_count=300,
            beta=0.4,
            gamma=0.4,
            permuted=True,
            seed=20260721,
        )
    )

    assert problem.dimension == 1000
    assert problem.spec.overlap_ratio == pytest.approx(0.3)
    assert len(problem.shared_variables) == 300
    np.testing.assert_allclose(problem(problem.global_optimum), [0.0], atol=1e-12)


def test_equal_grouping_preserves_final_group_size_with_overlap() -> None:
    problem = Wang2025OverlappingProblem.generate(
        Wang2025OverlappingSpec(
            dimension=100,
            min_group_size=5,
            max_group_size=5,
            alpha=0.1,
            overlap_count=30,
            beta=0.8,
            gamma=0.4,
            seed=9,
        )
    )

    assert len(problem.groups) == 26
    assert problem.group_sizes == (5,) * 26
    assert len(problem.shared_variables) == 30


def test_manifest_round_trip_and_hash_guard() -> None:
    problem = Wang2025OverlappingProblem.generate(
        Wang2025OverlappingSpec(100, 2, 5, 0.5, 20, seed=31)
    )
    manifest = problem.to_manifest()

    assert Wang2025OverlappingProblem.from_manifest(manifest) == problem

    modified = copy.deepcopy(manifest)
    modified["template"][0] = 1 - modified["template"][0]
    with pytest.raises(ValueError, match="hash mismatch"):
        Wang2025OverlappingProblem.from_manifest(modified)


def test_zero_overlap_remains_a_true_partition() -> None:
    problem = Wang2025OverlappingProblem.generate(
        Wang2025OverlappingSpec(100, 2, 5, 0.0, 0, seed=4)
    )

    assert problem.shared_variables == ()
    assert problem.overlap_relations == ()
    assert sorted(variable for group in problem.groups for variable in group) == list(range(100))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (
            {
                "dimension": 10,
                "min_group_size": 2,
                "max_group_size": 5,
                "alpha": 0.5,
                "overlap_count": 11,
            },
            "overlap_count",
        ),
        (
            {
                "dimension": 10,
                "min_group_size": 2,
                "max_group_size": 5,
                "alpha": 0.5,
                "overlap_count": 2,
                "beta": 0.0,
            },
            "beta",
        ),
        (
            {
                "dimension": 10,
                "min_group_size": 2,
                "max_group_size": 5,
                "alpha": 0.5,
                "overlap_count": 2,
                "gamma": 1.1,
            },
            "gamma",
        ),
        (
            {
                "dimension": 10,
                "min_group_size": 3,
                "max_group_size": 3,
                "alpha": 0.5,
                "overlap_count": 1,
            },
            "divisible",
        ),
    ],
)
def test_invalid_specs_fail_closed(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        Wang2025OverlappingSpec(**kwargs)


def test_invalid_candidate_is_not_silently_rounded() -> None:
    problem = Wang2025OverlappingProblem.generate(Wang2025OverlappingSpec(20, 2, 5, 0.5, 4, seed=7))
    with pytest.raises(ValueError, match="only 0 and 1"):
        problem(np.full(20, 0.5))

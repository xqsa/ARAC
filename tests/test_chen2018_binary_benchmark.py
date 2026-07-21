from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from arac.benchmarks import Chen2018BinaryProblem, Chen2018Spec


def test_equal_contiguous_problem_matches_original_matlab_values() -> None:
    problem = Chen2018BinaryProblem.generate(
        Chen2018Spec(
            dimension=6,
            min_group_size=3,
            max_group_size=3,
            alpha=0.5,
            seed=17,
        ),
        template=np.zeros(6, dtype=int),
    )
    candidates = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ]
    )

    np.testing.assert_allclose(problem.legacy_objective(candidates), [-6.0, -5.4, -3.45])
    np.testing.assert_allclose(problem(candidates), [0.0, 0.6, 2.55])
    assert problem.groups == ((0, 1, 2), (3, 4, 5))
    assert problem.info()["best"] == 0.0


def test_zero_deception_error_is_hamming_distance_from_template() -> None:
    problem = Chen2018BinaryProblem.generate(
        Chen2018Spec(8, 2, 2, alpha=0.0, seed=5),
        template=[0, 1, 0, 1, 0, 1, 0, 1],
    )
    candidate = np.asarray([0, 0, 0, 1, 1, 1, 0, 0])

    np.testing.assert_array_equal(problem(candidate), [3.0])
    np.testing.assert_array_equal(problem.group_errors(candidate), [[1.0, 0.0, 1.0, 1.0]])


def test_1000d_permuted_instance_is_frozen_and_reproducible() -> None:
    spec = Chen2018Spec(1000, 2, 5, alpha=0.8, permuted=True, seed=20260721)
    first = Chen2018BinaryProblem.generate(spec)
    second = Chen2018BinaryProblem.generate(spec)

    assert first == second
    assert first.instance_hash == second.instance_hash
    assert len(first.groups) == math.ceil(2 * 1000 / 7)
    assert sum(first.group_sizes) == 1000
    assert min(first.group_sizes) >= 2
    assert max(first.group_sizes) <= 5
    assert sorted(index for group in first.groups for index in group) == list(range(1000))

    optimum = np.asarray(first.template)
    complement = 1 - optimum
    np.testing.assert_allclose(first(optimum), [0.0], atol=1e-12)
    np.testing.assert_allclose(first(complement), [100.0], atol=1e-12)
    np.testing.assert_array_equal(first(optimum), first(optimum))


def test_manifest_round_trip_and_hash_guard() -> None:
    problem = Chen2018BinaryProblem.generate(
        Chen2018Spec(24, 2, 5, alpha=0.1, permuted=True, seed=11)
    )
    manifest = problem.to_manifest()

    assert Chen2018BinaryProblem.from_manifest(manifest) == problem

    modified = copy.deepcopy(manifest)
    modified["template"][0] = 1 - modified["template"][0]
    with pytest.raises(ValueError, match="hash mismatch"):
        Chen2018BinaryProblem.from_manifest(modified)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"dimension": 0, "min_group_size": 2, "max_group_size": 5, "alpha": 0.5}, "positive"),
        ({"dimension": 7, "min_group_size": 2, "max_group_size": 2, "alpha": 0.5}, "divisible"),
        ({"dimension": 10, "min_group_size": 5, "max_group_size": 2, "alpha": 0.5}, "group sizes"),
        ({"dimension": 10, "min_group_size": 2, "max_group_size": 5, "alpha": 0.9}, "alpha"),
    ],
)
def test_invalid_specs_fail_closed(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        Chen2018Spec(**kwargs)


@pytest.mark.parametrize(
    "candidate",
    [
        [0, 1, 0],
        [[0, 1, 0, 1], [0, 1, 0]],
        [0, 1, 0, 2],
        [0.0, 1.0, float("nan"), 1.0],
        ["0", "1", "0", "1"],
    ],
)
def test_invalid_candidates_fail_closed(candidate: object) -> None:
    problem = Chen2018BinaryProblem.generate(Chen2018Spec(4, 2, 2, alpha=0.5))
    with pytest.raises(ValueError, match="candidate"):
        problem(candidate)

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from arac.benchmarks import (
    WANG2025_CONTINUOUS_SCHEMA_VERSION,
    WANG2025_LOCAL_ESCAPE_CASES,
    Wang2025ContinuousProblem,
    Wang2025OverlappingProblem,
    Wang2025OverlappingSpec,
)


def _manual_problem(
    *,
    alpha: float = 0.5,
    conflict_ratio: float = 0.0,
) -> Wang2025OverlappingProblem:
    return Wang2025OverlappingProblem(
        spec=Wang2025OverlappingSpec(
            dimension=4,
            min_group_size=1,
            max_group_size=3,
            alpha=alpha,
            overlap_count=1,
            beta=1.0,
            gamma=1.0,
            conflict_ratio=conflict_ratio,
        ),
        template=(0, 0, 0, 0),
        groups=((0, 1), (1, 2), (3,)),
        base_owner_by_variable=(0, 0, 1, 2),
        conflict_target_by_variable=(-1, 1, -1, -1) if conflict_ratio else (),
    )


def _pair_signal(problem: Wang2025ContinuousProblem, first: int, second: int) -> float:
    anchor = np.zeros(problem.dimension)
    first_only = anchor.copy()
    second_only = anchor.copy()
    both = anchor.copy()
    first_only[first] = 0.5
    second_only[second] = 0.5
    both[[first, second]] = 0.5
    values = problem(np.vstack((anchor, first_only, second_only, both)))
    return float(abs((values[1] - values[0]) - (values[3] - values[2])))


def _rdg3_pair_signal(problem: Wang2025ContinuousProblem, first: int, second: int) -> float:
    anchor = np.zeros(problem.dimension)
    first_upper = anchor.copy()
    second_middle = anchor.copy()
    upper_and_middle = anchor.copy()
    first_upper[first] = 1.0
    second_middle[second] = 0.5
    upper_and_middle[first] = 1.0
    upper_and_middle[second] = 0.5
    values = problem(np.vstack((anchor, first_upper, second_middle, upper_and_middle)))
    return float(abs((values[0] - values[1]) - (values[2] - values[3])))


@pytest.mark.parametrize("conflict_ratio", [0.0, 1.0])
def test_continuous_extension_matches_every_binary_vertex(conflict_ratio: float) -> None:
    source = _manual_problem(conflict_ratio=conflict_ratio)
    problem = Wang2025ContinuousProblem(source)
    vertices = np.asarray(tuple(product((0.0, 1.0), repeat=source.dimension)))

    np.testing.assert_allclose(problem(vertices), source(vertices), rtol=0.0, atol=1e-12)
    assert problem.source_instance_hash == source.instance_hash
    assert problem.objective_hash != source.instance_hash
    assert WANG2025_CONTINUOUS_SCHEMA_VERSION == "wang2025-continuous-interaction-v2"
    assert problem.info()["schema_version"] == WANG2025_CONTINUOUS_SCHEMA_VERSION
    assert problem.info()["encoding"] == "continuous"


@pytest.mark.parametrize("case_index", [0, 6, 12])
def test_continuous_extension_preserves_deceptive_local_and_global_optima(
    case_index: int,
) -> None:
    source = WANG2025_LOCAL_ESCAPE_CASES[case_index].generate()
    problem = Wang2025ContinuousProblem(source)
    local = np.asarray(source.template, dtype=float)
    global_optimum = 1.0 - local
    toward_global = local + 1e-3 * (global_optimum - local)

    assert problem(toward_global)[0] > problem(local)[0]
    assert problem(global_optimum)[0] == pytest.approx(0.0, abs=1e-12)
    assert problem(np.full(source.dimension, 0.5))[0] > 0.0
    np.testing.assert_allclose(problem.global_optimum, global_optimum)


def test_pair_coupling_exposes_exact_direct_interactions() -> None:
    problem = Wang2025ContinuousProblem(_manual_problem(alpha=0.0))
    expected = problem.expected_interaction_matrix()

    assert expected.dtype == np.bool_
    assert np.all(np.diag(expected))
    assert expected[0, 1] and expected[1, 0]
    assert expected[1, 2] and expected[2, 1]
    assert not expected[0, 2] and not expected[2, 0]
    assert _pair_signal(problem, 0, 1) > 1e-6
    assert _pair_signal(problem, 1, 2) > 1e-6
    assert _pair_signal(problem, 0, 2) == pytest.approx(0.0, abs=1e-12)
    assert _rdg3_pair_signal(problem, 0, 1) > 1e-6
    assert _rdg3_pair_signal(problem, 1, 2) > 1e-6
    assert _rdg3_pair_signal(problem, 0, 2) == pytest.approx(0.0, abs=1e-12)


def test_continuous_candidates_are_batched_and_fail_closed_at_domain_boundary() -> None:
    problem = Wang2025ContinuousProblem(_manual_problem())
    candidate = np.asarray([0.1, 0.2, 0.3, 0.4])

    assert problem(candidate).shape == (1,)
    np.testing.assert_allclose(problem(np.vstack((candidate, candidate))), problem(candidate)[0])
    for invalid in (
        np.asarray([-1e-12, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, 1.0 + 1e-12]),
        np.asarray([0.0, np.nan, 0.0, 0.0]),
        np.asarray([0.0, np.inf, 0.0, 0.0]),
    ):
        with pytest.raises(ValueError, match=r"finite values in.*\[0, 1\]"):
            problem(invalid)
    with pytest.raises(ValueError, match="numeric"):
        problem(["0", "0", "0", "0"])
    with pytest.raises(ValueError, match="shape"):
        problem(np.zeros(3))


def test_1000_dimensional_catalog_case_keeps_topology_and_gets_a_stable_objective_hash() -> None:
    source = WANG2025_LOCAL_ESCAPE_CASES[-1].generate()
    first = Wang2025ContinuousProblem(source)
    second = Wang2025ContinuousProblem(source)
    interactions = first.expected_interaction_matrix()

    assert first.dimension == 1000
    assert first.objective_hash == second.objective_hash
    assert first.groups == source.groups
    assert interactions.shape == (1000, 1000)
    expected = np.eye(source.dimension, dtype=bool)
    for group in source.groups:
        indices = np.asarray(group, dtype=int)
        expected[np.ix_(indices, indices)] = True
    np.testing.assert_array_equal(interactions, expected)

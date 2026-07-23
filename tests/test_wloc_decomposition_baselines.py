from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from arac.baselines import (
    GroupingResult,
    design_matrix_from_groups,
    dg2_grouping,
    random_grouping,
    rddsm_grouping,
    rdg3_grouping,
    run_cooperative_cmaes,
)


class CountingObjective:
    def __init__(self, function) -> None:
        self.function = function
        self.evaluations = 0

    def __call__(self, candidate: np.ndarray) -> float:
        self.evaluations += 1
        return float(self.function(np.asarray(candidate, dtype=float)))


def _component_sets(result: GroupingResult) -> set[frozenset[int]]:
    return {frozenset(group) for group in result.groups}


def test_grouping_contract_validates_coverage_overlap_and_read_only_matrix() -> None:
    matrix = np.eye(3, dtype=bool)
    result = GroupingResult(
        method="fixture",
        dimension=3,
        groups=((0, 1), (1, 2)),
        decomposition_fes=7,
        allows_overlap=True,
        origin="test",
        matrix=matrix,
        matrix_kind="interaction",
    )

    assert len(result.grouping_hash) == 64
    assert result.grouping_hash == GroupingResult(
        method="fixture",
        dimension=3,
        groups=((0, 1), (1, 2)),
        decomposition_fes=7,
        allows_overlap=True,
        origin="test",
        matrix=matrix,
        matrix_kind="interaction",
    ).grouping_hash
    with pytest.raises(ValueError):
        result.matrix[0, 0] = False
    with pytest.raises(FrozenInstanceError):
        result.groups = ((0,), (1,), (2,))
    with pytest.raises(ValueError, match="overlap is not allowed"):
        GroupingResult(
            method="invalid",
            dimension=3,
            groups=((0, 1), (1, 2)),
            decomposition_fes=0,
            allows_overlap=False,
            origin="test",
        )
    with pytest.raises(ValueError, match="cover every"):
        GroupingResult(
            method="invalid",
            dimension=3,
            groups=((0,), (1,)),
            decomposition_fes=0,
            allows_overlap=False,
            origin="test",
        )


def test_dg2_finds_connected_components_and_counts_every_probe() -> None:
    objective = CountingObjective(
        lambda x: np.sum(x) + 4.0 * x[0] * x[1] + 3.0 * x[2] * x[3]
    )

    result = dg2_grouping(objective, 5)

    assert _component_sets(result) == {
        frozenset((0, 1)),
        frozenset((2, 3)),
        frozenset((4,)),
    }
    assert result.decomposition_fes == 1 + 5 + 5 * 4 // 2
    assert objective.evaluations == result.decomposition_fes
    assert result.matrix_kind == "interaction"
    assert result.matrix[0, 1]
    assert not result.matrix[0, 2]


def test_rdg3_breaks_an_overlap_link_at_the_nonseparable_threshold() -> None:
    objective = CountingObjective(
        lambda x: np.sum(x)
        + 5.0 * x[0] * x[1]
        + 5.0 * x[1] * x[2]
        + 5.0 * x[3] * x[4]
    )

    result = rdg3_grouping(
        objective,
        5,
        nonseparable_threshold=2,
        separable_chunk_size=1,
    )

    assert _component_sets(result) == {
        frozenset((0, 1)),
        frozenset((2,)),
        frozenset((3, 4)),
    }
    assert objective.evaluations == result.decomposition_fes
    assert result.decomposition_fes > 1


def test_random_baseline_builds_twenty_seeded_disjoint_subspaces() -> None:
    first = random_grouping(1000, seed=20260723)
    second = random_grouping(1000, seed=20260723)
    changed = random_grouping(1000, seed=20260724)

    assert len(first.groups) == 20
    assert {len(group) for group in first.groups} == {50}
    assert sorted(index for group in first.groups for index in group) == list(range(1000))
    assert first.grouping_hash == second.grouping_hash
    assert first.grouping_hash != changed.grouping_hash
    assert first.decomposition_fes == 0


def test_rddsm_recovers_overlapping_groups_from_membership_design() -> None:
    expected = ((0, 1), (1, 2), (3,))
    design = design_matrix_from_groups(4, expected)

    result = rddsm_grouping(design)

    assert _component_sets(result) == {frozenset(group) for group in expected}
    assert result.allows_overlap is True
    assert result.decomposition_fes == 0
    assert result.matrix_kind == "design"
    np.testing.assert_array_equal(result.matrix, design)


def test_1000d_rddsm_structural_decomposition_keeps_all_variables() -> None:
    groups = tuple(tuple(range(start, start + 50)) for start in range(0, 1000, 50))
    result = rddsm_grouping(design_matrix_from_groups(1000, groups))

    assert result.dimension == 1000
    assert _component_sets(result) == {frozenset(group) for group in groups}


def test_cooperative_cmaes_is_deterministic_bounded_and_exactly_budgeted() -> None:
    grouping = GroupingResult(
        method="fixture",
        dimension=4,
        groups=((0, 1), (2, 3)),
        decomposition_fes=17,
        allows_overlap=False,
        origin="test",
    )

    def bounded_objective(candidate: np.ndarray) -> float:
        values = np.asarray(candidate, dtype=float)
        assert np.all((0.0 <= values) & (values <= 1.0))
        return float(np.sum(np.square(values - 0.2)))

    first = run_cooperative_cmaes(
        bounded_objective,
        grouping,
        max_function_evaluations=31,
        seed=117,
        sigma=0.8,
        group_block_fes=7,
    )
    second = run_cooperative_cmaes(
        bounded_objective,
        grouping,
        max_function_evaluations=31,
        seed=117,
        sigma=0.8,
        group_block_fes=7,
    )

    assert first.result_hash == second.result_hash
    assert first.optimization_fes == 31
    assert first.decomposition_fes == 17
    assert len(first.best_so_far_trace) == 31
    assert sum(count for _, count in first.phase_fes) == 31
    assert first.phase_fes[0] == ("initial_context", 1)
    assert first.repair_policy == "clip_to_bounds"
    assert first.repaired_candidate_count > 0
    assert np.all((0.0 <= np.asarray(first.best_x)) & (np.asarray(first.best_x) <= 1.0))
    assert first.best_y <= bounded_objective(np.full(4, 0.5))

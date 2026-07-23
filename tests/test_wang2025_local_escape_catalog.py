from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from arac.benchmarks import (
    WANG2025_LOCAL_ESCAPE_CASES,
    WANG2025_LOCAL_ESCAPE_SUITE_VERSION,
    Wang2025OverlappingProblem,
    get_wang2025_local_escape_case,
)


def test_catalog_matches_the_18_case_paper_grid_at_1000_dimensions() -> None:
    cases = WANG2025_LOCAL_ESCAPE_CASES

    assert WANG2025_LOCAL_ESCAPE_SUITE_VERSION == "wang2025-local-escape-screening-v1"
    assert len(cases) == 18
    assert [case.case_id for case in cases] == [f"WLOC{index:02d}" for index in range(1, 19)]
    assert [case.paper_function_id for case in cases] == [f"f{index}" for index in range(1, 19)]
    assert Counter(case.spec.alpha for case in cases) == {0.1: 6, 0.5: 6, 0.8: 6}
    assert Counter(case.grouping_mode for case in cases) == {"equal": 9, "unequal": 9}
    assert Counter(case.overlap_percent for case in cases) == {10: 6, 20: 6, 30: 6}

    for case in cases:
        assert case.spec.dimension == 1000
        assert case.spec.overlap_count == case.overlap_percent * 10
        assert case.spec.overlap_ratio == pytest.approx(case.overlap_percent / 100)
        assert case.spec.beta == pytest.approx(0.5)
        assert case.spec.gamma == pytest.approx(0.5)
        assert case.spec.conflict_ratio == 0.0
        assert case.spec.permuted is False
        if case.grouping_mode == "equal":
            assert (case.spec.min_group_size, case.spec.max_group_size) == (5, 5)
        else:
            assert (case.spec.min_group_size, case.spec.max_group_size) == (2, 5)


@pytest.mark.parametrize("case", WANG2025_LOCAL_ESCAPE_CASES, ids=lambda case: case.case_id)
def test_catalog_case_generates_the_frozen_valid_instance(case) -> None:
    problem = case.generate()

    assert problem.instance_hash == case.expected_instance_hash
    assert problem.dimension == 1000
    assert len(problem.shared_variables) == case.spec.overlap_count
    assert problem.conflicting_variables == ()
    assert problem.global_optimum is not None
    assert sum(problem.group_sizes) == problem.dimension + case.spec.overlap_count
    np.testing.assert_allclose(problem(problem.global_optimum), [0.0], atol=1e-12)
    assert Wang2025OverlappingProblem.from_manifest(problem.to_manifest()) == problem


@pytest.mark.parametrize(
    ("indices", "expected_seed"),
    [
        ((0, 6, 12), 2026072301),
        ((1, 7, 13), 2026072302),
        ((2, 8, 14), 2026072303),
        ((3, 9, 15), 2026072304),
        ((4, 10, 16), 2026072305),
        ((5, 11, 17), 2026072306),
    ],
)
def test_alpha_triplets_change_only_the_deception_parameter(
    indices: tuple[int, int, int],
    expected_seed: int,
) -> None:
    cases = [WANG2025_LOCAL_ESCAPE_CASES[index] for index in indices]
    problems = [case.generate() for case in cases]

    assert [case.spec.alpha for case in cases] == [0.1, 0.5, 0.8]
    assert {case.spec.seed for case in cases} == {expected_seed}
    assert problems[0].template == problems[1].template == problems[2].template
    assert problems[0].groups == problems[1].groups == problems[2].groups
    assert (
        problems[0].base_owner_by_variable
        == problems[1].base_owner_by_variable
        == problems[2].base_owner_by_variable
    )


def test_catalog_lookup_accepts_suite_and_paper_ids() -> None:
    assert get_wang2025_local_escape_case("WLOC01") is WANG2025_LOCAL_ESCAPE_CASES[0]
    assert get_wang2025_local_escape_case(" f18 ") is WANG2025_LOCAL_ESCAPE_CASES[-1]
    with pytest.raises(KeyError, match="unknown Wang 2025 local-escape case"):
        get_wang2025_local_escape_case("R4")


def test_catalog_hash_guard_rejects_generator_drift() -> None:
    changed = replace(
        WANG2025_LOCAL_ESCAPE_CASES[0],
        expected_instance_hash="0" * 64,
    )

    with pytest.raises(RuntimeError, match="frozen catalog"):
        changed.generate()

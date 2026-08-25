from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.benchmarks.overlap24 import (
    OVERLAP24_BASE_FUNCTIONS,
    OVERLAP24_DEFAULT_SEED,
    OVERLAP24_DIMENSION,
    OVERLAP24_RECIPES,
    Overlap24Benchmark,
    overlap24_cases,
)


def test_overlap24_matrix_is_fixed_and_balanced() -> None:
    cases = overlap24_cases()
    assert len(cases) == 24
    assert tuple(case.case_id for case in cases) == tuple(f"O{index:02d}" for index in range(1, 25))
    assert len({case.case_id for case in cases}) == 24
    assert {case.base_function for case in cases} == set(OVERLAP24_BASE_FUNCTIONS)
    assert all(sum(case.base_function == family for case in cases) == 6 for family in OVERLAP24_BASE_FUNCTIONS)
    assert sum(case.conflict_mode == "conforming" for case in cases) == 12
    assert sum(case.conflict_mode == "conflicting" for case in cases) == 12
    assert {case.recipe for case in cases} == {recipe.name for recipe in OVERLAP24_RECIPES}


def test_instance_seed_is_shared_across_matched_base_families() -> None:
    cases = overlap24_cases()
    for recipe in OVERLAP24_RECIPES:
        recipe_rows = [case for case in cases if case.recipe == recipe.name]
        assert len(recipe_rows) == 4
        assert len({case.instance_seed for case in recipe_rows}) == 1


def test_run_seed_is_not_part_of_function_identity() -> None:
    benchmark = Overlap24Benchmark()
    problem_a, truth_a, spec_a = benchmark.load_with_truth("O01")
    problem_b, truth_b, spec_b = benchmark.load_with_truth("O01")
    assert spec_a.instance_seed == spec_b.instance_seed
    assert truth_a.structure.groups == truth_b.structure.groups
    points = np.random.default_rng(17).uniform(-2.0, 2.0, size=(2, OVERLAP24_DIMENSION))
    np.testing.assert_allclose(problem_a.objective(points), problem_b.objective(points))

    # The benchmark API intentionally has no run_seed parameter: it is supplied
    # to the stochastic algorithm, not used to regenerate the fixed function.
    assert "run_seed" not in {field.name for field in fields(spec_a)}


@pytest.mark.parametrize("case_id", tuple(f"O{index:02d}" for index in range(1, 25)))
def test_case_surface_and_manifest(case_id: str) -> None:
    benchmark = Overlap24Benchmark()
    problem, truth, spec = benchmark.load_with_truth(case_id)
    manifest = benchmark.truth_manifest(case_id)

    assert isinstance(problem, OptimizationProblem)
    assert problem.dimension == OVERLAP24_DIMENSION
    assert len(truth.structure.groups) == benchmark._num_groups
    assert manifest["schema_version"] == "arac-overlap24-cross-suite-v1"
    assert manifest["case_id"] == case_id
    assert manifest["dimension"] == OVERLAP24_DIMENSION
    assert manifest["overlap_slots"] == spec.overlap_budget
    assert manifest["shared_variable_count"] > 0
    assert manifest["component_count"] >= 1
    assert manifest["graph_connected"] is (manifest["component_count"] == 1)
    assert manifest["optimum"] == 0.0
    assert manifest["optimum_is_attainable"] is (spec.conflict_mode == "conforming")

    # The runtime surface remains identity-free; structure/truth is offline only.
    assert {field.name for field in fields(problem)} == {
        "objective",
        "dimension",
        "lower_bounds",
        "upper_bounds",
        "optimum",
    }


def test_manifest_uses_realised_graph_for_structured_recipes() -> None:
    benchmark = Overlap24Benchmark()
    by_mode: dict[str, list[dict[str, object]]] = {"conforming": [], "conflicting": []}
    for case in benchmark.cases:
        manifest = benchmark.truth_manifest(case.case_id)
        by_mode[case.conflict_mode].append(manifest)
        edges = [tuple(edge) for edge in manifest["graph_edges"]]
        if case.topology == "chain":
            assert all(right == left + 1 for left, right in edges)
        elif case.topology == "star":
            assert all(left == 0 for left, _ in edges)
        assert all(left < right for left, right in edges)
        assert manifest["component_count"] >= 1

    # The matrix must exercise both structural route regimes in each conflict
    # mode.  Route selection itself still consumes discovered evidence; this
    # assertion only checks that the benchmark contains the required truth
    # cases for an offline audit.
    for manifests in by_mode.values():
        components = {int(manifest["component_count"]) for manifest in manifests}
        assert 1 in components
        assert any(component_count > 1 for component_count in components)


def test_objective_scalar_and_batch_match() -> None:
    problem = Overlap24Benchmark(rotation=False, transforms=False).load("O03")
    points = np.random.default_rng(19).uniform(-10.0, 10.0, size=(3, OVERLAP24_DIMENSION))
    batch = np.asarray(problem.objective(points), dtype=float)
    scalar = np.asarray([problem.objective(point) for point in points], dtype=float)
    np.testing.assert_allclose(batch, scalar, rtol=1.0e-12, atol=1.0e-8)


def test_suite_seed_releases_a_different_fixed_instance() -> None:
    default = Overlap24Benchmark(suite_seed=OVERLAP24_DEFAULT_SEED)
    alternate = Overlap24Benchmark(suite_seed=OVERLAP24_DEFAULT_SEED + 1)
    assert default.spec("O01").instance_seed != alternate.spec("O01").instance_seed
    p1, _t1, _s1 = default.load_with_truth("O01")
    p2, _t2, _s2 = alternate.load_with_truth("O01")
    point = np.zeros(OVERLAP24_DIMENSION)
    assert float(p1.objective(point)) != float(p2.objective(point))


def test_unknown_case_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown overlap24 case_id"):
        Overlap24Benchmark().load("A1")

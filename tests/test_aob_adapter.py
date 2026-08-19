from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from arac.benchmarks.aob import AobBenchmark, OptimizationProblem


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("A4", 4865.544687570594),
        ("R4", 462974.34727924026),
        ("S5", 25792.95204593285),
    ],
)
def test_aob_adapter_preserves_benchmark_values(case_id: str, expected: float) -> None:
    problem = AobBenchmark().load(case_id)
    objective = problem.objective
    candidate = np.clip(
        objective.Ovector + np.linspace(-0.25, 0.25, problem.dimension),
        problem.lower_array,
        problem.upper_array,
    )

    assert problem.dimension == 1000
    assert float(objective(candidate)[0]) == pytest.approx(expected, rel=1e-13)


def test_problem_surface_contains_no_benchmark_identity_or_topology() -> None:
    names = {field.name for field in fields(OptimizationProblem)}

    assert names == {"objective", "dimension", "lower_bounds", "upper_bounds", "optimum"}
    assert not names & {"case_id", "family", "groups", "permutation", "overlap"}

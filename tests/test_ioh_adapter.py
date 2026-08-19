from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.benchmarks.ioh_bbob import IohBbobBenchmark

ioh = pytest.importorskip("ioh")


def test_ioh_bbob_adapter_preserves_error_values_and_batches() -> None:
    function_id = 8
    instance = 11
    dimension = 40
    direct = ioh.get_problem(
        function_id,
        instance=instance,
        dimension=dimension,
        problem_class=ioh.ProblemClass.BBOB,
    )
    adapted = IohBbobBenchmark().load(
        function_id,
        instance=instance,
        dimension=dimension,
    )
    candidates = np.vstack((np.zeros(dimension), np.ones(dimension)))

    expected = np.maximum(
        np.asarray(direct(candidates), dtype=float) - float(direct.optimum.y),
        0.0,
    )

    assert np.asarray(adapted.objective(candidates)).tolist() == pytest.approx(
        expected.tolist()
    )
    assert adapted.objective(candidates[0]) == pytest.approx(expected[0])
    assert adapted.objective(np.asarray(direct.optimum.x, dtype=float)) == pytest.approx(0.0)
    assert adapted.optimum == 0.0


def test_ioh_problem_surface_contains_no_benchmark_identity() -> None:
    problem = IohBbobBenchmark().load(1, instance=11, dimension=40)
    names = {field.name for field in fields(OptimizationProblem)}

    assert problem.dimension == 40
    assert problem.lower_bounds == (-5.0,) * 40
    assert problem.upper_bounds == (5.0,) * 40
    assert not names & {"function_id", "instance", "suite", "family", "case_id"}


@pytest.mark.parametrize(
    ("function_id", "instance", "dimension"),
    ((0, 1, 40), (25, 1, 40), (1, 0, 40), (1, 1, 0)),
)
def test_ioh_adapter_rejects_invalid_instance_coordinates(
    function_id: int,
    instance: int,
    dimension: int,
) -> None:
    with pytest.raises(ValueError):
        IohBbobBenchmark().load(
            function_id,
            instance=instance,
            dimension=dimension,
        )

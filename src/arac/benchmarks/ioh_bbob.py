"""Identity-blind adapter for the IOH BBOB benchmark suite."""

from __future__ import annotations

import importlib

import numpy as np

from arac.benchmarks.aob import OptimizationProblem


class IohBbobBenchmark:
    """Load one IOH BBOB instance and expose only its numeric surface."""

    def load(
        self,
        function_id: int,
        *,
        instance: int,
        dimension: int,
    ) -> OptimizationProblem:
        for value, name in (
            (function_id, "function_id"),
            (instance, "instance"),
            (dimension, "dimension"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if function_id > 24:
            raise ValueError("IOH BBOB function_id must be in 1..24")

        try:
            ioh = importlib.import_module("ioh")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "IOH BBOB validation requires the external-validation dependency"
            ) from exc

        source = ioh.get_problem(
            function_id,
            instance=instance,
            dimension=dimension,
            problem_class=ioh.ProblemClass.BBOB,
        )
        optimum = float(source.optimum.y)

        def objective(values: np.ndarray) -> float | np.ndarray:
            candidates = np.asarray(values, dtype=float)
            single = candidates.ndim == 1
            raw = np.asarray(source(candidates), dtype=float).reshape(-1)
            errors = np.maximum(raw - optimum, 0.0)
            return float(errors[0]) if single else errors

        return OptimizationProblem(
            objective=objective,
            dimension=dimension,
            lower_bounds=tuple(float(value) for value in source.bounds.lb),
            upper_bounds=tuple(float(value) for value in source.bounds.ub),
            optimum=0.0,
        )


__all__ = ["IohBbobBenchmark"]

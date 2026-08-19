"""Thin, versioned ports to upstream optimization packages."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import math
import time

import numpy as np
from pypop7.optimizers.es.cmaes import CMAES
from pypop7.optimizers.es.mmes import MMES
from pypop7.optimizers.es.sepcmaes import SEPCMAES

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.ledger import EvaluationLedger


class _BatchedEvaluation:
    def _evaluate_population(self, candidates: np.ndarray, args: object = None) -> np.ndarray:
        rows = np.asarray(candidates, dtype=float)
        if args is not None:
            raise ValueError("batched optimizer ports do not accept objective args")
        if len(rows) > self.max_function_evaluations - self.n_function_evaluations:
            raise RuntimeError("optimizer population exceeds its remaining FE")
        started = time.time()
        values = np.asarray(self.fitness_function(rows), dtype=float).reshape(-1)
        self.time_function_evaluations += time.time() - started
        if values.shape != (len(rows),) or not np.all(np.isfinite(values)):
            raise ValueError("objective must return one finite value per optimizer candidate")
        for candidate, value in zip(rows, values, strict=True):
            numeric = float(value)
            self.n_function_evaluations += 1
            if numeric < self.best_so_far_y:
                self.best_so_far_x = candidate.copy()
                self.best_so_far_y = numeric
            if (self._base_early_stopping - numeric) <= self.early_stopping_threshold:
                self._counter_early_stopping += 1
            else:
                self._counter_early_stopping = 0
                self._base_early_stopping = numeric
        return values


class _BatchedCMAES(_BatchedEvaluation, CMAES):
    def iterate(self, x=None, mean=None, e_ve=None, e_va=None, y=None, d=None, args=None):
        active = min(
            self.n_individuals,
            int(self.max_function_evaluations - self.n_function_evaluations),
        )
        transform = e_ve @ np.diag(e_va)
        for index in range(active):
            noise = self.rng_optimization.standard_normal((self.ndim_problem,))
            d[index] = transform @ noise
            x[index] = mean + self.sigma * d[index]
        np.clip(x[:active], self.lower_boundary, self.upper_boundary, out=x[:active])
        d[:active] = (x[:active] - mean) / self.sigma
        y[:active] = self._evaluate_population(x[:active], args)
        if active < self.n_individuals:
            y[active:] = math.inf
        return x, y, d


class _BatchedSEPCMAES(_BatchedEvaluation, SEPCMAES):
    def iterate(self, z=None, x=None, mean=None, d=None, y=None, args=None):
        active = min(
            self.n_individuals,
            int(self.max_function_evaluations - self.n_function_evaluations),
        )
        z[:active] = self.rng_optimization.standard_normal((active, self.ndim_problem))
        x[:active] = mean + self.sigma * d * z[:active]
        np.clip(x[:active], self.lower_boundary, self.upper_boundary, out=x[:active])
        z[:active] = (x[:active] - mean) / (self.sigma * d)
        y[:active] = self._evaluate_population(x[:active], args)
        if active < self.n_individuals:
            y[active:] = math.inf
        return z, x, y


class _BatchedMMES(_BatchedEvaluation, MMES):
    def iterate(self, x=None, mean=None, q=None, v=None, y=None, args=None):
        for index in range(self._n_mirror_sampling):
            mixed_direction = np.zeros((self.ndim_problem,))
            for _ in range(self.ms):
                direction = v[
                    (self.m - self.rng_optimization.geometric(self.c_a) % self.m) - 1
                ]
                mixed_direction += self.rng_optimization.standard_normal() * q[direction]
            step = self._z_1 * self.rng_optimization.standard_normal((self.ndim_problem,))
            step += self._z_2 * mixed_direction
            x[index] = mean + self.sigma * step
            mirror_index = self._n_mirror_sampling + index
            if mirror_index < self.n_individuals:
                x[mirror_index] = mean - self.sigma * step
        active = min(
            self.n_individuals,
            int(self.max_function_evaluations - self.n_function_evaluations),
        )
        np.clip(x[:active], self.lower_boundary, self.upper_boundary, out=x[:active])
        y[:active] = self._evaluate_population(x[:active], args)
        if active < self.n_individuals:
            y[active:] = math.inf
        return x, y


_OPTIMIZERS = {
    "cmaes": _BatchedCMAES,
    "mmes": _BatchedMMES,
    "sepcmaes": _BatchedSEPCMAES,
}


@dataclass(frozen=True)
class OptimizationRun:
    algorithm: str
    consumed_fes: int
    best_x: tuple[float, ...]
    best_error: float
    package: str
    package_version: str


class PypopOptimizerPort:
    """Execute one upstream optimizer without leaking package-specific state."""

    package = "pypop7"

    def __init__(self) -> None:
        self.package_version = importlib.metadata.version(self.package)

    def run(
        self,
        algorithm: str,
        *,
        problem: OptimizationProblem,
        ledger: EvaluationLedger,
        initial_mean: tuple[float, ...] | np.ndarray,
        sigma: float,
        seed: int,
        budget_fes: int,
        population_size: int | None = None,
        restart: bool = False,
    ) -> OptimizationRun:
        if algorithm not in _OPTIMIZERS:
            raise ValueError(f"unsupported upstream optimizer: {algorithm}")
        if problem is not ledger.problem:
            raise ValueError("optimizer problem and ledger problem must be identical")
        mean = np.asarray(initial_mean, dtype=float)
        if mean.shape != (problem.dimension,) or not np.all(np.isfinite(mean)):
            raise ValueError("initial_mean is invalid")
        step_size = float(sigma)
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(budget_fes, bool) or not 0 < budget_fes <= ledger.remaining:
            raise ValueError("optimizer budget is outside the remaining FE")
        if algorithm == "cmaes" and problem.dimension > 256:
            raise ValueError("full CMA-ES is disabled above 256 dimensions")
        if population_size is not None and population_size < 2:
            raise ValueError("population_size must be at least two")

        count_before = ledger.count
        upstream_problem = {
            "fitness_function": ledger.evaluate,
            "ndim_problem": problem.dimension,
            "lower_boundary": problem.lower_array,
            "upper_boundary": problem.upper_array,
        }
        options: dict[str, object] = {
            "max_function_evaluations": int(budget_fes),
            "mean": np.clip(mean, problem.lower_array, problem.upper_array),
            "sigma": step_size,
            "seed_rng": int(seed),
            "is_restart": bool(restart),
            "verbose": 0,
        }
        if population_size is not None:
            options["n_individuals"] = int(population_size)
        optimizer = _OPTIMIZERS[algorithm](upstream_problem, options)
        optimizer.optimize()
        consumed = ledger.count - count_before
        if consumed != budget_fes:
            raise RuntimeError(
                f"{algorithm} consumed {consumed} FE instead of the required {budget_fes}"
            )
        return OptimizationRun(
            algorithm=algorithm,
            consumed_fes=consumed,
            best_x=tuple(float(value) for value in ledger.best_x),
            best_error=float(ledger.best_error),
            package=self.package,
            package_version=self.package_version,
        )


__all__ = ["OptimizationRun", "PypopOptimizerPort"]

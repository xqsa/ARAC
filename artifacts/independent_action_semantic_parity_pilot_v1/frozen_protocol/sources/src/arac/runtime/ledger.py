"""Exact objective-evaluation accounting and strict-best archive."""

from __future__ import annotations

import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import Phase2Snapshot


class BudgetExceededError(RuntimeError):
    """Raised before an objective call would exceed the declared FE budget."""


class EvaluationLedger:
    """Own every objective call made by one ARAC run."""

    def __init__(
        self,
        problem: OptimizationProblem,
        total_budget: int,
        *,
        initial_count: int = 0,
        initial_incumbent: tuple[float, ...] | None = None,
        initial_error: float | None = None,
    ) -> None:
        if not isinstance(problem, OptimizationProblem):
            raise TypeError("problem must be OptimizationProblem")
        if isinstance(total_budget, bool) or total_budget <= 0:
            raise ValueError("total_budget must be a positive integer")
        if isinstance(initial_count, bool) or not 0 <= initial_count <= total_budget:
            raise ValueError("initial_count is outside the FE budget")
        if (initial_incumbent is None) != (initial_error is None):
            raise ValueError("initial incumbent and error must be supplied together")
        self.problem = problem
        self.total_budget = int(total_budget)
        self._lower_bounds = problem.lower_array
        self._upper_bounds = problem.upper_array
        self._count = int(initial_count)
        self._best_x: np.ndarray | None = None
        self._best_error = math.inf
        if initial_incumbent is not None:
            incumbent = np.asarray(initial_incumbent, dtype=float)
            if incumbent.shape != (problem.dimension,) or not np.all(np.isfinite(incumbent)):
                raise ValueError("initial incumbent is invalid")
            error = float(initial_error)
            if not math.isfinite(error):
                raise ValueError("initial error must be finite")
            self._best_x = incumbent.copy()
            self._best_error = error

    @property
    def count(self) -> int:
        return self._count

    @property
    def remaining(self) -> int:
        return self.total_budget - self._count

    @property
    def best_error(self) -> float:
        if self._best_x is None:
            raise RuntimeError("the archive is empty")
        return self._best_error

    @property
    def best_x(self) -> np.ndarray:
        if self._best_x is None:
            raise RuntimeError("the archive is empty")
        return self._best_x.copy()

    def evaluate(self, candidate: np.ndarray) -> float | np.ndarray:
        values = np.asarray(candidate, dtype=float)
        single = values.ndim == 1
        batch = values[np.newaxis, :] if single else values
        if batch.ndim != 2 or batch.shape[1] != self.problem.dimension:
            raise ValueError("candidate shape does not match the problem dimension")
        if not np.all(np.isfinite(batch)):
            raise ValueError("candidate values must be finite")
        if np.any(batch < self._lower_bounds) or np.any(batch > self._upper_bounds):
            raise ValueError("candidate escaped the problem bounds")
        requested = int(batch.shape[0])
        if requested > self.remaining:
            raise BudgetExceededError(
                f"objective call requests {requested} FE with only {self.remaining} remaining"
            )
        raw = self.problem.objective(values if single else batch)
        results = np.asarray(raw, dtype=float).reshape(-1)
        if results.size != requested or not np.all(np.isfinite(results)):
            raise ValueError("objective must return one finite value per candidate")
        self._count += requested
        for vector, error in zip(batch, results, strict=True):
            numeric = float(error)
            if self._best_x is None or numeric < self._best_error:
                self._best_x = vector.copy()
                self._best_error = numeric
        return float(results[0]) if single else results

    @classmethod
    def from_checkpoint(
        cls,
        problem: OptimizationProblem,
        *,
        total_budget: int,
        phase1_fes: int,
        incumbent: tuple[float, ...],
        incumbent_error: float,
    ) -> EvaluationLedger:
        return cls(
            problem,
            total_budget,
            initial_count=phase1_fes,
            initial_incumbent=incumbent,
            initial_error=incumbent_error,
        )

    @classmethod
    def from_phase2_snapshot(
        cls,
        problem: OptimizationProblem,
        snapshot: Phase2Snapshot,
    ) -> EvaluationLedger:
        """Recreate the strict-best archive at a Phase-II snapshot boundary."""

        if not isinstance(snapshot, Phase2Snapshot):
            raise TypeError("snapshot must be Phase2Snapshot")
        return cls(
            problem,
            snapshot.total_fes,
            initial_count=snapshot.start_fes + snapshot.consumed_fes,
            initial_incumbent=snapshot.incumbent,
            initial_error=snapshot.best_error,
        )


__all__ = ["BudgetExceededError", "EvaluationLedger"]

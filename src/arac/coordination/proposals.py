"""Black-box local proposal production for overlap components."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

from arac.coordination.overlap import LocalProposal, OverlapStructure
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import ResumableOptimizerSession


@dataclass(frozen=True)
class LocalProposalRun:
    """Auditable output of one owner-local optimizer session."""

    proposal: LocalProposal
    best_x: tuple[float, ...]
    best_error: float
    consumed_fes: int
    global_start_fes: int
    global_end_fes: int
    algorithm: str


class _MirroredLedger(EvaluationLedger):
    """Keep a private owner archive while charging a shared global ledger."""

    def __init__(
        self,
        problem: OptimizationProblem,
        budget_fes: int,
        *,
        dimensions: tuple[int, ...],
        initial_incumbent: tuple[float, ...],
        initial_error: float,
        global_ledger: EvaluationLedger,
    ) -> None:
        super().__init__(
            problem,
            budget_fes,
            initial_incumbent=initial_incumbent,
            initial_error=initial_error,
        )
        self.global_ledger = global_ledger
        self.dimensions = np.asarray(dimensions, dtype=int)
        self.observations: list[tuple[np.ndarray, float]] = []

    def evaluate(self, candidate: np.ndarray) -> float | np.ndarray:
        values = np.asarray(candidate, dtype=float)
        single = values.ndim == 1
        batch = values[np.newaxis, :] if single else values
        if batch.ndim != 2 or batch.shape[1] != self.problem.dimension:
            raise ValueError("candidate shape does not match the problem dimension")
        requested = int(batch.shape[0])
        if requested > self.remaining:
            raise RuntimeError("local proposal exceeded its owner budget")
        if requested > self.global_ledger.remaining:
            raise RuntimeError("global Phase-II ledger cannot pay local proposal FE")
        raw = self.global_ledger.evaluate(values if single else batch)
        results = np.asarray(raw, dtype=float).reshape(-1)
        if results.shape != (requested,) or not np.all(np.isfinite(results)):
            raise ValueError("global objective returned invalid local proposal values")
        self._count += requested
        for vector, result in zip(batch, results, strict=True):
            numeric = float(result)
            self.observations.append((vector[self.dimensions].copy(), numeric))
            if self._best_x is None or numeric < self._best_error:
                self._best_x = vector.copy()
                self._best_error = numeric
        return float(results[0]) if single else results


def produce_local_proposal(
    structure: OverlapStructure,
    group: int,
    *,
    problem: OptimizationProblem,
    global_ledger: EvaluationLedger,
    anchor: tuple[float, ...] | np.ndarray,
    anchor_error: float,
    budget_fes: int,
    seed: int,
    algorithm: str = "sepcmaes",
    population_size: int = 8,
    sigma: float = 0.5,
    variables: Iterable[int] | None = None,
) -> LocalProposalRun:
    """Optimize one component group from a common anchor and emit a proposal."""

    if not isinstance(structure, OverlapStructure):
        raise TypeError("structure must be OverlapStructure")
    if problem is not global_ledger.problem or structure.dimension != problem.dimension:
        raise ValueError("proposal structure, problem and ledger dimensions must agree")
    if isinstance(group, bool) or not isinstance(group, int) or group not in range(len(structure.groups)):
        raise ValueError("group must identify a known structure group")
    if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes < 8:
        raise ValueError("budget_fes must be at least eight")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    anchor_vector = np.asarray(anchor, dtype=float)
    if anchor_vector.shape != (problem.dimension,) or not np.all(np.isfinite(anchor_vector)):
        raise ValueError("anchor must match the problem dimension and be finite")
    if np.any(anchor_vector < problem.lower_array) or np.any(anchor_vector > problem.upper_array):
        raise ValueError("anchor escaped the problem bounds")
    numeric_anchor_error = float(anchor_error)
    if not math.isfinite(numeric_anchor_error):
        raise ValueError("anchor_error must be finite")
    if budget_fes > global_ledger.remaining:
        raise ValueError("local proposal budget exceeds the global ledger remainder")

    group_variables = tuple(structure.groups[group])
    if variables is None:
        dimensions = group_variables
    else:
        selected = tuple(variables)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("variables must be a non-empty set of unique group variables")
        if any(variable not in group_variables for variable in selected):
            raise ValueError("variables must be contained in the selected group")
        dimensions = tuple(sorted(selected))
    local_ledger = _MirroredLedger(
        problem,
        budget_fes,
        dimensions=dimensions,
        initial_incumbent=tuple(float(value) for value in anchor_vector),
        initial_error=numeric_anchor_error,
        global_ledger=global_ledger,
    )
    start_fes = global_ledger.count
    session = ResumableOptimizerSession(
        algorithm,
        problem=problem,
        ledger=local_ledger,
        initial_mean=anchor_vector[np.asarray(dimensions, dtype=int)],
        sigma=sigma,
        seed=seed,
        budget_fes=budget_fes,
        population_size=population_size,
        dimensions=dimensions,
        anchor=anchor_vector,
    )
    session.step(budget_fes)
    if local_ledger.count != budget_fes or global_ledger.count - start_fes != budget_fes:
        raise RuntimeError("local proposal optimizer drifted from exact FE accounting")

    observations = sorted(local_ledger.observations, key=lambda item: item[1])
    elite_count = max(2, min(len(observations), max(4, budget_fes // 8)))
    elite = np.asarray([row for row, _ in observations[:elite_count]], dtype=float)
    best_x = local_ledger.best_x
    uncertainty = tuple(
        (
            variable,
            float(max(np.std(elite[:, local_index], ddof=1), np.finfo(float).eps)),
        )
        for local_index, variable in enumerate(dimensions)
    )
    proposal = LocalProposal(
        group=group,
        values=tuple((variable, float(best_x[variable])) for variable in dimensions),
        improvement=max(0.0, numeric_anchor_error - float(local_ledger.best_error)),
        uncertainty=uncertainty,
    )
    return LocalProposalRun(
        proposal=proposal,
        best_x=tuple(float(value) for value in best_x),
        best_error=float(local_ledger.best_error),
        consumed_fes=budget_fes,
        global_start_fes=start_fes,
        global_end_fes=global_ledger.count,
        algorithm=algorithm,
    )


__all__ = ["LocalProposalRun", "produce_local_proposal"]

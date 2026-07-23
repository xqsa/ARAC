"""Deterministic optimization adapters for continuous WLOC baselines."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Callable, Sequence

import numpy as np
from pypop7.optimizers.es.cmaes import CMAES

from .contracts import BaselineResult, GroupingResult


Objective = Callable[[np.ndarray], object]


def cmaes_population_size(dimension: int) -> int:
    """Return the population rule used by the HCC paper implementation."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return 4 + 3 * math.ceil(math.log(dimension))


def derive_optimizer_seed(base_seed: int, *namespace: object) -> int:
    payload = ":".join((str(base_seed), *(str(value) for value in namespace))).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


def _vector(
    values: float | Sequence[float],
    dimension: int,
    name: str,
) -> np.ndarray:
    normalized = np.broadcast_to(np.asarray(values, dtype=float), (dimension,)).copy()
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must contain only finite values")
    return normalized


class EvaluationLedger:
    """Count every objective call and retain the repaired best phenotype."""

    def __init__(
        self,
        objective: Objective,
        dimension: int,
        lower: np.ndarray,
        upper: np.ndarray,
        max_fes: int,
    ) -> None:
        self.objective = objective
        self.dimension = dimension
        self.lower = lower
        self.upper = upper
        self.max_fes = max_fes
        self.best_x: np.ndarray | None = None
        self.best_y = math.inf
        self.trace: list[float] = []
        self.phase_fes: OrderedDict[str, int] = OrderedDict()
        self.repaired_candidate_count = 0

    @property
    def evaluations(self) -> int:
        return len(self.trace)

    def evaluate(self, candidate: object, phase: str) -> float:
        if self.evaluations >= self.max_fes:
            raise RuntimeError("optimization FE budget exhausted")
        raw = np.asarray(candidate, dtype=float)
        if raw.shape != (self.dimension,) or not np.all(np.isfinite(raw)):
            raise ValueError(f"optimizer candidate must have shape ({self.dimension},) and be finite")
        repaired = np.clip(raw, self.lower, self.upper)
        if not np.array_equal(raw, repaired):
            self.repaired_candidate_count += 1
        output = np.asarray(self.objective(repaired), dtype=float)
        if output.size != 1:
            raise ValueError("objective must return exactly one value per candidate")
        value = float(output.reshape(-1)[0])
        if not math.isfinite(value):
            raise ValueError("objective returned a non-finite value")
        if value < self.best_y:
            self.best_y = value
            self.best_x = repaired.copy()
        self.trace.append(self.best_y)
        self.phase_fes[phase] = self.phase_fes.get(phase, 0) + 1
        return value


def _run_cmaes_block(
    ledger: EvaluationLedger,
    context: np.ndarray,
    group: tuple[int, ...],
    *,
    budget: int,
    seed: int,
    sigma: float,
    phase: str,
) -> tuple[np.ndarray, float]:
    indices = np.asarray(group, dtype=int)
    snapshot = context.copy()
    block_best_y = math.inf
    block_best_values: np.ndarray | None = None

    def objective(raw_values: np.ndarray) -> float:
        nonlocal block_best_y, block_best_values
        raw = np.asarray(raw_values, dtype=float)
        values = np.clip(raw, ledger.lower[indices], ledger.upper[indices])
        candidate = snapshot.copy()
        candidate[indices] = raw
        fitness = ledger.evaluate(candidate, phase)
        if fitness < block_best_y:
            block_best_y = fitness
            block_best_values = values.copy()
        return fitness

    problem = {
        "fitness_function": objective,
        "ndim_problem": len(group),
        "lower_boundary": ledger.lower[indices],
        "upper_boundary": ledger.upper[indices],
    }
    options = {
        "max_function_evaluations": budget,
        "mean": snapshot[indices],
        "sigma": sigma,
        "n_individuals": cmaes_population_size(len(group)),
        "seed_rng": seed,
        "is_restart": False,
        "verbose": 0,
    }
    before = ledger.evaluations
    result = CMAES(problem, options).optimize()
    consumed = ledger.evaluations - before
    if consumed != budget or int(result["n_function_evaluations"]) != budget:
        raise RuntimeError("CMA-ES did not consume its exact assigned FE budget")
    if block_best_values is None:
        raise RuntimeError("CMA-ES block produced no evaluated candidate")
    candidate = snapshot.copy()
    candidate[indices] = block_best_values
    return candidate, block_best_y


def run_cooperative_cmaes(
    objective: Objective,
    grouping: GroupingResult,
    *,
    max_function_evaluations: int,
    seed: int,
    initial_mean: float | Sequence[float] = 0.5,
    sigma: float = 0.5,
    lower: float | Sequence[float] = 0.0,
    upper: float | Sequence[float] = 1.0,
    method_name: str | None = None,
    group_block_fes: int | None = None,
) -> BaselineResult:
    """Optimize supplied groups in deterministic round-robin order with CMA-ES."""

    dimension = grouping.dimension
    if max_function_evaluations <= 0:
        raise ValueError("max_function_evaluations must be positive")
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    if group_block_fes is not None and group_block_fes <= 0:
        raise ValueError("group_block_fes must be positive when supplied")
    lower_values = _vector(lower, dimension, "lower")
    upper_values = _vector(upper, dimension, "upper")
    if np.any(lower_values >= upper_values):
        raise ValueError("every lower bound must be smaller than its upper bound")
    mean = _vector(initial_mean, dimension, "initial_mean")
    if np.any(mean < lower_values) or np.any(mean > upper_values):
        raise ValueError("initial_mean must lie within the bounds")

    ledger = EvaluationLedger(
        objective,
        dimension,
        lower_values,
        upper_values,
        max_function_evaluations,
    )
    context = mean.copy()
    context_y = ledger.evaluate(context, "initial_context")
    stage = 0
    while ledger.evaluations < max_function_evaluations:
        for position, group in enumerate(grouping.groups):
            remaining = max_function_evaluations - ledger.evaluations
            if remaining == 0:
                break
            if group_block_fes is None:
                groups_left = len(grouping.groups) - position
                budget = math.ceil(remaining / groups_left)
            else:
                budget = min(group_block_fes, remaining)
            stage_seed = derive_optimizer_seed(seed, method_name or grouping.method, stage)
            phase = f"group_{position}"
            candidate, candidate_y = _run_cmaes_block(
                ledger,
                context,
                group,
                budget=budget,
                seed=stage_seed,
                sigma=sigma,
                phase=phase,
            )
            if candidate_y < context_y:
                context = candidate
                context_y = candidate_y
            stage += 1

    if ledger.best_x is None:
        raise RuntimeError("optimizer completed without a best candidate")
    return BaselineResult(
        method=method_name or f"{grouping.method}-CMAES",
        backend="PyPop7.CMAES@0.0.82",
        dimension=dimension,
        optimizer_seed=int(seed),
        optimization_fes=ledger.evaluations,
        decomposition_fes=grouping.decomposition_fes,
        best_x=tuple(float(value) for value in ledger.best_x),
        best_y=float(ledger.best_y),
        best_so_far_trace=tuple(float(value) for value in ledger.trace),
        initial_mean=tuple(float(value) for value in mean),
        sigma=float(sigma),
        repair_policy="clip_to_bounds",
        repaired_candidate_count=ledger.repaired_candidate_count,
        phase_fes=tuple(ledger.phase_fes.items()),
        grouping_hash=grouping.grouping_hash,
    )

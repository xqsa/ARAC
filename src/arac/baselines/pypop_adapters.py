"""Thin PyPop7 adapters for the four non-decomposition AOB baselines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from pypop7.optimizers.es.lmcma import LMCMA
from pypop7.optimizers.es.lmmaes import LMMAES
from pypop7.optimizers.es.mmes import MMES
from pypop7.optimizers.es.sepcmaes import SEPCMAES

from .contracts import BaselineResult
from .optimization import EvaluationLedger, Objective, _vector


PYPOP7_VERSION = "0.0.82"
PYPOP_METHODS = {
    "Sep-CMAES": SEPCMAES,
    "LM-MA-ES": LMMAES,
    "LMCMA": LMCMA,
    "MM-ES": MMES,
}


def run_pypop_block(
    optimizer_class: type,
    ledger: EvaluationLedger,
    mean: np.ndarray,
    *,
    budget: int,
    seed: int,
    sigma: float,
    phase: str,
    restart: bool = False,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Run one full-space PyPop7 block against an existing FE ledger."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    block_best_y = math.inf
    block_best_x: np.ndarray | None = None

    def objective(raw_candidate: np.ndarray) -> float:
        nonlocal block_best_y, block_best_x
        raw = np.asarray(raw_candidate, dtype=float)
        fitness = ledger.evaluate(raw, phase)
        if fitness < block_best_y:
            block_best_y = fitness
            block_best_x = np.clip(raw, ledger.lower, ledger.upper)
        return fitness

    problem = {
        "fitness_function": objective,
        "ndim_problem": ledger.dimension,
        "lower_boundary": ledger.lower,
        "upper_boundary": ledger.upper,
    }
    options = {
        "max_function_evaluations": budget,
        "mean": mean.copy(),
        "sigma": sigma,
        "seed_rng": seed,
        "is_restart": restart,
        "verbose": 0,
    }
    before = ledger.evaluations
    result = optimizer_class(problem, options).optimize()
    consumed = ledger.evaluations - before
    if consumed != budget or int(result["n_function_evaluations"]) != budget:
        raise RuntimeError("PyPop7 optimizer did not consume its exact assigned FE budget")
    if block_best_x is None:
        raise RuntimeError("PyPop7 optimizer block produced no evaluated candidate")
    return block_best_x, block_best_y, result


def run_pypop_baseline(
    objective: Objective,
    method: str,
    dimension: int,
    *,
    max_function_evaluations: int,
    seed: int,
    initial_mean: float | Sequence[float] = 0.5,
    sigma: float = 0.5,
    lower: float | Sequence[float] = 0.0,
    upper: float | Sequence[float] = 1.0,
) -> BaselineResult:
    """Run one of Sep-CMAES, LM-MA-ES, LMCMA, or MM-ES."""

    try:
        optimizer_class = PYPOP_METHODS[method]
    except KeyError as error:
        raise ValueError(f"unsupported PyPop7 baseline method: {method}") from error
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if max_function_evaluations <= 0:
        raise ValueError("max_function_evaluations must be positive")
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
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
    ledger.evaluate(mean, "initial_context")
    remaining = max_function_evaluations - ledger.evaluations
    if remaining:
        run_pypop_block(
            optimizer_class,
            ledger,
            mean,
            budget=remaining,
            seed=int(seed),
            sigma=sigma,
            phase="optimizer",
        )
    if ledger.best_x is None:
        raise RuntimeError("optimizer completed without a best candidate")
    return BaselineResult(
        method=method,
        backend=f"PyPop7.{optimizer_class.__name__}@{PYPOP7_VERSION}",
        dimension=dimension,
        optimizer_seed=int(seed),
        optimization_fes=ledger.evaluations,
        decomposition_fes=0,
        best_x=tuple(float(value) for value in ledger.best_x),
        best_y=float(ledger.best_y),
        best_so_far_trace=tuple(float(value) for value in ledger.trace),
        initial_mean=tuple(float(value) for value in mean),
        sigma=float(sigma),
        repair_policy="clip_to_bounds",
        repaired_candidate_count=ledger.repaired_candidate_count,
        phase_fes=tuple(ledger.phase_fes.items()),
    )


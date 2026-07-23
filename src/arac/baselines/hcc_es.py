"""Generic continuous-domain adapter for the paper's Hybrid HCC-ES."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from pypop7.optimizers.es.mmes import MMES

from .contracts import BaselineResult, GroupingResult
from .optimization import (
    EvaluationLedger,
    Objective,
    _run_cmaes_block,
    _vector,
    derive_optimizer_seed,
)
from .pypop_adapters import PYPOP7_VERSION, run_pypop_block


def overlap_degree(grouping: GroupingResult) -> float:
    counts = Counter(index for group in grouping.groups for index in group)
    shared = sum(count > 1 for count in counts.values())
    return shared / grouping.dimension


def hcc_global_phase_fes(
    max_function_evaluations: int,
    grouping: GroupingResult,
) -> int:
    """Return the HCC global-stage allocation, including its initial context FE."""

    if max_function_evaluations <= 0:
        raise ValueError("max_function_evaluations must be positive")
    coefficient = 0.2 + 0.8 * overlap_degree(grouping)
    return max(1, min(max_function_evaluations, int(coefficient * max_function_evaluations)))


def _blend(
    previous_value: float,
    current_value: float,
    previous_delta: float,
    current_delta: float,
) -> float:
    denominator = previous_delta + current_delta
    if denominator == 0.0:
        return (previous_value + current_value) / 2.0
    return (
        previous_delta * previous_value + current_delta * current_value
    ) / denominator


def run_hcc_es(
    objective: Objective,
    grouping: GroupingResult,
    *,
    max_function_evaluations: int,
    seed: int,
    initial_mean: float | Sequence[float] = 0.5,
    sigma: float = 0.5,
    lower: float | Sequence[float] = 0.0,
    upper: float | Sequence[float] = 1.0,
    group_block_fes: int | None = None,
) -> BaselineResult:
    """Run global MM-ES followed by topology-driven cooperative CMA-ES."""

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
    global_target = hcc_global_phase_fes(max_function_evaluations, grouping)
    global_budget = global_target - ledger.evaluations
    if global_budget > 0:
        global_candidate, global_y, _ = run_pypop_block(
            MMES,
            ledger,
            context,
            budget=global_budget,
            seed=derive_optimizer_seed(seed, "HCC-ES", "global"),
            sigma=sigma,
            phase="global_mmes",
            restart=True,
        )
        if global_y < context_y:
            context = global_candidate
            context_y = global_y

    membership_counts = Counter(index for group in grouping.groups for index in group)
    shared_variables = {index for index, count in membership_counts.items() if count > 1}
    stage = 0
    while ledger.evaluations < max_function_evaluations:
        proposals: dict[int, float] = {}
        for position, group in enumerate(grouping.groups):
            remaining = max_function_evaluations - ledger.evaluations
            if remaining == 0:
                break
            repeated = shared_variables.intersection(group).intersection(proposals)
            reconciliation_reserve = int(bool(repeated) and remaining > 1)
            available = remaining - reconciliation_reserve
            if group_block_fes is None:
                groups_left = len(grouping.groups) - position
                budget = math.ceil(available / groups_left)
            else:
                budget = min(group_block_fes, available)
            if budget <= 0:
                break
            snapshot = context.copy()
            snapshot_y = context_y
            candidate, candidate_y = _run_cmaes_block(
                ledger,
                context,
                group,
                budget=budget,
                seed=derive_optimizer_seed(seed, "HCC-ES", "group", stage),
                sigma=sigma,
                phase=f"group_{position}",
                restart=True,
            )
            current_delta = max(0.0, snapshot_y - candidate_y)
            proposed = candidate if candidate_y < snapshot_y else snapshot
            proposed_y = candidate_y if candidate_y < snapshot_y else snapshot_y
            if repeated:
                blended = proposed.copy()
                for variable in repeated:
                    blended[variable] = _blend(
                        snapshot[variable],
                        proposed[variable],
                        proposals[variable],
                        current_delta,
                    )
                context = blended
                if ledger.evaluations < max_function_evaluations:
                    context_y = ledger.evaluate(context, "overlap_reconciliation")
                else:
                    context_y = proposed_y
            else:
                context = proposed
                context_y = proposed_y
            for variable in shared_variables.intersection(group):
                proposals[variable] = proposals.get(variable, 0.0) + current_delta
            stage += 1

    if ledger.best_x is None:
        raise RuntimeError("HCC-ES completed without a best candidate")
    return BaselineResult(
        method="HCC-ES",
        backend=f"HCC-ES[PyPop7.MMES+CMAES]@{PYPOP7_VERSION}",
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


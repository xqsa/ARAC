"""Counted two-sided conflict probes for shared variables.

This module implements the design contract's authoritative conflict-grading
primitive (docs/arac-oc-design.md §5): for each shared variable in a scope,
evaluate ``x_plus`` and ``x_minus`` around the strict-best incumbent in the
full global context and derive directional bias ``B_j``, local response width
``W_j`` and a normalized conflict score ``C_j`` from real function values.

Status: v1 formulas, unit-tested but NOT calibrated and NOT wired into any
dispatch path.  Wiring requires an offline calibration gate per the frozen
discipline (see the topology-signal precedent in Gate 37/38).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from arac.coordination.overlap import OverlapStructure
from arac.runtime.ledger import EvaluationLedger


@dataclass(frozen=True)
class CountedProbeResult:
    """Per-variable counted-probe evidence for one shared variable."""

    variable: int
    step: float
    f_plus: float
    f_minus: float
    bias: float
    width: float
    conflict_score: float


def _step_for(
    structure: OverlapStructure,
    proposals,
    variable: int,
    *,
    step_floor: float,
    bounds_span: np.ndarray,
) -> float:
    """Probe scale from owner proposal disagreement, floored for degenerate cases."""

    owners = structure.owners(variable)
    by_group = {proposal.group: proposal for proposal in proposals}
    values = [by_group[group].value(variable) for group in owners if group in by_group]
    sigmas = [by_group[group].sigma(variable) for group in owners if group in by_group]
    disagreement = max(values) - min(values) if len(values) >= 2 else 0.0
    spread = max([disagreement, *sigmas]) if (values or sigmas) else 0.0
    scale = float(np.median(bounds_span))
    return max(step_floor, min(spread, 0.25 * scale))


def counted_probe(
    structure: OverlapStructure,
    ledger: EvaluationLedger,
    scope,
    *,
    proposals=(),
    step_floor: float = 1e-6,
) -> tuple[CountedProbeResult, ...]:
    """Probe every shared variable in ``scope`` from the strict-best incumbent.

    Consumes exactly ``2 * len(scope)`` FE billed to the global ledger;
    ``f(x0)`` reuses the incumbent's existing evaluation.  ``proposals`` only
    set the probe scale (owner disagreement / uncertainty); they never enter
    the formulas, so stale SMP state cannot lower a conflict level.
    """

    if not isinstance(structure, OverlapStructure):
        raise TypeError("structure must be OverlapStructure")
    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be EvaluationLedger")
    if isinstance(step_floor, bool) or not isinstance(step_floor, float) or not step_floor > 0.0:
        raise ValueError("step_floor must be a positive float")
    scope = tuple(scope)
    if not scope or len(set(scope)) != len(scope):
        raise ValueError("scope must be a non-empty set of unique variables")
    shared = set(structure.shared_variables)
    if any(variable not in shared for variable in scope):
        raise ValueError("scope must contain only shared variables")
    if 2 * len(scope) > ledger.remaining:
        raise ValueError("counted probe exceeds the remaining FE budget")

    incumbent = ledger.best_x
    f0 = float(ledger.best_error)
    lower = ledger.problem.lower_array
    upper = ledger.problem.upper_array
    span = upper - lower
    start = ledger.count
    batch = np.repeat(incumbent[np.newaxis, :], 2 * len(scope), axis=0)
    steps: list[float] = []
    for index, variable in enumerate(scope):
        step = _step_for(structure, proposals, variable, step_floor=step_floor, bounds_span=span)
        steps.append(step)
        batch[2 * index, variable] = incumbent[variable] + step
        batch[2 * index + 1, variable] = incumbent[variable] - step
    np.clip(batch, lower, upper, out=batch)
    errors = np.asarray(ledger.evaluate(batch), dtype=float)
    if ledger.count - start != 2 * len(scope):
        raise RuntimeError("counted probe FE accounting drifted")

    results = []
    for index, variable in enumerate(scope):
        f_plus = float(errors[2 * index])
        f_minus = float(errors[2 * index + 1])
        plus_delta = f_plus - f0
        minus_delta = f_minus - f0
        denominator = abs(plus_delta) + abs(minus_delta)
        bias = (minus_delta - plus_delta) / (denominator + 1e-12) if denominator > 0 else 0.0
        bias = float(np.clip(bias, -1.0, 1.0))
        width = max(abs(plus_delta), abs(minus_delta))
        scale = abs(f0) + width + 1e-12
        conflict = abs(bias) * min(1.0, width / scale)
        results.append(
            CountedProbeResult(
                variable=variable,
                step=steps[index],
                f_plus=f_plus,
                f_minus=f_minus,
                bias=bias,
                width=float(width),
                conflict_score=float(conflict),
            )
        )
    if not all(math.isfinite(item.conflict_score) for item in results):
        raise RuntimeError("counted probe produced a non-finite conflict score")
    return tuple(results)


__all__ = ["CountedProbeResult", "counted_probe"]

"""L1 runtime consistency classifier for shared variables (Gate 54a).

Instrument v3 (frozen after two declared pilot rejections, 2026-08-22):
owner-preference calibration.  For every owner group it runs a bounded,
seeded block-local search from the strict-best incumbent (only that
group's coordinates move; every evaluation bills the global ledger), then
reads each owning group's preferred value for its shared variables.

Why owner conditioning is necessary (the pilot-proven negative): on
additive objectives a conforming shared variable (single shared well) and
a conflicting one (weighted sum of disagreeing owner wells) induce the
SAME aggregated 1-D response up to an unobservable constant — with a
sphere base they are both exactly parabolic.  Cross-context co-movement
is zero by construction on separable instances (pilot 1), and scale
instability of the bias is drowned by incumbent-position effects
(pilot 2).  Owner preference disagreement is the only observable channel,
and it is exactly the bounded calibration window that L2b reuses for
contribution attribution.

FE accounting: ``rounds * population`` per owner group (default 3 rounds
x 4 candidates = 12 FE/group), independent of the shared-variable count;
the strict-best archive only ever improves, and owner endpoints are read
from each search's own best point (they may be globally worse than the
incumbent — that is the signal, not an archive violation).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from arac.coordination.overlap import OverlapStructure
from arac.runtime.ledger import EvaluationLedger


@dataclass(frozen=True)
class OwnerPreference:
    """One owner group's calibrated preferred value for one shared variable."""

    group: int
    variable: int
    value: float


@dataclass(frozen=True)
class ConsistencyLabel:
    """Per shared-variable consistency evidence and label."""

    variable: int
    owners: tuple[int, ...]
    owner_values: tuple[float, ...]
    disagreement: float
    label: str
    confidence: float
    group_fes: int


def owner_preference_calibration(
    structure: OverlapStructure,
    ledger: EvaluationLedger,
    scope,
    *,
    rounds: int = 3,
    population: int = 4,
    step_fraction: float = 0.10,
    seed: int = 0,
) -> tuple[ConsistencyLabel, ...]:
    """Calibrate owner preferences and classify every shared variable.

    Per owner group: ``rounds * population`` seeded gaussian steps on the
    group's coordinates from the incumbent, greedy accept within the
    search (the group endpoint is its own best seen point).  A shared
    variable is conflicting iff the owning groups' preferred values
    disagree by >= 0.05 of the bounds span (freeze candidate), conforming
    otherwise; confidence is the margin to the threshold.
    """

    if not isinstance(structure, OverlapStructure):
        raise TypeError("structure must be OverlapStructure")
    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be EvaluationLedger")
    if isinstance(rounds, bool) or rounds < 1 or isinstance(population, bool) or population < 1:
        raise ValueError("rounds and population must be positive integers")
    scope = tuple(sorted(set(int(variable) for variable in scope)))
    if not scope:
        raise ValueError("scope must be non-empty")
    shared = set(structure.shared_variables)
    if any(variable not in shared for variable in scope):
        raise ValueError("scope must contain only shared variables")
    groups = tuple(tuple(int(v) for v in group) for group in structure.groups)
    owners_by_variable = {
        variable: tuple(
            index for index, group in enumerate(groups) if variable in group
        )
        for variable in scope
    }
    owner_groups = sorted({index for owners in owners_by_variable.values() for index in owners})
    per_group_fes = rounds * population
    needed = per_group_fes * len(owner_groups)
    if needed > ledger.remaining:
        raise ValueError("owner calibration exceeds the remaining FE budget")

    incumbent = np.asarray(ledger.best_x, dtype=float)
    f0 = float(ledger.best_error)
    lower = ledger.problem.lower_array
    upper = ledger.problem.upper_array
    span = float(np.median(upper - lower))
    rng = np.random.default_rng(seed)
    start = ledger.count

    endpoints: dict[int, tuple[np.ndarray, float]] = {}
    for index in owner_groups:
        coordinates = np.asarray(groups[index], dtype=int)
        best = incumbent.copy()
        best_error = f0
        sigma = step_fraction * span
        for _ in range(rounds):
            batch = np.repeat(best[np.newaxis, :], population, axis=0)
            noise = rng.normal(0.0, sigma, size=(population, coordinates.size))
            batch[:, coordinates] += noise
            np.clip(batch, lower, upper, out=batch)
            errors = np.asarray(ledger.evaluate(batch), dtype=float)
            choice = int(np.argmin(errors))
            if float(errors[choice]) < best_error:
                best = batch[choice]
                best_error = float(errors[choice])
        endpoints[index] = (best, best_error)

    if ledger.count - start != needed:
        raise RuntimeError("owner calibration FE accounting drifted")

    disagreement_threshold = 0.05
    results: list[ConsistencyLabel] = []
    for variable in scope:
        owners = owners_by_variable[variable]
        values = tuple(float(endpoints[owner][0][variable]) for owner in owners)
        disagreement = (max(values) - min(values)) / span if len(values) > 1 else 0.0
        if disagreement >= disagreement_threshold:
            label = "conflicting"
            confidence = min(1.0, disagreement / (2.0 * disagreement_threshold))
        else:
            label = "conforming"
            confidence = max(
                0.0, 1.0 - disagreement / disagreement_threshold
            ) * 0.5
        results.append(
            ConsistencyLabel(
                variable=variable,
                owners=owners,
                owner_values=values,
                disagreement=float(disagreement),
                label=label,
                confidence=float(confidence),
                group_fes=per_group_fes,
            )
        )
    if not all(math.isfinite(item.disagreement) for item in results):
        raise RuntimeError("owner calibration produced a non-finite disagreement")
    return tuple(results)


# Backwards-compatible alias for the gate scripts.
consistency_probe = owner_preference_calibration

__all__ = [
    "ConsistencyLabel",
    "OwnerPreference",
    "owner_preference_calibration",
    "consistency_probe",
]

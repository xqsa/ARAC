"""Structure-certified shared-coordinate re-centering kernel for the
shared_transaction_v2 candidate.

Root-cause motivation (shared_transaction_v1 T3 screen, 2026-08-24): under
the frozen disjoint-partition action set a shared coordinate j is only ever
moved by its PRIMARY block, so every owner proposal snapshot carries the
same value at j, P_j = [v, v], and an arbitration-style transaction
(median/mean) is structurally a no-op - zero acceptances across all ten
screen pairs, each candidate evaluating to exactly the incumbent error.

The v2 kernel repairs the actual damage mechanism instead: when the
non-primary owner's block has moved (its writebacks exist but j was frozen
at the primary's stale value), the full-objective slice along j has shifted
its optimum.  At the T1-qualified boundary the kernel re-centers up to
``MAX_COORDINATES_PER_BOUNDARY`` certified coordinates with a three-point
quadratic fit:

```text
for each selected coordinate j (rule: (-|M(j)|, j), certified links with
both owners fresh, at most 4):
    f0 = ledger best error                        (0 FE, never re-evaluated)
    f_plus  = F(x with j -> j + delta_j)          (1 FE)
    f_minus = F(x with j -> j - delta_j)          (1 FE)
    vertex  = j + delta_j * (f_minus - f_plus) / (2 * (f_plus + f_minus - 2 f0))
    if the curvature is positive and the vertex is interior:
        candidate = x with j -> clip(vertex)      (1 FE, strict-best accepts)
```

- 3 FE per coordinate, at most 12 FE per boundary, hard cap recorded;
- trusts only full-objective feedback - no owner authority, no classifier,
  no direction belief, no persistent state;
- probes use the moderate-range convention delta_j = 0.25 * span_j
  (matching the frozen discovery instrument's perturbation scale);
- acceptance is exactly the ledger strict-best rule;
- fail-closed: non-finite curvature or non-interior vertex skips the third
  evaluation and returns the unspent FE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from experiments.upgrade.shared_transaction_v1.transaction_kernel import TransactionMount

MAX_COORDINATES_PER_BOUNDARY = 4
MAX_FE_PER_BOUNDARY = 3 * MAX_COORDINATES_PER_BOUNDARY
PROBE_FRACTION = 0.25


@dataclass(frozen=True)
class CertifiedLink:
    variable: int
    owner_blocks: tuple[int, int]


@dataclass(frozen=True)
class RecenterCandidateReceipt:
    coordinate: int
    f0: float
    f_plus: float
    f_minus: float
    delta: float
    vertex: float | None
    clipped_vertex: float | None
    evaluated_fes: int
    vertex_error: float | None
    accepted: bool


@dataclass(frozen=True)
class RecenterReceipt:
    boundary_phase: str
    selected_coordinates: tuple[int, ...]
    eligible_coordinates: tuple[int, ...]
    consumed_fes: int
    reserved_fes: int
    returned_fes: int
    best_error_before: float
    best_error_after: float
    accepted_count: int
    candidates: tuple[RecenterCandidateReceipt, ...]
    silent_reason: str | None

    def payload(self) -> dict[str, Any]:
        return {
            "boundary_phase": self.boundary_phase,
            "selected_coordinates": list(self.selected_coordinates),
            "eligible_coordinates": list(self.eligible_coordinates),
            "consumed_fes": self.consumed_fes,
            "reserved_fes": self.reserved_fes,
            "returned_fes": self.returned_fes,
            "best_error_before": self.best_error_before,
            "best_error_after": self.best_error_after,
            "accepted_count": self.accepted_count,
            "candidates": [candidate.__dict__ for candidate in self.candidates],
            "silent_reason": self.silent_reason,
        }


def run_recentering_transaction(
    context,
    links: list[CertifiedLink],
    proposals: Mapping[int, Any],
    *,
    boundary_phase: str,
) -> RecenterReceipt:
    """Re-center certified shared coordinates at a qualified boundary."""

    ledger = context.ledger
    problem = context.problem
    span = problem.upper_array - problem.lower_array
    eligible = [
        link.variable
        for link in links
        if all(owner in proposals for owner in link.owner_blocks)
    ]
    eligible.sort()
    selected = eligible[:MAX_COORDINATES_PER_BOUNDARY]
    best_before = float(ledger.best_error)
    if not selected:
        return RecenterReceipt(
            boundary_phase=boundary_phase,
            selected_coordinates=(),
            eligible_coordinates=tuple(eligible),
            consumed_fes=0,
            reserved_fes=MAX_FE_PER_BOUNDARY,
            returned_fes=MAX_FE_PER_BOUNDARY,
            best_error_before=best_before,
            best_error_after=best_before,
            accepted_count=0,
            candidates=(),
            silent_reason="no_link_with_both_owners_fresh",
        )

    consumed = 0
    accepted = 0
    receipts: list[RecenterCandidateReceipt] = []
    incumbent = ledger.best_x.copy()
    for variable in selected:
        if consumed + 3 > MAX_FE_PER_BOUNDARY:
            break
        delta = PROBE_FRACTION * float(span[variable])
        f0 = float(ledger.best_error)

        plus = incumbent.copy()
        plus[variable] = float(np.clip(incumbent[variable] + delta, problem.lower_array[variable], problem.upper_array[variable]))
        error_before = float(ledger.best_error)
        f_plus = float(np.asarray(ledger.evaluate(plus[np.newaxis, :])).reshape(-1)[0])
        consumed += 1
        if float(ledger.best_error) < error_before:
            # The probe itself was a strict improvement: keep it as the new
            # incumbent and continue re-centering from there.
            accepted += 1
            incumbent = ledger.best_x.copy()
            receipts.append(
                RecenterCandidateReceipt(
                    coordinate=variable, f0=f0, f_plus=f_plus, f_minus=float("nan"),
                    delta=delta, vertex=None, clipped_vertex=float(plus[variable]),
                    evaluated_fes=1, vertex_error=f_plus, accepted=True,
                )
            )
            continue

        minus = incumbent.copy()
        minus[variable] = float(np.clip(incumbent[variable] - delta, problem.lower_array[variable], problem.upper_array[variable]))
        error_before = float(ledger.best_error)
        f_minus = float(np.asarray(ledger.evaluate(minus[np.newaxis, :])).reshape(-1)[0])
        consumed += 1
        if float(ledger.best_error) < error_before:
            accepted += 1
            incumbent = ledger.best_x.copy()
            receipts.append(
                RecenterCandidateReceipt(
                    coordinate=variable, f0=f0, f_plus=f_plus, f_minus=f_minus,
                    delta=delta, vertex=None, clipped_vertex=float(minus[variable]),
                    evaluated_fes=2, vertex_error=f_minus, accepted=True,
                )
            )
            continue

        curvature = f_plus + f_minus - 2.0 * f0
        vertex: float | None = None
        clipped: float | None = None
        vertex_error: float | None = None
        candidate_accepted = False
        if np.isfinite(curvature) and curvature > 0.0:
            shift = delta * (f_minus - f_plus) / (2.0 * curvature)
            vertex = float(incumbent[variable] + shift)
            clipped = float(np.clip(vertex, problem.lower_array[variable], problem.upper_array[variable]))
            lower = min(float(plus[variable]), float(minus[variable]))
            upper = max(float(plus[variable]), float(minus[variable]))
            if np.isfinite(clipped) and lower < clipped < upper and abs(clipped - float(incumbent[variable])) > 1e-12:
                candidate = incumbent.copy()
                candidate[variable] = clipped
                error_before = float(ledger.best_error)
                vertex_error = float(np.asarray(ledger.evaluate(candidate[np.newaxis, :])).reshape(-1)[0])
                consumed += 1
                candidate_accepted = float(ledger.best_error) < error_before
                if candidate_accepted:
                    accepted += 1
                    incumbent = ledger.best_x.copy()
        receipts.append(
            RecenterCandidateReceipt(
                coordinate=variable,
                f0=f0,
                f_plus=f_plus,
                f_minus=f_minus,
                delta=delta,
                vertex=vertex,
                clipped_vertex=clipped,
                evaluated_fes=3 if vertex_error is not None else 2,
                vertex_error=vertex_error,
                accepted=candidate_accepted,
            )
        )
    return RecenterReceipt(
        boundary_phase=boundary_phase,
        selected_coordinates=tuple(selected),
        eligible_coordinates=tuple(eligible),
        consumed_fes=consumed,
        reserved_fes=MAX_FE_PER_BOUNDARY,
        returned_fes=MAX_FE_PER_BOUNDARY - consumed,
        best_error_before=best_before,
        best_error_after=float(ledger.best_error),
        accepted_count=accepted,
        candidates=tuple(receipts),
        silent_reason=None,
    )


class RecenteringMount(TransactionMount):
    """The v1 transaction mount with the kernel swapped for re-centering."""

    def __init__(self, links: list[CertifiedLink], *, enabled: bool) -> None:
        super().__init__(links, enabled=enabled)
        self.recenter_receipts: list[dict[str, Any]] = []

    def _maybe_fire(self, phase_tag: str, context: Any) -> None:
        if not self._enabled or self._fired or self._context is None:
            return
        self._fired = True
        proposals = self.proposals_by_block(self._source_phase)
        receipt = run_recentering_transaction(
            context,
            self._links,
            proposals,
            boundary_phase=phase_tag,
        )
        self.recenter_receipts.append(receipt.payload())


__all__ = [
    "CertifiedLink",
    "MAX_COORDINATES_PER_BOUNDARY",
    "MAX_FE_PER_BOUNDARY",
    "PROBE_FRACTION",
    "RecenteringMount",
    "run_recentering_transaction",
]

"""Stateless shared-variable transaction kernel and boundary mount for
shared_transaction_v1 (SCST v3.0 stages T2/T3).

Kernel contract (SCST §4, frozen):

- at a qualified boundary, select at most 4 coordinates by
  ``(-|M(j)|, coordinate_id)`` among certified links whose BOTH owners carry
  fresh strict-best proposals in the source phase;
- each coordinate yields two authority-free candidates, ``median(P_j)`` and
  ``mean(P_j)``, clipped to bounds;
- each candidate is one full-vector evaluation against the current incumbent
  (all other coordinates unchanged); ``consumed = 2 x |coordinates| <= 8``;
- ``best_error_before`` is read from the ledger, never re-evaluated;
- acceptance is exactly the ledger's strict-best rule (a strictly better
  full-objective value writes itself back); duplicates are rejected, not
  skipped;
- the lane is capped at 8 FE per boundary; unused cap is recorded as
  ``returned_fes``;
- stateless: no persistent z/r/u, no weights, no classifier, no direction
  authority, no probe FE.

The mount extends ``TransactionAuditRecorder``: T1 runs it with no boundary
action (pure observation); T2 exercises the kernel contract on toy and
mini-host problems; T3 arms run A0 (no action, bit-equal to the frozen
baseline lane) or A1 (kernel fires once at the whitelisted boundary before
the downstream phase starts, so the downstream phase re-anchors on the
post-transaction incumbent - the T1-proven propagation path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from experiments.upgrade.shared_transaction_v1.scst_instrumentation import (
    TransactionAuditRecorder,
)

MAX_COORDINATES_PER_BOUNDARY = 4
MAX_FE_PER_BOUNDARY = 8
AGGREGATORS: tuple[str, ...] = ("median", "mean")


@dataclass(frozen=True)
class CertifiedLink:
    variable: int
    owner_blocks: tuple[int, int]


@dataclass(frozen=True)
class KernelCandidateReceipt:
    coordinate: int
    aggregator: str
    raw_value: float
    clipped_value: float
    evaluated_fes: int
    error: float
    accepted: bool
    duplicate_of_previous: bool


@dataclass(frozen=True)
class KernelReceipt:
    boundary_phase: str
    requested_coordinates: tuple[int, ...]
    selected_coordinates: tuple[int, ...]
    consumed_fes: int
    reserved_fes: int
    returned_fes: int
    best_error_before: float
    best_error_after: float
    accepted_count: int
    candidates: tuple[KernelCandidateReceipt, ...]
    silent_reason: str | None

    def payload(self) -> dict[str, Any]:
        return {
            "boundary_phase": self.boundary_phase,
            "requested_coordinates": list(self.requested_coordinates),
            "selected_coordinates": list(self.selected_coordinates),
            "consumed_fes": self.consumed_fes,
            "reserved_fes": self.reserved_fes,
            "returned_fes": self.returned_fes,
            "best_error_before": self.best_error_before,
            "best_error_after": self.best_error_after,
            "accepted_count": self.accepted_count,
            "candidates": [candidate.__dict__ for candidate in self.candidates],
            "silent_reason": self.silent_reason,
        }


def build_transaction_links(certificates: Mapping[str, Any]) -> list[CertifiedLink]:
    links: list[CertifiedLink] = []
    for certificate in certificates.get("pairwise_certificates", []):
        links.append(
            CertifiedLink(
                variable=int(certificate["variable"]),
                owner_blocks=(
                    int(certificate["region_a"]) - 1,
                    int(certificate["region_b"]) - 1,
                ),
            )
        )
    return links


def run_stateless_transaction(
    context,
    links: list[CertifiedLink],
    proposals: Mapping[int, Any],
    *,
    boundary_phase: str,
) -> KernelReceipt:
    """Execute one fixed-FE stateless transaction at a qualified boundary."""

    ledger = context.ledger
    problem = context.problem
    eligible: list[tuple[int, tuple[int, int]]] = []
    for link in links:
        owners = link.owner_blocks
        if all(owner in proposals for owner in owners):
            eligible.append((link.variable, owners))
    eligible.sort(key=lambda item: (-len(item[1]), item[0]))
    selected = eligible[:MAX_COORDINATES_PER_BOUNDARY]
    best_before = float(ledger.best_error)
    if not selected:
        return KernelReceipt(
            boundary_phase=boundary_phase,
            requested_coordinates=tuple(variable for variable, _ in eligible),
            selected_coordinates=(),
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
    candidate_receipts: list[KernelCandidateReceipt] = []
    incumbent = ledger.best_x.copy()
    for variable, owners in selected:
        if consumed + len(AGGREGATORS) > MAX_FE_PER_BOUNDARY:
            break
        values = np.asarray(
            [float(proposals[owner].committed_x[variable]) for owner in owners],
            dtype=float,
        )
        previous_value: float | None = None
        for aggregator in AGGREGATORS:
            if consumed >= MAX_FE_PER_BOUNDARY:
                break
            raw = float(np.median(values)) if aggregator == "median" else float(np.mean(values))
            clipped = float(
                np.clip(raw, problem.lower_array[variable], problem.upper_array[variable])
            )
            if not np.isfinite(clipped):
                candidate_receipts.append(
                    KernelCandidateReceipt(
                        coordinate=variable,
                        aggregator=aggregator,
                        raw_value=raw,
                        clipped_value=clipped,
                        evaluated_fes=0,
                        error=float(ledger.best_error),
                        accepted=False,
                        duplicate_of_previous=False,
                    )
                )
                continue
            candidate = incumbent.copy()
            candidate[variable] = clipped
            error_before_eval = float(ledger.best_error)
            value = float(np.asarray(ledger.evaluate(candidate[np.newaxis, :])).reshape(-1)[0])
            consumed += 1
            # ledger.evaluate already applies the strict-best writeback, so
            # acceptance is "the ledger best moved during this evaluation";
            # comparing the returned value against the (post-update) best
            # would always read as rejected.
            accepted_now = float(ledger.best_error) < error_before_eval
            if accepted_now:
                accepted += 1
                incumbent = ledger.best_x.copy()
            candidate_receipts.append(
                KernelCandidateReceipt(
                    coordinate=variable,
                    aggregator=aggregator,
                    raw_value=raw,
                    clipped_value=clipped,
                    evaluated_fes=1,
                    error=value,
                    accepted=accepted_now,
                    duplicate_of_previous=(
                        previous_value is not None and clipped == previous_value
                    ),
                )
            )
            previous_value = clipped
    return KernelReceipt(
        boundary_phase=boundary_phase,
        requested_coordinates=tuple(variable for variable, _ in eligible),
        selected_coordinates=tuple(variable for variable, _ in selected),
        consumed_fes=consumed,
        reserved_fes=MAX_FE_PER_BOUNDARY,
        returned_fes=MAX_FE_PER_BOUNDARY - consumed,
        best_error_before=best_before,
        best_error_after=float(ledger.best_error),
        accepted_count=accepted,
        candidates=tuple(candidate_receipts),
        silent_reason=None,
    )


class TransactionMount(TransactionAuditRecorder):
    """Audit recorder with an optional one-shot kernel at one boundary."""

    def __init__(self, links: list[CertifiedLink], *, enabled: bool) -> None:
        super().__init__()
        self._links = list(links)
        self._enabled = bool(enabled)
        self.kernel_receipts: list[dict[str, Any]] = []
        self._context: Any = None
        self._fired = False

    def install(self, ledger: Any, context: Any = None) -> None:  # type: ignore[override]
        self._context = context
        super().install(ledger)

    def _maybe_fire(self, phase_tag: str, context: Any) -> None:
        if not self._enabled or self._fired or self._context is None:
            return
        self._fired = True
        proposals = self.proposals_by_block(self._source_phase)
        receipt = run_stateless_transaction(
            context,
            self._links,
            proposals,
            boundary_phase=phase_tag,
        )
        self.kernel_receipts.append(receipt.payload())

    def configure_boundary(self, source_phase: str, boundary_phase: str) -> None:
        self._source_phase = source_phase
        self._boundary_phase = boundary_phase

    def _wrap_runner(self, original: Callable[..., Any], name: str) -> Callable[..., Any]:
        outer = super()._wrap_runner(original, name)

        def wrapper_with_action(*args, **kwargs):
            if name == getattr(self, "_boundary_phase", None) and not self._fired and self._enabled:
                context = args[0] if args else kwargs.get("context")
                self._maybe_fire(name, context)
            return outer(*args, **kwargs)

        wrapper_with_action._scst_audit_wrapped = True
        return wrapper_with_action


__all__ = [
    "AGGREGATORS",
    "CertifiedLink",
    "KernelReceipt",
    "MAX_COORDINATES_PER_BOUNDARY",
    "MAX_FE_PER_BOUNDARY",
    "TransactionMount",
    "build_transaction_links",
    "run_stateless_transaction",
]

"""Certified-link joint rescue for the shared_transaction_v3 candidate.

Scale-axiom motivation (v1 + v2 screen receipts, 2026-08-24): both boundary
transaction kernels (8 FE arbitration, 12 FE re-centering) accepted strictly
better candidates at the boundary yet moved terminal errors by trajectory
perturbation noise - the documented F2/Gate-47b failure family (micro-window
interventions on multi-million-FE trajectories).  The meta-law's positive
side (G1/G3/G5) says real gains live at budget-ownership / lifecycle scale,
conditioned on static structure - and the frozen baseline itself contains
the proven pattern (topology-conditioned SMP lifecycle dispatch).

v3 mechanism: at the T1-qualified SMP boundary (stateful visits -> rescue),
each T0-certified link (A, B) gets ONE joint block session over the merged
scope A u B - the shared coordinates are re-optimized TOGETHER with both
owners by the same persistent block-CMA machinery rescue already uses - and
the joint budget is carved from the rescue window the host already reserved
(per-link ``FE_PER_LINK``, total capped at ``MAX_TOTAL_SHARE`` of the
requested rescue budget).  The remaining rescue budget flows to the frozen
``run_stalled_block_rescue`` untouched.  No owner authority, no classifier,
no state across boundaries; structure certification (T0) decides the scope,
the strict-best ledger decides every writeback, and the host's total FE
contract is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arac.actions import _execution as execution_module
from experiments.upgrade.shared_transaction_v1.scst_instrumentation import (
    TransactionAuditRecorder,
)

FE_PER_LINK = 20_000
MAX_TOTAL_SHARE = 0.5


@dataclass(frozen=True)
class CertifiedLink:
    variable: int
    owner_blocks: tuple[int, int]


@dataclass(frozen=True)
class JointRescueReceipt:
    boundary_phase: str
    link_scopes: tuple[tuple[int, int, int], ...]
    joint_fes: int
    forwarded_fes: int
    best_error_before: float
    best_error_after_joint: float
    accepted_links: int

    def payload(self) -> dict[str, Any]:
        return {
            "boundary_phase": self.boundary_phase,
            "link_scopes": [
                {"left_block": left, "right_block": right, "scope_size": size}
                for left, right, size in self.link_scopes
            ],
            "joint_fes": self.joint_fes,
            "forwarded_fes": self.forwarded_fes,
            "best_error_before": self.best_error_before,
            "best_error_after_joint": self.best_error_after_joint,
            "accepted_links": self.accepted_links,
        }


class JointRescueMount(TransactionAuditRecorder):
    """Audit recorder that prepends certified-link joint sessions to rescue.

    Installs the observation patches (zero-tax, T1-proven) and additionally
    wraps ``run_stalled_block_rescue``: on its first invocation the mount
    spends the preregistered joint budget on merged-scope sessions for every
    certified link whose owners produced fresh proposals in the source
    phase, then forwards the reduced ``requested_fes`` to the frozen rescue.
    """

    def __init__(self, links: list[CertifiedLink], *, enabled: bool) -> None:
        super().__init__()
        self._links = list(links)
        self._enabled = bool(enabled)
        self._source_phase = "run_stateful_block_visits"
        self._boundary_phase = "run_stalled_block_rescue"
        self._context: Any = None
        self._fired = False
        self.joint_receipts: list[dict[str, Any]] = []

    def configure_boundary(self, source_phase: str, boundary_phase: str) -> None:
        self._source_phase = source_phase
        self._boundary_phase = boundary_phase

    def install(self, ledger: Any, context: Any = None) -> None:  # type: ignore[override]
        self._context = context
        super().install(ledger)

    def _wrap_runner(self, original, name):
        parent_wrapper = super()._wrap_runner(original, name)
        if name != self._boundary_phase:
            return parent_wrapper
        mount = self

        def wrapper(*args, **kwargs):
            if mount._enabled and not mount._fired and mount._context is not None:
                mount._fired = True
                if "requested_fes" in kwargs:
                    requested = int(kwargs["requested_fes"])
                elif len(args) > 1:
                    requested = int(args[1])
                else:
                    requested = 0
                receipt = mount._run_joint_sessions(requested)
                forwarded = requested - receipt.joint_fes
                if forwarded <= 0:
                    raise RuntimeError(
                        "joint rescue consumed the entire rescue window; "
                        "the preregistered cap is violated"
                    )
                mount.joint_receipts.append(receipt.payload())
                if "requested_fes" in kwargs:
                    kwargs["requested_fes"] = forwarded
                elif len(args) > 1:
                    args = (args[0], forwarded, *args[2:])
            return parent_wrapper(*args, **kwargs)

        wrapper._scst_audit_wrapped = True
        return wrapper

    def _run_joint_sessions(self, requested_fes: int) -> JointRescueReceipt:
        context = self._context
        ledger = context.ledger
        proposals = self.proposals_by_block(self._source_phase)
        blocks = context.checkpoint.blocks
        eligible = [
            link
            for link in self._links
            if all(owner in proposals for owner in link.owner_blocks)
        ]
        best_before = float(ledger.best_error)
        if not eligible:
            return JointRescueReceipt(
                boundary_phase=self._boundary_phase,
                link_scopes=(),
                joint_fes=0,
                forwarded_fes=requested_fes,
                best_error_before=best_before,
                best_error_after_joint=best_before,
                accepted_links=0,
            )
        total_cap = int(MAX_TOTAL_SHARE * requested_fes)
        per_link = min(FE_PER_LINK, total_cap // len(eligible))
        scopes: list[tuple[int, int, int]] = []
        accepted = 0
        start = ledger.count
        for index, link in enumerate(eligible):
            left, right = link.owner_blocks
            scope = tuple(sorted(set(blocks[left]) | set(blocks[right])))
            if per_link <= 0 or ledger.remaining < per_link:
                break
            # Block sessions reject partial-population budgets; align the
            # joint window down to a whole number of generations exactly as
            # the frozen rescue machinery does.
            population = execution_module._block_population_size(len(scope))
            aligned = (per_link // population) * population
            if aligned <= 0 or ledger.remaining < aligned:
                continue
            error_before = float(ledger.best_error)
            session = execution_module._PersistentBlockSession(
                context,
                scope,
                min(left, right),
                aligned,
                population_size=population,
                seed_namespace=f"scst-v3-joint-{index}",
            )
            while not session.complete:
                session.advance()
            scopes.append((left, right, len(scope)))
            if float(ledger.best_error) < error_before:
                accepted += 1
        return JointRescueReceipt(
            boundary_phase=self._boundary_phase,
            link_scopes=tuple(scopes),
            joint_fes=ledger.count - start,
            forwarded_fes=requested_fes - (ledger.count - start),
            best_error_before=best_before,
            best_error_after_joint=float(ledger.best_error),
            accepted_links=accepted,
        )


__all__ = [
    "CertifiedLink",
    "FE_PER_LINK",
    "JointRescueMount",
    "MAX_TOTAL_SHARE",
]

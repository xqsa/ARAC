"""Certified-link joint coverage for the shared_transaction_v6 candidate.

Evidence chain that motivates v5 (all receipts from 2026-08-24):

- T1 (CTP census): under interleaved persistent coverage the strict-best
  writeback stream concentrates in 1-4 blocks of 10 - the concentration
  LOCKS IN during coverage, and no certified link ever gets both owners
  onto the writeback stream.
- v3/v4 (joint rescue): intervening at the LATE stateful->rescue boundary
  has no headroom - the incumbent there is polished by 1.2M FE of stateful
  visits, and fresh joint sessions (default or rescue-native warm-up)
  produce zero writebacks while stealing budget.
- v2 (re-centering): micro-window boundary transactions accept locally but
  their terminal effect is noise (scale axiom).

v5 therefore intervenes EARLY, inside the CTP host's own budget class: at
the entry of ``run_persistent_blocks`` (the coverage phase), every
T0-certified link (A, B) is visited as ONE EXTRA COVERAGE UNIT over the
merged scope A u B, using exactly the coverage session recipe (default
population ``BLOCK_POPULATION_SIZE``, default sigma, fresh session) - so
the shared coordinates are co-optimized with both owners BEFORE the
single-block writeback concentration locks in.  The joint window is carved
from the coverage budget the host already reserved (``FE_PER_LINK`` per
link, total capped at ``MAX_TOTAL_SHARE`` of the requested coverage FE);
the remainder flows to the frozen interleaved coverage untouched.

Selection is structure-only (all certified links): at coverage entry no
source-phase proposals exist by construction, and the T0 certificate is
the qualification.  Acceptance is the ledger strict-best rule; stateless
across runs; host total FE contract unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arac.actions import _execution as execution_module
from experiments.upgrade.shared_transaction_v1.scst_instrumentation import (
    TransactionAuditRecorder,
)

FE_PER_LINK = 20_000
MAX_TOTAL_SHARE = 0.25


@dataclass(frozen=True)
class CertifiedLink:
    variable: int
    owner_blocks: tuple[int, int]


@dataclass(frozen=True)
class JointCoverageReceipt:
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


class JointCoverageMount(TransactionAuditRecorder):
    """Audit recorder that prepends certified-link joint units to coverage."""

    def __init__(self, links: list[CertifiedLink], *, enabled: bool) -> None:
        super().__init__()
        # One joint unit per unique certified block pair.  v5's screen ran
        # one unit per certificate (eight duplicated scopes per pair, each
        # starved to ~4.7k FE by the cap) and produced zero writebacks; the
        # frozen FE_PER_LINK applies to the pair-level link.
        deduped: dict[tuple[int, int], int] = {}
        for link in links:
            key = (min(link.owner_blocks), max(link.owner_blocks))
            deduped.setdefault(key, link.variable)
        self._links = [
            CertifiedLink(variable=variable, owner_blocks=(left, right))
            for (left, right), variable in sorted(deduped.items())
        ]
        self._enabled = bool(enabled)
        self._boundary_phase = "run_persistent_blocks"
        self._context: Any = None
        self._fired = False
        self.joint_receipts: list[dict[str, Any]] = []

    def configure_boundary(self, boundary_phase: str) -> None:
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
                receipt = mount._run_joint_units(requested)
                forwarded = requested - receipt.joint_fes
                if forwarded <= 0:
                    raise RuntimeError(
                        "joint coverage consumed the entire coverage window; "
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

    def _run_joint_units(self, requested_fes: int) -> JointCoverageReceipt:
        context = self._context
        ledger = context.ledger
        blocks = context.checkpoint.blocks
        best_before = float(ledger.best_error)
        if not self._links:
            return JointCoverageReceipt(
                boundary_phase=self._boundary_phase,
                link_scopes=(),
                joint_fes=0,
                forwarded_fes=requested_fes,
                best_error_before=best_before,
                best_error_after_joint=best_before,
                accepted_links=0,
            )
        total_cap = int(MAX_TOTAL_SHARE * requested_fes)
        per_link = min(FE_PER_LINK, total_cap // len(self._links))
        scopes: list[tuple[int, int, int]] = []
        accepted = 0
        start = ledger.count
        for index, link in enumerate(self._links):
            left, right = link.owner_blocks
            scope = tuple(sorted(set(blocks[left]) | set(blocks[right])))
            # Coverage sessions use the default population constant; align
            # the joint window to whole generations of exactly that recipe.
            population = execution_module.BLOCK_POPULATION_SIZE
            aligned = (per_link // population) * population
            if aligned <= 0 or ledger.remaining < aligned:
                continue
            error_before = float(ledger.best_error)
            session = execution_module._PersistentBlockSession(
                context,
                scope,
                min(left, right),
                aligned,
                seed_namespace=f"scst-v6-joint-{index}",
            )
            while not session.complete:
                session.advance()
            scopes.append((left, right, len(scope)))
            if float(ledger.best_error) < error_before:
                accepted += 1
        return JointCoverageReceipt(
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
    "JointCoverageMount",
    "MAX_TOTAL_SHARE",
]

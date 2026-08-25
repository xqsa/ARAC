"""Per-visit shared-coordinate re-centering for shared_transaction_v7.

The terminal design of the receipt chain (2026-08-24):

- v2: the 3-point quadratic re-centering of a shared coordinate is always
  accepted by strict-best (4/4 coordinates on every arm) - the repair
  itself works; only its ONCE-PER-RUN timing drowns in trajectory noise.
- v3-v6: one-shot restart-scale interventions (late or early) never write
  back; fresh sessions cannot compete with the hosts' persistent machinery.

v7 therefore applies the proven local repair CONTINUOUSLY, at the moment
the damage occurs: every time a certified-owner block finishes a stateful
visit (which moves that owner's coordinates while the shared coordinate
stays frozen at the other owner's stale value), the kernel re-centers the
owner's certified shared coordinates with the v2 quadratic fit (3 FE per
coordinate).  Budget discipline: at most ``MAX_FE_PER_VISIT`` per visit and
a hard run-level cap of ``MAX_TOTAL_FRACTION`` of the Phase-II budget; once
the cap is hit the kernel goes silent for the rest of the run and the cap
consumption is recorded.  No classifier, no online selection signal, no
state across visits - a deterministic schedule over T0-certified structure,
with strict-best as the only acceptance authority.  On AOB (no certified
structure in the contract) the mount is structurally silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arac.actions import _execution as execution_module
from experiments.upgrade.shared_transaction_v1.scst_instrumentation import (
    TransactionAuditRecorder,
)

MAX_FE_PER_VISIT = 24
MAX_TOTAL_FRACTION = 0.01
PROBE_FRACTION = 0.25


@dataclass(frozen=True)
class CertifiedLink:
    variable: int
    owner_blocks: tuple[int, int]


@dataclass
class VisitRecenterLedger:
    visits_seen: int = 0
    recentered_visits: int = 0
    coordinates_evaluated: int = 0
    accepted: int = 0
    consumed_fes: int = 0
    cap_hit: bool = False
    error_before_first: float | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "visits_seen": self.visits_seen,
            "recentered_visits": self.recentered_visits,
            "coordinates_evaluated": self.coordinates_evaluated,
            "accepted": self.accepted,
            "consumed_fes": self.consumed_fes,
            "cap_hit": self.cap_hit,
            "error_before_first": self.error_before_first,
        }


class PerVisitRecenterMount(TransactionAuditRecorder):
    """Audit recorder that re-centers shared coordinates after owner visits.

    Patches ``_execution._run_block_visit`` (called once per stateful visit):
    after the frozen visit completes, if the visited block is a certified
    owner, each of its certified shared coordinates gets one quadratic
    re-centering against the live ledger incumbent.
    """

    def __init__(self, links: list[CertifiedLink], *, enabled: bool, phase2_budget: int) -> None:
        super().__init__()
        # owner block -> certified shared coordinates (unique)
        owner_coordinates: dict[int, list[int]] = {}
        for link in links:
            left, right = link.owner_blocks
            if link.variable not in owner_coordinates.setdefault(left, []):
                owner_coordinates[left].append(link.variable)
            if link.variable not in owner_coordinates.setdefault(right, []):
                owner_coordinates[right].append(link.variable)
        self._owner_coordinates = {owner: tuple(sorted(coords)) for owner, coords in owner_coordinates.items()}
        self._enabled = bool(enabled)
        self._budget_cap = int(MAX_TOTAL_FRACTION * phase2_budget)
        self._context: Any = None
        self._orig_run_block_visit = execution_module._run_block_visit
        self.ledger_stats = VisitRecenterLedger()

    def install(self, ledger: Any, context: Any = None) -> None:  # type: ignore[override]
        self._context = context
        super().install(ledger)
        if not self._enabled:
            return
        mount = self
        original = self._orig_run_block_visit

        def patched_visit(session, budget_fes):
            consumed = original(session, budget_fes)
            mount._after_visit(session)
            return consumed

        execution_module._run_block_visit = patched_visit

    def uninstall(self) -> None:
        if self._enabled and self._orig_run_block_visit is not None:
            execution_module._run_block_visit = self._orig_run_block_visit
        super().uninstall()

    def _after_visit(self, session) -> None:
        stats = self.ledger_stats
        stats.visits_seen += 1
        if self._context is None or stats.cap_hit:
            return
        coordinates = self._owner_coordinates.get(int(session.block_index))
        if not coordinates:
            return
        ledger = self._context.ledger
        problem = self._context.problem
        span = problem.upper_array - problem.lower_array
        stats.recentered_visits += 1
        if stats.error_before_first is None:
            stats.error_before_first = float(ledger.best_error)
        budget_left = self._budget_cap - stats.consumed_fes
        visit_budget = min(MAX_FE_PER_VISIT, budget_left)
        if visit_budget < 3:
            stats.cap_hit = True
            return
        incumbent = ledger.best_x.copy()
        for variable in coordinates:
            if visit_budget < 3:
                stats.cap_hit = True
                break
            delta = PROBE_FRACTION * float(span[variable])
            f0 = float(ledger.best_error)
            plus = incumbent.copy()
            plus[variable] = float(np.clip(incumbent[variable] + delta, problem.lower_array[variable], problem.upper_array[variable]))
            error_before = float(ledger.best_error)
            f_plus = float(np.asarray(ledger.evaluate(plus[np.newaxis, :])).reshape(-1)[0])
            stats.consumed_fes += 1
            stats.coordinates_evaluated += 1
            visit_budget -= 1
            if float(ledger.best_error) < error_before:
                stats.accepted += 1
                incumbent = ledger.best_x.copy()
                continue
            if visit_budget < 2:
                break
            minus = incumbent.copy()
            minus[variable] = float(np.clip(incumbent[variable] - delta, problem.lower_array[variable], problem.upper_array[variable]))
            error_before = float(ledger.best_error)
            f_minus = float(np.asarray(ledger.evaluate(minus[np.newaxis, :])).reshape(-1)[0])
            stats.consumed_fes += 1
            stats.coordinates_evaluated += 1
            visit_budget -= 1
            if float(ledger.best_error) < error_before:
                stats.accepted += 1
                incumbent = ledger.best_x.copy()
                continue
            curvature = f_plus + f_minus - 2.0 * f0
            if not np.isfinite(curvature) or curvature <= 0.0 or visit_budget < 1:
                continue
            shift = delta * (f_minus - f_plus) / (2.0 * curvature)
            vertex = float(incumbent[variable] + shift)
            clipped = float(np.clip(vertex, problem.lower_array[variable], problem.upper_array[variable]))
            lower = min(float(plus[variable]), float(minus[variable]))
            upper = max(float(plus[variable]), float(minus[variable]))
            if not (lower < clipped < upper) or abs(clipped - float(incumbent[variable])) <= 1e-12:
                continue
            candidate = incumbent.copy()
            candidate[variable] = clipped
            error_before = float(ledger.best_error)
            ledger.evaluate(candidate[np.newaxis, :])
            stats.consumed_fes += 1
            stats.coordinates_evaluated += 1
            visit_budget -= 1
            if float(ledger.best_error) < error_before:
                stats.accepted += 1
                incumbent = ledger.best_x.copy()


__all__ = [
    "CertifiedLink",
    "MAX_FE_PER_VISIT",
    "MAX_TOTAL_FRACTION",
    "PerVisitRecenterMount",
    "PROBE_FRACTION",
]

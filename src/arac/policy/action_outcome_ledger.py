"""Offline delayed-outcome ledger for future selector evaluation.

Two-phase lifecycle:
  1. record_pending  — called at action application time (sweep N).
  2. close_pending   — called at sweep N+1 end (after all groups complete).

The ledger is intentionally not instantiated by the HCC runner during Action
Validation. It provides a tested N-to-N+1 outcome contract without enabling a
selector or changing runtime optimization.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from arac.policy.overlap_hypergraph import DelayedHyperedgeCredit, build_delayed_hyperedge_credit


@dataclass(frozen=True)
class ActionOutcome:
    action_name: str
    semantic_surface: str
    sweep_index: int
    penalized_credit: float
    survival: float
    next_sweep_log_improvement: float
    evidence_snapshot: dict[str, float]

    def __post_init__(self) -> None:
        if not self.action_name:
            raise ValueError("action_name must be non-empty")
        if not math.isfinite(self.penalized_credit):
            raise ValueError("penalized_credit must be finite")


@dataclass
class _PendingRecord:
    action_name: str
    semantic_surface: str
    sweep_index: int
    anchor_error: float
    anchor_shared_values: tuple[float, ...]
    candidate_shared_values: tuple[float, ...]
    shared_variable_indices: tuple[int, ...]
    evidence_snapshot: dict[str, float]


@dataclass
class ActionOutcomeLedger:
    """Accumulates ActionOutcome records and provides per-action credit statistics.

    Not thread-safe — one instance per run.
    """

    _pending: dict[str, _PendingRecord] = field(default_factory=dict, init=False, repr=False)
    _outcomes: list[ActionOutcome] = field(default_factory=list, init=False, repr=False)
    _credit_sums: dict[str, float] = field(
        default_factory=lambda: defaultdict(float), init=False, repr=False
    )
    _credit_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int), init=False, repr=False
    )

    def record_pending(
        self,
        *,
        relation_id: str,
        action_name: str,
        semantic_surface: str,
        sweep_index: int,
        anchor_error: float,
        anchor_shared_values: Sequence[float],
        candidate_shared_values: Sequence[float],
        shared_variable_indices: Sequence[int],
        evidence_snapshot: dict[str, float],
    ) -> None:
        """Store the action application state at sweep N for later closure."""
        self._pending[relation_id] = _PendingRecord(
            action_name=action_name,
            semantic_surface=semantic_surface,
            sweep_index=sweep_index,
            anchor_error=float(anchor_error),
            anchor_shared_values=tuple(float(v) for v in anchor_shared_values),
            candidate_shared_values=tuple(float(v) for v in candidate_shared_values),
            shared_variable_indices=tuple(int(i) for i in shared_variable_indices),
            evidence_snapshot=dict(evidence_snapshot),
        )

    def close_pending(
        self,
        *,
        relation_id: str,
        resolution_sweep_index: int,
        next_sweep_error: float,
        next_sweep_shared_values: Sequence[float],
        all_groups_completed: bool,
        native_sweep_end_completed: bool,
    ) -> ActionOutcome | None:
        """Close the pending record for relation_id at sweep N+1.

        Returns the ActionOutcome on success, None if no pending record exists
        or the sweep constraints aren't met (partial sweep).
        """
        pending = self._pending.get(relation_id)
        if pending is None:
            return None
        if not all_groups_completed or not native_sweep_end_completed:
            return None
        if resolution_sweep_index != pending.sweep_index + 1:
            raise ValueError("action outcome must close at the next complete sweep")
        credit: DelayedHyperedgeCredit = build_delayed_hyperedge_credit(
            action_sweep_index=pending.sweep_index,
            resolution_sweep_index=resolution_sweep_index,
            all_groups_completed=all_groups_completed,
            native_sweep_end_completed=native_sweep_end_completed,
            anchor_error=pending.anchor_error,
            next_sweep_error=float(next_sweep_error),
            anchor_shared_values=pending.anchor_shared_values,
            candidate_shared_values=pending.candidate_shared_values,
            next_sweep_shared_values=tuple(float(v) for v in next_sweep_shared_values),
        )
        outcome = ActionOutcome(
            action_name=pending.action_name,
            semantic_surface=pending.semantic_surface,
            sweep_index=pending.sweep_index,
            penalized_credit=credit.penalized_credit,
            survival=credit.survival,
            next_sweep_log_improvement=credit.next_sweep_log_improvement,
            evidence_snapshot=pending.evidence_snapshot,
        )
        self._outcomes.append(outcome)
        self._credit_sums[pending.action_name] += credit.penalized_credit
        self._credit_counts[pending.action_name] += 1
        self._pending.pop(relation_id)
        return outcome

    def close_sweep(
        self,
        *,
        best_individual: Sequence[float],
        next_sweep_error: float,
        completed_sweep_index: int,
        all_groups_completed: bool,
        native_sweep_end_completed: bool,
    ) -> list[ActionOutcome]:
        """Close all records issued in the sweep before completed_sweep_index.

        Called right before outer_iter += 1.  Uses stored shared_variable_indices
        to extract next_sweep_shared_values from best_individual.
        Returns the list of newly closed outcomes (may be empty).
        """
        if not all_groups_completed or not native_sweep_end_completed:
            return []
        if any(
            rec.sweep_index + 1 < completed_sweep_index
            for rec in self._pending.values()
        ):
            raise ValueError("pending action outcome missed its next sweep")
        keys_to_close = [
            key for key, rec in self._pending.items()
            if rec.sweep_index + 1 == completed_sweep_index
        ]
        closed: list[ActionOutcome] = []
        for key in keys_to_close:
            pending = self._pending[key]
            indices = pending.shared_variable_indices
            next_vals = tuple(float(best_individual[i]) for i in indices)
            outcome = self.close_pending(
                relation_id=key,
                resolution_sweep_index=completed_sweep_index,
                next_sweep_error=next_sweep_error,
                next_sweep_shared_values=next_vals,
                all_groups_completed=all_groups_completed,
                native_sweep_end_completed=native_sweep_end_completed,
            )
            if outcome is None:
                raise RuntimeError("eligible pending outcome was not closed")
            closed.append(outcome)
        return closed

    def mean_credit(self, action_name: str) -> float | None:
        """Return mean penalized credit for an action, or None if unseen."""

        n = self._credit_counts.get(action_name, 0)
        if n == 0:
            return None
        return self._credit_sums[action_name] / n

    def all_outcomes(self) -> list[ActionOutcome]:
        return list(self._outcomes)

    def pending_relation_ids(self) -> set[str]:
        return set(self._pending)

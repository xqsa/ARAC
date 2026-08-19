"""Shared state-machine boundaries for interruptible Phase-II actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import (
    ACTION_NAMES,
    ActionContext,
    ActionExecutionRegistry,
    ActionResult,
    PhaseCheckpoint,
    Phase2Snapshot,
    Phase2StepResult,
)
from arac.runtime.ledger import EvaluationLedger


class Phase2StateError(RuntimeError):
    """Raised when a Phase-II state transition violates its contract."""


@dataclass(frozen=True)
class EpisodeProgress:
    """Unified scheduler-facing progress contract (v4 plan section 3).

    Protocol facts only.  ``protocol_mature`` reports whether the episode
    has completed a legal semantic state unit given the caller's
    ``maturity_window_fes`` (CTP: coverage end + one polish window; GSS:
    warmup sweep + one continuation window; SMP: one full visit + window;
    AOR: one independent correction window).  Evidence revelation
    (cumulative development budget versus a measured horizon) is a
    scheduler-ledger concept and deliberately absent from this contract.

    ``maturity_target_fes`` is a planning estimate of the cumulative
    Phase-II FE needed to become protocol-mature; ``protocol_mature`` is
    the authoritative live predicate.
    """

    episode: str
    phase: str
    consumed_fes: int
    next_boundary_fes: int
    min_step_fes: int
    maturity_target_fes: int
    protocol_mature: bool
    contract: str


class ResumablePhase2State(ABC):
    """Base class enforcing exact FE accounting and checkpoint binding.

    Subclasses implement only the action-specific transition and state payload.
    The transition must consume exactly the requested number of objective calls;
    returning early would make branch prefixes incomparable.
    """

    def __init__(self, context: ActionContext) -> None:
        if not isinstance(context, ActionContext):
            raise TypeError("Phase-II state requires ActionContext")
        self.context = context
        self.action_name = context.action_name
        self.checkpoint_hash = context.checkpoint.checkpoint_hash
        self.action_seed = context.action_seed
        self.start_fes = context.checkpoint.phase1_fes
        self.total_fes = context.checkpoint.total_budget_fes
        self._last_step_snapshot: Phase2Snapshot | None = None
        self._validate_position()

    @property
    def consumed_fes(self) -> int:
        return self.context.ledger.count - self.start_fes

    @property
    def complete(self) -> bool:
        return self.context.ledger.count == self.total_fes

    def _validate_position(self) -> None:
        ledger = self.context.ledger
        if ledger.count < self.start_fes or ledger.count > self.total_fes:
            raise Phase2StateError("ledger is outside the Phase-II state boundary")
        if ledger.total_budget != self.total_fes:
            raise Phase2StateError("ledger budget is not bound to the Phase-II state")
        if self.context.checkpoint.checkpoint_hash != self.checkpoint_hash:
            raise Phase2StateError("checkpoint hash changed after state creation")

    def step(self, budget_fes: int) -> Phase2StepResult:
        if self.complete:
            raise Phase2StateError("completed Phase-II state cannot advance")
        if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes <= 0:
            raise ValueError("budget_fes must be a positive integer")
        remaining = self.total_fes - self.context.ledger.count
        if budget_fes > remaining:
            raise ValueError("step budget exceeds the remaining Phase-II budget")
        before = self.context.ledger.count
        self._advance(budget_fes)
        consumed = self.context.ledger.count - before
        if consumed != budget_fes:
            raise Phase2StateError(
                f"Phase-II state consumed {consumed} FE instead of {budget_fes}"
            )
        self._validate_position()
        snapshot = self.snapshot()
        self._last_step_snapshot = snapshot
        return Phase2StepResult(
            action_name=self.action_name,
            checkpoint_hash=self.checkpoint_hash,
            action_seed=self.action_seed,
            step_fes=consumed,
            consumed_fes=self.consumed_fes,
            total_fes=self.total_fes - self.start_fes,
            best_error=float(self.context.ledger.best_error),
            complete=self.complete,
            state_hash=snapshot.state_hash,
        )

    @property
    def last_step_snapshot(self) -> Phase2Snapshot:
        """Return the snapshot already produced by the last successful step.

        Coordinators often need both hashes from a completed transition.  The
        state transition already built the full snapshot, so exposing it avoids
        serializing the same state a second time at the receipt boundary.
        """

        if self._last_step_snapshot is None:
            raise Phase2StateError("no successful Phase-II step has produced a snapshot")
        return self._last_step_snapshot

    def snapshot(self) -> Phase2Snapshot:
        self._validate_position()
        payload = self._snapshot_payload()
        if not isinstance(payload, bytes) or not payload:
            raise Phase2StateError("state payload must be non-empty bytes")
        return Phase2Snapshot(
            action_name=self.action_name,
            checkpoint_hash=self.checkpoint_hash,
            action_seed=self.action_seed,
            start_fes=self.start_fes,
            consumed_fes=self.consumed_fes,
            total_fes=self.total_fes,
            incumbent=tuple(float(value) for value in self.context.ledger.best_x),
            best_error=float(self.context.ledger.best_error),
            state_payload=payload,
            state_hash=hashlib.sha256(payload).hexdigest(),
        )

    def progress(self, *, maturity_window_fes: int = 20_000) -> EpisodeProgress:
        """Generic single-window progress: one window is one semantic unit.

        Subclasses with internal phase structure (coverage/polish, warmup
        sweeps, stateful block visits) override this with their own
        contract; the default treats the whole Phase-II run as a single
        regime whose maturity is one ``maturity_window_fes`` window.
        """

        if isinstance(maturity_window_fes, bool) or not isinstance(maturity_window_fes, int):
            raise ValueError("maturity_window_fes must be an integer")
        if maturity_window_fes <= 0:
            raise ValueError("maturity_window_fes must be positive")
        total_phase2 = self.total_fes - self.start_fes
        remaining = self.context.ledger.remaining
        target = min(maturity_window_fes, total_phase2)
        return EpisodeProgress(
            episode=self.action_name,
            phase="single_window",
            consumed_fes=self.consumed_fes,
            next_boundary_fes=min(maturity_window_fes, remaining),
            min_step_fes=1,
            maturity_target_fes=target,
            protocol_mature=self.consumed_fes >= target,
            contract="resumable-single-window-v1",
        )

    @abstractmethod
    def _advance(self, budget_fes: int) -> None:
        """Consume exactly ``budget_fes`` objective evaluations."""

    @abstractmethod
    def _snapshot_payload(self) -> bytes:
        """Return a deterministic, action-private state payload."""


def execute_phase2_action(
    action_name: str,
    checkpoint: PhaseCheckpoint,
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    action_seed: int,
    registry: ActionExecutionRegistry,
) -> ActionResult:
    """Execute one named action from one exact Phase-I checkpoint."""

    if action_name not in ACTION_NAMES:
        raise ValueError("Phase-II action is not in the frozen action set")
    if not isinstance(checkpoint, PhaseCheckpoint):
        raise TypeError("Phase-II execution requires PhaseCheckpoint")
    if not isinstance(problem, OptimizationProblem):
        raise TypeError("Phase-II execution requires OptimizationProblem")
    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("Phase-II execution requires EvaluationLedger")
    if problem is not ledger.problem:
        raise ValueError("Phase-II problem and ledger problem must be identical")
    if ledger.count != checkpoint.phase1_fes:
        raise ValueError("Phase-II execution must start at the Phase-I boundary")
    if (
        tuple(ledger.best_x) != checkpoint.incumbent
        or ledger.best_error != checkpoint.incumbent_error
    ):
        raise ValueError("Phase-II ledger incumbent does not match the Phase-I checkpoint")
    if tuple(registry.action_names) != ACTION_NAMES:
        raise ValueError("Phase-II registry does not cover the frozen action set")
    if ledger.allow_out_of_bounds is not registry.allow_out_of_bounds:
        raise ValueError("Phase-II ledger does not match the registry execution profile")

    result = registry.execute(
        ActionContext(action_name, checkpoint, problem, ledger, action_seed)
    )
    if result.action_name != action_name:
        raise RuntimeError("Phase-II registry executed an unexpected action")
    if result.checkpoint_hash != checkpoint.checkpoint_hash:
        raise RuntimeError("Phase-II action result is not bound to the Phase-I checkpoint")
    if ledger.count != checkpoint.total_budget_fes:
        raise RuntimeError("Phase-II action did not consume the exact terminal FE budget")
    if result.terminal_fes != checkpoint.total_budget_fes:
        raise RuntimeError("Phase-II action result reported the wrong terminal FE")
    return result


def validate_snapshot_context(context: ActionContext, snapshot: Phase2Snapshot) -> None:
    """Validate the public bindings before a subclass restores private state."""

    if not isinstance(context, ActionContext):
        raise TypeError("snapshot restore requires ActionContext")
    if not isinstance(snapshot, Phase2Snapshot):
        raise TypeError("snapshot must be Phase2Snapshot")
    if snapshot.action_name != context.action_name:
        raise Phase2StateError("snapshot action does not match context")
    if snapshot.checkpoint_hash != context.checkpoint.checkpoint_hash:
        raise Phase2StateError("snapshot checkpoint hash does not match context")
    if snapshot.action_seed != context.action_seed:
        raise Phase2StateError("snapshot action seed does not match context")
    if snapshot.start_fes != context.checkpoint.phase1_fes:
        raise Phase2StateError("snapshot start FE does not match checkpoint")
    if snapshot.total_fes != context.checkpoint.total_budget_fes:
        raise Phase2StateError("snapshot total FE does not match checkpoint")
    expected_count = snapshot.start_fes + snapshot.consumed_fes
    if context.ledger.count != expected_count:
        raise Phase2StateError("ledger position does not match snapshot")
    if tuple(float(value) for value in context.ledger.best_x) != snapshot.incumbent:
        raise Phase2StateError("ledger incumbent does not match snapshot")
    if float(context.ledger.best_error) != snapshot.best_error:
        raise Phase2StateError("ledger best error does not match snapshot")


__all__ = [
    "EpisodeProgress",
    "Phase2StateError",
    "ResumablePhase2State",
    "execute_phase2_action",
    "validate_snapshot_context",
]

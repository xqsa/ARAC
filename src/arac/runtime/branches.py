"""Common-anchor branch probing for the Phase-II v2 protocol."""

from __future__ import annotations

from dataclasses import dataclass

from arac.runtime.contracts import ACTION_NAMES, ActionContext, Phase2Snapshot, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import Phase2StateError


BRANCH_PROBE_SCHEMA = "arac-common-anchor-probe-v1"


@dataclass(frozen=True)
class CommonAnchorProbeSnapshot:
    """Immutable common-boundary record containing one snapshot per action."""

    checkpoint_hash: str
    branch_budget_fes: int
    action_seeds: tuple[tuple[str, int], ...]
    branches: tuple[Phase2Snapshot, ...]
    schema_version: str = BRANCH_PROBE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BRANCH_PROBE_SCHEMA:
            raise ValueError("common-anchor probe schema drifted")
        if len(self.checkpoint_hash) != 64:
            raise ValueError("checkpoint_hash must be SHA-256")
        if isinstance(self.branch_budget_fes, bool) or self.branch_budget_fes <= 0:
            raise ValueError("branch_budget_fes must be positive")
        names = tuple(name for name, _ in self.action_seeds)
        if names != ACTION_NAMES:
            raise ValueError("probe action seeds must cover the frozen action set")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for _, seed in self.action_seeds
        ):
            raise ValueError("probe action seeds must be non-negative integers")
        if len(self.branches) != len(ACTION_NAMES):
            raise ValueError("probe must contain one branch per action")
        if tuple(snapshot.action_name for snapshot in self.branches) != ACTION_NAMES:
            raise ValueError("probe branch order drifted")
        consumed = {snapshot.consumed_fes for snapshot in self.branches}
        if len(consumed) != 1 or next(iter(consumed)) > self.branch_budget_fes:
            raise ValueError("probe snapshots must share one bounded branch position")

    @property
    def snapshot_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "checkpoint_hash": self.checkpoint_hash,
            "branch_budget_fes": self.branch_budget_fes,
            "action_seeds": [list(item) for item in self.action_seeds],
            "branches": [snapshot.snapshot_hash for snapshot in self.branches],
        }
        return canonical_sha256(payload)


class CommonAnchorProbe:
    """Run equal-budget action branches from one frozen Phase-I incumbent.

    The probe is deliberately separate from a terminal action run.  Every
    branch receives its own ledger and exact FE budget; committing one branch
    only exposes that branch for the remaining terminal budget.
    """

    def __init__(
        self,
        context: ActionContext,
        *,
        branch_budget_fes: int,
        action_seeds: dict[str, int] | None = None,
    ) -> None:
        if not isinstance(context, ActionContext):
            raise TypeError("common-anchor probe requires ActionContext")
        if context.ledger.count != context.checkpoint.phase1_fes:
            raise Phase2StateError("common-anchor probe must start at the Phase-I boundary")
        if isinstance(branch_budget_fes, bool) or not isinstance(branch_budget_fes, int):
            raise ValueError("branch_budget_fes must be a positive integer")
        if not 0 < branch_budget_fes <= context.checkpoint.remaining_fes:
            raise ValueError("branch budget is outside the Phase-II budget")
        supplied = {} if action_seeds is None else dict(action_seeds)
        unknown = set(supplied) - set(ACTION_NAMES)
        if unknown:
            raise ValueError(f"unknown action seeds: {sorted(unknown)}")
        self.checkpoint = context.checkpoint
        self.problem = context.problem
        self.branch_budget_fes = branch_budget_fes
        self.action_seeds = {
            action: int(supplied.get(action, context.action_seed)) for action in ACTION_NAMES
        }
        if any(seed < 0 for seed in self.action_seeds.values()):
            raise ValueError("action seeds must be non-negative")
        self._states = self._create_states()
        self._committed_action: str | None = None

    def _create_states(self):
        from arac.actions.phase2_v2 import initialize_state

        states = {}
        for action in ACTION_NAMES:
            ledger = EvaluationLedger.from_checkpoint(
                self.problem,
                total_budget=self.checkpoint.total_budget_fes,
                phase1_fes=self.checkpoint.phase1_fes,
                incumbent=self.checkpoint.incumbent,
                incumbent_error=self.checkpoint.incumbent_error,
            )
            branch_context = ActionContext(
                action,
                self.checkpoint,
                self.problem,
                ledger,
                self.action_seeds[action],
            )
            states[action] = initialize_state(branch_context)
        return states

    @property
    def complete(self) -> bool:
        return all(state.consumed_fes == self.branch_budget_fes for state in self._states.values())

    @property
    def committed_action(self) -> str | None:
        return self._committed_action

    @property
    def states(self):
        return dict(self._states)

    @property
    def trajectories(self) -> dict[str, tuple[float, ...]]:
        """Return read-only best-so-far traces at the shared probe boundary."""

        positions = {state.consumed_fes for state in self._states.values()}
        if len(positions) != 1:
            raise Phase2StateError("common-anchor branches are not at one shared position")
        return {
            action: tuple(float(value) for value in self._states[action].best_trace)
            for action in ACTION_NAMES
        }

    def advance(self, step_fes: int) -> None:
        """Advance every uncommitted branch by the same bounded FE slice."""

        if self._committed_action is not None:
            raise Phase2StateError("common-anchor probe is already committed")
        if isinstance(step_fes, bool) or not isinstance(step_fes, int) or step_fes <= 0:
            raise ValueError("step_fes must be a positive integer")
        for state in self._states.values():
            remaining = self.branch_budget_fes - state.consumed_fes
            if remaining:
                state.step(min(step_fes, remaining))

    def extend(self, new_branch_budget_fes: int) -> None:
        """Increase the protected probe cap without replaying prior evaluations."""

        if self._committed_action is not None:
            raise Phase2StateError("common-anchor probe is already committed")
        if (
            isinstance(new_branch_budget_fes, bool)
            or not isinstance(new_branch_budget_fes, int)
            or new_branch_budget_fes <= self.branch_budget_fes
        ):
            raise ValueError("new branch budget must strictly increase")
        if new_branch_budget_fes > self.checkpoint.remaining_fes:
            raise ValueError("new branch budget exceeds the Phase-II budget")
        self.branch_budget_fes = new_branch_budget_fes

    def best_action(self) -> str:
        if not self.complete:
            raise Phase2StateError("common-anchor probe is not complete")
        return min(ACTION_NAMES, key=lambda action: (self._states[action].context.ledger.best_error, action))

    def commit(self, action_name: str):
        """Commit one probed branch and return its resumable state."""

        if action_name not in ACTION_NAMES:
            raise ValueError("unsupported committed action")
        if not self.complete:
            raise Phase2StateError("common-anchor probe must finish before commit")
        self._committed_action = action_name
        return self._states[action_name]

    def snapshot(self) -> CommonAnchorProbeSnapshot:
        if self._committed_action is not None:
            raise Phase2StateError("a committed probe cannot be snapshotted")
        positions = {state.consumed_fes for state in self._states.values()}
        if len(positions) != 1:
            raise Phase2StateError("common-anchor branches are not at one shared position")
        return CommonAnchorProbeSnapshot(
            checkpoint_hash=self.checkpoint.checkpoint_hash,
            branch_budget_fes=self.branch_budget_fes,
            action_seeds=tuple((action, self.action_seeds[action]) for action in ACTION_NAMES),
            branches=tuple(self._states[action].snapshot() for action in ACTION_NAMES),
        )

    @classmethod
    def restore(
        cls,
        context: ActionContext,
        snapshot: CommonAnchorProbeSnapshot,
    ) -> CommonAnchorProbe:
        """Restore all isolated branches at one common probe boundary."""

        if not isinstance(snapshot, CommonAnchorProbeSnapshot):
            raise TypeError("snapshot must be CommonAnchorProbeSnapshot")
        if snapshot.checkpoint_hash != context.checkpoint.checkpoint_hash:
            raise Phase2StateError("common-anchor checkpoint does not match snapshot")
        probe = cls(
            context,
            branch_budget_fes=snapshot.branch_budget_fes,
            action_seeds=dict(snapshot.action_seeds),
        )
        from arac.actions.phase2_v2 import restore_state

        restored = {}
        for action, branch_snapshot in zip(ACTION_NAMES, snapshot.branches, strict=True):
            ledger = EvaluationLedger.from_phase2_snapshot(context.problem, branch_snapshot)
            branch_context = ActionContext(
                action,
                context.checkpoint,
                context.problem,
                ledger,
                probe.action_seeds[action],
            )
            restored[action] = restore_state(branch_context, branch_snapshot)
        probe._states = restored
        return probe


__all__ = ["BRANCH_PROBE_SCHEMA", "CommonAnchorProbe", "CommonAnchorProbeSnapshot"]

"""Train-free common-anchor probe and delayed-commit runtime policy."""

from __future__ import annotations

from dataclasses import dataclass

from arac.analysis.delayed_commit import CommitDecision, decide_delayed_commit
from arac.runtime.branches import CommonAnchorProbe
from arac.runtime.contracts import ACTION_NAMES, ActionContext


@dataclass(frozen=True)
class ProbePolicyResult:
    """Auditable global-budget result of one probe-and-commit run."""

    selected_action: str
    archive_source_action: str
    commit_reason: str
    checkpoint_hash: str
    global_total_fes: int
    action_schedule_total_fes: int
    selected_ledger_fes: int
    phase1_fes: int
    branch_probe_fes: int
    continuation_fes: int
    aggregate_fes: int
    selected_action_fes: int
    final_error: float
    incumbent: tuple[float, ...]
    probe_final_errors: tuple[tuple[str, float], ...]
    decision: CommitDecision
    route: str
    optimizer_package: str
    optimizer_version: str
    selected_state_hash: str
    numerical_repair_count: int

    def __post_init__(self) -> None:
        if self.selected_action not in ACTION_NAMES:
            raise ValueError("probe policy selected an unsupported action")
        if self.archive_source_action not in ACTION_NAMES:
            raise ValueError("probe policy archive source is invalid")
        if len(self.checkpoint_hash) != 64:
            raise ValueError("probe policy checkpoint hash is invalid")
        if self.aggregate_fes != self.global_total_fes:
            raise ValueError("probe policy did not consume the global FE budget")
        if self.action_schedule_total_fes != self.global_total_fes:
            raise ValueError("probe policy action schedule budget drifted")
        if self.selected_ledger_fes + 3 * self.branch_probe_fes != self.global_total_fes:
            raise ValueError("probe policy selected ledger did not reserve branch probes")
        if tuple(action for action, _ in self.probe_final_errors) != ACTION_NAMES:
            raise ValueError("probe policy errors do not cover the frozen action set")
        if not self.route or not self.optimizer_package or not self.optimizer_version:
            raise ValueError("probe policy optimizer provenance is incomplete")
        if len(self.selected_state_hash) != 64:
            raise ValueError("probe policy state hash is invalid")
        if self.numerical_repair_count < 0:
            raise ValueError("probe policy numerical repair count is invalid")


def run_probe_commit_policy(
    context: ActionContext,
    *,
    global_total_fes: int,
    branch_probe_fes: int,
    decision_horizon_fes: int,
    exploration_floor_fes: int,
    min_relative_margin: float = 0.05,
    min_leader_stability: float = 0.20,
) -> ProbePolicyResult:
    """Probe all actions equally, commit one, and charge every objective FE."""

    if not isinstance(context, ActionContext):
        raise TypeError("probe policy requires ActionContext")
    if isinstance(global_total_fes, bool) or not isinstance(global_total_fes, int):
        raise ValueError("global_total_fes must be a positive integer")
    if global_total_fes != context.checkpoint.total_budget_fes:
        raise ValueError("global_total_fes must match the shared checkpoint budget")
    if (
        isinstance(decision_horizon_fes, bool)
        or not isinstance(decision_horizon_fes, int)
        or not 0 < decision_horizon_fes < branch_probe_fes
    ):
        raise ValueError("decision_horizon_fes must be inside the branch probe")
    phase2_fes = global_total_fes - context.checkpoint.phase1_fes
    aggregate_probe_fes = len(ACTION_NAMES) * branch_probe_fes
    if aggregate_probe_fes > phase2_fes:
        raise ValueError("all branch probes must fit inside the global Phase-II budget")

    probe = CommonAnchorProbe(context, branch_budget_fes=branch_probe_fes)
    probe.advance(branch_probe_fes)
    probe_final_errors = tuple(
        (action, float(probe.states[action].context.ledger.best_error))
        for action in ACTION_NAMES
    )
    decision = decide_delayed_commit(
        probe.trajectories,
        horizon_index=decision_horizon_fes,
        observed_fes=branch_probe_fes,
        exploration_floor_fes=exploration_floor_fes,
        min_relative_margin=min_relative_margin,
        min_leader_stability=min_leader_stability,
    )
    if decision.action_name is None:
        selected_action = probe.best_action()
        commit_reason = f"probe_cap_{decision.reason}"
    else:
        selected_action = decision.action_name
        commit_reason = decision.reason

    selected_state = probe.commit(selected_action)
    continuation_fes = phase2_fes - aggregate_probe_fes
    if continuation_fes:
        selected_state.step(continuation_fes)
    selected_snapshot = selected_state.snapshot()

    states = probe.states
    archive_source_action = min(
        ACTION_NAMES,
        key=lambda action: (states[action].context.ledger.best_error, action),
    )
    archive_ledger = states[archive_source_action].context.ledger
    aggregate_fes = context.checkpoint.phase1_fes + sum(
        state.consumed_fes for state in states.values()
    )
    return ProbePolicyResult(
        selected_action=selected_action,
        archive_source_action=archive_source_action,
        commit_reason=commit_reason,
        checkpoint_hash=context.checkpoint.checkpoint_hash,
        global_total_fes=global_total_fes,
        action_schedule_total_fes=context.checkpoint.total_budget_fes,
        selected_ledger_fes=selected_state.context.ledger.count,
        phase1_fes=context.checkpoint.phase1_fes,
        branch_probe_fes=branch_probe_fes,
        continuation_fes=continuation_fes,
        aggregate_fes=aggregate_fes,
        selected_action_fes=selected_state.consumed_fes,
        final_error=float(archive_ledger.best_error),
        incumbent=tuple(float(value) for value in archive_ledger.best_x),
        probe_final_errors=probe_final_errors,
        decision=decision,
        route=selected_state.route,
        optimizer_package=selected_state.optimizer_package,
        optimizer_version=selected_state.optimizer_version,
        selected_state_hash=selected_snapshot.snapshot_hash,
        numerical_repair_count=selected_state.numerical_repair_count,
    )


__all__ = ["ProbePolicyResult", "run_probe_commit_policy"]

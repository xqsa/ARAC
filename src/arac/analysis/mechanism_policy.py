"""Deterministic Phase-I mechanism baseline with no fitted parameters."""

from __future__ import annotations

from dataclasses import dataclass

from arac.actions.registry import ActionRegistry
from arac.core import AracCoreDecision, select_core_action
from arac.runtime.contracts import ActionContext, ActionResult, PhaseCheckpoint


MechanismDecision = AracCoreDecision


@dataclass(frozen=True)
class MechanismBaselineResult:
    decision: MechanismDecision
    action_result: ActionResult
    numerical_repair_count: int

    def __post_init__(self) -> None:
        if self.decision.action_name != self.action_result.action_name:
            raise ValueError("mechanism decision and action result drifted")
        if self.numerical_repair_count < 0:
            raise ValueError("mechanism numerical repair count is invalid")


def select_mechanism_action(checkpoint: PhaseCheckpoint) -> MechanismDecision:
    """Compatibility name for the active ARAC-Core selection rule."""

    return select_core_action(checkpoint)


def run_mechanism_baseline(context: ActionContext) -> MechanismBaselineResult:
    """Select once from Phase-I evidence and run that action to the checkpoint budget."""

    if not isinstance(context, ActionContext):
        raise TypeError("mechanism baseline requires ActionContext")
    decision = select_mechanism_action(context.checkpoint)
    action_context = ActionContext(
        decision.action_name,
        context.checkpoint,
        context.problem,
        context.ledger,
        context.action_seed,
    )
    state = ActionRegistry().initialize(action_context)
    state.step(state.total_fes - state.context.ledger.count)
    return MechanismBaselineResult(
        decision=decision,
        action_result=state.result(),
        numerical_repair_count=state.numerical_repair_count,
    )


__all__ = [
    "MechanismBaselineResult",
    "MechanismDecision",
    "run_mechanism_baseline",
    "select_mechanism_action",
]

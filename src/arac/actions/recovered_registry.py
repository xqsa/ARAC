"""Explicit registry for the verified recovered terminal actions."""

from __future__ import annotations

from arac.actions.ctp import CtpExecutor
from arac.actions.gcb import GcbExecutor
from arac.actions.recovered import (
    RecoveredAorExecutor,
    RecoveredHistoricalSmpExecutor,
)
from arac.runtime.contracts import ACTION_NAMES, ActionContext, ActionResult


class RecoveredActionRegistry:
    """Dispatch the recovered four-action set under one execution profile."""

    def __init__(self) -> None:
        executors = (
            CtpExecutor(),
            RecoveredHistoricalSmpExecutor(),
            GcbExecutor(),
            RecoveredAorExecutor(),
        )
        self._executors = {executor.name: executor for executor in executors}
        if set(self._executors) != set(ACTION_NAMES):
            raise RuntimeError("recovered registry does not match the frozen action set")

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTION_NAMES

    @property
    def allow_out_of_bounds(self) -> bool:
        """Permit recovered SMP offspring before public-bound clipping."""

        return True

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext):
            raise TypeError("context must be ActionContext")
        if context.ledger.allow_out_of_bounds is not self.allow_out_of_bounds:
            raise ValueError("ledger does not match the recovered execution profile")
        return self._executors[context.action_name].execute(context)


__all__ = ["RecoveredActionRegistry"]

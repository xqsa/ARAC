"""Single-dispatch registry for the four ARAC actions."""

from __future__ import annotations

from arac.actions.aor import AorExecutor
from arac.actions.ctp import CtpExecutor
from arac.actions.gcb import GcbExecutor
from arac.actions.smp import SmpExecutor
from arac.runtime.contracts import ACTION_NAMES, ActionContext, ActionResult


class ActionRegistry:
    def __init__(self) -> None:
        executors = (CtpExecutor(), SmpExecutor(), GcbExecutor(), AorExecutor())
        self._executors = {executor.name: executor for executor in executors}
        if set(self._executors) != set(ACTION_NAMES):
            raise RuntimeError("action registry does not match the frozen action set")

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTION_NAMES

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext):
            raise TypeError("context must be ActionContext")
        return self._executors[context.action_name].execute(context)


__all__ = ["ActionRegistry"]

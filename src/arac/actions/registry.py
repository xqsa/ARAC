"""Single-dispatch registry for the four ARAC actions."""

from __future__ import annotations

from arac.actions.aor import AorExecutor
from arac.actions.ctp import CtpExecutor
from arac.actions.gcb import GcbExecutor
from arac.actions.smp import SmpExecutor
from arac.actions.phase2_v2 import Phase2V2State
from arac.runtime.contracts import ACTION_NAMES, ActionContext, ActionResult, Phase2Snapshot


class ActionRegistry:
    def __init__(self) -> None:
        executors = (CtpExecutor(), SmpExecutor(), GcbExecutor(), AorExecutor())
        self._executors = {executor.name: executor for executor in executors}
        if set(self._executors) != set(ACTION_NAMES):
            raise RuntimeError("action registry does not match the frozen action set")

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTION_NAMES

    @property
    def allow_out_of_bounds(self) -> bool:
        """Keep the existing production execution profile bounded."""

        return False

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext):
            raise TypeError("context must be ActionContext")
        return self._executors[context.action_name].execute(context)

    def initialize(self, context: ActionContext) -> Phase2V2State:
        """Create a resumable Phase-II v2 state at the Phase-I boundary."""

        if not isinstance(context, ActionContext):
            raise TypeError("context must be ActionContext")
        return self._executors[context.action_name].initialize(context)

    def resume(self, context: ActionContext, snapshot: Phase2Snapshot) -> Phase2V2State:
        """Restore one action state after validating all public bindings."""

        if not isinstance(context, ActionContext):
            raise TypeError("context must be ActionContext")
        return self._executors[context.action_name].resume(context, snapshot)

    def execute_v2(self, context: ActionContext, *, step_fes: int | None = None) -> ActionResult:
        """Run the v2 state machine to completion, optionally in fixed-size steps."""

        state = self.initialize(context)
        while not state.complete:
            budget = state.total_fes - state.context.ledger.count
            if step_fes is not None:
                if isinstance(step_fes, bool) or not isinstance(step_fes, int) or step_fes <= 0:
                    raise ValueError("step_fes must be a positive integer")
                budget = min(budget, step_fes)
            state.step(budget)
        return state.result()


__all__ = ["ActionRegistry"]

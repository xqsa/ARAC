"""State-memory persistence (SMP) action."""

from __future__ import annotations

from arac.actions._execution import run_full_space, run_persistent_blocks, terminal_result
from arac.runtime.contracts import ActionContext, ActionResult


SMP_ACTION = "smp"


class SmpExecutor:
    """Persist one upstream block-CMA state per evidence block across all sweeps."""

    name = SMP_ACTION

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("SMP requires an SMP ActionContext")
        persistent_fes = run_persistent_blocks(
            context,
            requested_fes=context.ledger.remaining,
        )
        if context.ledger.remaining:
            run_full_space(context, algorithm="sepcmaes", namespace="smp-terminal")
        return terminal_result(
            context,
            route=f"persistent_block_cma_{persistent_fes}",
        )


__all__ = ["SMP_ACTION", "SmpExecutor"]

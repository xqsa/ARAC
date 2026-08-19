"""Coverage-to-polish (CTP) action."""

from __future__ import annotations

from arac.actions._execution import (
    run_full_space,
    run_persistent_blocks,
    run_sequential_blocks,
    terminal_result,
)
from arac.runtime.contracts import ActionContext, ActionResult


CTP_ACTION = "ctp"


class CtpExecutor:
    """Cover every evidence block, then polish the shared archive with MMES."""

    name = CTP_ACTION

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("CTP requires a CTP ActionContext")
        coverage_sweeps = 4 if context.checkpoint.overlap_relation_count == 0 else 2
        coverage_fes = run_persistent_blocks(
            context,
            requested_fes=context.ledger.remaining,
            sweep_limit=coverage_sweeps,
        )
        if context.checkpoint.overlap_relation_count == 0:
            polish_fes = run_sequential_blocks(
                context,
                requested_fes=context.ledger.remaining,
            )
            if context.ledger.remaining:
                run_full_space(context, algorithm="mmes", namespace="ctp-terminal")
            route = f"coverage_{coverage_fes}_then_sequential_block_polish_{polish_fes}"
        else:
            run_full_space(context, algorithm="mmes", namespace="ctp-polish")
            route = f"coverage_{coverage_fes}_then_mmes_polish"
        return terminal_result(
            context,
            route=route,
        )


__all__ = ["CTP_ACTION", "CtpExecutor"]

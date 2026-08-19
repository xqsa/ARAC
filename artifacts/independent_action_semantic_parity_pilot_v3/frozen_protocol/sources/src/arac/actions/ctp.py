"""Coverage-to-polish (CTP) action."""

from __future__ import annotations

from arac.actions._execution import (
    BLOCK_POPULATION_SIZE,
    run_full_space,
    run_persistent_blocks,
    run_sequential_blocks,
    terminal_result,
)
from arac.actions.phase2_v2 import CtpPhase2State
from arac.runtime.contracts import ActionContext, ActionResult
from arac.runtime.contracts import Phase2Snapshot


CTP_ACTION = "ctp"
_COVERAGE_FRACTION = 0.20


def _relation_cover(context: ActionContext) -> tuple[tuple[int, ...], ...]:
    """Build an evidence-derived overlapping cover for relation-aware polish."""

    base_blocks = tuple(tuple(block) for block in context.checkpoint.blocks)
    relation_blocks = []
    for relation in sorted(
        context.checkpoint.relations,
        key=lambda item: (-item.strength * (1.0 + item.disagreement), item.left_block, item.right_block),
    ):
        merged = tuple(
            sorted(
                set(base_blocks[relation.left_block])
                | set(base_blocks[relation.right_block])
            )
        )
        if merged not in relation_blocks:
            relation_blocks.append(merged)
    return base_blocks + tuple(relation_blocks)


class CtpExecutor:
    """Cover every evidence block, then polish the shared archive by block."""

    name = CTP_ACTION

    def initialize(self, context: ActionContext) -> CtpPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("CTP requires a CTP ActionContext")
        return CtpPhase2State(context)

    def resume(self, context: ActionContext, snapshot: Phase2Snapshot) -> CtpPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("CTP requires a CTP ActionContext")
        return CtpPhase2State.restore(context, snapshot)

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("CTP requires a CTP ActionContext")
        sweep_fes = len(context.checkpoint.blocks) * BLOCK_POPULATION_SIZE
        coverage_budget = min(
            context.ledger.remaining,
            max(sweep_fes, int(context.ledger.remaining * _COVERAGE_FRACTION)),
        )
        coverage_fes = run_persistent_blocks(
            context,
            requested_fes=coverage_budget,
        )
        polish_blocks = (
            context.checkpoint.blocks
            if context.checkpoint.overlap_relation_count == 0
            else _relation_cover(context)
        )
        polish_fes = run_sequential_blocks(
            context,
            requested_fes=context.ledger.remaining,
            blocks=polish_blocks,
        )
        if context.ledger.remaining:
            run_full_space(context, algorithm="mmes", namespace="ctp-terminal")
        route = (
            f"coverage_{coverage_fes}_then_sequential_block_polish_{polish_fes}"
            if context.checkpoint.overlap_relation_count == 0
            else f"coverage_{coverage_fes}_then_relation_cover_polish_{polish_fes}"
        )
        return terminal_result(
            context,
            route=route,
        )


__all__ = ["CTP_ACTION", "CtpExecutor"]

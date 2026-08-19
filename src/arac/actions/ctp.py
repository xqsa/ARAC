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
_POSITIVE_RELATION_MMES_TAIL_FRACTION = 0.20


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
        available = context.ledger.remaining
        sweep_fes = len(context.checkpoint.blocks) * BLOCK_POPULATION_SIZE
        positive_relations = context.checkpoint.overlap_relation_count > 0
        tail_reserve = (
            int(available * _POSITIVE_RELATION_MMES_TAIL_FRACTION)
            if positive_relations
            else 0
        )
        coverage_cap = max(0, available - tail_reserve)
        coverage_budget = min(
            coverage_cap,
            max(sweep_fes, int(coverage_cap * _COVERAGE_FRACTION)),
        )
        coverage_fes = run_persistent_blocks(
            context,
            requested_fes=coverage_budget,
        )
        polish_blocks = context.checkpoint.blocks if not positive_relations else _relation_cover(context)
        polish_budget = (
            context.ledger.remaining
            if not positive_relations
            else max(0, context.ledger.remaining - tail_reserve)
        )
        polish_fes = run_sequential_blocks(
            context,
            requested_fes=polish_budget,
            blocks=polish_blocks,
        )
        tail_fes = 0
        if positive_relations and context.ledger.remaining:
            tail_fes = run_full_space(
                context,
                algorithm="mmes",
                namespace="ctp-relation-mmes-tail",
            ).consumed_fes
        elif context.ledger.remaining:
            tail_fes = run_full_space(
                context,
                algorithm="mmes",
                namespace="ctp-terminal",
            ).consumed_fes
        route = f"coverage_{coverage_fes}_then_"
        route += "relation_cover_polish_" if positive_relations else "sequential_block_polish_"
        route += f"{polish_fes}"
        if positive_relations:
            route += f"_then_mmes_tail_{tail_fes}"
        return terminal_result(
            context,
            route=route,
        )


__all__ = ["CTP_ACTION", "CtpExecutor"]

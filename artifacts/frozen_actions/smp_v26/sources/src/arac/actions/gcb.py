"""Graph-conditioned balancing (GCB) action."""

from __future__ import annotations

from arac.actions._execution import (
    run_cold_start_block_sweeps,
    run_full_space,
    terminal_result,
)
from arac.runtime.contracts import ActionContext, ActionResult


GCB_ACTION = "gcb"


class GcbExecutor:
    """Condition coordination only on the observed Phase-I relation graph."""

    name = GCB_ACTION

    @staticmethod
    def _block_order(context: ActionContext) -> tuple[int, ...]:
        scores = [0.0] * len(context.checkpoint.blocks)
        for relation in context.checkpoint.relations:
            score = relation.strength * (1.0 + relation.disagreement)
            scores[relation.left_block] += score
            scores[relation.right_block] += score
        return tuple(sorted(range(len(scores)), key=lambda index: (-scores[index], index)))

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("GCB requires a GCB ActionContext")
        relation_mode = (
            "zero_relation"
            if context.checkpoint.overlap_relation_count == 0
            else "positive_relation_graph"
        )
        block_order = (
            self._block_order(context)
            if context.checkpoint.overlap_relation_count > 0
            else None
        )
        warmup_fes, warmup_sweeps = run_cold_start_block_sweeps(
            context,
            requested_fes=context.ledger.remaining,
            sweep_limit=3,
            block_order=block_order,
            namespace="gcb-warmup",
        )
        coordination_budget = min(
            context.ledger.remaining,
            warmup_sweeps[-1] if warmup_sweeps else context.ledger.remaining,
        )
        if coordination_budget:
            run_full_space(
                context,
                algorithm="sepcmaes",
                budget_fes=coordination_budget,
                namespace="gcb-global-coordination",
            )
        continuation_fes, continuation_sweeps = run_cold_start_block_sweeps(
            context,
            requested_fes=context.ledger.remaining,
            block_order=block_order,
            namespace="gcb-continuation",
        )
        tail_fes = context.ledger.remaining
        if tail_fes:
            run_full_space(
                context,
                algorithm="sepcmaes",
                namespace="gcb-terminal-alignment",
            )
        route = (
            f"{relation_mode}_cold_warmup_{warmup_fes}_sweeps_{len(warmup_sweeps)}_"
            f"coordination_{coordination_budget}_cold_continuation_{continuation_fes}_"
            f"sweeps_{len(continuation_sweeps)}_tail_{tail_fes}"
        )
        return terminal_result(context, route=route)


__all__ = ["GCB_ACTION", "GcbExecutor"]

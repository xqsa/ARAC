"""Graph-conditioned balancing (GCB) action."""

from __future__ import annotations

from arac.actions._execution import run_full_space, run_persistent_blocks, terminal_result
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
        if context.checkpoint.overlap_relation_count == 0:
            run_full_space(context, algorithm="sepcmaes", namespace="gcb-global")
            route = "zero_relation_global_coordination"
        else:
            requested = max(240, context.checkpoint.remaining_fes // 10)
            graph_fes = run_persistent_blocks(
                context,
                requested_fes=requested,
                block_order=self._block_order(context),
            )
            if context.ledger.remaining:
                run_full_space(context, algorithm="sepcmaes", namespace="gcb-bridge")
            route = f"positive_relation_graph_{graph_fes}_then_global_coordination"
        return terminal_result(context, route=route)


__all__ = ["GCB_ACTION", "GcbExecutor"]

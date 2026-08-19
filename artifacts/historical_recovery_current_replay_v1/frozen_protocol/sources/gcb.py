"""Graph-conditioned balancing (GCB) action."""

from __future__ import annotations

from arac.actions._execution import (
    run_cold_start_block_sweeps,
    run_full_space,
    terminal_result,
)
from arac.evidence.mechanism_features import summarize_relation_topology
from arac.runtime.contracts import ActionContext, ActionResult, PhaseCheckpoint
from arac.runtime.contracts import Phase2Snapshot
from arac.actions.phase2_v2 import GcbPhase2State


GCB_ACTION = "gcb"
_WARMUP_FRACTION = 0.20
_MIN_COORDINATION_FRACTION = 0.10
_MAX_COORDINATION_FRACTION = 0.40


def _coordination_fraction(checkpoint: PhaseCheckpoint) -> float:
    if checkpoint.overlap_relation_count == 0:
        return 0.0
    _, entropy, largest_component = summarize_relation_topology(
        checkpoint.blocks,
        checkpoint.relations,
    )
    return min(
        _MAX_COORDINATION_FRACTION,
        _MIN_COORDINATION_FRACTION
        + (_MAX_COORDINATION_FRACTION - _MIN_COORDINATION_FRACTION)
        * entropy
        * largest_component,
    )


class GcbExecutor:
    """Condition coordination only on the observed Phase-I relation graph."""

    name = GCB_ACTION

    def initialize(self, context: ActionContext) -> GcbPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("GCB requires a GCB ActionContext")
        return GcbPhase2State(context)

    def resume(self, context: ActionContext, snapshot: Phase2Snapshot) -> GcbPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("GCB requires a GCB ActionContext")
        return GcbPhase2State.restore(context, snapshot)

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
        available = context.ledger.remaining
        warmup_budget = int(available * _WARMUP_FRACTION)
        warmup_fes, warmup_sweeps = run_cold_start_block_sweeps(
            context,
            requested_fes=warmup_budget,
            sweep_limit=3,
            block_order=block_order,
            namespace="gcb-warmup",
        )
        coordination_budget = min(
            context.ledger.remaining,
            int(available * _coordination_fraction(context.checkpoint)),
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

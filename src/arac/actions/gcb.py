"""Graph-conditioned balancing (GCB) action."""

from __future__ import annotations

import math

from arac.actions._execution import (
    derived_seed,
    run_cold_start_block_sweeps,
    run_full_space,
    terminal_result,
)
from arac.actions.phase2_v2 import GcbPhase2State
from arac.runtime.contracts import ActionContext, ActionResult, Phase2Snapshot


GCB_ACTION = "gcb"
_SOURCE_WINDOW_FRACTION = 0.08
_SOURCE_SWEEP_COUNT = 3
_NATIVE_WINDOW_COUNT = 3


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
        return self.execute_schedule(context)

    def execute_schedule(
        self,
        context: ActionContext,
        *,
        event_trace: list[dict[str, object]] | None = None,
    ) -> ActionResult:
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
        blocks = context.checkpoint.blocks
        minimum_window_fes = sum(
            4 + 3 * math.ceil(math.log(len(block))) for block in blocks
        )
        available = context.ledger.remaining
        minimum_contract_fes = minimum_window_fes * (
            1 + _SOURCE_SWEEP_COUNT + _NATIVE_WINDOW_COUNT
        )
        if available < minimum_contract_fes:
            compact_fes = context.ledger.remaining
            run_full_space(
                context,
                algorithm="sepcmaes",
                namespace=f"gcb-compact-{context.checkpoint.checkpoint_hash}",
            )
            return terminal_result(
                context,
                route=f"{relation_mode}_compact_coordination_{compact_fes}",
            )

        source_window_budget = max(
            minimum_window_fes,
            int(available * _SOURCE_WINDOW_FRACTION),
        )
        source_window_budget = min(
            source_window_budget,
            available // (1 + _SOURCE_SWEEP_COUNT + _NATIVE_WINDOW_COUNT),
        )
        seed_namespace = f"gcb-native-{context.checkpoint.checkpoint_hash}"
        def seed_factory(stage_index: int) -> int:
            return derived_seed(context, seed_namespace, stage_index)
        source_fes = 0
        source_sweeps: list[int] = []
        for source_index in range(_SOURCE_SWEEP_COUNT):
            consumed, sweeps = run_cold_start_block_sweeps(
                context,
                requested_fes=min(source_window_budget, context.ledger.remaining),
                sweep_limit=1,
                block_order=block_order,
                namespace="gcb-source",
                seed_factory=seed_factory,
                start_sweep_index=source_index,
                event_trace=event_trace,
            )
            if len(sweeps) != 1 or consumed < minimum_window_fes:
                raise RuntimeError("GCB source sweep did not cover every block")
            source_fes += consumed
            source_sweeps.extend(sweeps)
        coordination_budget = min(
            context.ledger.remaining,
            source_sweeps[-1],
        )
        if coordination_budget:
            coordination_start = context.ledger.count
            run_full_space(
                context,
                algorithm="sepcmaes",
                budget_fes=coordination_budget,
                namespace=f"gcb-global-coordination-{context.checkpoint.checkpoint_hash}",
            )
            if event_trace is not None:
                event_trace.append(
                    {
                        "event": "full_space_coordination",
                        "trigger": (
                            "phase_boundary"
                            if context.checkpoint.overlap_relation_count == 0
                            else "relation_dispatch"
                        ),
                        "start_fes": coordination_start,
                        "requested_fes": coordination_budget,
                        "actual_fes": context.ledger.count - coordination_start,
                        "end_fes": context.ledger.count,
                    }
                )
        native_fes = 0
        native_sweeps: list[int] = []
        for window_offset in range(_NATIVE_WINDOW_COUNT):
            consumed, sweeps = run_cold_start_block_sweeps(
                context,
                requested_fes=min(source_window_budget, context.ledger.remaining),
                sweep_limit=1,
                block_order=block_order,
                namespace="gcb-native",
                seed_factory=seed_factory,
                start_sweep_index=_SOURCE_SWEEP_COUNT + window_offset,
                event_trace=event_trace,
            )
            if len(sweeps) != 1 or consumed < minimum_window_fes:
                raise RuntimeError("GCB native window did not cover every block")
            native_fes += consumed
            native_sweeps.extend(sweeps)
        continuation_fes, continuation_sweeps = run_cold_start_block_sweeps(
            context,
            requested_fes=context.ledger.remaining,
            block_order=block_order,
            namespace="gcb-native",
            seed_factory=seed_factory,
            start_sweep_index=_SOURCE_SWEEP_COUNT + _NATIVE_WINDOW_COUNT,
            event_trace=event_trace,
        )
        native_fes += continuation_fes
        native_sweeps.extend(continuation_sweeps)
        tail_fes = context.ledger.remaining
        if tail_fes:
            run_full_space(
                context,
                algorithm="sepcmaes",
                namespace=f"gcb-terminal-alignment-{context.checkpoint.checkpoint_hash}",
            )
        route = (
            f"{relation_mode}_source_{source_fes}_sweeps_{len(source_sweeps)}_"
            f"coordination_{coordination_budget}_cold_native_{native_fes}_"
            f"windows_{len(native_sweeps)}_tail_{tail_fes}"
        )
        return terminal_result(context, route=route)


__all__ = ["GCB_ACTION", "GcbExecutor"]

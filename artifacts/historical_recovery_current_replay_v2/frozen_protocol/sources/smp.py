"""State-memory persistence (SMP) action."""

from __future__ import annotations

from arac.actions._execution import (
    STATE_RESCUE_FRACTION,
    STATE_RESCUE_MIN_FES,
    run_full_space,
    run_stalled_block_rescue,
    run_stateful_block_visits,
    run_stateful_block_visits_with_sessions,
    run_zero_relation_hybrid_rescue,
    terminal_result,
)
from arac.actions.phase2_v2 import SmpPhase2State
from arac.runtime.contracts import ActionContext, ActionResult, Phase2Snapshot


SMP_ACTION = "smp"
_POSITIVE_RELATION_GLOBAL_POLISH_FRACTION = 0.50


class SmpExecutor:
    """Persist one upstream block-CMA state per evidence block across all sweeps."""

    name = SMP_ACTION

    def initialize(self, context: ActionContext) -> SmpPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("SMP requires an SMP ActionContext")
        return SmpPhase2State(context)

    def resume(self, context: ActionContext, snapshot: Phase2Snapshot) -> SmpPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("SMP requires an SMP ActionContext")
        return SmpPhase2State.restore(context, snapshot)

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("SMP requires an SMP ActionContext")
        available = context.ledger.remaining
        global_polish_budget = (
            int(available * _POSITIVE_RELATION_GLOBAL_POLISH_FRACTION)
            if context.checkpoint.overlap_relation_count > 0
            else 0
        )
        state_budget = available - global_polish_budget
        rescue_budget = (
            int(state_budget * STATE_RESCUE_FRACTION) if state_budget >= STATE_RESCUE_MIN_FES else 0
        )
        if context.checkpoint.overlap_relation_count > 0:
            stateful_fes, visit_count, restart_count = run_stateful_block_visits(
                context,
                requested_fes=state_budget - rescue_budget,
            )
            rescue_fes, probe_fes, rescue_visits = run_stalled_block_rescue(
                context,
                requested_fes=min(rescue_budget, context.ledger.remaining),
            )
            rescue_route = f"rescue_visits_{rescue_visits}"
        else:
            (
                stateful_fes,
                visit_count,
                restart_count,
                sessions,
            ) = run_stateful_block_visits_with_sessions(
                context,
                requested_fes=state_budget - rescue_budget,
            )
            (
                rescue_fes,
                probe_fes,
                coverage_fes,
                cold_rescue_visits,
                persistent_rescue_visits,
            ) = run_zero_relation_hybrid_rescue(
                context,
                requested_fes=min(rescue_budget, context.ledger.remaining),
                sessions=sessions,
            )
            rescue_route = (
                f"coverage_{coverage_fes}_cold_rescue_visits_{cold_rescue_visits}_"
                f"persistent_rescue_visits_{persistent_rescue_visits}"
            )
        global_polish_fes = 0
        if global_polish_budget and context.ledger.remaining:
            global_polish_fes = run_full_space(
                context,
                algorithm="mmes",
                budget_fes=min(global_polish_budget, context.ledger.remaining),
                namespace="smp-global-polish",
            ).consumed_fes
        if context.ledger.remaining:
            run_full_space(context, algorithm="sepcmaes", namespace="smp-terminal")
        return terminal_result(
            context,
            route=(
                f"stateful_block_visits_{stateful_fes}_visits_{visit_count}_"
                f"stale_restarts_{restart_count}_rescue_{rescue_fes}_"
                f"probes_{probe_fes}_{rescue_route}_"
                f"global_polish_{global_polish_fes}"
            ),
        )


__all__ = ["SMP_ACTION", "SmpExecutor"]

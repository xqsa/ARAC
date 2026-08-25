"""Historically recovered terminal action mechanisms."""

from __future__ import annotations

import numpy as np

from arac.actions._execution import (
    DEFAULT_SIGMA,
    FULL_SPACE_POPULATION_SIZE,
    run_stateful_block_visits_with_sessions,
    terminal_result,
)
from arac.actions.phase2_v2 import RecoveredAorPhase2State, RecoveredSmpPhase2State
from arac.actions.smp import SmpExecutor
from arac.runtime.contracts import ActionContext, ActionResult, Phase2Snapshot
from arac.runtime.optimizers import PypopOptimizerPort


class RecoveredAorExecutor:
    """Run the recovered fresh full-space Sep-CMA route."""

    name = "aor"

    def initialize(self, context: ActionContext) -> RecoveredAorPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("recovered AOR requires an AOR ActionContext")
        return RecoveredAorPhase2State(context)

    def resume(self, context: ActionContext, snapshot: Phase2Snapshot) -> RecoveredAorPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("recovered AOR requires an AOR ActionContext")
        return RecoveredAorPhase2State.restore(context, snapshot)

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("recovered AOR requires an AOR ActionContext")
        budget = context.ledger.remaining
        PypopOptimizerPort().run(
            "sepcmaes",
            problem=context.problem,
            ledger=context.ledger,
            initial_mean=np.zeros(context.problem.dimension),
            sigma=DEFAULT_SIGMA,
            seed=context.action_seed,
            budget_fes=budget,
            population_size=FULL_SPACE_POPULATION_SIZE,
            restart=False,
        )
        return terminal_result(
            context,
            route=f"recovered_fresh_zero_mean_sepcmaes_{budget}",
        )


class RecoveredSmpExecutor:
    """Run the recovered identity-blind state-memory lifecycle."""

    name = "smp"

    def initialize(self, context: ActionContext) -> RecoveredSmpPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("recovered SMP requires an SMP ActionContext")
        if not context.ledger.allow_out_of_bounds:
            raise ValueError("recovered SMP requires the explicit unbounded-offspring profile")
        return RecoveredSmpPhase2State(context)

    def resume(self, context: ActionContext, snapshot: Phase2Snapshot) -> RecoveredSmpPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("recovered SMP requires an SMP ActionContext")
        if not context.ledger.allow_out_of_bounds:
            raise ValueError("recovered SMP requires the explicit unbounded-offspring profile")
        return RecoveredSmpPhase2State.restore(context, snapshot)

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("recovered SMP requires an SMP ActionContext")
        if not context.ledger.allow_out_of_bounds:
            raise ValueError("recovered SMP requires the explicit unbounded-offspring profile")
        requested_fes = context.ledger.remaining
        consumed, visits, resets, _ = run_stateful_block_visits_with_sessions(
            context,
            requested_fes=requested_fes,
            clip_offspring=False,
            precheck_incumbent=True,
            strict_material_gain=True,
        )
        noop_fes = 0
        while context.ledger.remaining:
            context.ledger.evaluate(context.ledger.best_x)
            noop_fes += 1
        return terminal_result(
            context,
            route=(
                f"recovered_stateful_visits_{consumed}_visits_{visits}_"
                f"stale_resets_{resets}_noop_{noop_fes}"
            ),
        )


class RecoveredHistoricalSmpExecutor:
    """Select the recovered SMP lifecycle by relation topology.

    ``RecoveredSmpExecutor`` remains the resumable v2 state-machine wrapper;
    this adapter is used by the recovered one-shot registry only.  Conflicting
    overlap uses the evidenced rescue, global-polish, and terminal-tail budget
    ownership, while zero-relation cases retain the recovered hybrid rescue
    route that was already validated on E1.
    """

    name = "smp"
    historical_lifecycle_profile = "historical_compatible_smp_v1_clip_offspring_true"
    zero_relation_lifecycle_profile = "zero_relation_recovered_smp_v1_clip_offspring_false"

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("recovered historical SMP requires an SMP ActionContext")
        if not context.ledger.allow_out_of_bounds:
            raise ValueError("recovered historical SMP requires the explicit unbounded-offspring profile")
        if context.checkpoint.overlap_relation_count == 0:
            result = RecoveredSmpExecutor().execute(context)
            lifecycle_profile = self.zero_relation_lifecycle_profile
        else:
            result = SmpExecutor().execute(context)
            lifecycle_profile = self.historical_lifecycle_profile
        return terminal_result(
            context,
            route=f"recovered_{lifecycle_profile}_{result.route}",
        )


__all__ = [
    "RecoveredAorExecutor",
    "RecoveredHistoricalSmpExecutor",
    "RecoveredSmpExecutor",
]

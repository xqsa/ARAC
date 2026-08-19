"""Adaptive optimizer routing (AOR) action."""

from __future__ import annotations

from collections.abc import Mapping

from arac.actions._execution import run_full_space, terminal_result
from arac.actions.phase2_v2 import AorPhase2State
from arac.runtime.contracts import ActionContext, ActionResult, Phase2Snapshot


AOR_ACTION = "aor"


def _optimizer_route(evidence: Mapping[str, float]) -> str:
    """Choose the frozen legacy full-space optimizer route."""

    center_scale = float(evidence["log10_center_error"])
    roughness = float(evidence["line_high_frequency_fraction_median"])
    return "sepcmaes" if center_scale < 10.0 or roughness < 0.3 else "mmes"


class AorExecutor:
    """Route one full-space optimizer from Phase-I numerical evidence."""

    name = AOR_ACTION

    def initialize(self, context: ActionContext) -> AorPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("AOR requires an AOR ActionContext")
        return AorPhase2State(context)

    def resume(self, context: ActionContext, snapshot: Phase2Snapshot) -> AorPhase2State:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("AOR requires an AOR ActionContext")
        return AorPhase2State.restore(context, snapshot)

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("AOR requires an AOR ActionContext")
        evidence = dict(
            zip(
                context.checkpoint.feature_names,
                context.checkpoint.feature_values,
                strict=True,
            )
        )
        algorithm = _optimizer_route(evidence)
        run_full_space(context, algorithm=algorithm, namespace=f"aor-{algorithm}")
        return terminal_result(context, route=f"evidence_routed_{algorithm}")


__all__ = ["AOR_ACTION", "AorExecutor"]

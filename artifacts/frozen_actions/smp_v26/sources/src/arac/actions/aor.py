"""Adaptive optimizer routing (AOR) action."""

from __future__ import annotations

from arac.actions._execution import run_full_space, terminal_result
from arac.runtime.contracts import ActionContext, ActionResult


AOR_ACTION = "aor"


class AorExecutor:
    """Route one full-space optimizer from Phase-I numerical evidence."""

    name = AOR_ACTION

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
        center_scale = evidence["log10_center_error"]
        roughness = evidence["line_high_frequency_fraction_median"]
        algorithm = "sepcmaes" if center_scale < 10.0 or roughness < 0.3 else "mmes"
        run_full_space(context, algorithm=algorithm, namespace=f"aor-{algorithm}")
        return terminal_result(context, route=f"evidence_routed_{algorithm}")


__all__ = ["AOR_ACTION", "AorExecutor"]

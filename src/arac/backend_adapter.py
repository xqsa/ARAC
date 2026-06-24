"""Backend intervention adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from .policy import ActionDecision


@dataclass(frozen=True)
class BackendSemanticsDiff:
    variable_owner_changed: bool = False
    relation_handling_changed: bool = False
    coordination_mode_changed: bool = False
    budget_allocation_changed: bool = False
    update_order_changed: bool = False
    acceptance_rule_changed: bool = False

    @property
    def changed(self) -> bool:
        return any(
            (
                self.variable_owner_changed,
                self.relation_handling_changed,
                self.coordination_mode_changed,
                self.budget_allocation_changed,
                self.update_order_changed,
                self.acceptance_rule_changed,
            )
        )


class BackendAdapter:
    """Interface for optimizer-consumed action semantics."""

    def apply(self, decision: ActionDecision) -> BackendSemanticsDiff:
        raise NotImplementedError


class NullBackendAdapter(BackendAdapter):
    """Safe placeholder backend.

    It never reports active semantics. Use this in schema tests and replace it
    before claiming runtime-connected action execution.
    """

    def apply(self, decision: ActionDecision) -> BackendSemanticsDiff:
        return BackendSemanticsDiff()


class ToyBackendAdapter(BackendAdapter):
    """Deterministic backend semantics for schema-smoke experiments.

    The adapter is intentionally small: it proves that an action label can be
    bound to optimizer-consumed semantics without importing the historical HCC
    execution surface.
    """

    def apply(self, decision: ActionDecision) -> BackendSemanticsDiff:
        match decision.action_name:
            case "isolate_conflicting_relation":
                return BackendSemanticsDiff(relation_handling_changed=True)
            case "protect_high_margin_group":
                return BackendSemanticsDiff(budget_allocation_changed=True)
            case "repair_shared_variable_binding":
                return BackendSemanticsDiff(variable_owner_changed=True)
            case "allow_beneficial_coordination":
                return BackendSemanticsDiff(coordination_mode_changed=True)
            case _:
                return BackendSemanticsDiff()

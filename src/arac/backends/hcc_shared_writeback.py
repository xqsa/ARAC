"""Reference-blind mapping from ARAC actions to HCC semantics."""

from __future__ import annotations

from arac.actions import ActionDecision, ActionFamily
from arac.execution import BackendSemanticsDiff

def hcc_backend_semantics_for(
    decision: ActionDecision,
    *,
    optimizer_consumed: bool,
) -> BackendSemanticsDiff:
    """Map clean ARAC actions onto HCC optimizer-consumed semantic surfaces."""

    if not optimizer_consumed:
        return BackendSemanticsDiff()
    if decision.action_family == ActionFamily.ISOLATE:
        return BackendSemanticsDiff(relation_handling_changed=True)
    if decision.action_family == ActionFamily.PROTECT:
        return BackendSemanticsDiff(budget_allocation_changed=True)
    if decision.action_family == ActionFamily.REASSIGN_REPAIR:
        return BackendSemanticsDiff(variable_owner_changed=True)
    if decision.action_family == ActionFamily.COORDINATE:
        return BackendSemanticsDiff(coordination_mode_changed=True)
    if decision.action_family == ActionFamily.TRAJECTORY:
        if decision.action_name in {"repair_protect_refine", "repair_protect_deep_refine"}:
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                budget_allocation_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name in {
            "repair_bipop_search_state_restart",
            "repair_phase_rescue_multistart",
        }:
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name == "cc_harm_guarded_sep_refresh":
            return BackendSemanticsDiff(
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name == "separable_cmaes_dispatch_action":
            return BackendSemanticsDiff(
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name == "arac_evidence_action_controller_v1":
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                relation_handling_changed=True,
                coordination_mode_changed=True,
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name == "arac_evidence_action_controller_v2":
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                relation_handling_changed=True,
                coordination_mode_changed=True,
            )
        if decision.action_name == "arac_evidence_action_controller_v3":
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                relation_handling_changed=True,
                coordination_mode_changed=True,
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name in {
            "arac_evidence_action_controller_v31",
            "arac_evidence_action_controller_v32",
            "arac_evidence_action_controller_v33",
            "arac_evidence_action_controller_v34",
        }:
            return BackendSemanticsDiff(
                variable_owner_changed=True,
                relation_handling_changed=True,
                coordination_mode_changed=True,
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        if decision.action_name in {"bipop_search_state_restart", "phase_rescue_multistart"}:
            return BackendSemanticsDiff(
                budget_allocation_changed=True,
                update_order_changed=True,
                acceptance_rule_changed=True,
            )
        return BackendSemanticsDiff(budget_allocation_changed=True)
    return BackendSemanticsDiff()

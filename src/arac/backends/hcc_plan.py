"""Pure HCC action-plan contracts and action bindings.

This module contains only auditable action metadata. It does not execute the
optimizer or read paper/historical result data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from arac.actions import ActionDecision
from arac.actions.controller_profiles import controller_action_effects

AOB_FUNCTION_NAMES = {
    "E": "elliptic",
    "S": "schwefel",
    "R": "rastrigin",
    "A": "ackley",
}

@dataclass(frozen=True)
class HccActionExecutionPlan:
    """Audit row describing whether an ARAC action reaches HCC runtime."""

    problem_id: str
    selected_action_name: str
    selected_action_family: str
    backend_effect_kind: str
    optimizer_consumed: bool
    optimizer_consumed_parameters: dict[str, object]
    execution_mode: str
    blocker_reason: str
    runtime_dispatch_allowed: bool

    def to_csv_row(self) -> dict[str, str]:
        return {
            "problem_id": self.problem_id,
            "selected_action_name": self.selected_action_name,
            "selected_action_family": self.selected_action_family,
            "backend_effect_kind": self.backend_effect_kind,
            "optimizer_consumed": "1" if self.optimizer_consumed else "0",
            "optimizer_consumed_parameters": _format_json_parameters(
                self.optimizer_consumed_parameters
            ),
            "execution_mode": self.execution_mode,
            "blocker_reason": self.blocker_reason,
            "runtime_dispatch_allowed": "1" if self.runtime_dispatch_allowed else "0",
        }

HCC_ACTION_EFFECTS = {
    "conservative_no_action": (
        "no_op_safe_fallback",
        {"backend": "repo_default_hcc_no_action"},
        "hcc_noop_baseline",
        True,
        "",
    ),
    "isolate_conflicting_relation": (
        "shared_variable_value_selection",
        {"runtime_hook": "overlap_value_selection_rule"},
        "hcc_relation_value_selection_consumed",
        True,
        "",
    ),
    "protect_high_margin_group": (
        "protect_resource_priority",
        {},
        "audit_only_not_executed",
        False,
        "no_hcc_runtime_consumer_yet",
    ),
    "budget_shift_mean_blend": (
        "optimizer_budget_and_mean_trajectory",
        {"runtime_hook": "budget_shift_mean_blend"},
        "hcc_trajectory_runtime_consumed",
        True,
        "",
    ),
    "bipop_search_state_restart": (
        "optimizer_search_state_restart",
        {"runtime_hook": "bipop_search_state_restart"},
        "hcc_search_state_runtime_consumed",
        True,
        "",
    ),
    "resume_phase_i_search_state": (
        "resumable_mmes_state_block",
        {
            "runtime_hook": "resume_phase_i_search_state",
            "backend": "saved_phase_i_mmes_state",
            "acceptance_rule": "strict_global_incumbent_improvement",
        },
        "hcc_stateful_search_action",
        True,
        "",
    ),
    "phase_rescue_multistart": (
        "optimizer_phase_rescue_multistart",
        {
            "runtime_hook": "phase_rescue_multistart",
            "acceptance_rule": "best_improving_candidate_only",
        },
        "hcc_phase_rescue_runtime_consumed",
        True,
        "",
    ),
    "repair_phase_rescue_multistart": (
        "repair_guided_phase_rescue_multistart",
        {
            "overlap_runtime_hook": "overlap_repair_rule",
            "search_state_runtime_hook": "phase_rescue_multistart",
            "acceptance_rule": "best_improving_candidate_only",
        },
        "hcc_repair_phase_rescue_runtime_consumed",
        True,
        "",
    ),
    "cc_harm_guarded_sep_refresh": (
        "cc_harm_guarded_sep_or_nda_refresh",
        {
            "runtime_hook": "cc_harm_guarded_sep_refresh",
            "guard": "phase_i_or_current_incumbent_no_harm",
            "refresh_backend": "full_space_mmes_nda_continuation",
            "acceptance_rule": "guarded_incumbent_improving_candidate_only",
        },
        "hcc_cc_harm_guarded_refresh_runtime_consumed",
        True,
        "",
    ),
    "separable_cmaes_dispatch_action": (
        "full_space_diagonal_separable_search_takeover",
        {
            "runtime_hook": "separable_cmaes_dispatch_action",
            "backend": "direct_separable_cmaes",
            "search_distribution": "diagonal_sigma_full_space",
            "acceptance_rule": "optimizer_best_so_far",
        },
        "hcc_direct_separable_cmaes_runtime_consumed",
        True,
        "",
    ),
    "arac_evidence_action_controller_v1": (
        "evidence_action_runtime_controller",
        {
            "relation_runtime_hook": "adaptive_v26_relation_dispatch",
            "overlap_runtime_hook": "evidence_triggered_overlap_action",
            "search_state_runtime_hooks": [
                "phase_rescue_multistart",
                "cc_harm_guarded_sep_refresh",
            ],
            "dispatch_boundary": "runtime_evidence_only",
        },
        "hcc_evidence_action_controller_runtime_consumed",
        True,
        "",
    ),
    "arac_evidence_action_controller_v2": (
        "evidence_action_runtime_controller_v2",
        {
            "relation_runtime_hook": "adaptive_v24_relation_dispatch",
            "overlap_runtime_hook": "relation_first_evidence_triggered_overlap_action",
            "search_state_runtime_hooks": [],
            "dispatch_boundary": "runtime_evidence_only",
        },
        "hcc_evidence_action_controller_v2_runtime_consumed",
        True,
        "",
    ),
    "arac_evidence_action_controller_v3": (
        "evidence_action_runtime_controller_v3",
        {
            "relation_runtime_hook": "controller_v3_relation_dispatch",
            "mode_selector": "early_runtime_overlap_relation_evidence",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": [
                "phase_rescue_multistart",
                "cc_harm_guarded_sep_refresh",
            ],
            "dispatch_boundary": "runtime_evidence_only",
        },
        "hcc_evidence_action_controller_v3_runtime_consumed",
        True,
        "",
    ),
    "arac_evidence_action_controller_v31": (
        "evidence_action_runtime_controller_v31",
        {
            "relation_runtime_hook": "controller_v31_guarded_relation_dispatch",
            "mode_selector": "early_runtime_overlap_relation_evidence_with_relation_first_lock",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": [
                "resume_phase_i_search_state",
            ],
            "guard": "stable_relation_first_no_harm_gate",
            "dispatch_boundary": "runtime_evidence_only",
        },
        "hcc_evidence_action_controller_v31_runtime_consumed",
        True,
        "",
    ),
    "arac_evidence_action_controller_v32": (
        "evidence_action_runtime_controller_v32",
        {
            "relation_runtime_hook": "controller_v31_guarded_relation_dispatch",
            "mode_selector": "early_runtime_overlap_relation_evidence_with_relation_first_lock",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": [
                "phase_rescue_multistart",
            ],
            "guard": "stable_relation_first_no_harm_gate",
            "dispatch_boundary": "runtime_evidence_only",
        },
        "hcc_evidence_action_controller_v32_runtime_consumed",
        True,
        "",
    ),
    "cross_sweep_cma_sigma_continuation": (
        "optimizer_search_scale_continuation",
        {
            "runtime_hook": "cross_sweep_cma_terminal_sigma_continuation",
            "state_scope": "current_run_phase_i_group",
        },
        "hcc_cma_sigma_continuation_runtime_consumed",
        True,
        "",
    ),
    "repair_bipop_search_state_restart": (
        "repair_guided_optimizer_search_state_restart",
        {
            "overlap_runtime_hook": "overlap_repair_rule",
            "search_state_runtime_hook": "bipop_search_state_restart",
        },
        "hcc_repair_bipop_runtime_consumed",
        True,
        "",
    ),
    "repair_protect_refine": (
        "repair_guided_local_refinement",
        {
            "overlap_runtime_hook": "overlap_repair_rule",
            "optimizer_runtime_hook": "protected_small_sigma_refine",
        },
        "hcc_repair_refine_runtime_consumed",
        True,
        "",
    ),
    "repair_protect_deep_refine": (
        "repair_guided_deep_local_refinement",
        {
            "overlap_runtime_hook": "overlap_repair_rule",
            "optimizer_runtime_hook": "protected_deep_sigma_refine",
        },
        "hcc_repair_deep_refine_runtime_consumed",
        True,
        "",
    ),
    "repair_shared_variable_binding": (
        "shared_variable_owner_rebinding",
        {"runtime_hook": "overlap_repair_rule"},
        "hcc_smoke_runtime_consumed",
        True,
        "",
    ),
    "allow_beneficial_coordination": (
        "coordination_mode_switch",
        {"runtime_hook": "overlap_clipped_consensus_blend"},
        "hcc_relation_runtime_consumed",
        True,
        "",
    ),
}
HCC_ACTION_EFFECTS.update(controller_action_effects())

def _problem_parts(problem_id: str) -> tuple[str, str, int]:
    problem = str(problem_id).strip().upper()
    if len(problem) != 2 or problem[0] not in AOB_FUNCTION_NAMES or not problem[1].isdigit():
        raise ValueError(f"unsupported AOB problem_id: {problem_id}")
    function_id = int(problem[1])
    if function_id < 1 or function_id > 6:
        raise ValueError(f"unsupported AOB function id: {problem_id}")
    return problem, AOB_FUNCTION_NAMES[problem[0]], function_id

def _format_json_parameters(parameters: dict[str, object]) -> str:
    if not parameters:
        return ""
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def build_hcc_action_execution_plan(
    problem_id: str,
    decision: ActionDecision,
) -> HccActionExecutionPlan:
    """Describe whether an ARAC action is optimizer-consumed by HCC today."""

    effect = HCC_ACTION_EFFECTS.get(decision.action_name)
    if effect is None:
        backend_effect_kind = "unknown_action"
        parameters: dict[str, object] = {}
        execution_mode = "audit_only_not_executed"
        optimizer_consumed = False
        blocker = "unknown_hcc_action_binding"
    else:
        backend_effect_kind, parameters, execution_mode, optimizer_consumed, blocker = effect

    return HccActionExecutionPlan(
        problem_id=_problem_parts(problem_id)[0],
        selected_action_name=decision.action_name,
        selected_action_family=decision.action_family.value,
        backend_effect_kind=backend_effect_kind,
        optimizer_consumed=bool(optimizer_consumed),
        optimizer_consumed_parameters=dict(parameters),
        execution_mode=execution_mode,
        blocker_reason=blocker,
        runtime_dispatch_allowed=bool(optimizer_consumed and not blocker),
    )

from __future__ import annotations

from arac.actions import ActionFamily
from arac.backends.hcc import build_hcc_action_execution_plan
from arac.actions import ActionDecision


def test_hcc_action_execution_plan_marks_no_action_as_optimizer_consumed_noop() -> None:
    decision = ActionDecision(
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "fallback",
        "test",
        0.0,
    )

    plan = build_hcc_action_execution_plan("E1", decision)

    assert plan.problem_id == "E1"
    assert plan.selected_action_name == "conservative_no_action"
    assert plan.backend_effect_kind == "no_op_safe_fallback"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {"backend": "repo_default_hcc_no_action"}
    assert plan.execution_mode == "hcc_noop_baseline"
    assert plan.blocker_reason == ""
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_repair_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.REASSIGN_REPAIR,
        "repair_shared_variable_binding",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("S6", decision)

    assert plan.selected_action_name == "repair_shared_variable_binding"
    assert plan.backend_effect_kind == "shared_variable_owner_rebinding"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {"runtime_hook": "overlap_repair_rule"}
    assert plan.execution_mode == "hcc_smoke_runtime_consumed"
    assert plan.blocker_reason == ""
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_trajectory_action_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "budget_shift_mean_blend",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("E4", decision)

    assert plan.selected_action_name == "budget_shift_mean_blend"
    assert plan.backend_effect_kind == "optimizer_budget_and_mean_trajectory"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {"runtime_hook": "budget_shift_mean_blend"}
    assert plan.execution_mode == "hcc_trajectory_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_bipop_search_state_action_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "bipop_search_state_restart",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("A4", decision)

    assert plan.selected_action_name == "bipop_search_state_restart"
    assert plan.backend_effect_kind == "optimizer_search_state_restart"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {"runtime_hook": "bipop_search_state_restart"}
    assert plan.execution_mode == "hcc_search_state_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_phase_rescue_multistart_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "phase_rescue_multistart",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("R4", decision)

    assert plan.selected_action_name == "phase_rescue_multistart"
    assert plan.backend_effect_kind == "optimizer_phase_rescue_multistart"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "runtime_hook": "phase_rescue_multistart",
        "acceptance_rule": "best_improving_candidate_only",
    }
    assert plan.execution_mode == "hcc_phase_rescue_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_repair_phase_rescue_as_composite_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "repair_phase_rescue_multistart",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("R3", decision)

    assert plan.selected_action_name == "repair_phase_rescue_multistart"
    assert plan.backend_effect_kind == "repair_guided_phase_rescue_multistart"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "overlap_runtime_hook": "overlap_repair_rule",
        "search_state_runtime_hook": "phase_rescue_multistart",
        "acceptance_rule": "best_improving_candidate_only",
    }
    assert plan.execution_mode == "hcc_repair_phase_rescue_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_repair_bipop_as_composite_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "repair_bipop_search_state_restart",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("A5", decision)

    assert plan.selected_action_name == "repair_bipop_search_state_restart"
    assert plan.backend_effect_kind == "repair_guided_optimizer_search_state_restart"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "overlap_runtime_hook": "overlap_repair_rule",
        "search_state_runtime_hook": "bipop_search_state_restart",
    }
    assert plan.execution_mode == "hcc_repair_bipop_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_repair_protect_refine_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "repair_protect_refine",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("A5", decision)

    assert plan.selected_action_name == "repair_protect_refine"
    assert plan.backend_effect_kind == "repair_guided_local_refinement"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "overlap_runtime_hook": "overlap_repair_rule",
        "optimizer_runtime_hook": "protected_small_sigma_refine",
    }
    assert plan.execution_mode == "hcc_repair_refine_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_repair_protect_deep_refine_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "repair_protect_deep_refine",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("A3", decision)

    assert plan.selected_action_name == "repair_protect_deep_refine"
    assert plan.backend_effect_kind == "repair_guided_deep_local_refinement"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "overlap_runtime_hook": "overlap_repair_rule",
        "optimizer_runtime_hook": "protected_deep_sigma_refine",
    }
    assert plan.execution_mode == "hcc_repair_deep_refine_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_cc_harm_guarded_sep_refresh_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "cc_harm_guarded_sep_refresh",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("R4", decision)

    assert plan.selected_action_name == "cc_harm_guarded_sep_refresh"
    assert plan.backend_effect_kind == "cc_harm_guarded_sep_or_nda_refresh"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "runtime_hook": "cc_harm_guarded_sep_refresh",
        "guard": "phase_i_or_current_incumbent_no_harm",
        "refresh_backend": "full_space_mmes_nda_continuation",
        "acceptance_rule": "guarded_incumbent_improving_candidate_only",
    }
    assert plan.execution_mode == "hcc_cc_harm_guarded_refresh_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_separable_cmaes_dispatch_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "separable_cmaes_dispatch_action",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("R5", decision)

    assert plan.selected_action_name == "separable_cmaes_dispatch_action"
    assert plan.backend_effect_kind == "full_space_diagonal_separable_search_takeover"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "runtime_hook": "separable_cmaes_dispatch_action",
        "backend": "direct_separable_cmaes",
        "search_distribution": "diagonal_sigma_full_space",
        "acceptance_rule": "optimizer_best_so_far",
    }
    assert plan.execution_mode == "hcc_direct_separable_cmaes_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_evidence_action_controller_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v1",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("R4", decision)

    assert plan.selected_action_name == "arac_evidence_action_controller_v1"
    assert plan.backend_effect_kind == "evidence_action_runtime_controller"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "relation_runtime_hook": "adaptive_v26_relation_dispatch",
        "overlap_runtime_hook": "evidence_triggered_overlap_action",
        "search_state_runtime_hooks": [
            "phase_rescue_multistart",
            "cc_harm_guarded_sep_refresh",
        ],
        "dispatch_boundary": "runtime_evidence_only",
    }
    assert plan.execution_mode == "hcc_evidence_action_controller_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_evidence_action_controller_v2_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v2",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("S6", decision)

    assert plan.selected_action_name == "arac_evidence_action_controller_v2"
    assert plan.backend_effect_kind == "evidence_action_runtime_controller_v2"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "relation_runtime_hook": "adaptive_v24_relation_dispatch",
        "overlap_runtime_hook": "relation_first_evidence_triggered_overlap_action",
        "search_state_runtime_hooks": [],
        "dispatch_boundary": "runtime_evidence_only",
    }
    assert plan.execution_mode == "hcc_evidence_action_controller_v2_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_evidence_action_controller_v3_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v3",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("S6", decision)

    assert plan.selected_action_name == "arac_evidence_action_controller_v3"
    assert plan.backend_effect_kind == "evidence_action_runtime_controller_v3"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "relation_runtime_hook": "controller_v3_relation_dispatch",
        "mode_selector": "early_runtime_overlap_relation_evidence",
        "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
        "search_state_runtime_hooks": [
            "phase_rescue_multistart",
            "cc_harm_guarded_sep_refresh",
        ],
        "dispatch_boundary": "runtime_evidence_only",
    }
    assert plan.execution_mode == "hcc_evidence_action_controller_v3_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_evidence_action_controller_v31_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v31",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("S6", decision)

    assert plan.selected_action_name == "arac_evidence_action_controller_v31"
    assert plan.backend_effect_kind == "evidence_action_runtime_controller_v31"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "relation_runtime_hook": "controller_v31_guarded_relation_dispatch",
        "mode_selector": "early_runtime_overlap_relation_evidence_with_relation_first_lock",
        "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
        "search_state_runtime_hooks": [
            "resume_phase_i_search_state",
        ],
        "guard": "stable_relation_first_no_harm_gate",
        "dispatch_boundary": "runtime_evidence_only",
    }
    assert plan.execution_mode == "hcc_evidence_action_controller_v31_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_evidence_action_controller_v32_as_group_local_rescue() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v32",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("R2", decision)

    assert plan.selected_action_name == "arac_evidence_action_controller_v32"
    assert plan.backend_effect_kind == "evidence_action_runtime_controller_v32"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "relation_runtime_hook": "controller_v31_guarded_relation_dispatch",
        "mode_selector": "early_runtime_overlap_relation_evidence_with_relation_first_lock",
        "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
        "search_state_runtime_hooks": ["phase_rescue_multistart"],
        "guard": "stable_relation_first_no_harm_gate",
        "dispatch_boundary": "runtime_evidence_only",
    }
    assert plan.execution_mode == "hcc_evidence_action_controller_v32_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_v33_as_risk_aware_runtime_guard() -> None:
    decision = ActionDecision(
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v33",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("E2", decision)

    assert plan.selected_action_name == "arac_evidence_action_controller_v33"
    assert plan.backend_effect_kind == "evidence_action_runtime_controller_v33"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "relation_runtime_hook": "controller_v33_risk_aware_action_guard",
        "mode_selector": "early_runtime_overlap_relation_evidence_with_relation_first_lock",
        "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
        "search_state_runtime_hooks": ["phase_rescue_multistart"],
        "guard": "probation_trust_quarantine_and_exposure_cap",
        "writeback": "topology_scoped_fallback_and_bounded_active_damping",
        "dispatch_boundary": "runtime_evidence_only",
    }
    assert plan.execution_mode == "hcc_evidence_action_controller_v33_runtime_consumed"
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_marks_isolate_as_runtime_consumed() -> None:
    decision = ActionDecision(
        ActionFamily.ISOLATE,
        "isolate_conflicting_relation",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("S6", decision)

    assert plan.selected_action_name == "isolate_conflicting_relation"
    assert plan.backend_effect_kind == "shared_variable_value_selection"
    assert plan.optimizer_consumed is True
    assert plan.optimizer_consumed_parameters == {
        "runtime_hook": "overlap_value_selection_rule"
    }
    assert plan.execution_mode == "hcc_relation_value_selection_consumed"
    assert plan.blocker_reason == ""
    assert plan.runtime_dispatch_allowed is True


def test_hcc_action_execution_plan_blocks_unwired_active_action() -> None:
    decision = ActionDecision(
        ActionFamily.PROTECT,
        "protect_high_margin_group",
        "allow",
        "test",
        0.5,
    )

    plan = build_hcc_action_execution_plan("S6", decision)

    assert plan.selected_action_name == "protect_high_margin_group"
    assert plan.optimizer_consumed is False
    assert plan.execution_mode == "audit_only_not_executed"
    assert plan.blocker_reason == "no_hcc_runtime_consumer_yet"
    assert plan.runtime_dispatch_allowed is False

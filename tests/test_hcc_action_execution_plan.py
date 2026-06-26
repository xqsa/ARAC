from __future__ import annotations

from arac.action_space import ActionFamily
from arac.backends.hcc import build_hcc_action_execution_plan
from arac.policy import ActionDecision


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


def test_hcc_action_execution_plan_blocks_unwired_active_action() -> None:
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
    assert plan.optimizer_consumed is False
    assert plan.optimizer_consumed_parameters == {}
    assert plan.execution_mode == "audit_only_not_executed"
    assert plan.blocker_reason == "no_hcc_runtime_consumer_yet"
    assert plan.runtime_dispatch_allowed is False

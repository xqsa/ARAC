from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_relation_policy_is_a_compatibility_shim_for_action_policy() -> None:
    from arac.policy import action_policy, evidence_model, relation_policy

    assert relation_policy.decide_action is action_policy.decide_action
    assert relation_policy.score_relation_actions is action_policy.score_relation_actions
    assert relation_policy.ActionDecision is action_policy.ActionDecision
    assert action_policy.ActionDecision is evidence_model.ActionDecision
    assert action_policy.ScoredActionDecision is evidence_model.ScoredActionDecision


def test_action_policy_does_not_import_hcc_or_run_optimizer() -> None:
    tree = ast.parse(
        (ROOT / "src" / "arac" / "policy" / "action_policy.py").read_text(
            encoding="utf-8"
        )
    )
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert not any(module.startswith("arac.backends") for module in imported_modules)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        for node in ast.walk(tree)
    )


def test_hcc_backend_does_not_import_paper_or_historical_evidence() -> None:
    source = (ROOT / "src" / "arac" / "backends" / "hcc.py").read_text(
        encoding="utf-8"
    )

    assert "paper/" not in source
    assert "references/paper" not in source
    assert "references/historical" not in source


def test_hcc_runtime_pure_responsibilities_have_dedicated_modules() -> None:
    from arac.backends import hcc, hcc_budget, hcc_plan, hcc_shared_writeback, hcc_trace

    assert hcc_plan.build_hcc_action_execution_plan is hcc.build_hcc_action_execution_plan
    assert hcc_plan.HccActionExecutionPlan is hcc.HccActionExecutionPlan
    assert hcc_shared_writeback.hcc_backend_semantics_for is hcc.hcc_backend_semantics_for
    assert hcc_budget._parse_hcc_evaluation_record is hcc._parse_hcc_evaluation_record
    assert (
        hcc_budget._parse_hcc_evaluation_record_with_optimizer_final_fe
        is hcc._parse_hcc_evaluation_record_with_optimizer_final_fe
    )
    assert hcc_budget._parse_hcc_budget_summary_final_fe is hcc._parse_hcc_budget_summary_final_fe
    assert hcc_budget._parse_hcc_budget_summary is hcc._parse_hcc_budget_summary
    assert hcc_trace._find_hcc_action_trace is hcc._find_hcc_action_trace
    assert hcc_trace._tail is hcc._tail


def test_hcc_budget_and_trace_modules_do_not_own_process_execution() -> None:
    from arac.backends import hcc_budget, hcc_trace

    budget_source = (ROOT / "src" / "arac" / "backends" / "hcc_budget.py").read_text(
        encoding="utf-8"
    )
    trace_source = (ROOT / "src" / "arac" / "backends" / "hcc_trace.py").read_text(
        encoding="utf-8"
    )

    assert "subprocess" not in budget_source
    assert "subprocess" not in trace_source
    assert not hasattr(hcc_budget, "run_hcc_aob_smoke_execution")
    assert not hasattr(hcc_trace, "run_hcc_aob_smoke_execution")


def test_hcc_shared_writeback_is_reference_blind() -> None:
    source = (
        ROOT / "src" / "arac" / "backends" / "hcc_shared_writeback.py"
    ).read_text(encoding="utf-8")

    forbidden_tokens = (
        "paper",
        "historical",
        "relative_gain",
        "function_family",
        "final_outcome",
    )
    lowered = source.casefold()
    assert not any(token in lowered for token in forbidden_tokens)


def test_hcc_budget_parser_is_reference_blind() -> None:
    source = (ROOT / "src" / "arac" / "backends" / "hcc_budget.py").read_text(
        encoding="utf-8"
    )

    lowered = source.casefold()
    assert "paper" not in lowered
    assert "historical" not in lowered

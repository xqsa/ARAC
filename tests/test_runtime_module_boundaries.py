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

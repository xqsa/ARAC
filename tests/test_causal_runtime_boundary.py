from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from arac.policy.causal_risk_scheduler import (
    PRE_ACTION_UTILITY_SCHEMA_VERSION,
    UTILITY_FEATURE_NAMES,
    CausalRiskInvariantError,
    CausalRiskModelBundle,
    PreActionUtilityState,
    UtilityEstimate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/hcc_smoke_runner.py"
POLICY_PATH = REPO_ROOT / "src/arac/policy/causal_risk_scheduler.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one {name}, found {len(matches)}"
    return matches[0]


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one {name}, found {len(matches)}"
    return matches[0]


def _dataclass_fields(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and "ClassVar" not in ast.unparse(statement.annotation)
    )


def _mapping_field(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _accessed_fields(node: ast.AST) -> set[str]:
    accessed = {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
    }
    accessed.update(
        field
        for child in ast.walk(node)
        if (field := _mapping_field(child)) is not None
    )
    accessed.update(
        str(child.args[0].value)
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "get"
        and child.args
        and isinstance(child.args[0], ast.Constant)
        and isinstance(child.args[0].value, str)
    )
    return accessed


def _state_payload() -> dict[str, object]:
    return {
        "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
        **{name: 0.5 for name in UTILITY_FEATURE_NAMES},
    }


def test_hcc_runner_constructs_only_the_pre_action_allowlist() -> None:
    function = _function(_tree(RUNNER_PATH), "build_precision_causal_snapshot")
    payload_assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets)
        and isinstance(node.value, ast.Dict)
    ]
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "PreActionUtilityState"
        and node.func.attr == "from_runtime_payload"
    ]

    assert len(payload_assignments) == 1
    assert len(calls) == 1
    assert len(calls[0].args) == 1
    assert isinstance(calls[0].args[0], ast.Name)
    assert calls[0].args[0].id == "payload"
    keys = tuple(
        key.value
        for key in payload_assignments[0].value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    )
    assert keys == ("schema_version", *UTILITY_FEATURE_NAMES)


def test_policy_contract_and_model_call_are_identity_and_outcome_free() -> None:
    tree = _tree(POLICY_PATH)
    state_contract = _class(tree, "PreActionUtilityState")
    estimate_contract = _class(tree, "UtilityEstimate")
    model_contract = _class(tree, "CausalRiskModelBundle")
    model_estimate = next(
        node
        for node in model_contract.body
        if isinstance(node, ast.FunctionDef) and node.name == "estimate"
    )
    safe_release = _function(tree, "decide_safe_release")
    forbidden = {
        "case",
        "case_id",
        "problem_id",
        "seed",
        "family",
        "function_family",
        "fingerprint",
        "graph_fingerprint",
        "component_fingerprint",
        "relation_fingerprint",
        "outcome",
        "final_outcome",
        "final_error",
        "paper_best",
        "historical_best",
    }

    assert _dataclass_fields(state_contract) == (
        "schema_version",
        *UTILITY_FEATURE_NAMES,
    )
    assert not forbidden.intersection(_dataclass_fields(estimate_contract))
    assert not forbidden.intersection(_accessed_fields(model_estimate))
    assert not forbidden.intersection(_accessed_fields(safe_release))
    assert ast.unparse(model_estimate.args.args[1].annotation) == "PreActionUtilityState"

    estimate_calls = [
        node
        for node in ast.walk(safe_release)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "estimate"
    ]
    assert len(estimate_calls) == 1
    assert isinstance(estimate_calls[0].func.value, ast.Name)
    assert estimate_calls[0].func.value.id == "model_bundle"
    assert len(estimate_calls[0].args) == 1
    assert isinstance(estimate_calls[0].args[0], ast.Name)
    assert estimate_calls[0].args[0].id == "state"

    runtime_fields = {
        field.name for field in (*fields(PreActionUtilityState), *fields(UtilityEstimate))
    }
    assert not forbidden.intersection(runtime_fields)


def test_every_declared_forbidden_runtime_field_fails_closed() -> None:
    forbidden = PreActionUtilityState.forbidden_field_names()
    assert forbidden
    for field in forbidden:
        payload = _state_payload()
        payload[field] = "forbidden"
        with pytest.raises(CausalRiskInvariantError, match="forbidden runtime field"):
            PreActionUtilityState.from_runtime_payload(payload)


@pytest.mark.parametrize("field", ("family", "fingerprint", "outcome"))
def test_forbidden_category_aliases_are_rejected_as_unknown(field: str) -> None:
    payload = _state_payload()
    payload[field] = "forbidden"
    with pytest.raises(CausalRiskInvariantError, match="unknown runtime field"):
        PreActionUtilityState.from_runtime_payload(payload)


def test_snapshot_cannot_be_backfilled_and_model_rejects_raw_payload() -> None:
    payload = _state_payload()
    state = PreActionUtilityState.from_runtime_payload(payload)
    original_hash = state.feature_sha256
    payload["remaining_fe_ratio"] = 0.9

    assert state.remaining_fe_ratio == 0.5
    assert state.feature_sha256 == original_hash
    with pytest.raises(FrozenInstanceError):
        state.remaining_fe_ratio = 0.9  # type: ignore[misc]
    with pytest.raises(CausalRiskInvariantError, match="requires PreActionUtilityState"):
        CausalRiskModelBundle.estimate(object(), payload)  # type: ignore[arg-type]

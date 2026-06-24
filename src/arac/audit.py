"""Audit helpers for leakage and claim gates."""

from __future__ import annotations

from .backend_adapter import BackendSemanticsDiff
from .evidence import FORBIDDEN_RUNTIME_FIELDS
from .evaluation import SameBudgetLedger
from .policy import ActionDecision


def find_forbidden_runtime_fields(payload: dict) -> list[str]:
    return sorted(FORBIDDEN_RUNTIME_FIELDS.intersection(payload))


def active_action_has_effect(decision: ActionDecision, diff: BackendSemanticsDiff) -> bool:
    if decision.action_family.value == "fallback":
        return True
    return diff.changed


def claim_gate(
    *,
    runtime_payload: dict,
    decision: ActionDecision,
    semantics_diff: BackendSemanticsDiff,
    ledger: SameBudgetLedger,
    utility_label: str,
    negative_control_pass: bool,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    forbidden = find_forbidden_runtime_fields(runtime_payload)
    if forbidden:
        blockers.append("forbidden_runtime_fields:" + ",".join(forbidden))
    if not active_action_has_effect(decision, semantics_diff):
        blockers.append("active_action_without_backend_semantic_effect")
    if ledger.violation:
        blockers.append("same_budget_violation")
    if not ledger.fresh_execution:
        blockers.append("not_fresh_execution")
    if utility_label == "catastrophic_loss":
        blockers.append("catastrophic_loss")
    if not negative_control_pass:
        blockers.append("negative_control_failed")
    return not blockers, blockers


from __future__ import annotations

import pytest

from arac.audit import claim_gate
from arac.action_space import ActionFamily
from arac.backends.hcc import build_hcc_action_execution_plan
from arac.backend_adapter import BackendSemanticsDiff
from arac.evaluation import SameBudgetLedger, classify_utility
from arac.evidence import EvidenceProfile, validate_runtime_payload
from arac.policy import ActionDecision, decide_action


def make_evidence(**overrides: object) -> EvidenceProfile:
    values = {
        "run_id": "run-001",
        "problem_id": "problem-a",
        "seed": 1,
        "unit_type": "relation",
        "unit_id": "rel-001",
        "feature_coverage": 1.0,
        "overlap_degree": 0.2,
        "shared_var_support_ratio": 0.1,
        "direction_disagreement": 0.9,
        "harmful_coord_score": 0.8,
        "group_gain_asymmetry": 0.1,
        "priority_spread": 0.2,
        "rank_stability": 0.9,
        "budget_remaining_ratio": 0.5,
        "fallback_margin_proxy": 0.8,
    }
    values.update(overrides)
    return EvidenceProfile(**values)


def test_arac_package_imports_policy_and_selects_reference_blind_action() -> None:
    decision = decide_action(make_evidence())

    assert decision.action_family.value == "isolate"
    assert decision.action_name == "isolate_conflicting_relation"
    assert decision.decision == "allow"


def test_policy_selects_coordinate_for_stable_beneficial_evidence() -> None:
    decision = decide_action(
        make_evidence(
            overlap_degree=0.2,
            shared_var_support_ratio=0.1,
            direction_disagreement=0.05,
            harmful_coord_score=0.05,
            group_gain_asymmetry=0.2,
            priority_spread=0.25,
            rank_stability=0.95,
            fallback_margin_proxy=0.85,
        )
    )

    assert decision.action_family.value == "coordinate"
    assert decision.action_name == "allow_beneficial_coordination"
    assert decision.decision == "allow"


def test_runtime_payload_rejects_forbidden_outcome_fields() -> None:
    with pytest.raises(ValueError, match="final_error"):
        validate_runtime_payload({"overlap_degree": 0.5, "final_error": 1.2})


def test_runtime_payload_rejects_paper_and_problem_family_fields() -> None:
    with pytest.raises(ValueError, match="paper_reported_mean"):
        validate_runtime_payload({"overlap_degree": 0.5, "paper_reported_mean": 6.87e6})

    with pytest.raises(ValueError, match="base_function"):
        validate_runtime_payload({"overlap_degree": 0.5, "base_function": "elliptic"})


def test_claim_gate_blocks_active_action_without_backend_semantics() -> None:
    decision = decide_action(make_evidence())
    ledger = SameBudgetLedger(
        phase_i_fe=10,
        phase_ii_fe=20,
        budget_limit=40,
        fresh_execution=True,
    )

    allowed, blockers = claim_gate(
        runtime_payload={"overlap_degree": 0.2, "direction_disagreement": 0.9},
        decision=decision,
        semantics_diff=BackendSemanticsDiff(),
        ledger=ledger,
        utility_label="meaningful_win",
        negative_control_pass=True,
    )

    assert allowed is False
    assert "active_action_without_backend_semantic_effect" in blockers


def test_claim_gate_blocks_active_action_without_hcc_runtime_consumption() -> None:
    decision = ActionDecision(
        ActionFamily.PROTECT,
        "protect_high_margin_group",
        "allow",
        "test",
        0.4,
    )
    plan = build_hcc_action_execution_plan("E2", decision)

    allowed, blockers = claim_gate(
        runtime_payload={"overlap_degree": 0.9, "direction_disagreement": 0.1},
        decision=decision,
        semantics_diff=BackendSemanticsDiff(budget_allocation_changed=True),
        ledger=SameBudgetLedger(
            phase_i_fe=100,
            phase_ii_fe=100,
            budget_limit=200,
            fresh_execution=True,
        ),
        utility_label="beneficial",
        negative_control_pass=True,
        optimizer_consumed=plan.optimizer_consumed,
    )

    assert allowed is False
    assert "active_action_not_consumed_by_hcc_runtime" in blockers


def test_utility_classification_uses_meaningful_and_catastrophic_thresholds() -> None:
    assert classify_utility(100.0, 94.0) == "meaningful_win"
    assert classify_utility(100.0, 125.0) == "catastrophic_loss"
    assert classify_utility(100.0, 98.0) == "tie_or_small_effect"

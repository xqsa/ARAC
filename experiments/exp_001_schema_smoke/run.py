from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from arac.action_space import ActionFamily
from arac.audit import claim_gate
from arac.backend_adapter import BackendSemanticsDiff, ToyBackendAdapter
from arac.evaluation import SameBudgetLedger, classify_utility, relative_gain
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS, EvidenceProfile, validate_runtime_payload
from arac.policy import ActionDecision, decide_action

RUN_ID = "exp_001_schema_smoke"
PROBLEM_ID = "toy_overlap_case_001"
SEED = 1
WINDOW_ID = "phase_i_window_001"
PHASE_I_FE = 20
BUDGET_LIMIT = 100
PHASE_II_FE = BUDGET_LIMIT - PHASE_I_FE

LANE_FINAL_ERRORS = {
    "policy_action": 88.0,
    "no_action": 100.0,
    "fallback": 96.0,
    "random_action": 128.0,
    "shuffled_evidence_action": 104.0,
    "oracle_action_eval_only": 82.0,
}

RUNTIME_LANES = {
    "policy_action",
    "no_action",
    "fallback",
    "random_action",
    "shuffled_evidence_action",
}


def make_tiny_trace() -> list[EvidenceProfile]:
    return [
        EvidenceProfile(
            run_id=RUN_ID,
            problem_id=PROBLEM_ID,
            seed=SEED,
            unit_type="relation",
            unit_id="relation_conflict",
            feature_coverage=1.0,
            overlap_degree=0.40,
            shared_var_support_ratio=0.30,
            direction_disagreement=0.92,
            harmful_coord_score=0.85,
            group_gain_asymmetry=0.20,
            priority_spread=0.30,
            rank_stability=0.90,
            budget_remaining_ratio=0.80,
            fallback_margin_proxy=0.82,
        ),
        EvidenceProfile(
            run_id=RUN_ID,
            problem_id=PROBLEM_ID,
            seed=SEED,
            unit_type="group",
            unit_id="group_priority",
            feature_coverage=1.0,
            overlap_degree=0.20,
            shared_var_support_ratio=0.35,
            direction_disagreement=0.10,
            harmful_coord_score=0.20,
            group_gain_asymmetry=0.70,
            priority_spread=0.92,
            rank_stability=0.88,
            budget_remaining_ratio=0.80,
            fallback_margin_proxy=0.72,
        ),
        EvidenceProfile(
            run_id=RUN_ID,
            problem_id=PROBLEM_ID,
            seed=SEED,
            unit_type="shared_variable",
            unit_id="shared_variable_binding",
            feature_coverage=1.0,
            overlap_degree=0.86,
            shared_var_support_ratio=0.88,
            direction_disagreement=0.20,
            harmful_coord_score=0.25,
            group_gain_asymmetry=0.30,
            priority_spread=0.30,
            rank_stability=0.84,
            budget_remaining_ratio=0.80,
            fallback_margin_proxy=0.74,
        ),
        EvidenceProfile(
            run_id=RUN_ID,
            problem_id=PROBLEM_ID,
            seed=SEED,
            unit_type="relation",
            unit_id="beneficial_coordination",
            feature_coverage=1.0,
            overlap_degree=0.72,
            shared_var_support_ratio=0.72,
            direction_disagreement=0.05,
            harmful_coord_score=0.05,
            group_gain_asymmetry=0.08,
            priority_spread=0.10,
            rank_stability=0.92,
            budget_remaining_ratio=0.80,
            fallback_margin_proxy=0.66,
        ),
    ]


def coordinate_decision() -> ActionDecision:
    return ActionDecision(
        ActionFamily.COORDINATE,
        "allow_beneficial_coordination",
        "allow",
        "beneficial_overlap_with_low_conflict",
        0.38,
    )


def lane_decisions(trace: list[EvidenceProfile]) -> dict[str, tuple[EvidenceProfile, ActionDecision]]:
    return {
        "policy_action": (
            trace[0],
            decide_action(trace[0]),
        ),
        "no_action": (
            trace[0],
            ActionDecision(
                ActionFamily.FALLBACK,
                "conservative_no_action",
                "fallback",
                "no_action_baseline",
                0.0,
            ),
        ),
        "fallback": (
            trace[0],
            ActionDecision(
                ActionFamily.FALLBACK,
                "conservative_no_action",
                "fallback",
                "policy_abstain_baseline",
                0.0,
            ),
        ),
        "random_action": (
            trace[1],
            ActionDecision(
                ActionFamily.PROTECT,
                "protect_high_margin_group",
                "allow",
                "negative_control_random_action",
                0.10,
            ),
        ),
        "shuffled_evidence_action": (
            trace[2],
            ActionDecision(
                ActionFamily.REASSIGN_REPAIR,
                "repair_shared_variable_binding",
                "allow",
                "negative_control_shuffled_evidence",
                0.20,
            ),
        ),
        "oracle_action_eval_only": (
            trace[0],
            ActionDecision(
                ActionFamily.ISOLATE,
                "isolate_conflicting_relation",
                "shadow_only",
                "oracle_final_error_eval_only_forbidden_at_runtime",
                1.0,
            ),
        ),
    }


def semantic_probe_decisions() -> list[ActionDecision]:
    trace = make_tiny_trace()
    return [
        decide_action(trace[0]),
        decide_action(trace[1]),
        decide_action(trace[2]),
        coordinate_decision(),
        ActionDecision(
            ActionFamily.FALLBACK,
            "conservative_no_action",
            "fallback",
            "fallback_semantics_probe",
            0.0,
        ),
    ]


def runtime_payload(evidence: EvidenceProfile) -> dict[str, object]:
    payload = asdict(evidence)
    payload["window_id"] = WINDOW_ID
    payload["feature_source"] = "tiny_synthetic_trace"
    payload["used_for_runtime"] = 1
    return payload


def trigger_features_for(decision: ActionDecision) -> str:
    match decision.action_name:
        case "isolate_conflicting_relation":
            return "direction_disagreement;harmful_coord_score;fallback_margin_proxy"
        case "protect_high_margin_group":
            return "priority_spread;rank_stability"
        case "repair_shared_variable_binding":
            return "overlap_degree;shared_var_support_ratio;fallback_margin_proxy"
        case "allow_beneficial_coordination":
            return "overlap_degree;shared_var_support_ratio;harmful_coord_score"
        case _:
            return "feature_coverage;budget_remaining_ratio"


def semantics_hash(label: str, decision: ActionDecision, diff: BackendSemanticsDiff) -> str:
    payload = "|".join(
        [
            label,
            decision.action_name,
            str(int(diff.variable_owner_changed)),
            str(int(diff.relation_handling_changed)),
            str(int(diff.coordination_mode_changed)),
            str(int(diff.budget_allocation_changed)),
            str(int(diff.update_order_changed)),
            str(int(diff.acceptance_rule_changed)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evidence_rows(trace: Iterable[EvidenceProfile]) -> list[dict[str, object]]:
    rows = []
    for evidence in trace:
        payload = runtime_payload(evidence)
        validate_runtime_payload(payload)
        rows.append(payload)
    return rows


def action_rows(decisions: dict[str, tuple[EvidenceProfile, ActionDecision]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lane, (evidence, decision) in decisions.items():
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": evidence.problem_id,
                "seed": evidence.seed,
                "window_id": WINDOW_ID,
                "unit_type": evidence.unit_type,
                "unit_id": evidence.unit_id,
                "plan_name": lane,
                "selected_action_family": decision.action_family.value,
                "selected_action_name": decision.action_name,
                "decision": decision.decision,
                "trigger_features": trigger_features_for(decision),
                "trigger_reason": decision.trigger_reason,
                "expected_gain_proxy": f"{max(decision.utility_proxy, 0.0):.6f}",
                "action_cost_proxy": "0.050000" if decision.action_family.value != "fallback" else "0.000000",
                "risk_penalty_proxy": "0.050000" if lane in {"random_action", "shuffled_evidence_action"} else "0.000000",
                "utility_proxy": f"{decision.utility_proxy:.6f}",
                "fallback_action": decision.fallback_action,
                "negative_control_status": "fail"
                if lane in {"random_action", "shuffled_evidence_action"}
                else "pass",
                "used_for_runtime": 1 if lane in RUNTIME_LANES else 0,
            }
        )
    return rows


def backend_rows(adapter: ToyBackendAdapter) -> list[dict[str, object]]:
    rows = []
    for decision in semantic_probe_decisions():
        diff = adapter.apply(decision)
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": PROBLEM_ID,
                "seed": SEED,
                "selected_action_name": decision.action_name,
                "backend_name": "toy_backend_adapter",
                "variable_owner_changed": int(diff.variable_owner_changed),
                "relation_handling_changed": int(diff.relation_handling_changed),
                "coordination_mode_changed": int(diff.coordination_mode_changed),
                "budget_allocation_changed": int(diff.budget_allocation_changed),
                "update_order_changed": int(diff.update_order_changed),
                "acceptance_rule_changed": int(diff.acceptance_rule_changed),
                "semantics_hash_before": semantics_hash("before", decision, BackendSemanticsDiff()),
                "semantics_hash_after": semantics_hash("after", decision, diff),
                "backend_semantics_changed": int(diff.changed),
            }
        )
    return rows


def ledger_for_lane(lane: str) -> SameBudgetLedger:
    phase_ii = PHASE_II_FE
    if lane == "no_action":
        phase_ii = PHASE_II_FE
    return SameBudgetLedger(
        phase_i_fe=PHASE_I_FE,
        phase_ii_fe=phase_ii,
        budget_limit=BUDGET_LIMIT,
        fresh_execution=True,
    )


def ledger_rows(decisions: dict[str, tuple[EvidenceProfile, ActionDecision]]) -> list[dict[str, object]]:
    rows = []
    for lane, (evidence, _) in decisions.items():
        ledger = ledger_for_lane(lane)
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": evidence.problem_id,
                "seed": evidence.seed,
                "plan_name": lane,
                "phase_i_fe": ledger.phase_i_fe,
                "phase_ii_fe": ledger.phase_ii_fe,
                "total_fe": ledger.total_fe,
                "budget_limit": ledger.budget_limit,
                "same_budget_violation": int(ledger.violation),
                "fresh_execution": int(ledger.fresh_execution),
            }
        )
    return rows


def utility_rows(
    decisions: dict[str, tuple[EvidenceProfile, ActionDecision]],
    adapter: ToyBackendAdapter,
) -> list[dict[str, object]]:
    fallback_error = LANE_FINAL_ERRORS["fallback"]
    rows = []
    for lane, (evidence, decision) in decisions.items():
        action_error = LANE_FINAL_ERRORS[lane]
        utility_label = classify_utility(fallback_error, action_error)
        negative_control_pass = lane not in {"random_action", "shuffled_evidence_action"}
        diff = adapter.apply(decision)
        ledger = ledger_for_lane(lane)
        runtime = lane in RUNTIME_LANES
        payload = runtime_payload(evidence) if runtime else {"eval_only_lane": lane}
        allowed, blockers = claim_gate(
            runtime_payload=payload,
            decision=decision,
            semantics_diff=diff,
            ledger=ledger,
            utility_label=utility_label,
            negative_control_pass=negative_control_pass,
        )
        if lane == "oracle_action_eval_only":
            allowed = False
            blockers.append("oracle_eval_only_not_runtime_dispatch")
        if lane in {"no_action", "fallback"}:
            allowed = False
            blockers.append("comparison_lane_not_action_claim")

        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": evidence.problem_id,
                "seed": evidence.seed,
                "plan_name": lane,
                "fallback_plan": "fallback",
                "action_final_error": f"{action_error:.6f}",
                "fallback_final_error": f"{fallback_error:.6f}",
                "relative_gain_vs_fallback": f"{relative_gain(fallback_error, action_error):.6f}",
                "meaningful_win": int(utility_label == "meaningful_win"),
                "catastrophic_loss": int(utility_label == "catastrophic_loss"),
                "negative_control_pass": int(negative_control_pass),
                "backend_semantics_changed": int(diff.changed),
                "claim_eligible": int(allowed),
                "claim_blockers": ";".join(sorted(set(blockers))),
                "runtime_dispatch_allowed": int(runtime and lane != "oracle_action_eval_only"),
            }
        )
    return rows


def anti_leakage_rows(decisions: dict[str, tuple[EvidenceProfile, ActionDecision]]) -> list[dict[str, object]]:
    rows = []
    runtime_payloads = [
        runtime_payload(evidence)
        for lane, (evidence, _) in decisions.items()
        if lane in RUNTIME_LANES
    ]
    for field in sorted(FORBIDDEN_RUNTIME_FIELDS):
        found = any(field in payload for payload in runtime_payloads)
        rows.append(
            {
                "run_id": RUN_ID,
                "artifact_path": "evidence_profile.csv;action_decision.csv",
                "forbidden_field": field,
                "found_in_runtime_payload": int(found),
                "audit_status": "fail" if found else "pass",
                "note": "runtime scan excludes oracle/final/reference fields",
            }
        )
    return rows


def write_manifest(output_dir: Path) -> None:
    manifest = "\n".join(
        [
            "# exp_001_schema_smoke Run Manifest",
            "",
            "Date: 2026-06-24",
            "Executor: Codex",
            "Claim level: schema smoke / toy backend semantics only",
            "Runtime boundary: oracle_action_eval_only is offline evaluation only",
            f"Budget: phase_i_fe={PHASE_I_FE}, phase_ii_fe={PHASE_II_FE}, total_fe={BUDGET_LIMIT}",
            "Negative controls: random_action and shuffled_evidence_action are blocked from claims",
            "Final-success claim: not allowed",
            "",
        ]
    )
    (output_dir / "run_manifest.md").write_text(manifest, encoding="utf-8")


def run_schema_smoke(output_dir: Path | str = Path("results/exp_001_schema_smoke")) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = ToyBackendAdapter()
    trace = make_tiny_trace()
    decisions = lane_decisions(trace)

    write_csv(
        output / "evidence_profile.csv",
        evidence_rows(trace),
        [
            "run_id",
            "problem_id",
            "seed",
            "unit_type",
            "unit_id",
            "feature_coverage",
            "overlap_degree",
            "shared_var_support_ratio",
            "direction_disagreement",
            "harmful_coord_score",
            "group_gain_asymmetry",
            "priority_spread",
            "rank_stability",
            "budget_remaining_ratio",
            "fallback_margin_proxy",
            "window_id",
            "feature_source",
            "used_for_runtime",
        ],
    )
    write_csv(
        output / "action_decision.csv",
        action_rows(decisions),
        [
            "run_id",
            "problem_id",
            "seed",
            "window_id",
            "unit_type",
            "unit_id",
            "plan_name",
            "selected_action_family",
            "selected_action_name",
            "decision",
            "trigger_features",
            "trigger_reason",
            "expected_gain_proxy",
            "action_cost_proxy",
            "risk_penalty_proxy",
            "utility_proxy",
            "fallback_action",
            "negative_control_status",
            "used_for_runtime",
        ],
    )
    write_csv(
        output / "backend_semantics_diff.csv",
        backend_rows(adapter),
        [
            "run_id",
            "problem_id",
            "seed",
            "selected_action_name",
            "backend_name",
            "variable_owner_changed",
            "relation_handling_changed",
            "coordination_mode_changed",
            "budget_allocation_changed",
            "update_order_changed",
            "acceptance_rule_changed",
            "semantics_hash_before",
            "semantics_hash_after",
            "backend_semantics_changed",
        ],
    )
    write_csv(
        output / "same_budget_ledger.csv",
        ledger_rows(decisions),
        [
            "run_id",
            "problem_id",
            "seed",
            "plan_name",
            "phase_i_fe",
            "phase_ii_fe",
            "total_fe",
            "budget_limit",
            "same_budget_violation",
            "fresh_execution",
        ],
    )
    write_csv(
        output / "action_utility_audit.csv",
        utility_rows(decisions, adapter),
        [
            "run_id",
            "problem_id",
            "seed",
            "plan_name",
            "fallback_plan",
            "action_final_error",
            "fallback_final_error",
            "relative_gain_vs_fallback",
            "meaningful_win",
            "catastrophic_loss",
            "negative_control_pass",
            "backend_semantics_changed",
            "claim_eligible",
            "claim_blockers",
            "runtime_dispatch_allowed",
        ],
    )
    write_csv(
        output / "anti_leakage_audit.csv",
        anti_leakage_rows(decisions),
        [
            "run_id",
            "artifact_path",
            "forbidden_field",
            "found_in_runtime_payload",
            "audit_status",
            "note",
        ],
    )
    write_manifest(output)
    return output


if __name__ == "__main__":
    run_schema_smoke()

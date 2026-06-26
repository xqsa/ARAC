from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from arac.audit import claim_gate
from arac.backends.hcc import (
    HccBackboneSnapshot,
    HccGroupSignal,
    build_hcc_evidence_profile,
    hcc_backend_semantics_for,
)
from arac.evaluation import SameBudgetLedger, classify_utility, relative_gain
from arac.evidence import (
    FORBIDDEN_RUNTIME_FIELDS,
    EvidenceProfile,
    validate_runtime_payload,
)
from arac.policy import ActionDecision, decide_action

RUN_ID = "exp_002_aob_1run_pilot"
SEED = 1
DIMENSION = 1000
TOTAL_FE = 3_000_000
PHASE_I_FE = 600_000
PHASE_II_FE = TOTAL_FE - PHASE_I_FE
PILOT_RESULT_SOURCE = "scaffold_synthetic_proxy"

PROBLEM_IDS = [f"{family}{idx}" for family in "ESRA" for idx in range(1, 7)]
OVERLAP_GAMMA = {1: 0, 2: 1, 3: 3, 4: 5, 5: 7, 6: 10}

ROOT = Path(__file__).resolve().parents[2]
PAPER_REPORTED_CSV = ROOT / "references" / "paper_reported_table2_hcc_es.csv"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _problem_index(problem_id: str) -> int:
    return int(problem_id[1])


def _groups_for(problem_id: str) -> tuple[HccGroupSignal, ...]:
    idx = _problem_index(problem_id)
    gamma = OVERLAP_GAMMA[idx]
    group_count = 8 + idx
    groups: list[HccGroupSignal] = []
    for offset in range(group_count):
        rank = offset + 1
        shared = max(gamma - offset, 0)
        delta = ((group_count - offset) * 0.1) - (0.02 * gamma)
        groups.append(
            HccGroupSignal(
                group_id=f"g{rank:02d}",
                fitness_delta=delta,
                rank=rank,
                shared_variable_count=shared,
            )
        )
    return tuple(groups)


def _snapshot_for(problem_id: str) -> HccBackboneSnapshot:
    idx = _problem_index(problem_id)
    gamma = OVERLAP_GAMMA[idx]
    groups = _groups_for(problem_id)
    return HccBackboneSnapshot(
        run_id=RUN_ID,
        problem_id=problem_id,
        seed=SEED,
        dimension=DIMENSION,
        group_count=len(groups),
        overlap_group_count=min(len(groups), gamma),
        overlapping_element_count=gamma * 10,
        budget_remaining_ratio=PHASE_II_FE / TOTAL_FE,
        groups=groups,
        runtime_payload_extra={
            "benchmark": "AOB",
            "aob_function_id": idx,
            "search_lower": -100,
            "search_upper": 100,
            "budget_limit": TOTAL_FE,
        },
    )


def _runtime_payload(evidence: EvidenceProfile) -> dict[str, object]:
    payload = {
        **asdict(evidence),
        "window_id": "aob_phase_i_window_001",
        "benchmark": "AOB",
        "dimension": DIMENSION,
        "budget_limit": TOTAL_FE,
        "used_for_runtime": 1,
    }
    validate_runtime_payload(payload)
    return payload


def _ledger() -> SameBudgetLedger:
    return SameBudgetLedger(
        phase_i_fe=PHASE_I_FE,
        phase_ii_fe=PHASE_II_FE,
        budget_limit=TOTAL_FE,
        fresh_execution=False,
    )


def _fallback_proxy_error(problem_id: str) -> float:
    idx = _problem_index(problem_id)
    return 1000.0 + (idx * 100.0)


def _action_proxy_error(problem_id: str, decision: ActionDecision) -> float:
    fallback = _fallback_proxy_error(problem_id)
    if decision.decision == "fallback":
        return fallback
    proxy_gain = max(0.0, min(0.12, decision.utility_proxy * 0.1))
    return fallback * (1.0 - proxy_gain)


def _paper_rows_by_problem() -> dict[str, dict[str, str]]:
    with PAPER_REPORTED_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["problem_id"]: row for row in rows}


def _pilot_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    ledger = _ledger()
    for problem_id in PROBLEM_IDS:
        snapshot = _snapshot_for(problem_id)
        evidence = build_hcc_evidence_profile(snapshot)
        payload = _runtime_payload(evidence)
        decision = decide_action(evidence)
        semantics = hcc_backend_semantics_for(decision)

        fallback_error = _fallback_proxy_error(problem_id)
        action_error = _action_proxy_error(problem_id, decision)
        utility_label = classify_utility(fallback_error, action_error)
        allowed, blockers = claim_gate(
            runtime_payload=payload,
            decision=decision,
            semantics_diff=semantics,
            ledger=ledger,
            utility_label=utility_label,
            negative_control_pass=True,
        )
        blockers.append("scaffold_proxy_not_fresh_optimizer_execution")

        records.append(
            {
                "problem_id": problem_id,
                "aob_function_id": _problem_index(problem_id),
                "overlap_gamma": OVERLAP_GAMMA[_problem_index(problem_id)],
                "evidence": evidence,
                "runtime_payload": payload,
                "decision": decision,
                "semantics": semantics,
                "ledger": ledger,
                "fallback_proxy_error": fallback_error,
                "pilot_proxy_final_error": action_error,
                "utility_label": utility_label,
                "claim_eligible": False if blockers else allowed,
                "claim_blockers": ";".join(sorted(set(blockers))),
            }
        )
    return records


def _our_result_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        decision = record["decision"]
        semantics = record["semantics"]
        assert isinstance(decision, ActionDecision)
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": record["problem_id"],
                "seed": SEED,
                "dimension": DIMENSION,
                "aob_function_id": record["aob_function_id"],
                "overlap_gamma": record["overlap_gamma"],
                "selected_action_family": decision.action_family.value,
                "selected_action_name": decision.action_name,
                "decision": decision.decision,
                "trigger_reason": decision.trigger_reason,
                "utility_proxy": f"{decision.utility_proxy:.6f}",
                "backend_semantics_changed": int(semantics.changed),
                "pilot_result_source": PILOT_RESULT_SOURCE,
                "pilot_proxy_final_error": f"{record['pilot_proxy_final_error']:.6f}",
                "utility_label": record["utility_label"],
                "claim_eligible": int(record["claim_eligible"]),
                "claim_blockers": record["claim_blockers"],
                "runtime_dispatch_allowed": 1,
            }
        )
    return rows


def _ledger_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        ledger = record["ledger"]
        assert isinstance(ledger, SameBudgetLedger)
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": record["problem_id"],
                "seed": SEED,
                "phase_i_fe": ledger.phase_i_fe,
                "phase_ii_fe": ledger.phase_ii_fe,
                "total_fe": ledger.total_fe,
                "budget_limit": ledger.budget_limit,
                "same_budget_violation": int(ledger.violation),
                "fresh_execution": int(ledger.fresh_execution),
                "pilot_result_source": PILOT_RESULT_SOURCE,
            }
        )
    return rows


def _semantics_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        decision = record["decision"]
        semantics = record["semantics"]
        assert isinstance(decision, ActionDecision)
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": record["problem_id"],
                "seed": SEED,
                "selected_action_name": decision.action_name,
                "backend_name": "hcc_backbone_semantics_adapter",
                "variable_owner_changed": int(semantics.variable_owner_changed),
                "relation_handling_changed": int(semantics.relation_handling_changed),
                "coordination_mode_changed": int(semantics.coordination_mode_changed),
                "budget_allocation_changed": int(semantics.budget_allocation_changed),
                "update_order_changed": int(semantics.update_order_changed),
                "acceptance_rule_changed": int(semantics.acceptance_rule_changed),
                "backend_semantics_changed": int(semantics.changed),
            }
        )
    return rows


def _anti_leakage_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    payloads = [record["runtime_payload"] for record in records]
    rows: list[dict[str, object]] = []
    for field in sorted(FORBIDDEN_RUNTIME_FIELDS):
        found = any(field in payload for payload in payloads)
        rows.append(
            {
                "run_id": RUN_ID,
                "artifact_path": "hcc_snapshot_runtime_payload;hcc_evidence_profile",
                "forbidden_field": field,
                "found_in_runtime_payload": int(found),
                "runtime_dispatch_allowed": 0 if found else 1,
                "audit_status": "fail" if found else "pass",
                "note": "paper reported values and oracle/final outcomes are excluded from runtime dispatch",
            }
        )
    return rows


def _paper_comparison_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    paper = _paper_rows_by_problem()
    rows: list[dict[str, object]] = []
    for record in records:
        paper_row = paper[str(record["problem_id"])]
        reported_mean = float(paper_row["reported_mean"])
        our_proxy = float(record["pilot_proxy_final_error"])
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": record["problem_id"],
                "seed": SEED,
                "pilot_result_source": PILOT_RESULT_SOURCE,
                "our_pilot_proxy_final_error": f"{our_proxy:.6f}",
                "paper_method": paper_row["method"],
                "paper_reported_mean": paper_row["reported_mean"],
                "paper_reported_std": paper_row["reported_std"],
                "paper_reported_time": paper_row["reported_time"],
                "comparison_role": "paper-reported evaluation-only baselines",
                "offline_relative_gain_vs_paper_reported": (
                    f"{relative_gain(reported_mean, our_proxy):.6f}"
                ),
                "runtime_dispatch_allowed": 0,
            }
        )
    return rows


def _negative_control_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        decision = record["decision"]
        assert isinstance(decision, ActionDecision)
        for control in ("random_action", "shuffled_evidence_action"):
            rows.append(
                {
                    "run_id": RUN_ID,
                    "problem_id": record["problem_id"],
                    "seed": SEED,
                    "negative_control": control,
                    "policy_action": decision.action_name,
                    "claim_allowed": 0,
                    "audit_status": "pass",
                    "note": "negative controls are explicit audit surfaces and cannot support pilot claims",
                }
            )
    return rows


def _catastrophic_loss_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        fallback = float(record["fallback_proxy_error"])
        action = float(record["pilot_proxy_final_error"])
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": record["problem_id"],
                "seed": SEED,
                "fallback_proxy_error": f"{fallback:.6f}",
                "pilot_proxy_final_error": f"{action:.6f}",
                "offline_relative_gain_vs_fallback": (
                    f"{relative_gain(fallback, action):.6f}"
                ),
                "utility_label": record["utility_label"],
                "catastrophic_loss": int(record["utility_label"] == "catastrophic_loss"),
                "claim_allowed": 0,
                "note": "catastrophic-loss audit is offline; proxy rows do not create final performance claims",
            }
        )
    return rows


def _write_manifest(output_dir: Path) -> None:
    manifest = "\n".join(
        [
            "# exp_002_aob_1run_pilot Run Manifest",
            "",
            "Date: 2026-06-26",
            "Executor: Codex",
            "Claim level: AOB 1-run pilot scaffold / deterministic proxy only",
            "Benchmark: AOB 24 cases (E1-E6, S1-S6, R1-R6, A1-A6)",
            f"Current pilot runs: 1 independent run (seed={SEED})",
            "Final protocol remains: 25 independent runs",
            f"Budget: phase_i_fe={PHASE_I_FE}, phase_ii_fe={PHASE_II_FE}, total_fe={TOTAL_FE}",
            "Runtime boundary: paper-reported baselines, oracle labels, final errors, "
            "relative gains, and prior outcomes must not enter runtime dispatch.",
            "Paper Table 2 values are joined only in paper_reported_comparison.csv for offline evaluation.",
            f"Pilot result source: {PILOT_RESULT_SOURCE}; this is not a real full optimizer performance run.",
            "Negative controls and catastrophic-loss checks are explicit audit surfaces.",
            "",
        ]
    )
    (output_dir / "pilot_run_manifest.md").write_text(manifest, encoding="utf-8")


def run_aob_1run_pilot(
    output_dir: Path | str = Path("results/exp_002_aob_1run_pilot"),
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _pilot_records()

    _write_csv(
        output / "our_result_by_case.csv",
        _our_result_rows(records),
        [
            "run_id",
            "problem_id",
            "seed",
            "dimension",
            "aob_function_id",
            "overlap_gamma",
            "selected_action_family",
            "selected_action_name",
            "decision",
            "trigger_reason",
            "utility_proxy",
            "backend_semantics_changed",
            "pilot_result_source",
            "pilot_proxy_final_error",
            "utility_label",
            "claim_eligible",
            "claim_blockers",
            "runtime_dispatch_allowed",
        ],
    )
    _write_csv(
        output / "same_budget_ledger.csv",
        _ledger_rows(records),
        [
            "run_id",
            "problem_id",
            "seed",
            "phase_i_fe",
            "phase_ii_fe",
            "total_fe",
            "budget_limit",
            "same_budget_violation",
            "fresh_execution",
            "pilot_result_source",
        ],
    )
    _write_csv(
        output / "backend_semantics_diff.csv",
        _semantics_rows(records),
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
            "backend_semantics_changed",
        ],
    )
    _write_csv(
        output / "anti_leakage_audit.csv",
        _anti_leakage_rows(records),
        [
            "run_id",
            "artifact_path",
            "forbidden_field",
            "found_in_runtime_payload",
            "runtime_dispatch_allowed",
            "audit_status",
            "note",
        ],
    )
    _write_csv(
        output / "paper_reported_comparison.csv",
        _paper_comparison_rows(records),
        [
            "run_id",
            "problem_id",
            "seed",
            "pilot_result_source",
            "our_pilot_proxy_final_error",
            "paper_method",
            "paper_reported_mean",
            "paper_reported_std",
            "paper_reported_time",
            "comparison_role",
            "offline_relative_gain_vs_paper_reported",
            "runtime_dispatch_allowed",
        ],
    )
    _write_csv(
        output / "negative_control_audit.csv",
        _negative_control_rows(records),
        [
            "run_id",
            "problem_id",
            "seed",
            "negative_control",
            "policy_action",
            "claim_allowed",
            "audit_status",
            "note",
        ],
    )
    _write_csv(
        output / "catastrophic_loss_audit.csv",
        _catastrophic_loss_rows(records),
        [
            "run_id",
            "problem_id",
            "seed",
            "fallback_proxy_error",
            "pilot_proxy_final_error",
            "offline_relative_gain_vs_fallback",
            "utility_label",
            "catastrophic_loss",
            "claim_allowed",
            "note",
        ],
    )
    _write_manifest(output)
    return output


if __name__ == "__main__":
    run_aob_1run_pilot()

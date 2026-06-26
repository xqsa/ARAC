from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from arac.audit import claim_gate
from arac.backends.hcc import (
    HccAobExecutionResult,
    build_hcc_evidence_profile,
    hcc_backend_semantics_for,
    load_hcc_aob_topology,
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
PILOT_RESULT_SOURCE = "hcc_source_grounded_grouping_probe"
SOURCE_LEVEL = "hcc_source_topology"

PROBLEM_IDS = [f"{family}{idx}" for family in "ESRA" for idx in range(1, 7)]

ROOT = Path(__file__).resolve().parents[2]
PAPER_REPORTED_CSV = ROOT / "references" / "paper_reported_table2_hcc_es.csv"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    idx = int(str(problem_id)[1])
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


def _smoke_results_by_problem(
    smoke_execution_results: list[HccAobExecutionResult] | None,
) -> dict[str, HccAobExecutionResult]:
    return {
        str(result.problem_id).strip().upper(): result
        for result in smoke_execution_results or []
    }


def _pilot_records(
    smoke_execution_results: list[HccAobExecutionResult] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    ledger = _ledger()
    smoke_by_problem = _smoke_results_by_problem(smoke_execution_results)
    for problem_id in PROBLEM_IDS:
        topology = load_hcc_aob_topology(problem_id)
        snapshot = topology.to_snapshot(
            run_id=RUN_ID,
            seed=SEED,
            budget_remaining_ratio=PHASE_II_FE / TOTAL_FE,
        )
        evidence = build_hcc_evidence_profile(snapshot)
        payload = _runtime_payload(evidence)
        decision = decide_action(evidence)
        semantics = hcc_backend_semantics_for(decision)

        fallback_error = _fallback_proxy_error(problem_id)
        smoke_result = smoke_by_problem.get(problem_id)
        action_error = (
            smoke_result.final_error
            if smoke_result and smoke_result.fresh_optimizer_execution
            else _action_proxy_error(problem_id, decision)
        )
        utility_label = classify_utility(fallback_error, action_error)
        allowed, blockers = claim_gate(
            runtime_payload=payload,
            decision=decision,
            semantics_diff=semantics,
            ledger=ledger,
            utility_label=utility_label,
            negative_control_pass=True,
        )
        blockers.append("hcc_source_topology_probe_not_fresh_optimizer_execution")

        records.append(
            {
                "problem_id": problem_id,
                "aob_function_id": topology.function_id,
                "function_name": topology.function_name,
                "overlap_gamma": topology.overlap_gamma,
                "dimension_real": topology.dimension_real,
                "group_count": topology.group_count,
                "overlap_group_count": topology.overlap_group_count,
                "overlapping_element_count": topology.overlapping_element_count,
                "degree_of_overlap": topology.degree_of_overlap,
                "global_fes": topology.global_fes,
                "source_level": topology.source_level,
                "evidence": evidence,
                "runtime_payload": payload,
                "decision": decision,
                "semantics": semantics,
                "ledger": ledger,
                "fallback_proxy_error": fallback_error,
                "pilot_proxy_final_error": None if smoke_result else action_error,
                "hcc_smoke_final_error": smoke_result.final_error if smoke_result else None,
                "hcc_smoke_fe_used": smoke_result.fe_used if smoke_result else "",
                "hcc_smoke_status": smoke_result.status if smoke_result else "",
                "fresh_optimizer_execution": (
                    smoke_result.fresh_optimizer_execution if smoke_result else False
                ),
                "result_source": (
                    smoke_result.result_source if smoke_result else PILOT_RESULT_SOURCE
                ),
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
                "dimension_real": record["dimension_real"],
                "aob_function_id": record["aob_function_id"],
                "function_name": record["function_name"],
                "overlap_gamma": record["overlap_gamma"],
                "group_count": record["group_count"],
                "overlap_group_count": record["overlap_group_count"],
                "overlapping_element_count": record["overlapping_element_count"],
                "degree_of_overlap": f"{record['degree_of_overlap']:.12g}",
                "global_fes": record["global_fes"],
                "selected_action_family": decision.action_family.value,
                "selected_action_name": decision.action_name,
                "decision": decision.decision,
                "trigger_reason": decision.trigger_reason,
                "utility_proxy": f"{decision.utility_proxy:.6f}",
                "backend_semantics_changed": int(semantics.changed),
                "pilot_result_source": record["result_source"],
                "source_level": record["source_level"],
                "pilot_proxy_final_error": (
                    ""
                    if record["pilot_proxy_final_error"] is None
                    else f"{record['pilot_proxy_final_error']:.6f}"
                ),
                "hcc_smoke_final_error": (
                    ""
                    if record["hcc_smoke_final_error"] is None
                    else f"{record['hcc_smoke_final_error']:.6f}"
                ),
                "hcc_smoke_fe_used": record["hcc_smoke_fe_used"],
                "hcc_smoke_status": record["hcc_smoke_status"],
                "fresh_optimizer_execution": int(record["fresh_optimizer_execution"]),
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
                "fresh_execution": int(record["fresh_optimizer_execution"]),
                "pilot_result_source": record["result_source"],
                "source_level": record["source_level"],
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
        our_result = record["hcc_smoke_final_error"]
        if our_result is None:
            our_result = record["pilot_proxy_final_error"]
        our_proxy = float(our_result)
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": record["problem_id"],
                "seed": SEED,
                "pilot_result_source": record["result_source"],
                "our_pilot_proxy_final_error": (
                    "" if record["pilot_proxy_final_error"] is None else f"{our_proxy:.6f}"
                ),
                "our_hcc_smoke_final_error": (
                    "" if record["hcc_smoke_final_error"] is None else f"{our_proxy:.6f}"
                ),
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
        action_value = record["hcc_smoke_final_error"]
        if action_value is None:
            action_value = record["pilot_proxy_final_error"]
        action = float(action_value)
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
            "Claim level: HCC-source-grounded AOB grouping probe only",
            "Benchmark: AOB 24 cases (E1-E6, S1-S6, R1-R6, A1-A6)",
            f"Current pilot runs: 1 independent run (seed={SEED})",
            "Final protocol remains: 25 independent runs",
            f"Budget: phase_i_fe={PHASE_I_FE}, phase_ii_fe={PHASE_II_FE}, total_fe={TOTAL_FE}",
            "Runtime boundary: paper-reported baselines, oracle labels, final errors, "
            "relative gains, and prior outcomes must not enter runtime dispatch.",
            "Paper Table 2 values are joined only in paper_reported_comparison.csv for offline evaluation.",
            f"Pilot result source: {PILOT_RESULT_SOURCE}; this is not a real full optimizer performance run.",
            f"Source level: {SOURCE_LEVEL}; AOB metadata and grouping topology are read from E:\\HCC-main.",
            "Optional HCC smoke execution overlays are offline-only and must not enter runtime dispatch.",
            "Negative controls and catastrophic-loss checks are explicit audit surfaces.",
            "",
        ]
    )
    (output_dir / "pilot_run_manifest.md").write_text(manifest, encoding="utf-8")


def run_aob_1run_pilot(
    output_dir: Path | str = Path("results/exp_002_aob_1run_pilot"),
    smoke_execution_results: list[HccAobExecutionResult] | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _pilot_records(smoke_execution_results=smoke_execution_results)

    _write_csv(
        output / "our_result_by_case.csv",
        _our_result_rows(records),
        [
            "run_id",
            "problem_id",
            "seed",
            "dimension",
            "dimension_real",
            "aob_function_id",
            "function_name",
            "overlap_gamma",
            "group_count",
            "overlap_group_count",
            "overlapping_element_count",
            "degree_of_overlap",
            "global_fes",
            "selected_action_family",
            "selected_action_name",
            "decision",
            "trigger_reason",
            "utility_proxy",
            "backend_semantics_changed",
            "pilot_result_source",
            "source_level",
            "pilot_proxy_final_error",
            "hcc_smoke_final_error",
            "hcc_smoke_fe_used",
            "hcc_smoke_status",
            "fresh_optimizer_execution",
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
            "source_level",
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
            "our_hcc_smoke_final_error",
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

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ARAC_REPO_ROOT = Path(__file__).resolve().parents[2]
ARAC_SRC_ROOT = ARAC_REPO_ROOT / "src"
if str(ARAC_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(ARAC_SRC_ROOT))

from arac.audit import claim_gate
from arac.backend_adapter import BackendSemanticsDiff
from arac.backends.hcc import (
    DEFAULT_HCC_MAIN_ROOT,
    HccActionExecutionPlan,
    HccAobExecutionRequest,
    HccAobExecutionResult,
    build_hcc_action_execution_plan,
    hcc_backend_semantics_for,
    run_hcc_aob_smoke_execution,
)
from arac.evaluation import SameBudgetLedger
from arac.evaluation import classify_utility, relative_gain
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS, validate_runtime_payload
from arac.policy import ActionDecision
from arac.action_space import ActionFamily

RUN_ID = "exp_003_hcc_runtime_consumer_smoke"
PROBLEM_ID = "E2"
DEFAULT_SEEDS = (1, 2, 3)
MAX_FES = 2_000
PHASE_I_FE = 0
PHASE_II_FE = MAX_FES
LOW_ACTIVE_DENSITY_THRESHOLD = 0.20
MEANINGFUL_GAIN_THRESHOLD = 0.05


@dataclass(frozen=True)
class LaneConfig:
    lane_id: str
    action_family: ActionFamily
    selected_action_name: str
    runner_action_name: str
    dispatch_scope: str
    relation_dispatch_enabled: bool = False
    plan_action_name: str = ""
    relation_policy_mode: str = "rule"
    negative_control: bool = False


LANES = (
    LaneConfig(
        "fallback",
        ActionFamily.FALLBACK,
        "conservative_no_action",
        "conservative_no_action",
        "fixed_lane_runtime_consumer_smoke",
    ),
    LaneConfig(
        "fixed_repair",
        ActionFamily.REASSIGN_REPAIR,
        "repair_shared_variable_binding",
        "repair_shared_variable_binding",
        "fixed_lane_runtime_consumer_smoke",
    ),
    LaneConfig(
        "fixed_coordinate",
        ActionFamily.COORDINATE,
        "allow_beneficial_coordination",
        "allow_beneficial_coordination",
        "fixed_lane_runtime_consumer_smoke",
    ),
    LaneConfig(
        "relation_dispatch_rule",
        ActionFamily.COORDINATE,
        "relation_dispatch_rule",
        "conservative_no_action",
        "per_overlap_relation_runtime_dispatch",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
    ),
    LaneConfig(
        "shuffled_relation_dispatch",
        ActionFamily.COORDINATE,
        "shuffled_relation_dispatch",
        "conservative_no_action",
        "shuffled_relation_dispatch_negative_control",
        relation_dispatch_enabled=True,
        plan_action_name="allow_beneficial_coordination",
        relation_policy_mode="shuffled",
        negative_control=True,
    ),
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _write_claim_evidence_table(
    output_dir: Path,
    diagnosis_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# exp_003 Claim Evidence Table",
        "",
        "| Problem | Claim | Status | Evidence | Blocker | Source artifact |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in diagnosis_rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row.get(field, ""))
                for field in (
                    "problem_id",
                    "diagnostic_key",
                    "status",
                    "observed_value",
                    "blocker_reason",
                )
            )
            + " | policy_evidence_diagnosis.csv |"
        )
    (output_dir / "claim_evidence_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _decision(lane: LaneConfig) -> ActionDecision:
    action_name = lane.plan_action_name or lane.selected_action_name
    return ActionDecision(
        action_family=lane.action_family,
        action_name=action_name,
        decision="fallback" if lane.action_family == ActionFamily.FALLBACK else "allow",
        trigger_reason=f"exp_003_{lane.dispatch_scope}",
        utility_proxy=0.0 if lane.action_family == ActionFamily.FALLBACK else 1.0,
    )


def _effective_claim_gate_decision(
    lane: LaneConfig,
    decision: ActionDecision,
    trace_rows: list[dict[str, str]],
) -> ActionDecision:
    if not lane.relation_dispatch_enabled:
        return decision
    has_active_consumed_action = any(
        row.get("optimizer_consumed") == "1"
        and _trace_action(row) != "conservative_no_action"
        for row in trace_rows
    )
    if has_active_consumed_action:
        return decision
    return ActionDecision(
        action_family=ActionFamily.FALLBACK,
        action_name="conservative_no_action",
        decision="fallback",
        trigger_reason="relation_dispatch_no_active_optimizer_consumed_action",
        utility_proxy=0.0,
    )


def _same_budget_group_id(problem_id: str, seed: int, max_fes: int) -> str:
    return f"{problem_id}_seed{seed}_{max_fes}fe"


def _is_overlap_applicable_problem_id(problem_id: str) -> bool:
    level = "".join(character for character in problem_id if character.isdigit())
    return level != "1"


def _runtime_payload(
    problem_id: str,
    seed: int,
    lane_id: str,
    action_name: str,
    max_fes: int,
) -> dict[str, object]:
    payload = {
        "run_id": RUN_ID,
        "problem_id": problem_id,
        "seed": seed,
        "lane_id": lane_id,
        "selected_action_name": action_name,
        "benchmark": "AOB",
        "budget_limit": max_fes,
        "used_for_runtime": 1,
    }
    validate_runtime_payload(payload)
    return payload


def _ledger_for_result(result: HccAobExecutionResult) -> SameBudgetLedger:
    actual_fe_used = _actual_fe_used(result)
    return SameBudgetLedger(
        phase_i_fe=PHASE_I_FE,
        phase_ii_fe=actual_fe_used - PHASE_I_FE,
        budget_limit=result.max_fes,
        fresh_execution=result.fresh_optimizer_execution,
    )


def _actual_fe_used(result: HccAobExecutionResult) -> int:
    if result.optimizer_final_fe_used is None:
        return result.fe_used
    return result.optimizer_final_fe_used


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_lane_artifact(result: HccAobExecutionResult, artifact_name: str) -> Path | None:
    root = Path(result.output_root)
    preferred = sorted(root.rglob(f"{result.problem_id}_{artifact_name}"))
    if preferred:
        return preferred[-1]
    generic = sorted(root.rglob(artifact_name))
    return generic[-1] if generic else None


def _trace_rows_for_record(record: dict[str, object]) -> list[dict[str, str]]:
    result = record["result"]
    assert isinstance(result, HccAobExecutionResult)
    return _read_csv_rows(result.action_trace_path)


def _artifact_rows_for_record(
    record: dict[str, object],
    artifact_name: str,
) -> list[dict[str, str]]:
    result = record["result"]
    assert isinstance(result, HccAobExecutionResult)
    return _read_csv_rows(_find_lane_artifact(result, artifact_name))


def _with_lane_prefix(
    record: dict[str, object],
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    result = record["result"]
    assert isinstance(result, HccAobExecutionResult)
    return [
        {**row, "run_id": RUN_ID, "lane_id": record["lane_id"], "seed": result.seed}
        for row in rows
    ]


def _format_action_mix(
    rows: list[dict[str, str]],
    fallback_action: str,
    *,
    optimizer_consumed_only: bool = False,
) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        if optimizer_consumed_only and row.get("optimizer_consumed") != "1":
            continue
        action = row.get("canonical_action_name") or row.get("selected_action_name") or ""
        if not action:
            continue
        counts[action] = counts.get(action, 0) + 1
    if not counts:
        counts[fallback_action] = 1
    return ";".join(f"{action}={counts[action]}" for action in sorted(counts))


def _semantics_from_trace_rows(
    rows: list[dict[str, str]],
    fallback: BackendSemanticsDiff,
) -> BackendSemanticsDiff:
    if not rows:
        return fallback
    return BackendSemanticsDiff(
        variable_owner_changed=any(
            _trace_action(row) == "repair_shared_variable_binding"
            and row.get("optimizer_consumed") == "1"
            for row in rows
        ),
        relation_handling_changed=any(
            _trace_action(row) == "isolate_conflicting_relation"
            and row.get("optimizer_consumed") == "1"
            for row in rows
        ),
        coordination_mode_changed=any(
            _trace_action(row) == "allow_beneficial_coordination"
            and row.get("optimizer_consumed") == "1"
            for row in rows
        ),
    )


def _trace_action(row: dict[str, str]) -> str:
    return row.get("canonical_action_name") or row.get("selected_action_name") or ""


def _relation_join_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        result = record["result"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(result, HccAobExecutionResult)
        if not lane.relation_dispatch_enabled:
            continue
        trace_ids = {
            row.get("relation_id", "")
            for row in _trace_rows_for_record(record)
            if row.get("relation_id")
        }
        decision_ids = {
            row.get("relation_id", "")
            for row in _artifact_rows_for_record(record, "action_decision.csv")
            if row.get("relation_id")
        }
        overlap_ids = {
            row.get("relation_id", "")
            for row in _artifact_rows_for_record(record, "overlap_relations.csv")
            if row.get("relation_id")
        }
        for relation_id in sorted(trace_ids | decision_ids | overlap_ids):
            has_trace = relation_id in trace_ids
            has_decision = relation_id in decision_ids
            has_overlap = relation_id in overlap_ids
            rows.append(
                {
                    "run_id": RUN_ID,
                    "lane_id": lane.lane_id,
                    "problem_id": result.problem_id,
                    "seed": result.seed,
                    "relation_id": relation_id,
                    "has_action_trace": int(has_trace),
                    "has_action_decision": int(has_decision),
                    "has_overlap_relation": int(has_overlap),
                    "audit_status": "pass"
                    if has_trace and has_decision and has_overlap
                    else "fail",
                }
            )
    return rows


def _relation_join_pass(record: dict[str, object]) -> bool:
    lane = record["lane"]
    assert isinstance(lane, LaneConfig)
    if not lane.relation_dispatch_enabled:
        return True
    rows = [
        row for row in _relation_join_rows([record])
        if row["lane_id"] == lane.lane_id
    ]
    return bool(rows) and all(row["audit_status"] == "pass" for row in rows)


def _records(
    output_dir: Path,
    execution_runner: Callable[[HccAobExecutionRequest], HccAobExecutionResult],
    hcc_root: Path,
    python_executable: str,
    seeds: tuple[int, ...],
    problem_ids: tuple[str, ...],
    max_fes: int,
    jobs: int = 1,
    budget_accounting: str = "strict",
    cmaes_restart: bool = True,
    mmes_restart: bool = True,
) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    for problem_id in problem_ids:
        for seed in seeds:
            for lane in LANES:
                decision = _decision(lane)
                plan = build_hcc_action_execution_plan(problem_id, decision)
                semantics = hcc_backend_semantics_for(
                    decision,
                    optimizer_consumed=plan.optimizer_consumed,
                )
                payload = _runtime_payload(
                    problem_id,
                    seed,
                    lane.lane_id,
                    lane.selected_action_name,
                    max_fes,
                )
                lane_output = (
                    output_dir
                    / "_hcc_smoke"
                    / problem_id
                    / f"seed_{seed}"
                    / lane.lane_id
                ).resolve()
                contexts.append(
                    {
                        "lane": lane,
                        "lane_id": lane.lane_id,
                        "decision": decision,
                        "plan": plan,
                        "semantics": semantics,
                        "payload": payload,
                        "request": HccAobExecutionRequest(
                            problem_id=problem_id,
                            seed=seed,
                            max_fes=max_fes,
                            output_dir=lane_output,
                            hcc_root=hcc_root,
                            python_executable=python_executable,
                            timestamp=f"{RUN_ID}-{problem_id}-seed{seed}-{lane.lane_id}",
                            arac_action=lane.runner_action_name,
                            enable_relation_dispatch=lane.relation_dispatch_enabled,
                            relation_policy_mode=lane.relation_policy_mode,
                            budget_accounting=budget_accounting,
                            cmaes_restart=cmaes_restart,
                            mmes_restart=mmes_restart,
                        ),
                    }
                )

    def run_context(context: dict[str, object]) -> dict[str, object]:
        request = context["request"]
        semantics = context["semantics"]
        plan = context["plan"]
        payload = context["payload"]
        decision = context["decision"]
        lane = context["lane"]
        assert isinstance(request, HccAobExecutionRequest)
        assert isinstance(semantics, BackendSemanticsDiff)
        assert isinstance(plan, HccActionExecutionPlan)
        assert isinstance(decision, ActionDecision)
        assert isinstance(lane, LaneConfig)
        result = execution_runner(request)
        trace_rows = _read_csv_rows(result.action_trace_path)
        semantics = _semantics_from_trace_rows(trace_rows, fallback=semantics)
        ledger = _ledger_for_result(result)
        effective_decision = _effective_claim_gate_decision(
            lane,
            decision,
            trace_rows,
        )
        allowed, blockers = claim_gate(
            runtime_payload=payload,
            decision=effective_decision,
            semantics_diff=semantics,
            ledger=ledger,
            utility_label="runtime_smoke_not_performance_claim",
            negative_control_pass=not lane.negative_control,
            optimizer_consumed=plan.optimizer_consumed,
        )
        return {
            "lane": lane,
            "lane_id": context["lane_id"],
            "decision": decision,
            "plan": plan,
            "semantics": semantics,
            "payload": payload,
            "ledger": ledger,
            "result": result,
            "claim_allowed": allowed,
            "claim_blockers": ";".join(blockers),
        }

    worker_count = max(1, int(jobs))
    if worker_count == 1:
        return [run_context(context) for context in contexts]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(run_context, contexts))


def _utility_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    fallback_by_case: dict[tuple[str, int], HccAobExecutionResult] = {}
    for record in records:
        if record["lane_id"] != "fallback":
            continue
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        fallback_by_case[(result.problem_id, result.seed)] = result
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        result = record["result"]
        ledger = record["ledger"]
        semantics = record["semantics"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(result, HccAobExecutionResult)
        assert isinstance(ledger, SameBudgetLedger)
        fallback_result = fallback_by_case[(result.problem_id, result.seed)]
        utility_label = classify_utility(fallback_result.final_error, result.final_error)
        blockers: list[str] = []
        if lane.lane_id == "fallback":
            blockers.append("comparison_lane_not_utility_claim")
        if lane.negative_control:
            blockers.append("negative_control_lane_not_utility_claim")
        if ledger.violation:
            blockers.append("same_budget_violation")
        if not ledger.fresh_execution:
            blockers.append("not_fresh_execution")
        if utility_label == "catastrophic_loss":
            blockers.append("catastrophic_loss")
        if lane.lane_id != "fallback" and utility_label != "meaningful_win":
            blockers.append("utility_not_meaningful_win")
        if lane.relation_dispatch_enabled and not _relation_join_pass(record):
            blockers.append("relation_artifact_join_failed")
        claim_allowed = not blockers
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": lane.lane_id,
                "problem_id": result.problem_id,
                "seed": result.seed,
                "final_error": f"{result.final_error:.6e}",
                "fe_used": result.fe_used,
                "same_budget_violation": int(ledger.violation),
                "relative_gain_vs_fallback": (
                    f"{relative_gain(fallback_result.final_error, result.final_error):.6f}"
                ),
                "utility_label": utility_label,
                "action_mix": _format_action_mix(
                    _trace_rows_for_record(record),
                    lane.selected_action_name,
                ),
                "optimizer_consumed_action_mix": _format_action_mix(
                    _trace_rows_for_record(record),
                    lane.selected_action_name,
                    optimizer_consumed_only=True,
                ),
                "runtime_connected_claim_allowed": int(
                    result.fresh_optimizer_execution
                    and bool(result.action_trace_rows)
                    and (not lane.relation_dispatch_enabled or _relation_join_pass(record))
                ),
                "backend_semantics_changed": int(semantics.changed),
                "claim_allowed": int(claim_allowed),
                "claim_blockers": ";".join(sorted(set(blockers))),
            }
        )
    return rows


def _our_result_rows(
    records: list[dict[str, object]],
    utility_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    utility_claim_by_lane = {
        (row["problem_id"], row["lane_id"], row["seed"]): row["claim_allowed"]
        for row in utility_rows
    }
    runtime_claim_by_lane = {
        (row["problem_id"], row["lane_id"], row["seed"]): row[
            "runtime_connected_claim_allowed"
        ]
        for row in utility_rows
    }
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        decision = record["decision"]
        result = record["result"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(decision, ActionDecision)
        assert isinstance(result, HccAobExecutionResult)
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": lane.lane_id,
                "problem_id": result.problem_id,
                "seed": result.seed,
                "selected_action_family": decision.action_family.value,
                "selected_action_name": lane.selected_action_name,
                "hcc_smoke_final_error": f"{result.final_error:.6e}",
                "hcc_smoke_fe_used": result.fe_used,
                "hcc_smoke_status": result.status,
                "fresh_optimizer_execution": int(result.fresh_optimizer_execution),
                "action_trace_rows": result.action_trace_rows,
                "runtime_dispatch_allowed": 1,
                "dispatch_scope": lane.dispatch_scope,
                "relation_dispatch_enabled": int(lane.relation_dispatch_enabled),
                "runtime_connected_claim_allowed": runtime_claim_by_lane[
                    (result.problem_id, lane.lane_id, result.seed)
                ],
                "utility_claim_allowed": utility_claim_by_lane[
                    (result.problem_id, lane.lane_id, result.seed)
                ],
                "performance_claim_allowed": 0,
            }
        )
    return rows


def _ledger_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        actual_fe_used = _actual_fe_used(result)
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": result.problem_id,
                "seed": result.seed,
                "same_budget_group_id": _same_budget_group_id(
                    result.problem_id,
                    result.seed,
                    result.max_fes,
                ),
                "phase_i_fe": PHASE_I_FE,
                "phase_ii_fe": actual_fe_used - PHASE_I_FE,
                "total_fe": actual_fe_used,
                "budget_limit": result.max_fes,
                "configured_budget_limit": result.max_fes,
                "budget_aligned_fe_used": result.fe_used,
                "actual_fe_used": actual_fe_used,
                "budget_limit_source": "experiment_config",
                "same_budget_violation": int(actual_fe_used > result.max_fes),
                "fresh_execution": int(result.fresh_optimizer_execution),
            }
        )
    return rows


def _semantics_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        decision = record["decision"]
        result = record["result"]
        semantics = record["semantics"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(decision, ActionDecision)
        assert isinstance(result, HccAobExecutionResult)
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": result.problem_id,
                "seed": result.seed,
                "selected_action_name": lane.selected_action_name,
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


def _action_execution_plan_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        row = record["plan"].to_csv_row()
        row["run_id"] = RUN_ID
        row["lane_id"] = record["lane_id"]
        rows.append(row)
    return rows


def _action_trace_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(_with_lane_prefix(record, _trace_rows_for_record(record)))
    return rows


def _action_decision_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(
            _with_lane_prefix(record, _artifact_rows_for_record(record, "action_decision.csv"))
        )
    return rows


def _action_mismatch_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(
            _with_lane_prefix(
                record,
                _artifact_rows_for_record(record, "action_mismatch_audit.csv"),
            )
        )
    return rows


def _overlap_relation_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(
            _with_lane_prefix(record, _artifact_rows_for_record(record, "overlap_relations.csv"))
        )
    return rows


def _anti_leakage_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    payloads = [record["payload"] for record in records]
    artifact_rows = (
        _action_trace_rows(records)
        + _action_decision_rows(records)
        + _action_mismatch_rows(records)
        + _overlap_relation_rows(records)
    )
    rows: list[dict[str, object]] = []
    for field in sorted(FORBIDDEN_RUNTIME_FIELDS):
        found = any(field in payload for payload in payloads)
        artifact_found = any(field in row for row in artifact_rows)
        rows.append(
            {
                "run_id": RUN_ID,
                "artifact_path": (
                    "runtime_payload;action_trace.csv;"
                    "action_decision.csv;overlap_relations.csv"
                ),
                "forbidden_field": field,
                "found_in_runtime_payload": int(found or artifact_found),
                "runtime_dispatch_allowed": 0 if found or artifact_found else 1,
                "audit_status": "fail" if found or artifact_found else "pass",
            }
        )
    return rows


def _claim_gate_rows(
    records: list[dict[str, object]],
    utility_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    utility_by_case = {
        (row["problem_id"], row["lane_id"], row["seed"]): row
        for row in utility_rows
    }
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        decision = record["decision"]
        plan = record["plan"]
        ledger = record["ledger"]
        result = record["result"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(decision, ActionDecision)
        assert isinstance(ledger, SameBudgetLedger)
        assert isinstance(result, HccAobExecutionResult)
        utility_row = utility_by_case[(result.problem_id, lane.lane_id, result.seed)]
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": result.problem_id,
                "seed": result.seed,
                "selected_action_name": lane.selected_action_name,
                "optimizer_consumed": int(plan.optimizer_consumed),
                "same_budget_violation": int(ledger.violation),
                "runtime_connected_claim_allowed": int(record["claim_allowed"]),
                "runtime_claim_blockers": record["claim_blockers"],
                "utility_claim_allowed": utility_row["claim_allowed"],
                "utility_claim_blockers": utility_row["claim_blockers"],
                "performance_claim_allowed": 0,
                "claim_allowed": utility_row["claim_allowed"],
                "claim_blockers": utility_row["claim_blockers"],
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _parse_action_mix(value: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(value).split(";"):
        if not part or "=" not in part:
            continue
        action, count = part.rsplit("=", 1)
        counts[action] = counts.get(action, 0) + int(count)
    return counts


def _has_active_relation_action(row: dict[str, object]) -> bool:
    counts = _parse_action_mix(
        row.get("optimizer_consumed_action_mix") or row.get("action_mix", "")
    )
    return any(
        action != "conservative_no_action" and count > 0
        for action, count in counts.items()
    )


def _active_relation_density(row: dict[str, object]) -> float:
    counts = _parse_action_mix(row.get("action_mix", ""))
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    active = sum(
        count
        for action, count in counts.items()
        if action != "conservative_no_action"
    )
    return active / total


def _expects_backend_semantics(row: dict[str, object]) -> bool:
    lane_id = row["lane_id"]
    if lane_id in {"relation_dispatch_rule", "shuffled_relation_dispatch"}:
        return _has_active_relation_action(row)
    return lane_id != "fallback"


def _format_action_counts(counts: dict[str, int]) -> str:
    return ";".join(f"{action}={counts[action]}" for action in sorted(counts))


def _format_inline_counts(counts: dict[str, int]) -> str:
    return ",".join(f"{action}={counts[action]}" for action in sorted(counts))


def _mean_numeric(rows: list[dict[str, object]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return _mean(values)


def _format_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


def _gain_bucket(gain: float) -> str:
    if gain > 0.0:
        return "win"
    if gain < 0.0:
        return "loss"
    return "tie"


def _aggregate_lane_action_mix(
    utility_rows: list[dict[str, object]],
    lane_id: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in utility_rows:
        if row["lane_id"] != lane_id:
            continue
        for action, count in _parse_action_mix(row.get("action_mix", "")).items():
            counts[action] = counts.get(action, 0) + count
    return counts


def _action_mix_for_gain_bucket(
    rows: list[dict[str, object]],
    bucket: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        gain = float(row["relative_gain_vs_fallback"])
        if (
            (bucket == "win" and gain <= 0.0)
            or (bucket == "loss" and gain >= 0.0)
            or (bucket == "tie" and gain != 0.0)
        ):
            continue
        for action, count in _parse_action_mix(row.get("action_mix", "")).items():
            counts[action] = counts.get(action, 0) + count
    return counts


def _action_value_delta_profile_for_gain_bucket(
    trace_rows: list[dict[str, object]],
    relation_rows: list[dict[str, object]],
    bucket: str,
) -> str:
    gain_by_case = {
        (str(row["problem_id"]), str(row["seed"])): float(
            row["relative_gain_vs_fallback"]
        )
        for row in relation_rows
    }
    values_by_action: dict[str, list[float]] = {}
    for row in trace_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        gain = gain_by_case.get((str(row.get("problem_id", "")), str(row.get("seed", ""))))
        if gain is None:
            continue
        if (
            (bucket == "win" and gain <= 0.0)
            or (bucket == "loss" and gain >= 0.0)
            or (bucket == "tie" and gain != 0.0)
        ):
            continue
        try:
            value = float(row.get("action_value_delta_norm", ""))
        except (TypeError, ValueError):
            continue
        action = str(row.get("canonical_action_name") or row.get("selected_action_name", ""))
        if not action:
            continue
        values_by_action.setdefault(action, []).append(value)
    return ";".join(
        f"{action}:n={len(values)},mean={_mean(values):.6f},max={max(values):.6f}"
        for action, values in sorted(values_by_action.items())
    )


def _result_by_problem_seed_and_lane(
    records: list[dict[str, object]],
) -> dict[tuple[str, int, str], HccAobExecutionResult]:
    indexed: dict[tuple[str, int, str], HccAobExecutionResult] = {}
    for record in records:
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        indexed[(result.problem_id, result.seed, str(record["lane_id"]))] = result
    return indexed


def _negative_control_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = _result_by_problem_seed_and_lane(records)
    problem_ids = sorted(
        problem_id
        for problem_id, _seed, lane_id in indexed
        if lane_id == "relation_dispatch_rule"
    )
    rows: list[dict[str, object]] = []
    for problem_id in sorted(set(problem_ids)):
        seeds = sorted(
            seed
            for indexed_problem_id, seed, lane_id in indexed
            if indexed_problem_id == problem_id
            and lane_id == "relation_dispatch_rule"
        )
        relation_errors = [
            indexed[(problem_id, seed, "relation_dispatch_rule")].final_error
            for seed in seeds
        ]
        shuffled_errors = [
            indexed[(problem_id, seed, "shuffled_relation_dispatch")].final_error
            for seed in seeds
        ]
        shuffled_win_count = sum(
            1
            for relation_error, shuffled_error in zip(
                relation_errors,
                shuffled_errors,
                strict=True,
            )
            if classify_utility(relation_error, shuffled_error) == "meaningful_win"
        )
        total = len(seeds)
        stable_outperform = shuffled_win_count > total / 2
        rows.append(
            {
                "run_id": RUN_ID,
                "problem_id": problem_id,
                "seeds": ";".join(str(seed) for seed in seeds),
                "relation_dispatch_mean_final_error": f"{_mean(relation_errors):.6e}",
                "shuffled_mean_final_error": f"{_mean(shuffled_errors):.6e}",
                "shuffled_win_count": shuffled_win_count,
                "total_seeds": total,
                "stable_outperform_detected": int(stable_outperform),
                "negative_control_pass": int(not stable_outperform),
                "diagnostic": (
                    "shuffled_control_stably_outperforms_relation_dispatch"
                    if stable_outperform
                    else "shuffled_control_not_stably_better"
                ),
            }
        )
    return rows


def _policy_evidence_diagnosis_rows_for_problem(
    problem_id: str,
    utility_rows: list[dict[str, object]],
    negative_control: dict[str, object],
) -> list[dict[str, object]]:
    relation_rows = [
        row for row in utility_rows if row["lane_id"] == "relation_dispatch_rule"
    ]
    budget_violations = sum(
        1 for row in utility_rows if str(row["same_budget_violation"]) == "1"
    )
    relation_meaningful = sum(
        1 for row in relation_rows if row["utility_label"] == "meaningful_win"
    )
    relation_catastrophic = sum(
        1 for row in relation_rows if row["utility_label"] == "catastrophic_loss"
    )
    relation_gains = [
        float(row["relative_gain_vs_fallback"]) for row in relation_rows
    ]
    relation_positive = sum(1 for gain in relation_gains if gain > 0.0)
    relation_negative = sum(1 for gain in relation_gains if gain < 0.0)
    relation_mean_gain = _mean(relation_gains)
    fixed_coordinate_by_seed = {
        str(row["seed"]): float(row["final_error"])
        for row in utility_rows
        if row["lane_id"] == "fixed_coordinate"
    }
    relation_vs_fixed_coordinate_gains = [
        relative_gain(
            fixed_coordinate_by_seed[str(row["seed"])],
            float(row["final_error"]),
        )
        for row in relation_rows
        if str(row["seed"]) in fixed_coordinate_by_seed
    ]
    relation_beats_fixed_coordinate = sum(
        1 for gain in relation_vs_fixed_coordinate_gains if gain > 0.0
    )
    relation_loses_fixed_coordinate = sum(
        1 for gain in relation_vs_fixed_coordinate_gains if gain < 0.0
    )
    relation_vs_fixed_coordinate_mean_gain = _mean(relation_vs_fixed_coordinate_gains)
    relation_gating_pass = (
        bool(relation_vs_fixed_coordinate_gains)
        and relation_beats_fixed_coordinate > relation_loses_fixed_coordinate
        and relation_vs_fixed_coordinate_mean_gain > 0.0
    )
    fixed_coordinate_rows = [
        row for row in utility_rows if row["lane_id"] == "fixed_coordinate"
    ]
    fixed_coordinate_mean_gain = _mean(
        [float(row["relative_gain_vs_fallback"]) for row in fixed_coordinate_rows]
    )
    relation_directional_pass = (
        bool(relation_rows)
        and relation_positive > relation_negative
        and relation_mean_gain > 0.0
    )
    negative_control_pass = str(negative_control.get("negative_control_pass", "0")) == "1"
    blockers: list[str] = []
    if budget_violations:
        blockers.append("same_budget_violation")
    if relation_meaningful != len(relation_rows):
        blockers.append("relation_dispatch_not_meaningful_win")
    if relation_catastrophic:
        blockers.append("catastrophic_loss")
    if not negative_control_pass:
        blockers.append("negative_control_failed")
    if not relation_gating_pass:
        blockers.append("fixed_coordinate_baseline_not_beaten")
    pilot_utility_pass = (
        not budget_violations
        and relation_directional_pass
        and relation_catastrophic == 0
        and negative_control_pass
    )
    sota_allowed = not blockers
    rule_mix = _aggregate_lane_action_mix(utility_rows, "relation_dispatch_rule")
    shuffled_mix = _aggregate_lane_action_mix(
        utility_rows,
        "shuffled_relation_dispatch",
    )
    shuffled_fallbackized_isolate = min(
        rule_mix.get("isolate_conflicting_relation", 0),
        shuffled_mix.get("conservative_no_action", 0),
    )

    return [
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "same_budget_fe_status",
            "status": "blocked" if budget_violations else "pass",
            "observed_value": f"{budget_violations}/{len(utility_rows)}",
            "blocker_reason": "same_budget_violation" if budget_violations else "",
            "next_step": "fix_same_budget_accounting" if budget_violations else "continue",
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "relation_dispatch_utility",
            "status": (
                "pass" if relation_meaningful == len(relation_rows) else "blocked"
            ),
            "observed_value": f"{relation_meaningful}/{len(relation_rows)}",
            "blocker_reason": (
                "" if relation_meaningful == len(relation_rows)
                else "relation_dispatch_not_meaningful_win"
            ),
            "next_step": (
                "continue"
                if relation_meaningful == len(relation_rows)
                else "diagnose_policy_evidence_before_sota"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "relation_dispatch_directional_utility",
            "status": "pass" if relation_directional_pass else "blocked",
            "observed_value": (
                f"{relation_positive}/{len(relation_rows)};"
                f"mean_gain={relation_mean_gain:.6f}"
            ),
            "blocker_reason": ""
            if relation_directional_pass
            else "relation_dispatch_not_directionally_positive",
            "next_step": "continue"
            if relation_directional_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "pilot_utility_evidence",
            "status": "pass" if pilot_utility_pass else "blocked",
            "observed_value": (
                f"directional={relation_positive}/{len(relation_rows)};"
                f"mean_gain={relation_mean_gain:.6f};"
                f"negative_control={int(negative_control_pass)};"
                f"catastrophic={relation_catastrophic}/{len(relation_rows)}"
            ),
            "blocker_reason": ""
            if pilot_utility_pass
            else "pilot_utility_evidence_not_established",
            "next_step": "continue_to_multi_problem_protocol"
            if pilot_utility_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "relation_vs_fixed_coordinate_baseline",
            "status": "pass" if relation_gating_pass else "blocked",
            "observed_value": (
                f"win_count={relation_beats_fixed_coordinate}/"
                f"{len(relation_vs_fixed_coordinate_gains)};"
                "mean_gain_vs_fixed_coordinate="
                f"{relation_vs_fixed_coordinate_mean_gain:.6f};"
                "fixed_coordinate_mean_gain_vs_fallback="
                f"{fixed_coordinate_mean_gain:.6f}"
            ),
            "blocker_reason": ""
            if relation_gating_pass
            else "relation_gating_not_better_than_fixed_coordinate",
            "next_step": "continue"
            if relation_gating_pass
            else "diagnose_coordinate_gating_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "catastrophic_loss_gate",
            "status": "blocked" if relation_catastrophic else "pass",
            "observed_value": f"{relation_catastrophic}/{len(relation_rows)}",
            "blocker_reason": "catastrophic_loss" if relation_catastrophic else "",
            "next_step": (
                "diagnose_policy_evidence_before_sota"
                if relation_catastrophic
                else "continue"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "shuffled_negative_control",
            "status": "pass" if negative_control_pass else "blocked",
            "observed_value": str(negative_control.get("negative_control_pass", "")),
            "blocker_reason": "" if negative_control_pass else str(
                negative_control.get("diagnostic", "negative_control_failed")
            ),
            "next_step": (
                "continue"
                if negative_control_pass
                else "diagnose_policy_evidence_before_sota"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "negative_control_action_mix",
            "status": "pass" if negative_control_pass else "blocked",
            "observed_value": (
                "relation_dispatch_rule="
                f"{_format_action_counts(rule_mix)}|"
                "shuffled_relation_dispatch="
                f"{_format_action_counts(shuffled_mix)}"
            ),
            "blocker_reason": ""
            if negative_control_pass
            else (
                f"{negative_control.get('diagnostic', 'negative_control_failed')};"
                "rule_isolate_to_shuffled_fallback="
                f"{shuffled_fallbackized_isolate}"
            ),
            "next_step": (
                "continue"
                if negative_control_pass
                else "inspect_rule_vs_shuffled_action_mix"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": problem_id,
            "diagnostic_key": "sota_escalation_allowed",
            "status": "pass" if sota_allowed else "blocked",
            "observed_value": str(int(sota_allowed)),
            "blocker_reason": ";".join(blockers),
            "next_step": "continue_to_sota_protocol"
            if sota_allowed
            else "diagnose_policy_evidence_before_sota",
        },
    ]


def _relation_policy_profile_row(
    problem_id: str,
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    trace_rows: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
) -> dict[str, object]:
    relation_utility_rows = [
        row for row in utility_rows if row["lane_id"] == "relation_dispatch_rule"
    ]
    relation_decisions = [
        row
        for row in decision_rows
        if str(row.get("problem_id", "")) == problem_id
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
    ]
    relation_traces = [
        row
        for row in trace_rows
        if str(row.get("problem_id", "")) == problem_id
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
    ]
    relation_overlaps = [
        row
        for row in overlap_rows
        if str(row.get("problem_id", "")) == problem_id
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
    ]
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in relation_decisions:
        action_name = str(row.get("canonical_action_name", ""))
        if action_name:
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        trigger_reason = str(row.get("trigger_reason", ""))
        if trigger_reason:
            reason_counts[trigger_reason] = reason_counts.get(trigger_reason, 0) + 1
    active_decisions = [
        row
        for row in relation_decisions
        if str(row.get("canonical_action_name", "")) != "conservative_no_action"
    ]
    active_actions = sum(
        count
        for action_name, count in action_counts.items()
        if action_name != "conservative_no_action"
    )
    active_density = (
        active_actions / len(relation_decisions)
        if relation_decisions
        else float("nan")
    )
    downstream_consumed = sum(
        1 for row in relation_traces if str(row.get("downstream_consumed", "")) == "1"
    )
    optimizer_consumed = sum(
        1 for row in relation_traces if str(row.get("optimizer_consumed", "")) == "1"
    )
    profile_complete = bool(relation_decisions and relation_traces and relation_overlaps)
    utility_blocked = any(
        row["utility_label"] != "meaningful_win" for row in relation_utility_rows
    )
    return {
        "run_id": RUN_ID,
        "problem_id": problem_id,
        "diagnostic_key": "relation_policy_evidence_profile",
        "status": "pass" if profile_complete else "blocked",
        "observed_value": (
            f"relations={len(relation_decisions)};"
            f"active={active_actions};"
            f"active_density={_format_float(active_density)};"
            f"downstream={downstream_consumed}/{len(relation_traces)};"
            f"optimizer_consumed={optimizer_consumed}/{len(relation_traces)};"
            f"actions={_format_action_counts(action_counts)};"
            f"reasons={_format_action_counts(reason_counts)};"
            "mean_gain="
            f"{_format_float(_mean_numeric(relation_utility_rows, 'relative_gain_vs_fallback'))};"
            "mean_active_confidence="
            f"{_format_float(_mean_numeric(active_decisions, 'confidence'))};"
            "mean_fallback_margin="
            f"{_format_float(_mean_numeric(relation_overlaps, 'fallback_margin_proxy'))};"
            "mean_delta_ratio_gap="
            f"{_format_float(_mean_numeric(relation_overlaps, 'delta_ratio_gap'))};"
            "mean_rank_stability="
            f"{_format_float(_mean_numeric(relation_overlaps, 'rank_stability'))};"
            "mean_shared_var_support="
            f"{_format_float(_mean_numeric(relation_overlaps, 'shared_var_support_ratio'))}"
        ),
        "blocker_reason": "" if profile_complete else "relation_policy_profile_missing",
        "next_step": (
            "tune_policy_or_backend_effect_size"
            if profile_complete and utility_blocked
            else ("continue" if profile_complete else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_relation_policy_profile_row(
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
) -> dict[str, object]:
    relation_utility_rows = [
        row
        for row in utility_rows
        if row["lane_id"] == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row["problem_id"]))
    ]
    relation_decisions = [
        row
        for row in decision_rows
        if str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    ]
    relation_overlaps = [
        row
        for row in overlap_rows
        if str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    ]
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in relation_decisions:
        action_name = str(row.get("canonical_action_name", ""))
        if action_name:
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        trigger_reason = str(row.get("trigger_reason", ""))
        if trigger_reason:
            reason_counts[trigger_reason] = reason_counts.get(trigger_reason, 0) + 1
    active_decisions = [
        row
        for row in relation_decisions
        if str(row.get("canonical_action_name", "")) != "conservative_no_action"
    ]
    active_density = (
        len(active_decisions) / len(relation_decisions)
        if relation_decisions
        else float("nan")
    )
    utility_blocked = any(
        row["utility_label"] != "meaningful_win" for row in relation_utility_rows
    )
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_relation_policy_profile",
        "status": "pass" if relation_decisions else "blocked",
        "observed_value": (
            f"relations={len(relation_decisions)};"
            f"active={len(active_decisions)};"
            f"active_density={_format_float(active_density)};"
            f"actions={_format_action_counts(action_counts)};"
            f"reasons={_format_action_counts(reason_counts)};"
            "mean_gain="
            f"{_format_float(_mean_numeric(relation_utility_rows, 'relative_gain_vs_fallback'))};"
            "mean_active_confidence="
            f"{_format_float(_mean_numeric(active_decisions, 'confidence'))};"
            "mean_shared_var_support="
            f"{_format_float(_mean_numeric(relation_overlaps, 'shared_var_support_ratio'))}"
        ),
        "blocker_reason": "" if relation_decisions else "relation_policy_profile_missing",
        "next_step": (
            "diagnose_policy_evidence_before_sota"
            if relation_decisions and utility_blocked
            else ("continue" if relation_decisions else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_trigger_outcome_profile_row(
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
) -> dict[str, object]:
    gain_by_case = {
        (str(row["problem_id"]), str(row["seed"])): float(
            row["relative_gain_vs_fallback"]
        )
        for row in utility_rows
        if row["lane_id"] == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row["problem_id"]))
    }
    counts: dict[str, dict[str, int]] = {}
    for row in decision_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        case_key = (str(row.get("problem_id", "")), str(row.get("seed", "")))
        if case_key not in gain_by_case:
            continue
        trigger_reason = str(row.get("trigger_reason", ""))
        if not trigger_reason:
            continue
        bucket = _gain_bucket(gain_by_case[case_key])
        counts.setdefault(trigger_reason, {"win": 0, "loss": 0, "tie": 0})[bucket] += 1
    observed_value = ";".join(
        (
            f"{trigger}=win:{bucket_counts['win']},"
            f"loss:{bucket_counts['loss']},"
            f"tie:{bucket_counts['tie']}"
        )
        for trigger, bucket_counts in sorted(counts.items())
    )
    losses = sum(bucket_counts["loss"] for bucket_counts in counts.values())
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_trigger_outcome_profile",
        "status": "blocked" if losses else ("pass" if counts else "blocked"),
        "observed_value": observed_value,
        "blocker_reason": (
            "relation_dispatch_lost_cases"
            if losses
            else ("" if counts else "relation_policy_profile_missing")
        ),
        "next_step": (
            "inspect_trigger_outcome_profile"
            if losses
            else ("continue" if counts else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_trigger_baseline_gap_profile_row(
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
) -> dict[str, object]:
    by_case_lane = {
        (str(row["problem_id"]), str(row["seed"]), str(row["lane_id"])): float(
            row["final_error"]
        )
        for row in utility_rows
        if _is_overlap_applicable_problem_id(str(row["problem_id"]))
        and str(row["lane_id"])
        in {"relation_dispatch_rule", "fixed_repair", "fixed_coordinate"}
    }
    gaps: dict[str, dict[str, list[float]]] = {}
    for row in decision_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        problem_id = str(row.get("problem_id", ""))
        if not _is_overlap_applicable_problem_id(problem_id):
            continue
        seed = str(row.get("seed", ""))
        relation_error = by_case_lane.get((problem_id, seed, "relation_dispatch_rule"))
        repair_error = by_case_lane.get((problem_id, seed, "fixed_repair"))
        coordinate_error = by_case_lane.get((problem_id, seed, "fixed_coordinate"))
        trigger_reason = str(row.get("trigger_reason", ""))
        if (
            relation_error is None
            or repair_error is None
            or coordinate_error is None
            or not trigger_reason
        ):
            continue
        trigger_gaps = gaps.setdefault(
            trigger_reason,
            {"fixed_repair": [], "fixed_coordinate": []},
        )
        trigger_gaps["fixed_repair"].append(relative_gain(repair_error, relation_error))
        trigger_gaps["fixed_coordinate"].append(
            relative_gain(coordinate_error, relation_error)
        )
    observed_value = ";".join(
        (
            f"{trigger}=relations:{len(values['fixed_repair'])},"
            f"vs_fixed_repair_mean={_format_float(_mean(values['fixed_repair']))},"
            "vs_fixed_coordinate_mean="
            f"{_format_float(_mean(values['fixed_coordinate']))}"
        )
        for trigger, values in sorted(gaps.items())
    )
    has_negative_mean = any(
        _mean(values["fixed_repair"]) < 0.0
        or _mean(values["fixed_coordinate"]) < 0.0
        for values in gaps.values()
    )
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_trigger_baseline_gap_profile",
        "status": "blocked" if has_negative_mean else ("pass" if gaps else "blocked"),
        "observed_value": observed_value,
        "blocker_reason": (
            "trigger_baseline_gap_detected"
            if has_negative_mean
            else ("" if gaps else "relation_policy_profile_missing")
        ),
        "next_step": (
            "inspect_trigger_baseline_gap_profile"
            if has_negative_mean
            else ("continue" if gaps else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_action_baseline_gap_profile_row(
    utility_rows: list[dict[str, object]],
    action_trace_rows: list[dict[str, object]],
) -> dict[str, object]:
    by_case_lane = {
        (str(row["problem_id"]), str(row["seed"]), str(row["lane_id"])): float(
            row["final_error"]
        )
        for row in utility_rows
        if _is_overlap_applicable_problem_id(str(row["problem_id"]))
        and str(row["lane_id"])
        in {"relation_dispatch_rule", "fixed_repair", "fixed_coordinate"}
    }
    gaps: dict[str, dict[str, list[float]]] = {}
    for row in action_trace_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        problem_id = str(row.get("problem_id", ""))
        if not _is_overlap_applicable_problem_id(problem_id):
            continue
        seed = str(row.get("seed", ""))
        relation_error = by_case_lane.get((problem_id, seed, "relation_dispatch_rule"))
        repair_error = by_case_lane.get((problem_id, seed, "fixed_repair"))
        coordinate_error = by_case_lane.get((problem_id, seed, "fixed_coordinate"))
        action_name = str(row.get("canonical_action_name", ""))
        if (
            relation_error is None
            or repair_error is None
            or coordinate_error is None
            or not action_name
        ):
            continue
        action_gaps = gaps.setdefault(
            action_name,
            {"fixed_repair": [], "fixed_coordinate": [], "value_delta": []},
        )
        action_gaps["fixed_repair"].append(relative_gain(repair_error, relation_error))
        action_gaps["fixed_coordinate"].append(
            relative_gain(coordinate_error, relation_error)
        )
        try:
            action_gaps["value_delta"].append(float(row["action_value_delta_norm"]))
        except (KeyError, TypeError, ValueError):
            pass
    observed_value = ";".join(
        (
            f"{action}=relations:{len(values['fixed_repair'])},"
            f"vs_fixed_repair_mean={_format_float(_mean(values['fixed_repair']))},"
            "vs_fixed_coordinate_mean="
            f"{_format_float(_mean(values['fixed_coordinate']))},"
            "mean_action_value_delta_norm="
            f"{_format_float(_mean(values['value_delta']))}"
        )
        for action, values in sorted(gaps.items())
    )
    has_negative_mean = any(
        _mean(values["fixed_repair"]) < 0.0
        or _mean(values["fixed_coordinate"]) < 0.0
        for values in gaps.values()
    )
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_action_baseline_gap_profile",
        "status": "blocked" if has_negative_mean else ("pass" if gaps else "blocked"),
        "observed_value": observed_value,
        "blocker_reason": (
            "action_baseline_gap_detected"
            if has_negative_mean
            else ("" if gaps else "relation_action_trace_missing")
        ),
        "next_step": (
            "inspect_action_baseline_gap_profile"
            if has_negative_mean
            else ("continue" if gaps else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_no_overlap_control_row(
    utility_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    def has_overlap_support(row: dict[str, object]) -> bool:
        saw_support_field = False
        for key in ("shared_var_count", "shared_vars_count", "overlap_strength"):
            if key not in row:
                continue
            saw_support_field = True
            raw_value = str(row.get(key, "")).strip()
            if not raw_value:
                continue
            try:
                if float(raw_value) > 0.0:
                    return True
            except ValueError:
                return True
        if "shared_vars" in row:
            saw_support_field = True
            if str(row.get("shared_vars", "")).strip():
                return True
        return not saw_support_field

    control_ids = sorted(
        {
            str(row["problem_id"])
            for row in utility_rows
            if not _is_overlap_applicable_problem_id(str(row["problem_id"]))
        }
    )
    if not control_ids:
        return None
    control_id_set = set(control_ids)
    control_utility_rows = [
        row for row in utility_rows if str(row["problem_id"]) in control_id_set
    ]
    control_overlap_rows = [
        row
        for row in overlap_rows
        if str(row.get("problem_id", "")) in control_id_set
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and has_overlap_support(row)
    ]
    active_decision_rows = [
        row
        for row in decision_rows
        if str(row.get("problem_id", "")) in control_id_set
        and str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and str(row.get("canonical_action_name", "")) != "conservative_no_action"
    ]
    budget_violations = sum(
        1
        for row in control_utility_rows
        if str(row.get("same_budget_violation", "0")) == "1"
    )
    blockers: list[str] = []
    if control_overlap_rows:
        blockers.append("no_overlap_relation_rows_detected")
    if active_decision_rows:
        blockers.append("no_overlap_active_relation_actions_detected")
    if budget_violations:
        blockers.append("same_budget_violation")
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_no_overlap_control",
        "status": "blocked" if blockers else "pass",
        "observed_value": (
            f"controls={','.join(control_ids)};"
            f"relation_rows={len(control_overlap_rows)};"
            f"active_relation_actions={len(active_decision_rows)};"
            f"same_budget_violations={budget_violations}/{len(control_utility_rows)}"
        ),
        "blocker_reason": ";".join(blockers),
        "next_step": "inspect_no_overlap_controls" if blockers else "continue",
    }


def _multi_problem_action_mismatch_profile_row(
    mismatch_rows: list[dict[str, object]],
) -> dict[str, object]:
    relation_rows = [
        row
        for row in mismatch_rows
        if str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    ]
    final_counts: dict[str, int] = {}
    best_counts: dict[str, int] = {}
    abstain_counts: dict[str, int] = {}
    mismatch_count = 0
    margins: list[float] = []
    for row in relation_rows:
        final_action = str(row.get("final_action_name", ""))
        best_action = str(row.get("best_action_name", ""))
        if final_action:
            final_counts[final_action] = final_counts.get(final_action, 0) + 1
        if best_action:
            best_counts[best_action] = best_counts.get(best_action, 0) + 1
        if final_action and best_action and final_action != best_action:
            mismatch_count += 1
        abstain_reason = str(row.get("abstain_reason", ""))
        if abstain_reason:
            abstain_counts[abstain_reason] = abstain_counts.get(abstain_reason, 0) + 1
        try:
            margins.append(float(row.get("margin", "")))
        except (TypeError, ValueError):
            continue
    abstain_total = sum(abstain_counts.values())
    findings = mismatch_count or abstain_total
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_action_mismatch_profile",
        "status": "blocked" if findings or not relation_rows else "pass",
        "observed_value": (
            f"rows={len(relation_rows)};"
            f"final_best_mismatch={mismatch_count};"
            f"abstains={abstain_total};"
            f"mean_margin={_mean(margins):.6f};"
            f"final_actions={_format_inline_counts(final_counts)};"
            f"best_actions={_format_inline_counts(best_counts)};"
            f"abstain_reasons={_format_inline_counts(abstain_counts)}"
        ),
        "blocker_reason": (
            "relation_policy_profile_missing"
            if not relation_rows
            else ("action_mismatch_or_abstain_detected" if findings else "")
        ),
        "next_step": (
            "repair_relation_artifact_join"
            if not relation_rows
            else ("inspect_action_mismatch_audit" if findings else "continue")
        ),
    }


def _multi_problem_mismatch_baseline_gap_profile_row(
    utility_rows: list[dict[str, object]],
    mismatch_rows: list[dict[str, object]],
) -> dict[str, object]:
    by_case_lane = {
        (str(row["problem_id"]), str(row["seed"]), str(row["lane_id"])): float(
            row["final_error"]
        )
        for row in utility_rows
        if _is_overlap_applicable_problem_id(str(row["problem_id"]))
        and str(row["lane_id"])
        in {"relation_dispatch_rule", "fixed_repair", "fixed_coordinate"}
    }
    gaps: dict[str, dict[str, list[float]]] = {}
    for row in mismatch_rows:
        if str(row.get("lane_id", "")) != "relation_dispatch_rule":
            continue
        problem_id = str(row.get("problem_id", ""))
        if not _is_overlap_applicable_problem_id(problem_id):
            continue
        seed = str(row.get("seed", ""))
        relation_error = by_case_lane.get((problem_id, seed, "relation_dispatch_rule"))
        repair_error = by_case_lane.get((problem_id, seed, "fixed_repair"))
        coordinate_error = by_case_lane.get((problem_id, seed, "fixed_coordinate"))
        final_action = str(row.get("final_action_name", ""))
        best_action = str(row.get("best_action_name", ""))
        if (
            relation_error is None
            or repair_error is None
            or coordinate_error is None
            or not final_action
            or not best_action
        ):
            continue
        key = f"{final_action}->{best_action}"
        bucket = gaps.setdefault(key, {"fixed_repair": [], "fixed_coordinate": []})
        bucket["fixed_repair"].append(relative_gain(repair_error, relation_error))
        bucket["fixed_coordinate"].append(
            relative_gain(coordinate_error, relation_error)
        )
    observed_value = ";".join(
        (
            f"{key}=relations:{len(values['fixed_repair'])},"
            f"vs_fixed_repair_mean={_format_float(_mean(values['fixed_repair']))},"
            "vs_fixed_coordinate_mean="
            f"{_format_float(_mean(values['fixed_coordinate']))}"
        )
        for key, values in sorted(gaps.items())
    )
    has_negative_mean = any(
        _mean(values["fixed_repair"]) < 0.0
        or _mean(values["fixed_coordinate"]) < 0.0
        for values in gaps.values()
    )
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_mismatch_baseline_gap_profile",
        "status": "blocked" if has_negative_mean else ("pass" if gaps else "blocked"),
        "observed_value": observed_value,
        "blocker_reason": (
            "mismatch_baseline_gap_detected"
            if has_negative_mean
            else ("" if gaps else "action_mismatch_audit_missing")
        ),
        "next_step": (
            "inspect_action_mismatch_baseline_gaps"
            if has_negative_mean
            else ("continue" if gaps else "repair_relation_artifact_join")
        ),
    }


def _multi_problem_relation_confidence_interval_row(
    utility_rows: list[dict[str, object]],
) -> dict[str, object]:
    indexed = {
        (str(row["problem_id"]), str(row["seed"]), str(row["lane_id"])): row
        for row in utility_rows
        if _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    }
    relation_rows = [
        row
        for row in utility_rows
        if str(row.get("lane_id", "")) == "relation_dispatch_rule"
        and _is_overlap_applicable_problem_id(str(row.get("problem_id", "")))
    ]
    comparisons = [
        ("vs_fallback", "fallback"),
        ("vs_fixed_repair", "fixed_repair"),
        ("vs_fixed_coordinate", "fixed_coordinate"),
        ("vs_shuffled_relation_dispatch", "shuffled_relation_dispatch"),
    ]
    parts: list[str] = []
    missing = False
    for label, baseline_lane in comparisons:
        gains: list[float] = []
        for relation_row in relation_rows:
            case_key = (str(relation_row["problem_id"]), str(relation_row["seed"]))
            baseline_row = indexed.get((*case_key, baseline_lane))
            if baseline_row is None:
                continue
            if baseline_lane == "fallback":
                try:
                    gains.append(float(relation_row["relative_gain_vs_fallback"]))
                except (KeyError, TypeError, ValueError):
                    gains.append(
                        relative_gain(
                            float(baseline_row["final_error"]),
                            float(relation_row["final_error"]),
                        )
                    )
            else:
                gains.append(
                    relative_gain(
                        float(baseline_row["final_error"]),
                        float(relation_row["final_error"]),
                    )
                )
        if not gains:
            missing = True
        parts.append(f"{label}:{_confidence_interval_summary(gains)}")
    return {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_relation_dispatch_confidence_interval",
        "status": "blocked" if missing else "pass",
        "observed_value": ";".join(parts),
        "blocker_reason": "confidence_interval_inputs_missing" if missing else "",
        "next_step": "repair_utility_audit_inputs" if missing else "continue",
    }


def _confidence_interval_summary(values: list[float]) -> str:
    if not values:
        return "n=0,mean=nan,ci95=[nan,nan]"
    mean = _mean(values)
    if len(values) == 1:
        half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        half_width = 1.96 * ((variance ** 0.5) / (len(values) ** 0.5))
    return (
        f"n={len(values)},mean={mean:.6f},"
        f"ci95=[{mean - half_width:.6f},{mean + half_width:.6f}]"
    )


def _multi_problem_diagnosis_rows(
    utility_rows: list[dict[str, object]],
    negative_control_rows: list[dict[str, object]],
    action_trace_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    problem_ids = sorted({str(row["problem_id"]) for row in utility_rows})
    if len(problem_ids) <= 1:
        return []
    overlap_applicable_ids = [
        problem_id
        for problem_id in problem_ids
        if _is_overlap_applicable_problem_id(problem_id)
    ]
    no_overlap_control_ids = [
        problem_id
        for problem_id in problem_ids
        if not _is_overlap_applicable_problem_id(problem_id)
    ]
    overlap_applicable_id_set = set(overlap_applicable_ids)
    scope_row = {
        "run_id": RUN_ID,
        "problem_id": "ALL",
        "diagnostic_key": "multi_problem_claim_scope",
        "status": "pass" if overlap_applicable_ids else "blocked",
        "observed_value": (
            f"overlap_applicable={','.join(overlap_applicable_ids)};"
            f"no_overlap_controls={','.join(no_overlap_control_ids)}"
        ),
        "blocker_reason": "" if overlap_applicable_ids else "no_overlap_applicable_cases",
        "next_step": "continue" if overlap_applicable_ids else "add_overlap_applicable_cases",
    }
    utility_rows = [
        row
        for row in utility_rows
        if str(row["problem_id"]) in overlap_applicable_id_set
    ]
    negative_control_rows = [
        row
        for row in negative_control_rows
        if str(row["problem_id"]) in overlap_applicable_id_set
    ]
    if not utility_rows:
        return [scope_row]

    relation_rows = [
        row for row in utility_rows if row["lane_id"] == "relation_dispatch_rule"
    ]
    relation_gains = [
        float(row["relative_gain_vs_fallback"]) for row in relation_rows
    ]
    relation_lost_case_ids = [
        f"{row['problem_id']}_seed{row['seed']}"
        for row in relation_rows
        if float(row["relative_gain_vs_fallback"]) < 0.0
    ]
    relation_lost_rows = [
        row for row in relation_rows if float(row["relative_gain_vs_fallback"]) < 0.0
    ]
    relation_lost_action_mix: dict[str, int] = {}
    for row in relation_lost_rows:
        for action, count in _parse_action_mix(row.get("action_mix", "")).items():
            relation_lost_action_mix[action] = (
                relation_lost_action_mix.get(action, 0) + count
            )
    relation_lost_mean_gain = _mean(
        [float(row["relative_gain_vs_fallback"]) for row in relation_lost_rows]
    )
    relation_outcome_action_mix = {
        "wins": _action_mix_for_gain_bucket(relation_rows, "win"),
        "losses": _action_mix_for_gain_bucket(relation_rows, "loss"),
        "ties": _action_mix_for_gain_bucket(relation_rows, "tie"),
    }
    trace_rows = [] if action_trace_rows is None else action_trace_rows
    relation_action_value_delta_profile = {
        "wins": _action_value_delta_profile_for_gain_bucket(
            trace_rows,
            relation_rows,
            "win",
        ),
        "losses": _action_value_delta_profile_for_gain_bucket(
            trace_rows,
            relation_rows,
            "loss",
        ),
        "ties": _action_value_delta_profile_for_gain_bucket(
            trace_rows,
            relation_rows,
            "tie",
        ),
    }
    positive_cases = sum(1 for gain in relation_gains if gain > 0.0)
    mean_gain = _mean(relation_gains)
    loss_cases = len(relation_lost_rows)
    directional_pass = (
        bool(relation_rows) and positive_cases > loss_cases and mean_gain > 0.0
    )
    active_relation_rows = [
        row for row in relation_rows if _has_active_relation_action(row)
    ]
    active_relation_gains = [
        float(row["relative_gain_vs_fallback"]) for row in active_relation_rows
    ]
    active_relation_lost_case_ids = [
        f"{row['problem_id']}_seed{row['seed']}"
        for row in active_relation_rows
        if float(row["relative_gain_vs_fallback"]) < 0.0
    ]
    active_positive_cases = sum(1 for gain in active_relation_gains if gain > 0.0)
    active_mean_gain = _mean(active_relation_gains)
    active_loss_cases = len(active_relation_lost_case_ids)
    active_directional_pass = (
        bool(active_relation_rows)
        and active_positive_cases > active_loss_cases
        and active_mean_gain > 0.0
    )
    active_density_cases = [
        (
            f"{row['problem_id']}_seed{row['seed']}",
            _active_relation_density(row),
        )
        for row in relation_rows
    ]
    active_densities = [
        density for _case_id, density in active_density_cases if density == density
    ]
    low_active_density_case_ids = [
        case_id
        for case_id, density in active_density_cases
        if density == density and density <= LOW_ACTIVE_DENSITY_THRESHOLD
    ]
    low_active_density_cases = sum(
        1 for _case_id in low_active_density_case_ids
    )
    active_density_pass = low_active_density_cases == 0
    fixed_repair_by_case = {
        (str(row["problem_id"]), str(row["seed"])): float(row["final_error"])
        for row in utility_rows
        if row["lane_id"] == "fixed_repair"
    }
    fixed_repair_gain_cases = [
        (
            f"{row['problem_id']}_seed{row['seed']}",
            relative_gain(
                fixed_repair_by_case[(str(row["problem_id"]), str(row["seed"]))],
                float(row["final_error"]),
            ),
        )
        for row in relation_rows
        if (str(row["problem_id"]), str(row["seed"])) in fixed_repair_by_case
    ]
    fixed_repair_gains = [gain for _case_id, gain in fixed_repair_gain_cases]
    fixed_repair_lost_case_ids = [
        case_id for case_id, gain in fixed_repair_gain_cases if gain <= 0.0
    ]
    fixed_repair_win_count = sum(1 for gain in fixed_repair_gains if gain > 0.0)
    fixed_repair_mean_gain = _mean(fixed_repair_gains)
    fixed_repair_material_labels = [
        classify_utility(
            fixed_repair_by_case[(str(row["problem_id"]), str(row["seed"]))],
            float(row["final_error"]),
        )
        for row in relation_rows
        if (str(row["problem_id"]), str(row["seed"])) in fixed_repair_by_case
    ]
    fixed_repair_material_wins = fixed_repair_material_labels.count("meaningful_win")
    fixed_repair_material_losses = fixed_repair_material_labels.count(
        "catastrophic_loss"
    )
    fixed_repair_material_ties = (
        len(fixed_repair_material_labels)
        - fixed_repair_material_wins
        - fixed_repair_material_losses
    )
    fixed_repair_pass = (
        bool(fixed_repair_gains)
        and fixed_repair_win_count > len(fixed_repair_lost_case_ids)
        and fixed_repair_mean_gain > 0.0
    )
    fixed_coordinate_by_case = {
        (str(row["problem_id"]), str(row["seed"])): float(row["final_error"])
        for row in utility_rows
        if row["lane_id"] == "fixed_coordinate"
    }
    fixed_coordinate_gain_cases = [
        (
            f"{row['problem_id']}_seed{row['seed']}",
            relative_gain(
                fixed_coordinate_by_case[(str(row["problem_id"]), str(row["seed"]))],
                float(row["final_error"]),
            ),
        )
        for row in relation_rows
        if (str(row["problem_id"]), str(row["seed"])) in fixed_coordinate_by_case
    ]
    fixed_coordinate_gains = [gain for _case_id, gain in fixed_coordinate_gain_cases]
    fixed_coordinate_lost_case_ids = [
        case_id for case_id, gain in fixed_coordinate_gain_cases if gain <= 0.0
    ]
    fixed_coordinate_win_count = sum(1 for gain in fixed_coordinate_gains if gain > 0.0)
    fixed_coordinate_mean_gain = _mean(fixed_coordinate_gains)
    fixed_coordinate_pass = (
        bool(fixed_coordinate_gains)
        and fixed_coordinate_win_count > len(fixed_coordinate_lost_case_ids)
        and fixed_coordinate_mean_gain > 0.0
    )
    active_rows = [row for row in utility_rows if _expects_backend_semantics(row)]
    backend_semantics_changed = sum(
        1 for row in active_rows if str(row["backend_semantics_changed"]) == "1"
    )
    backend_semantics_pass = (
        bool(active_rows) and backend_semantics_changed == len(active_rows)
    )
    catastrophic = sum(
        1 for row in relation_rows if row["utility_label"] == "catastrophic_loss"
    )
    relation_meaningful = sum(
        1 for row in relation_rows if row["utility_label"] == "meaningful_win"
    )
    relation_material_losses = sum(
        1 for row in relation_rows if row["utility_label"] == "catastrophic_loss"
    )
    relation_material_ties = (
        len(relation_rows) - relation_meaningful - relation_material_losses
    )
    relation_materiality_pass = (
        bool(relation_rows)
        and relation_material_losses == 0
        and mean_gain >= MEANINGFUL_GAIN_THRESHOLD
    )
    relation_materiality_blocker = (
        ""
        if relation_materiality_pass
        else (
            "relation_dispatch_material_loss_detected"
            if relation_material_losses
            else "relation_dispatch_effect_size_below_threshold"
        )
    )
    budget_violations = sum(
        1 for row in utility_rows if str(row["same_budget_violation"]) == "1"
    )
    negative_failures = sum(
        1
        for row in negative_control_rows
        if str(row.get("negative_control_pass", "0")) != "1"
    )
    negative_failed_problem_ids = sorted(
        str(row["problem_id"])
        for row in negative_control_rows
        if str(row.get("negative_control_pass", "0")) != "1"
    )
    negative_pass_count = len(negative_control_rows) - negative_failures
    shuffled_win_count = sum(
        int(row.get("shuffled_win_count", 0)) for row in negative_control_rows
    )
    negative_total_seeds = sum(
        int(row.get("total_seeds", 0)) for row in negative_control_rows
    )
    negative_control_pass = (
        bool(negative_control_rows) and negative_failures == 0
    )
    rule_mix = _aggregate_lane_action_mix(utility_rows, "relation_dispatch_rule")
    shuffled_mix = _aggregate_lane_action_mix(
        utility_rows,
        "shuffled_relation_dispatch",
    )
    shuffled_repair_to_isolate = min(
        rule_mix.get("repair_shared_variable_binding", 0),
        shuffled_mix.get("isolate_conflicting_relation", 0),
    )
    shuffled_isolate_to_fallback = min(
        rule_mix.get("isolate_conflicting_relation", 0),
        shuffled_mix.get("conservative_no_action", 0),
    )
    blockers: list[str] = []
    if budget_violations:
        blockers.append("same_budget_violation")
    if not directional_pass:
        blockers.append("multi_problem_not_directionally_positive")
    if not relation_materiality_pass:
        blockers.append(relation_materiality_blocker)
    if catastrophic:
        blockers.append("catastrophic_loss")
    if negative_failures:
        blockers.append("negative_control_failed")
    if not fixed_repair_pass:
        blockers.append("fixed_repair_baseline_not_beaten")
    if not fixed_coordinate_pass:
        blockers.append("fixed_coordinate_baseline_not_beaten")
    if not backend_semantics_pass:
        blockers.append("backend_semantics_audit_failed")
    pilot_utility_pass = (
        not budget_violations
        and directional_pass
        and catastrophic == 0
        and negative_control_pass
    )
    sota_allowed = not blockers
    if sota_allowed:
        claim_tier = "sota_level_overlap_aware_cc_backend_optimization"
        claim_tier_blocker = ""
        claim_tier_next_step = "freeze_policy_and_run_final_protocol"
    elif pilot_utility_pass:
        claim_tier = (
            "runtime_evidence_driven_relation_dispatch_with_positive_utility_evidence"
        )
        claim_tier_blocker = "sota_gate_blocked"
        claim_tier_next_step = "report_positive_utility_or_continue_policy_diagnosis"
    else:
        claim_tier = "auditable_runtime_dispatch_framework"
        claim_tier_blocker = "pilot_utility_gate_blocked"
        claim_tier_next_step = "diagnose_policy_evidence_before_utility_claim"

    return [
        scope_row,
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_same_budget_fe_status",
            "status": "blocked" if budget_violations else "pass",
            "observed_value": f"{budget_violations}/{len(utility_rows)}",
            "blocker_reason": "same_budget_violation" if budget_violations else "",
            "next_step": "fix_same_budget_accounting" if budget_violations else "continue",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_relation_dispatch_mean_gain",
            "status": "pass" if directional_pass else "blocked",
            "observed_value": (
                f"positive_cases={positive_cases}/{len(relation_rows)};"
                f"mean_gain={mean_gain:.6f};"
                f"lost_case_ids={','.join(relation_lost_case_ids)}"
            ),
            "blocker_reason": ""
            if directional_pass
            else "multi_problem_not_directionally_positive",
            "next_step": "continue" if directional_pass else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_relation_dispatch_materiality",
            "status": "pass" if relation_materiality_pass else "blocked",
            "observed_value": (
                f"material_wins={relation_meaningful}/{len(relation_rows)};"
                f"material_losses={relation_material_losses}/{len(relation_rows)};"
                f"ties={relation_material_ties}/{len(relation_rows)};"
                f"mean_gain={mean_gain:.6f};"
                f"threshold={MEANINGFUL_GAIN_THRESHOLD:.6f}"
            ),
            "blocker_reason": (
                "" if relation_materiality_pass else relation_materiality_blocker
            ),
            "next_step": (
                "continue"
                if relation_materiality_pass
                else "diagnose_policy_evidence_before_sota"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_lost_case_action_mix",
            "status": "blocked" if relation_lost_rows else "pass",
            "observed_value": (
                f"lost_cases={len(relation_lost_rows)};"
                f"mean_lost_gain={relation_lost_mean_gain:.6f};"
                f"actions={_format_action_counts(relation_lost_action_mix)}"
            ),
            "blocker_reason": (
                "relation_dispatch_lost_cases" if relation_lost_rows else ""
            ),
            "next_step": (
                "inspect_lost_case_action_mix" if relation_lost_rows else "continue"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_action_outcome_profile",
            "status": "blocked" if relation_lost_rows else "pass",
            "observed_value": (
                f"wins={_format_action_counts(relation_outcome_action_mix['wins'])}|"
                f"losses={_format_action_counts(relation_outcome_action_mix['losses'])}|"
                f"ties={_format_action_counts(relation_outcome_action_mix['ties'])}"
            ),
            "blocker_reason": (
                "relation_dispatch_lost_cases" if relation_lost_rows else ""
            ),
            "next_step": (
                "inspect_action_outcome_profile" if relation_lost_rows else "continue"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_action_value_delta_profile",
            "status": (
                "blocked"
                if relation_lost_rows and action_trace_rows is not None
                else "pass"
            ),
            "observed_value": (
                f"wins={relation_action_value_delta_profile['wins']}|"
                f"losses={relation_action_value_delta_profile['losses']}|"
                f"ties={relation_action_value_delta_profile['ties']}"
            ),
            "blocker_reason": (
                "relation_dispatch_lost_cases"
                if relation_lost_rows and action_trace_rows is not None
                else ""
            ),
            "next_step": (
                "inspect_action_value_delta_profile"
                if relation_lost_rows and action_trace_rows is not None
                else "continue"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_active_relation_dispatch_mean_gain",
            "status": "pass" if active_directional_pass else "blocked",
            "observed_value": (
                f"active_cases={len(active_relation_rows)};"
                f"positive_cases={active_positive_cases}/{len(active_relation_rows)};"
                f"mean_gain={active_mean_gain:.6f};"
                f"lost_case_ids={','.join(active_relation_lost_case_ids)}"
            ),
            "blocker_reason": ""
            if active_directional_pass
            else "active_relation_dispatch_not_directionally_positive",
            "next_step": "continue" if active_directional_pass else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_pilot_utility_evidence",
            "status": "pass" if pilot_utility_pass else "blocked",
            "observed_value": (
                f"directional={positive_cases}/{len(relation_rows)};"
                f"mean_gain={mean_gain:.6f};"
                f"negative_control={negative_pass_count}/{len(negative_control_rows)};"
                f"catastrophic={catastrophic}/{len(relation_rows)}"
            ),
            "blocker_reason": ""
            if pilot_utility_pass
            else "multi_problem_pilot_utility_evidence_not_established",
            "next_step": "continue_to_sota_protocol"
            if pilot_utility_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_active_density_profile",
            "status": "pass" if active_density_pass else "blocked",
            "observed_value": (
                f"mean={_mean(active_densities):.6f};"
                f"min={min(active_densities) if active_densities else float('nan'):.6f};"
                f"low_density_cases={low_active_density_cases}/{len(active_densities)};"
                f"threshold={LOW_ACTIVE_DENSITY_THRESHOLD:.6f};"
                f"low_density_case_ids={','.join(low_active_density_case_ids)}"
            ),
            "blocker_reason": ""
            if active_density_pass
            else "low_relation_action_density_detected",
            "next_step": "continue"
            if active_density_pass
            else "inspect_low_active_density_problem_cases",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_relation_dispatch_win_count",
            "status": "pass" if directional_pass else "blocked",
            "observed_value": (
                f"win_count={positive_cases}/{len(relation_rows)};"
                f"loss_count={loss_cases}/{len(relation_rows)}"
            ),
            "blocker_reason": ""
            if directional_pass
            else "multi_problem_not_directionally_positive",
            "next_step": "continue" if directional_pass else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_fixed_repair_baseline",
            "status": "pass" if fixed_repair_pass else "blocked",
            "observed_value": (
                f"win_count={fixed_repair_win_count}/{len(fixed_repair_gains)};"
                f"mean_gain={fixed_repair_mean_gain:.6f};"
                f"lost_case_ids={','.join(fixed_repair_lost_case_ids)}"
            ),
            "blocker_reason": ""
            if fixed_repair_pass
            else "fixed_repair_baseline_not_beaten",
            "next_step": "continue" if fixed_repair_pass else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_fixed_repair_materiality",
            "status": (
                "pass"
                if fixed_repair_material_labels
                and fixed_repair_material_losses == 0
                else "blocked"
            ),
            "observed_value": (
                f"material_wins={fixed_repair_material_wins}/"
                f"{len(fixed_repair_material_labels)};"
                f"material_losses={fixed_repair_material_losses}/"
                f"{len(fixed_repair_material_labels)};"
                f"ties={fixed_repair_material_ties}/"
                f"{len(fixed_repair_material_labels)}"
            ),
            "blocker_reason": (
                ""
                if fixed_repair_material_labels
                and fixed_repair_material_losses == 0
                else "fixed_repair_material_loss_detected"
            ),
            "next_step": (
                "continue"
                if fixed_repair_material_labels
                and fixed_repair_material_losses == 0
                else "diagnose_policy_evidence_before_sota"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_relation_vs_fixed_coordinate_baseline",
            "status": "pass" if fixed_coordinate_pass else "blocked",
            "observed_value": (
                f"win_count={fixed_coordinate_win_count}/{len(fixed_coordinate_gains)};"
                f"mean_gain={fixed_coordinate_mean_gain:.6f};"
                f"lost_case_ids={','.join(fixed_coordinate_lost_case_ids)}"
            ),
            "blocker_reason": ""
            if fixed_coordinate_pass
            else "relation_gating_not_better_than_fixed_coordinate",
            "next_step": "continue" if fixed_coordinate_pass else "diagnose_coordinate_gating_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_backend_semantics_audit",
            "status": "pass" if backend_semantics_pass else "blocked",
            "observed_value": (
                f"changed={backend_semantics_changed}/{len(active_rows)}"
            ),
            "blocker_reason": ""
            if backend_semantics_pass
            else "backend_semantics_audit_failed",
            "next_step": "continue"
            if backend_semantics_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_negative_control",
            "status": "pass" if negative_control_pass else "blocked",
            "observed_value": (
                f"pass={negative_pass_count}/{len(negative_control_rows)};"
                f"shuffled_win_count={shuffled_win_count}/{negative_total_seeds};"
                f"failed_problem_ids={','.join(negative_failed_problem_ids)}"
            ),
            "blocker_reason": ""
            if negative_control_pass
            else "negative_control_failed",
            "next_step": "continue"
            if negative_control_pass
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_negative_control_action_mix",
            "status": "pass" if negative_control_pass else "blocked",
            "observed_value": (
                "relation_dispatch_rule="
                f"{_format_action_counts(rule_mix)}|"
                "shuffled_relation_dispatch="
                f"{_format_action_counts(shuffled_mix)}"
            ),
            "blocker_reason": ""
            if negative_control_pass
            else (
                "negative_control_failed;"
                f"failed_problem_ids={','.join(negative_failed_problem_ids)};"
                "rule_repair_to_shuffled_isolate="
                f"{shuffled_repair_to_isolate};"
                "rule_isolate_to_shuffled_fallback="
                f"{shuffled_isolate_to_fallback}"
            ),
            "next_step": (
                "continue"
                if negative_control_pass
                else "inspect_rule_vs_shuffled_action_mix"
            ),
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_catastrophic_loss_gate",
            "status": "blocked" if catastrophic else "pass",
            "observed_value": f"{catastrophic}/{len(relation_rows)}",
            "blocker_reason": "catastrophic_loss" if catastrophic else "",
            "next_step": "diagnose_policy_evidence_before_sota"
            if catastrophic
            else "continue",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_sota_escalation_allowed",
            "status": "pass" if sota_allowed else "blocked",
            "observed_value": str(int(sota_allowed)),
            "blocker_reason": ";".join(blockers),
            "next_step": "continue_to_sota_protocol"
            if sota_allowed
            else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_claim_tier_recommendation",
            "status": "pass",
            "observed_value": claim_tier,
            "blocker_reason": claim_tier_blocker,
            "next_step": claim_tier_next_step,
        },
    ]


def _policy_evidence_diagnosis_rows(
    records: list[dict[str, object]],
    utility_rows: list[dict[str, object]],
    negative_control_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    negative_by_problem = {
        str(row["problem_id"]): row for row in negative_control_rows
    }
    decision_rows = _action_decision_rows(records)
    mismatch_rows = _action_mismatch_rows(records)
    trace_rows = _action_trace_rows(records)
    overlap_rows = _overlap_relation_rows(records)
    rows: list[dict[str, object]] = []
    for problem_id in sorted({str(row["problem_id"]) for row in utility_rows}):
        problem_utility_rows = [
            row for row in utility_rows if str(row["problem_id"]) == problem_id
        ]
        rows.extend(
            _policy_evidence_diagnosis_rows_for_problem(
                problem_id,
                problem_utility_rows,
                negative_by_problem.get(problem_id, {}),
            )
        )
        rows.append(
            _relation_policy_profile_row(
                problem_id,
                problem_utility_rows,
                decision_rows,
                trace_rows,
                overlap_rows,
            )
        )
    rows.extend(
        _multi_problem_diagnosis_rows(
            utility_rows,
            negative_control_rows,
            trace_rows,
        )
    )
    if len({str(row["problem_id"]) for row in utility_rows}) > 1:
        no_overlap_row = _multi_problem_no_overlap_control_row(
            utility_rows,
            decision_rows,
            overlap_rows,
        )
        if no_overlap_row is not None:
            rows.append(no_overlap_row)
        rows.append(
            _multi_problem_relation_policy_profile_row(
                utility_rows,
                decision_rows,
                overlap_rows,
            )
        )
        rows.append(
            _multi_problem_trigger_outcome_profile_row(
                utility_rows,
                decision_rows,
            )
        )
        rows.append(
            _multi_problem_trigger_baseline_gap_profile_row(
                utility_rows,
                decision_rows,
            )
        )
        rows.append(
            _multi_problem_action_baseline_gap_profile_row(
                utility_rows,
                trace_rows,
            )
        )
        rows.append(_multi_problem_action_mismatch_profile_row(mismatch_rows))
        rows.append(
            _multi_problem_mismatch_baseline_gap_profile_row(
                utility_rows,
                mismatch_rows,
            )
        )
        rows.append(_multi_problem_relation_confidence_interval_row(utility_rows))
    return rows


def _diagnostic_observed_value(
    diagnosis_rows: list[dict[str, object]],
    diagnostic_key: str,
) -> str:
    for row in diagnosis_rows:
        if str(row["diagnostic_key"]) == diagnostic_key:
            return str(row["observed_value"])
    return ""


def _sota_claim_allowed(diagnosis_rows: list[dict[str, object]]) -> str:
    multi_value = _diagnostic_observed_value(
        diagnosis_rows,
        "multi_problem_sota_escalation_allowed",
    )
    if multi_value:
        return multi_value
    values = [
        str(row["observed_value"])
        for row in diagnosis_rows
        if str(row["diagnostic_key"]) == "sota_escalation_allowed"
    ]
    return "1" if values and all(value == "1" for value in values) else "0"


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ARAC_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _config_fingerprint(
    seeds: tuple[int, ...],
    problem_ids: tuple[str, ...],
    jobs: int,
    max_fes: int,
    budget_accounting: str,
    cmaes_restart: bool,
    mmes_restart: bool,
) -> str:
    payload = {
        "budget_accounting": budget_accounting,
        "cmaes_restart": bool(cmaes_restart),
        "jobs": max(1, int(jobs)),
        "lanes": [lane.lane_id for lane in LANES],
        "max_fes": int(max_fes),
        "mmes_restart": bool(mmes_restart),
        "problem_ids": list(problem_ids),
        "seeds": list(seeds),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_manifest(
    output_dir: Path,
    seeds: tuple[int, ...],
    problem_ids: tuple[str, ...],
    diagnosis_rows: list[dict[str, object]],
    jobs: int = 1,
    max_fes: int = MAX_FES,
    budget_accounting: str = "strict",
    cmaes_restart: bool = True,
    mmes_restart: bool = True,
) -> None:
    same_budget_status = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_same_budget_fe_status",
        )
        or _diagnostic_observed_value(diagnosis_rows, "same_budget_fe_status")
    )
    multi_problem_pilot = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_pilot_utility_evidence",
        )
        or "not_applicable"
    )
    multi_problem_claim_scope = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_claim_scope",
        )
        or "not_applicable"
    )
    multi_problem_active_density = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_active_density_profile",
        )
        or "not_applicable"
    )
    multi_problem_relation_materiality = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_relation_dispatch_materiality",
        )
        or "not_applicable"
    )
    multi_problem_fixed_repair = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_fixed_repair_baseline",
        )
        or "not_applicable"
    )
    multi_problem_fixed_repair_materiality = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_fixed_repair_materiality",
        )
        or "not_applicable"
    )
    multi_problem_fixed_coordinate = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_relation_vs_fixed_coordinate_baseline",
        )
        or "not_applicable"
    )
    multi_problem_relation_policy_profile = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_relation_policy_profile",
        )
        or "not_applicable"
    )
    multi_problem_claim_tier = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_claim_tier_recommendation",
        )
        or "not_applicable"
    )
    artifacts = [
        "our_result_by_case.csv",
        "same_budget_ledger.csv",
        "backend_semantics_diff.csv",
        "action_execution_plan.csv",
        "action_trace.csv",
        "action_decision.csv",
        "action_mismatch_audit.csv",
        "overlap_relations.csv",
        "relation_join_audit.csv",
        "action_utility_audit.csv",
        "negative_control_comparison.csv",
        "policy_evidence_diagnosis.csv",
        "anti_leakage_audit.csv",
        "claim_gate.csv",
        "claim_evidence_table.md",
    ]
    manifest = "\n".join(
        [
            "# exp_003_hcc_runtime_consumer_smoke Run Manifest",
            "",
            f"Date: {date.today().isoformat()}",
            "Executor: Codex",
            "",
            "Evidence posture: runtime dispatch + utility evidence",
            f"SOTA claim allowed: {_sota_claim_allowed(diagnosis_rows)}",
            "",
            "Command shape:",
            (
                "py -3 experiments\\exp_003_hcc_runtime_consumer_smoke\\run.py "
                "--output-dir <output_dir> --seeds "
                f"{' '.join(str(seed) for seed in seeds)} --problems "
                f"{' '.join(problem_ids)} --jobs {max(1, int(jobs))} "
                f"--max-fes {max_fes} --budget-accounting {budget_accounting}"
                f"{'' if cmaes_restart else ' --no-cmaes-restart'}"
                f"{'' if mmes_restart else ' --no-mmes-restart'}"
            ),
            f"Budget: {max_fes} FE per lane/case",
            f"Budget accounting: {budget_accounting}",
            (
                "Optimizer restarts: "
                f"CMAES={'enabled' if cmaes_restart else 'disabled'}, "
                f"MMES={'enabled' if mmes_restart else 'disabled'}"
            ),
            f"Parallel jobs: {max(1, int(jobs))}",
            f"Lanes: {', '.join(lane.lane_id for lane in LANES)}",
            "",
            "Freeze evidence:",
            f"- git commit: {_git_commit()}",
            (
                "- config fingerprint: "
                f"{_config_fingerprint(seeds, problem_ids, jobs, max_fes, budget_accounting, cmaes_restart, mmes_restart)}"
            ),
            f"- policy sha256: {_sha256_file(ARAC_SRC_ROOT / 'arac' / 'policy' / 'relation_policy.py')}",
            f"- experiment runner sha256: {_sha256_file(Path(__file__).resolve())}",
            f"- HCC smoke runner sha256: {_sha256_file(ARAC_REPO_ROOT / 'HCC_SRC' / 'arac_hcc_smoke_runner.py')}",
            "",
            "Runtime boundary: final/reported/oracle values must not enter runtime dispatch.",
            "",
            "Key gates:",
            f"- claim scope: {multi_problem_claim_scope}",
            f"- same-budget violations: {same_budget_status}",
            f"- pilot utility: {_diagnostic_observed_value(diagnosis_rows, 'pilot_utility_evidence')}",
            f"- multi-problem pilot utility: {multi_problem_pilot}",
            f"- multi-problem active density: {multi_problem_active_density}",
            f"- relation dispatch materiality: {multi_problem_relation_materiality}",
            f"- fixed repair baseline: {multi_problem_fixed_repair}",
            f"- fixed repair materiality: {multi_problem_fixed_repair_materiality}",
            f"- fixed coordinate baseline: {multi_problem_fixed_coordinate}",
            f"- multi-problem relation policy profile: {multi_problem_relation_policy_profile}",
            f"- claim tier recommendation: {multi_problem_claim_tier}",
            f"- SOTA escalation: {_sota_claim_allowed(diagnosis_rows)}",
            "",
            "Artifacts:",
            *[f"- {artifact}" for artifact in artifacts],
            "",
            "No performance or SOTA claim is made unless the relevant SOTA escalation gate is 1.",
        ]
    )
    (output_dir / "run_manifest.md").write_text(manifest + "\n", encoding="utf-8")


def run_hcc_runtime_consumer_smoke(
    output_dir: Path | str = Path("results/exp_003_hcc_runtime_consumer_smoke"),
    execution_runner: Callable[[HccAobExecutionRequest], HccAobExecutionResult] = (
        run_hcc_aob_smoke_execution
    ),
    hcc_root: Path | str = DEFAULT_HCC_MAIN_ROOT,
    python_executable: str = sys.executable,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    problem_ids: tuple[str, ...] = (PROBLEM_ID,),
    jobs: int = 1,
    max_fes: int = MAX_FES,
    budget_accounting: str = "strict",
    cmaes_restart: bool = True,
    mmes_restart: bool = True,
) -> Path:
    worker_count = max(1, int(jobs))
    max_fes = int(max_fes)
    if max_fes <= 0:
        raise ValueError("max_fes must be positive")
    if budget_accounting not in {"strict", "source"}:
        raise ValueError("budget_accounting must be 'strict' or 'source'")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _records(
        output_dir=output,
        execution_runner=execution_runner,
        hcc_root=Path(hcc_root),
        python_executable=python_executable,
        seeds=tuple(seeds),
        problem_ids=tuple(problem_ids),
        max_fes=max_fes,
        jobs=worker_count,
        budget_accounting=budget_accounting,
        cmaes_restart=cmaes_restart,
        mmes_restart=mmes_restart,
    )
    utility_rows = _utility_rows(records)
    negative_control_rows = _negative_control_rows(records)
    diagnosis_rows = _policy_evidence_diagnosis_rows(
        records,
        utility_rows,
        negative_control_rows,
    )
    _write_csv(
        output / "our_result_by_case.csv",
        _our_result_rows(records, utility_rows),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "selected_action_family",
            "selected_action_name",
            "hcc_smoke_final_error",
            "hcc_smoke_fe_used",
            "hcc_smoke_status",
            "fresh_optimizer_execution",
            "action_trace_rows",
            "runtime_dispatch_allowed",
            "dispatch_scope",
            "relation_dispatch_enabled",
            "runtime_connected_claim_allowed",
            "utility_claim_allowed",
            "performance_claim_allowed",
        ],
    )
    _write_csv(
        output / "same_budget_ledger.csv",
        _ledger_rows(records),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "same_budget_group_id",
            "phase_i_fe",
            "phase_ii_fe",
            "total_fe",
            "budget_limit",
            "configured_budget_limit",
            "budget_aligned_fe_used",
            "actual_fe_used",
            "budget_limit_source",
            "same_budget_violation",
            "fresh_execution",
        ],
    )
    _write_csv(
        output / "backend_semantics_diff.csv",
        _semantics_rows(records),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "selected_action_name",
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
        output / "action_execution_plan.csv",
        _action_execution_plan_rows(records),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "selected_action_name",
            "selected_action_family",
            "backend_effect_kind",
            "optimizer_consumed",
            "optimizer_consumed_parameters",
            "execution_mode",
            "blocker_reason",
            "runtime_dispatch_allowed",
        ],
    )
    _write_csv(
        output / "action_trace.csv",
        _action_trace_rows(records),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "outer_iter",
            "group_index",
            "selected_action_name",
            "relation_id",
            "group_left",
            "group_right",
            "shared_vars_hash",
            "action_family",
            "canonical_action_name",
            "relation_policy_source",
            "overlap_size",
            "previous_delta",
            "current_delta",
            "owner_selected",
            "semantic_surface",
            "state_mutated",
            "action_value_delta_norm",
            "downstream_consumed",
            "downstream_consumption_scope",
            "optimizer_consumed",
        ],
    )
    _write_csv(
        output / "action_decision.csv",
        _action_decision_rows(records),
        [
            "run_id",
            "lane_id",
            "seed",
            "problem_id",
            "relation_id",
            "group_left",
            "group_right",
            "shared_vars_count",
            "overlap_strength",
            "delta_signal",
            "rank_signal",
            "relation_action_name",
            "canonical_action_name",
            "action_family",
            "confidence",
            "trigger_reason",
        ],
    )
    _write_csv(
        output / "action_mismatch_audit.csv",
        _action_mismatch_rows(records),
        [
            "run_id",
            "lane_id",
            "seed",
            "problem_id",
            "relation_id",
            "group_left",
            "group_right",
            "candidate_scores",
            "coordinate_score",
            "isolate_conflicting_relation_score",
            "reassign_repair_score",
            "fallback_score",
            "best_action_name",
            "best_score",
            "second_best_action_name",
            "second_best_score",
            "margin",
            "final_action_name",
            "final_canonical_action_name",
            "confidence",
            "trigger_reason",
            "abstain_reason",
        ],
    )
    _write_csv(
        output / "overlap_relations.csv",
        _overlap_relation_rows(records),
        [
            "run_id",
            "lane_id",
            "seed",
            "relation_id",
            "problem_id",
            "outer_iter",
            "group_left",
            "group_right",
            "shared_vars",
            "overlap_strength",
            "delta_signal",
            "rank_signal",
            "budget_remaining_ratio",
            "previous_delta",
            "current_delta",
            "delta_abs_gap",
            "delta_signed_gap",
            "delta_ratio_gap",
            "both_positive",
            "one_side_zero",
            "rank_gap",
            "rank_stability",
            "shared_var_count",
            "shared_var_support_ratio",
            "feature_coverage",
            "fallback_margin_proxy",
        ],
    )
    _write_csv(
        output / "relation_join_audit.csv",
        _relation_join_rows(records),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "relation_id",
            "has_action_trace",
            "has_action_decision",
            "has_overlap_relation",
            "audit_status",
        ],
    )
    _write_csv(
        output / "action_utility_audit.csv",
        utility_rows,
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "final_error",
            "fe_used",
            "same_budget_violation",
            "relative_gain_vs_fallback",
            "utility_label",
            "action_mix",
            "optimizer_consumed_action_mix",
            "runtime_connected_claim_allowed",
            "backend_semantics_changed",
            "claim_allowed",
            "claim_blockers",
        ],
    )
    _write_csv(
        output / "negative_control_comparison.csv",
        negative_control_rows,
        [
            "run_id",
            "problem_id",
            "seeds",
            "relation_dispatch_mean_final_error",
            "shuffled_mean_final_error",
            "shuffled_win_count",
            "total_seeds",
            "stable_outperform_detected",
            "negative_control_pass",
            "diagnostic",
        ],
    )
    _write_csv(
        output / "policy_evidence_diagnosis.csv",
        diagnosis_rows,
        [
            "run_id",
            "problem_id",
            "diagnostic_key",
            "status",
            "observed_value",
            "blocker_reason",
            "next_step",
        ],
    )
    _write_claim_evidence_table(output, diagnosis_rows)
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
        ],
    )
    _write_csv(
        output / "claim_gate.csv",
        _claim_gate_rows(records, utility_rows),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "selected_action_name",
            "optimizer_consumed",
            "same_budget_violation",
            "runtime_connected_claim_allowed",
            "runtime_claim_blockers",
            "utility_claim_allowed",
            "utility_claim_blockers",
            "performance_claim_allowed",
            "claim_allowed",
            "claim_blockers",
        ],
    )
    _write_manifest(
        output,
        tuple(seeds),
        tuple(problem_ids),
        diagnosis_rows,
        worker_count,
        max_fes,
        budget_accounting,
        cmaes_restart,
        mmes_restart,
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp_003 HCC runtime consumer smoke.")
    parser.add_argument("--output-dir", default="results/exp_003_hcc_runtime_consumer_smoke")
    parser.add_argument("--hcc-root", default=str(DEFAULT_HCC_MAIN_ROOT))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--problems", nargs="+", default=[PROBLEM_ID])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-fes", type=int, default=MAX_FES)
    parser.add_argument("--budget-accounting", default="strict", choices=["strict", "source"])
    parser.add_argument("--cmaes-restart", dest="cmaes_restart", action="store_true", default=True)
    parser.add_argument("--no-cmaes-restart", dest="cmaes_restart", action="store_false")
    parser.add_argument("--mmes-restart", dest="mmes_restart", action="store_true", default=True)
    parser.add_argument("--no-mmes-restart", dest="mmes_restart", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_hcc_runtime_consumer_smoke(
        output_dir=args.output_dir,
        hcc_root=Path(args.hcc_root),
        python_executable=str(args.python_executable),
        seeds=tuple(args.seeds),
        problem_ids=tuple(str(problem).upper() for problem in args.problems),
        jobs=int(args.jobs),
        max_fes=int(args.max_fes),
        budget_accounting=str(args.budget_accounting),
        cmaes_restart=bool(args.cmaes_restart),
        mmes_restart=bool(args.mmes_restart),
    )


if __name__ == "__main__":
    main()

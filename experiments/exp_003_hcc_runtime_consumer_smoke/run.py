from __future__ import annotations

import argparse
import csv
import sys
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


def _decision(lane: LaneConfig) -> ActionDecision:
    action_name = lane.plan_action_name or lane.selected_action_name
    return ActionDecision(
        action_family=lane.action_family,
        action_name=action_name,
        decision="fallback" if lane.action_family == ActionFamily.FALLBACK else "allow",
        trigger_reason=f"exp_003_{lane.dispatch_scope}",
        utility_proxy=0.0 if lane.action_family == ActionFamily.FALLBACK else 1.0,
    )


def _same_budget_group_id(problem_id: str, seed: int) -> str:
    return f"{problem_id}_seed{seed}_{MAX_FES}fe"


def _runtime_payload(
    problem_id: str,
    seed: int,
    lane_id: str,
    action_name: str,
) -> dict[str, object]:
    payload = {
        "run_id": RUN_ID,
        "problem_id": problem_id,
        "seed": seed,
        "lane_id": lane_id,
        "selected_action_name": action_name,
        "benchmark": "AOB",
        "budget_limit": MAX_FES,
        "used_for_runtime": 1,
    }
    validate_runtime_payload(payload)
    return payload


def _ledger_for_result(result: HccAobExecutionResult) -> SameBudgetLedger:
    actual_fe_used = _actual_fe_used(result)
    return SameBudgetLedger(
        phase_i_fe=PHASE_I_FE,
        phase_ii_fe=actual_fe_used - PHASE_I_FE,
        budget_limit=MAX_FES,
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
        {"run_id": RUN_ID, "lane_id": record["lane_id"], "seed": result.seed, **row}
        for row in rows
    ]


def _format_action_mix(rows: list[dict[str, str]], fallback_action: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
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
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
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
                )
                lane_output = (
                    output_dir
                    / "_hcc_smoke"
                    / problem_id
                    / f"seed_{seed}"
                    / lane.lane_id
                ).resolve()
                result = execution_runner(
                    HccAobExecutionRequest(
                        problem_id=problem_id,
                        seed=seed,
                        max_fes=MAX_FES,
                        output_dir=lane_output,
                        hcc_root=hcc_root,
                        python_executable=python_executable,
                        timestamp=f"{RUN_ID}-{problem_id}-seed{seed}-{lane.lane_id}",
                        arac_action=lane.runner_action_name,
                        enable_relation_dispatch=lane.relation_dispatch_enabled,
                        relation_policy_mode=lane.relation_policy_mode,
                    )
                )
                semantics = _semantics_from_trace_rows(
                    _read_csv_rows(result.action_trace_path),
                    fallback=semantics,
                )
                ledger = _ledger_for_result(result)
                allowed, blockers = claim_gate(
                    runtime_payload=payload,
                    decision=decision,
                    semantics_diff=semantics,
                    ledger=ledger,
                    utility_label="runtime_smoke_not_performance_claim",
                    negative_control_pass=not lane.negative_control,
                    optimizer_consumed=plan.optimizer_consumed,
                )
                records.append(
                    {
                        "lane": lane,
                        "lane_id": lane.lane_id,
                        "decision": decision,
                        "plan": plan,
                        "semantics": semantics,
                        "payload": payload,
                        "ledger": ledger,
                        "result": result,
                        "claim_allowed": allowed,
                        "claim_blockers": ";".join(blockers),
                    }
                )
    return records


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
                ),
                "phase_i_fe": PHASE_I_FE,
                "phase_ii_fe": actual_fe_used - PHASE_I_FE,
                "total_fe": actual_fe_used,
                "budget_limit": MAX_FES,
                "configured_budget_limit": MAX_FES,
                "budget_aligned_fe_used": result.fe_used,
                "actual_fe_used": actual_fe_used,
                "budget_limit_source": "experiment_config",
                "same_budget_violation": int(actual_fe_used > MAX_FES),
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
    counts = _parse_action_mix(row.get("action_mix", ""))
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
            if shuffled_error < relation_error
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
    relation_vs_fixed_coordinate_mean_gain = _mean(relation_vs_fixed_coordinate_gains)
    relation_gating_pass = (
        bool(relation_vs_fixed_coordinate_gains)
        and relation_beats_fixed_coordinate == len(relation_vs_fixed_coordinate_gains)
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
        and relation_positive == len(relation_rows)
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
    for row in relation_decisions:
        action_name = str(row.get("canonical_action_name", ""))
        if action_name:
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
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


def _multi_problem_diagnosis_rows(
    utility_rows: list[dict[str, object]],
    negative_control_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    problem_ids = sorted({str(row["problem_id"]) for row in utility_rows})
    if len(problem_ids) <= 1:
        return []

    relation_rows = [
        row for row in utility_rows if row["lane_id"] == "relation_dispatch_rule"
    ]
    relation_gains = [
        float(row["relative_gain_vs_fallback"]) for row in relation_rows
    ]
    positive_cases = sum(1 for gain in relation_gains if gain > 0.0)
    mean_gain = _mean(relation_gains)
    active_relation_rows = [
        row for row in relation_rows if _has_active_relation_action(row)
    ]
    active_relation_gains = [
        float(row["relative_gain_vs_fallback"]) for row in active_relation_rows
    ]
    active_positive_cases = sum(1 for gain in active_relation_gains if gain > 0.0)
    active_mean_gain = _mean(active_relation_gains)
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
    fixed_repair_pass = (
        bool(fixed_repair_gains)
        and fixed_repair_win_count == len(fixed_repair_gains)
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
        and fixed_coordinate_win_count == len(fixed_coordinate_gains)
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
    budget_violations = sum(
        1 for row in utility_rows if str(row["same_budget_violation"]) == "1"
    )
    negative_failures = sum(
        1
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
    blockers: list[str] = []
    if budget_violations:
        blockers.append("same_budget_violation")
    if positive_cases != len(relation_rows) or mean_gain <= 0.0:
        blockers.append("multi_problem_not_directionally_positive")
    if relation_meaningful != len(relation_rows):
        blockers.append("relation_dispatch_not_meaningful_win")
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
    directional_pass = positive_cases == len(relation_rows) and mean_gain > 0.0
    active_directional_pass = (
        bool(active_relation_rows)
        and active_positive_cases == len(active_relation_rows)
        and active_mean_gain > 0.0
    )
    pilot_utility_pass = (
        not budget_violations
        and directional_pass
        and catastrophic == 0
        and negative_control_pass
    )
    sota_allowed = not blockers

    return [
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_relation_dispatch_mean_gain",
            "status": "pass" if directional_pass else "blocked",
            "observed_value": (
                f"positive_cases={positive_cases}/{len(relation_rows)};"
                f"mean_gain={mean_gain:.6f}"
            ),
            "blocker_reason": ""
            if directional_pass
            else "multi_problem_not_directionally_positive",
            "next_step": "continue" if directional_pass else "diagnose_policy_evidence_before_sota",
        },
        {
            "run_id": RUN_ID,
            "problem_id": "ALL",
            "diagnostic_key": "multi_problem_active_relation_dispatch_mean_gain",
            "status": "pass" if active_directional_pass else "blocked",
            "observed_value": (
                f"active_cases={len(active_relation_rows)};"
                f"positive_cases={active_positive_cases}/{len(active_relation_rows)};"
                f"mean_gain={active_mean_gain:.6f}"
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
            "observed_value": f"win_count={positive_cases}/{len(relation_rows)}",
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
                f"shuffled_win_count={shuffled_win_count}/{negative_total_seeds}"
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
    rows.extend(_multi_problem_diagnosis_rows(utility_rows, negative_control_rows))
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


def _write_manifest(
    output_dir: Path,
    seeds: tuple[int, ...],
    problem_ids: tuple[str, ...],
    diagnosis_rows: list[dict[str, object]],
) -> None:
    multi_problem_pilot = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_pilot_utility_evidence",
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
    multi_problem_fixed_repair = (
        _diagnostic_observed_value(
            diagnosis_rows,
            "multi_problem_fixed_repair_baseline",
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
    artifacts = [
        "our_result_by_case.csv",
        "same_budget_ledger.csv",
        "backend_semantics_diff.csv",
        "action_execution_plan.csv",
        "action_trace.csv",
        "action_decision.csv",
        "overlap_relations.csv",
        "relation_join_audit.csv",
        "action_utility_audit.csv",
        "negative_control_comparison.csv",
        "policy_evidence_diagnosis.csv",
        "anti_leakage_audit.csv",
        "claim_gate.csv",
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
                f"{' '.join(problem_ids)}"
            ),
            f"Budget: {MAX_FES} FE per lane/case",
            f"Lanes: {', '.join(lane.lane_id for lane in LANES)}",
            "",
            "Runtime boundary: final/reported/oracle values must not enter runtime dispatch.",
            "",
            "Key gates:",
            f"- same-budget: {_diagnostic_observed_value(diagnosis_rows, 'same_budget_fe_status')}",
            f"- pilot utility: {_diagnostic_observed_value(diagnosis_rows, 'pilot_utility_evidence')}",
            f"- multi-problem pilot utility: {multi_problem_pilot}",
            f"- multi-problem active density: {multi_problem_active_density}",
            f"- fixed repair baseline: {multi_problem_fixed_repair}",
            f"- fixed coordinate baseline: {multi_problem_fixed_coordinate}",
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
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _records(
        output_dir=output,
        execution_runner=execution_runner,
        hcc_root=Path(hcc_root),
        python_executable=python_executable,
        seeds=tuple(seeds),
        problem_ids=tuple(problem_ids),
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
    _write_manifest(output, tuple(seeds), tuple(problem_ids), diagnosis_rows)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp_003 HCC runtime consumer smoke.")
    parser.add_argument("--output-dir", default="results/exp_003_hcc_runtime_consumer_smoke")
    parser.add_argument("--hcc-root", default=str(DEFAULT_HCC_MAIN_ROOT))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--problems", nargs="+", default=[PROBLEM_ID])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_hcc_runtime_consumer_smoke(
        output_dir=args.output_dir,
        hcc_root=Path(args.hcc_root),
        python_executable=str(args.python_executable),
        seeds=tuple(args.seeds),
        problem_ids=tuple(str(problem).upper() for problem in args.problems),
    )


if __name__ == "__main__":
    main()

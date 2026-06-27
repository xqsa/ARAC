from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ARAC_REPO_ROOT = Path(__file__).resolve().parents[2]
ARAC_SRC_ROOT = ARAC_REPO_ROOT / "src"
if str(ARAC_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(ARAC_SRC_ROOT))

from arac.audit import claim_gate
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


def _same_budget_group_id(seed: int) -> str:
    return f"{PROBLEM_ID}_seed{seed}_{MAX_FES}fe"


def _runtime_payload(seed: int, lane_id: str, action_name: str) -> dict[str, object]:
    payload = {
        "run_id": RUN_ID,
        "problem_id": PROBLEM_ID,
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
    return SameBudgetLedger(
        phase_i_fe=PHASE_I_FE,
        phase_ii_fe=result.fe_used - PHASE_I_FE,
        budget_limit=MAX_FES,
        fresh_execution=result.fresh_optimizer_execution,
    )


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
                    "problem_id": PROBLEM_ID,
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
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed in seeds:
        for lane in LANES:
            decision = _decision(lane)
            plan = build_hcc_action_execution_plan(PROBLEM_ID, decision)
            semantics = hcc_backend_semantics_for(
                decision,
                optimizer_consumed=plan.optimizer_consumed,
            )
            payload = _runtime_payload(seed, lane.lane_id, lane.selected_action_name)
            lane_output = (output_dir / "_hcc_smoke" / f"seed_{seed}" / lane.lane_id).resolve()
            result = execution_runner(
                HccAobExecutionRequest(
                    problem_id=PROBLEM_ID,
                    seed=seed,
                    max_fes=MAX_FES,
                    output_dir=lane_output,
                    hcc_root=hcc_root,
                    python_executable=python_executable,
                    timestamp=f"{RUN_ID}-seed{seed}-{lane.lane_id}",
                    arac_action=lane.runner_action_name,
                    enable_relation_dispatch=lane.relation_dispatch_enabled,
                    relation_policy_mode=lane.relation_policy_mode,
                )
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
    fallback_by_seed: dict[int, HccAobExecutionResult] = {}
    for record in records:
        if record["lane_id"] != "fallback":
            continue
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        fallback_by_seed[result.seed] = result
    rows: list[dict[str, object]] = []
    for record in records:
        lane = record["lane"]
        result = record["result"]
        ledger = record["ledger"]
        semantics = record["semantics"]
        assert isinstance(lane, LaneConfig)
        assert isinstance(result, HccAobExecutionResult)
        assert isinstance(ledger, SameBudgetLedger)
        fallback_result = fallback_by_seed[result.seed]
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
        (row["lane_id"], row["seed"]): row["claim_allowed"]
        for row in utility_rows
    }
    runtime_claim_by_lane = {
        (row["lane_id"], row["seed"]): row["runtime_connected_claim_allowed"]
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
                    (lane.lane_id, result.seed)
                ],
                "utility_claim_allowed": utility_claim_by_lane[(lane.lane_id, result.seed)],
                "performance_claim_allowed": 0,
            }
        )
    return rows


def _ledger_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": result.problem_id,
                "seed": result.seed,
                "same_budget_group_id": _same_budget_group_id(result.seed),
                "phase_i_fe": PHASE_I_FE,
                "phase_ii_fe": result.fe_used - PHASE_I_FE,
                "total_fe": result.fe_used,
                "budget_limit": MAX_FES,
                "configured_budget_limit": MAX_FES,
                "actual_fe_used": result.fe_used,
                "budget_limit_source": "experiment_config",
                "same_budget_violation": int(result.fe_used > MAX_FES),
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
                "problem_id": PROBLEM_ID,
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


def _claim_gate_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
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
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": PROBLEM_ID,
                "seed": result.seed,
                "selected_action_name": lane.selected_action_name,
                "optimizer_consumed": int(plan.optimizer_consumed),
                "same_budget_violation": int(ledger.violation),
                "performance_claim_allowed": 0,
                "claim_allowed": int(record["claim_allowed"]),
                "claim_blockers": record["claim_blockers"],
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _result_by_seed_and_lane(
    records: list[dict[str, object]],
) -> dict[tuple[int, str], HccAobExecutionResult]:
    indexed: dict[tuple[int, str], HccAobExecutionResult] = {}
    for record in records:
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        indexed[(result.seed, str(record["lane_id"]))] = result
    return indexed


def _negative_control_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = _result_by_seed_and_lane(records)
    seeds = sorted(seed for seed, lane_id in indexed if lane_id == "relation_dispatch_rule")
    relation_errors = [
        indexed[(seed, "relation_dispatch_rule")].final_error for seed in seeds
    ]
    shuffled_errors = [
        indexed[(seed, "shuffled_relation_dispatch")].final_error for seed in seeds
    ]
    shuffled_win_count = sum(
        1
        for relation_error, shuffled_error in zip(relation_errors, shuffled_errors, strict=True)
        if shuffled_error < relation_error
    )
    total = len(seeds)
    stable_outperform = shuffled_win_count > total / 2
    return [
        {
            "run_id": RUN_ID,
            "problem_id": PROBLEM_ID,
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
    ]


def run_hcc_runtime_consumer_smoke(
    output_dir: Path | str = Path("results/exp_003_hcc_runtime_consumer_smoke"),
    execution_runner: Callable[[HccAobExecutionRequest], HccAobExecutionResult] = (
        run_hcc_aob_smoke_execution
    ),
    hcc_root: Path | str = DEFAULT_HCC_MAIN_ROOT,
    python_executable: str = sys.executable,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _records(
        output_dir=output,
        execution_runner=execution_runner,
        hcc_root=Path(hcc_root),
        python_executable=python_executable,
        seeds=tuple(seeds),
    )
    utility_rows = _utility_rows(records)
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
        _negative_control_rows(records),
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
        _claim_gate_rows(records),
        [
            "run_id",
            "lane_id",
            "problem_id",
            "seed",
            "selected_action_name",
            "optimizer_consumed",
            "same_budget_violation",
            "performance_claim_allowed",
            "claim_allowed",
            "claim_blockers",
        ],
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp_003 HCC runtime consumer smoke.")
    parser.add_argument("--output-dir", default="results/exp_003_hcc_runtime_consumer_smoke")
    parser.add_argument("--hcc-root", default=str(DEFAULT_HCC_MAIN_ROOT))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_hcc_runtime_consumer_smoke(
        output_dir=args.output_dir,
        hcc_root=Path(args.hcc_root),
        python_executable=str(args.python_executable),
        seeds=tuple(args.seeds),
    )


if __name__ == "__main__":
    main()

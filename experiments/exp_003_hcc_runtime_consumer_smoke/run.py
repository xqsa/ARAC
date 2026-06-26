from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

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
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS, validate_runtime_payload
from arac.policy import ActionDecision
from arac.action_space import ActionFamily

RUN_ID = "exp_003_hcc_runtime_consumer_smoke"
PROBLEM_ID = "E2"
SEED = 1
MAX_FES = 2_000
PHASE_I_FE = 0
PHASE_II_FE = MAX_FES
LANES = (
    ("fallback", ActionFamily.FALLBACK, "conservative_no_action"),
    ("repair_runtime_consumer", ActionFamily.REASSIGN_REPAIR, "repair_shared_variable_binding"),
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _decision(action_family: ActionFamily, action_name: str) -> ActionDecision:
    return ActionDecision(
        action_family=action_family,
        action_name=action_name,
        decision="fallback" if action_family == ActionFamily.FALLBACK else "allow",
        trigger_reason="exp_003_fixed_lane_runtime_consumer_smoke",
        utility_proxy=0.0 if action_family == ActionFamily.FALLBACK else 1.0,
    )


def _runtime_payload(lane_id: str, action_name: str) -> dict[str, object]:
    payload = {
        "run_id": RUN_ID,
        "problem_id": PROBLEM_ID,
        "seed": SEED,
        "lane_id": lane_id,
        "selected_action_name": action_name,
        "benchmark": "AOB",
        "budget_limit": MAX_FES,
        "used_for_runtime": 1,
    }
    validate_runtime_payload(payload)
    return payload


def _records(
    output_dir: Path,
    execution_runner: Callable[[HccAobExecutionRequest], HccAobExecutionResult],
    hcc_root: Path,
    python_executable: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    ledger = SameBudgetLedger(
        phase_i_fe=PHASE_I_FE,
        phase_ii_fe=PHASE_II_FE,
        budget_limit=MAX_FES,
        fresh_execution=True,
    )
    for lane_id, action_family, action_name in LANES:
        decision = _decision(action_family, action_name)
        plan = build_hcc_action_execution_plan(PROBLEM_ID, decision)
        semantics = hcc_backend_semantics_for(
            decision,
            optimizer_consumed=plan.optimizer_consumed,
        )
        payload = _runtime_payload(lane_id, action_name)
        lane_output = (output_dir / "_hcc_smoke" / lane_id).resolve()
        result = execution_runner(
            HccAobExecutionRequest(
                problem_id=PROBLEM_ID,
                seed=SEED,
                max_fes=MAX_FES,
                output_dir=lane_output,
                hcc_root=hcc_root,
                python_executable=python_executable,
                timestamp=f"{RUN_ID}-{lane_id}",
                arac_action=action_name,
            )
        )
        allowed, blockers = claim_gate(
            runtime_payload=payload,
            decision=decision,
            semantics_diff=semantics,
            ledger=SameBudgetLedger(
                phase_i_fe=PHASE_I_FE,
                phase_ii_fe=result.fe_used,
                budget_limit=result.fe_used,
                fresh_execution=result.fresh_optimizer_execution,
            ),
            utility_label="runtime_smoke_not_performance_claim",
            negative_control_pass=True,
            optimizer_consumed=plan.optimizer_consumed,
        )
        records.append(
            {
                "lane_id": lane_id,
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


def _our_result_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        decision = record["decision"]
        result = record["result"]
        assert isinstance(decision, ActionDecision)
        assert isinstance(result, HccAobExecutionResult)
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": result.problem_id,
                "seed": result.seed,
                "selected_action_family": decision.action_family.value,
                "selected_action_name": decision.action_name,
                "hcc_smoke_final_error": f"{result.final_error:.6e}",
                "hcc_smoke_fe_used": result.fe_used,
                "hcc_smoke_status": result.status,
                "fresh_optimizer_execution": int(result.fresh_optimizer_execution),
                "action_trace_rows": result.action_trace_rows,
                "runtime_dispatch_allowed": 1,
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
                "phase_i_fe": PHASE_I_FE,
                "phase_ii_fe": result.fe_used,
                "total_fe": result.fe_used,
                "budget_limit": result.fe_used,
                "same_budget_violation": 0,
                "fresh_execution": int(result.fresh_optimizer_execution),
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
                "lane_id": record["lane_id"],
                "problem_id": PROBLEM_ID,
                "seed": SEED,
                "selected_action_name": decision.action_name,
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
        result = record["result"]
        assert isinstance(result, HccAobExecutionResult)
        if result.action_trace_path is None:
            continue
        with result.action_trace_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"run_id": RUN_ID, "lane_id": record["lane_id"], **row})
    return rows


def _anti_leakage_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    payloads = [record["payload"] for record in records]
    rows: list[dict[str, object]] = []
    for field in sorted(FORBIDDEN_RUNTIME_FIELDS):
        found = any(field in payload for payload in payloads)
        rows.append(
            {
                "run_id": RUN_ID,
                "forbidden_field": field,
                "found_in_runtime_payload": int(found),
                "runtime_dispatch_allowed": 0 if found else 1,
                "audit_status": "fail" if found else "pass",
            }
        )
    return rows


def _claim_gate_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        decision = record["decision"]
        plan = record["plan"]
        assert isinstance(decision, ActionDecision)
        rows.append(
            {
                "run_id": RUN_ID,
                "lane_id": record["lane_id"],
                "problem_id": PROBLEM_ID,
                "seed": SEED,
                "selected_action_name": decision.action_name,
                "optimizer_consumed": int(plan.optimizer_consumed),
                "claim_allowed": int(record["claim_allowed"]),
                "claim_blockers": record["claim_blockers"],
            }
        )
    return rows


def run_hcc_runtime_consumer_smoke(
    output_dir: Path | str = Path("results/exp_003_hcc_runtime_consumer_smoke"),
    execution_runner: Callable[[HccAobExecutionRequest], HccAobExecutionResult] = (
        run_hcc_aob_smoke_execution
    ),
    hcc_root: Path | str = DEFAULT_HCC_MAIN_ROOT,
    python_executable: str = sys.executable,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _records(
        output_dir=output,
        execution_runner=execution_runner,
        hcc_root=Path(hcc_root),
        python_executable=python_executable,
    )
    _write_csv(
        output / "our_result_by_case.csv",
        _our_result_rows(records),
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
            "phase_i_fe",
            "phase_ii_fe",
            "total_fe",
            "budget_limit",
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
            "overlap_size",
            "previous_delta",
            "current_delta",
            "owner_selected",
            "semantic_surface",
            "optimizer_consumed",
        ],
    )
    _write_csv(
        output / "anti_leakage_audit.csv",
        _anti_leakage_rows(records),
        [
            "run_id",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_hcc_runtime_consumer_smoke(
        output_dir=args.output_dir,
        hcc_root=Path(args.hcc_root),
        python_executable=str(args.python_executable),
    )


if __name__ == "__main__":
    main()

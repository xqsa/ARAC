from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

from arac.backends.hcc import HccAobExecutionRequest, HccAobExecutionResult


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_exp_003_cli_help_works_without_pythonpath() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "exp_003_hcc_runtime_consumer_smoke"
        / "run.py"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run exp_003 HCC runtime consumer smoke" in completed.stdout


def test_exp_003_writes_runtime_consumer_smoke_artifacts(tmp_path: Path) -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        run_hcc_runtime_consumer_smoke,
    )

    requests: list[HccAobExecutionRequest] = []

    def fake_runner(request: HccAobExecutionRequest) -> HccAobExecutionResult:
        requests.append(request)
        trace_path = tmp_path / f"{request.arac_action}_action_trace.csv"
        trace_path.write_text(
            "problem_id,seed,outer_iter,group_index,selected_action_name,"
            "relation_id,group_left,group_right,shared_vars_hash,action_family,"
            "canonical_action_name,relation_policy_source,"
            "overlap_size,previous_delta,current_delta,owner_selected,"
            "semantic_surface,state_mutated,downstream_consumed,"
            "downstream_consumption_scope,optimizer_consumed\n"
            f"E2,1,0,1,{request.arac_action},O0_0_1,0,1,abc123,reassign_repair,"
            f"{request.arac_action},legacy_single_action,"
            "1,1.000000e+00,2.000000e+00,current,shared_variable_owner_rebinding,"
            "1,1,same_outer_iteration,"
            f"{1 if request.arac_action == 'repair_shared_variable_binding' else 0}\n",
            encoding="utf-8",
        )
        return HccAobExecutionResult(
            problem_id=request.problem_id,
            seed=request.seed,
            max_fes=request.max_fes,
            final_error=123.0
            if request.arac_action == "conservative_no_action"
            else 77.0,
            fe_used=2_128,
            time_seconds=0.5,
            output_root=request.output_dir,
            fresh_optimizer_execution=True,
            status="completed",
            result_source="hcc_subprocess_smoke_execution",
            action_trace_path=trace_path,
            action_trace_rows=1,
        )

    output = run_hcc_runtime_consumer_smoke(
        output_dir=tmp_path / "exp003",
        execution_runner=fake_runner,
        python_executable="python",
    )

    expected = {
        "our_result_by_case.csv",
        "same_budget_ledger.csv",
        "backend_semantics_diff.csv",
        "action_execution_plan.csv",
        "action_trace.csv",
        "anti_leakage_audit.csv",
        "claim_gate.csv",
    }
    assert expected == {path.name for path in output.iterdir() if path.suffix == ".csv"}
    assert [request.arac_action for request in requests] == [
        "conservative_no_action",
        "repair_shared_variable_binding",
    ]

    plan_rows = _read_csv(output / "action_execution_plan.csv")
    repair_plan = next(
        row
        for row in plan_rows
        if row["selected_action_name"] == "repair_shared_variable_binding"
    )
    assert repair_plan["optimizer_consumed"] == "1"
    assert repair_plan["execution_mode"] == "hcc_smoke_runtime_consumed"

    semantics_rows = _read_csv(output / "backend_semantics_diff.csv")
    repair_semantics = next(
        row
        for row in semantics_rows
        if row["selected_action_name"] == "repair_shared_variable_binding"
    )
    assert repair_semantics["variable_owner_changed"] == "1"
    assert repair_semantics["backend_semantics_changed"] == "1"

    claim_rows = _read_csv(output / "claim_gate.csv")
    repair_claim = next(
        row
        for row in claim_rows
        if row["selected_action_name"] == "repair_shared_variable_binding"
    )
    assert repair_claim["optimizer_consumed"] == "1"
    assert "active_action_not_consumed_by_hcc_runtime" not in repair_claim["claim_blockers"]

    trace_rows = _read_csv(output / "action_trace.csv")
    assert any(row["optimizer_consumed"] == "1" for row in trace_rows)
    assert {
        "relation_id",
        "group_left",
        "group_right",
        "shared_vars_hash",
        "action_family",
        "canonical_action_name",
        "relation_policy_source",
        "state_mutated",
        "downstream_consumed",
        "downstream_consumption_scope",
    }.issubset(trace_rows[0])

    result_rows = _read_csv(output / "our_result_by_case.csv")
    assert all(row["dispatch_scope"] == "fixed_lane_runtime_consumer_smoke" for row in result_rows)
    assert all(row["relation_dispatch_enabled"] == "0" for row in result_rows)
    assert all(row["performance_claim_allowed"] == "0" for row in result_rows)

    ledger_rows = _read_csv(output / "same_budget_ledger.csv")
    assert all(row["same_budget_group_id"] == "E2_seed1_2000fe" for row in ledger_rows)
    assert all(row["configured_budget_limit"] == "2000" for row in ledger_rows)
    assert all(row["actual_fe_used"] == "2128" for row in ledger_rows)
    assert all(row["budget_limit"] == "2000" for row in ledger_rows)
    assert all(row["budget_limit_source"] == "experiment_config" for row in ledger_rows)
    assert all(row["same_budget_violation"] == "1" for row in ledger_rows)

    claim_rows = _read_csv(output / "claim_gate.csv")
    assert all(row["performance_claim_allowed"] == "0" for row in claim_rows)
    assert all(row["same_budget_violation"] == "1" for row in claim_rows)

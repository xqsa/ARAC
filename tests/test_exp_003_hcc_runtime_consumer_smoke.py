from __future__ import annotations

import csv
from pathlib import Path

from arac.backends.hcc import HccAobExecutionRequest, HccAobExecutionResult


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
            "overlap_size,previous_delta,current_delta,owner_selected,"
            "semantic_surface,optimizer_consumed\n"
            f"E2,1,0,1,{request.arac_action},1,1.000000e+00,"
            "2.000000e+00,current,shared_variable_owner_rebinding,"
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

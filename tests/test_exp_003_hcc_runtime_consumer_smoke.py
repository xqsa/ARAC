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
        problem_id = request.problem_id
        lane_id = request.output_dir.name
        trace_path = request.output_dir / "action_trace.csv"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if request.enable_relation_dispatch:
            policy_source = (
                "deterministic_shuffled_negative_control"
                if request.relation_policy_mode == "shuffled"
                else "rule_based_relation_policy"
            )
            first_action = (
                "repair_shared_variable_binding"
                if request.relation_policy_mode == "shuffled"
                else "allow_beneficial_coordination"
            )
            first_family = (
                "reassign_repair"
                if request.relation_policy_mode == "shuffled"
                else "coordinate"
            )
            trace_path.write_text(
                "problem_id,seed,outer_iter,group_index,selected_action_name,"
                "relation_id,group_left,group_right,shared_vars_hash,action_family,"
                "canonical_action_name,relation_policy_source,"
                "overlap_size,previous_delta,current_delta,owner_selected,"
                "semantic_surface,state_mutated,downstream_consumed,"
                "downstream_consumption_scope,optimizer_consumed\n"
                f"{problem_id},{request.seed},0,1,{first_action},O0_0_1,0,1,abc123,"
                f"{first_family},{first_action},{policy_source},"
                "1,1.000000e+00,1.100000e+00,clipped_consensus_blend,"
                "coordination_clipped_consensus_blend,1,1,same_outer_iteration,1\n"
                f"{problem_id},{request.seed},0,2,repair_shared_variable_binding,O0_1_2,1,2,def456,"
                f"reassign_repair,repair_shared_variable_binding,{policy_source},"
                "1,1.100000e+00,1.200000e+00,current,"
                "shared_variable_owner_rebinding,1,0,same_outer_iteration,0\n",
                encoding="utf-8",
            )
            (request.output_dir / f"{problem_id}_action_decision.csv").write_text(
                "run_id,problem_id,relation_id,group_left,group_right,shared_vars_count,"
                "overlap_strength,delta_signal,rank_signal,relation_action_name,"
                "canonical_action_name,action_family,confidence,trigger_reason\n"
                f"run,{problem_id},O0_0_1,0,1,1,1.000000,0.100000,0.900000,coordinate,"
                "allow_beneficial_coordination,coordinate,0.800000,stable\n"
                f"run,{problem_id},O0_1_2,1,2,1,1.000000,0.100000,0.900000,reassign_repair,"
                "repair_shared_variable_binding,reassign_repair,0.700000,repair\n",
                encoding="utf-8",
            )
            (request.output_dir / f"{problem_id}_overlap_relations.csv").write_text(
                "relation_id,problem_id,outer_iter,group_left,group_right,shared_vars,"
                "overlap_strength,delta_signal,rank_signal,budget_remaining_ratio,"
                "previous_delta,current_delta,delta_abs_gap,delta_signed_gap,"
                "delta_ratio_gap,both_positive,one_side_zero,rank_gap,rank_stability,"
                "shared_var_count,shared_var_support_ratio,feature_coverage,"
                "fallback_margin_proxy\n"
                f"O0_0_1,{problem_id},0,0,1,7,1.000000,0.100000,0.900000,1.000000,"
                "1.000000,1.100000,0.100000,0.100000,0.090909,1,0,"
                "0.000000,0.900000,1,0.050000,1.000000,0.900000\n"
                f"O0_1_2,{problem_id},0,1,2,9,1.000000,0.100000,0.900000,1.000000,"
                "1.100000,1.200000,0.100000,0.100000,0.083333,1,0,"
                "0.000000,0.900000,1,0.050000,1.000000,0.900000\n",
                encoding="utf-8",
            )
        else:
            trace_path.write_text(
                "problem_id,seed,outer_iter,group_index,selected_action_name,"
                "relation_id,group_left,group_right,shared_vars_hash,action_family,"
                "canonical_action_name,relation_policy_source,"
                "overlap_size,previous_delta,current_delta,owner_selected,"
                "semantic_surface,state_mutated,downstream_consumed,"
                "downstream_consumption_scope,optimizer_consumed\n"
                f"{problem_id},{request.seed},0,1,{request.arac_action},O0_0_1,0,1,abc123,reassign_repair,"
                f"{request.arac_action},legacy_single_action,"
                "1,1.000000e+00,2.000000e+00,current,shared_variable_owner_rebinding,"
                "1,1,same_outer_iteration,"
                f"{1 if request.arac_action == 'repair_shared_variable_binding' else 0}\n",
                encoding="utf-8",
            )
        final_error = {
            "fallback": 120.0 + request.seed,
            "fixed_repair": 80.0 + request.seed,
            "relation_dispatch_rule": 121.0 + request.seed,
            "shuffled_relation_dispatch": 119.0 + request.seed,
        }[lane_id]
        return HccAobExecutionResult(
            problem_id=problem_id,
            seed=request.seed,
            max_fes=request.max_fes,
            final_error=final_error,
            fe_used=2_000,
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
        "action_decision.csv",
        "overlap_relations.csv",
        "relation_join_audit.csv",
        "action_utility_audit.csv",
        "anti_leakage_audit.csv",
        "claim_gate.csv",
        "negative_control_comparison.csv",
        "policy_evidence_diagnosis.csv",
    }
    assert expected == {path.name for path in output.iterdir() if path.suffix == ".csv"}
    assert len(requests) == 12
    assert {request.seed for request in requests} == {1, 2, 3}
    assert {
        (request.output_dir.name, request.arac_action, request.enable_relation_dispatch, request.relation_policy_mode)
        for request in requests
    } == {
        ("fallback", "conservative_no_action", False, "rule"),
        ("fixed_repair", "repair_shared_variable_binding", False, "rule"),
        ("relation_dispatch_rule", "conservative_no_action", True, "rule"),
        ("shuffled_relation_dispatch", "conservative_no_action", True, "shuffled"),
    }

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
    assert {row["lane_id"] for row in trace_rows} == {
        "fallback",
        "fixed_repair",
        "relation_dispatch_rule",
        "shuffled_relation_dispatch",
    }
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

    decision_rows = _read_csv(output / "action_decision.csv")
    overlap_rows = _read_csv(output / "overlap_relations.csv")
    assert {row["relation_id"] for row in decision_rows} == {"O0_0_1", "O0_1_2"}
    assert {row["relation_id"] for row in overlap_rows} == {"O0_0_1", "O0_1_2"}
    assert overlap_rows[0]["fallback_margin_proxy"] == "0.900000"
    assert overlap_rows[0]["feature_coverage"] == "1.000000"

    join_rows = _read_csv(output / "relation_join_audit.csv")
    assert all(row["audit_status"] == "pass" for row in join_rows)
    assert {row["relation_id"] for row in join_rows} == {"O0_0_1", "O0_1_2"}

    utility_rows = _read_csv(output / "action_utility_audit.csv")
    assert {
        "final_error",
        "fe_used",
        "same_budget_violation",
        "relative_gain_vs_fallback",
        "utility_label",
        "action_mix",
        "claim_allowed",
        "claim_blockers",
    }.issubset(utility_rows[0])
    by_lane = {row["lane_id"]: row for row in utility_rows}
    assert by_lane["fixed_repair"]["utility_label"] == "meaningful_win"
    assert by_lane["fixed_repair"]["claim_allowed"] == "1"
    assert by_lane["relation_dispatch_rule"]["utility_label"] == "tie_or_small_effect"
    assert by_lane["relation_dispatch_rule"]["claim_allowed"] == "0"
    assert "utility_not_meaningful_win" in by_lane["relation_dispatch_rule"]["claim_blockers"]
    assert by_lane["shuffled_relation_dispatch"]["utility_label"] == "tie_or_small_effect"
    assert by_lane["shuffled_relation_dispatch"]["claim_allowed"] == "0"
    assert "negative_control_lane_not_utility_claim" in by_lane["shuffled_relation_dispatch"]["claim_blockers"]

    negative_control_rows = _read_csv(output / "negative_control_comparison.csv")
    assert negative_control_rows == [
        {
            "run_id": "exp_003_hcc_runtime_consumer_smoke",
            "problem_id": "E2",
            "seeds": "1;2;3",
            "relation_dispatch_mean_final_error": "1.230000e+02",
            "shuffled_mean_final_error": "1.210000e+02",
            "shuffled_win_count": "3",
            "total_seeds": "3",
            "stable_outperform_detected": "1",
            "negative_control_pass": "0",
            "diagnostic": "shuffled_control_stably_outperforms_relation_dispatch",
        }
    ]

    diagnosis_rows = _read_csv(output / "policy_evidence_diagnosis.csv")
    diagnosis_by_key = {row["diagnostic_key"]: row for row in diagnosis_rows}
    assert diagnosis_by_key["relation_dispatch_utility"]["status"] == "blocked"
    assert diagnosis_by_key["relation_dispatch_utility"]["observed_value"] == "0/3"
    assert diagnosis_by_key["relation_dispatch_directional_utility"]["status"] == "blocked"
    assert "mean_gain=" in diagnosis_by_key["relation_dispatch_directional_utility"]["observed_value"]
    assert diagnosis_by_key["shuffled_negative_control"]["status"] == "blocked"
    assert diagnosis_by_key["negative_control_action_mix"]["status"] == "blocked"
    assert "relation_dispatch_rule=" in diagnosis_by_key["negative_control_action_mix"]["observed_value"]
    assert "shuffled_relation_dispatch=" in diagnosis_by_key["negative_control_action_mix"]["observed_value"]
    assert diagnosis_by_key["sota_escalation_allowed"]["observed_value"] == "0"
    assert diagnosis_by_key["sota_escalation_allowed"]["next_step"] == (
        "diagnose_policy_evidence_before_sota"
    )

    result_rows = _read_csv(output / "our_result_by_case.csv")
    assert {row["dispatch_scope"] for row in result_rows} == {
        "fixed_lane_runtime_consumer_smoke",
        "per_overlap_relation_runtime_dispatch",
        "shuffled_relation_dispatch_negative_control",
    }
    assert {row["relation_dispatch_enabled"] for row in result_rows} == {"0", "1"}
    assert all(row["performance_claim_allowed"] == "0" for row in result_rows)
    result_by_lane = {row["lane_id"]: row for row in result_rows}
    assert result_by_lane["fixed_repair"]["runtime_connected_claim_allowed"] == "1"
    assert result_by_lane["fixed_repair"]["utility_claim_allowed"] == "1"
    assert result_by_lane["relation_dispatch_rule"]["runtime_connected_claim_allowed"] == "1"
    assert result_by_lane["relation_dispatch_rule"]["utility_claim_allowed"] == "0"
    assert result_by_lane["shuffled_relation_dispatch"]["runtime_connected_claim_allowed"] == "1"
    assert result_by_lane["shuffled_relation_dispatch"]["utility_claim_allowed"] == "0"

    ledger_rows = _read_csv(output / "same_budget_ledger.csv")
    assert {row["same_budget_group_id"] for row in ledger_rows} == {
        "E2_seed1_2000fe",
        "E2_seed2_2000fe",
        "E2_seed3_2000fe",
    }
    assert all(row["configured_budget_limit"] == "2000" for row in ledger_rows)
    assert all(row["actual_fe_used"] == "2000" for row in ledger_rows)
    assert all(row["budget_limit"] == "2000" for row in ledger_rows)
    assert all(row["budget_limit_source"] == "experiment_config" for row in ledger_rows)
    assert all(row["same_budget_violation"] == "0" for row in ledger_rows)

    claim_rows = _read_csv(output / "claim_gate.csv")
    assert all(row["performance_claim_allowed"] == "0" for row in claim_rows)
    assert all(row["same_budget_violation"] == "0" for row in claim_rows)

    requests.clear()
    multi_output = run_hcc_runtime_consumer_smoke(
        output_dir=tmp_path / "exp003-multi",
        execution_runner=fake_runner,
        python_executable="python",
        seeds=(1,),
        problem_ids=("E1", "E2"),
    )

    assert len(requests) == 8
    assert {request.problem_id for request in requests} == {"E1", "E2"}
    assert {request.seed for request in requests} == {1}
    multi_utility = _read_csv(multi_output / "action_utility_audit.csv")
    assert {row["problem_id"] for row in multi_utility} == {"E1", "E2"}
    multi_negative = _read_csv(multi_output / "negative_control_comparison.csv")
    assert {row["problem_id"] for row in multi_negative} == {"E1", "E2"}
    multi_diagnosis = _read_csv(multi_output / "policy_evidence_diagnosis.csv")
    assert {row["problem_id"] for row in multi_diagnosis} == {"ALL", "E1", "E2"}
    aggregate_by_key = {
        row["diagnostic_key"]: row
        for row in multi_diagnosis
        if row["problem_id"] == "ALL"
    }
    assert aggregate_by_key["multi_problem_relation_dispatch_mean_gain"]["observed_value"] == (
        "positive_cases=0/2;mean_gain=-0.008264"
    )
    assert aggregate_by_key["multi_problem_catastrophic_loss_gate"]["observed_value"] == "0/2"
    assert aggregate_by_key["multi_problem_sota_escalation_allowed"]["status"] == "blocked"
    assert aggregate_by_key["multi_problem_sota_escalation_allowed"]["observed_value"] == "0"
    multi_ledger = _read_csv(multi_output / "same_budget_ledger.csv")
    assert {row["same_budget_group_id"] for row in multi_ledger} == {
        "E1_seed1_2000fe",
        "E2_seed1_2000fe",
    }

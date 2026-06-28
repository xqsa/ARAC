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


def _hcc_result(
    problem_id: str,
    seed: int,
    final_error: float,
    tmp_path: Path,
) -> HccAobExecutionResult:
    return HccAobExecutionResult(
        problem_id=problem_id,
        seed=seed,
        max_fes=2000,
        final_error=final_error,
        fe_used=1999,
        time_seconds=0.1,
        output_root=tmp_path,
        fresh_optimizer_execution=True,
        status="ok",
        result_source="test",
    )


def test_exp_003_normalizes_subprocess_run_id_in_relation_artifacts(tmp_path: Path) -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import RUN_ID, _with_lane_prefix

    result = HccAobExecutionResult(
        problem_id="E2",
        seed=7,
        max_fes=5000,
        final_error=1.0,
        fe_used=4999,
        time_seconds=0.1,
        output_root=tmp_path,
        fresh_optimizer_execution=True,
        status="ok",
        result_source="test",
    )

    rows = _with_lane_prefix(
        {"lane_id": "relation_dispatch_rule", "result": result},
        [{"run_id": "subprocess-case-id", "relation_id": "O0_0_1"}],
    )

    assert rows == [
        {
            "run_id": RUN_ID,
            "lane_id": "relation_dispatch_rule",
            "seed": 7,
            "relation_id": "O0_0_1",
        }
    ]


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
    assert "--jobs" in completed.stdout
    assert "--max-fes" in completed.stdout


def test_negative_control_ignores_tiny_shuffled_advantage(tmp_path: Path) -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import _negative_control_rows

    records = []
    for seed in (1, 2, 3):
        records.extend(
            [
                {
                    "lane_id": "relation_dispatch_rule",
                    "result": _hcc_result("E6", seed, 100.0, tmp_path),
                },
                {
                    "lane_id": "shuffled_relation_dispatch",
                    "result": _hcc_result("E6", seed, 99.99, tmp_path),
                },
            ]
        )

    rows = _negative_control_rows(records)

    assert rows[0]["shuffled_win_count"] == 0
    assert rows[0]["stable_outperform_detected"] == 0
    assert rows[0]["negative_control_pass"] == 1


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
                "semantic_surface,state_mutated,action_value_delta_norm,downstream_consumed,"
                "downstream_consumption_scope,optimizer_consumed\n"
                f"{problem_id},{request.seed},0,1,{first_action},O0_0_1,0,1,abc123,"
                f"{first_family},{first_action},{policy_source},"
                "1,1.000000e+00,1.100000e+00,clipped_consensus_blend,"
                "coordination_clipped_consensus_blend,1,2.000000e-01,1,same_outer_iteration,1\n"
                f"{problem_id},{request.seed},0,2,repair_shared_variable_binding,O0_1_2,1,2,def456,"
                f"reassign_repair,repair_shared_variable_binding,{policy_source},"
                "1,1.100000e+00,1.200000e+00,current,"
                "shared_variable_owner_rebinding,1,3.000000e-01,0,same_outer_iteration,0\n",
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
            (request.output_dir / f"{problem_id}_action_mismatch_audit.csv").write_text(
                "run_id,problem_id,relation_id,group_left,group_right,candidate_scores,"
                "coordinate_score,isolate_conflicting_relation_score,reassign_repair_score,"
                "fallback_score,best_action_name,best_score,second_best_action_name,"
                "second_best_score,margin,final_action_name,final_canonical_action_name,"
                "confidence,trigger_reason,abstain_reason\n"
                f"run,{problem_id},O0_0_1,0,1,coordinate=0.800000;fallback=0.100000,"
                "0.800000,0.000000,0.000000,0.100000,coordinate,0.800000,"
                "fallback,0.100000,0.700000,coordinate,"
                "allow_beneficial_coordination,0.800000,stable,\n"
                f"run,{problem_id},O0_1_2,1,2,reassign_repair=0.700000;fallback=0.100000,"
                "0.000000,0.000000,0.700000,0.100000,reassign_repair,0.700000,"
                "fallback,0.100000,0.600000,reassign_repair,"
                "repair_shared_variable_binding,0.700000,repair,\n",
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
            action_family = (
                "coordinate"
                if request.arac_action == "allow_beneficial_coordination"
                else "reassign_repair"
            )
            owner_selected = (
                "clipped_consensus_blend"
                if request.arac_action == "allow_beneficial_coordination"
                else "current"
            )
            semantic_surface = (
                "coordination_clipped_consensus_blend"
                if request.arac_action == "allow_beneficial_coordination"
                else "shared_variable_owner_rebinding"
            )
            trace_path.write_text(
                "problem_id,seed,outer_iter,group_index,selected_action_name,"
                "relation_id,group_left,group_right,shared_vars_hash,action_family,"
                "canonical_action_name,relation_policy_source,"
                "overlap_size,previous_delta,current_delta,owner_selected,"
                "semantic_surface,state_mutated,action_value_delta_norm,downstream_consumed,"
                "downstream_consumption_scope,optimizer_consumed\n"
                f"{problem_id},{request.seed},0,1,{request.arac_action},O0_0_1,0,1,abc123,{action_family},"
                f"{request.arac_action},legacy_single_action,"
                f"1,1.000000e+00,2.000000e+00,{owner_selected},{semantic_surface},"
                "1,1.000000e-01,1,same_outer_iteration,"
                f"{1 if request.arac_action in {'repair_shared_variable_binding', 'allow_beneficial_coordination'} else 0}\n",
                encoding="utf-8",
            )
        final_error = {
            "fallback": 120.0 + request.seed,
            "fixed_repair": 80.0 + request.seed,
            "fixed_coordinate": 130.0 + request.seed,
            "relation_dispatch_rule": 126.0 + request.seed,
            "shuffled_relation_dispatch": 119.0 + request.seed,
        }[lane_id]
        return HccAobExecutionResult(
            problem_id=problem_id,
            seed=request.seed,
            max_fes=request.max_fes,
            final_error=final_error,
            fe_used=request.max_fes,
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
        "action_mismatch_audit.csv",
        "overlap_relations.csv",
        "relation_join_audit.csv",
        "action_utility_audit.csv",
        "anti_leakage_audit.csv",
        "claim_gate.csv",
        "negative_control_comparison.csv",
        "policy_evidence_diagnosis.csv",
    }
    assert expected == {path.name for path in output.iterdir() if path.suffix == ".csv"}
    manifest = (output / "run_manifest.md").read_text(encoding="utf-8")
    assert "runtime dispatch + utility evidence" in manifest
    assert "final/reported/oracle values must not enter runtime dispatch" in manifest
    assert "SOTA claim allowed: 0" in manifest
    assert "multi-problem pilot utility: not_applicable" in manifest
    assert "- claim_evidence_table.md" in manifest
    assert "Freeze evidence:" in manifest
    assert "- git commit:" in manifest
    assert "- config fingerprint:" in manifest
    assert "- policy sha256:" in manifest
    assert "- experiment runner sha256:" in manifest
    assert "- HCC smoke runner sha256:" in manifest
    claim_table = (output / "claim_evidence_table.md").read_text(encoding="utf-8")
    assert "# exp_003 Claim Evidence Table" in claim_table
    assert "| E2 | relation_dispatch_utility | blocked | 0/3 |" in claim_table
    assert "| E2 | sota_escalation_allowed | blocked | 0 |" in claim_table
    assert "policy_evidence_diagnosis.csv" in claim_table
    assert len(requests) == 15
    assert {request.seed for request in requests} == {1, 2, 3}
    assert {
        (request.output_dir.name, request.arac_action, request.enable_relation_dispatch, request.relation_policy_mode)
        for request in requests
    } == {
        ("fallback", "conservative_no_action", False, "rule"),
        ("fixed_repair", "repair_shared_variable_binding", False, "rule"),
        ("fixed_coordinate", "allow_beneficial_coordination", False, "rule"),
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
    coordinate_semantics = next(
        row
        for row in semantics_rows
        if row["lane_id"] == "fixed_coordinate"
    )
    assert coordinate_semantics["coordination_mode_changed"] == "1"
    assert coordinate_semantics["variable_owner_changed"] == "0"
    relation_semantics = next(
        row
        for row in semantics_rows
        if row["lane_id"] == "relation_dispatch_rule"
    )
    assert relation_semantics["coordination_mode_changed"] == "1"
    assert relation_semantics["variable_owner_changed"] == "0"

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
        "fixed_coordinate",
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
        "action_value_delta_norm",
        "downstream_consumed",
        "downstream_consumption_scope",
    }.issubset(trace_rows[0])

    decision_rows = _read_csv(output / "action_decision.csv")
    mismatch_rows = _read_csv(output / "action_mismatch_audit.csv")
    overlap_rows = _read_csv(output / "overlap_relations.csv")
    assert {row["relation_id"] for row in decision_rows} == {"O0_0_1", "O0_1_2"}
    assert {row["relation_id"] for row in mismatch_rows} == {"O0_0_1", "O0_1_2"}
    assert {
        "candidate_scores",
        "final_action_name",
        "second_best_action_name",
        "margin",
        "abstain_reason",
    }.issubset(mismatch_rows[0])
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
        "optimizer_consumed_action_mix",
        "claim_allowed",
        "claim_blockers",
    }.issubset(utility_rows[0])
    by_lane = {row["lane_id"]: row for row in utility_rows}
    assert by_lane["fixed_repair"]["utility_label"] == "meaningful_win"
    assert by_lane["fixed_repair"]["claim_allowed"] == "1"
    assert by_lane["fixed_coordinate"]["utility_label"] == "tie_or_small_effect"
    assert by_lane["fixed_coordinate"]["claim_allowed"] == "0"
    assert "utility_not_meaningful_win" in by_lane["fixed_coordinate"]["claim_blockers"]
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
            "relation_dispatch_mean_final_error": "1.280000e+02",
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
    coordinate_baseline = diagnosis_by_key["relation_vs_fixed_coordinate_baseline"]
    assert coordinate_baseline["status"] == "pass"
    assert "win_count=3/3" in coordinate_baseline["observed_value"]
    assert "fixed_coordinate_mean_gain_vs_fallback=" in coordinate_baseline["observed_value"]
    assert diagnosis_by_key["shuffled_negative_control"]["status"] == "blocked"
    assert diagnosis_by_key["negative_control_action_mix"]["status"] == "blocked"
    assert "relation_dispatch_rule=" in diagnosis_by_key["negative_control_action_mix"]["observed_value"]
    assert "shuffled_relation_dispatch=" in diagnosis_by_key["negative_control_action_mix"]["observed_value"]
    policy_profile = diagnosis_by_key["relation_policy_evidence_profile"]
    assert policy_profile["status"] == "pass"
    assert "relations=6" in policy_profile["observed_value"]
    assert "actions=allow_beneficial_coordination=3;repair_shared_variable_binding=3" in policy_profile["observed_value"]
    assert "reasons=repair=3;stable=3" in policy_profile["observed_value"]
    assert "active_density=1.000000" in policy_profile["observed_value"]
    assert "mean_active_confidence=0.750000" in policy_profile["observed_value"]
    assert policy_profile["next_step"] == "tune_policy_or_backend_effect_size"
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
    assert result_by_lane["fixed_coordinate"]["runtime_connected_claim_allowed"] == "1"
    assert result_by_lane["fixed_coordinate"]["utility_claim_allowed"] == "0"
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
    assert all(row["budget_aligned_fe_used"] == "2000" for row in ledger_rows)
    assert all(row["actual_fe_used"] == "2000" for row in ledger_rows)
    assert all(row["budget_limit"] == "2000" for row in ledger_rows)
    assert all(row["budget_limit_source"] == "experiment_config" for row in ledger_rows)
    assert all(row["same_budget_violation"] == "0" for row in ledger_rows)

    requests.clear()
    budget_output = run_hcc_runtime_consumer_smoke(
        output_dir=tmp_path / "exp003-budget",
        execution_runner=fake_runner,
        python_executable="python",
        seeds=(1,),
        problem_ids=("E2",),
        max_fes=3_000,
        budget_accounting="source",
    )

    assert {request.max_fes for request in requests} == {3_000}
    assert {request.budget_accounting for request in requests} == {"source"}
    budget_ledger_rows = _read_csv(budget_output / "same_budget_ledger.csv")
    assert {row["same_budget_group_id"] for row in budget_ledger_rows} == {
        "E2_seed1_3000fe"
    }
    assert all(row["configured_budget_limit"] == "3000" for row in budget_ledger_rows)
    assert all(row["actual_fe_used"] == "3000" for row in budget_ledger_rows)
    budget_manifest = (budget_output / "run_manifest.md").read_text(encoding="utf-8")
    assert "--max-fes 3000 --budget-accounting source" in budget_manifest
    assert "Budget: 3000 FE per lane/case" in budget_manifest
    assert "Budget accounting: source" in budget_manifest

    claim_rows = _read_csv(output / "claim_gate.csv")
    assert all(row["performance_claim_allowed"] == "0" for row in claim_rows)
    assert all(row["same_budget_violation"] == "0" for row in claim_rows)
    claim_by_lane = {row["lane_id"]: row for row in claim_rows}
    assert claim_by_lane["relation_dispatch_rule"]["runtime_connected_claim_allowed"] == "1"
    assert claim_by_lane["relation_dispatch_rule"]["utility_claim_allowed"] == "0"
    assert claim_by_lane["relation_dispatch_rule"]["claim_allowed"] == "0"
    assert "utility_not_meaningful_win" in claim_by_lane["relation_dispatch_rule"]["claim_blockers"]

    requests.clear()
    multi_output = run_hcc_runtime_consumer_smoke(
        output_dir=tmp_path / "exp003-multi",
        execution_runner=fake_runner,
        python_executable="python",
        seeds=(1,),
        problem_ids=("E1", "E2"),
        jobs=2,
    )

    assert len(requests) == 10
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
    assert aggregate_by_key["multi_problem_claim_scope"]["observed_value"] == (
        "overlap_applicable=E2;no_overlap_controls=E1"
    )
    assert aggregate_by_key["multi_problem_relation_dispatch_mean_gain"]["observed_value"] == (
        "positive_cases=0/1;mean_gain=-0.049587;lost_case_ids=E2_seed1"
    )
    lost_case_mix = aggregate_by_key["multi_problem_lost_case_action_mix"]
    assert lost_case_mix["status"] == "blocked"
    assert lost_case_mix["observed_value"] == (
        "lost_cases=1;mean_lost_gain=-0.049587;"
        "actions=allow_beneficial_coordination=1;repair_shared_variable_binding=1"
    )
    action_outcome = aggregate_by_key["multi_problem_action_outcome_profile"]
    assert action_outcome["status"] == "blocked"
    assert action_outcome["observed_value"] == (
        "wins=|losses=allow_beneficial_coordination=1;"
        "repair_shared_variable_binding=1|ties="
    )
    assert aggregate_by_key["multi_problem_relation_dispatch_win_count"]["observed_value"] == (
        "win_count=0/1;loss_count=1/1"
    )
    assert aggregate_by_key["multi_problem_relation_dispatch_win_count"]["status"] == "blocked"
    assert aggregate_by_key[
        "multi_problem_active_relation_dispatch_mean_gain"
    ]["observed_value"] == (
        "active_cases=1;positive_cases=0/1;mean_gain=-0.049587;"
        "lost_case_ids=E2_seed1"
    )
    assert aggregate_by_key[
        "multi_problem_active_relation_dispatch_mean_gain"
    ]["status"] == "blocked"
    assert aggregate_by_key["multi_problem_fixed_repair_baseline"]["observed_value"] == (
        "win_count=0/1;mean_gain=-0.567901;lost_case_ids=E2_seed1"
    )
    assert aggregate_by_key["multi_problem_fixed_repair_baseline"]["status"] == "blocked"
    assert aggregate_by_key[
        "multi_problem_relation_vs_fixed_coordinate_baseline"
    ]["observed_value"] == "win_count=1/1;mean_gain=0.030534;lost_case_ids="
    assert aggregate_by_key[
        "multi_problem_relation_vs_fixed_coordinate_baseline"
    ]["status"] == "pass"
    assert aggregate_by_key["multi_problem_backend_semantics_audit"]["observed_value"] == (
        "changed=4/4"
    )
    assert aggregate_by_key["multi_problem_backend_semantics_audit"]["status"] == "pass"
    assert aggregate_by_key["multi_problem_negative_control"]["observed_value"] == (
        "pass=0/1;shuffled_win_count=1/1;failed_problem_ids=E2"
    )
    assert aggregate_by_key["multi_problem_negative_control"]["status"] == "blocked"
    aggregate_action_mix = aggregate_by_key["multi_problem_negative_control_action_mix"]
    assert aggregate_action_mix["status"] == "blocked"
    assert aggregate_action_mix["observed_value"] == (
        "relation_dispatch_rule=allow_beneficial_coordination=1;repair_shared_variable_binding=1|"
        "shuffled_relation_dispatch=repair_shared_variable_binding=2"
    )
    assert "failed_problem_ids=E2" in aggregate_action_mix["blocker_reason"]
    assert aggregate_by_key["multi_problem_catastrophic_loss_gate"]["observed_value"] == "0/1"
    aggregate_policy_profile = aggregate_by_key["multi_problem_relation_policy_profile"]
    assert aggregate_policy_profile["status"] == "pass"
    assert "relations=2" in aggregate_policy_profile["observed_value"]
    assert "active_density=1.000000" in aggregate_policy_profile["observed_value"]
    assert "actions=allow_beneficial_coordination=1;repair_shared_variable_binding=1" in (
        aggregate_policy_profile["observed_value"]
    )
    assert "reasons=repair=1;stable=1" in aggregate_policy_profile["observed_value"]
    assert aggregate_by_key["multi_problem_sota_escalation_allowed"]["status"] == "blocked"
    assert aggregate_by_key["multi_problem_sota_escalation_allowed"]["observed_value"] == "0"
    assert "negative_control_failed" in aggregate_by_key[
        "multi_problem_sota_escalation_allowed"
    ]["blocker_reason"]
    assert "fixed_repair_baseline_not_beaten" in aggregate_by_key[
        "multi_problem_sota_escalation_allowed"
    ]["blocker_reason"]
    multi_manifest = (multi_output / "run_manifest.md").read_text(encoding="utf-8")
    assert "- claim scope: overlap_applicable=E2;no_overlap_controls=E1" in multi_manifest
    assert "- same-budget violations: 0/5" in multi_manifest
    assert (
        "- multi-problem active density: "
        "mean=1.000000;min=1.000000;low_density_cases=0/1;"
        "threshold=0.200000;low_density_case_ids="
    ) in multi_manifest
    assert (
        "- relation dispatch materiality: "
        "material_wins=0/1;material_losses=0/1;ties=1/1;"
        "mean_gain=-0.049587;threshold=0.050000"
    ) in multi_manifest
    assert (
        "- fixed repair baseline: "
        "win_count=0/1;mean_gain=-0.567901;lost_case_ids=E2_seed1"
    ) in multi_manifest
    assert (
        "- fixed repair materiality: "
        "material_wins=0/1;material_losses=1/1;ties=0/1"
    ) in multi_manifest
    assert (
        "- fixed coordinate baseline: "
        "win_count=1/1;mean_gain=0.030534;lost_case_ids="
    ) in multi_manifest
    assert (
        "- multi-problem relation policy profile: "
        "relations=2;active=2;active_density=1.000000"
    ) in multi_manifest
    multi_ledger = _read_csv(multi_output / "same_budget_ledger.csv")
    assert {row["same_budget_group_id"] for row in multi_ledger} == {
        "E1_seed1_2000fe",
        "E2_seed1_2000fe",
    }


def test_multi_problem_semantics_audit_allows_fallback_only_relation_dispatch() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_diagnosis_rows,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "final_error": str(final_error),
            "relative_gain_vs_fallback": gain,
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "backend_semantics_changed": changed,
            "action_mix": action_mix,
        }
        for problem_id, lane_id, final_error, gain, changed, action_mix in [
            ("E1", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("E1", "fixed_repair", 99.0, "0.010000", "1", "repair_shared_variable_binding=1"),
            ("E1", "fixed_coordinate", 99.0, "0.010000", "1", "allow_beneficial_coordination=1"),
            ("E1", "relation_dispatch_rule", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("E1", "shuffled_relation_dispatch", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("E2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("E2", "fixed_repair", 99.0, "0.010000", "1", "repair_shared_variable_binding=1"),
            ("E2", "fixed_coordinate", 99.0, "0.010000", "1", "allow_beneficial_coordination=1"),
            ("E2", "relation_dispatch_rule", 99.0, "0.010000", "1", "allow_beneficial_coordination=1"),
            ("E2", "shuffled_relation_dispatch", 99.0, "0.010000", "1", "allow_beneficial_coordination=1"),
        ]
    ]
    negative_rows = [
        {
            "problem_id": "E1",
            "negative_control_pass": "1",
            "shuffled_win_count": "0",
            "total_seeds": "1",
        },
        {
            "problem_id": "E2",
            "negative_control_pass": "1",
            "shuffled_win_count": "0",
            "total_seeds": "1",
        },
    ]

    rows = _multi_problem_diagnosis_rows(utility_rows, negative_rows)
    by_key = {row["diagnostic_key"]: row for row in rows}

    assert by_key["multi_problem_claim_scope"]["observed_value"] == (
        "overlap_applicable=E2;no_overlap_controls=E1"
    )
    assert by_key["multi_problem_backend_semantics_audit"]["observed_value"] == "changed=4/4"
    assert by_key["multi_problem_backend_semantics_audit"]["status"] == "pass"
    assert by_key["multi_problem_active_relation_dispatch_mean_gain"]["observed_value"] == (
        "active_cases=1;positive_cases=1/1;mean_gain=0.010000;lost_case_ids="
    )


def test_pilot_utility_evidence_is_separate_from_sota_claim_gate() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _policy_evidence_diagnosis_rows_for_problem,
    )

    utility_rows = [
        {
            "problem_id": "E2",
            "seed": "1",
            "lane_id": "fallback",
            "final_error": "100.000000",
            "relative_gain_vs_fallback": "0.000000",
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "action_mix": "conservative_no_action=1",
        },
        {
            "problem_id": "E2",
            "seed": "1",
            "lane_id": "fixed_coordinate",
            "final_error": "99.995000",
            "relative_gain_vs_fallback": "0.000050",
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "action_mix": "allow_beneficial_coordination=1",
        },
        {
            "problem_id": "E2",
            "seed": "1",
            "lane_id": "relation_dispatch_rule",
            "final_error": "99.990000",
            "relative_gain_vs_fallback": "0.000100",
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "action_mix": "allow_beneficial_coordination=1;conservative_no_action=2",
        },
    ]
    negative_control = {
        "negative_control_pass": "1",
        "diagnostic": "shuffled_control_not_stably_better",
    }

    rows = _policy_evidence_diagnosis_rows_for_problem(
        "E2",
        utility_rows,
        negative_control,
    )
    by_key = {row["diagnostic_key"]: row for row in rows}

    assert by_key["pilot_utility_evidence"]["status"] == "pass"
    assert by_key["pilot_utility_evidence"]["observed_value"] == (
        "directional=1/1;mean_gain=0.000100;negative_control=1;catastrophic=0/1"
    )
    assert by_key["sota_escalation_allowed"]["status"] == "blocked"
    assert by_key["sota_escalation_allowed"]["blocker_reason"] == (
        "relation_dispatch_not_meaningful_win"
    )


def test_problem_diagnostics_use_directional_gate_for_pilot_and_coordinate() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _policy_evidence_diagnosis_rows_for_problem,
    )

    utility_rows = [
        {
            "problem_id": "E2",
            "seed": seed,
            "lane_id": lane_id,
            "final_error": str(final_error),
            "relative_gain_vs_fallback": gain,
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "action_mix": action_mix,
        }
        for seed, lane_id, final_error, gain, action_mix in [
            ("1", "fallback", 100.0, "0.000000", "conservative_no_action=1"),
            ("1", "fixed_coordinate", 100.0, "0.000000", "allow_beneficial_coordination=1"),
            ("1", "relation_dispatch_rule", 99.0, "0.010000", "allow_beneficial_coordination=1"),
            ("2", "fallback", 100.0, "0.000000", "conservative_no_action=1"),
            ("2", "fixed_coordinate", 100.0, "0.000000", "allow_beneficial_coordination=1"),
            ("2", "relation_dispatch_rule", 99.5, "0.005000", "allow_beneficial_coordination=1"),
            ("3", "fallback", 100.0, "0.000000", "conservative_no_action=1"),
            ("3", "fixed_coordinate", 100.0, "0.000000", "allow_beneficial_coordination=1"),
            ("3", "relation_dispatch_rule", 100.2, "-0.002000", "allow_beneficial_coordination=1"),
        ]
    ]
    negative_control = {
        "negative_control_pass": "1",
        "diagnostic": "shuffled_control_not_stably_better",
    }

    rows = _policy_evidence_diagnosis_rows_for_problem(
        "E2",
        utility_rows,
        negative_control,
    )
    by_key = {row["diagnostic_key"]: row for row in rows}

    assert by_key["relation_dispatch_directional_utility"]["status"] == "pass"
    assert by_key["relation_dispatch_directional_utility"]["observed_value"] == (
        "2/3;mean_gain=0.004333"
    )
    assert by_key["pilot_utility_evidence"]["status"] == "pass"
    assert by_key["relation_vs_fixed_coordinate_baseline"]["status"] == "pass"
    assert by_key["relation_vs_fixed_coordinate_baseline"]["observed_value"] == (
        "win_count=2/3;mean_gain_vs_fixed_coordinate=0.004333;"
        "fixed_coordinate_mean_gain_vs_fallback=0.000000"
    )


def test_multi_problem_pilot_utility_evidence_is_separate_from_sota_gate() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_diagnosis_rows,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "final_error": str(final_error),
            "relative_gain_vs_fallback": gain,
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "backend_semantics_changed": changed,
            "action_mix": action_mix,
        }
        for problem_id, lane_id, final_error, gain, changed, action_mix in [
            ("E2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("E2", "fixed_repair", 99.995, "0.000050", "1", "repair_shared_variable_binding=1"),
            ("E2", "fixed_coordinate", 99.995, "0.000050", "1", "allow_beneficial_coordination=1"),
            ("E2", "relation_dispatch_rule", 99.990, "0.000100", "1", "allow_beneficial_coordination=1;conservative_no_action=4"),
            ("S2", "fallback", 200.0, "0.000000", "0", "conservative_no_action=1"),
            ("S2", "fixed_repair", 199.990, "0.000050", "1", "repair_shared_variable_binding=1"),
            ("S2", "fixed_coordinate", 199.990, "0.000050", "1", "allow_beneficial_coordination=1"),
            ("S2", "relation_dispatch_rule", 199.980, "0.000100", "1", "allow_beneficial_coordination=1"),
        ]
    ]
    negative_rows = [
        {
            "problem_id": problem_id,
            "negative_control_pass": "1",
            "shuffled_win_count": "0",
            "total_seeds": "1",
        }
        for problem_id in ("E2", "S2")
    ]

    rows = _multi_problem_diagnosis_rows(utility_rows, negative_rows)
    by_key = {row["diagnostic_key"]: row for row in rows}

    assert by_key["multi_problem_pilot_utility_evidence"]["status"] == "pass"
    assert by_key["multi_problem_pilot_utility_evidence"]["observed_value"] == (
        "directional=2/2;mean_gain=0.000100;negative_control=2/2;catastrophic=0/2"
    )
    assert by_key["multi_problem_active_density_profile"]["status"] == "blocked"
    assert by_key["multi_problem_active_density_profile"]["observed_value"] == (
        "mean=0.600000;min=0.200000;low_density_cases=1/2;"
        "threshold=0.200000;low_density_case_ids=E2_seed1"
    )
    assert by_key["multi_problem_sota_escalation_allowed"]["status"] == "blocked"
    assert by_key["multi_problem_sota_escalation_allowed"]["blocker_reason"] == (
        "relation_dispatch_effect_size_below_threshold"
    )
    claim_tier = by_key["multi_problem_claim_tier_recommendation"]
    assert claim_tier["status"] == "pass"
    assert claim_tier["observed_value"] == (
        "runtime_evidence_driven_relation_dispatch_with_positive_utility_evidence"
    )
    assert claim_tier["blocker_reason"] == "sota_gate_blocked"


def test_multi_problem_pilot_utility_allows_positive_mean_with_more_wins_than_losses() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_diagnosis_rows,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "final_error": str(final_error),
            "relative_gain_vs_fallback": gain,
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "backend_semantics_changed": changed,
            "action_mix": action_mix,
        }
        for problem_id, lane_id, final_error, gain, changed, action_mix in [
            ("E2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("E2", "fixed_repair", 100.0, "0.000000", "1", "repair_shared_variable_binding=1"),
            ("E2", "fixed_coordinate", 100.0, "0.000000", "1", "allow_beneficial_coordination=1"),
            ("E2", "relation_dispatch_rule", 99.0, "0.010000", "1", "allow_beneficial_coordination=1"),
            ("S2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("S2", "fixed_repair", 100.0, "0.000000", "1", "repair_shared_variable_binding=1"),
            ("S2", "fixed_coordinate", 100.0, "0.000000", "1", "allow_beneficial_coordination=1"),
            ("S2", "relation_dispatch_rule", 99.5, "0.005000", "1", "allow_beneficial_coordination=1"),
            ("R2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("R2", "fixed_repair", 100.0, "0.000000", "1", "repair_shared_variable_binding=1"),
            ("R2", "fixed_coordinate", 100.0, "0.000000", "1", "allow_beneficial_coordination=1"),
            ("R2", "relation_dispatch_rule", 100.2, "-0.002000", "1", "allow_beneficial_coordination=1"),
        ]
    ]
    negative_rows = [
        {
            "problem_id": problem_id,
            "negative_control_pass": "1",
            "shuffled_win_count": "0",
            "total_seeds": "1",
        }
        for problem_id in ("E2", "S2", "R2")
    ]

    rows = _multi_problem_diagnosis_rows(utility_rows, negative_rows)
    by_key = {row["diagnostic_key"]: row for row in rows}

    assert by_key["multi_problem_relation_dispatch_mean_gain"]["status"] == "pass"
    assert by_key["multi_problem_relation_dispatch_mean_gain"]["observed_value"] == (
        "positive_cases=2/3;mean_gain=0.004333;lost_case_ids=R2_seed1"
    )
    assert by_key["multi_problem_pilot_utility_evidence"]["status"] == "pass"
    assert by_key["multi_problem_pilot_utility_evidence"]["observed_value"] == (
        "directional=2/3;mean_gain=0.004333;negative_control=3/3;catastrophic=0/3"
    )
    assert by_key["multi_problem_relation_dispatch_win_count"]["observed_value"] == (
        "win_count=2/3;loss_count=1/3"
    )
    assert by_key["multi_problem_sota_escalation_allowed"]["status"] == "blocked"
    assert "relation_dispatch_effect_size_below_threshold" in by_key[
        "multi_problem_sota_escalation_allowed"
    ]["blocker_reason"]


def test_multi_problem_diagnostics_report_action_value_delta_profile() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_diagnosis_rows,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": "relation_dispatch_rule",
            "final_error": "99.0",
            "relative_gain_vs_fallback": gain,
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "backend_semantics_changed": "1",
            "action_mix": action_mix,
        }
        for problem_id, gain, action_mix in [
            ("E2", "0.010000", "allow_beneficial_coordination=1;conservative_no_action=1"),
            ("S2", "-0.010000", "repair_shared_variable_binding=1"),
        ]
    ]
    negative_rows = [
        {
            "problem_id": problem_id,
            "negative_control_pass": "1",
            "shuffled_win_count": "0",
            "total_seeds": "1",
        }
        for problem_id in ("E2", "S2")
    ]
    trace_rows = [
        {
            "problem_id": "E2",
            "seed": "1",
            "lane_id": "relation_dispatch_rule",
            "canonical_action_name": "allow_beneficial_coordination",
            "action_value_delta_norm": "2.000000e+00",
        },
        {
            "problem_id": "E2",
            "seed": "1",
            "lane_id": "relation_dispatch_rule",
            "canonical_action_name": "conservative_no_action",
            "action_value_delta_norm": "5.000000e-01",
        },
        {
            "problem_id": "S2",
            "seed": "1",
            "lane_id": "relation_dispatch_rule",
            "canonical_action_name": "repair_shared_variable_binding",
            "action_value_delta_norm": "1.000000e+00",
        },
    ]

    rows = _multi_problem_diagnosis_rows(utility_rows, negative_rows, trace_rows)
    by_key = {row["diagnostic_key"]: row for row in rows}

    value_delta = by_key["multi_problem_action_value_delta_profile"]
    assert value_delta["status"] == "blocked"
    assert value_delta["observed_value"] == (
        "wins=allow_beneficial_coordination:n=1,mean=2.000000,max=2.000000;"
        "conservative_no_action:n=1,mean=0.500000,max=0.500000|"
        "losses=repair_shared_variable_binding:n=1,mean=1.000000,max=1.000000|"
        "ties="
    )
    assert value_delta["blocker_reason"] == "relation_dispatch_lost_cases"


def test_multi_problem_baseline_diagnostics_report_lost_case_ids() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_diagnosis_rows,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "final_error": str(final_error),
            "relative_gain_vs_fallback": gain,
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "backend_semantics_changed": changed,
            "action_mix": action_mix,
        }
        for problem_id, lane_id, final_error, gain, changed, action_mix in [
            ("A2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("A2", "fixed_repair", 99.0, "0.010000", "1", "repair_shared_variable_binding=1"),
            ("A2", "fixed_coordinate", 98.0, "0.020000", "1", "allow_beneficial_coordination=1"),
            ("A2", "relation_dispatch_rule", 99.5, "0.005000", "1", "allow_beneficial_coordination=1"),
            ("E2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("E2", "fixed_repair", 99.5, "0.005000", "1", "repair_shared_variable_binding=1"),
            ("E2", "fixed_coordinate", 99.5, "0.005000", "1", "allow_beneficial_coordination=1"),
            ("E2", "relation_dispatch_rule", 99.0, "0.010000", "1", "allow_beneficial_coordination=1"),
        ]
    ]
    negative_rows = [
        {
            "problem_id": problem_id,
            "negative_control_pass": "1",
            "shuffled_win_count": "0",
            "total_seeds": "1",
        }
        for problem_id in ("A2", "E2")
    ]

    rows = _multi_problem_diagnosis_rows(utility_rows, negative_rows)
    by_key = {row["diagnostic_key"]: row for row in rows}

    assert by_key["multi_problem_fixed_repair_baseline"]["observed_value"] == (
        "win_count=1/2;mean_gain=-0.000013;lost_case_ids=A2_seed1"
    )
    assert by_key["multi_problem_relation_vs_fixed_coordinate_baseline"]["observed_value"] == (
        "win_count=1/2;mean_gain=-0.005140;lost_case_ids=A2_seed1"
    )


def test_multi_problem_fixed_coordinate_baseline_uses_directional_gate() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_diagnosis_rows,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "final_error": str(final_error),
            "relative_gain_vs_fallback": gain,
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "backend_semantics_changed": changed,
            "action_mix": action_mix,
        }
        for problem_id, lane_id, final_error, gain, changed, action_mix in [
            ("E2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("E2", "fixed_repair", 99.0, "0.010000", "1", "repair_shared_variable_binding=1"),
            ("E2", "fixed_coordinate", 100.0, "0.000000", "1", "allow_beneficial_coordination=1"),
            ("E2", "relation_dispatch_rule", 99.0, "0.010000", "1", "allow_beneficial_coordination=1"),
            ("S2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("S2", "fixed_repair", 99.0, "0.010000", "1", "repair_shared_variable_binding=1"),
            ("S2", "fixed_coordinate", 100.0, "0.000000", "1", "allow_beneficial_coordination=1"),
            ("S2", "relation_dispatch_rule", 99.5, "0.005000", "1", "allow_beneficial_coordination=1"),
            ("R2", "fallback", 100.0, "0.000000", "0", "conservative_no_action=1"),
            ("R2", "fixed_repair", 99.0, "0.010000", "1", "repair_shared_variable_binding=1"),
            ("R2", "fixed_coordinate", 100.0, "0.000000", "1", "allow_beneficial_coordination=1"),
            ("R2", "relation_dispatch_rule", 100.2, "-0.002000", "1", "allow_beneficial_coordination=1"),
        ]
    ]
    negative_rows = [
        {
            "problem_id": problem_id,
            "negative_control_pass": "1",
            "shuffled_win_count": "0",
            "total_seeds": "1",
        }
        for problem_id in ("E2", "S2", "R2")
    ]

    rows = _multi_problem_diagnosis_rows(utility_rows, negative_rows)
    by_key = {row["diagnostic_key"]: row for row in rows}

    coordinate = by_key["multi_problem_relation_vs_fixed_coordinate_baseline"]
    assert coordinate["status"] == "pass"
    assert coordinate["observed_value"] == (
        "win_count=2/3;mean_gain=0.004333;lost_case_ids=R2_seed1"
    )


def test_multi_problem_diagnostics_report_fixed_repair_materiality() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_diagnosis_rows,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "final_error": str(final_error),
            "relative_gain_vs_fallback": gain,
            "utility_label": "tie_or_small_effect",
            "same_budget_violation": "0",
            "backend_semantics_changed": "1" if lane_id != "fallback" else "0",
            "action_mix": action_mix,
        }
        for problem_id, lane_id, final_error, gain, action_mix in [
            ("E2", "fallback", 100.0, "0.000000", "conservative_no_action=1"),
            ("E2", "fixed_repair", 100.0, "0.000000", "repair_shared_variable_binding=1"),
            ("E2", "relation_dispatch_rule", 94.0, "0.060000", "allow_beneficial_coordination=1"),
            ("S2", "fallback", 100.0, "0.000000", "conservative_no_action=1"),
            ("S2", "fixed_repair", 100.0, "0.000000", "repair_shared_variable_binding=1"),
            ("S2", "relation_dispatch_rule", 101.0, "-0.010000", "conservative_no_action=1"),
            ("R2", "fallback", 100.0, "0.000000", "conservative_no_action=1"),
            ("R2", "fixed_repair", 100.0, "0.000000", "repair_shared_variable_binding=1"),
            ("R2", "relation_dispatch_rule", 125.0, "-0.250000", "reassign_repair=1"),
        ]
    ]
    negative_rows = [
        {
            "problem_id": problem_id,
            "negative_control_pass": "1",
            "shuffled_win_count": "0",
            "total_seeds": "1",
        }
        for problem_id in ("E2", "S2", "R2")
    ]

    rows = _multi_problem_diagnosis_rows(utility_rows, negative_rows)
    by_key = {row["diagnostic_key"]: row for row in rows}

    materiality = by_key["multi_problem_fixed_repair_materiality"]
    assert materiality["status"] == "blocked"
    assert materiality["observed_value"] == (
        "material_wins=1/3;material_losses=1/3;ties=1/3"
    )
    assert materiality["blocker_reason"] == "fixed_repair_material_loss_detected"


def test_multi_problem_diagnostics_report_relation_dispatch_materiality() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_diagnosis_rows,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "final_error": str(final_error),
            "relative_gain_vs_fallback": gain,
            "utility_label": utility_label,
            "same_budget_violation": "0",
            "backend_semantics_changed": "1" if lane_id != "fallback" else "0",
            "action_mix": action_mix,
        }
        for problem_id, lane_id, final_error, gain, utility_label, action_mix in [
            ("E2", "fallback", 100.0, "0.000000", "tie_or_small_effect", "conservative_no_action=1"),
            ("E2", "fixed_repair", 100.0, "0.000000", "tie_or_small_effect", "repair_shared_variable_binding=1"),
            ("E2", "fixed_coordinate", 100.0, "0.000000", "tie_or_small_effect", "allow_beneficial_coordination=1"),
            ("E2", "relation_dispatch_rule", 94.0, "0.060000", "meaningful_win", "allow_beneficial_coordination=1"),
            ("S2", "fallback", 100.0, "0.000000", "tie_or_small_effect", "conservative_no_action=1"),
            ("S2", "fixed_repair", 100.0, "0.000000", "tie_or_small_effect", "repair_shared_variable_binding=1"),
            ("S2", "fixed_coordinate", 100.0, "0.000000", "tie_or_small_effect", "allow_beneficial_coordination=1"),
            ("S2", "relation_dispatch_rule", 99.0, "0.010000", "tie_or_small_effect", "conservative_no_action=1"),
            ("R2", "fallback", 100.0, "0.000000", "tie_or_small_effect", "conservative_no_action=1"),
            ("R2", "fixed_repair", 100.0, "0.000000", "tie_or_small_effect", "repair_shared_variable_binding=1"),
            ("R2", "fixed_coordinate", 100.0, "0.000000", "tie_or_small_effect", "allow_beneficial_coordination=1"),
            ("R2", "relation_dispatch_rule", 125.0, "-0.250000", "catastrophic_loss", "reassign_repair=1"),
        ]
    ]
    negative_rows = [
        {
            "problem_id": problem_id,
            "negative_control_pass": "1",
            "shuffled_win_count": "0",
            "total_seeds": "1",
        }
        for problem_id in ("E2", "S2", "R2")
    ]

    rows = _multi_problem_diagnosis_rows(utility_rows, negative_rows)
    by_key = {row["diagnostic_key"]: row for row in rows}

    materiality = by_key["multi_problem_relation_dispatch_materiality"]
    assert materiality["status"] == "blocked"
    assert materiality["observed_value"] == (
        "material_wins=1/3;material_losses=1/3;ties=1/3;"
        "mean_gain=-0.060000;threshold=0.050000"
    )
    assert materiality["blocker_reason"] == "relation_dispatch_material_loss_detected"


def test_multi_problem_trigger_outcome_profile_groups_reasons_by_case_gain() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_trigger_outcome_profile_row,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": "relation_dispatch_rule",
            "relative_gain_vs_fallback": gain,
        }
        for problem_id, gain in [
            ("E1", "0.500000"),
            ("E2", "0.010000"),
            ("S2", "-0.010000"),
            ("R2", "0.000000"),
        ]
    ]
    decision_rows = [
        {
            "lane_id": "relation_dispatch_rule",
            "problem_id": problem_id,
            "seed": "1",
            "trigger_reason": trigger_reason,
        }
        for problem_id, trigger_reason in [
            ("E1", "ignored_no_overlap"),
            ("E2", "balanced_mid_support_coordinate_mode"),
            ("E2", "dense_prefix_coordinate_mode"),
            ("S2", "dense_prefix_coordinate_mode"),
            ("R2", "balanced_mid_support_coordinate_mode"),
        ]
    ]

    row = _multi_problem_trigger_outcome_profile_row(utility_rows, decision_rows)

    assert row["diagnostic_key"] == "multi_problem_trigger_outcome_profile"
    assert row["status"] == "blocked"
    assert row["observed_value"] == (
        "balanced_mid_support_coordinate_mode=win:1,loss:0,tie:1;"
        "dense_prefix_coordinate_mode=win:1,loss:1,tie:0"
    )
    assert row["blocker_reason"] == "relation_dispatch_lost_cases"


def test_multi_problem_trigger_baseline_gap_profile_reports_strong_baseline_gaps() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_trigger_baseline_gap_profile_row,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "final_error": str(final_error),
        }
        for problem_id, lane_id, final_error in [
            ("E1", "relation_dispatch_rule", 1.0),
            ("E2", "relation_dispatch_rule", 90.0),
            ("E2", "fixed_repair", 95.0),
            ("E2", "fixed_coordinate", 100.0),
            ("S2", "relation_dispatch_rule", 110.0),
            ("S2", "fixed_repair", 100.0),
            ("S2", "fixed_coordinate", 100.0),
            ("R2", "relation_dispatch_rule", 90.0),
            ("R2", "fixed_repair", 100.0),
            ("R2", "fixed_coordinate", 100.0),
        ]
    ]
    decision_rows = [
        {
            "lane_id": "relation_dispatch_rule",
            "problem_id": problem_id,
            "seed": "1",
            "trigger_reason": trigger_reason,
        }
        for problem_id, trigger_reason in [
            ("E1", "ignored_no_overlap"),
            ("E2", "dense_prefix_coordinate_mode"),
            ("S2", "dense_prefix_coordinate_mode"),
            ("R2", "balanced_mid_support_coordinate_mode"),
        ]
    ]

    row = _multi_problem_trigger_baseline_gap_profile_row(utility_rows, decision_rows)

    assert row["diagnostic_key"] == "multi_problem_trigger_baseline_gap_profile"
    assert row["status"] == "blocked"
    assert row["observed_value"] == (
        "balanced_mid_support_coordinate_mode=relations:1,"
        "vs_fixed_repair_mean=0.100000,vs_fixed_coordinate_mean=0.100000;"
        "dense_prefix_coordinate_mode=relations:2,"
        "vs_fixed_repair_mean=-0.023684,vs_fixed_coordinate_mean=0.000000"
    )
    assert row["blocker_reason"] == "trigger_baseline_gap_detected"


def test_multi_problem_no_overlap_control_reports_unwanted_relation_activity() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_no_overlap_control_row,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "same_budget_violation": "0",
        }
        for problem_id, lane_id in [
            ("E1", "fallback"),
            ("E1", "relation_dispatch_rule"),
            ("E1", "fixed_repair"),
            ("E1", "fixed_coordinate"),
            ("E1", "shuffled_relation_dispatch"),
            ("E2", "relation_dispatch_rule"),
        ]
    ]

    clean = _multi_problem_no_overlap_control_row(utility_rows, [], [])

    assert clean is not None
    assert clean["diagnostic_key"] == "multi_problem_no_overlap_control"
    assert clean["status"] == "pass"
    assert clean["observed_value"] == (
        "controls=E1;relation_rows=0;active_relation_actions=0;"
        "same_budget_violations=0/5"
    )

    zero_support = _multi_problem_no_overlap_control_row(
        utility_rows,
        [],
        [
            {
                "problem_id": "E1",
                "lane_id": "relation_dispatch_rule",
                "relation_id": "O0_0_1",
                "shared_var_count": "0",
                "overlap_strength": "0.000000",
                "shared_vars": "",
            }
        ],
    )

    assert zero_support is not None
    assert zero_support["status"] == "pass"
    assert zero_support["observed_value"] == (
        "controls=E1;relation_rows=0;active_relation_actions=0;"
        "same_budget_violations=0/5"
    )

    blocked = _multi_problem_no_overlap_control_row(
        utility_rows,
        [
            {
                "problem_id": "E1",
                "lane_id": "relation_dispatch_rule",
                "canonical_action_name": "allow_beneficial_coordination",
            }
        ],
        [
            {
                "problem_id": "E1",
                "lane_id": "relation_dispatch_rule",
                "relation_id": "O0_0_1",
            }
        ],
    )

    assert blocked is not None
    assert blocked["status"] == "blocked"
    assert blocked["blocker_reason"] == (
        "no_overlap_relation_rows_detected;"
        "no_overlap_active_relation_actions_detected"
    )


def test_multi_problem_action_baseline_gap_profile_reports_action_gaps() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_action_baseline_gap_profile_row,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": "1",
            "lane_id": lane_id,
            "final_error": str(final_error),
        }
        for problem_id, lane_id, final_error in [
            ("E1", "relation_dispatch_rule", 1.0),
            ("E2", "relation_dispatch_rule", 90.0),
            ("E2", "fixed_repair", 95.0),
            ("E2", "fixed_coordinate", 100.0),
            ("S2", "relation_dispatch_rule", 110.0),
            ("S2", "fixed_repair", 100.0),
            ("S2", "fixed_coordinate", 100.0),
            ("R2", "relation_dispatch_rule", 90.0),
            ("R2", "fixed_repair", 100.0),
            ("R2", "fixed_coordinate", 100.0),
        ]
    ]
    trace_rows = [
        {
            "lane_id": "relation_dispatch_rule",
            "problem_id": problem_id,
            "seed": "1",
            "canonical_action_name": action_name,
            "action_value_delta_norm": str(delta_norm),
        }
        for problem_id, action_name, delta_norm in [
            ("E1", "ignored_no_overlap", 0.0),
            ("E2", "allow_beneficial_coordination", 1.0),
            ("S2", "conservative_no_action", 0.0),
            ("R2", "allow_beneficial_coordination", 2.0),
        ]
    ]

    row = _multi_problem_action_baseline_gap_profile_row(utility_rows, trace_rows)

    assert row["diagnostic_key"] == "multi_problem_action_baseline_gap_profile"
    assert row["status"] == "blocked"
    assert row["observed_value"] == (
        "allow_beneficial_coordination=relations:2,"
        "vs_fixed_repair_mean=0.076316,vs_fixed_coordinate_mean=0.100000,"
        "mean_action_value_delta_norm=1.500000;"
        "conservative_no_action=relations:1,"
        "vs_fixed_repair_mean=-0.100000,vs_fixed_coordinate_mean=-0.100000,"
        "mean_action_value_delta_norm=0.000000"
    )
    assert row["blocker_reason"] == "action_baseline_gap_detected"


def test_multi_problem_action_mismatch_profile_summarizes_candidate_gaps() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_action_mismatch_profile_row,
    )

    mismatch_rows = [
        {
            "lane_id": "relation_dispatch_rule",
            "problem_id": "E1",
            "final_action_name": "fallback",
            "best_action_name": "fallback",
            "margin": "0.000000",
            "abstain_reason": "",
        },
        {
            "lane_id": "relation_dispatch_rule",
            "problem_id": "E2",
            "final_action_name": "coordinate",
            "best_action_name": "fallback",
            "margin": "0.040000",
            "abstain_reason": "",
        },
        {
            "lane_id": "relation_dispatch_rule",
            "problem_id": "S2",
            "final_action_name": "fallback",
            "best_action_name": "coordinate",
            "margin": "0.030000",
            "abstain_reason": "candidate_margin_below_threshold",
        },
        {
            "lane_id": "relation_dispatch_rule",
            "problem_id": "R2",
            "final_action_name": "coordinate",
            "best_action_name": "coordinate",
            "margin": "0.100000",
            "abstain_reason": "",
        },
    ]

    row = _multi_problem_action_mismatch_profile_row(mismatch_rows)

    assert row["diagnostic_key"] == "multi_problem_action_mismatch_profile"
    assert row["status"] == "blocked"
    assert row["observed_value"] == (
        "rows=3;final_best_mismatch=2;abstains=1;"
        "mean_margin=0.056667;final_actions=coordinate=2,fallback=1;"
        "best_actions=coordinate=2,fallback=1;"
        "abstain_reasons=candidate_margin_below_threshold=1"
    )
    assert row["blocker_reason"] == "action_mismatch_or_abstain_detected"


def test_multi_problem_mismatch_baseline_gap_profile_reports_final_best_gaps() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_mismatch_baseline_gap_profile_row,
    )

    utility_rows = [
        {
            "problem_id": problem_id,
            "seed": seed,
            "lane_id": lane_id,
            "final_error": str(final_error),
        }
        for problem_id, seed, lane_id, final_error in [
            ("E1", "1", "relation_dispatch_rule", 50.0),
            ("E2", "1", "relation_dispatch_rule", 90.0),
            ("E2", "1", "fixed_repair", 80.0),
            ("E2", "1", "fixed_coordinate", 95.0),
            ("E2", "2", "relation_dispatch_rule", 90.0),
            ("E2", "2", "fixed_repair", 100.0),
            ("E2", "2", "fixed_coordinate", 80.0),
        ]
    ]
    mismatch_rows = [
        {
            "problem_id": "E1",
            "seed": "1",
            "lane_id": "relation_dispatch_rule",
            "final_action_name": "fallback",
            "best_action_name": "fallback",
        },
        {
            "problem_id": "E2",
            "seed": "1",
            "lane_id": "relation_dispatch_rule",
            "final_action_name": "fallback",
            "best_action_name": "coordinate",
        },
        {
            "problem_id": "E2",
            "seed": "2",
            "lane_id": "relation_dispatch_rule",
            "final_action_name": "coordinate",
            "best_action_name": "coordinate",
        },
    ]

    row = _multi_problem_mismatch_baseline_gap_profile_row(
        utility_rows,
        mismatch_rows,
    )

    assert row["diagnostic_key"] == "multi_problem_mismatch_baseline_gap_profile"
    assert row["status"] == "blocked"
    assert row["observed_value"] == (
        "coordinate->coordinate=relations:1,"
        "vs_fixed_repair_mean=0.100000,"
        "vs_fixed_coordinate_mean=-0.125000;"
        "fallback->coordinate=relations:1,"
        "vs_fixed_repair_mean=-0.125000,"
        "vs_fixed_coordinate_mean=0.052632"
    )
    assert row["blocker_reason"] == "mismatch_baseline_gap_detected"


def test_multi_problem_relation_confidence_interval_reports_baseline_deltas() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _multi_problem_relation_confidence_interval_row,
    )

    utility_rows = []
    for seed in ("1", "2", "3"):
        utility_rows.extend(
            [
                {
                    "problem_id": "E2",
                    "seed": seed,
                    "lane_id": "fallback",
                    "final_error": "100.0",
                },
                {
                    "problem_id": "E2",
                    "seed": seed,
                    "lane_id": "fixed_repair",
                    "final_error": "95.0",
                },
                {
                    "problem_id": "E2",
                    "seed": seed,
                    "lane_id": "fixed_coordinate",
                    "final_error": "92.0",
                },
                {
                    "problem_id": "E2",
                    "seed": seed,
                    "lane_id": "shuffled_relation_dispatch",
                    "final_error": "91.0",
                },
                {
                    "problem_id": "E2",
                    "seed": seed,
                    "lane_id": "relation_dispatch_rule",
                    "final_error": "90.0",
                    "relative_gain_vs_fallback": "0.100000",
                },
            ]
        )

    row = _multi_problem_relation_confidence_interval_row(utility_rows)

    assert row["diagnostic_key"] == "multi_problem_relation_dispatch_confidence_interval"
    assert row["status"] == "pass"
    assert row["observed_value"] == (
        "vs_fallback:n=3,mean=0.100000,ci95=[0.100000,0.100000];"
        "vs_fixed_repair:n=3,mean=0.052632,ci95=[0.052632,0.052632];"
        "vs_fixed_coordinate:n=3,mean=0.021739,ci95=[0.021739,0.021739];"
        "vs_shuffled_relation_dispatch:n=3,mean=0.010989,ci95=[0.010989,0.010989]"
    )


def test_backend_semantics_expectation_uses_optimizer_consumed_action_mix() -> None:
    from experiments.exp_003_hcc_runtime_consumer_smoke.run import (
        _expects_backend_semantics,
    )

    row = {
        "lane_id": "relation_dispatch_rule",
        "action_mix": "allow_beneficial_coordination=1;conservative_no_action=24",
        "optimizer_consumed_action_mix": "conservative_no_action=24",
    }

    assert _expects_backend_semantics(row) is False

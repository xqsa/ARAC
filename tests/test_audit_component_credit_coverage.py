from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_component_credit_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_component_credit_coverage", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _valid_run_inputs() -> tuple[
    dict[str, str],
    dict[str, str],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, str],
]:
    result = {
        "problem_id": "A4",
        "seed": "31",
        "hcc_smoke_final_error": "1.000000e+01",
        "hcc_smoke_fe_used": "100",
        "hcc_smoke_status": "completed",
        "fresh_optimizer_execution": "1",
    }
    ledger = {
        "problem_id": "A4",
        "seed": "31",
        "actual_fe_used": "100",
        "total_fe": "100",
        "budget_limit": "100",
        "configured_budget_limit": "100",
        "same_budget_violation": "0",
        "fresh_execution": "1",
    }
    relation = {
        "problem_id": "A4",
        "seed": "31",
        "selected_action_name": "conservative_no_action",
        "component_id": "component_1",
        "component_group_count": "2",
        "component_shared_var_count": "1",
        "component_action_id": "",
        "component_action_scope": "shared_relation_observation",
        "component_credit_status": "relation_observation",
        "component_decision_fe": "5",
        "component_remaining_budget_ratio": "9.500000e-01",
        "component_proposal_disagreement": "2.000000e-01",
        "component_credit_reason": "paired_shared_value_proposals_observed",
    }
    resolved = {
        "problem_id": "A4",
        "seed": "31",
        "selected_action_name": "post_retirement_precision_reanchor",
        "component_id": "component_1",
        "component_group_count": "2",
        "component_shared_var_count": "1",
        "component_action_id": (
            "post_retirement_precision_reanchor:component_1:1:0:1"
        ),
        "component_action_scope": "group_search_start_component_credit",
        "component_credit_status": "resolved",
        "component_decision_fe": "10",
        "component_remaining_budget_ratio": "9.000000e-01",
        "component_resolution_fe": "20",
        "component_resolution_delay_fe": "10",
        "component_resolution_window": "next_canonical_group_revisit",
        "component_pending_before": "0",
        "component_lock_conflict": "0",
        "component_proposal_disagreement": "2.000000e-01",
        "component_local_gain": "1.000000e-01",
        "component_gain": "2.000000e-01",
        "component_neighbor_gain": "1.000000e-01",
        "component_neighbor_spillover": "0.000000e+00",
        "shared_var_overwrite_rate": "5.000000e-01",
        "shared_var_survival_rate": "5.000000e-01",
        "component_credit_reason": "resolved_next_canonical_group_revisit",
    }
    unresolved = {
        **resolved,
        "component_action_id": (
            "post_retirement_precision_reanchor:component_1:2:1:2"
        ),
        "component_credit_status": "unresolved_run_end",
        "component_decision_fe": "90",
        "component_remaining_budget_ratio": "1.000000e-01",
        "component_resolution_fe": "100",
        "component_resolution_delay_fe": "10",
        "component_pending_before": "1",
        "component_lock_conflict": "1",
        "component_gain": "",
        "component_neighbor_gain": "",
        "component_neighbor_spillover": "",
        "shared_var_overwrite_rate": "",
        "shared_var_survival_rate": "",
        "component_credit_reason": "budget_ended_before_next_group_revisit",
    }
    aob_rows = [
        {
            "problem_id": "A4",
            "seed": "31",
            "file": "F4-R1.txt",
            "sha256_before": "abc",
            "sha256_after": "abc",
            "unchanged": "1",
        }
    ]
    plan = {
        "problem_id": "A4",
        "optimizer_consumed": "1",
        "runtime_dispatch_allowed": "1",
        "optimizer_consumed_parameters": json.dumps(
            {
                "optimizer_runtime_hook": "post_retirement_precision_reanchor",
                "trace_runtime_hook": (
                    "component_locked_action_specific_delayed_credit"
                ),
                "trace_affects_dispatch": False,
                "dispatch_boundary": "runtime_evidence_only",
            }
        ),
    }
    return result, ledger, [relation, resolved, unresolved], aob_rows, plan


def test_audit_run_accepts_resolved_unresolved_and_overwrite_semantics() -> None:
    result, ledger, trace_rows, aob_rows, plan = _valid_run_inputs()

    summary, blockers = MODULE.audit_run(
        result=result,
        ledger=ledger,
        trace_rows=trace_rows,
        aob_rows=aob_rows,
        plan=plan,
    )

    assert blockers == []
    assert summary["precision_rows"] == 2
    assert summary["resolved_rows"] == 1
    assert summary["unresolved_rows"] == 1
    assert summary["relation_observation_rows"] == 1
    assert summary["overwrite_observation_rows"] == 1
    assert summary["lock_conflict_rows"] == 1
    assert summary["resolution_rate"] == 0.5


def test_audit_run_rejects_backward_resolution_and_inconsistent_survival() -> None:
    result, ledger, trace_rows, aob_rows, plan = _valid_run_inputs()
    trace_rows[1]["component_resolution_fe"] = "9"
    trace_rows[1]["component_resolution_delay_fe"] = "-1"
    trace_rows[1]["shared_var_survival_rate"] = "0.7"

    _summary, blockers = MODULE.audit_run(
        result=result,
        ledger=ledger,
        trace_rows=trace_rows,
        aob_rows=aob_rows,
        plan=plan,
    )

    assert "non_monotonic_resolution_fe" in blockers
    assert "overwrite_survival_not_complementary" in blockers


def test_compare_parity_run_ignores_only_lane_and_v40_component_fields() -> None:
    result, _ledger, trace_rows, aob_rows, _plan = _valid_run_inputs()
    v38_result = {**result, "lane_id": "arac_evidence_action_controller_v38"}
    v40_result = {**result, "lane_id": "arac_evidence_action_controller_v40"}
    v38_trace = [
        {
            "run_id": "run",
            "lane_id": "arac_evidence_action_controller_v38",
            "problem_id": "A4",
            "seed": "31",
            "selected_action_name": row["selected_action_name"],
        }
        for row in trace_rows
    ]
    v40_trace = [
        {
            **row,
            "run_id": "run",
            "lane_id": "arac_evidence_action_controller_v40",
        }
        for row in v38_trace
    ]

    parity = MODULE.compare_parity_run(
        v38_result=v38_result,
        v40_result=v40_result,
        v38_trace=v38_trace,
        v40_trace=v40_trace,
        v38_aob=aob_rows,
        v40_aob=aob_rows,
    )

    assert parity["status"] == "pass"
    assert parity["common_trace_differences"] == 0
    assert parity["final_error_equal"] == 1
    assert parity["fe_equal"] == 1


def test_build_gate_uses_run_level_coverage_not_row_volume() -> None:
    summaries = []
    for problem_id in ("A4", "S2", "E2"):
        for seed in (31, 32, 33):
            has_precision = problem_id != "E2" or seed == 31
            summaries.append(
                {
                    "problem_id": problem_id,
                    "seed": seed,
                    "precision_rows": 10 if has_precision else 0,
                    "resolved_rows": 9 if has_precision else 0,
                    "unresolved_rows": 1 if has_precision else 0,
                    "relation_observation_rows": 1,
                    "overwrite_observation_rows": 1 if has_precision else 0,
                    "lock_conflict_rows": 1 if has_precision else 0,
                }
            )
    integrity = [
        {"problem_id": row["problem_id"], "seed": row["seed"], "status": "pass"}
        for row in summaries
    ]
    parity = [
        {"problem_id": problem_id, "seed": 31, "status": "pass"}
        for problem_id in ("A4", "S2", "E2")
    ]

    gate = MODULE.build_gate(
        summaries=summaries,
        integrity_rows=integrity,
        parity_rows=parity,
        anti_leakage_pass=True,
    )

    assert gate["overall_status"] == "pass"
    assert gate["precision_run_count"] == 7
    assert gate["cases_with_two_precision_seeds"] == 2
    assert gate["resolution_rate"] == 0.9

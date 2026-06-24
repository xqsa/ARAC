from __future__ import annotations

import csv
from pathlib import Path

from experiments.exp_001_schema_smoke.run import run_schema_smoke


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_exp_001_generates_required_truth_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp_001"

    run_schema_smoke(output_dir)

    expected_files = {
        "evidence_profile.csv",
        "action_decision.csv",
        "backend_semantics_diff.csv",
        "same_budget_ledger.csv",
        "action_utility_audit.csv",
        "anti_leakage_audit.csv",
        "run_manifest.md",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}


def test_exp_001_records_all_lanes_with_same_budget_fe(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp_001"

    run_schema_smoke(output_dir)

    ledger_rows = read_rows(output_dir / "same_budget_ledger.csv")
    lanes = {row["plan_name"] for row in ledger_rows}

    assert lanes == {
        "policy_action",
        "no_action",
        "fallback",
        "random_action",
        "shuffled_evidence_action",
        "oracle_action_eval_only",
    }
    assert all(int(row["phase_i_fe"]) + int(row["phase_ii_fe"]) == int(row["total_fe"]) for row in ledger_rows)
    assert {row["same_budget_violation"] for row in ledger_rows} == {"0"}
    assert {row["fresh_execution"] for row in ledger_rows} == {"1"}
    assert {row["budget_limit"] for row in ledger_rows} == {"100"}
    assert {row["total_fe"] for row in ledger_rows} == {"100"}


def test_exp_001_toy_backend_records_real_action_semantics(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp_001"

    run_schema_smoke(output_dir)

    rows = read_rows(output_dir / "backend_semantics_diff.csv")
    by_action = {row["selected_action_name"]: row for row in rows}

    assert by_action["isolate_conflicting_relation"]["relation_handling_changed"] == "1"
    assert by_action["protect_high_margin_group"]["budget_allocation_changed"] == "1"
    assert by_action["repair_shared_variable_binding"]["variable_owner_changed"] == "1"
    assert by_action["allow_beneficial_coordination"]["coordination_mode_changed"] == "1"
    assert by_action["conservative_no_action"]["backend_semantics_changed"] == "0"


def test_exp_001_keeps_oracle_out_of_runtime_and_blocks_failed_negative_controls(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp_001"

    run_schema_smoke(output_dir)

    decisions = read_rows(output_dir / "action_decision.csv")
    runtime_decisions = [row for row in decisions if row["used_for_runtime"] == "1"]
    oracle_decisions = [row for row in decisions if row["plan_name"] == "oracle_action_eval_only"]

    assert runtime_decisions
    assert oracle_decisions
    assert all(row["plan_name"] != "oracle_action_eval_only" for row in runtime_decisions)
    assert {row["decision"] for row in oracle_decisions} == {"shadow_only"}
    assert {row["used_for_runtime"] for row in oracle_decisions} == {"0"}

    utility_rows = read_rows(output_dir / "action_utility_audit.csv")
    by_lane = {row["plan_name"]: row for row in utility_rows}
    assert by_lane["policy_action"]["claim_eligible"] == "1"
    assert by_lane["no_action"]["claim_eligible"] == "0"
    assert by_lane["fallback"]["claim_eligible"] == "0"
    assert by_lane["random_action"]["negative_control_pass"] == "0"
    assert by_lane["random_action"]["claim_eligible"] == "0"
    assert "negative_control_failed" in by_lane["random_action"]["claim_blockers"]
    assert by_lane["oracle_action_eval_only"]["claim_eligible"] == "0"
    assert "oracle_eval_only_not_runtime_dispatch" in by_lane["oracle_action_eval_only"]["claim_blockers"]

    leakage_rows = read_rows(output_dir / "anti_leakage_audit.csv")
    assert {row["audit_status"] for row in leakage_rows} == {"pass"}
    assert all(row["found_in_runtime_payload"] == "0" for row in leakage_rows)

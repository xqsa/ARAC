from __future__ import annotations

import pytest

from experiments.audit_historical_recovery import parse_target, render_report, run_audit
from experiments.historical_recovery.diagnose_fixed_expert_drift import run_diagnosis
from experiments.historical_recovery.replay import load_replay_plan


def test_parse_target_reads_historical_scientific_notation() -> None:
    assert parse_target("5.69E+05 +/- 1.57E+06") == (569000.0, 1570000.0)
    with pytest.raises(ValueError, match="invalid historical target"):
        parse_target("5.69E+05")


def test_repository_audit_recovers_a_s_r_and_exposes_missing_e() -> None:
    audit = run_audit()

    assert audit["gate_passed"] is False
    assert audit["counts"] == {"recovered": 18, "failed": 0, "missing": 6}
    statuses = {row["case"]: row["status"] for row in audit["cases"]}
    assert all(statuses[f"A{index}"] == "recovered" for index in range(1, 7))
    assert all(statuses[f"S{index}"] == "recovered" for index in range(1, 7))
    assert all(statuses[f"R{index}"] == "recovered" for index in range(1, 7))
    assert all(statuses[f"E{index}"] == "missing" for index in range(1, 7))
    matrix = audit["frozen_independent_matrix"]
    assert matrix["case_count"] == 24
    assert matrix["seed_count"] == 25
    assert matrix["rounded_historical_mean_met_count"] == 12
    assert matrix["rounded_historical_mean_not_met_count"] == 12
    assert matrix["source_hash_match_count"] == 3
    assert matrix["source_hash_total"] == 12
    assert matrix["current_source_compatible"] is False
    replay = audit["current_replay"]
    assert replay["status"] == "failed"
    assert replay["passed_count"] == 1
    assert replay["failed_count"] == 3
    assert all(row["terminal_fes_match"] for row in replay["cases"])
    control = audit["frozen_source_control"]
    assert control["status"] == "matched"
    assert control["matched_count"] == 4
    assert control["context_count"] == 4
    assert control["source_hash_match_count"] == control["source_hash_total"] == 13
    current = audit["current_fixed_expert"]
    assert current["status"] == "failed"
    assert current["gate_passed"] is False
    assert current["context_count"] == current["expected_context_count"] == 600
    assert current["case_count"] == 24
    assert current["seed_count_per_case"] == 25
    assert current["all_terminal_fes_exact"] is True
    assert current["summary_consistent"] is True
    assert current["manifest_consistent"] is True
    assert current["counts"] == {"recovered": 0, "failed": 24, "missing": 0}
    assert current["mean_match_count"] == 6
    assert current["sample_std_match_count"] == 0


def test_report_does_not_turn_missing_evidence_into_success() -> None:
    report = render_report(run_audit())

    assert "Recovered: **18/24**" in report
    assert "Missing: **6/24**" in report
    assert "Historical mean met at displayed precision: **12/24**" in report
    assert "Exact replay passed: **1/4**" in report
    assert "Current results matching the manifest-bound frozen source: **4/4**" in report
    assert "Complete arms: **600/600**" in report
    assert "Mean matches at displayed precision: **6/24**" in report
    assert "Sample-std matches at displayed precision: **0/24**" in report
    assert "Selector correctness and ARAC-Core end-to-end claims must remain deferred." in report
    assert "stored v5 block-action arms do not reproduce" in report


def test_current_replay_plan_binds_four_families_and_frozen_inputs() -> None:
    plan = load_replay_plan()

    contexts = plan["config"]["contexts"]
    assert {(item["case"], item["action"]) for item in contexts} == {
        ("A1", "aor"),
        ("E1", "smp"),
        ("R1", "gcb"),
        ("S1", "ctp"),
    }
    assert all(item["expected_terminal_fes"] == 3_000_000 for item in contexts)
    assert len(plan["current_source_hashes"]) == 12


def test_fixed_expert_diagnosis_exposes_protocol_drift() -> None:
    diagnosis = run_diagnosis()

    assert diagnosis["current_campaign"]["context_count"] == 600
    assert diagnosis["current_campaign"]["phase1_fes"] == 180_000
    assert diagnosis["current_campaign"]["action_consumed_fes"] == 2_820_000
    assert diagnosis["current_campaign"]["action_optimizer_package"] == "pypop7"
    assert diagnosis["source_drift"]["current_phase1_protocol"] == (
        "arac-identity-blind-evidence-v9"
    )
    assert diagnosis["source_drift"]["frozen_matrix_phase1_protocol"] == (
        "arac-identity-blind-evidence-v8"
    )
    assert diagnosis["source_drift"]["matching_component_count"] == 3
    assert diagnosis["source_drift"]["common_component_count"] == 14
    assert "reconstruct" in diagnosis["decision"]

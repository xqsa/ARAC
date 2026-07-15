from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_runtime_component_lease.py"
SPEC = importlib.util.spec_from_file_location("audit_runtime_component_lease", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cli_help_runs_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=SCRIPT_PATH.parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def _selected_attempt() -> dict[str, str]:
    return {
        "problem_id": "E2",
        "seed": "37",
        "outer_iter": "4",
        "group_index": "18",
        "selected_action_name": "post_retirement_precision_reanchor",
        "downstream_consumed": "1",
        "component_id": "component_1",
        "component_action_id": "precision:component_1:4:18:1",
        "component_credit_status": "resolved",
        "component_decision_fe": "2987567",
        "component_resolution_fe": "2999070",
        "component_scheduler_sweep_start_fe": "2935070",
        "component_scheduler_cc_budget_limit_fe": "3000000",
        "component_scheduler_group_budget_fe": "3247",
        "component_scheduler_optimizer_budget_fe": "3232",
        "component_scheduler_population_sizes": ";".join(["16"] * 20),
        "component_scheduler_revisit_cap_fe": "11871",
        "component_scheduler_revisit_reachable": "1",
        "component_scheduler_revisit_reason": "scheduler_revisit_cap_available",
        "component_active_lease_action_id": "",
        "component_lease_decision": "selected",
        "component_lease_reason": "component_lease_available",
        "component_precision_consumed": "1",
        "component_gain": "0.5",
        "shared_var_overwrite_rate": "0.25",
    }


def test_replay_eligibility_is_unchanged_by_current_action_outcome() -> None:
    original = MODULE.replay_runtime_run([_selected_attempt()])[0]
    mutated_row = _selected_attempt()
    mutated_row.update(
        {
            "component_credit_status": "unresolved_run_end",
            "component_resolution_fe": "3000000",
            "component_gain": "-999",
            "shared_var_overwrite_rate": "1",
        }
    )
    mutated = MODULE.replay_runtime_run([mutated_row])[0]

    eligibility_fields = (
        "expected_decision",
        "expected_reason",
        "decision_match",
        "reason_match",
        "active_lease_match",
        "consumption_match",
        "cap_contract_status",
    )
    assert {field: original[field] for field in eligibility_fields} == {
        field: mutated[field] for field in eligibility_fields
    }


def _summaries() -> list[dict[str, object]]:
    return [
        {
            "problem_id": problem_id,
            "seed": seed,
            "selected_action_count": 1,
            "selected_unresolved_count": 0,
            "overlap_violation_count": 0,
            "cap_underprediction_count": 0,
            "contract_failure_count": 0,
            "decision_mismatch_count": 0,
        }
        for problem_id in ("A4", "S2", "E2")
        for seed in (37, 38)
    ]


def _performance() -> list[dict[str, object]]:
    return [
        {
            "paired_log_advantage": "0.01",
            "changed": 1,
            "win": 1,
            "loss": 0,
            "catastrophic_loss": 0,
        }
        for _ in range(6)
    ]


def test_gate_requires_runtime_closure_coverage_and_positive_performance() -> None:
    gate = MODULE.build_gate(
        run_summaries=_summaries(),
        performance_rows=_performance(),
        integrity_blockers=[],
        aob_equal=True,
    )

    assert gate["overall_status"] == "pass"
    summaries = _summaries()
    summaries[0]["selected_unresolved_count"] = 1
    gate = MODULE.build_gate(
        run_summaries=summaries,
        performance_rows=_performance(),
        integrity_blockers=[],
        aob_equal=True,
    )
    assert gate["overall_status"] == "fail"
    assert "selected_horizon_not_closed" in gate["blockers"]


def test_gate_does_not_relax_catastrophic_or_mean_thresholds() -> None:
    performance = _performance()
    performance[0].update(
        {
            "paired_log_advantage": "-0.2",
            "win": 0,
            "loss": 1,
            "catastrophic_loss": 1,
        }
    )

    gate = MODULE.build_gate(
        run_summaries=_summaries(),
        performance_rows=performance,
        integrity_blockers=[],
        aob_equal=True,
    )

    assert gate["overall_status"] == "fail"
    assert "catastrophic_loss" in gate["blockers"]

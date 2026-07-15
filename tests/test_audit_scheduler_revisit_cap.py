from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_scheduler_revisit_cap.py"
SPEC = importlib.util.spec_from_file_location("audit_scheduler_revisit_cap", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _action(
    *,
    action_number: int,
    decision_fe: int,
    resolution_fe: int,
    group_index: int = 18,
    status: str = "resolved",
) -> dict[str, str]:
    return {
        "problem_id": "E2",
        "seed": "34",
        "outer_iter": "4",
        "group_index": str(group_index),
        "selected_action_name": MODULE.PRECISION_ACTION,
        "component_id": "component_1",
        "component_action_id": f"precision:component_1:4:{group_index}:{action_number}",
        "component_decision_fe": str(decision_fe),
        "component_credit_status": status,
        "component_resolution_fe": str(resolution_fe),
        "component_scheduler_sweep_start_fe": "2935070",
        "component_scheduler_cc_budget_limit_fe": "3000000",
        "component_scheduler_group_budget_fe": "3247",
        "component_scheduler_optimizer_budget_fe": "3232",
        "component_scheduler_population_sizes": ";".join(["16"] * 20),
        "component_scheduler_revisit_cap_fe": "11871",
        "component_scheduler_revisit_reachable": "1",
        "component_scheduler_revisit_reason": "scheduler_revisit_cap_available",
        "component_gain": "0.5",
        "component_neighbor_gain": "0.25",
        "shared_var_overwrite_rate": "0.25",
        "shared_var_survival_rate": "0.75",
    }


def test_cap_contract_recomputes_exact_pre_action_state() -> None:
    cap, blockers = MODULE.audit_cap_contract(_action(
        action_number=1,
        decision_fe=2_987_567,
        resolution_fe=2_999_070,
    ))

    assert blockers == []
    assert cap.reachable is True
    assert cap.cap_fe == 11_871


def test_cap_contract_rejects_serialized_cap_drift() -> None:
    row = _action(
        action_number=1,
        decision_fe=2_987_567,
        resolution_fe=2_999_070,
    )
    row["component_scheduler_revisit_cap_fe"] = "11870"

    _cap, blockers = MODULE.audit_cap_contract(row)

    assert blockers == ["scheduler_revisit_cap_mismatch"]


def test_current_action_outcome_cannot_change_scheduler_eligibility() -> None:
    row = _action(
        action_number=1,
        decision_fe=2_987_567,
        resolution_fe=2_999_070,
    )
    original = MODULE.replay_run([row])
    row.update(
        {
            "component_credit_status": "unresolved_run_end",
            "component_resolution_fe": "3000000",
            "component_gain": "-999",
            "component_neighbor_gain": "-999",
            "shared_var_overwrite_rate": "1",
            "shared_var_survival_rate": "0",
        }
    )
    mutated = MODULE.replay_run([row])

    eligibility = ("replay_decision", "abstain_reason", "scheduler_revisit_cap_fe")
    assert {key: original[0][key] for key in eligibility} == {
        key: mutated[0][key] for key in eligibility
    }


def test_gate_requires_zero_underprediction_and_cross_case_coverage() -> None:
    summaries = []
    for problem_id in ("A4", "S2", "E2"):
        for seed in (34, 35):
            summaries.append(
                {
                    "problem_id": problem_id,
                    "seed": seed,
                    "selected_action_count": 1,
                    "selected_resolved_count": 1,
                    "selected_unresolved_count": 0,
                    "selected_overlap_violation_count": 0,
                    "selected_cap_underprediction_count": 0,
                    "cap_contract_failure_count": 0,
                }
            )

    gate = MODULE.build_gate(
        run_summaries=summaries,
        input_integrity_pass=True,
    )

    assert gate["overall_status"] == "pass"
    summaries[0]["selected_cap_underprediction_count"] = 1
    gate = MODULE.build_gate(
        run_summaries=summaries,
        input_integrity_pass=True,
    )
    assert gate["overall_status"] == "fail"
    assert "scheduler_cap_underpredicted" in gate["blockers"]

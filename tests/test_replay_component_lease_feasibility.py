from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "replay_component_lease_feasibility.py"
)
SPEC = importlib.util.spec_from_file_location(
    "replay_component_lease_feasibility", SCRIPT_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _relation(*, group: int, outer: int, decision_fe: int) -> dict[str, str]:
    return {
        "problem_id": "A4",
        "seed": "31",
        "outer_iter": str(outer),
        "group_index": str(group),
        "selected_action_name": "conservative_no_action",
        "component_id": "component_1",
        "component_credit_status": "relation_observation",
        "component_decision_fe": str(decision_fe),
    }


def _action(
    *,
    action_number: int,
    group: int,
    outer: int,
    decision_fe: int,
    resolution_fe: int,
    status: str = "resolved",
) -> dict[str, str]:
    return {
        "problem_id": "A4",
        "seed": "31",
        "outer_iter": str(outer),
        "group_index": str(group),
        "selected_action_name": MODULE.PRECISION_ACTION,
        "component_id": "component_1",
        "component_action_id": (
            f"{MODULE.PRECISION_ACTION}:component_1:{outer}:{group}:{action_number}"
        ),
        "component_credit_status": status,
        "component_decision_fe": str(decision_fe),
        "component_resolution_fe": str(resolution_fe),
        "component_gain": "0.5",
        "component_neighbor_gain": "0.25",
        "shared_var_overwrite_rate": "0.25",
        "shared_var_survival_rate": "0.75",
    }


def _history() -> list[dict[str, str]]:
    return [
        _relation(group=1, outer=0, decision_fe=10),
        _relation(group=2, outer=0, decision_fe=20),
        _relation(group=1, outer=1, decision_fe=50),
        _relation(group=2, outer=1, decision_fe=65),
        _relation(group=1, outer=2, decision_fe=95),
        _relation(group=2, outer=2, decision_fe=100),
    ]


def test_projection_uses_only_latest_completed_prior_cycles_and_component_max() -> None:
    rows = _history()
    rows.append(_relation(group=1, outer=3, decision_fe=500))

    projection = MODULE.project_next_revisit_fe(
        trace_rows=rows,
        component_id="component_1",
        action_outer_iter=3,
        action_decision_fe=110,
    )

    assert projection.projected_fe == 45
    assert projection.history_group_count == 2
    assert projection.completed_cycle_count == 4


def test_projection_abstains_without_a_completed_prior_cycle() -> None:
    projection = MODULE.project_next_revisit_fe(
        trace_rows=[_relation(group=1, outer=0, decision_fe=10)],
        component_id="component_1",
        action_outer_iter=1,
        action_decision_fe=20,
    )

    assert projection.projected_fe is None
    assert projection.history_group_count == 0
    assert projection.completed_cycle_count == 0


def test_replay_holds_mutex_until_observed_resolution_then_releases() -> None:
    rows = [
        *_history(),
        _action(
            action_number=1,
            group=0,
            outer=3,
            decision_fe=110,
            resolution_fe=130,
        ),
        _action(
            action_number=2,
            group=1,
            outer=3,
            decision_fe=120,
            resolution_fe=135,
        ),
        _relation(group=1, outer=3, decision_fe=132),
        _relation(group=2, outer=3, decision_fe=134),
        _action(
            action_number=3,
            group=0,
            outer=4,
            decision_fe=140,
            resolution_fe=180,
        ),
    ]

    replay = MODULE.replay_run(trace_rows=rows, budget_limit=200)

    assert [row["replay_decision"] for row in replay] == [
        "selected",
        "abstained",
        "selected",
    ]
    assert replay[1]["abstain_reason"] == "abstain_component_mutex"
    assert replay[1]["active_lease_action_id_before"] == rows[6]["component_action_id"]
    assert replay[2]["released_lease_action_id"] == rows[6]["component_action_id"]
    assert sum(int(row["overlap_violation"]) for row in replay) == 0


def test_replay_abstains_when_projected_horizon_exceeds_remaining_fe() -> None:
    action = _action(
        action_number=1,
        group=0,
        outer=3,
        decision_fe=110,
        resolution_fe=120,
        status="unresolved_run_end",
    )

    replay = MODULE.replay_run(trace_rows=[*_history(), action], budget_limit=150)

    assert replay[0]["projected_next_revisit_fe"] == 45
    assert replay[0]["remaining_fe"] == 40
    assert replay[0]["replay_decision"] == "abstained"
    assert replay[0]["abstain_reason"] == "abstain_unresolvable_horizon"


def test_current_action_outcomes_cannot_change_eligibility() -> None:
    action = _action(
        action_number=1,
        group=0,
        outer=3,
        decision_fe=110,
        resolution_fe=150,
    )
    original = MODULE.replay_run(trace_rows=[*_history(), action], budget_limit=200)
    action.update(
        {
            "component_credit_status": "unresolved_run_end",
            "component_resolution_fe": "200",
            "component_gain": "-999",
            "component_neighbor_gain": "-999",
            "shared_var_overwrite_rate": "1",
            "shared_var_survival_rate": "0",
        }
    )
    mutated = MODULE.replay_run(trace_rows=[*_history(), action], budget_limit=200)

    eligibility_fields = (
        "replay_decision",
        "abstain_reason",
        "projected_next_revisit_fe",
        "remaining_fe",
    )
    assert {field: original[0][field] for field in eligibility_fields} == {
        field: mutated[0][field] for field in eligibility_fields
    }

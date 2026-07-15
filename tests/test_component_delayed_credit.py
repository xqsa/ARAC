from __future__ import annotations

import math

import numpy as np
import pytest

from arac.policy.component_delayed_credit import (
    COMPONENT_CREDIT_TRACE_FIELDS,
    ComponentDelayedCreditTrace,
    SchedulerRevisitCap,
    build_overlap_components,
    decide_component_lease,
)


def _reachable_cap() -> SchedulerRevisitCap:
    return SchedulerRevisitCap(
        sweep_start_fe=0,
        decision_fe=30,
        cc_budget_limit_fe=100,
        current_group_index=0,
        current_sweep_group_budget_fe=50,
        current_optimizer_budget_fe=50,
        group_population_sizes=(10, 10),
        reachable=True,
        cap_fe=61,
        current_tail_cap_fe=50,
        next_sweep_min_group_budget_fe=10,
        reason="scheduler_revisit_cap_available",
    )


def test_component_lease_eligibility_uses_only_cap_and_active_lease() -> None:
    cap = _reachable_cap()

    selected = decide_component_lease(
        scheduler_revisit_cap=cap,
        active_component_action_id="",
    )
    locked = decide_component_lease(
        scheduler_revisit_cap=cap,
        active_component_action_id="precision:component:0",
    )
    unreachable = decide_component_lease(
        scheduler_revisit_cap=SchedulerRevisitCap(
            **{
                **cap.__dict__,
                "reachable": False,
                "cap_fe": None,
                "reason": "next_sweep_not_reachable",
            }
        ),
        active_component_action_id="",
    )

    assert selected.selected is True
    assert selected.reason == "component_lease_available"
    assert locked.selected is False
    assert locked.reason == "abstain_component_mutex"
    assert locked.active_component_action_id == "precision:component:0"
    assert unreachable.selected is False
    assert unreachable.reason == "abstain_scheduler_unreachable"


def test_overlap_components_lock_connected_groups_and_isolate_disjoint_groups() -> None:
    topology = build_overlap_components([[0, 1], [1, 2], [3, 4]])

    first = topology.for_group(0)
    second = topology.for_group(1)
    isolated = topology.for_group(2)

    assert first.component_id == second.component_id
    assert first.component_id != isolated.component_id
    assert first.group_indices == (0, 1)
    assert first.shared_variables == (1,)
    assert isolated.group_indices == (2,)
    assert isolated.shared_variables == ()


def test_component_trace_observes_lock_conflict_and_resolves_at_next_revisit() -> None:
    trace = ComponentDelayedCreditTrace(
        [[0, 1], [1, 2]],
        lower=-5.0,
        upper=5.0,
    )
    relation_row = {field: "" for field in COMPONENT_CREDIT_TRACE_FIELDS}
    disagreement = trace.annotate_relation_observation(
        relation_row,
        outer_iter=0,
        group_left=0,
        group_right=1,
        previous_values=np.array([1.0]),
        current_values=np.array([3.0]),
        decision_fe=20,
        max_fes=100,
    )
    trace.complete_sweep(outer_iter=0, optimized_group_count=2)

    first_row = {field: "" for field in COMPONENT_CREDIT_TRACE_FIELDS}
    trace.register_search_action(
        first_row,
        action_name="post_retirement_precision_reanchor",
        outer_iter=1,
        group_index=0,
        decision_fe=30,
        max_fes=100,
        pre_action_fitness=100.0,
        post_action_fitness=90.0,
        pre_action_candidate=np.array([0.0, 0.0, 0.0]),
        post_action_candidate=np.array([0.0, 2.0, 0.0]),
    )
    second_row = {field: "" for field in COMPONENT_CREDIT_TRACE_FIELDS}
    trace.register_search_action(
        second_row,
        action_name="post_retirement_precision_reanchor",
        outer_iter=1,
        group_index=1,
        decision_fe=40,
        max_fes=100,
        pre_action_fitness=90.0,
        post_action_fitness=85.0,
        pre_action_candidate=np.array([0.0, 2.0, 0.0]),
        post_action_candidate=np.array([0.0, 2.0, 1.0]),
    )

    assert math.isclose(disagreement, 0.2)
    assert first_row["component_pending_before"] == "0"
    assert first_row["component_action_id"].startswith(
        "post_retirement_precision_reanchor:component_"
    )
    assert first_row["component_lock_conflict"] == "0"
    assert first_row["component_credit_status"] == "pending"
    assert first_row["component_proposal_disagreement"] == "2.000000e-01"
    assert second_row["component_pending_before"] == "1"
    assert second_row["component_lock_conflict"] == "1"

    current = np.array([0.0, 1.0, 1.0])
    original = current.copy()
    resolved = trace.resolve_group_revisit(
        group_index=0,
        resolution_fe=70,
        current_fitness=80.0,
        current_candidate=current,
    )

    assert resolved == 1
    assert np.array_equal(current, original)
    assert first_row["component_credit_status"] == "resolved"
    assert first_row["component_resolution_fe"] == "70"
    assert first_row["component_resolution_delay_fe"] == "40"
    assert float(first_row["component_local_gain"]) == 0.1
    assert float(first_row["component_gain"]) == 0.2
    assert math.isclose(
        float(first_row["component_neighbor_gain"]),
        1.0 / 9.0,
        rel_tol=1e-6,
    )
    assert first_row["component_neighbor_spillover"] == "0.000000e+00"
    assert first_row["shared_var_overwrite_rate"] == "1.000000e+00"
    assert first_row["shared_var_survival_rate"] == "0.000000e+00"


def test_component_trace_marks_unresolved_actions_without_fabricated_credit() -> None:
    trace = ComponentDelayedCreditTrace([[0, 1], [1, 2]], lower=-5.0, upper=5.0)
    row = {field: "" for field in COMPONENT_CREDIT_TRACE_FIELDS}
    trace.register_search_action(
        row,
        action_name="post_retirement_precision_reanchor",
        outer_iter=0,
        group_index=0,
        decision_fe=90,
        max_fes=100,
        pre_action_fitness=100.0,
        post_action_fitness=100.0,
        pre_action_candidate=np.zeros(3),
        post_action_candidate=np.zeros(3),
    )

    assert trace.finalize_unresolved(resolution_fe=100) == 1
    assert row["component_credit_status"] == "unresolved_run_end"
    assert row["component_resolution_fe"] == "100"
    assert row["component_gain"] == ""
    assert row["component_neighbor_gain"] == ""
    assert row["component_credit_reason"] == "budget_ended_before_next_group_revisit"


def test_component_runtime_mutex_spans_connected_groups_until_origin_revisit() -> None:
    trace = ComponentDelayedCreditTrace([[0, 1], [1, 2]], lower=-5.0, upper=5.0)
    first = trace.component_lease_eligibility(
        group_index=0,
        scheduler_revisit_cap=_reachable_cap(),
    )
    row = {field: "" for field in COMPONENT_CREDIT_TRACE_FIELDS}
    trace.register_search_action(
        row,
        action_name="post_retirement_precision_reanchor",
        outer_iter=0,
        group_index=0,
        decision_fe=30,
        max_fes=100,
        pre_action_fitness=100.0,
        post_action_fitness=90.0,
        pre_action_candidate=np.zeros(3),
        post_action_candidate=np.ones(3),
        require_component_unlocked=True,
    )

    locked = trace.component_lease_eligibility(
        group_index=1,
        scheduler_revisit_cap=_reachable_cap(),
    )
    assert first.selected is True
    assert locked.selected is False
    assert locked.reason == "abstain_component_mutex"
    assert locked.active_component_action_id == row["component_action_id"]

    trace.resolve_group_revisit(
        group_index=0,
        resolution_fe=70,
        current_fitness=80.0,
        current_candidate=np.ones(3),
    )
    released = trace.component_lease_eligibility(
        group_index=1,
        scheduler_revisit_cap=_reachable_cap(),
    )
    assert released.selected is True


def test_component_trace_rejects_unidentified_actions_and_backward_fe() -> None:
    trace = ComponentDelayedCreditTrace([[0, 1], [1, 2]], lower=-5.0, upper=5.0)
    row = {field: "" for field in COMPONENT_CREDIT_TRACE_FIELDS}
    common = {
        "trace_row": row,
        "outer_iter": 0,
        "group_index": 0,
        "decision_fe": 30,
        "max_fes": 100,
        "pre_action_fitness": 100.0,
        "post_action_fitness": 90.0,
        "pre_action_candidate": np.zeros(3),
        "post_action_candidate": np.ones(3),
    }

    with pytest.raises(ValueError, match="action_name must be non-empty"):
        trace.register_search_action(action_name="", **common)

    trace.register_search_action(action_name="precision", **common)
    with pytest.raises(ValueError, match="must not precede"):
        trace.resolve_group_revisit(
            group_index=0,
            resolution_fe=29,
            current_fitness=80.0,
            current_candidate=np.ones(3),
        )

    assert trace.resolve_group_revisit(
        group_index=0,
        resolution_fe=31,
        current_fitness=80.0,
        current_candidate=np.ones(3),
    ) == 1

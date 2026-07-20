from __future__ import annotations

import math

import numpy as np
import pytest

from arac.actions.action_spec import ActionSpec
from arac.actions.budget_reallocation import apply_budget_reallocation_action
from arac.actions.shared_variable_blend import (
    NATIVE_EQ8_ACTION,
    TRUE_NO_WRITEBACK_ACTION,
    apply_shared_variable_action,
)
from arac.actions.sweep_ordering import apply_sweep_ordering_action
from arac.actions.warm_start import apply_warm_start_action
from arac.policy.action_bandit import ActionBandit
from arac.policy.action_outcome_ledger import ActionOutcomeLedger


def test_action_spec_contains_parameters_but_no_selector_precondition() -> None:
    spec = ActionSpec(
        name="explicit_action",
        semantic_surface="test",
        parameter_names=("value",),
    )

    assert spec.parameter_names == ("value",)
    assert not hasattr(spec, "precondition")


def test_budget_action_executes_exact_frozen_vector() -> None:
    result = apply_budget_reallocation_action(
        "efficiency_budget_reallocation",
        group_budgets=(20, 40),
        population_sizes=(10, 10),
        frozen_total=60,
    )

    assert result == (20, 40)


def test_budget_action_rejects_total_drift() -> None:
    with pytest.raises(ValueError, match="frozen FE total"):
        apply_budget_reallocation_action(
            "efficiency_budget_reallocation",
            group_budgets=(20, 39),
            population_sizes=(10, 10),
            frozen_total=60,
        )


def test_sweep_order_action_requires_complete_permutation() -> None:
    assert apply_sweep_ordering_action(
        "delta_priority_scan",
        group_order=(2, 0, 1),
        group_count=3,
    ) == (2, 0, 1)

    with pytest.raises(ValueError, match="complete permutation"):
        apply_sweep_ordering_action(
            "delta_priority_scan",
            group_order=(2, 0, 0),
            group_count=3,
        )


def test_warm_start_changes_only_explicit_unique_positions() -> None:
    current = np.array([1.0, 2.0, 3.0, 4.0])
    updated = apply_warm_start_action(
        "stagnation_cross_group_warm_start",
        current_mean=current,
        unique_positions=(0, 2),
        mean_shift=np.array([0.5, -1.0]),
        lower_bound=-10.0,
        upper_bound=10.0,
    )

    np.testing.assert_allclose(updated, [1.5, 2.0, 2.0, 4.0])
    np.testing.assert_allclose(current, [1.0, 2.0, 3.0, 4.0])


def test_shared_action_writes_only_frozen_values() -> None:
    current = np.array([1.0, 2.0])
    frozen = np.array([3.0, 4.0])

    result = apply_shared_variable_action(NATIVE_EQ8_ACTION, current, frozen)

    np.testing.assert_allclose(result, frozen)
    assert result is not frozen
    assert apply_shared_variable_action(
        TRUE_NO_WRITEBACK_ACTION,
        current,
        None,
    ) is None


def test_outcome_ledger_waits_for_the_next_complete_sweep() -> None:
    ledger = ActionOutcomeLedger()
    ledger.record_pending(
        relation_id="g0-1",
        action_name="exact_left",
        semantic_surface="shared_variable_value",
        sweep_index=2,
        anchor_error=10.0,
        anchor_shared_values=(0.0,),
        candidate_shared_values=(1.0,),
        shared_variable_indices=(0,),
        evidence_snapshot={"delta_ratio_gap": 0.5},
    )

    assert ledger.close_sweep(
        best_individual=(1.0,),
        next_sweep_error=5.0,
        completed_sweep_index=2,
        all_groups_completed=True,
        native_sweep_end_completed=True,
    ) == []
    assert ledger.pending_relation_ids() == {"g0-1"}
    assert ledger.close_sweep(
        best_individual=(1.0,),
        next_sweep_error=5.0,
        completed_sweep_index=3,
        all_groups_completed=False,
        native_sweep_end_completed=False,
    ) == []
    assert ledger.pending_relation_ids() == {"g0-1"}

    outcomes = ledger.close_sweep(
        best_individual=(1.0,),
        next_sweep_error=5.0,
        completed_sweep_index=3,
        all_groups_completed=True,
        native_sweep_end_completed=True,
    )

    assert len(outcomes) == 1
    assert outcomes[0].sweep_index == 2
    assert math.isclose(outcomes[0].next_sweep_log_improvement, math.log(2.0))
    assert ledger.pending_relation_ids() == set()
    assert math.isclose(ledger.mean_credit("exact_left") or 0.0, math.log(2.0))


def test_outcome_ledger_keeps_pending_record_on_wrong_resolution() -> None:
    ledger = ActionOutcomeLedger()
    ledger.record_pending(
        relation_id="g0-1",
        action_name="exact_left",
        semantic_surface="shared_variable_value",
        sweep_index=2,
        anchor_error=10.0,
        anchor_shared_values=(0.0,),
        candidate_shared_values=(1.0,),
        shared_variable_indices=(0,),
        evidence_snapshot={"delta_ratio_gap": 0.5},
    )

    with pytest.raises(ValueError, match="next complete sweep"):
        ledger.close_pending(
            relation_id="g0-1",
            resolution_sweep_index=4,
            next_sweep_error=5.0,
            next_sweep_shared_values=(1.0,),
            all_groups_completed=True,
            native_sweep_end_completed=True,
        )

    assert ledger.pending_relation_ids() == {"g0-1"}


def test_bandit_counts_every_observation() -> None:
    bandit = ActionBandit(candidate_actions=["native_eq8", "exact_left"])
    evidence = {"delta_ratio_gap": 0.5}

    for _ in range(5):
        bandit.update("native_eq8", evidence, 1.0)

    assert bandit._bucket_pulls["conflict"] == 5
    assert bandit._stats[("native_eq8", "conflict")].count == 5


def test_bandit_rejects_unregistered_updates() -> None:
    bandit = ActionBandit(candidate_actions=["native_eq8"])

    with pytest.raises(ValueError, match="unregistered"):
        bandit.update("exact_left", {"delta_ratio_gap": 0.5}, 1.0)

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from arac.actions import (
    PERSISTENT_EFFICIENCY_BUDGET_REALLOCATION_ACTION,
    PersistentBudgetAllocationAction,
    PersistentBudgetAllocationExecutionState,
    execute_persistent_budget_allocation_action,
)
from arac.actions.budget_reallocation import budget_allocation_parameter_hash


def _action() -> PersistentBudgetAllocationAction:
    return PersistentBudgetAllocationAction(
        problem_id="S5",
        run_seed=117,
        checkpoint_fe=300_000,
        checkpoint_hash="a" * 64,
        action_set_hash="b" * 64,
        source_efficiency_ewma=(9.0, 1.0, 0.0),
        population_sizes=(2, 2, 2),
        uniform_group_budgets=(6, 6, 6),
        group_budgets=(10, 4, 4),
        frozen_total_fes=18,
        start_sweep=4,
        end_absolute_fe=300_046,
    )


def _record_normal_application(
    lifecycle: PersistentBudgetAllocationExecutionState,
    action: PersistentBudgetAllocationAction,
) -> tuple[int, ...]:
    return lifecycle.record_application(
        action,
        current_sweep=4,
        application_fe=300_001,
        checkpoint_hash=action.checkpoint_hash,
        action_set_hash=action.action_set_hash,
        applied_group_budgets=action.group_budgets,
        actual_optimizer_fes=(10, 4, 4),
        group_interval_fes=(11, 5, 5),
        terminal_truncated=False,
    )


def test_persistent_action_is_frozen_and_executes_exact_stored_budgets() -> None:
    action = _action()

    assert execute_persistent_budget_allocation_action(action) == (10, 4, 4)
    assert action.budget_parameter_hash == budget_allocation_parameter_hash(
        population_sizes=action.population_sizes,
        uniform_group_budgets=action.uniform_group_budgets,
        group_budgets=action.group_budgets,
        frozen_total_fes=action.frozen_total_fes,
    )
    assert action.audit_payload()["action"] == (
        PERSISTENT_EFFICIENCY_BUDGET_REALLOCATION_ACTION
    )
    assert PERSISTENT_EFFICIENCY_BUDGET_REALLOCATION_ACTION == (
        "persistent_frozen_efficiency_budget_reallocation"
    )
    assert len(action.action_hash) == 64
    assert not hasattr(action, "ttl_sweeps")
    with pytest.raises(FrozenInstanceError):
        action.start_sweep = 5  # type: ignore[misc]


def test_persistent_lifecycle_reuses_one_allocation_until_absolute_fe() -> None:
    action = _action()
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)

    assert _record_normal_application(lifecycle, action) == action.group_budgets
    assert lifecycle.status == "active"
    assert lifecycle.record_application(
        action,
        current_sweep=5,
        application_fe=300_022,
        checkpoint_hash=action.checkpoint_hash,
        action_set_hash=action.action_set_hash,
        applied_group_budgets=action.group_budgets,
        actual_optimizer_fes=action.group_budgets,
        group_interval_fes=(11, 5, 5),
        terminal_truncated=False,
    ) == action.group_budgets
    assert lifecycle.record_application(
        action,
        current_sweep=6,
        application_fe=300_043,
        checkpoint_hash=action.checkpoint_hash,
        action_set_hash=action.action_set_hash,
        applied_group_budgets=(3, 0, 0),
        actual_optimizer_fes=(3, 0, 0),
        group_interval_fes=(4, 0, 0),
        terminal_truncated=True,
    ) == action.group_budgets

    assert lifecycle.status == "completed"
    assert lifecycle.completed_fe == action.end_absolute_fe
    assert [record.requested_group_budgets for record in lifecycle.applications] == [
        action.group_budgets,
        action.group_budgets,
        action.group_budgets,
    ]
    assert lifecycle.applications[-1].terminal_truncated is True
    assert lifecycle.applications[-1].applied_group_budgets == (3, 0, 0)
    assert lifecycle.applications[-1].actual_optimizer_fes == (3, 0, 0)
    assert lifecycle.applications[-1].group_interval_fes == (4, 0, 0)


def test_persistent_lifecycle_payload_is_reconstructible() -> None:
    action = _action()
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)
    lifecycle.record_application(
        action,
        current_sweep=action.start_sweep,
        application_fe=action.checkpoint_fe + 1,
        checkpoint_hash=action.checkpoint_hash,
        action_set_hash=action.action_set_hash,
        applied_group_budgets=action.group_budgets,
        actual_optimizer_fes=(2, 1, 1),
        group_interval_fes=(3, 2, 2),
        terminal_truncated=False,
    )
    payload = lifecycle.audit_payload(action)
    payload.pop("action")

    reconstructed = PersistentBudgetAllocationExecutionState(**payload)

    reconstructed.validate_for(action)
    assert reconstructed.state_hash(action) == lifecycle.state_hash(action)
    assert reconstructed.applications == lifecycle.applications


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("checkpoint_hash", "c" * 64, "checkpoint_hash mismatch"),
        ("action_set_hash", "c" * 64, "action_set_hash mismatch"),
        ("current_sweep", 5, "consecutive sweeps"),
    ),
)
def test_persistent_lifecycle_fails_closed_on_identity_or_start_mismatch(
    field: str,
    value: str | int,
    message: str,
) -> None:
    action = _action()
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)
    arguments: dict[str, object] = {
        "current_sweep": action.start_sweep,
        "application_fe": action.checkpoint_fe + 1,
        "checkpoint_hash": action.checkpoint_hash,
        "action_set_hash": action.action_set_hash,
        "applied_group_budgets": action.group_budgets,
        "actual_optimizer_fes": action.group_budgets,
        "group_interval_fes": (11, 5, 5),
        "terminal_truncated": False,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        lifecycle.record_application(action, **arguments)  # type: ignore[arg-type]

    assert lifecycle.status == "issued"
    assert lifecycle.applications == ()


def test_persistent_lifecycle_requires_truthful_terminal_truncation() -> None:
    action = _action()
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)

    with pytest.raises(ValueError, match="sequential absolute FE caps"):
        lifecycle.record_application(
            action,
            current_sweep=action.start_sweep,
            application_fe=action.checkpoint_fe + 1,
            checkpoint_hash=action.checkpoint_hash,
            action_set_hash=action.action_set_hash,
            applied_group_budgets=(1, 0, 0),
            actual_optimizer_fes=(1, 0, 0),
            group_interval_fes=(2, 0, 0),
            terminal_truncated=False,
        )

    _record_normal_application(lifecycle, action)
    lifecycle.record_application(
        action,
        current_sweep=5,
        application_fe=300_022,
        checkpoint_hash=action.checkpoint_hash,
        action_set_hash=action.action_set_hash,
        applied_group_budgets=action.group_budgets,
        actual_optimizer_fes=action.group_budgets,
        group_interval_fes=(11, 5, 5),
        terminal_truncated=False,
    )
    with pytest.raises(ValueError, match="terminal_truncated does not match"):
        lifecycle.record_application(
            action,
            current_sweep=6,
            application_fe=action.end_absolute_fe - 3,
            checkpoint_hash=action.checkpoint_hash,
            action_set_hash=action.action_set_hash,
            applied_group_budgets=(3, 0, 0),
            actual_optimizer_fes=(3, 0, 0),
            group_interval_fes=(4, 0, 0),
            terminal_truncated=False,
        )
    with pytest.raises(ValueError, match="sequential absolute FE caps"):
        lifecycle.record_application(
            action,
            current_sweep=6,
            application_fe=action.end_absolute_fe - 3,
            checkpoint_hash=action.checkpoint_hash,
            action_set_hash=action.action_set_hash,
            applied_group_budgets=action.group_budgets,
            actual_optimizer_fes=action.group_budgets,
            group_interval_fes=(11, 5, 5),
            terminal_truncated=True,
        )

    assert lifecycle.status == "active"


def test_persistent_lifecycle_accepts_early_stop_below_applied_cap() -> None:
    action = _action()
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)

    lifecycle.record_application(
        action,
        current_sweep=action.start_sweep,
        application_fe=action.checkpoint_fe + 1,
        checkpoint_hash=action.checkpoint_hash,
        action_set_hash=action.action_set_hash,
        applied_group_budgets=action.group_budgets,
        actual_optimizer_fes=(2, 1, 1),
        group_interval_fes=(3, 2, 2),
        terminal_truncated=False,
    )

    assert lifecycle.status == "active"
    assert lifecycle.applications[-1].actual_optimizer_fes == (2, 1, 1)

    with pytest.raises(ValueError, match="cannot exceed"):
        lifecycle.record_application(
            action,
            current_sweep=action.start_sweep + 1,
            application_fe=300_008,
            checkpoint_hash=action.checkpoint_hash,
            action_set_hash=action.action_set_hash,
            applied_group_budgets=action.group_budgets,
            actual_optimizer_fes=(11, 1, 1),
            group_interval_fes=(12, 2, 2),
            terminal_truncated=False,
        )
    with pytest.raises(ValueError, match="exactly one incumbent precheck"):
        lifecycle.record_application(
            action,
            current_sweep=action.start_sweep + 1,
            application_fe=300_008,
            checkpoint_hash=action.checkpoint_hash,
            action_set_hash=action.action_set_hash,
            applied_group_budgets=action.group_budgets,
            actual_optimizer_fes=(2, 1, 1),
            group_interval_fes=(4, 2, 2),
            terminal_truncated=False,
        )


def test_nominal_terminal_sweep_can_remain_untruncated_after_early_stop() -> None:
    action = PersistentBudgetAllocationAction(
        **{
            **_action().__dict__,
            "checkpoint_fe": 300_029,
        }
    )
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)

    lifecycle.record_application(
        action,
        current_sweep=action.start_sweep,
        application_fe=300_030,
        checkpoint_hash=action.checkpoint_hash,
        action_set_hash=action.action_set_hash,
        applied_group_budgets=action.group_budgets,
        actual_optimizer_fes=(1, 1, 1),
        group_interval_fes=(2, 2, 2),
        terminal_truncated=False,
    )

    assert lifecycle.status == "active"
    assert lifecycle.applications[-1].actual_end_fe == 300_035


def test_terminal_clipping_can_require_an_additional_sweep_after_early_stop() -> None:
    action = PersistentBudgetAllocationAction(
        **{
            **_action().__dict__,
            "checkpoint_fe": 300_039,
        }
    )
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)

    lifecycle.record_application(
        action,
        current_sweep=action.start_sweep,
        application_fe=300_040,
        checkpoint_hash=action.checkpoint_hash,
        action_set_hash=action.action_set_hash,
        applied_group_budgets=(6, 4, 2),
        actual_optimizer_fes=(1, 1, 1),
        group_interval_fes=(2, 2, 2),
        terminal_truncated=True,
    )

    assert lifecycle.status == "active"
    assert lifecycle.applications[-1].actual_end_fe == 300_045
    lifecycle.record_application(
        action,
        current_sweep=action.start_sweep + 1,
        application_fe=300_046,
        checkpoint_hash=action.checkpoint_hash,
        action_set_hash=action.action_set_hash,
        applied_group_budgets=(0, 0, 0),
        actual_optimizer_fes=(0, 0, 0),
        group_interval_fes=(1, 0, 0),
        terminal_truncated=True,
    )

    assert lifecycle.status == "completed"
    assert lifecycle.completed_fe == action.end_absolute_fe


def test_terminal_application_fails_closed_on_impossible_cap_or_boundary() -> None:
    action = PersistentBudgetAllocationAction(
        **{
            **_action().__dict__,
            "checkpoint_fe": 300_039,
        }
    )
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)

    terminal_arguments = {
        "current_sweep": action.start_sweep,
        "application_fe": action.checkpoint_fe + 1,
        "checkpoint_hash": action.checkpoint_hash,
        "action_set_hash": action.action_set_hash,
        "terminal_truncated": True,
    }
    with pytest.raises(ValueError, match="sequential absolute FE caps"):
        lifecycle.record_application(
            action,
            **terminal_arguments,
            applied_group_budgets=(5, 4, 2),
            actual_optimizer_fes=(1, 1, 1),
            group_interval_fes=(2, 2, 2),
        )
    with pytest.raises(ValueError, match="absolute FE boundary"):
        lifecycle.record_application(
            action,
            **terminal_arguments,
            applied_group_budgets=(6, 4, 2),
            actual_optimizer_fes=(6, 4, 2),
            group_interval_fes=(7, 5, 3),
        )

    assert lifecycle.status == "issued"


def test_persistent_lifecycle_can_abstain_after_activation() -> None:
    action = _action()
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)
    _record_normal_application(lifecycle, action)

    lifecycle.abstain(action, reason="checkpoint_invalidated")

    assert lifecycle.status == "abstained"
    assert lifecycle.invalidation_reason == "checkpoint_invalidated"
    assert len(lifecycle.applications) == 1
    with pytest.raises(ValueError, match="issued or active"):
        _record_normal_application(lifecycle, action)


def test_persistent_execution_api_has_no_phase2_evidence_input() -> None:
    parameters = inspect.signature(
        PersistentBudgetAllocationExecutionState.record_application
    ).parameters

    assert "source_efficiency_ewma" not in parameters
    assert "current_efficiency_ewma" not in parameters
    assert "fitness_delta" not in parameters


def test_persistent_lifecycle_hash_binding_fails_closed() -> None:
    action = _action()
    lifecycle = PersistentBudgetAllocationExecutionState.for_action(action)

    mismatched = PersistentBudgetAllocationExecutionState(action_hash="c" * 64)
    with pytest.raises(ValueError, match="does not match action_hash"):
        mismatched.validate_for(action)

    invalid_active = PersistentBudgetAllocationExecutionState(
        action_hash=action.action_hash,
        status="active",
    )
    with pytest.raises(ValueError, match="must contain unfinished applications"):
        invalid_active.validate_for(action)

    assert lifecycle.state_hash(action) != mismatched.action_hash


def test_persistent_action_rejects_invalid_total_cap_and_end_fe() -> None:
    action = _action()
    with pytest.raises(ValueError, match="frozen FE total"):
        PersistentBudgetAllocationAction(
            **{**action.__dict__, "group_budgets": (10, 4, 3)}
        )
    with pytest.raises(ValueError, match="3x uniform cap"):
        PersistentBudgetAllocationAction(
            **{
                **action.__dict__,
                "population_sizes": (1, 1, 1),
                "uniform_group_budgets": (2, 8, 8),
                "group_budgets": (7, 5, 6),
            }
        )
    with pytest.raises(ValueError, match="after checkpoint_fe"):
        PersistentBudgetAllocationAction(
            **{**action.__dict__, "end_absolute_fe": action.checkpoint_fe}
        )

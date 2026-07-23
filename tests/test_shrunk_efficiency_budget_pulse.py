from __future__ import annotations

from dataclasses import replace

import pytest

from arac.actions.shrunk_budget_pulse import (
    SHRUNK_BUDGET_PULSE_SCHEMA,
    SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION,
    ShrunkBudgetPulseExecutionState,
    ShrunkEfficiencyBudgetPulseAction,
    allocate_shrunk_efficiency_budgets,
    execute_shrunk_efficiency_budget_pulse_action,
    shrunk_budget_pulse_anchor_hash,
    shrunk_budget_pulse_parameter_hash,
)


def _action() -> ShrunkEfficiencyBudgetPulseAction:
    raw = (2, 4, 12)
    uniform = (6, 6, 6)
    populations = (2, 2, 2)
    budgets = allocate_shrunk_efficiency_budgets(raw, uniform, populations)
    anchor_hash = shrunk_budget_pulse_anchor_hash(
        problem_id="S5",
        run_seed=117,
        checkpoint_fe=300_000,
        dispatch_checkpoint_hash="a" * 64,
        raw_group_budgets=raw,
        uniform_group_budgets=uniform,
        population_sizes=populations,
        issued_sweep=3,
    )
    parameter_hash = shrunk_budget_pulse_parameter_hash(
        raw_group_budgets=raw,
        uniform_group_budgets=uniform,
        group_budgets=budgets,
        population_sizes=populations,
        frozen_total_fes=sum(uniform),
    )
    return ShrunkEfficiencyBudgetPulseAction(
        problem_id="S5",
        run_seed=117,
        checkpoint_fe=300_000,
        dispatch_checkpoint_hash="a" * 64,
        anchor_hash=anchor_hash,
        raw_group_budgets=raw,
        uniform_group_budgets=uniform,
        group_budgets=budgets,
        population_sizes=populations,
        frozen_total_fes=sum(uniform),
        issued_sweep=3,
        target_sweep=4,
        ttl_sweeps=1,
        expires_sweep=4,
        parameter_hash=parameter_hash,
    )


def test_compiler_applies_fixed_half_shrink_and_preserves_total() -> None:
    budgets = allocate_shrunk_efficiency_budgets(
        raw_group_budgets=(2, 4, 12),
        uniform_group_budgets=(6, 6, 6),
        population_sizes=(2, 2, 2),
    )

    assert budgets == (4, 5, 9)
    assert sum(budgets) == 18
    assert all(
        2 * budget >= uniform and budget <= 2 * uniform
        for budget, uniform in zip(budgets, (6, 6, 6), strict=True)
    )


def test_compiler_uses_group_index_to_break_equal_remainders() -> None:
    budgets = allocate_shrunk_efficiency_budgets(
        raw_group_budgets=(4, 6),
        uniform_group_budgets=(5, 5),
        population_sizes=(1, 1),
    )

    # Both exact quotas end in .5; stable largest-remainder awards group 0.
    assert budgets == (5, 5)


def test_compiler_respects_natural_upper_bound_after_rounding() -> None:
    budgets = allocate_shrunk_efficiency_budgets(
        raw_group_budgets=(6, 1, 5),
        uniform_group_budgets=(2, 4, 6),
        population_sizes=(1, 1, 1),
    )

    assert budgets == (4, 3, 5)
    assert budgets[0] == 2 * 2


@pytest.mark.parametrize(
    ("raw", "uniform", "match"),
    (
        ((2, 4, 11), (6, 6, 6), "preserve the uniform FE total"),
        ((7, 5), (2, 10), "3x uniform cap"),
    ),
)
def test_compiler_rejects_invalid_raw_allocation(
    raw: tuple[int, ...],
    uniform: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        allocate_shrunk_efficiency_budgets(
            raw_group_budgets=raw,
            uniform_group_budgets=uniform,
            population_sizes=tuple(1 for _ in raw),
        )


def test_typed_action_binds_new_semantics_and_executor_returns_frozen_vector() -> None:
    action = _action()

    assert action.group_budgets == (4, 5, 9)
    assert execute_shrunk_efficiency_budget_pulse_action(action) == action.group_budgets
    assert action.audit_payload()["action"] == SHRUNK_EFFICIENCY_BUDGET_PULSE_ACTION
    assert action.audit_payload()["schema"] == SHRUNK_BUDGET_PULSE_SCHEMA
    assert len(action.action_hash) == 64


def test_typed_action_rejects_non_formula_allocation_even_with_matching_hash() -> None:
    action = _action()
    wrong_budgets = (5, 4, 9)
    wrong_hash = shrunk_budget_pulse_parameter_hash(
        raw_group_budgets=action.raw_group_budgets,
        uniform_group_budgets=action.uniform_group_budgets,
        group_budgets=wrong_budgets,
        population_sizes=action.population_sizes,
        frozen_total_fes=action.frozen_total_fes,
    )

    with pytest.raises(ValueError, match="fixed 50/50 pulse"):
        replace(action, group_budgets=wrong_budgets, parameter_hash=wrong_hash)


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"ttl_sweeps": 2}, "ttl_sweeps=1"),
        ({"target_sweep": 5, "expires_sweep": 5}, "next sweep"),
        ({"expires_sweep": 5}, "equal the target sweep"),
    ),
)
def test_typed_action_enforces_one_sweep_ttl(
    changes: dict[str, int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_action(), **changes)


def test_lifecycle_consumes_once_and_records_exact_allocation() -> None:
    action = _action()
    lifecycle = ShrunkBudgetPulseExecutionState.for_action(action)

    budgets = lifecycle.consume(
        action,
        current_sweep=action.target_sweep,
        application_fe=300_001,
        dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
        anchor_hash=action.anchor_hash,
    )

    assert budgets == action.group_budgets
    assert lifecycle.status == "consumed"
    assert lifecycle.audit_payload(action)["applied_group_budgets"] == [4, 5, 9]
    with pytest.raises(ValueError, match="only an issued"):
        lifecycle.consume(
            action,
            current_sweep=action.target_sweep,
            application_fe=300_002,
            dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
            anchor_hash=action.anchor_hash,
        )


def test_lifecycle_rejects_expired_pulse_and_can_record_abstention() -> None:
    action = _action()
    lifecycle = ShrunkBudgetPulseExecutionState.for_action(action)

    with pytest.raises(ValueError, match="TTL expired"):
        lifecycle.consume(
            action,
            current_sweep=action.expires_sweep + 1,
            application_fe=300_001,
            dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
            anchor_hash=action.anchor_hash,
        )

    lifecycle.abstain(action, reason="expired before target dispatch")
    assert lifecycle.status == "abstained"
    assert lifecycle.audit_payload(action)["invalidation_reason"] == (
        "expired before target dispatch"
    )

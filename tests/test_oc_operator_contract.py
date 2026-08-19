"""Unit tests for the frozen ARAC-OC operator contract."""

from __future__ import annotations

from dataclasses import fields
import pytest

from arac.coordination.contract import (
    OC_ACTION_AOR,
    OC_ACTION_ARBITRATION,
    OC_ACTION_CTP_SHARED_CORE,
    OC_ACTION_SMP,
    OC_PROBE_FES_PER_VARIABLE,
    OC_STATUS_COMPLETED,
    OC_STATUS_NO_GAIN,
    OC_STATUS_OPERATOR_FAILED,
    UNCALIBRATED_FIELDS,
    OcCoordinatorConfig,
    OperatorPlan,
    OperatorReceipt,
    receipt_from_plan,
)


def _plan(**overrides) -> OperatorPlan:
    base = dict(
        cycle_index=3,
        component=(0, 1),
        scope=(4, 5),
        conflict_level="high",
        action=OC_ACTION_CTP_SHARED_CORE,
        reserved_fes=16,
        predicted_gain=0.5,
        seed=11,
        reason="ema_enter_high",
        hub_degree=2,
        relative_hub=0.5,
    )
    base.update(overrides)
    return OperatorPlan(**base)


def test_plan_hash_is_canonical_and_stable() -> None:
    plan = _plan()
    same = _plan()
    assert plan.plan_hash == same.plan_hash
    assert _plan(reserved_fes=32).plan_hash != plan.plan_hash


def test_plan_rejects_level_action_mismatch() -> None:
    with pytest.raises(ValueError, match="not admissible at conflict level"):
        _plan(action=OC_ACTION_SMP, conflict_level="medium")
    # high admits the SMP trust-decay alternative
    _plan(action=OC_ACTION_SMP)


def test_arbitration_only_reserves_no_operator_fe() -> None:
    plan = _plan(
        conflict_level="low",
        action=OC_ACTION_ARBITRATION,
        reserved_fes=0,
        predicted_gain=0.0,
    )
    assert plan.reserved_fes == 0
    with pytest.raises(ValueError, match="reserve no operator FE"):
        _plan(conflict_level="low", action=OC_ACTION_ARBITRATION, reserved_fes=8)
    with pytest.raises(ValueError, match="positive FE budget"):
        _plan(reserved_fes=0)


def test_plan_scope_must_be_sorted_unique_variables() -> None:
    with pytest.raises(ValueError, match="must not repeat"):
        _plan(scope=(4, 4))
    with pytest.raises(ValueError, match="sorted"):
        _plan(scope=(5, 4))


def test_plan_rejects_unknown_levels_and_actions() -> None:
    with pytest.raises(ValueError, match="unknown conflict level"):
        _plan(conflict_level="extreme", action=OC_ACTION_AOR)
    with pytest.raises(ValueError, match="unknown operator action"):
        _plan(action="restart")


def test_config_hysteresis_and_pulse_invariants() -> None:
    with pytest.raises(ValueError, match="tau_exit < tau_enter"):
        OcCoordinatorConfig(tau_enter=0.2, tau_exit=0.2)
    with pytest.raises(ValueError, match="gamma_up must exceed 1"):
        OcCoordinatorConfig(gamma_up=1.0)
    with pytest.raises(ValueError, match="gamma_down must be in"):
        OcCoordinatorConfig(gamma_down=1.5)
    with pytest.raises(ValueError, match="pulse_min_fes must not exceed"):
        OcCoordinatorConfig(pulse_min_fes=64, pulse_max_fes=8)
    config = OcCoordinatorConfig()
    assert config.config_hash == OcCoordinatorConfig().config_hash
    assert OcCoordinatorConfig(ema_alpha=0.5).config_hash != config.config_hash


def test_uncalibrated_registry_matches_config_fields() -> None:
    names = {field.name for field in fields(OcCoordinatorConfig)}
    assert UNCALIBRATED_FIELDS <= names


def _receipt(**overrides) -> OperatorReceipt:
    base = dict(
        plan_hash=_plan().plan_hash,
        cycle_index=3,
        component=(0, 1),
        action=OC_ACTION_CTP_SHARED_CORE,
        conflict_level="high",
        reason="ema_enter_high",
        hub_degree=2,
        relative_hub=0.5,
        reserved_fes=16,
        actual_fes=16,
        status=OC_STATUS_COMPLETED,
        realized_gain=1.25,
        best_error_before=10.0,
        best_error_after=8.75,
        state_hash="0" * 64,
    )
    base.update(overrides)
    return OperatorReceipt(**base)


def test_receipt_requires_exact_parity_on_normal_completion() -> None:
    assert _receipt().receipt_hash == _receipt().receipt_hash
    with pytest.raises(ValueError, match="exact FE parity"):
        _receipt(actual_fes=15)
    with pytest.raises(ValueError, match="status and realized_gain disagree"):
        _receipt(status=OC_STATUS_NO_GAIN, realized_gain=1.25)
    with pytest.raises(ValueError, match="status and realized_gain disagree"):
        _receipt(status=OC_STATUS_COMPLETED, realized_gain=0.0)


def test_receipt_fail_closed_semantics() -> None:
    failed = _receipt(
        status=OC_STATUS_OPERATOR_FAILED,
        actual_fes=6,
        realized_gain=0.0,
        exception_name="RuntimeError",
        remaining_fes=1024,
    )
    assert failed.remaining_fes == 1024
    with pytest.raises(ValueError, match="must name the exception"):
        _receipt(status=OC_STATUS_OPERATOR_FAILED, actual_fes=6, realized_gain=0.0)
    with pytest.raises(ValueError, match="cannot exceed its reservation"):
        _receipt(
            status=OC_STATUS_OPERATOR_FAILED,
            actual_fes=32,
            realized_gain=0.0,
            exception_name="RuntimeError",
        )
    with pytest.raises(ValueError, match="reserved for operator_failed"):
        _receipt(exception_name="RuntimeError")


def test_receipt_from_plan_derives_status_from_gain() -> None:
    plan = _plan()
    gained = receipt_from_plan(
        plan,
        actual_fes=16,
        best_error_before=10.0,
        best_error_after=8.0,
        state_hash="0" * 64,
    )
    assert gained.status == OC_STATUS_COMPLETED
    assert gained.realized_gain == pytest.approx(2.0)
    flat = receipt_from_plan(
        plan,
        actual_fes=16,
        best_error_before=10.0,
        best_error_after=10.0,
        state_hash="0" * 64,
    )
    assert flat.status == OC_STATUS_NO_GAIN
    failed = receipt_from_plan(
        plan,
        actual_fes=4,
        best_error_before=10.0,
        best_error_after=10.0,
        state_hash="0" * 64,
        remaining_fes=1000,
        exception_name="BudgetExceededError",
    )
    assert failed.status == OC_STATUS_OPERATOR_FAILED
    assert failed.actual_fes == 4
    assert failed.remaining_fes == 1000


def test_probe_cost_constant_is_counted_per_variable() -> None:
    assert OC_PROBE_FES_PER_VARIABLE == 2

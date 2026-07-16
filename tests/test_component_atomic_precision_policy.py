from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields

import pytest

from arac.policy.component_atomic_precision import (
    FORBIDDEN_SCHEMA_FIELD_FRAGMENTS,
    ComponentAtomicPlan,
    ComponentEndpointResult,
    ComponentReward,
    build_component_endpoint_result,
    build_component_reward,
    paired_endpoint_tau,
    plan_component_atomic_precision,
)


def _endpoint(
    *,
    checkpoint_error: float = 100.0,
    endpoint_error: float = 90.0,
) -> ComponentEndpointResult:
    return build_component_endpoint_result(
        checkpoint_error=checkpoint_error,
        endpoint_error=endpoint_error,
        canonical_shared_path=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        next_shared_values=(1.0, 0.5),
    )


def _execution() -> dict[str, object]:
    return {
        "group_indices": (2, 3, 4),
        "group_budgets": (32, 48, 64),
        "population_sizes": (16, 16, 16),
        "normal_sigma": 0.25,
        "precision_sigma": 0.125,
    }


def test_first_feasible_component_acquires_the_once_lock() -> None:
    skipped = plan_component_atomic_precision(
        candidate_feasible=False,
        component_unlocked=True,
        horizon_reachable=True,
        once_lock_consumed=False,
        **_execution(),
    )
    selected = plan_component_atomic_precision(
        candidate_feasible=True,
        component_unlocked=True,
        horizon_reachable=True,
        once_lock_consumed=skipped.once_lock_consumed_after,
        **_execution(),
    )
    repeated = plan_component_atomic_precision(
        candidate_feasible=True,
        component_unlocked=True,
        horizon_reachable=True,
        once_lock_consumed=selected.once_lock_consumed_after,
        **_execution(),
    )

    assert skipped.reason == "abstain_candidate_infeasible"
    assert skipped.once_lock_consumed_after is False
    assert selected.execute_precision is True
    assert selected.once_lock_consumed_after is True
    assert repeated.execute_precision is False
    assert repeated.reason == "abstain_once_lock_consumed"


@pytest.mark.parametrize(
    ("component_unlocked", "horizon_reachable", "reason"),
    [
        (False, True, "abstain_component_locked"),
        (True, False, "abstain_component_horizon_unreachable"),
    ],
)
def test_failed_component_guards_do_not_consume_the_once_lock(
    component_unlocked: bool,
    horizon_reachable: bool,
    reason: str,
) -> None:
    plan = plan_component_atomic_precision(
        candidate_feasible=True,
        component_unlocked=component_unlocked,
        horizon_reachable=horizon_reachable,
        once_lock_consumed=False,
        **_execution(),
    )

    assert plan.execute_precision is False
    assert plan.reason == reason
    assert plan.once_lock_consumed_after is False


def test_component_survival_scores_follow_the_frozen_formulas() -> None:
    endpoint = _endpoint()

    assert endpoint.s_h == pytest.approx(1.0)
    assert endpoint.s_d == pytest.approx(0.75)
    assert endpoint.strict_survival is True

    reversing = build_component_endpoint_result(
        checkpoint_error=10.0,
        endpoint_error=9.0,
        canonical_shared_path=((0.0,), (1.0,), (0.5,)),
        next_shared_values=(0.75,),
    )
    assert reversing.s_h == pytest.approx(1.0 / 3.0)
    assert reversing.s_d == pytest.approx(0.5)


def test_zero_endpoint_displacement_is_not_strict_survival() -> None:
    endpoint = build_component_endpoint_result(
        checkpoint_error=10.0,
        endpoint_error=10.0,
        canonical_shared_path=((0.0,), (1.0,), (0.0,)),
        next_shared_values=(0.0,),
    )

    assert endpoint.s_h == 0.0
    assert endpoint.s_d == 0.0
    assert endpoint.strict_survival is False


def test_reward_and_paired_endpoint_tau_use_finite_log_errors() -> None:
    baseline = _endpoint(endpoint_error=90.0)
    precision = _endpoint(endpoint_error=80.0)
    reward = build_component_reward(precision)

    assert reward.log_gain == pytest.approx(math.log(100.0 / 80.0))
    assert reward.material is True
    assert paired_endpoint_tau(baseline, precision) == pytest.approx(
        math.log(90.0 / 80.0)
    )

    floored = build_component_reward(
        ComponentEndpointResult(0.0, 0.0, 0.0, 0.0, False)
    )
    assert floored.log_gain == 0.0
    assert floored.material is False


def test_reward_material_flag_uses_frozen_log_gain_threshold() -> None:
    threshold_endpoint = 100.0 / 1.01

    assert build_component_reward(
        _endpoint(endpoint_error=threshold_endpoint - 1e-6)
    ).material is True
    assert build_component_reward(
        _endpoint(endpoint_error=threshold_endpoint + 1e-6)
    ).material is False


def test_endpoint_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="equal width"):
        build_component_endpoint_result(
            checkpoint_error=1.0,
            endpoint_error=1.0,
            canonical_shared_path=((0.0,), (1.0, 2.0)),
            next_shared_values=(1.0,),
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        ComponentEndpointResult(-1.0, 1.0, 0.0, 0.0, False)
    with pytest.raises(ValueError, match="L1 distance must be finite"):
        build_component_endpoint_result(
            checkpoint_error=1.0,
            endpoint_error=1.0,
            canonical_shared_path=((1e308,), (-1e308,)),
            next_shared_values=(-1e308,),
        )
    with pytest.raises(ValueError, match="same checkpoint"):
        paired_endpoint_tau(
            _endpoint(checkpoint_error=100.0),
            _endpoint(checkpoint_error=101.0),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"group_indices": (2, 4)}, "contiguous"),
        ({"group_budgets": (31, 48, 64)}, "complete populations"),
        ({"population_sizes": (16, 16)}, "equal length"),
        ({"group_budgets": (32.5, 48, 64)}, "only integers"),
        ({"precision_sigma": 0.2}, "precision_sigma must equal"),
    ],
)
def test_atomic_plan_rejects_invalid_component_execution_contract(
    overrides: dict[str, object],
    message: str,
) -> None:
    execution = {**_execution(), **overrides}
    with pytest.raises(ValueError, match=message):
        plan_component_atomic_precision(
            candidate_feasible=True,
            component_unlocked=True,
            horizon_reachable=True,
            once_lock_consumed=False,
            **execution,
        )


def test_atomic_policy_schemas_are_immutable_and_identity_free() -> None:
    schemas = (ComponentAtomicPlan, ComponentEndpointResult, ComponentReward)
    names = {item.name.lower() for schema in schemas for item in fields(schema)}

    assert not {
        name
        for name in names
        if any(fragment in name for fragment in FORBIDDEN_SCHEMA_FIELD_FRAGMENTS)
    }
    plan = plan_component_atomic_precision(
        candidate_feasible=True,
        component_unlocked=True,
        horizon_reachable=True,
        once_lock_consumed=False,
        **_execution(),
    )
    with pytest.raises(FrozenInstanceError):
        plan.execute_precision = False

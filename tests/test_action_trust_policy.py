from __future__ import annotations

import numpy as np
import pytest

from arac.policy.action_trust_policy import (
    ActionTrustConfig,
    ActionTrustPolicy,
    make_action_key,
    normalized_objective_credit,
    robust_damped_writeback,
)


def test_new_action_shadows_existing_writeback_until_risk_is_observed() -> None:
    policy = ActionTrustPolicy(ActionTrustConfig(probation_strength=0.20))

    decision = policy.decide("0:1:shared:coordinate")

    assert decision.phase == "probation"
    assert decision.allow_intervention is True
    assert decision.blend_strength == pytest.approx(1.0)
    assert decision.reason == "probation_shadow"
    assert decision.attempt_count == 1
    assert decision.exposure == pytest.approx(1.0)


def test_one_weak_credit_limits_next_probation_writeback() -> None:
    policy = ActionTrustPolicy(ActionTrustConfig(probation_strength=0.20))
    key = "0:1:shared:coordinate"

    policy.decide(key)
    policy.observe(key, credit=0.0, unstable=False)
    limited = policy.decide(key)

    assert limited.phase == "probation"
    assert limited.allow_intervention is True
    assert limited.blend_strength == pytest.approx(0.20)
    assert limited.reason == "probation_limited"


def test_two_material_positive_credits_promote_action_to_trusted() -> None:
    policy = ActionTrustPolicy(
        ActionTrustConfig(
            probation_strength=0.20,
            trusted_strength=0.60,
            promotion_streak=2,
        )
    )
    key = "0:1:shared:coordinate"

    policy.decide(key)
    policy.observe(key, credit=0.10, unstable=False)
    policy.decide(key)
    promoted = policy.observe(key, credit=0.20, unstable=False)
    decision = policy.decide(key)

    assert promoted.phase == "trusted"
    assert decision.phase == "trusted"
    assert decision.allow_intervention is True
    assert decision.blend_strength == pytest.approx(0.60)
    assert decision.trust_score > 0.0


def test_consecutive_low_gain_quarantines_action() -> None:
    policy = ActionTrustPolicy(
        ActionTrustConfig(quarantine_streak=2, cooldown_steps=2)
    )
    key = "0:1:shared:repair"

    policy.decide(key)
    policy.observe(key, credit=0.0, unstable=False)
    policy.decide(key)
    quarantined = policy.observe(key, credit=-0.01, unstable=False)
    blocked = policy.decide(key)

    assert quarantined.phase == "quarantined"
    assert quarantined.cooldown_remaining == 2
    assert blocked.phase == "quarantined"
    assert blocked.allow_intervention is False
    assert blocked.blend_strength == 0.0
    assert blocked.reason == "cooldown_active"


def test_cooldown_recovers_to_probation_without_resetting_exposure() -> None:
    policy = ActionTrustPolicy(
        ActionTrustConfig(
            probation_strength=0.20,
            quarantine_streak=1,
            cooldown_steps=2,
        )
    )
    key = "0:1:shared:repair"

    first = policy.decide(key)
    policy.observe(key, credit=0.0, unstable=True)
    first_cooldown = policy.decide(key)
    second_cooldown = policy.decide(key)
    recovered = policy.decide(key)

    assert first_cooldown.allow_intervention is False
    assert second_cooldown.allow_intervention is False
    assert recovered.phase == "probation"
    assert recovered.allow_intervention is True
    assert recovered.exposure == pytest.approx(first.exposure + 0.20)


def test_exposure_cap_forces_protected_fallback() -> None:
    policy = ActionTrustPolicy(
        ActionTrustConfig(
            initial_strength=0.20,
            probation_strength=0.20,
            exposure_cap=0.20,
        )
    )
    key = "0:1:shared:coordinate"

    first = policy.decide(key)
    blocked = policy.decide(key)

    assert first.allow_intervention is True
    assert blocked.allow_intervention is False
    assert blocked.reason == "exposure_cap_reached"
    assert blocked.exposure == pytest.approx(0.20)


def test_rollback_latest_decision_restores_attempt_and_exposure() -> None:
    policy = ActionTrustPolicy(ActionTrustConfig(probation_strength=0.20))
    key = "0:1:shared:coordinate"

    policy.decide(key)
    policy.observe(key, credit=0.0, unstable=False)
    before = policy.state_for(key)
    decision = policy.decide(key)

    rolled_back = policy.rollback_decision(decision)

    assert rolled_back.attempt_count == before.attempt_count
    assert rolled_back.exposure == pytest.approx(before.exposure)


def test_make_action_key_is_stable_across_shared_variable_order() -> None:
    first = make_action_key(
        group_left=2,
        group_right=3,
        shared_vars=(9, 4, 9),
        canonical_action_name="allow_beneficial_coordination",
    )
    second = make_action_key(
        group_left=2,
        group_right=3,
        shared_vars=(4, 9),
        canonical_action_name="allow_beneficial_coordination",
    )

    assert first == second
    assert "allow_beneficial_coordination" in first


def test_robust_writeback_damps_and_clips_outlier_proposal() -> None:
    current = np.array([0.0, 0.0])
    proposal = np.array([1000.0, -1000.0])

    adjusted = robust_damped_writeback(
        current_values=current,
        proposed_values=proposal,
        blend_strength=0.50,
        max_delta_norm=1.0,
    )

    assert np.linalg.norm(adjusted - current) == pytest.approx(1.0)
    assert np.all(np.isfinite(adjusted))


def test_robust_writeback_rejects_non_finite_proposal() -> None:
    with pytest.raises(ValueError, match="finite"):
        robust_damped_writeback(
            current_values=np.array([0.0]),
            proposed_values=np.array([np.nan]),
            blend_strength=0.20,
            max_delta_norm=1.0,
        )


def test_objective_credit_is_scale_free_and_uses_minimization_direction() -> None:
    assert normalized_objective_credit(15.0, 10.0) == pytest.approx(1.0 / 3.0)
    assert normalized_objective_credit(10.0, 15.0) == pytest.approx(-1.0 / 3.0)
    assert normalized_objective_credit(0.0, 0.0) == 0.0
    assert normalized_objective_credit(-10.0, 10.0) == -1.0

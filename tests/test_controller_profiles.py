from __future__ import annotations

import pytest

from arac.actions.controller_profiles import (
    controller_profile_by_action,
    controller_profile_by_version,
)


def test_v37_profile_is_the_only_runtime_controller() -> None:
    profile = controller_profile_by_action("arac_evidence_action_controller_v37")

    assert profile == controller_profile_by_version(37)
    assert profile.relation_policy_mode == "controller_v31"
    assert profile.capabilities == {
        "guarded",
        "requires_pinned_environment",
        "trust_trace",
        "risk_aware_trust",
        "maturity",
        "rescue_retirement",
    }
    assert profile.runtime_dispatch_allowed


@pytest.mark.parametrize(
    "action_name",
    (
        "arac_evidence_action_controller_v36",
        "arac_evidence_action_controller_v38",
        "arac_counterfactual_action_racing_w3",
    ),
)
def test_controller_registry_rejects_removed_actions(action_name: str) -> None:
    with pytest.raises(KeyError, match="unknown controller action"):
        controller_profile_by_action(action_name)

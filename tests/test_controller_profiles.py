from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("action_name", "capabilities"),
    [
        ("arac_evidence_action_controller_v35", {"trust_trace", "transparent_trust"}),
        ("arac_evidence_action_controller_v36", {"trust_trace", "maturity"}),
        (
            "arac_evidence_action_controller_v37",
            {"trust_trace", "maturity", "rescue_retirement"},
        ),
        (
            "arac_evidence_action_controller_v38",
            {"trust_trace", "maturity", "rescue_retirement", "precision_reanchor"},
        ),
    ],
)
def test_controller_registry_is_the_single_runtime_capability_source(
    action_name: str,
    capabilities: set[str],
) -> None:
    from arac.actions.controller_profiles import controller_profile_by_action

    profile = controller_profile_by_action(action_name)

    assert profile.action_name == action_name
    assert profile.lane_profile == action_name.removeprefix("arac_")
    assert capabilities.issubset(profile.capabilities)
    assert profile.optimizer_consumed is True
    assert profile.runtime_dispatch_allowed is True
    assert profile.dispatch_boundary == "runtime_evidence_only"


def test_controller_registry_rejects_unknown_actions() -> None:
    from arac.actions.controller_profiles import controller_profile_by_action

    with pytest.raises(KeyError, match="unknown controller action"):
        controller_profile_by_action("arac_evidence_action_controller_v40")


def test_pinned_hcc_environment_includes_deterministic_thread_settings() -> None:
    from arac.execution.environment import PINNED_HCC_RUNTIME_ENVIRONMENT

    assert {
        name: PINNED_HCC_RUNTIME_ENVIRONMENT[name]
        for name in (
            "pythonhashseed",
            "omp_num_threads",
            "openblas_num_threads",
            "mkl_num_threads",
            "numexpr_num_threads",
        )
    } == {
        "pythonhashseed": "0",
        "omp_num_threads": "1",
        "openblas_num_threads": "1",
        "mkl_num_threads": "1",
        "numexpr_num_threads": "1",
    }

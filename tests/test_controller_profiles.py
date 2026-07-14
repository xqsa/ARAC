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
        (
            "arac_counterfactual_action_racing_w",
            {"paired_probe", "branch_isolation", "single_fe_ledger", "writeback_only"},
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
    assert profile.dispatch_boundary.endswith("runtime_evidence_only")


def test_controller_registry_rejects_unknown_actions() -> None:
    from arac.actions.controller_profiles import controller_profile_by_action

    with pytest.raises(KeyError, match="unknown controller action"):
        controller_profile_by_action("arac_evidence_action_controller_v40")


def test_car_w_profile_freezes_the_audited_protocol() -> None:
    from arac.actions.controller_profiles import controller_profile_by_action

    profile = controller_profile_by_action("arac_counterfactual_action_racing_w")

    assert profile.version is None
    assert profile.relation_policy_mode == "controller_v31"
    assert profile.dispatch_boundary == "identity_free_runtime_evidence_only"
    assert profile.optimizer_consumed_parameters == {
        "candidate_proposal": "controller_v31_relation_proposal",
        "candidate_blend_alpha": 0.20,
        "candidate_norm_guard": "controller_v33_norm_guard",
        "fallback": "arac_evidence_action_controller_v33",
        "minimum_complete_evidence_sweeps": 2,
        "stable_support_rule": "two_sweep_non_fallback_subgraph",
        "paired_probe_count": 3,
        "writeback_probe_budget_fraction": 0.03,
        "deployment_pair_index": 2,
        "candidate_lease": "final_pair_component_horizon_only",
        "dispatch_boundary": "identity_free_runtime_evidence_only",
    }


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

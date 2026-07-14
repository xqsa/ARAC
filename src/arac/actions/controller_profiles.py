"""Single source of truth for audited HCC controller metadata.

The registry contains only pre-registered runtime contracts. It must never
contain case identities, function families, reported values, or run outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerProfile:
    version: int
    lane_profile: str
    action_name: str
    dispatch_scope: str
    relation_policy_mode: str
    backend_effect_kind: str
    optimizer_consumed_parameters: dict[str, object]
    execution_mode: str
    capabilities: frozenset[str]
    optimizer_consumed: bool = True
    runtime_dispatch_allowed: bool = True
    dispatch_boundary: str = "runtime_evidence_only"

    def hcc_action_effect(self) -> tuple[str, dict[str, object], str, bool, str]:
        parameters = dict(self.optimizer_consumed_parameters)
        parameters["dispatch_boundary"] = self.dispatch_boundary
        return (
            self.backend_effect_kind,
            parameters,
            self.execution_mode,
            self.optimizer_consumed,
            "" if self.runtime_dispatch_allowed else "runtime_dispatch_blocked",
        )


def _profile(
    version: int,
    *,
    dispatch_scope: str,
    backend_parameters: dict[str, object],
    capabilities: set[str],
) -> ControllerProfile:
    suffix = str(version)
    return ControllerProfile(
        version=version,
        lane_profile=f"evidence_action_controller_v{suffix}",
        action_name=f"arac_evidence_action_controller_v{suffix}",
        dispatch_scope=dispatch_scope,
        relation_policy_mode="controller_v31",
        backend_effect_kind=f"evidence_action_runtime_controller_v{suffix}",
        optimizer_consumed_parameters=backend_parameters,
        execution_mode=f"hcc_evidence_action_controller_v{suffix}_runtime_consumed",
        capabilities=frozenset({"guarded", "requires_pinned_environment", *capabilities}),
    )


CONTROLLER_PROFILES = (
    _profile(
        33,
        dispatch_scope="single_run_risk_aware_runtime_evidence_controller_v33",
        backend_parameters={
            "relation_runtime_hook": "controller_v33_risk_aware_action_guard",
            "mode_selector": "early_runtime_overlap_relation_evidence_with_relation_first_lock",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": ["phase_rescue_multistart"],
            "guard": "probation_trust_quarantine_and_exposure_cap",
            "writeback": "topology_scoped_fallback_and_bounded_active_damping",
        },
        capabilities={"trust_trace", "risk_aware_trust"},
    ),
    _profile(
        34,
        dispatch_scope="single_run_downstream_recovery_runtime_evidence_controller_v34",
        backend_parameters={
            "relation_runtime_hook": "controller_v33_risk_aware_action_guard",
            "mode_selector": "early_runtime_overlap_relation_evidence_with_relation_first_lock",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": ["phase_rescue_multistart"],
            "guard": "probation_trust_quarantine_and_exposure_cap",
            "writeback": "topology_scoped_fallback_and_bounded_active_damping",
            "trajectory_guard": "downstream_recovery_checkpoint",
        },
        capabilities={"trust_trace", "risk_aware_trust", "trajectory_guard"},
    ),
    _profile(
        35,
        dispatch_scope="single_run_transparent_trust_topology_guard_controller_v35",
        backend_parameters={
            "relation_runtime_hook": "controller_v35_transparent_topology_guard",
            "mode_selector": "current_run_phase_i_relation_evidence",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": ["phase_rescue_multistart"],
            "guard": "topology_scoped_fallback_only",
            "writeback": "transparent_active_and_topology_scoped_fallback",
        },
        capabilities={"trust_trace", "transparent_trust"},
    ),
    _profile(
        36,
        dispatch_scope="single_run_first_sweep_evidence_maturity_controller_v36",
        backend_parameters={
            "relation_runtime_hook": "controller_v36_maturity_guarded_relation_dispatch",
            "mode_selector": "current_run_first_sweep_relation_evidence",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": ["phase_rescue_multistart"],
            "guard": "repair_lock_and_first_sweep_evidence_maturity",
            "writeback": "maturity_scoped_active_and_topology_scoped_fallback",
        },
        capabilities={"trust_trace", "risk_aware_trust", "maturity"},
    ),
    _profile(
        37,
        dispatch_scope="single_run_zero_yield_phase_rescue_retirement_controller_v37",
        backend_parameters={
            "relation_runtime_hook": "controller_v36_maturity_guarded_relation_dispatch",
            "mode_selector": "current_run_first_sweep_relation_and_rescue_evidence",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": [
                "phase_rescue_multistart",
                "zero_yield_phase_rescue_retirement",
            ],
            "resource_runtime_hook": "zero_yield_phase_rescue_retirement",
            "guard": "repair_lock_first_sweep_maturity_and_rescue_productivity",
        },
        capabilities={
            "trust_trace",
            "risk_aware_trust",
            "maturity",
            "rescue_retirement",
        },
    ),
    _profile(
        38,
        dispatch_scope="single_run_post_retirement_precision_reanchor_controller_v38",
        backend_parameters={
            "relation_runtime_hook": "controller_v36_maturity_guarded_relation_dispatch",
            "mode_selector": "current_run_relation_rescue_and_search_state_evidence",
            "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
            "search_state_runtime_hooks": [
                "phase_rescue_multistart",
                "zero_yield_phase_rescue_retirement",
                "post_retirement_precision_reanchor",
            ],
            "optimizer_runtime_hook": "post_retirement_precision_reanchor",
            "guard": "repair_lock_first_sweep_maturity_and_rescue_productivity",
        },
        capabilities={
            "trust_trace",
            "risk_aware_trust",
            "maturity",
            "rescue_retirement",
            "precision_reanchor",
        },
    ),
    _profile(
        39,
        dispatch_scope="single_run_cross_sweep_cma_sigma_continuation_controller_v39",
        backend_parameters={
            "relation_runtime_hook": "controller_v36_maturity_guarded_relation_dispatch",
            "mode_selector": "current_run_phase_i_relation_and_optimizer_evidence",
            "search_state_runtime_hooks": [
                "phase_rescue_multistart",
                "zero_yield_phase_rescue_retirement",
                "post_retirement_precision_reanchor",
                "cross_sweep_cma_terminal_sigma_continuation",
            ],
            "optimizer_runtime_hook": "cross_sweep_cma_terminal_sigma_continuation",
        },
        capabilities={
            "trust_trace",
            "risk_aware_trust",
            "maturity",
            "rescue_retirement",
            "precision_reanchor",
            "sigma_continuation",
        },
    ),
)

_PROFILES_BY_ACTION = {profile.action_name: profile for profile in CONTROLLER_PROFILES}
_PROFILES_BY_LANE = {profile.lane_profile: profile for profile in CONTROLLER_PROFILES}
_PROFILES_BY_VERSION = {profile.version: profile for profile in CONTROLLER_PROFILES}


def controller_profile_by_action(action_name: str) -> ControllerProfile:
    try:
        return _PROFILES_BY_ACTION[action_name]
    except KeyError as exc:
        raise KeyError(f"unknown controller action: {action_name}") from exc


def controller_profile_by_lane(lane_profile: str) -> ControllerProfile:
    try:
        return _PROFILES_BY_LANE[lane_profile]
    except KeyError as exc:
        raise KeyError(f"unknown controller lane profile: {lane_profile}") from exc


def controller_profile_by_version(version: int) -> ControllerProfile:
    try:
        return _PROFILES_BY_VERSION[int(version)]
    except KeyError as exc:
        raise KeyError(f"unknown controller version: {version}") from exc


def controller_has_capability(action_name: str, capability: str) -> bool:
    profile = _PROFILES_BY_ACTION.get(action_name)
    return profile is not None and capability in profile.capabilities


def controller_lane_profile_names() -> tuple[str, ...]:
    return tuple(profile.lane_profile for profile in CONTROLLER_PROFILES)


def controller_action_effects() -> dict[str, tuple[str, dict[str, object], str, bool, str]]:
    return {
        profile.action_name: profile.hcc_action_effect()
        for profile in CONTROLLER_PROFILES
    }

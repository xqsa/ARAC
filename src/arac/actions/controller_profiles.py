"""Frozen runtime metadata for the exp_018 v37 controller."""

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


V37_PROFILE = ControllerProfile(
    version=37,
    lane_profile="evidence_action_controller_v37",
    action_name="arac_evidence_action_controller_v37",
    dispatch_scope="single_run_zero_yield_phase_rescue_retirement_controller_v37",
    relation_policy_mode="controller_v31",
    backend_effect_kind="evidence_action_runtime_controller_v37",
    optimizer_consumed_parameters={
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
    execution_mode="hcc_evidence_action_controller_v37_runtime_consumed",
    capabilities=frozenset(
        {
            "guarded",
            "requires_pinned_environment",
            "trust_trace",
            "risk_aware_trust",
            "maturity",
            "rescue_retirement",
        }
    ),
)

CONTROLLER_PROFILES = (V37_PROFILE,)


def controller_profile_by_action(action_name: str) -> ControllerProfile:
    if action_name != V37_PROFILE.action_name:
        raise KeyError(f"unknown controller action: {action_name}")
    return V37_PROFILE


def controller_profile_by_lane(lane_profile: str) -> ControllerProfile:
    if lane_profile != V37_PROFILE.lane_profile:
        raise KeyError(f"unknown controller lane profile: {lane_profile}")
    return V37_PROFILE


def controller_profile_by_version(version: int) -> ControllerProfile:
    if int(version) != V37_PROFILE.version:
        raise KeyError(f"unknown controller version: {version}")
    return V37_PROFILE


def controller_has_capability(action_name: str, capability: str) -> bool:
    return action_name == V37_PROFILE.action_name and capability in V37_PROFILE.capabilities


def controller_lane_profile_names() -> tuple[str, ...]:
    return (V37_PROFILE.lane_profile,)


def controller_action_effects() -> dict[str, tuple[str, dict[str, object], str, bool, str]]:
    return {V37_PROFILE.action_name: V37_PROFILE.hcc_action_effect()}

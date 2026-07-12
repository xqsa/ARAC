"""Core action taxonomy for ARAC.

Backend optimizers and executors are support surfaces. They should not be
counted as core intervention actions unless they change optimizer-consumed
semantics through one of these action families.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ActionFamily(StrEnum):
    COORDINATE = "coordinate"
    ISOLATE = "isolate"
    PROTECT = "protect"
    TRAJECTORY = "trajectory"
    REASSIGN_REPAIR = "reassign_repair"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ActionDecision:
    action_family: ActionFamily
    action_name: str
    decision: str
    trigger_reason: str
    utility_proxy: float
    fallback_action: str = "conservative_no_action"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    family: ActionFamily
    backend_role: str
    requires_semantic_effect: bool = True


DEFAULT_ACTION_SPACE = (
    ActionSpec("allow_beneficial_coordination", ActionFamily.COORDINATE, "core_intervention"),
    ActionSpec("isolate_conflicting_relation", ActionFamily.ISOLATE, "core_intervention"),
    ActionSpec("protect_high_margin_group", ActionFamily.PROTECT, "core_intervention"),
    ActionSpec("budget_shift_mean_blend", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("bipop_search_state_restart", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("resume_phase_i_search_state", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("phase_rescue_multistart", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("repair_phase_rescue_multistart", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("cc_harm_guarded_sep_refresh", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("separable_cmaes_dispatch_action", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("arac_evidence_action_controller_v1", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("arac_evidence_action_controller_v2", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("arac_evidence_action_controller_v3", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("arac_evidence_action_controller_v31", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("arac_evidence_action_controller_v32", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("repair_bipop_search_state_restart", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("repair_protect_refine", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("repair_protect_deep_refine", ActionFamily.TRAJECTORY, "core_intervention"),
    ActionSpec("repair_shared_variable_binding", ActionFamily.REASSIGN_REPAIR, "core_intervention"),
    ActionSpec("conservative_no_action", ActionFamily.FALLBACK, "fallback", False),
)


def action_by_name(name: str, action_space=DEFAULT_ACTION_SPACE) -> ActionSpec:
    for action in action_space:
        if action.name == name:
            return action
    raise KeyError(f"unknown action: {name}")

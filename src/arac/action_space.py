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
    REASSIGN_REPAIR = "reassign_repair"
    FALLBACK = "fallback"


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
    ActionSpec("repair_shared_variable_binding", ActionFamily.REASSIGN_REPAIR, "core_intervention"),
    ActionSpec("conservative_no_action", ActionFamily.FALLBACK, "fallback", False),
)


def action_by_name(name: str, action_space=DEFAULT_ACTION_SPACE) -> ActionSpec:
    for action in action_space:
        if action.name == name:
            return action
    raise KeyError(f"unknown action: {name}")


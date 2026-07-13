"""Relation-level policy decision contracts."""

from __future__ import annotations

from dataclasses import dataclass


ACTION_NAMES = (
    "coordinate",
    "isolate_conflicting_relation",
    "reassign_repair",
    "fallback",
)
RELATION_ACTION_ALIASES = {
    "fallback": "conservative_no_action",
    "coordinate": "allow_beneficial_coordination",
    "reassign_repair": "repair_shared_variable_binding",
    "isolate_conflicting_relation": "isolate_conflicting_relation",
}


@dataclass
class ActionDecision:
    relation_id: str
    action_name: str
    action_family: str
    confidence: float
    trigger_reason: str
    relation_action_name: str = ""
    canonical_action_name: str = ""

    def __post_init__(self) -> None:
        if not self.relation_action_name:
            self.relation_action_name = self.action_name
        if not self.canonical_action_name:
            self.canonical_action_name = RELATION_ACTION_ALIASES[self.relation_action_name]


@dataclass(frozen=True)
class ScoredActionDecision:
    relation_id: str
    candidate_scores: dict[str, float]
    final_action: ActionDecision
    best_action_name: str
    best_score: float
    second_best_action_name: str
    second_best_score: float
    margin: float
    abstain_reason: str

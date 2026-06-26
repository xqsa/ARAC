"""Deterministic overlap-relation action policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from arac.evidence.overlap_relation_builder import OverlapRelation

HIGH_OVERLAP_THRESHOLD = 1.0
CONFLICT_THRESHOLD = 1.0
STABILITY_THRESHOLD = 0.75
ACTION_NAMES = (
    "coordinate",
    "isolate_conflicting_relation",
    "reassign_repair",
    "fallback",
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ActionDecision:
    relation_id: str
    action_name: str
    action_family: str
    confidence: float
    trigger_reason: str


def decide_action(relation: OverlapRelation) -> ActionDecision:
    """Choose one deterministic action for an overlap relation."""

    abs_delta = abs(relation.delta_signal)
    high_overlap = relation.overlap_strength >= HIGH_OVERLAP_THRESHOLD
    stable_delta = abs_delta <= (1.0 - STABILITY_THRESHOLD)
    stable_rank = relation.rank_signal >= STABILITY_THRESHOLD

    if relation.delta_signal < 0 or abs_delta >= CONFLICT_THRESHOLD:
        return ActionDecision(
            relation_id=relation.relation_id,
            action_name="isolate_conflicting_relation",
            action_family="isolate",
            confidence=_clamp(abs_delta / max(CONFLICT_THRESHOLD, 1e-12)),
            trigger_reason="large_delta_conflict_or_negative_divergence",
        )

    if high_overlap and stable_delta and stable_rank:
        confidence = _mean(
            _overlap_confidence(relation.overlap_strength),
            1.0 - _clamp(abs_delta / max(CONFLICT_THRESHOLD, 1e-12)),
            relation.rank_signal,
        )
        return ActionDecision(
            relation_id=relation.relation_id,
            action_name="coordinate",
            action_family="coordinate",
            confidence=confidence,
            trigger_reason="high_overlap_with_stable_delta_and_rank",
        )

    if high_overlap and (abs_delta > (1.0 - STABILITY_THRESHOLD) or not stable_rank):
        confidence = _mean(
            _overlap_confidence(relation.overlap_strength),
            _clamp(abs_delta / max(CONFLICT_THRESHOLD, 1e-12)),
            1.0 - _clamp(relation.rank_signal),
        )
        return ActionDecision(
            relation_id=relation.relation_id,
            action_name="reassign_repair",
            action_family="reassign_repair",
            confidence=confidence,
            trigger_reason="overlap_relation_has_imbalance_or_unstable_rank",
        )

    return ActionDecision(
        relation_id=relation.relation_id,
        action_name="fallback",
        action_family="fallback",
        confidence=0.0,
        trigger_reason="no_deterministic_relation_rule_triggered",
    )


def decide_actions_for_relations(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [decide_action(relation) for relation in relations]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.action_name] += 1
    LOGGER.info(
        "relation policy action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def _overlap_confidence(overlap_strength: float) -> float:
    return _clamp(overlap_strength / max(HIGH_OVERLAP_THRESHOLD, 1e-12))


def _mean(*values: float) -> float:
    return _clamp(sum(values) / len(values))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

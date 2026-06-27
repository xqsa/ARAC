"""Deterministic overlap-relation action policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from arac.evidence.overlap_relation_builder import OverlapRelation

HIGH_OVERLAP_THRESHOLD = 1.0
CONFLICT_THRESHOLD = 0.75
STABILITY_THRESHOLD = 0.75
MIN_FEATURE_COVERAGE = 0.80
MIN_BUDGET_REMAINING_RATIO = 0.05
MIN_FALLBACK_MARGIN_PROXY = 0.20
HIGH_FALLBACK_MARGIN_THRESHOLD = 0.95
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

LOGGER = logging.getLogger(__name__)


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


def decide_action(relation: OverlapRelation) -> ActionDecision:
    """Choose one deterministic action for an overlap relation."""

    abs_delta = relation.delta_abs_gap or abs(relation.delta_signal)
    signed_delta = relation.delta_signed_gap
    delta_ratio_gap = relation.delta_ratio_gap
    rank_stability = relation.rank_stability or relation.rank_signal
    high_overlap = relation.overlap_strength >= HIGH_OVERLAP_THRESHOLD
    stable_delta = delta_ratio_gap <= (1.0 - STABILITY_THRESHOLD)
    stable_rank = rank_stability >= STABILITY_THRESHOLD

    if relation.overlap_strength < HIGH_OVERLAP_THRESHOLD or relation.shared_var_count <= 0:
        return _decision(
            relation,
            "fallback",
            "fallback",
            0.0,
            "no_shared_overlap_support",
        )

    if (
        relation.feature_coverage < MIN_FEATURE_COVERAGE
        or relation.budget_remaining_ratio < MIN_BUDGET_REMAINING_RATIO
        or relation.fallback_margin_proxy < MIN_FALLBACK_MARGIN_PROXY
    ):
        return _decision(
            relation,
            "fallback",
            "fallback",
            0.0,
            "insufficient_relation_policy_safety_margin",
        )

    if (
        relation.both_positive
        and signed_delta < 0.0
        and delta_ratio_gap >= CONFLICT_THRESHOLD
        and relation.fallback_margin_proxy >= HIGH_FALLBACK_MARGIN_THRESHOLD
    ):
        confidence = _mean(
            _overlap_confidence(relation.overlap_strength),
            relation.fallback_margin_proxy,
            1.0 - relation.shared_var_support_ratio,
        )
        return _decision(
            relation,
            "coordinate",
            "coordinate",
            confidence,
            "high_fallback_margin_supports_safe_coordination",
        )

    if signed_delta < 0.0 and delta_ratio_gap >= CONFLICT_THRESHOLD:
        return _decision(
            relation,
            "isolate_conflicting_relation",
            "isolate",
            _clamp(delta_ratio_gap / max(CONFLICT_THRESHOLD, 1e-12)),
            "large_delta_conflict_or_negative_divergence",
        )

    if high_overlap and relation.both_positive and stable_delta and stable_rank:
        confidence = _mean(
            _overlap_confidence(relation.overlap_strength),
            1.0 - _clamp(abs_delta / max(CONFLICT_THRESHOLD, 1e-12)),
            rank_stability,
        )
        return _decision(
            relation,
            "coordinate",
            "coordinate",
            confidence,
            "high_overlap_with_stable_delta_and_rank",
        )

    if (
        high_overlap
        and relation.both_positive
        and relation.fallback_margin_proxy >= HIGH_FALLBACK_MARGIN_THRESHOLD
    ):
        return _decision(
            relation,
            "fallback",
            "fallback",
            0.0,
            "high_fallback_margin_keeps_native_overlap_blend",
        )

    if high_overlap and (
        relation.one_side_zero
        or delta_ratio_gap > (1.0 - STABILITY_THRESHOLD)
        or not stable_rank
    ):
        confidence = _mean(
            _overlap_confidence(relation.overlap_strength),
            _clamp(max(delta_ratio_gap, abs_delta / max(CONFLICT_THRESHOLD, 1e-12))),
            1.0 - _clamp(rank_stability),
        )
        return _decision(
            relation,
            "reassign_repair",
            "reassign_repair",
            confidence,
            "overlap_relation_has_imbalance_or_unstable_rank",
        )

    return _decision(
        relation,
        "fallback",
        "fallback",
        0.0,
        "no_deterministic_relation_rule_triggered",
    )


def decide_actions_for_relations(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [decide_action(relation) for relation in relations]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.relation_action_name] += 1
    LOGGER.info(
        "relation policy action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def _overlap_confidence(overlap_strength: float) -> float:
    return _clamp(overlap_strength / max(HIGH_OVERLAP_THRESHOLD, 1e-12))


def _decision(
    relation: OverlapRelation,
    relation_action_name: str,
    action_family: str,
    confidence: float,
    trigger_reason: str,
) -> ActionDecision:
    return ActionDecision(
        relation_id=relation.relation_id,
        action_name=relation_action_name,
        relation_action_name=relation_action_name,
        canonical_action_name=RELATION_ACTION_ALIASES[relation_action_name],
        action_family=action_family,
        confidence=_clamp(confidence),
        trigger_reason=trigger_reason,
    )


def _mean(*values: float) -> float:
    return _clamp(sum(values) / len(values))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

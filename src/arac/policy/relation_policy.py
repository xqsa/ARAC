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
MIN_ACTIVE_REBIND_SUPPORT_RATIO = 0.05
MAX_ACTIVE_REBIND_SUPPORT_RATIO = 0.20
DENSE_PREFIX_COORDINATE_SUPPORT_THRESHOLD = 0.24
DENSE_PREFIX_COORDINATE_SUPPORT_MAX = 0.30
BALANCED_MID_SUPPORT_COORDINATE_MIN = 0.14
BALANCED_MID_SUPPORT_COORDINATE_MAX = 0.17
HIGH_FALLBACK_MARGIN_THRESHOLD = 0.95
ACTION_MARGIN_THRESHOLD = 0.05
FALLBACK_SCORE_DISCOUNT = 0.10
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


def decide_action(relation: OverlapRelation) -> ActionDecision:
    """Choose one deterministic action for an overlap relation."""

    return score_relation_actions(relation).final_action


def score_relation_actions(relation: OverlapRelation) -> ScoredActionDecision:
    """Score deterministic action candidates and apply a margin abstain rule."""

    abs_delta = relation.delta_abs_gap or abs(relation.delta_signal)
    signed_delta = relation.delta_signed_gap
    delta_ratio_gap = relation.delta_ratio_gap
    rank_stability = relation.rank_stability or relation.rank_signal
    high_overlap = relation.overlap_strength >= HIGH_OVERLAP_THRESHOLD
    stable_delta = delta_ratio_gap <= (1.0 - STABILITY_THRESHOLD)
    stable_rank = rank_stability >= STABILITY_THRESHOLD
    strong_rebinding_allowed = (
        relation.shared_var_support_ratio >= MIN_ACTIVE_REBIND_SUPPORT_RATIO
    )
    dense_rebinding_blocked = (
        relation.shared_var_support_ratio >= MAX_ACTIVE_REBIND_SUPPORT_RATIO
    )
    scores = {action_name: 0.0 for action_name in ACTION_NAMES}
    reasons = {action_name: "" for action_name in ACTION_NAMES}
    fallback_reason = "no_deterministic_relation_rule_triggered"
    scores["fallback"] = _clamp(relation.fallback_margin_proxy - FALLBACK_SCORE_DISCOUNT)

    if (
        relation.overlap_strength < HIGH_OVERLAP_THRESHOLD
        or relation.shared_var_count <= 0
    ):
        fallback_reason = "no_shared_overlap_support"
    elif (
        relation.feature_coverage < MIN_FEATURE_COVERAGE
        or relation.budget_remaining_ratio < MIN_BUDGET_REMAINING_RATIO
        or relation.fallback_margin_proxy < MIN_FALLBACK_MARGIN_PROXY
    ):
        fallback_reason = "insufficient_relation_policy_safety_margin"
    else:
        if (
            relation.both_positive
            and signed_delta < 0.0
            and delta_ratio_gap >= CONFLICT_THRESHOLD
            and relation.fallback_margin_proxy >= HIGH_FALLBACK_MARGIN_THRESHOLD
        ):
            _set_candidate_score(
                scores,
                reasons,
                "coordinate",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    relation.fallback_margin_proxy,
                    1.0 - relation.shared_var_support_ratio,
                ),
                "high_fallback_margin_supports_safe_coordination",
            )

        if (
            signed_delta < 0.0
            and delta_ratio_gap >= CONFLICT_THRESHOLD
            and dense_rebinding_blocked
        ):
            fallback_reason = "very_dense_shared_support_blocks_active_relation_dispatch"
        elif (
            signed_delta < 0.0
            and delta_ratio_gap >= CONFLICT_THRESHOLD
            and strong_rebinding_allowed
        ):
            _set_candidate_score(
                scores,
                reasons,
                "isolate_conflicting_relation",
                _clamp(delta_ratio_gap / max(CONFLICT_THRESHOLD, 1e-12)),
                "large_delta_conflict_or_negative_divergence",
            )

        if high_overlap and relation.both_positive and stable_delta and stable_rank:
            _set_candidate_score(
                scores,
                reasons,
                "coordinate",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    1.0 - _clamp(delta_ratio_gap),
                    rank_stability,
                ),
                "high_overlap_with_stable_delta_and_rank",
            )

        if (
            high_overlap
            and relation.both_positive
            and relation.fallback_margin_proxy >= HIGH_FALLBACK_MARGIN_THRESHOLD
            and not scores["coordinate"]
        ):
            fallback_reason = "high_fallback_margin_keeps_native_overlap_blend"

        if (
            fallback_reason != "high_fallback_margin_keeps_native_overlap_blend"
            and high_overlap
            and not strong_rebinding_allowed
            and (
            relation.one_side_zero
            or signed_delta < 0.0
            or delta_ratio_gap > (1.0 - STABILITY_THRESHOLD)
            or not stable_rank
            )
        ):
            fallback_reason = "low_shared_support_blocks_strong_relation_rebinding"
        elif high_overlap and dense_rebinding_blocked and (
            relation.one_side_zero
            or delta_ratio_gap > (1.0 - STABILITY_THRESHOLD)
            or not stable_rank
        ):
            fallback_reason = "very_dense_shared_support_blocks_active_relation_dispatch"
        elif (
            fallback_reason != "high_fallback_margin_keeps_native_overlap_blend"
            and high_overlap
            and relation.one_side_zero
            and (
                delta_ratio_gap > (1.0 - STABILITY_THRESHOLD)
                or not stable_rank
            )
        ):
            _set_candidate_score(
                scores,
                reasons,
                "reassign_repair",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    _clamp(max(delta_ratio_gap, abs_delta / max(CONFLICT_THRESHOLD, 1e-12))),
                    1.0 - _clamp(rank_stability),
                ),
                "overlap_relation_has_imbalance_or_unstable_rank",
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], _action_sort_order(item[0])),
    )
    best_action_name, best_score = ranked[0]
    second_best_action_name, second_best_score = ranked[1]
    margin = best_score - second_best_score
    abstain_reason = ""
    if best_action_name != "fallback" and margin >= ACTION_MARGIN_THRESHOLD:
        final_action = _decision(
            relation,
            best_action_name,
            _action_family(best_action_name),
            best_score,
            reasons[best_action_name],
        )
    else:
        if best_action_name != "fallback" and margin < ACTION_MARGIN_THRESHOLD:
            abstain_reason = "candidate_margin_below_threshold"
            fallback_reason = abstain_reason
        final_action = _decision(
            relation,
            "fallback",
            "fallback",
            0.0,
            fallback_reason,
        )

    return ScoredActionDecision(
        relation_id=relation.relation_id,
        candidate_scores={name: _clamp(scores[name]) for name in ACTION_NAMES},
        final_action=final_action,
        best_action_name=best_action_name,
        best_score=_clamp(best_score),
        second_best_action_name=second_best_action_name,
        second_best_score=_clamp(second_best_score),
        margin=_clamp(margin),
        abstain_reason=abstain_reason,
    )


def action_mismatch_audit_row(
    relation: OverlapRelation,
    scored: ScoredActionDecision | None = None,
    final_action: ActionDecision | None = None,
) -> dict[str, str]:
    if scored is None:
        scored = score_relation_actions(relation)
    if final_action is None:
        final_action = scored.final_action
    return {
        "problem_id": relation.problem_id,
        "relation_id": relation.relation_id,
        "group_left": str(relation.group_left),
        "group_right": str(relation.group_right),
        "candidate_scores": ";".join(
            f"{action_name}={scored.candidate_scores[action_name]:.6f}"
            for action_name in ACTION_NAMES
        ),
        "coordinate_score": f"{scored.candidate_scores['coordinate']:.6f}",
        "isolate_conflicting_relation_score": (
            f"{scored.candidate_scores['isolate_conflicting_relation']:.6f}"
        ),
        "reassign_repair_score": f"{scored.candidate_scores['reassign_repair']:.6f}",
        "fallback_score": f"{scored.candidate_scores['fallback']:.6f}",
        "best_action_name": scored.best_action_name,
        "best_score": f"{scored.best_score:.6f}",
        "second_best_action_name": scored.second_best_action_name,
        "second_best_score": f"{scored.second_best_score:.6f}",
        "margin": f"{scored.margin:.6f}",
        "final_action_name": final_action.relation_action_name,
        "final_canonical_action_name": final_action.canonical_action_name,
        "confidence": f"{final_action.confidence:.6f}",
        "trigger_reason": final_action.trigger_reason,
        "abstain_reason": scored.abstain_reason,
    }


def decide_actions_for_relations(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [decide_action(relation) for relation in relations]
    dense_prefix_seen = False
    balanced_mid_support_seen = False
    prefix_has_one_side_zero = False
    for index, relation in enumerate(relations):
        prefix_has_one_side_zero = prefix_has_one_side_zero or relation.one_side_zero
        dense_prefix_seen = dense_prefix_seen or (
            relation.shared_var_support_ratio
            >= DENSE_PREFIX_COORDINATE_SUPPORT_THRESHOLD
            and relation.shared_var_support_ratio <= DENSE_PREFIX_COORDINATE_SUPPORT_MAX
        )
        balanced_mid_support_seen = balanced_mid_support_seen or (
            not prefix_has_one_side_zero
            and relation.both_positive
            and relation.shared_var_support_ratio >= BALANCED_MID_SUPPORT_COORDINATE_MIN
            and relation.shared_var_support_ratio <= BALANCED_MID_SUPPORT_COORDINATE_MAX
            and relation.delta_ratio_gap >= CONFLICT_THRESHOLD
            and relation.rank_stability >= STABILITY_THRESHOLD
        )
        if (
            dense_prefix_seen
            and relation.both_positive
            and not relation.one_side_zero
            and decisions[index].relation_action_name == "coordinate"
        ):
            decisions[index] = _decision(
                relation,
                "coordinate",
                "coordinate",
                max(decisions[index].confidence, relation.fallback_margin_proxy),
                "dense_prefix_coordinate_mode",
            )
        elif balanced_mid_support_seen:
            decisions[index] = _decision(
                relation,
                "coordinate",
                "coordinate",
                max(decisions[index].confidence, relation.fallback_margin_proxy),
                "balanced_mid_support_coordinate_mode",
            )
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


def _set_candidate_score(
    scores: dict[str, float],
    reasons: dict[str, str],
    action_name: str,
    score: float,
    reason: str,
) -> None:
    score = _clamp(score)
    if score > scores[action_name]:
        scores[action_name] = score
        reasons[action_name] = reason


def _action_sort_order(action_name: str) -> int:
    order = {
        "fallback": 0,
        "coordinate": 1,
        "reassign_repair": 2,
        "isolate_conflicting_relation": 3,
    }
    return order[action_name]


def _action_family(action_name: str) -> str:
    if action_name == "coordinate":
        return "coordinate"
    if action_name == "reassign_repair":
        return "reassign_repair"
    if action_name == "isolate_conflicting_relation":
        return "isolate"
    return "fallback"


def _mean(*values: float) -> float:
    return _clamp(sum(values) / len(values))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

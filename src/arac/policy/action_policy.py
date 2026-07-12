"""Deterministic overlap-relation action policy."""

from __future__ import annotations

import logging

from arac.evidence.overlap_relation_builder import OverlapRelation

from .evidence_model import (
    ACTION_NAMES,
    RELATION_ACTION_ALIASES,
    ActionDecision,
    ScoredActionDecision,
)

HIGH_OVERLAP_THRESHOLD = 1.0
CONFLICT_THRESHOLD = 0.75
STABILITY_THRESHOLD = 0.75
MIN_FEATURE_COVERAGE = 0.80
MIN_BUDGET_REMAINING_RATIO = 0.05
MIN_FALLBACK_MARGIN_PROXY = 0.20
MIN_ACTIVE_REBIND_SUPPORT_RATIO = 0.05
MAX_ACTIVE_REBIND_SUPPORT_RATIO = 0.20
BALANCED_MID_SUPPORT_COORDINATE_MIN = 0.14
BALANCED_MID_SUPPORT_COORDINATE_MAX = 0.17
DENSE_REPAIR_SUPPORT_THRESHOLD = 0.20
DENSE_REPAIR_DELTA_MIN = 0.30
DENSE_REPAIR_DELTA_MAX = 0.75
DENSE_REPAIR_RANK_STABILITY_MIN = 0.50
MID_DENSE_ACTIVE_SUPPORT_MAX = 0.25
HIGH_FALLBACK_MARGIN_THRESHOLD = 0.95
ACTION_MARGIN_THRESHOLD = 0.05
FALLBACK_SCORE_DISCOUNT = 0.10
V2_HIGH_CONFLICT_MIN = 0.40
V2_COORDINATE_SUPPORT_MIN = 0.10
V2_COORDINATE_SUPPORT_MAX = 0.28
V2_REPAIR_SUPPORT_MIN = 0.18
V2_REPAIR_SUPPORT_MAX = 0.32
V2_REPAIR_STABILITY_MIN = 0.65
V2_REPAIR_FALLBACK_MARGIN_MIN = 0.80
V21_COORDINATE_CONTEXT_TRIGGER_COUNT = 2
V22_EARLY_LOCK_TRIGGER_COUNT = 1
V24_CONTEXT_SUPPORT_MIN = 0.14
V24_CONTEXT_RANK_STABILITY_MIN = 0.65
V24_CHAIN_REPAIR_SUPPORT_MIN = 0.14
V24_CHAIN_REPAIR_SUPPORT_MAX = 0.18
V24_CHAIN_REPAIR_DELTA_MIN = 0.55
V24_CHAIN_REPAIR_RANK_MAX = 0.68
V24_CHAIN_REPAIR_FALLBACK_MARGIN_MIN = 0.75
V26_CHAIN_REPAIR_MAX_SHARED_VAR_COUNT = 5
V3_RELATION_FIRST_SUPPORT_MIN = 0.10
V3_RELATION_FIRST_SUPPORT_MAX = 0.32
V3_RELATION_FIRST_DELTA_GAP_MIN = 0.55
V3_RELATION_FIRST_RANK_STABILITY_MAX = 0.65
V3_RELATION_FIRST_FALLBACK_MARGIN_MIN = 0.75
V31_RELATION_FIRST_LOCK_MIN_COUNT = 2
V31_RELATION_FIRST_LOCK_SUPPORT_MIN = 0.10
V31_RELATION_FIRST_LOCK_SUPPORT_MAX = 0.32
V31_RELATION_FIRST_LOCK_RANK_STABILITY_MIN = 0.85
V31_SEARCH_STATE_LOW_GAIN_COUNT = 3
V31_DENSE_OVERLAP_THRESHOLD = 0.18
V31_DENSE_LOCK_PREFIX_COUNT = 3
V31_DENSE_V24_DELTA_RATIO_MIN = 0.70
V31_DENSE_V24_RANK_STABILITY_MAX = 0.34
V31_DENSE_V24_FALLBACK_MARGIN_MAX = 0.81
# Keep the historical logger name so existing audit filters remain compatible
# while the implementation lives in the new action-policy module.
LOGGER = logging.getLogger("arac.policy.relation_policy")


def decide_action(relation: OverlapRelation) -> ActionDecision:
    """Choose one deterministic action for an overlap relation."""

    return score_relation_actions(relation).final_action


def select_evidence_action_controller_v3_mode(
    relations: list[OverlapRelation],
) -> str:
    """Select a v3 controller mode from runtime overlap-relation evidence only.

    The selector deliberately ignores problem labels, function families, paper
    scores, and historical outcomes. Empty evidence stays relation-first so a
    run cannot trigger search-state rescue before seeing relation evidence.
    """

    if not relations:
        return "relation_first"

    recent_relations = relations[-3:]
    for relation in recent_relations:
        mid_supported = (
            V3_RELATION_FIRST_SUPPORT_MIN
            <= relation.shared_var_support_ratio
            <= V3_RELATION_FIRST_SUPPORT_MAX
        )
        unstable_delta = (
            relation.delta_ratio_gap >= V3_RELATION_FIRST_DELTA_GAP_MIN
        )
        weak_rank_stability = (
            relation.rank_stability <= V3_RELATION_FIRST_RANK_STABILITY_MAX
        )
        enough_margin = (
            relation.fallback_margin_proxy
            >= V3_RELATION_FIRST_FALLBACK_MARGIN_MIN
        )
        if mid_supported and enough_margin and (unstable_delta or weak_rank_stability):
            return "relation_first"

    return "search_state_assisted"


def relation_policy_mode_for_evidence_action_controller_v3(
    relations: list[OverlapRelation],
) -> str:
    if select_evidence_action_controller_v3_mode(relations) == "relation_first":
        return "adaptive_v24"
    return "adaptive_v26"


def select_evidence_action_controller_v31_mode(
    relations: list[OverlapRelation],
) -> str:
    """Guarded v3.1 selector with a relation-first no-harm lock.

    V3.1 is intentionally more conservative than v3. Once runtime evidence
    shows stable positive relation-first behavior, search-state rescue stays
    locked out. It only opens when repeated low-gain evidence appears without
    that lock.
    """

    if not relations:
        return "relation_first"
    if _has_stable_relation_first_lock(relations):
        return "relation_first"
    if _has_repeated_low_gain_without_lock(relations):
        return "search_state_assisted"
    return select_evidence_action_controller_v3_mode(relations)


def relation_policy_mode_for_evidence_action_controller_v31(
    relations: list[OverlapRelation],
) -> str:
    if select_evidence_action_controller_v31_mode(relations) == "relation_first":
        return "adaptive_v24"
    return "adaptive_v26"


def is_evidence_action_controller_v31_dense_overlap(
    degree_of_overlap: float,
) -> bool:
    """Return whether runtime topology requires the dense-overlap route."""

    return float(degree_of_overlap) >= V31_DENSE_OVERLAP_THRESHOLD


def select_evidence_action_controller_v31_dense_lock_mode(
    relations: list[OverlapRelation],
) -> str | None:
    """Select one dense policy before the first evidence-backed intervention."""

    if len(relations) < V31_DENSE_LOCK_PREFIX_COUNT:
        return None
    relation = relations[V31_DENSE_LOCK_PREFIX_COUNT - 1]
    early_chain_instability = (
        relation.shared_var_count > V26_CHAIN_REPAIR_MAX_SHARED_VAR_COUNT
        and relation.both_positive
        and not relation.one_side_zero
        and relation.delta_ratio_gap >= V31_DENSE_V24_DELTA_RATIO_MIN
        and relation.rank_stability <= V31_DENSE_V24_RANK_STABILITY_MAX
        and relation.fallback_margin_proxy <= V31_DENSE_V24_FALLBACK_MARGIN_MAX
    )
    return "adaptive_v24" if early_chain_instability else "adaptive_v26"


def _has_stable_relation_first_lock(relations: list[OverlapRelation]) -> bool:
    stable_count = 0
    for relation in relations[-4:]:
        supported = (
            V31_RELATION_FIRST_LOCK_SUPPORT_MIN
            <= relation.shared_var_support_ratio
            <= V31_RELATION_FIRST_LOCK_SUPPORT_MAX
        )
        stable_positive = (
            relation.both_positive
            and relation.previous_delta > 0.0
            and relation.current_delta > 0.0
            and relation.rank_stability >= V31_RELATION_FIRST_LOCK_RANK_STABILITY_MIN
        )
        if supported and stable_positive:
            stable_count += 1
    return stable_count >= V31_RELATION_FIRST_LOCK_MIN_COUNT


def _has_repeated_low_gain_without_lock(relations: list[OverlapRelation]) -> bool:
    if len(relations) < V31_SEARCH_STATE_LOW_GAIN_COUNT:
        return False
    low_gain_count = 0
    for relation in relations[-V31_SEARCH_STATE_LOW_GAIN_COUNT:]:
        if relation.previous_delta <= 0.0 and relation.current_delta <= 0.0:
            low_gain_count += 1
    return low_gain_count >= V31_SEARCH_STATE_LOW_GAIN_COUNT


def score_relation_actions(relation: OverlapRelation) -> ScoredActionDecision:
    """Score deterministic action candidates and apply a margin abstain rule."""

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
    mid_dense_active_blocked = (
        relation.shared_var_support_ratio >= DENSE_REPAIR_SUPPORT_THRESHOLD
        and relation.shared_var_support_ratio < MID_DENSE_ACTIVE_SUPPORT_MAX
    )
    scores = {action_name: 0.0 for action_name in ACTION_NAMES}
    reasons = {action_name: "" for action_name in ACTION_NAMES}
    fallback_reason = "no_deterministic_relation_rule_triggered"
    fallback_discount = (
        0.15 if stable_delta and rank_stability >= 0.85 else FALLBACK_SCORE_DISCOUNT
    )
    scores["fallback"] = _clamp(relation.fallback_margin_proxy - fallback_discount)

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
            fallback_reason = "active_isolate_conflict_abstained"

        dense_repair_signal = (
            high_overlap
            and relation.both_positive
            and relation.shared_var_support_ratio >= DENSE_REPAIR_SUPPORT_THRESHOLD
            and delta_ratio_gap >= DENSE_REPAIR_DELTA_MIN
            and delta_ratio_gap <= DENSE_REPAIR_DELTA_MAX
            and rank_stability >= DENSE_REPAIR_RANK_STABILITY_MIN
        )
        if dense_repair_signal and mid_dense_active_blocked:
            fallback_reason = "mid_dense_support_blocks_repair"
        elif dense_repair_signal:
            _set_candidate_score(
                scores,
                reasons,
                "reassign_repair",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    relation.fallback_margin_proxy,
                    rank_stability,
                ),
                "dense_two_sided_repair_mode",
            )

        if (
            high_overlap
            and relation.both_positive
            and stable_delta
            and stable_rank
            and mid_dense_active_blocked
        ):
            fallback_reason = "mid_dense_support_blocks_stable_coordinate"
        elif high_overlap and relation.both_positive and stable_delta and stable_rank:
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
            and fallback_reason != "mid_dense_support_blocks_stable_coordinate"
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
            fallback_reason != "mid_dense_support_blocks_repair"
            and (
                relation.one_side_zero
                or delta_ratio_gap > (1.0 - STABILITY_THRESHOLD)
                or not stable_rank
            )
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
            if fallback_reason != "active_isolate_conflict_abstained":
                fallback_reason = "active_reassign_repair_abstained"

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
    decisions = [scored.final_action for scored in score_actions_for_relations(relations)]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.relation_action_name] += 1
    LOGGER.info(
        "relation policy action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def decide_actions_for_relations_v2(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [
        scored.final_action for scored in score_actions_for_relations_v2(relations)
    ]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.relation_action_name] += 1
    LOGGER.info(
        "relation policy v2 action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def decide_actions_for_relations_v21(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [
        scored.final_action for scored in score_actions_for_relations_v21(relations)
    ]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.relation_action_name] += 1
    LOGGER.info(
        "relation policy v2.1 action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def decide_actions_for_relations_v22(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [
        scored.final_action for scored in score_actions_for_relations_v22(relations)
    ]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.relation_action_name] += 1
    LOGGER.info(
        "relation policy v2.2 action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def decide_actions_for_relations_v23(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [
        scored.final_action for scored in score_actions_for_relations_v23(relations)
    ]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.relation_action_name] += 1
    LOGGER.info(
        "relation policy v2.3 action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def decide_actions_for_relations_v24(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [
        scored.final_action for scored in score_actions_for_relations_v24(relations)
    ]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.relation_action_name] += 1
    LOGGER.info(
        "relation policy v2.4 action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def decide_actions_for_relations_v25(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [
        scored.final_action for scored in score_actions_for_relations_v25(relations)
    ]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.relation_action_name] += 1
    LOGGER.info(
        "relation policy v2.5 action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def decide_actions_for_relations_v26(
    relations: list[OverlapRelation],
) -> list[ActionDecision]:
    decisions = [
        scored.final_action for scored in score_actions_for_relations_v26(relations)
    ]
    counts = {action_name: 0 for action_name in ACTION_NAMES}
    for decision in decisions:
        counts[decision.relation_action_name] += 1
    LOGGER.info(
        "relation policy v2.6 action counts: %s",
        ", ".join(f"{action_name}={counts[action_name]}" for action_name in ACTION_NAMES),
    )
    return decisions


def score_actions_for_relations(
    relations: list[OverlapRelation],
) -> list[ScoredActionDecision]:
    scored_actions = [score_relation_actions(relation) for relation in relations]
    balanced_mid_support_seen = False
    prefix_has_one_side_zero = False
    for index, relation in enumerate(relations):
        prefix_has_one_side_zero = prefix_has_one_side_zero or relation.one_side_zero
        balanced_mid_support_seen = balanced_mid_support_seen or (
            _has_active_safety_margin(relation)
            and not prefix_has_one_side_zero
            and relation.both_positive
            and relation.shared_var_support_ratio >= BALANCED_MID_SUPPORT_COORDINATE_MIN
            and relation.shared_var_support_ratio <= BALANCED_MID_SUPPORT_COORDINATE_MAX
            and relation.delta_ratio_gap >= CONFLICT_THRESHOLD
            and relation.rank_stability >= STABILITY_THRESHOLD
        )
        if balanced_mid_support_seen and _has_active_safety_margin(relation):
            scored_actions[index] = _with_coordinate_context_score(
                scored_actions[index],
                relation,
                "balanced_mid_support_coordinate_mode",
            )
    return scored_actions


def score_actions_for_relations_v2(
    relations: list[OverlapRelation],
) -> list[ScoredActionDecision]:
    """Less conservative evidence router for full-budget probes.

    V1 intentionally abstains in mid-dense overlap regimes. The 3M-FE audit
    showed those abstentions dominate S5/E5-like traces, while fixed coordinate
    or repair lanes can be better. V2 only relaxes those abstentions when the
    same runtime relation carries enough support, margin, and stability.
    """

    scored_actions = score_actions_for_relations(relations)
    v2_actions = []
    for relation, scored in zip(relations, scored_actions):
        v2_actions.append(_with_v2_evidence_override(scored, relation))
    return v2_actions


def score_actions_for_relations_v21(
    relations: list[OverlapRelation],
) -> list[ScoredActionDecision]:
    """V2 plus prefix-level coordination consistency.

    Sparse relation-local coordinate decisions can produce an inconsistent
    incumbent when surrounding overlap relations stay on native blending. Once
    repeated runtime conflict-coordinate evidence appears in the same prefix,
    V2.1 keeps subsequent safe two-sided relations in a coordinate context.
    """

    scored_actions = score_actions_for_relations_v2(relations)
    coordinate_evidence_count = 0
    context_active = False
    v21_actions = []
    for relation, scored in zip(relations, scored_actions):
        trigger = scored.final_action.trigger_reason
        if trigger == "adaptive_v2_conflict_coordinate_evidence":
            coordinate_evidence_count += 1
        context_active = (
            context_active
            or coordinate_evidence_count >= V21_COORDINATE_CONTEXT_TRIGGER_COUNT
        )
        if (
            context_active
            and _has_active_safety_margin(relation)
            and relation.both_positive
            and not relation.one_side_zero
            and scored.final_action.relation_action_name == "fallback"
        ):
            scored = _with_forced_action_score(
                scored,
                relation,
                "coordinate",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    relation.fallback_margin_proxy,
                    relation.rank_stability or relation.rank_signal,
                ),
                "adaptive_v21_coordinate_context",
            )
        v21_actions.append(scored)
    return v21_actions


def score_actions_for_relations_v22(
    relations: list[OverlapRelation],
) -> list[ScoredActionDecision]:
    """V2.1 plus first-active-evidence coordinate early lock.

    When the first active relation evidence in an outer-iteration prefix is
    coordinate, later mixed repair/native writes can tear the incumbent state.
    V2.2 locks that prefix into coordinate context earlier. If repair evidence
    appears first, repair remains available so chain-coupled cases can still
    use owner rebinding.
    """

    scored_actions = score_actions_for_relations_v2(relations)
    coordinate_evidence_count = 0
    first_active_action = ""
    context_active = False
    v22_actions = []
    for relation, scored in zip(relations, scored_actions):
        action_name = scored.final_action.relation_action_name
        trigger = scored.final_action.trigger_reason
        if action_name != "fallback" and not first_active_action:
            first_active_action = action_name
        if trigger == "adaptive_v2_conflict_coordinate_evidence":
            coordinate_evidence_count += 1
        context_active = (
            context_active
            or (
                first_active_action == "coordinate"
                and coordinate_evidence_count >= V22_EARLY_LOCK_TRIGGER_COUNT
            )
        )
        if (
            context_active
            and _has_active_safety_margin(relation)
            and relation.both_positive
            and not relation.one_side_zero
            and action_name in {"fallback", "reassign_repair"}
        ):
            scored = _with_forced_action_score(
                scored,
                relation,
                "coordinate",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    relation.fallback_margin_proxy,
                    relation.rank_stability or relation.rank_signal,
                ),
                "adaptive_v22_coordinate_early_lock",
            )
        v22_actions.append(scored)
    return v22_actions


def score_actions_for_relations_v23(
    relations: list[OverlapRelation],
) -> list[ScoredActionDecision]:
    """V2.2 early coordinate context, but preserve repair actions.

    V2.2 showed that overwriting repair evidence can hurt chain-coupled cases.
    V2.3 keeps early coordinate context for fallback/native-blend relations only;
    explicit repair evidence remains repair.
    """

    scored_actions = score_actions_for_relations_v2(relations)
    coordinate_evidence_count = 0
    first_active_action = ""
    context_active = False
    v23_actions = []
    for relation, scored in zip(relations, scored_actions):
        action_name = scored.final_action.relation_action_name
        trigger = scored.final_action.trigger_reason
        if action_name != "fallback" and not first_active_action:
            first_active_action = action_name
        if trigger == "adaptive_v2_conflict_coordinate_evidence":
            coordinate_evidence_count += 1
        context_active = (
            context_active
            or (
                first_active_action == "coordinate"
                and coordinate_evidence_count >= V22_EARLY_LOCK_TRIGGER_COUNT
            )
        )
        if (
            context_active
            and _has_active_safety_margin(relation)
            and relation.both_positive
            and not relation.one_side_zero
            and action_name == "fallback"
        ):
            scored = _with_forced_action_score(
                scored,
                relation,
                "coordinate",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    relation.fallback_margin_proxy,
                    relation.rank_stability or relation.rank_signal,
                ),
                "adaptive_v23_repair_preserving_coordinate_context",
            )
        v23_actions.append(scored)
    return v23_actions


def score_actions_for_relations_v24(
    relations: list[OverlapRelation],
) -> list[ScoredActionDecision]:
    """V2.3 with a runtime-only no-harm gate for unstable coordinate context.

    V2.3 helped some dense overlap traces, but the 3M-FE audit showed that
    forcing coordinate context under low rank stability can tear chain-coupled
    incumbents. V2.4 keeps the repair-preserving idea, adds a chain-instability
    repair route, and only allows early coordinate context when the current
    relation carries enough support and rank stability.
    """

    scored_actions = score_actions_for_relations_v2(relations)
    coordinate_evidence_count = 0
    first_active_action = ""
    context_active = False
    v24_actions = []
    for relation, scored in zip(relations, scored_actions):
        scored = _with_v24_chain_instability_repair(scored, relation)
        action_name = scored.final_action.relation_action_name
        trigger = scored.final_action.trigger_reason
        if action_name != "fallback" and not first_active_action:
            first_active_action = action_name
        if trigger == "adaptive_v2_conflict_coordinate_evidence":
            coordinate_evidence_count += 1
        context_active = (
            context_active
            or (
                first_active_action == "coordinate"
                and coordinate_evidence_count >= V22_EARLY_LOCK_TRIGGER_COUNT
            )
        )
        rank_stability = relation.rank_stability or relation.rank_signal
        if (
            context_active
            and _has_active_safety_margin(relation)
            and relation.both_positive
            and not relation.one_side_zero
            and action_name == "fallback"
            and relation.shared_var_support_ratio >= V24_CONTEXT_SUPPORT_MIN
            and rank_stability >= V24_CONTEXT_RANK_STABILITY_MIN
        ):
            scored = _with_forced_action_score(
                scored,
                relation,
                "coordinate",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    relation.fallback_margin_proxy,
                    rank_stability,
                ),
                "adaptive_v24_stability_gated_coordinate_context",
            )
        v24_actions.append(scored)
    return v24_actions


def score_actions_for_relations_v25(
    relations: list[OverlapRelation],
) -> list[ScoredActionDecision]:
    """V2.3 coordinate context plus narrow chain-instability repair.

    The v2.4 audit showed the stability gate itself removed useful coordinate
    context on several cases. V2.5 therefore keeps v2.3's repair-preserving
    coordinate context and only adds the runtime chain-repair route for the
    narrow mid-support, low-stability regime.
    """

    scored_actions = score_actions_for_relations_v2(relations)
    coordinate_evidence_count = 0
    first_active_action = ""
    context_active = False
    v25_actions = []
    for relation, scored in zip(relations, scored_actions):
        scored = _with_v25_chain_instability_repair(scored, relation)
        action_name = scored.final_action.relation_action_name
        trigger = scored.final_action.trigger_reason
        if action_name != "fallback" and not first_active_action:
            first_active_action = action_name
        if trigger == "adaptive_v2_conflict_coordinate_evidence":
            coordinate_evidence_count += 1
        context_active = (
            context_active
            or (
                first_active_action == "coordinate"
                and coordinate_evidence_count >= V22_EARLY_LOCK_TRIGGER_COUNT
            )
        )
        if (
            context_active
            and _has_active_safety_margin(relation)
            and relation.both_positive
            and not relation.one_side_zero
            and action_name == "fallback"
        ):
            scored = _with_forced_action_score(
                scored,
                relation,
                "coordinate",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    relation.fallback_margin_proxy,
                    relation.rank_stability or relation.rank_signal,
                ),
                "adaptive_v25_chain_repair_preserving_coordinate_context",
            )
        v25_actions.append(scored)
    return v25_actions


def score_actions_for_relations_v26(
    relations: list[OverlapRelation],
) -> list[ScoredActionDecision]:
    """V2.5 with chain repair limited to low-overlap runtime relations."""

    scored_actions = score_actions_for_relations_v2(relations)
    coordinate_evidence_count = 0
    first_active_action = ""
    context_active = False
    v26_actions = []
    for relation, scored in zip(relations, scored_actions):
        scored = _with_v26_low_overlap_chain_instability_repair(scored, relation)
        action_name = scored.final_action.relation_action_name
        trigger = scored.final_action.trigger_reason
        if action_name != "fallback" and not first_active_action:
            first_active_action = action_name
        if trigger == "adaptive_v2_conflict_coordinate_evidence":
            coordinate_evidence_count += 1
        context_active = (
            context_active
            or (
                first_active_action == "coordinate"
                and coordinate_evidence_count >= V22_EARLY_LOCK_TRIGGER_COUNT
            )
        )
        if (
            context_active
            and _has_active_safety_margin(relation)
            and relation.both_positive
            and not relation.one_side_zero
            and action_name == "fallback"
        ):
            scored = _with_forced_action_score(
                scored,
                relation,
                "coordinate",
                _mean(
                    _overlap_confidence(relation.overlap_strength),
                    relation.fallback_margin_proxy,
                    relation.rank_stability or relation.rank_signal,
                ),
                "adaptive_v26_low_overlap_chain_repair_preserving_coordinate_context",
            )
        v26_actions.append(scored)
    return v26_actions


def _with_v2_evidence_override(
    scored: ScoredActionDecision,
    relation: OverlapRelation,
) -> ScoredActionDecision:
    if not _has_active_safety_margin(relation):
        return scored
    if scored.final_action.relation_action_name != "fallback":
        return scored
    if relation.one_side_zero or not relation.both_positive:
        return scored

    support = relation.shared_var_support_ratio
    delta_ratio = relation.delta_ratio_gap
    rank_stability = relation.rank_stability or relation.rank_signal

    if (
        support >= V2_REPAIR_SUPPORT_MIN
        and support <= V2_REPAIR_SUPPORT_MAX
        and delta_ratio >= DENSE_REPAIR_DELTA_MIN
        and rank_stability >= V2_REPAIR_STABILITY_MIN
        and relation.fallback_margin_proxy >= V2_REPAIR_FALLBACK_MARGIN_MIN
    ):
        return _with_forced_action_score(
            scored,
            relation,
            "reassign_repair",
            _mean(
                _overlap_confidence(relation.overlap_strength),
                relation.fallback_margin_proxy,
                rank_stability,
            ),
            "adaptive_v2_mid_dense_repair_evidence",
        )

    if (
        support >= V2_COORDINATE_SUPPORT_MIN
        and support <= V2_COORDINATE_SUPPORT_MAX
        and delta_ratio >= V2_HIGH_CONFLICT_MIN
        and relation.fallback_margin_proxy >= MIN_FALLBACK_MARGIN_PROXY
    ):
        return _with_forced_action_score(
            scored,
            relation,
            "coordinate",
            _mean(
                _overlap_confidence(relation.overlap_strength),
                relation.fallback_margin_proxy,
                1.0 - min(1.0, support),
            ),
            "adaptive_v2_conflict_coordinate_evidence",
        )

    return scored


def _with_v24_chain_instability_repair(
    scored: ScoredActionDecision,
    relation: OverlapRelation,
) -> ScoredActionDecision:
    if not _has_active_safety_margin(relation):
        return scored
    if relation.one_side_zero or not relation.both_positive:
        return scored

    support = relation.shared_var_support_ratio
    rank_stability = relation.rank_stability or relation.rank_signal
    if (
        support >= V24_CHAIN_REPAIR_SUPPORT_MIN
        and support <= V24_CHAIN_REPAIR_SUPPORT_MAX
        and relation.delta_ratio_gap >= V24_CHAIN_REPAIR_DELTA_MIN
        and rank_stability <= V24_CHAIN_REPAIR_RANK_MAX
        and relation.fallback_margin_proxy >= V24_CHAIN_REPAIR_FALLBACK_MARGIN_MIN
    ):
        return _with_forced_action_score(
            scored,
            relation,
            "reassign_repair",
            _mean(
                _overlap_confidence(relation.overlap_strength),
                relation.fallback_margin_proxy,
                1.0 - min(1.0, relation.delta_ratio_gap),
            ),
            "adaptive_v24_chain_instability_repair",
        )

    return scored


def _with_v25_chain_instability_repair(
    scored: ScoredActionDecision,
    relation: OverlapRelation,
) -> ScoredActionDecision:
    repaired = _with_v24_chain_instability_repair(scored, relation)
    if repaired is scored:
        return scored
    return _replace_trigger_reason(
        repaired,
        relation,
        "adaptive_v25_chain_instability_repair",
    )


def _with_v26_low_overlap_chain_instability_repair(
    scored: ScoredActionDecision,
    relation: OverlapRelation,
) -> ScoredActionDecision:
    if relation.shared_var_count > V26_CHAIN_REPAIR_MAX_SHARED_VAR_COUNT:
        return scored
    repaired = _with_v24_chain_instability_repair(scored, relation)
    if repaired is scored:
        return scored
    return _replace_trigger_reason(
        repaired,
        relation,
        "adaptive_v26_low_overlap_chain_instability_repair",
    )


def _with_coordinate_context_score(
    scored: ScoredActionDecision,
    relation: OverlapRelation,
    trigger_reason: str,
) -> ScoredActionDecision:
    scores = dict(scored.candidate_scores)
    other_best = max(score for action, score in scores.items() if action != "coordinate")
    coordinate_score = max(
        scores["coordinate"],
        relation.fallback_margin_proxy,
        min(1.0, other_best + ACTION_MARGIN_THRESHOLD),
    )
    if coordinate_score >= 1.0:
        for action_name in ACTION_NAMES:
            if action_name != "coordinate":
                scores[action_name] = min(scores[action_name], 1.0 - ACTION_MARGIN_THRESHOLD)
    scores["coordinate"] = _clamp(coordinate_score)
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], _action_sort_order(item[0])),
    )
    best_action_name, best_score = ranked[0]
    second_best_action_name, second_best_score = ranked[1]
    final_action = _decision(
        relation,
        "coordinate",
        "coordinate",
        best_score,
        trigger_reason,
    )
    return ScoredActionDecision(
        relation_id=relation.relation_id,
        candidate_scores={name: _clamp(scores[name]) for name in ACTION_NAMES},
        final_action=final_action,
        best_action_name=best_action_name,
        best_score=_clamp(best_score),
        second_best_action_name=second_best_action_name,
        second_best_score=_clamp(second_best_score),
        margin=_clamp(best_score - second_best_score),
        abstain_reason="",
    )


def _with_forced_action_score(
    scored: ScoredActionDecision,
    relation: OverlapRelation,
    action_name: str,
    score: float,
    trigger_reason: str,
) -> ScoredActionDecision:
    scores = dict(scored.candidate_scores)
    other_best = max(score for name, score in scores.items() if name != action_name)
    forced_score = max(_clamp(score), min(1.0, other_best + ACTION_MARGIN_THRESHOLD))
    if forced_score >= 1.0:
        for candidate_name in ACTION_NAMES:
            if candidate_name != action_name:
                scores[candidate_name] = min(
                    scores[candidate_name],
                    1.0 - ACTION_MARGIN_THRESHOLD,
                )
    scores[action_name] = _clamp(forced_score)
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], _action_sort_order(item[0])),
    )
    best_action_name, best_score = ranked[0]
    second_best_action_name, second_best_score = ranked[1]
    final_action = _decision(
        relation,
        action_name,
        _action_family(action_name),
        best_score,
        trigger_reason,
    )
    return ScoredActionDecision(
        relation_id=relation.relation_id,
        candidate_scores={name: _clamp(scores[name]) for name in ACTION_NAMES},
        final_action=final_action,
        best_action_name=best_action_name,
        best_score=_clamp(best_score),
        second_best_action_name=second_best_action_name,
        second_best_score=_clamp(second_best_score),
        margin=_clamp(best_score - second_best_score),
        abstain_reason="",
    )


def _overlap_confidence(overlap_strength: float) -> float:
    return _clamp(overlap_strength / max(HIGH_OVERLAP_THRESHOLD, 1e-12))


def _has_active_safety_margin(relation: OverlapRelation) -> bool:
    return (
        relation.overlap_strength >= HIGH_OVERLAP_THRESHOLD
        and relation.shared_var_count > 0
        and relation.feature_coverage >= MIN_FEATURE_COVERAGE
        and relation.budget_remaining_ratio >= MIN_BUDGET_REMAINING_RATIO
        and relation.fallback_margin_proxy >= MIN_FALLBACK_MARGIN_PROXY
    )


def _replace_trigger_reason(
    scored: ScoredActionDecision,
    relation: OverlapRelation,
    trigger_reason: str,
) -> ScoredActionDecision:
    action = scored.final_action
    final_action = _decision(
        relation,
        action.relation_action_name,
        action.action_family,
        action.confidence,
        trigger_reason,
    )
    return ScoredActionDecision(
        relation_id=scored.relation_id,
        candidate_scores=dict(scored.candidate_scores),
        final_action=final_action,
        best_action_name=scored.best_action_name,
        best_score=scored.best_score,
        second_best_action_name=scored.second_best_action_name,
        second_best_score=scored.second_best_score,
        margin=scored.margin,
        abstain_reason=scored.abstain_reason,
    )


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

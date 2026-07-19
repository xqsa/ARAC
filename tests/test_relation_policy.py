from __future__ import annotations

from arac.evidence.overlap_relation_builder import OverlapRelation
from arac.policy.relation_policy import (
    action_mismatch_audit_row,
    decide_action,
    decide_actions_for_relations,
    decide_actions_for_relations_v2,
    decide_actions_for_relations_v24,
    decide_actions_for_relations_v26,
    is_evidence_action_controller_v31_dense_overlap,
    relation_policy_mode_for_evidence_action_controller_v31,
    score_actions_for_relations,
    score_actions_for_relations_v2,
    score_actions_for_relations_v24,
    score_actions_for_relations_v26,
    score_relation_actions,
    select_evidence_action_controller_v31_dense_lock_mode,
    select_evidence_action_controller_v31_mode,
    soft_score_actions,
)


def make_relation(**overrides: object) -> OverlapRelation:
    values = {
        "relation_id": "O1_0_1",
        "problem_id": "E2",
        "outer_iter": 1,
        "group_left": 0,
        "group_right": 1,
        "shared_vars": (2,),
        "overlap_strength": 1.0,
        "delta_signal": 0.1,
        "rank_signal": 0.9,
        "budget_remaining_ratio": 0.8,
        "previous_delta": 1.0,
        "current_delta": 1.1,
        "delta_abs_gap": 0.1,
        "delta_signed_gap": 0.1,
        "delta_ratio_gap": 0.0909090909,
        "both_positive": True,
        "one_side_zero": False,
        "rank_gap": 0.0,
        "rank_stability": 0.9,
        "shared_var_count": 1,
        "shared_var_support_ratio": 1.0,
        "feature_coverage": 1.0,
        "fallback_margin_proxy": 0.9,
    }
    values.update(overrides)
    return OverlapRelation(**values)


def test_evidence_action_controller_v31_locks_relation_first_for_stable_positive_prefix() -> None:
    relations = [
        make_relation(
            relation_id="O0_0_1",
            shared_vars=(1, 2, 3, 4),
            shared_var_count=4,
            shared_var_support_ratio=0.16,
            previous_delta=10.0,
            current_delta=9.0,
            delta_ratio_gap=0.10,
            both_positive=True,
            rank_stability=0.92,
            fallback_margin_proxy=0.90,
        ),
        make_relation(
            relation_id="O0_1_2",
            shared_vars=(5, 6, 7, 8),
            shared_var_count=4,
            shared_var_support_ratio=0.16,
            previous_delta=9.0,
            current_delta=8.5,
            delta_ratio_gap=0.06,
            both_positive=True,
            rank_stability=0.90,
            fallback_margin_proxy=0.91,
        ),
    ]

    mode = select_evidence_action_controller_v31_mode(relations)

    assert mode == "relation_first"
    assert relation_policy_mode_for_evidence_action_controller_v31(relations) == "adaptive_v24"


def test_evidence_action_controller_v31_allows_search_state_only_for_repeated_low_gain_without_relation_lock() -> None:
    relations = [
        make_relation(
            relation_id=f"O0_{index}_{index + 1}",
            shared_vars=(index,),
            shared_var_count=1,
            shared_var_support_ratio=0.04,
            previous_delta=0.0,
            current_delta=0.0,
            delta_ratio_gap=0.0,
            both_positive=False,
            one_side_zero=False,
            rank_stability=0.20,
            fallback_margin_proxy=0.90,
        )
        for index in range(3)
    ]

    mode = select_evidence_action_controller_v31_mode(relations)

    assert mode == "search_state_assisted"
    assert relation_policy_mode_for_evidence_action_controller_v31(relations) == "adaptive_v26"


def test_evidence_action_controller_v31_dense_overlap_threshold_is_inclusive() -> None:
    assert is_evidence_action_controller_v31_dense_overlap(0.18) is True
    assert is_evidence_action_controller_v31_dense_overlap(0.179999) is False


def test_evidence_action_controller_v31_dense_lock_waits_for_three_relations() -> None:
    relations = [
        make_relation(
            relation_id=f"O0_{index}_{index + 1}",
            shared_var_count=10,
        )
        for index in range(2)
    ]

    mode = select_evidence_action_controller_v31_dense_lock_mode(relations)

    assert mode is None


def test_evidence_action_controller_v31_dense_lock_selects_v24_for_early_chain_instability() -> None:
    relations = [
        make_relation(
            relation_id=f"O0_{index}_{index + 1}",
            shared_var_count=10,
            both_positive=True,
            delta_ratio_gap=0.10 if index < 2 else 0.75,
            rank_stability=0.0 if index < 2 else 0.33,
            fallback_margin_proxy=0.90 if index < 2 else 0.79,
        )
        for index in range(3)
    ]

    mode = select_evidence_action_controller_v31_dense_lock_mode(relations)

    assert mode == "adaptive_v24"


def test_evidence_action_controller_v31_dense_lock_selects_v26_without_early_chain_instability() -> None:
    relations = [
        make_relation(
            relation_id=f"O0_{index}_{index + 1}",
            shared_var_count=10,
            both_positive=True,
            delta_ratio_gap=0.10 if index < 2 else 0.58,
            rank_stability=0.0 if index < 2 else 0.67,
            fallback_margin_proxy=0.90 if index < 2 else 0.83,
        )
        for index in range(3)
    ]

    mode = select_evidence_action_controller_v31_dense_lock_mode(relations)

    assert mode == "adaptive_v26"


def test_relation_policy_coordinates_stable_high_overlap_relation() -> None:
    decision = decide_action(make_relation())

    assert decision.relation_id == "O1_0_1"
    assert decision.action_name == "coordinate"
    assert decision.relation_action_name == "coordinate"
    assert decision.canonical_action_name == "allow_beneficial_coordination"
    assert decision.action_family == "coordinate"
    assert decision.confidence > 0.0
    assert decision.trigger_reason == "high_overlap_with_stable_delta_and_rank"


def test_relation_policy_abstains_on_mid_dense_stable_coordinate_signal() -> None:
    decision = decide_action(
        make_relation(
            shared_var_support_ratio=0.21875,
            delta_ratio_gap=0.113419,
            rank_signal=0.916667,
            rank_stability=0.916667,
            fallback_margin_proxy=0.975190,
        )
    )

    assert decision.relation_action_name == "fallback"
    assert decision.trigger_reason == "mid_dense_support_blocks_stable_coordinate"


def test_relation_policy_v2_repairs_mid_dense_stable_evidence() -> None:
    relation = make_relation(
        shared_var_support_ratio=0.21875,
        delta_ratio_gap=0.404862,
        delta_signed_gap=49_847.312436,
        rank_signal=0.888889,
        rank_stability=0.888889,
        fallback_margin_proxy=0.911436,
    )

    v1_decision = decide_action(relation)
    v2_decision = decide_actions_for_relations_v2([relation])[0]

    assert v1_decision.relation_action_name == "fallback"
    assert v2_decision.relation_action_name == "reassign_repair"
    assert v2_decision.canonical_action_name == "repair_shared_variable_binding"
    assert v2_decision.trigger_reason == "adaptive_v2_mid_dense_repair_evidence"


def test_relation_policy_v2_coordinates_supported_conflict_evidence() -> None:
    relation = make_relation(
        previous_delta=10.0,
        current_delta=4.0,
        delta_signal=6.0,
        delta_abs_gap=6.0,
        delta_signed_gap=-6.0,
        delta_ratio_gap=0.60,
        both_positive=True,
        one_side_zero=False,
        rank_signal=0.60,
        rank_stability=0.60,
        shared_var_support_ratio=0.16,
        fallback_margin_proxy=0.90,
    )

    v1_decision = decide_action(relation)
    v2_decision = decide_actions_for_relations_v2([relation])[0]

    assert v1_decision.relation_action_name == "fallback"
    assert v2_decision.relation_action_name == "coordinate"
    assert v2_decision.canonical_action_name == "allow_beneficial_coordination"
    assert v2_decision.trigger_reason == "adaptive_v2_conflict_coordinate_evidence"


def test_relation_policy_v2_keeps_one_side_zero_fallback() -> None:
    relation = make_relation(
        previous_delta=10.0,
        current_delta=0.0,
        delta_signal=10.0,
        delta_abs_gap=10.0,
        delta_signed_gap=-10.0,
        delta_ratio_gap=1.0,
        both_positive=False,
        one_side_zero=True,
        rank_signal=0.90,
        rank_stability=0.90,
        shared_var_support_ratio=0.20,
        fallback_margin_proxy=0.95,
    )

    v2_decision = decide_actions_for_relations_v2([relation])[0]

    assert v2_decision.relation_action_name == "fallback"
    assert v2_decision.canonical_action_name == "conservative_no_action"


def test_relation_policy_v2_scored_action_reports_override_margin() -> None:
    relation = make_relation(
        shared_var_support_ratio=0.21875,
        delta_ratio_gap=0.404862,
        delta_signed_gap=49_847.312436,
        rank_signal=0.888889,
        rank_stability=0.888889,
        fallback_margin_proxy=0.911436,
    )

    scored = score_actions_for_relations_v2([relation])[0]

    assert scored.final_action.relation_action_name == "reassign_repair"
    assert scored.best_action_name == "reassign_repair"
    assert scored.margin >= 0.05


def test_relation_policy_v24_repairs_chain_instability_signal() -> None:
    relation = make_relation(
        previous_delta=10.0,
        current_delta=4.0,
        delta_signal=6.0,
        delta_abs_gap=6.0,
        delta_signed_gap=-6.0,
        delta_ratio_gap=0.60,
        both_positive=True,
        one_side_zero=False,
        rank_signal=0.62,
        rank_stability=0.62,
        shared_var_support_ratio=0.16,
        fallback_margin_proxy=0.90,
    )

    v2_decision = decide_actions_for_relations_v2([relation])[0]
    v24_decision = decide_actions_for_relations_v24([relation])[0]

    assert v2_decision.relation_action_name == "coordinate"
    assert v24_decision.relation_action_name == "reassign_repair"
    assert v24_decision.canonical_action_name == "repair_shared_variable_binding"
    assert v24_decision.trigger_reason == "adaptive_v24_chain_instability_repair"


def test_relation_policy_v24_gates_low_stability_coordinate_context() -> None:
    relations = [
        make_relation(
            relation_id="O1_0_1",
            previous_delta=10.0,
            current_delta=4.0,
            delta_signal=6.0,
            delta_abs_gap=6.0,
            delta_signed_gap=-6.0,
            delta_ratio_gap=0.60,
            both_positive=True,
            one_side_zero=False,
            rank_signal=0.72,
            rank_stability=0.72,
            shared_var_support_ratio=0.16,
            fallback_margin_proxy=0.90,
        ),
        make_relation(
            relation_id="O1_1_2",
            previous_delta=3.0,
            current_delta=2.8,
            delta_signal=0.2,
            delta_abs_gap=0.2,
            delta_signed_gap=-0.2,
            delta_ratio_gap=0.066667,
            both_positive=True,
            one_side_zero=False,
            rank_signal=0.58,
            rank_stability=0.58,
            shared_var_support_ratio=0.16,
            fallback_margin_proxy=0.85,
        ),
    ]

    v24_actions = decide_actions_for_relations_v24(relations)

    assert [action.relation_action_name for action in v24_actions] == [
        "coordinate",
        "fallback",
    ]

    stable_context = [
        relations[0],
        make_relation(
            relation_id="O1_1_2",
            previous_delta=3.0,
            current_delta=2.4,
            delta_signal=0.6,
            delta_abs_gap=0.6,
            delta_signed_gap=-0.6,
            delta_ratio_gap=0.20,
            both_positive=True,
            one_side_zero=False,
            rank_signal=0.70,
            rank_stability=0.70,
            shared_var_support_ratio=0.16,
            fallback_margin_proxy=0.85,
        ),
    ]
    stable_v24_actions = decide_actions_for_relations_v24(stable_context)

    assert stable_v24_actions[-1].relation_action_name == "coordinate"
    assert (
        stable_v24_actions[-1].trigger_reason
        == "adaptive_v24_stability_gated_coordinate_context"
    )


def test_relation_policy_v24_scored_action_reports_repair_override() -> None:
    relation = make_relation(
        previous_delta=10.0,
        current_delta=4.0,
        delta_signal=6.0,
        delta_abs_gap=6.0,
        delta_signed_gap=-6.0,
        delta_ratio_gap=0.60,
        both_positive=True,
        one_side_zero=False,
        rank_signal=0.62,
        rank_stability=0.62,
        shared_var_support_ratio=0.16,
        fallback_margin_proxy=0.90,
    )

    scored = score_actions_for_relations_v24([relation])[0]

    assert scored.final_action.relation_action_name == "reassign_repair"
    assert scored.best_action_name == "reassign_repair"
    assert scored.margin >= 0.05


def test_relation_policy_v26_limits_chain_repair_to_low_overlap_relations() -> None:
    low_overlap_relation = make_relation(
        previous_delta=10.0,
        current_delta=4.0,
        delta_signal=6.0,
        delta_abs_gap=6.0,
        delta_signed_gap=-6.0,
        delta_ratio_gap=0.60,
        both_positive=True,
        one_side_zero=False,
        rank_signal=0.62,
        rank_stability=0.62,
        shared_vars=(1, 2, 3, 4, 5),
        shared_var_count=5,
        overlap_strength=5.0,
        shared_var_support_ratio=1.0 / 6.0,
        fallback_margin_proxy=0.90,
    )
    higher_overlap_relation = make_relation(
        previous_delta=10.0,
        current_delta=4.0,
        delta_signal=6.0,
        delta_abs_gap=6.0,
        delta_signed_gap=-6.0,
        delta_ratio_gap=0.60,
        both_positive=True,
        one_side_zero=False,
        rank_signal=0.62,
        rank_stability=0.62,
        shared_vars=(1, 2, 3, 4, 5, 6, 7),
        shared_var_count=7,
        overlap_strength=7.0,
        shared_var_support_ratio=0.14,
        fallback_margin_proxy=0.90,
    )

    low_overlap_action = decide_actions_for_relations_v26([low_overlap_relation])[0]
    higher_overlap_v26 = decide_actions_for_relations_v26(
        [higher_overlap_relation]
    )[0]

    assert low_overlap_action.relation_action_name == "reassign_repair"
    assert (
        low_overlap_action.trigger_reason
        == "adaptive_v26_low_overlap_chain_instability_repair"
    )
    assert higher_overlap_v26.relation_action_name == "coordinate"
    assert (
        higher_overlap_v26.trigger_reason
        == "adaptive_v2_conflict_coordinate_evidence"
    )


def test_relation_policy_v26_scored_action_reports_low_overlap_repair() -> None:
    relation = make_relation(
        previous_delta=10.0,
        current_delta=4.0,
        delta_signal=6.0,
        delta_abs_gap=6.0,
        delta_signed_gap=-6.0,
        delta_ratio_gap=0.60,
        both_positive=True,
        one_side_zero=False,
        rank_signal=0.62,
        rank_stability=0.62,
        shared_vars=(1, 2, 3, 4, 5),
        shared_var_count=5,
        overlap_strength=5.0,
        shared_var_support_ratio=1.0 / 6.0,
        fallback_margin_proxy=0.90,
    )

    scored = score_actions_for_relations_v26([relation])[0]

    assert scored.final_action.relation_action_name == "reassign_repair"
    assert (
        scored.final_action.trigger_reason
        == "adaptive_v26_low_overlap_chain_instability_repair"
    )
    assert scored.best_action_name == "reassign_repair"


def test_relation_policy_scores_candidates_and_reports_margin() -> None:
    relation = make_relation()
    scored = score_relation_actions(relation)

    assert scored.final_action.relation_action_name == "coordinate"
    assert scored.second_best_action_name == "fallback"
    assert scored.margin > 0.0
    assert scored.abstain_reason == ""
    assert scored.candidate_scores["coordinate"] > scored.candidate_scores["fallback"]

    row = action_mismatch_audit_row(relation, scored)

    assert row["relation_id"] == relation.relation_id
    assert row["final_action_name"] == "coordinate"
    assert row["second_best_action_name"] == "fallback"
    assert float(row["margin"]) > 0.0
    assert row["abstain_reason"] == ""
    assert "coordinate=" in row["candidate_scores"]


def test_relation_policy_safety_gate_falls_back_on_low_feature_coverage() -> None:
    decision = decide_action(
        make_relation(
            feature_coverage=0.75,
            previous_delta=100.0,
            current_delta=0.0,
            delta_signal=100.0,
            delta_abs_gap=100.0,
            delta_signed_gap=-100.0,
            delta_ratio_gap=1.0,
            both_positive=False,
            one_side_zero=True,
            fallback_margin_proxy=0.9,
        )
    )

    assert decision.action_name == "fallback"
    assert decision.canonical_action_name == "conservative_no_action"
    assert decision.trigger_reason == "insufficient_relation_policy_safety_margin"


def test_decide_actions_for_relations_preserves_order_and_logs_counts(caplog) -> None:
    relations = [
        make_relation(relation_id="O1_0_1"),
        make_relation(
            relation_id="O1_1_2",
            delta_signal=2.0,
            previous_delta=2.0,
            current_delta=0.0,
            delta_abs_gap=2.0,
            delta_signed_gap=-2.0,
            delta_ratio_gap=1.0,
            shared_var_support_ratio=0.10,
        ),
        make_relation(
            relation_id="O1_2_3",
            delta_signal=0.6,
            previous_delta=0.0,
            current_delta=0.6,
            delta_abs_gap=0.6,
            delta_signed_gap=0.6,
            delta_ratio_gap=1.0,
            one_side_zero=True,
            both_positive=False,
            rank_signal=0.4,
            rank_stability=0.4,
            shared_var_support_ratio=0.10,
        ),
        make_relation(
            relation_id="O1_3_4",
            overlap_strength=0.0,
            shared_vars=(),
            delta_signal=0.2,
        ),
    ]

    with caplog.at_level("INFO", logger="arac.policy.relation_policy"):
        decisions = decide_actions_for_relations(relations)

    assert [decision.relation_id for decision in decisions] == [
        "O1_0_1",
        "O1_1_2",
        "O1_2_3",
        "O1_3_4",
    ]
    assert [decision.action_name for decision in decisions] == [
        "coordinate",
        "fallback",
        "fallback",
        "fallback",
    ]
    assert [decision.canonical_action_name for decision in decisions] == [
        "allow_beneficial_coordination",
        "conservative_no_action",
        "conservative_no_action",
        "conservative_no_action",
    ]
    assert (
        "relation policy action counts: "
        "coordinate=1, isolate_conflicting_relation=0, reassign_repair=0, fallback=3"
    ) in caplog.text


def test_score_actions_for_relations_reflects_balanced_batch_coordinate_mode() -> None:
    relations = [
        make_relation(
            relation_id="O0_0_1",
            shared_var_support_ratio=0.10,
            delta_ratio_gap=0.3,
            rank_stability=0.0,
            fallback_margin_proxy=0.9,
        ),
        make_relation(
            relation_id="O0_1_2",
            shared_var_support_ratio=0.166667,
            delta_ratio_gap=0.80,
            rank_stability=0.75,
            fallback_margin_proxy=0.86,
        ),
    ]

    scored = score_actions_for_relations(relations)
    row = action_mismatch_audit_row(relations[1], scored[1])

    assert row["final_action_name"] == "coordinate"
    assert row["best_action_name"] == "coordinate"
    assert row["trigger_reason"] == "balanced_mid_support_coordinate_mode"


def test_balanced_batch_coordinate_mode_respects_budget_gate() -> None:
    relations = [
        make_relation(
            relation_id="O0_0_1",
            shared_var_support_ratio=0.166667,
            delta_ratio_gap=0.80,
            rank_stability=0.75,
            fallback_margin_proxy=0.86,
            budget_remaining_ratio=0.8,
        ),
        make_relation(
            relation_id="O0_1_2",
            shared_var_support_ratio=0.166667,
            delta_ratio_gap=0.50,
            rank_stability=0.60,
            fallback_margin_proxy=0.90,
            budget_remaining_ratio=0.01,
        ),
    ]

    decisions = decide_actions_for_relations(relations)

    assert decisions[0].trigger_reason == "balanced_mid_support_coordinate_mode"
    assert decisions[1].relation_action_name == "fallback"
    assert decisions[1].trigger_reason == "insufficient_relation_policy_safety_margin"


def test_soft_policy_changes_continuously_around_conflict_threshold() -> None:
    below = soft_score_actions(
        make_relation(
            shared_var_support_ratio=0.10,
            delta_ratio_gap=0.749,
            rank_stability=0.80,
            fallback_margin_proxy=0.50,
            cohen_d=1.0,
        )
    )
    above = soft_score_actions(
        make_relation(
            shared_var_support_ratio=0.10,
            delta_ratio_gap=0.751,
            rank_stability=0.80,
            fallback_margin_proxy=0.50,
            cohen_d=1.0,
        )
    )

    assert abs(
        below.candidate_scores["reassign_repair"]
        - above.candidate_scores["reassign_repair"]
    ) < 0.01


def test_soft_policy_uses_population_trajectory_and_synergy_evidence() -> None:
    stable = make_relation(
        shared_var_support_ratio=0.10,
        delta_ratio_gap=0.10,
        rank_stability=0.90,
        fallback_margin_proxy=0.50,
        cohen_d=2.0,
        delta_momentum=1.0,
        probe_synergy=0.8,
    )
    oscillating = make_relation(
        shared_var_support_ratio=0.10,
        delta_ratio_gap=0.10,
        rank_stability=0.90,
        fallback_margin_proxy=0.50,
        cohen_d=2.0,
        delta_momentum=-1.0,
        probe_synergy=-0.8,
    )

    stable_score = soft_score_actions(stable)
    oscillating_score = soft_score_actions(oscillating)

    assert stable_score.candidate_scores["coordinate"] > (
        oscillating_score.candidate_scores["coordinate"]
    )

    no_separation = soft_score_actions(
        make_relation(
            shared_var_support_ratio=0.10,
            delta_ratio_gap=0.90,
            rank_stability=0.40,
            fallback_margin_proxy=0.50,
            cohen_d=0.0,
        )
    )
    separated = soft_score_actions(
        make_relation(
            shared_var_support_ratio=0.10,
            delta_ratio_gap=0.90,
            rank_stability=0.40,
            fallback_margin_proxy=0.50,
            cohen_d=2.0,
            population_spread_asymmetry=1.0,
        )
    )

    assert no_separation.candidate_scores["reassign_repair"] == 0.0
    assert separated.candidate_scores["reassign_repair"] > 0.0


def test_soft_policy_safety_gate_is_fail_closed() -> None:
    scored = soft_score_actions(make_relation(feature_coverage=0.5))

    assert scored.final_action.relation_action_name == "fallback"
    assert scored.abstain_reason == "insufficient_relation_policy_safety_margin"

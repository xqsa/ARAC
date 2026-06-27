from __future__ import annotations

from arac.evidence.overlap_relation_builder import OverlapRelation
from arac.policy.relation_policy import decide_action, decide_actions_for_relations


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


def test_relation_policy_coordinates_stable_high_overlap_relation() -> None:
    decision = decide_action(make_relation())

    assert decision.relation_id == "O1_0_1"
    assert decision.action_name == "coordinate"
    assert decision.relation_action_name == "coordinate"
    assert decision.canonical_action_name == "allow_beneficial_coordination"
    assert decision.action_family == "coordinate"
    assert decision.confidence > 0.0
    assert decision.trigger_reason == "high_overlap_with_stable_delta_and_rank"


def test_relation_policy_uses_normalized_conflict_not_raw_aob_scale() -> None:
    decision = decide_action(
        make_relation(
            previous_delta=10_000.0,
            current_delta=10_100.0,
            delta_signal=100.0,
            delta_abs_gap=100.0,
            delta_signed_gap=100.0,
            delta_ratio_gap=0.0099009901,
            both_positive=True,
            rank_signal=0.95,
            rank_stability=0.95,
            fallback_margin_proxy=0.99,
        )
    )

    assert decision.action_name == "coordinate"
    assert decision.canonical_action_name == "allow_beneficial_coordination"


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


def test_relation_policy_isolates_large_delta_conflict() -> None:
    decision = decide_action(
        make_relation(
            delta_signal=2.5,
            previous_delta=3.0,
            current_delta=0.5,
            delta_abs_gap=2.5,
            delta_signed_gap=-2.5,
            delta_ratio_gap=0.8333333333,
            rank_signal=0.9,
        )
    )

    assert decision.action_name == "isolate_conflicting_relation"
    assert decision.canonical_action_name == "isolate_conflicting_relation"
    assert decision.action_family == "isolate"
    assert decision.trigger_reason == "large_delta_conflict_or_negative_divergence"


def test_relation_policy_coordinates_high_margin_positive_conflict() -> None:
    decision = decide_action(
        make_relation(
            previous_delta=13_516_227_713.169922,
            current_delta=1_157_557_952.257812,
            delta_signal=12_358_669_760.91211,
            delta_abs_gap=12_358_669_760.91211,
            delta_signed_gap=-12_358_669_760.91211,
            delta_ratio_gap=0.914358,
            both_positive=True,
            one_side_zero=False,
            rank_signal=0.285714,
            rank_stability=0.285714,
            shared_var_support_ratio=0.02,
            fallback_margin_proxy=0.98,
        )
    )

    assert decision.action_name == "coordinate"
    assert decision.canonical_action_name == "allow_beneficial_coordination"
    assert decision.trigger_reason == "high_fallback_margin_supports_safe_coordination"


def test_relation_policy_isolates_negative_divergence() -> None:
    decision = decide_action(
        make_relation(
            delta_signal=1.0,
            previous_delta=1.0,
            current_delta=0.0,
            delta_abs_gap=1.0,
            delta_signed_gap=-1.0,
            delta_ratio_gap=1.0,
            both_positive=False,
            one_side_zero=True,
            rank_signal=0.9,
        )
    )

    assert decision.action_name == "isolate_conflicting_relation"
    assert decision.action_family == "isolate"


def test_relation_policy_keeps_mixed_relation_actions_for_mixed_evidence() -> None:
    relations = [
        make_relation(
            relation_id="O1_0_1",
            previous_delta=10_000.0,
            current_delta=10_100.0,
            delta_signal=100.0,
            delta_abs_gap=100.0,
            delta_signed_gap=100.0,
            delta_ratio_gap=0.0099009901,
            both_positive=True,
            rank_signal=0.95,
            rank_stability=0.95,
        ),
        make_relation(
            relation_id="O1_1_2",
            previous_delta=0.0,
            current_delta=10_000.0,
            delta_signal=10_000.0,
            delta_abs_gap=10_000.0,
            delta_signed_gap=10_000.0,
            delta_ratio_gap=1.0,
            both_positive=False,
            one_side_zero=True,
            rank_signal=0.4,
            rank_stability=0.4,
        ),
        make_relation(
            relation_id="O1_2_3",
            shared_vars=(),
            overlap_strength=0.0,
            shared_var_count=0,
            shared_var_support_ratio=0.0,
        ),
    ]

    decisions = decide_actions_for_relations(relations)

    assert [decision.action_name for decision in decisions] == [
        "coordinate",
        "reassign_repair",
        "fallback",
    ]


def test_relation_policy_repairs_imbalanced_overlap() -> None:
    decision = decide_action(
        make_relation(
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
        )
    )

    assert decision.action_name == "reassign_repair"
    assert decision.canonical_action_name == "repair_shared_variable_binding"
    assert decision.action_family == "reassign_repair"
    assert decision.trigger_reason == "overlap_relation_has_imbalance_or_unstable_rank"


def test_relation_policy_keeps_native_blend_for_high_margin_positive_imbalance() -> None:
    decision = decide_action(
        make_relation(
            previous_delta=4_237_141.583984,
            current_delta=17_314_035.378906,
            delta_signal=13_076_893.794922,
            delta_abs_gap=13_076_893.794922,
            delta_signed_gap=13_076_893.794922,
            delta_ratio_gap=0.755277,
            both_positive=True,
            one_side_zero=False,
            rank_signal=0.0,
            rank_stability=0.0,
            shared_var_support_ratio=0.02,
            fallback_margin_proxy=0.98,
        )
    )

    assert decision.action_name == "fallback"
    assert decision.canonical_action_name == "conservative_no_action"
    assert decision.trigger_reason == "high_fallback_margin_keeps_native_overlap_blend"


def test_relation_policy_falls_back_when_no_rule_fires() -> None:
    decision = decide_action(
        make_relation(
            overlap_strength=0.0,
            shared_vars=(),
            delta_signal=0.2,
            rank_signal=0.9,
        )
    )

    assert decision.action_name == "fallback"
    assert decision.canonical_action_name == "conservative_no_action"
    assert decision.action_family == "fallback"
    assert decision.confidence == 0.0


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
        "isolate_conflicting_relation",
        "reassign_repair",
        "fallback",
    ]
    assert [decision.canonical_action_name for decision in decisions] == [
        "allow_beneficial_coordination",
        "isolate_conflicting_relation",
        "repair_shared_variable_binding",
        "conservative_no_action",
    ]
    assert (
        "relation policy action counts: "
        "coordinate=1, isolate_conflicting_relation=1, reassign_repair=1, fallback=1"
    ) in caplog.text

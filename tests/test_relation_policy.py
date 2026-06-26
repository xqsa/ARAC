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
    }
    values.update(overrides)
    return OverlapRelation(**values)


def test_relation_policy_coordinates_stable_high_overlap_relation() -> None:
    decision = decide_action(make_relation())

    assert decision.relation_id == "O1_0_1"
    assert decision.action_name == "coordinate"
    assert decision.action_family == "coordinate"
    assert decision.confidence > 0.0
    assert decision.trigger_reason == "high_overlap_with_stable_delta_and_rank"


def test_relation_policy_isolates_large_delta_conflict() -> None:
    decision = decide_action(make_relation(delta_signal=2.5, rank_signal=0.9))

    assert decision.action_name == "isolate_conflicting_relation"
    assert decision.action_family == "isolate"
    assert decision.trigger_reason == "large_delta_conflict_or_negative_divergence"


def test_relation_policy_isolates_negative_divergence() -> None:
    decision = decide_action(make_relation(delta_signal=-0.1, rank_signal=0.9))

    assert decision.action_name == "isolate_conflicting_relation"
    assert decision.action_family == "isolate"


def test_relation_policy_repairs_imbalanced_overlap() -> None:
    decision = decide_action(make_relation(delta_signal=0.6, rank_signal=0.4))

    assert decision.action_name == "reassign_repair"
    assert decision.action_family == "reassign_repair"
    assert decision.trigger_reason == "overlap_relation_has_imbalance_or_unstable_rank"


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
    assert decision.action_family == "fallback"
    assert decision.confidence == 0.0


def test_decide_actions_for_relations_preserves_order_and_logs_counts(caplog) -> None:
    relations = [
        make_relation(relation_id="O1_0_1"),
        make_relation(relation_id="O1_1_2", delta_signal=2.0),
        make_relation(relation_id="O1_2_3", delta_signal=0.6, rank_signal=0.4),
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
    assert (
        "relation policy action counts: "
        "coordinate=1, isolate_conflicting_relation=1, reassign_repair=1, fallback=1"
    ) in caplog.text

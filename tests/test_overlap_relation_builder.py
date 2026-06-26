from __future__ import annotations

from arac.evidence.overlap_relation_builder import build_overlap_relations


def test_build_overlap_relations_from_adjacent_groups() -> None:
    hcc_trace = {
        "outer_iter": 3,
        "groups": [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5],
        ],
        "fitness_deltas": [10.0, 7.5, 7.0],
        "group_ranks": [1, 2, 2],
        "budget_remaining_ratio": 0.25,
    }

    relations = build_overlap_relations(hcc_trace, "E2")

    assert [relation.relation_id for relation in relations] == ["O3_0_1", "O3_1_2"]
    assert relations[0].problem_id == "E2"
    assert relations[0].outer_iter == 3
    assert relations[0].group_left == 0
    assert relations[0].group_right == 1
    assert relations[0].shared_vars == (2,)
    assert relations[0].overlap_strength == 1.0
    assert relations[0].delta_signal == 2.5
    assert relations[0].rank_signal == 0.5
    assert relations[0].budget_remaining_ratio == 0.25
    assert relations[1].shared_vars == (4,)
    assert relations[1].delta_signal == 0.5
    assert relations[1].rank_signal == 1.0


def test_build_overlap_relations_from_iteration_payloads_and_overlap_groups() -> None:
    hcc_trace = {
        "iterations": [
            {
                "outer_iter": 0,
                "overlapping_elements": [[10, 11], [12]],
            },
            {
                "outer_iter": 1,
                "overlap_groups": [13],
                "scores": [4.0, 9.0],
                "ranks": [2, 1],
                "budget_remaining_ratio": 0.75,
            },
        ]
    }

    relations = build_overlap_relations(hcc_trace, "S6")

    assert [relation.relation_id for relation in relations] == ["O0_0_1", "O0_1_2", "O1_0_1"]
    assert relations[0].shared_vars == (10, 11)
    assert relations[0].overlap_strength == 2.0
    assert relations[0].delta_signal == 0.0
    assert relations[0].rank_signal == 0.0
    assert relations[0].budget_remaining_ratio == 1.0
    assert relations[2].shared_vars == (13,)
    assert relations[2].delta_signal == 5.0
    assert relations[2].rank_signal == 0.0
    assert relations[2].budget_remaining_ratio == 0.75

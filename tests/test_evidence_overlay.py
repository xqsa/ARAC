from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields

import pytest

from arac.policy.evidence_overlay import (
    FORBIDDEN_RUNTIME_FIELD_FRAGMENTS,
    SHADOW_GAIN_THRESHOLD,
    BridgeWeights,
    FourPointProbe,
    ProbeUtilities,
    ReferenceBlindOrdering,
    RelationCandidate,
    RelationKey,
    RelationSelection,
    RuntimeProbeAction,
    ScoredRelation,
    ShadowDecision,
    bridge_weights,
    build_four_point_probe,
    build_reference_blind_ordering,
    build_relation_candidates,
    decide_shadow_action,
    ordering_sha256,
    score_relations,
    select_top_relations,
    shuffle_relation_scores,
    summarize_probe_utilities,
    topology_sha256,
)


def _relation(
    variable: int,
    *,
    disagreement: float,
    priority: float,
    owners: tuple[int, int] | None = None,
) -> RelationCandidate:
    owner_pair = owners or (2 * variable, 2 * variable + 1)
    return RelationCandidate(
        key=RelationKey(owner_pair, (variable,)),
        owner_proposals=((0.0,), (disagreement,)),
        owner_reliabilities=(0.4, 0.6),
        proposal_disagreement=disagreement,
        owner_priority=priority,
        owner_population_centers=((0.0,), (disagreement,)),
        owner_population_standard_deviations=((0.0,), (0.0,)),
        owner_population_sizes=(1, 1),
    )


def _point_population_samples(
    proposals: dict[tuple[int, int], float],
) -> dict[tuple[int, int], tuple[float, ...]]:
    return {key: (value,) for key, value in proposals.items()}


def _scored(variable: int, score: float) -> ScoredRelation:
    relation = _relation(variable, disagreement=score, priority=score)
    return ScoredRelation(
        relation=relation,
        disagreement_rank=score,
        priority_rank=score,
        voi_score=score,
        score_source=relation.key,
    )


def test_disjoint_groups_use_lexicographic_membership_order() -> None:
    result = build_reference_blind_ordering(((7, 5), (2, 1), (4, 3)))

    assert result.groups == ((5, 7), (1, 2), (3, 4))
    assert result.group_order == (1, 2, 0)
    assert result.ordered_groups == ((1, 2), (3, 4), (5, 7))
    assert result.has_overlap is False


def test_overlap_path_uses_deterministic_structural_endpoint() -> None:
    first = build_reference_blind_ordering(((2, 3), (0, 1), (1, 2)))
    reordered = build_reference_blind_ordering(((1, 2), (2, 3), (0, 1)))

    assert first.group_order == (1, 2, 0)
    assert first.ordered_groups == ((0, 1), (1, 2), (2, 3))
    assert reordered.ordered_groups == first.ordered_groups
    assert reordered.topology_sha256 == first.topology_sha256
    assert reordered.ordering_sha256 == first.ordering_sha256
    assert first.has_overlap is True


@pytest.mark.parametrize(
    "groups",
    (
        ((0, 1), (1, 2), (0, 2)),
        ((0, 4), (1, 4), (2, 4), (3, 4)),
        ((0, 1), (1, 2), (3, 4), (4, 5)),
        ((0, 1), (1, 2), (3,)),
    ),
)
def test_non_path_overlap_topology_fails_closed(groups) -> None:
    with pytest.raises(ValueError, match="simple path"):
        build_reference_blind_ordering(groups)


def test_topology_and_order_hashes_are_structural_and_validated() -> None:
    groups = ((2, 3), (0, 1), (1, 2))
    result = build_reference_blind_ordering(groups)

    assert len(result.topology_sha256) == 64
    assert len(result.ordering_sha256) == 64
    assert result.topology_sha256 == topology_sha256(tuple(reversed(groups)))
    assert result.ordering_sha256 == ordering_sha256(groups, result.group_order)
    with pytest.raises(ValueError, match="permutation"):
        ordering_sha256(groups, (0, 0, 2))


def test_relation_candidates_require_exactly_two_structural_owners() -> None:
    groups = ((0, 1), (1, 2), (2, 3), (2, 4))
    proposals = {
        (0, 0): 0.0,
        (0, 1): 2.0,
        (1, 1): -1.0,
        (1, 2): 4.0,
        (2, 2): 6.0,
        (3, 2): 8.0,
    }

    result = build_relation_candidates(
        groups,
        proposals,
        _point_population_samples(proposals),
        group_priorities=(0.1, 0.8, 0.4, 0.7),
        owner_reliabilities=(0.2, 0.9, 0.3, 0.6),
        lower_bound=-5.0,
        upper_bound=5.0,
    )

    assert len(result) == 1
    assert result[0].key == RelationKey((0, 1), (1,))
    assert result[0].owner_proposals == ((2.0,), (-1.0,))
    assert result[0].owner_reliabilities == (0.2, 0.9)
    assert result[0].proposal_disagreement == pytest.approx(0.3)
    assert result[0].owner_priority == 0.8


def test_relation_candidate_missing_owner_proposal_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing proposal"):
        build_relation_candidates(
            ((0, 1), (1, 2)),
            {(0, 1): 0.2},
            {(0, 1): (0.2,)},
            group_priorities=(0.4, 0.6),
            owner_reliabilities=(0.5, 0.5),
            lower_bound=-1.0,
            upper_bound=1.0,
        )

    with pytest.raises(ValueError, match="structural group"):
        build_relation_candidates(
            ((0, 1), (1, 2)),
            {(0, 2): 0.2, (1, 1): 0.3},
            {},
            group_priorities=(0.4, 0.6),
            owner_reliabilities=(0.5, 0.5),
            lower_bound=-1.0,
            upper_bound=1.0,
        )


def test_group_pair_relation_aggregates_aligned_shared_variables() -> None:
    proposals = {
        (0, 1): -2.0,
        (0, 2): 4.0,
        (1, 1): 2.0,
        (1, 2): -2.0,
    }
    result = build_relation_candidates(
        ((0, 1, 2), (1, 2, 3), (4, 5)),
        proposals,
        _point_population_samples(proposals),
        group_priorities=(0.3, 0.7, 0.2),
        owner_reliabilities=(0.4, 0.8, 0.1),
        lower_bound=-10.0,
        upper_bound=10.0,
    )

    assert len(result) == 1
    relation = result[0]
    assert relation.key == RelationKey((0, 1), (1, 2))
    assert relation.owner_proposals == ((-2.0, 4.0), (2.0, -2.0))
    assert relation.proposal_disagreement == pytest.approx((0.2 + 0.3) / 2.0)
    assert relation.owner_priority == 0.7


def test_distribution_disagreement_includes_center_and_variance() -> None:
    relation = build_relation_candidates(
        ((0, 1), (1, 2)),
        {(0, 1): -4.0, (1, 1): 4.0},
        {(0, 1): (-1.0, 1.0), (1, 1): (0.0, 0.0)},
        group_priorities=(0.4, 0.6),
        owner_reliabilities=(0.5, 0.5),
        lower_bound=-5.0,
        upper_bound=5.0,
    )[0]

    assert relation.owner_proposals == ((-4.0,), (4.0,))
    assert relation.owner_population_centers == ((0.0,), (0.0,))
    assert relation.owner_population_standard_deviations == ((1.0,), (0.0,))
    assert relation.owner_population_sizes == (2, 2)
    assert relation.proposal_disagreement == pytest.approx(0.1)


def test_twenty_group_path_builds_nineteen_multivariable_relations() -> None:
    edge_variables = tuple(
        tuple(3 * edge + offset for offset in range(3)) for edge in range(19)
    )
    groups = tuple(
        tuple(
            sorted(
                {
                    1000 + group,
                    *(edge_variables[group - 1] if group > 0 else ()),
                    *(edge_variables[group] if group < 19 else ()),
                }
            )
        )
        for group in range(20)
    )
    proposals = {
        (group, variable): float(group + variable % 3)
        for group, variables in enumerate(groups)
        for variable in variables
        if variable < 1000
    }

    relations = build_relation_candidates(
        groups,
        proposals,
        _point_population_samples(proposals),
        group_priorities=tuple(group / 20.0 for group in range(20)),
        owner_reliabilities=(0.5,) * 20,
        lower_bound=-100.0,
        upper_bound=100.0,
    )

    assert len(relations) == 19
    assert tuple(relation.key.owner_group_indices for relation in relations) == tuple(
        (group, group + 1) for group in range(19)
    )
    assert all(len(relation.key.shared_variable_indices) == 3 for relation in relations)


def test_relation_scoring_uses_midrank_harmonic_voi() -> None:
    scored = score_relations(
        (
            _relation(0, disagreement=1.0, priority=3.0 / 4.0),
            _relation(1, disagreement=1.0, priority=1.0 / 4.0),
            _relation(2, disagreement=3.0, priority=1.0 / 2.0),
        )
    )

    assert tuple(item.disagreement_rank for item in scored) == pytest.approx(
        (1.0 / 3.0, 1.0 / 3.0, 5.0 / 6.0)
    )
    assert tuple(item.priority_rank for item in scored) == pytest.approx(
        (5.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0)
    )
    assert scored[0].voi_score == pytest.approx(10.0 / 21.0)
    assert scored[1].voi_score == pytest.approx(2.0 / 9.0)
    assert scored[2].voi_score == pytest.approx(5.0 / 8.0)


def test_top_four_abstains_only_when_cutoff_is_ambiguous() -> None:
    internal_tie = tuple(
        _scored(variable, score)
        for variable, score in enumerate((0.9, 0.9, 0.8, 0.7, 0.6))
    )
    cutoff_tie = tuple(
        _scored(variable, score)
        for variable, score in enumerate((0.9, 0.8, 0.7, 0.6, 0.6))
    )

    selected = select_top_relations(internal_tie)
    abstained = select_top_relations(cutoff_tie)

    assert selected.abstained is False
    assert len(selected.selected) == 4
    assert abstained == RelationSelection((), True, "non_unique_top_relation_cutoff")
    assert select_top_relations(internal_tie[:3]).reason == (
        "insufficient_eligible_relations"
    )


def test_shuffled_scores_are_deterministic_deranged_and_change_top_four() -> None:
    native = tuple(
        _scored(variable, score)
        for variable, score in enumerate((0.95, 0.8, 0.65, 0.5, 0.35, 0.2))
    )

    first = shuffle_relation_scores(native, seed=117)
    repeated = shuffle_relation_scores(tuple(reversed(native)), seed=117)

    assert first == repeated
    assert all(item.score_source != item.relation.key for item in first)
    assert sorted(item.voi_score for item in first) == sorted(
        item.voi_score for item in native
    )
    assert {
        item.relation.key for item in select_top_relations(first).selected
    } != {
        item.relation.key for item in select_top_relations(native).selected
    }
    with pytest.raises(ValueError, match="at least two"):
        shuffle_relation_scores(native[:1], seed=117)


def test_bridge_weights_are_proportional_and_capped() -> None:
    assert bridge_weights(0.5, 0.5) == BridgeWeights(0.5, 0.5)
    proportional = bridge_weights(0.2, 0.8)
    assert proportional.left_owner == pytest.approx(0.4)
    assert proportional.right_owner == pytest.approx(0.6)
    assert bridge_weights(0.0, 1.0) == BridgeWeights(0.35, 0.65)
    assert bridge_weights(1.0, 0.0) == BridgeWeights(0.65, 0.35)


def test_four_point_probe_changes_only_the_shared_coordinate() -> None:
    relation = RelationCandidate(
        key=RelationKey((0, 1), (1, 2)),
        owner_proposals=((10.0, 30.0), (-2.0, -4.0)),
        owner_reliabilities=(0.0, 1.0),
        proposal_disagreement=0.5,
        owner_priority=0.8,
        owner_population_centers=((10.0, 30.0), (-2.0, -4.0)),
        owner_population_standard_deviations=((0.0, 0.0), (0.0, 0.0)),
        owner_population_sizes=(1, 1),
    )

    probe = build_four_point_probe((1.0, 2.0, 3.0, 4.0), relation)

    assert probe == FourPointProbe(
        relation=relation.key,
        weights=BridgeWeights(0.35, 0.65),
        x0=(1.0, 2.0, 3.0, 4.0),
        x_left=(1.0, 10.0, 30.0, 4.0),
        x_right=(1.0, -2.0, -4.0, 4.0),
        x_bridge=(1.0, 2.2, 7.9, 4.0),
    )


def test_probe_utility_is_fixed_log_ratio_and_rejects_invalid_fitness() -> None:
    utilities = summarize_probe_utilities(
        anchor_fitness=100.0,
        left_fitness=50.0,
        right_fitness=100.0,
        bridge_fitness=25.0,
    )

    assert utilities.left_owner == pytest.approx(math.log(2.0))
    assert utilities.right_owner == pytest.approx(0.0)
    assert utilities.bridge == pytest.approx(math.log(4.0))

    optimum = summarize_probe_utilities(
        anchor_fitness=1.0,
        left_fitness=0.0,
        right_fitness=1.0,
        bridge_fitness=1.0,
    )
    assert math.isfinite(optimum.left_owner)
    assert optimum.left_owner > SHADOW_GAIN_THRESHOLD

    for invalid in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            summarize_probe_utilities(
                anchor_fitness=1.0,
                left_fitness=invalid,
                right_fitness=1.0,
                bridge_fitness=1.0,
            )


def test_shadow_decision_is_observer_only_and_uses_fixed_gate() -> None:
    owner = decide_shadow_action(
        ProbeUtilities(SHADOW_GAIN_THRESHOLD, 0.0, -0.1)
    )
    bridge = decide_shadow_action(ProbeUtilities(0.0, -0.1, 0.2))
    below = decide_shadow_action(
        ProbeUtilities(SHADOW_GAIN_THRESHOLD / 2.0, 0.0, -0.1)
    )
    tied = decide_shadow_action(ProbeUtilities(0.2, 0.2, 0.1))

    assert owner.shadow_action == "repair"
    assert owner.winner == "left_owner"
    assert bridge.shadow_action == "coordinate"
    assert bridge.winner == "bridge"
    assert below.shadow_action == "fallback"
    assert below.reason == "probe_gain_below_one_percent"
    assert tied.shadow_action == "fallback"
    assert tied.reason == "non_unique_best_probe_utility"
    assert all(
        decision.runtime_authorized is False
        for decision in (owner, bridge, below, tied)
    )
    with pytest.raises(FrozenInstanceError):
        owner.runtime_authorized = True  # type: ignore[misc]


def test_runtime_records_have_no_aob_identity_or_outcome_fields() -> None:
    record_types = (
        ReferenceBlindOrdering,
        RelationKey,
        RelationCandidate,
        ScoredRelation,
        RelationSelection,
        BridgeWeights,
        FourPointProbe,
        ProbeUtilities,
        ShadowDecision,
        RuntimeProbeAction,
    )
    names = {item.name.lower() for record in record_types for item in fields(record)}

    assert not {
        name
        for name in names
        if any(fragment in name for fragment in FORBIDDEN_RUNTIME_FIELD_FRAGMENTS)
    }

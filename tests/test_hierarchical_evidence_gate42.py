"""Gate 42: evidence-semantics property tests for the three-layer model.

Pass criteria (docs/arac-phase1-v10-design.md §6):
1. Region-only evidence can never yield shared variables (direct construction
   gives none; the conversion API fails closed without hyperedges);
2. The schema rejects every "variable belongs to two disjoint leaves" path;
3. Hyperedge -> OverlapStructure conversion carries a complete audit trail.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from arac.coordination.overlap import OverlapStructure
from arac.evidence.hierarchical import (
    MIN_SIGN_STABILITY,
    Phase1Evidence,
    RegionNode,
    RegionRelation,
    RegionStructure,
    RegionTree,
    ResolvedOverlapHyperedge,
    VariableRegionInteraction,
    mode_of_component,
    to_overlap_structure,
)


def _two_leaf_tree(dimension: int = 8) -> RegionTree:
    return RegionTree(
        dimension=dimension,
        nodes=(
            RegionNode(node_id=0, parent=None, depth=0, variables=tuple(range(dimension))),
            RegionNode(node_id=1, parent=0, depth=1, variables=(0, 1, 2, 3)),
            RegionNode(node_id=2, parent=0, depth=1, variables=(4, 5, 6, 7)),
        ),
    )


def _interaction(variable: int, source: int, target: int, sign_stability: float = 1.0):
    return VariableRegionInteraction(
        variable=variable,
        source_region=source,
        target_region=target,
        q_lb=0.8,
        support=6,
        sign_stability=sign_stability,
    )


def _status_all(dimension: int, status: str = "observed_separable"):
    return tuple((variable, status) for variable in range(dimension))


def test_gate42_property_1a_disjoint_leaves_have_no_shared_variables() -> None:
    """v10.0's bug is structurally impossible: leaf groups share nothing."""

    rng = random.Random(20260815)
    for _ in range(25):
        dimension = rng.randint(4, 40)
        cut = rng.randint(1, dimension - 1)
        variables = list(range(dimension))
        rng.shuffle(variables)
        left = tuple(sorted(variables[:cut]))
        right = tuple(sorted(variables[cut:]))
        structure = OverlapStructure(dimension=dimension, groups=(left, right))
        assert structure.shared_variables == ()


def test_gate42_property_1b_conversion_fails_closed_without_hyperedges() -> None:
    tree = _two_leaf_tree()
    evidence = Phase1Evidence(
        dimension=8,
        region_tree=tree,
        region_relations=(RegionRelation(left=1, right=2, score=3.0, stability=1.0, depth=1),),
        variable_region_interactions=(_interaction(1, source=1, target=2),),
        variable_status=_status_all(8, "member_candidate"),
    )

    with pytest.raises(ValueError, match="no resolved overlap hyperedges"):
        to_overlap_structure(evidence)


def test_gate42_property_2_schema_rejects_membership_semantics() -> None:
    tree = _two_leaf_tree()

    with pytest.raises(ValueError, match="own home leaf"):
        _interaction(0, source=1, target=1)
    with pytest.raises(ValueError, match="home leaf"):
        Phase1Evidence(
            dimension=8,
            region_tree=tree,
            region_relations=(),
            variable_region_interactions=(_interaction(4, source=1, target=2),),
            variable_status=_status_all(8),
        )
    with pytest.raises(ValueError, match="at least two regions"):
        ResolvedOverlapHyperedge(variable=1, regions=(1,), evidence=())
    with pytest.raises(ValueError, match="confirming evidence"):
        ResolvedOverlapHyperedge(variable=1, regions=(1, 2), evidence=())
    with pytest.raises(ValueError, match="sign-stability"):
        ResolvedOverlapHyperedge(
            variable=1,
            regions=(1, 2),
            evidence=(_interaction(1, 1, 2, sign_stability=0.5),),
        )
    with pytest.raises(ValueError, match="recorded interactions"):
        Phase1Evidence(
            dimension=8,
            region_tree=tree,
            region_relations=(),
            variable_region_interactions=(),
            resolved_hyperedges=(
                ResolvedOverlapHyperedge(
                    variable=1,
                    regions=(1, 2),
                    evidence=(_interaction(1, 1, 2),),
                ),
            ),
            variable_status=_status_all(8, "member_candidate"),
        )


def test_gate42_property_3_conversion_carries_audit_trail() -> None:
    tree = _two_leaf_tree()
    interaction = _interaction(1, source=1, target=2)
    evidence = Phase1Evidence(
        dimension=8,
        region_tree=tree,
        region_relations=(RegionRelation(left=1, right=2, score=3.0, stability=1.0, depth=1),),
        variable_region_interactions=(interaction,),
        resolved_hyperedges=(
            ResolvedOverlapHyperedge(variable=1, regions=(1, 2), evidence=(interaction,)),
        ),
        variable_status=_status_all(8, "member_candidate"),
        per_component_mode=(((1, 2), "HIERARCHICAL"),),
        level_budgets=(("coarse", 15_870), ("incumbent", 76_064)),
    )

    structure = to_overlap_structure(evidence)

    assert structure.shared_variables == (1,)
    memberships = structure.memberships
    assert sorted(memberships[1]) == [0, 1]
    groups = [set(group) for group in structure.groups]
    assert 1 in groups[0] and 1 in groups[1]
    assert set().union(*groups) == set(range(8))
    for variable in range(8):
        if variable != 1:
            assert len(memberships[variable]) == 1
    assert mode_of_component(evidence, (1, 2)) == "HIERARCHICAL"


def test_gate42_tree_partition_invariants_hold_under_random_splits() -> None:
    rng = random.Random(20260816)
    for _ in range(25):
        dimension = rng.randint(2, 30)
        variables = list(range(dimension))
        rng.shuffle(variables)
        cut = rng.randint(1, dimension - 1)
        nodes = (
            RegionNode(0, None, 0, tuple(range(dimension))),
            RegionNode(1, 0, 1, tuple(sorted(variables[:cut]))),
            RegionNode(2, 0, 1, tuple(sorted(variables[cut:]))),
        )
        tree = RegionTree(dimension=dimension, nodes=nodes)
        covered = sorted(v for leaf in tree.leaves for v in leaf.variables)
        assert covered == list(range(dimension))

    with pytest.raises(ValueError, match="partition"):
        RegionTree(
            dimension=4,
            nodes=(
                RegionNode(0, None, 0, (0, 1, 2, 3)),
                RegionNode(1, 0, 1, (0, 1)),
                RegionNode(2, 0, 1, (1, 2, 3)),
            ),
        )


def test_gate42_region_structure_has_no_conversion_path() -> None:
    tree = _two_leaf_tree()
    evidence = Phase1Evidence(
        dimension=8,
        region_tree=tree,
        region_relations=(RegionRelation(left=1, right=2, score=3.0, stability=1.0, depth=1),),
        variable_region_interactions=(),
        variable_status=_status_all(8),
        per_component_mode=(((1, 2), "EVIDENCE_DENSE"),),
    )
    region_structure = RegionStructure(evidence=evidence)

    assert not hasattr(region_structure, "to_overlap_structure")
    assert region_structure.partners(1) == (2,)
    assert region_structure.components() == ((1, 2),)
    assert region_structure.region_variables(1) == (0, 1, 2, 3)
    assert mode_of_component(evidence, (1, 2)) == "EVIDENCE_DENSE"


def test_gate42_evidence_validation_covers_status_modes_and_components() -> None:
    tree = _two_leaf_tree()
    base = dict(
        dimension=8,
        region_tree=tree,
        region_relations=(),
        variable_region_interactions=(),
    )

    with pytest.raises(ValueError, match="exactly once"):
        Phase1Evidence(**base, variable_status=((0, "observed_separable"),))
    with pytest.raises(ValueError, match="unknown variable status"):
        Phase1Evidence(**base, variable_status=_status_all(8, "member_of_two_leaves"))
    with pytest.raises(ValueError, match="unknown evidence mode"):
        Phase1Evidence(
            **base,
            variable_status=_status_all(8),
            per_component_mode=(((1, 2), "DENSE"),),
        )
    with pytest.raises(ValueError, match="partition the leaves"):
        Phase1Evidence(
            **base,
            variable_status=_status_all(8),
            per_component_mode=(((1,), "SPARSE"),),
        )


def test_gate42_sign_stability_threshold_matches_design() -> None:
    assert MIN_SIGN_STABILITY == pytest.approx(5.0 / 6.0)


def test_gate42_deterministic_hashability() -> None:
    interaction = _interaction(1, source=1, target=2)
    evidence = Phase1Evidence(
        dimension=8,
        region_tree=_two_leaf_tree(),
        region_relations=(RegionRelation(1, 2, 3.0, 1.0, 1),),
        variable_region_interactions=(interaction,),
        variable_status=_status_all(8, "member_candidate"),
    )
    assert hash(evidence) == hash(
        Phase1Evidence(
            dimension=8,
            region_tree=_two_leaf_tree(),
            region_relations=(RegionRelation(1, 2, 3.0, 1.0, 1),),
            variable_region_interactions=(interaction,),
            variable_status=_status_all(8, "member_candidate"),
        )
    )
    assert np.isfinite(interaction.q_lb)

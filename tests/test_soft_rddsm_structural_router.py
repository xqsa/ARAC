from __future__ import annotations

from arac.analysis.structural_router import route_from_overlap_evidence
from arac.evidence.hierarchical import (
    Phase1Evidence,
    RegionNode,
    RegionRelation,
    RegionTree,
    ResolvedOverlapHyperedge,
    VariableRegionInteraction,
)
from arac.evidence.overlap_adapter import Phase1OverlapEvidence
from arac.evidence.soft_rddsm_adapter import (
    overlap_evidence_hash,
    overlap_evidence_payload,
    soft_evidence_to_overlap_evidence,
)


def _hierarchical_evidence(*, shared: bool, unresolved: bool = False) -> Phase1Evidence:
    interaction = VariableRegionInteraction(
        variable=1,
        source_region=1,
        target_region=2,
        q_lb=0.8,
        support=5,
        sign_stability=1.0,
    )
    hyperedges = (
        ResolvedOverlapHyperedge(
            variable=1,
            regions=(1, 2),
            evidence=(interaction,),
        ),
    ) if shared else ()
    statuses = tuple(
        (
            variable,
            "not_yet_resolved"
            if variable == 1 and unresolved
            else "observed_separable",
        )
        for variable in range(6)
    )
    return Phase1Evidence(
        dimension=6,
        region_tree=RegionTree(
            dimension=6,
            nodes=(
                RegionNode(0, None, 0, (0, 1, 2, 3, 4, 5)),
                RegionNode(1, 0, 1, (0, 1)),
                RegionNode(2, 0, 1, (2, 3)),
                RegionNode(3, 0, 1, (4, 5)),
            ),
        ),
        region_relations=(
            (RegionRelation(left=1, right=2, score=0.5, stability=1.0, depth=1),)
            if shared
            else ()
        ),
        variable_region_interactions=(interaction,) if shared else (),
        resolved_hyperedges=hyperedges,
        variable_status=statuses,
        per_component_mode=(((1, 2, 3), "SPARSE"),),
        level_budgets=(("signature", 10),),
    )


def _sidecar(groups: tuple[tuple[int, ...], ...], *, complete: bool = True) -> Phase1OverlapEvidence:
    memberships = tuple(
        tuple(group for group, variables in enumerate(groups) if variable in variables)
        for variable in range(max(max(group) for group in groups) + 1)
    )
    confidences = tuple(
        (variable, group, 0.8)
        for variable, owners in enumerate(memberships)
        for group in owners
    )
    return Phase1OverlapEvidence(
        dimension=len(memberships),
        groups=groups,
        memberships=memberships,
        membership_confidences=confidences,
        complete=complete,
    )


def test_soft_rddsm_converter_preserves_disjoint_leaves_without_false_overlap() -> None:
    result = soft_evidence_to_overlap_evidence(_hierarchical_evidence(shared=False))

    assert result.complete is True
    assert result.memberships[1] == (0,)
    assert all(len(owners) == 1 for owners in result.memberships)


def test_soft_rddsm_converter_adds_only_confirmed_hyperedge_owners() -> None:
    result = soft_evidence_to_overlap_evidence(_hierarchical_evidence(shared=True))

    assert result.complete is True
    assert result.memberships[1] == (0, 1)
    assert result.groups[0] == (0, 1)
    assert result.groups[1] == (1, 2, 3)
    assert result.groups[2] == (4, 5)
    assert result.membership_confidences[1][2] == 0.8


def test_sidecar_payload_and_hash_are_deterministic() -> None:
    evidence = soft_evidence_to_overlap_evidence(_hierarchical_evidence(shared=True))

    payload = overlap_evidence_payload(evidence)
    assert payload["schema_version"] == "arac-soft-rddsm-overlap-evidence-v1"
    assert payload["complete"] is True
    assert overlap_evidence_hash(evidence) == overlap_evidence_hash(evidence)


def test_unresolved_soft_evidence_is_fail_closed_to_aor() -> None:
    sidecar = soft_evidence_to_overlap_evidence(
        _hierarchical_evidence(shared=False, unresolved=True)
    )

    decision = route_from_overlap_evidence(sidecar)
    assert sidecar.complete is False
    assert decision.action_name == "aor"
    assert decision.reason == "overlap_evidence_incomplete"
    assert decision.smp_compatible is False
    assert decision.smp_mode == "unavailable"


def test_region_interaction_without_hyperedge_is_not_called_disjoint() -> None:
    evidence = _hierarchical_evidence(shared=False)
    evidence = Phase1Evidence(
        dimension=evidence.dimension,
        region_tree=evidence.region_tree,
        region_relations=(
            RegionRelation(left=1, right=2, score=0.5, stability=1.0, depth=1),
        ),
        variable_region_interactions=(),
        resolved_hyperedges=(),
        variable_status=evidence.variable_status,
        per_component_mode=(((1, 2), "HIERARCHICAL"), ((3,), "SPARSE")),
        level_budgets=evidence.level_budgets,
    )

    sidecar = soft_evidence_to_overlap_evidence(evidence)
    assert sidecar.complete is False
    assert route_from_overlap_evidence(sidecar).action_name == "aor"


def test_missing_component_modes_are_not_called_complete() -> None:
    evidence = _hierarchical_evidence(shared=False)
    evidence = Phase1Evidence(
        dimension=evidence.dimension,
        region_tree=evidence.region_tree,
        region_relations=(),
        variable_region_interactions=(),
        resolved_hyperedges=(),
        variable_status=evidence.variable_status,
        per_component_mode=(),
        level_budgets=evidence.level_budgets,
    )

    sidecar = soft_evidence_to_overlap_evidence(evidence)
    assert sidecar.complete is False
    assert route_from_overlap_evidence(sidecar).action_name == "aor"


def test_hyperedge_in_one_component_cannot_close_another_component() -> None:
    interaction = VariableRegionInteraction(
        variable=1,
        source_region=1,
        target_region=2,
        q_lb=0.8,
        support=5,
        sign_stability=1.0,
    )
    evidence = Phase1Evidence(
        dimension=8,
        region_tree=RegionTree(
            dimension=8,
            nodes=(
                RegionNode(0, None, 0, tuple(range(8))),
                RegionNode(1, 0, 1, (0, 1)),
                RegionNode(2, 0, 1, (2, 3)),
                RegionNode(3, 0, 1, (4, 5)),
                RegionNode(4, 0, 1, (6, 7)),
            ),
        ),
        region_relations=(
            RegionRelation(left=1, right=2, score=0.5, stability=1.0, depth=1),
            RegionRelation(left=3, right=4, score=0.4, stability=1.0, depth=1),
        ),
        variable_region_interactions=(interaction,),
        resolved_hyperedges=(
            ResolvedOverlapHyperedge(variable=1, regions=(1, 2), evidence=(interaction,)),
        ),
        variable_status=tuple(
            (variable, "member_candidate" if variable == 1 else "observed_separable")
            for variable in range(8)
        ),
        per_component_mode=(((1, 2), "SPARSE"), ((3, 4), "HIERARCHICAL")),
        level_budgets=(("signature", 10),),
    )

    sidecar = soft_evidence_to_overlap_evidence(evidence)
    assert sidecar.complete is False
    assert route_from_overlap_evidence(sidecar).action_name == "aor"


def test_structural_router_selects_smp_for_complete_disjoint_evidence() -> None:
    decision = route_from_overlap_evidence(
        soft_evidence_to_overlap_evidence(_hierarchical_evidence(shared=False))
    )

    assert decision.action_name == "smp"
    assert decision.reason == "complete_disjoint_structure"
    assert decision.smp_compatible is True
    assert decision.smp_mode == "zero_relation"
    assert decision.shared_variables == ()


def test_structural_router_selects_ctp_for_disconnected_overlap_components() -> None:
    decision = route_from_overlap_evidence(
        _sidecar(((0, 1), (1, 2), (3, 4), (4, 5)))
    )

    assert decision.action_name == "ctp"
    assert decision.reason == "complete_disconnected_overlap_components"
    assert decision.smp_compatible is True
    assert decision.smp_mode == "overlap_aware"
    assert decision.shared_variables == (1, 4)
    assert len(decision.components) == 2
    assert sum(weight for _, weight in decision.component_weights) == 1.0


def test_structural_router_selects_gcb_for_connected_overlap_graph() -> None:
    decision = route_from_overlap_evidence(
        _sidecar(((0, 1), (1, 2), (2, 3)))
    )

    assert decision.action_name == "gcb"
    assert decision.reason == "complete_connected_overlap_graph"
    assert decision.smp_compatible is True
    assert decision.smp_mode == "overlap_aware"
    assert decision.largest_component_fraction == 1.0
    assert decision.maximum_shared_degree == 2

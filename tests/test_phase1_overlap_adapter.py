from __future__ import annotations

import pytest

from arac.evidence import (
    INFERENCE_INCOMPLETE,
    INFERENCE_READY,
    PARTITION_ONLY_REASON,
    Phase1OverlapAdapter,
    Phase1OverlapEvidence,
)
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence


def _checkpoint(*, with_relation: bool = True) -> PhaseCheckpoint:
    relations = (
        (RelationEvidence(0, 1, strength=0.8, disagreement=0.4),)
        if with_relation
        else ()
    )
    return PhaseCheckpoint(
        protocol="phase1-overlap-adapter-test-v1",
        run_seed=13,
        total_budget_fes=100,
        phase1_fes=10,
        incumbent=(0.0, 0.0, 0.0, 0.0),
        incumbent_error=0.0,
        feature_names=("structural_inference_complete",),
        feature_values=(1.0,),
        blocks=((0, 1), (2, 3)),
        relations=relations,
    )


def _complete_evidence() -> Phase1OverlapEvidence:
    return Phase1OverlapEvidence(
        dimension=4,
        groups=((0, 1), (1, 2, 3)),
        memberships=((0,), (0, 1), (1,), (1,)),
        membership_confidences=(
            (0, 0, 0.9),
            (1, 0, 0.8),
            (1, 1, 0.7),
            (2, 1, 0.9),
            (3, 1, 0.9),
        ),
        complete=True,
    )


def test_partition_checkpoint_fails_closed_even_with_a_relation_edge() -> None:
    checkpoint = _checkpoint(with_relation=True)

    result = Phase1OverlapAdapter().adapt(checkpoint)

    assert result.status == INFERENCE_INCOMPLETE
    assert result.reason == PARTITION_ONLY_REASON
    assert result.checkpoint_hash == checkpoint.checkpoint_hash
    assert result.structure is None


def test_relation_edge_is_not_reinterpreted_as_shared_variable_membership() -> None:
    checkpoint = _checkpoint(with_relation=True)
    left, right = checkpoint.relations[0].left_block, checkpoint.relations[0].right_block

    result = Phase1OverlapAdapter().adapt(checkpoint)

    assert (left, right) == (0, 1)
    assert set(checkpoint.blocks[left]).isdisjoint(checkpoint.blocks[right])
    assert result.structure is None


def test_incomplete_variable_level_evidence_does_not_construct_a_structure() -> None:
    evidence = Phase1OverlapEvidence(
        dimension=4,
        groups=((0, 1), (1, 2, 3)),
        memberships=((0,), (0, 1), (1,), (1,)),
        membership_confidences=(),
        complete=False,
    )

    result = Phase1OverlapAdapter().adapt(_checkpoint(), evidence)

    assert result.status == INFERENCE_INCOMPLETE
    assert result.reason == "variable_membership_evidence_incomplete"
    assert result.structure is None


def test_missing_membership_confidence_fails_closed() -> None:
    evidence = _complete_evidence()
    evidence = Phase1OverlapEvidence(
        dimension=evidence.dimension,
        groups=evidence.groups,
        memberships=evidence.memberships,
        membership_confidences=evidence.membership_confidences[:-1],
        complete=True,
    )

    result = Phase1OverlapAdapter().adapt(_checkpoint(), evidence)

    assert result.status == INFERENCE_INCOMPLETE
    assert result.reason == "membership_confidence_incomplete"
    assert result.structure is None


def test_complete_variable_level_evidence_constructs_overlap_structure() -> None:
    result = Phase1OverlapAdapter().adapt(_checkpoint(), _complete_evidence())

    assert result.status == INFERENCE_READY
    assert result.ready
    assert result.structure is not None
    assert result.structure.groups == ((0, 1), (1, 2, 3))
    assert result.structure.shared_variables == (1,)
    assert result.structure.owners(1) == (0, 1)
    assert result.structure.confidence(1, 0) == pytest.approx(0.8)
    assert result.structure.confidence(1, 1) == pytest.approx(0.7)


def test_inconsistent_membership_and_groups_are_rejected() -> None:
    evidence = _complete_evidence()
    inconsistent = Phase1OverlapEvidence(
        dimension=evidence.dimension,
        groups=evidence.groups,
        memberships=((0,), (0,), (1,), (1,)),
        membership_confidences=evidence.membership_confidences,
        complete=True,
    )

    with pytest.raises(ValueError, match="memberships disagree"):
        Phase1OverlapAdapter().adapt(_checkpoint(), inconsistent)

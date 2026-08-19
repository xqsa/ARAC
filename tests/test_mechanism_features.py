from __future__ import annotations

import math

import pytest

from arac.evidence.mechanism_features import (
    CANDIDATE_FEATURE_NAMES,
    CTP_COVER_FEATURE_NAMES,
    DISAGREEMENT_FEATURE_NAMES,
    PROGRESS_FEATURE_NAMES,
    PhaseProgressErrors,
    candidate_feature_items,
    candidate_feature_map,
    summarize_ctp_cover,
    summarize_mechanism_features,
    summarize_phase_progress,
    summarize_relation_disagreement,
    summarize_relation_topology,
)
from arac.runtime.contracts import RelationEvidence


def _relations() -> tuple[RelationEvidence, ...]:
    return (
        RelationEvidence(0, 1, strength=2.0, disagreement=0.0),
        RelationEvidence(1, 2, strength=1.0, disagreement=0.2),
        RelationEvidence(0, 2, strength=1.0, disagreement=0.8),
        RelationEvidence(0, 1, strength=1.0, disagreement=0.4),
    )


def test_disagreement_summary_has_fixed_order_and_expected_statistics() -> None:
    values = summarize_relation_disagreement(_relations())

    assert DISAGREEMENT_FEATURE_NAMES == (
        "relation_disagreement_median",
        "relation_disagreement_std",
        "relation_disagreement_q90",
        "relation_disagreement_max",
        "relation_disagreement_nonzero_fraction",
    )
    assert values[0] == pytest.approx(0.3)
    assert values[1] == pytest.approx(math.sqrt(0.0875))
    assert values[2] == pytest.approx(0.68)
    assert values[3] == pytest.approx(0.8)
    assert values[4] == pytest.approx(0.75)


def test_empty_relations_return_zero_candidate_summaries() -> None:
    assert summarize_relation_disagreement(()) == (0.0,) * 5
    assert candidate_feature_values_for_empty() == (0.0,) * len(CANDIDATE_FEATURE_NAMES)
    assert summarize_relation_topology(((0,), (1,)), ()) == (0.0, 0.0, 0.0)
    assert summarize_ctp_cover(((0,), (1,)), ()) == (0.0, 0.0)


def candidate_feature_values_for_empty() -> tuple[float, ...]:
    return tuple(value for _, value in candidate_feature_items((), None))


def test_topology_and_ctp_cover_are_deterministic_candidate_summaries() -> None:
    blocks = ((0, 1), (2, 3), (4,))
    relations = (
        RelationEvidence(0, 1, strength=2.0, disagreement=0.5),
        RelationEvidence(1, 2, strength=1.0, disagreement=0.0),
    )

    concentration, entropy, largest = summarize_relation_topology(blocks, relations)
    assert concentration == pytest.approx(4.0 / 8.0)
    assert entropy == pytest.approx(
        -(3 / 8 * math.log(3 / 8) + 4 / 8 * math.log(4 / 8) + 1 / 8 * math.log(1 / 8))
        / math.log(3)
    )
    assert largest == pytest.approx(1.0)
    assert summarize_ctp_cover(blocks, relations) == pytest.approx((2 / 3, 7 / 5))
    assert len(CTP_COVER_FEATURE_NAMES) == 2


def test_progress_summary_accepts_sequence_mapping_and_dataclass() -> None:
    sequence = summarize_phase_progress((1000.0, 100.0, 10.0, 1.0))
    mapping = summarize_phase_progress(
        {"probe": 1000.0, "warmup": 100.0, "structure": 10.0, "tail": 1.0}
    )
    dataclass_values = summarize_phase_progress(PhaseProgressErrors(1000.0, 100.0, 10.0, 1.0))

    assert sequence == pytest.approx(mapping)
    assert sequence == pytest.approx(dataclass_values)
    assert sequence[0] == pytest.approx(math.log10(1001.0 / 101.0))
    assert sequence[1] == pytest.approx(math.log10(101.0 / 11.0))
    assert sequence[2] == pytest.approx(math.log10(11.0 / 2.0))
    assert sequence[3] == pytest.approx((sequence[1] + sequence[2]) / sum(sequence[:3]))
    assert PROGRESS_FEATURE_NAMES[-1] == "late_gain_fraction"


def test_candidate_mapping_preserves_schema_order() -> None:
    items = candidate_feature_items((), None)
    mapping = candidate_feature_map((), None)

    assert tuple(name for name, _ in items) == CANDIDATE_FEATURE_NAMES
    assert tuple(mapping) == CANDIDATE_FEATURE_NAMES
    assert len(items) == len(CANDIDATE_FEATURE_NAMES)


def test_blocks_first_entrypoint_keeps_topology_candidates_out_of_active_schema() -> None:
    items = summarize_mechanism_features(((0,), (1,)), ())

    assert tuple(name for name, _ in items) == CANDIDATE_FEATURE_NAMES
    assert not set(name for name, _ in items) & set(CTP_COVER_FEATURE_NAMES)


@pytest.mark.parametrize(
    ("blocks", "relations"),
    [
        (((0,),), (RelationEvidence(0, 1, strength=1.0, disagreement=0.0),)),
        (((0, 1), (1, 2)), ()),
    ],
)
def test_graph_summaries_reject_invalid_block_inputs(blocks, relations) -> None:
    with pytest.raises(ValueError):
        summarize_relation_topology(blocks, relations)


def test_progress_summary_rejects_invalid_endpoints() -> None:
    with pytest.raises(ValueError):
        summarize_phase_progress((1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        summarize_phase_progress((1.0, math.nan, 3.0, 4.0))

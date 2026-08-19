from __future__ import annotations

from arac.coordination import LocalProposal, OverlapStructure
from experiments.proposal_conditioned_context_gate26 import (
    CONTINUATION_FES,
    build_candidate_stream,
)


def test_gate26_candidate_stream_is_deterministic_and_balanced() -> None:
    structure = OverlapStructure(3, ((0, 1), (1, 2)))
    proposals = (
        LocalProposal(0, ((0, 1.0), (1, 2.0)), 2.0, ((0, 0.1), (1, 0.2))),
        LocalProposal(1, ((1, 3.0), (2, 4.0)), 1.0, ((1, 0.3), (2, 0.4))),
    )

    first = build_candidate_stream(structure, (0, 1), proposals, seed=41)
    second = build_candidate_stream(structure, (0, 1), proposals, seed=41)

    assert first == second
    assert len(first) == CONTINUATION_FES
    assert {candidate.group for candidate in first} == {0, 1}
    assert sum(candidate.group == 0 for candidate in first) == CONTINUATION_FES // 2
    assert sum(candidate.group == 1 for candidate in first) == CONTINUATION_FES // 2
    for candidate in first:
        assert {variable for variable, _ in candidate.values} == set(
            structure.groups[candidate.group]
        )

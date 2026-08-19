from __future__ import annotations

from experiments.proposal_neighborhood_gate27 import _context


def test_gate27_context_is_complete_and_reproducible() -> None:
    first = _context("conflicting", "chain", 6, 32101)
    second = _context("conflicting", "chain", 6, 32101)

    assert first == second
    assert first.component_count >= 2
    assert first.proposals_identical is True
    assert first.fe_parity is True
    assert first.strict_best is True
    assert first.trace_complete is True
    assert first.proposal_neighborhood.continuation_fes == 32
    assert first.proposal_neighborhood.round_count == 26

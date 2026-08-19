from __future__ import annotations

from experiments.overlap_full_context_writeback_gate21 import _context


def test_gate21_full_context_has_equal_budget_and_trace() -> None:
    context = _context("conflicting", "chain", 6, 32101)

    assert context.component_count >= 2
    assert context.probes_identical is True
    assert context.proposals_identical is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert context.full_context.continuation_fes == 32
    assert context.full_context.consumed_fes == context.current.consumed_fes == 48
    assert len(context.full_context.accepted_rounds) <= 16


def test_gate21_is_reproducible() -> None:
    first = _context("conforming", "star", 12, 32102)
    second = _context("conforming", "star", 12, 32102)

    assert first == second

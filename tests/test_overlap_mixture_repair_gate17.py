from __future__ import annotations

from experiments.overlap_mixture_repair_gate17 import _context


def test_gate17_mixture_has_equal_budget_and_parity() -> None:
    context = _context("conflicting", "chain", 6, 31701)

    assert context.component_count >= 2
    assert context.probes_identical is True
    assert context.proposals_identical is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert context.mixture.continuation_fes == 32
    assert context.mixture.consumed_fes == context.current.consumed_fes == 48


def test_gate17_is_reproducible() -> None:
    first = _context("conforming", "star", 12, 31702)
    second = _context("conforming", "star", 12, 31702)

    assert first == second

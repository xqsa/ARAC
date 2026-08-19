from __future__ import annotations

from experiments.overlap_proposal_target_direction_gate19 import _context


def test_gate19_target_direction_has_equal_budget_and_trace() -> None:
    context = _context("conflicting", "chain", 6, 31901)
    assert context.component_count >= 2
    assert context.probes_identical is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert context.target.continuation_fes == 32
    assert context.target.consumed_fes == context.current.consumed_fes == 48
    assert len(context.target.radius_trace) == 16


def test_gate19_is_reproducible() -> None:
    first = _context("conforming", "star", 12, 31902)
    second = _context("conforming", "star", 12, 31902)
    assert first == second

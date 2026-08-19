from __future__ import annotations

from experiments.overlap_sequential_shared_patch_gate18 import _context


def test_gate18_sequential_has_equal_budget_and_trace() -> None:
    context = _context("conflicting", "chain", 6, 31801)

    assert context.component_count >= 2
    assert context.probes_identical is True
    assert context.proposals_identical is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert context.sequential.continuation_fes == 32
    assert context.sequential.consumed_fes == context.current.consumed_fes == 48
    assert len(context.sequential.radius_trace) == 16
    assert all(len(row) == len(context.sequential.radius_trace[0]) for row in context.sequential.radius_trace)


def test_gate18_is_reproducible() -> None:
    first = _context("conforming", "star", 12, 31802)
    second = _context("conforming", "star", 12, 31802)

    assert first == second

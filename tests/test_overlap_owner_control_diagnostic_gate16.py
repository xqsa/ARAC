from __future__ import annotations

from experiments.overlap_owner_control_diagnostic_gate16 import _context


def test_gate16_arms_have_equal_budget_and_parity() -> None:
    context = _context("conflicting", "chain", 6, 31601)

    assert context.component_count >= 2
    assert context.probe_parity is True
    assert context.proposals_identical is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert {context.current.consumed_fes, context.after_arbitration.consumed_fes, context.owner_shared_core.consumed_fes, context.owner_full.consumed_fes} == {48}


def test_gate16_is_reproducible() -> None:
    first = _context("conforming", "random", 12, 31602)
    second = _context("conforming", "random", 12, 31602)

    assert first == second

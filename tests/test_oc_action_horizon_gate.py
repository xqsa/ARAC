from __future__ import annotations

from experiments.oc_action_horizon_gate import WINDOW_COUNT, _context


def test_horizon_arms_chain_archive_with_equal_window_budgets() -> None:
    context = _context("conflicting", "chain", 6, 31601)

    assert context.component_count >= 2
    assert context.probes_identical is True
    assert context.proposals_identical is True
    assert context.boundary_parity is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert context.horizon_trace_complete is True
    for arm in (
        context.owner_control,
        context.shared_sequential,
        context.shared_joint,
    ):
        assert len(arm.windows) == WINDOW_COUNT
        assert arm.strict_best is True
        assert all(window.strict_best for window in arm.windows)
        assert {window.arbitration_fes for window in arm.windows} == {4}
        assert {window.action_fes for window in arm.windows} == {32}
        assert {window.handoff_fes for window in arm.windows} == {32}


def test_horizon_context_is_reproducible() -> None:
    first = _context("conforming", "random", 12, 31602)
    second = _context("conforming", "random", 12, 31602)

    assert first == second

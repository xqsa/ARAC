from __future__ import annotations

from experiments.overlap_value_aware_dispatch_gate15 import _context


def test_gate15_context_has_disconnected_components_and_equal_fe() -> None:
    context = _context("conflicting", "chain", 6, 31501)

    assert context.component_count >= 2
    assert context.probes_identical is True
    assert context.proposal_parity is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert context.value.probe_fes == 2 * context.component_count
    assert context.value.repair_or_control_fes == 32


def test_gate15_context_is_reproducible() -> None:
    first = _context("conforming", "star", 12, 31502)
    second = _context("conforming", "star", 12, 31502)

    assert first.value == second.value
    assert first.structural == second.structural
    assert first.value_selection_regret == second.value_selection_regret

from __future__ import annotations

from experiments.overlap_joint_patch_gate20 import _context


def test_gate20_joint_patch_has_equal_budget_and_trace() -> None:
    context = _context("conflicting", "chain", 6, 32001)
    assert context.component_count >= 2
    assert context.joint_variable_count >= context.shared_variable_count
    assert context.probes_identical is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert context.joint_patch.continuation_fes == 32
    assert context.joint_patch.consumed_fes == context.current.consumed_fes == 48
    assert len(context.joint_patch.radius_trace) == 16


def test_gate20_is_reproducible() -> None:
    first = _context("conforming", "star", 12, 32002)
    second = _context("conforming", "star", 12, 32002)
    assert first == second

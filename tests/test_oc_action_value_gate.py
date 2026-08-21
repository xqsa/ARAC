from __future__ import annotations

from experiments.oc_action_value_gate import _context


def test_action_value_arms_are_paired_and_handoff_is_complete() -> None:
    context = _context("conflicting", "chain", 6, 31601)

    assert context.component_count >= 2
    assert context.probes_identical is True
    assert context.proposals_identical is True
    assert context.fe_parity is True
    assert context.strict_best is True
    assert context.handoff_trace_complete is True
    assert {
        context.owner_control.consumed_fes,
        context.shared_sequential.consumed_fes,
        context.shared_joint.consumed_fes,
    } == {80}
    assert {
        context.owner_control.action_fes,
        context.shared_sequential.action_fes,
        context.shared_joint.action_fes,
    } == {32}
    assert {
        context.owner_control.handoff_fes,
        context.shared_sequential.handoff_fes,
        context.shared_joint.handoff_fes,
    } == {32}


def test_action_value_context_is_reproducible() -> None:
    first = _context("conforming", "random", 12, 31602)
    second = _context("conforming", "random", 12, 31602)

    assert first == second

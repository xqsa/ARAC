"""Integration tests for the gcb_coordinated overlap coordination mode."""

from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import ConflictLevel, GcbDispatchConfig, OverlapCoordinator
from arac.evidence import run_phase1_overlap_pilot
from arac.overlap_core import GCB_COORDINATED_MODE, run_overlap_from_pilot


def _overlap_problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        result += 0.25 * batch[:, 0] ** 2 * batch[:, 1] ** 2
        result += 0.25 * batch[:, 1] ** 2 * batch[:, 2] ** 2
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )


_PHASE1_KWARGS = {
    "anchors": ((-1.0,) * 4, (1.0,) * 4),
    "step": 0.25,
    "rounds": 8,
    "bucket_size": 2,
    "max_candidate_pairs": 16,
}


def test_gcb_coordinated_reaches_exact_terminal_budget_and_is_reproducible() -> None:
    problem = _overlap_problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=2_000,
        run_seed=101,
        **_PHASE1_KWARGS,
    )
    first = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=GCB_COORDINATED_MODE,
        refresh_cycles=3,
        neighborhood_fes=8,
    )
    second = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=GCB_COORDINATED_MODE,
        refresh_cycles=3,
        neighborhood_fes=8,
    )

    assert first == second
    assert first.terminal_fes == 2_000
    assert first.phase2_consumed_fes == 2_000 - pilot.checkpoint.phase1_fes
    assert all(cycle.best_error_after <= cycle.best_error_before for cycle in first.cycles)
    assert all(
        cycle.ctp_fes + cycle.neighborhood_fes == 8 * len(first.overlap_components)
        for cycle in first.cycles
    )
    assert all(
        receipt.consumed_fes == receipt.reserved_fes
        for cycle in first.cycles
        for receipt in cycle.dispatch_receipts
    )


def test_never_dispatching_planner_reproduces_proposal_neighborhood() -> None:
    problem = _overlap_problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=2_000,
        run_seed=103,
        **_PHASE1_KWARGS,
    )
    control = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode="proposal_neighborhood",
        refresh_cycles=3,
        neighborhood_fes=8,
    )
    coordinated = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=GCB_COORDINATED_MODE,
        refresh_cycles=3,
        neighborhood_fes=8,
        dispatch_config=GcbDispatchConfig(
            persistent_streak=10**6, escalation_streak=10**6 + 1
        ),
    )

    assert coordinated.final_error == control.final_error
    assert tuple(cycle.best_error_after for cycle in coordinated.cycles) == tuple(
        cycle.best_error_after for cycle in control.cycles
    )
    assert all(
        cycle.ctp_fes == 0 and not cycle.dispatch_receipts for cycle in coordinated.cycles
    )


def test_forced_persistent_conflict_dispatches_and_keeps_envelope_discipline(
    monkeypatch,
) -> None:
    problem = _overlap_problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=2_000,
        run_seed=107,
        **_PHASE1_KWARGS,
    )

    def always_high(self, residuals):
        return ConflictLevel.HIGH

    monkeypatch.setattr(OverlapCoordinator, "_level", always_high)
    result = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=GCB_COORDINATED_MODE,
        refresh_cycles=4,
        neighborhood_fes=8,
    )

    receipts = [
        receipt for cycle in result.cycles for receipt in cycle.dispatch_receipts
    ]
    assert receipts, "forced persistent conflict must produce dispatch receipts"
    assert all(receipt.consumed_fes == receipt.reserved_fes for receipt in receipts)
    assert all(receipt.reserved_fes <= 8 for receipt in receipts)
    dispatched_cycles = {receipt.cycle_index for receipt in receipts}
    assert all(result.cycles[index].ctp_fes > 0 for index in dispatched_cycles)
    assert result.terminal_fes == 2_000
    assert all(cycle.best_error_after <= cycle.best_error_before for cycle in result.cycles)
    for cycle in result.cycles:
        assert cycle.ctp_fes + cycle.neighborhood_fes == 8 * len(result.overlap_components)
    for receipt in receipts:
        assert receipt.best_error_after <= receipt.best_error_before

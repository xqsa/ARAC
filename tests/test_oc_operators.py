"""Unit tests for plan-driven operators (bounded windows, exact parity)."""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.contract import OperatorPlan, receipt_from_plan
from arac.coordination.operators import (
    AorOperator,
    CtpRestrictedOperator,
    SmpOperator,
    SmpSenseOperator,
    execute_bounded,
)
from arac.coordination.overlap import LocalProposal, OverlapCoordinator, OverlapStructure
from arac.runtime.ledger import BudgetExceededError, EvaluationLedger


def _problem() -> OptimizationProblem:
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


def _ledger(problem, budget: int = 4_000) -> EvaluationLedger:
    incumbent = np.zeros(problem.dimension)
    return EvaluationLedger(
        problem,
        budget,
        initial_incumbent=tuple(incumbent),
        initial_error=float(problem.objective(incumbent)),
    )


def _structure() -> OverlapStructure:
    return OverlapStructure(dimension=4, groups=((0, 1), (1, 2), (2, 3)))


def _proposals() -> tuple[LocalProposal, ...]:
    return (
        LocalProposal(
            group=0,
            values=((0, 0.2), (1, 0.4)),
            improvement=0.0,
            uncertainty=((0, 0.2), (1, 0.2)),
        ),
        LocalProposal(group=1, values=((1, -0.3), (2, 0.5)), improvement=0.0, uncertainty=((1, 0.2), (2, 0.2))),
        LocalProposal(group=2, values=((2, -0.4), (3, 0.1)), improvement=0.0, uncertainty=((2, 0.2), (3, 0.2))),
    )


def _plan(**overrides) -> OperatorPlan:
    base = dict(
        cycle_index=1,
        component=(0, 1, 2),
        scope=(1,),
        conflict_level="medium",
        action="ctp_restricted",
        reserved_fes=12,
        predicted_gain=0.5,
        seed=77,
        reason="medium_restricted_ctp",
        hub_degree=2,
        relative_hub=1.0,
    )
    base.update(overrides)
    return OperatorPlan(**base)


def test_execute_bounded_pads_remainder_to_exact_parity() -> None:
    problem = _problem()
    ledger = _ledger(problem)
    plan = _plan()
    start = ledger.count

    def executor() -> tuple[tuple[float, ...], ...]:
        ledger.evaluate(np.full((4, problem.dimension), 0.01))
        return ()

    execution = execute_bounded(plan, ledger, executor)
    assert execution.actual_fes == 12
    assert ledger.count - start == 12
    assert execution.best_error_after <= execution.best_error_before


def test_execute_bounded_guards_and_overconsumption() -> None:
    problem = _problem()
    ledger = _ledger(problem, budget=8)
    with pytest.raises(BudgetExceededError):
        execute_bounded(_plan(reserved_fes=16), ledger, lambda: ())
    tight = _ledger(problem, budget=4_000)
    with pytest.raises(RuntimeError, match="over-consumed"):

        def greedy() -> tuple[tuple[float, ...], ...]:
            tight.evaluate(np.zeros((16, problem.dimension)))
            return ()

        execute_bounded(_plan(), tight, greedy)


def test_ctp_restricted_operator_reaches_parity_and_feeds_receipt() -> None:
    problem = _problem()
    ledger = _ledger(problem)
    coordinator = OverlapCoordinator(_structure(), ledger)
    plan = _plan()
    execution = CtpRestrictedOperator().execute_plan(
        plan, coordinator=coordinator, proposals=_proposals()
    )
    assert execution.actual_fes == plan.reserved_fes
    receipt = receipt_from_plan(
        plan,
        actual_fes=execution.actual_fes,
        best_error_before=execution.best_error_before,
        best_error_after=execution.best_error_after,
        state_hash="0" * 64,
    )
    assert receipt.status in ("completed", "no_gain")


def test_repair_operator_forwards_the_gcb_scope(monkeypatch) -> None:
    problem = _problem()
    ledger = _ledger(problem)
    coordinator = OverlapCoordinator(_structure(), ledger)
    observed = []

    def record_dispatch(component, proposals, *, budget_fes, seed, strategy, scope=None):
        observed.append((tuple(component), tuple(scope or ()), strategy))
        ledger.evaluate(np.zeros((budget_fes, problem.dimension)))
        return budget_fes

    monkeypatch.setattr(coordinator, "dispatch_repair", record_dispatch)
    plan = _plan(scope=(1,), reserved_fes=12)
    CtpRestrictedOperator().execute_plan(plan, coordinator=coordinator, proposals=_proposals())
    assert observed == [((0, 1, 2), (1,), "sequential_coordinate_patch")]


def test_aor_operator_runs_the_declared_escalation_window() -> None:
    problem = _problem()
    ledger = _ledger(problem)
    coordinator = OverlapCoordinator(_structure(), ledger)
    plan = _plan(
        conflict_level="complex", action="aor", reason="complex_topology_aor", reserved_fes=16
    )
    execution = AorOperator().execute_plan(
        plan, coordinator=coordinator, proposals=_proposals()
    )
    assert execution.actual_fes == 16


def test_smp_operator_splits_reservation_across_groups() -> None:
    problem = _problem()
    ledger = _ledger(problem)
    coordinator = OverlapCoordinator(_structure(), ledger)
    plan = _plan(
        conflict_level="high", action="smp", reason="high_smp_trust_rebuild", reserved_fes=24
    )
    execution = SmpOperator(problem).execute_plan(plan, coordinator=coordinator)
    assert execution.actual_fes == 24
    assert len(execution.candidates) == 2


def test_smp_sense_bills_one_session_per_group() -> None:
    problem = _problem()
    ledger = _ledger(problem)
    structure = _structure()
    start = ledger.count
    runs = SmpSenseOperator().sense(
        structure,
        (0, 1),
        problem=problem,
        ledger=ledger,
        budget_fes_per_group=8,
        seed=11,
    )
    assert len(runs) == 2
    assert ledger.count - start == 16
    assert all(run.consumed_fes == 8 for run in runs)


def test_operator_exceptions_propagate_fail_closed(monkeypatch) -> None:
    problem = _problem()
    ledger = _ledger(problem)
    coordinator = OverlapCoordinator(_structure(), ledger)
    before = ledger.count

    def explode(*args, **kwargs):
        raise RuntimeError("primitive failure")

    monkeypatch.setattr(coordinator, "dispatch_repair", explode)
    with pytest.raises(RuntimeError, match="primitive failure"):
        CtpRestrictedOperator().execute_plan(_plan(), coordinator=coordinator, proposals=_proposals())
    assert ledger.count == before

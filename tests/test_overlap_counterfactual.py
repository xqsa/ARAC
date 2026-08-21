from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.counterfactual import (
    COUNTERFACTUAL_SCHEMA,
    evaluate_frozen_private_counterfactual,
)
from arac.coordination.loop import run_oc_unified_from_structure
from arac.coordination.overlap import OverlapStructure
from arac.runtime.ledger import EvaluationLedger


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum((batch - 1.0) ** 2, axis=1)
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=3,
        lower_bounds=(-2.0,) * 3,
        upper_bounds=(2.0,) * 3,
    )


def _loop_problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum((batch - 1.0) ** 2, axis=1)
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-2.0,) * 4,
        upper_bounds=(2.0,) * 4,
    )


def test_frozen_counterfactual_consumes_one_fe_and_preserves_archive() -> None:
    problem = _problem()
    ledger = EvaluationLedger(
        problem,
        total_budget=8,
        initial_count=1,
        initial_incumbent=(0.0, 0.0, 0.0),
        initial_error=3.0,
    )
    before = ledger.archive_snapshot()
    receipt = evaluate_frozen_private_counterfactual(
        ledger,
        component=(0, 1),
        scope=(1,),
        incumbent=(0.0, 0.0, 0.0),
        best_error_before=3.0,
        candidate_name="owner",
        candidate=(1.0, 1.0, 1.0),
        full_candidate_error=0.0,
    )

    assert receipt.schema_version == COUNTERFACTUAL_SCHEMA
    assert receipt.consumed_fes == 1
    assert receipt.full_gain == pytest.approx(3.0)
    assert receipt.frozen_gain == pytest.approx(2.0)
    assert receipt.coupled_gain == pytest.approx(1.0)
    assert receipt.archive_preserved is True
    assert ledger.count == 2
    assert ledger.archive_snapshot() == before


def test_frozen_counterfactual_rejects_unsorted_scope() -> None:
    problem = _problem()
    ledger = EvaluationLedger(
        problem,
        total_budget=8,
        initial_count=1,
        initial_incumbent=(0.0, 0.0, 0.0),
        initial_error=3.0,
    )
    with pytest.raises(ValueError, match="scope must be sorted"):
        evaluate_frozen_private_counterfactual(
            ledger,
            component=(0, 1),
            scope=(2, 1),
            incumbent=(0.0, 0.0, 0.0),
            best_error_before=3.0,
            candidate_name="owner",
            candidate=(1.0, 1.0, 1.0),
            full_candidate_error=0.0,
        )


def test_unified_loop_exposes_deterministic_counterfactual_receipt() -> None:
    problem = _loop_problem()
    structure = OverlapStructure(
        dimension=4,
        groups=((0, 1), (1, 2), (3,)),
        member_confidences=((1, 0, 0.8), (1, 1, 0.8)),
    )
    kwargs = dict(
        problem=problem,
        structure=structure,
        checkpoint_hash="0" * 64,
        total_budget_fes=2_000,
        phase1_fes=1,
        incumbent=(0.0, 0.0, 0.0, 0.0),
        incumbent_error=4.0,
        run_seed=1701,
        refresh_cycles=1,
        sense_budget_fes=8,
    )
    first = run_oc_unified_from_structure(**kwargs)
    second = run_oc_unified_from_structure(**kwargs)
    assert first == second
    assert first.cycles
    trace = first.cycles[0]
    assert trace.counterfactual_unavailable is False
    assert trace.counterfactual is not None
    assert trace.counterfactual.consumed_fes == 1
    assert trace.counterfactual.candidate_name != "incumbent"
    assert 3 <= trace.arbitration_fes <= 4
    assert trace.arbitration_fes >= trace.counterfactual.consumed_fes
    assert trace.best_error_after <= trace.best_error_before
    assert first.terminal_fes == 2_000

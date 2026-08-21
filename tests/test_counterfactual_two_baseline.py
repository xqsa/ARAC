from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import (
    TWO_BASELINE_COUNTERFACTUAL_SCHEMA,
    evaluate_two_baseline_counterfactual,
)
from arac.runtime.ledger import EvaluationLedger


def _problem(*, interaction: bool) -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        private_mean = 0.5 * (batch[:, 0] + batch[:, 2])
        result = (batch[:, 0] - 1.0) ** 2 + (batch[:, 2] - 1.0) ** 2
        if interaction:
            result = result + 2.0 * (private_mean - batch[:, 1]) ** 2
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=3,
        lower_bounds=(-2.0,) * 3,
        upper_bounds=(2.0,) * 3,
    )


def _ledger(problem: OptimizationProblem, initial_error: float) -> EvaluationLedger:
    return EvaluationLedger(
        problem,
        total_budget=8,
        initial_count=1,
        initial_incumbent=(0.0, 0.0, 0.0),
        initial_error=initial_error,
    )


def test_two_baseline_receipt_isolates_second_order_interaction() -> None:
    problem = _problem(interaction=True)
    ledger = _ledger(problem, 2.0)
    before = ledger.archive_snapshot()
    receipt = evaluate_two_baseline_counterfactual(
        ledger,
        component=(0, 1),
        scope=(1,),
        incumbent=(0.0, 0.0, 0.0),
        best_error_before=2.0,
        candidate_name="owner",
        candidate=(1.0, 1.0, 1.0),
        full_candidate_error=0.0,
    )

    assert receipt.schema_version == TWO_BASELINE_COUNTERFACTUAL_SCHEMA
    assert receipt.consumed_fes == 2
    assert receipt.private_candidate_error == pytest.approx(2.0)
    assert receipt.shared_candidate_error == pytest.approx(4.0)
    assert receipt.interaction_gain == pytest.approx(4.0)
    assert receipt.archive_preserved is True
    assert ledger.count == 3
    assert ledger.archive_snapshot() == before


def test_two_baseline_receipt_is_zero_for_additive_objective() -> None:
    problem = _problem(interaction=False)
    ledger = _ledger(problem, 2.0)
    receipt = evaluate_two_baseline_counterfactual(
        ledger,
        component=(0, 1),
        scope=(1,),
        incumbent=(0.0, 0.0, 0.0),
        best_error_before=2.0,
        candidate_name="owner",
        candidate=(1.0, 1.0, 1.0),
        full_candidate_error=0.0,
    )

    assert receipt.interaction_gain == pytest.approx(0.0)

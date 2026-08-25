"""Strict FE reservation tests for the soft-RDDSM discovery branch."""

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.soft_rddsm import (
    SoftDsmConfig,
    build_soft_dsm,
    discover_hierarchical_soft,
)
from arac.runtime.ledger import EvaluationLedger


DIMENSION = 4
SIGNATURE_PROBE_COUNT = 2
SIGNATURE_PROBE_SIZE = 1
SIGNATURE_FES = 1 + DIMENSION + SIGNATURE_PROBE_COUNT + DIMENSION * SIGNATURE_PROBE_COUNT + (
    SIGNATURE_PROBE_COUNT * SIGNATURE_PROBE_SIZE
)


def _problem() -> OptimizationProblem:
    def objective(values: np.ndarray) -> float | np.ndarray:
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _config(*, dsm_budget: int = 1) -> SoftDsmConfig:
    return SoftDsmConfig(
        dsm_budget=dsm_budget,
        # Keep the fixture focused on accounting.  A separable objective
        # should not pass this deliberately high interaction threshold.
        edge_threshold=0.9,
        confirm_threshold=0.95,
    )


def _discover(*, ledger: EvaluationLedger, budget_fes: int):
    return discover_hierarchical_soft(
        ledger.problem,
        ledger,
        run_seed=20260825,
        config=_config(),
        signature_probe_count=SIGNATURE_PROBE_COUNT,
        signature_probe_size=SIGNATURE_PROBE_SIZE,
        budget_fes=budget_fes,
    )


def test_discovery_rejects_reservation_before_signature_evaluations() -> None:
    problem = _problem()
    ledger = EvaluationLedger(problem, total_budget=100)

    with pytest.raises(ValueError, match="variable-signature stage"):
        _discover(ledger=ledger, budget_fes=SIGNATURE_FES - 1)

    # Failing the reservation must not consume a prefix of the signature
    # batch.  This keeps the caller's phase boundary auditable.
    assert ledger.count == 0


def test_discovery_rejects_reservation_above_ledger_headroom() -> None:
    problem = _problem()
    ledger = EvaluationLedger(problem, total_budget=20, initial_count=5)

    with pytest.raises(ValueError, match="ledger headroom"):
        _discover(ledger=ledger, budget_fes=16)

    assert ledger.count == 5


def test_discovery_receipt_reconciles_and_stays_inside_reservation() -> None:
    problem = _problem()
    ledger = EvaluationLedger(problem, total_budget=100, initial_count=5)
    budget_fes = SIGNATURE_FES + 1

    result = _discover(ledger=ledger, budget_fes=budget_fes)

    assert result.discovery_start_fes == 5
    assert result.discovery_start_fes + result.discovery_consumed_fes == result.discovery_end_fes
    assert result.discovery_end_fes == ledger.count
    assert result.discovery_consumed_fes <= budget_fes
    assert sum(fes for _stage, fes in result.level_budgets) == result.discovery_consumed_fes


def test_dsm_screen_requires_the_full_multi_anchor_batch() -> None:
    problem = _problem()
    signatures = np.ones((DIMENSION, SIGNATURE_PROBE_COUNT), dtype=float)
    config = _config(dsm_budget=7)

    ledger = EvaluationLedger(problem, total_budget=100)
    _edges, _candidates, consumed = build_soft_dsm(
        problem,
        ledger,
        signatures,
        config=config,
        budget_fes=7,
    )

    # Two anchors cost 2 * 4 FE.  A reservation of seven must not execute a
    # partial screen and must leave both the ledger and stage receipt intact.
    assert consumed == 0
    assert ledger.count == 0


def test_dsm_reservation_cannot_exceed_ledger_headroom() -> None:
    problem = _problem()
    ledger = EvaluationLedger(problem, total_budget=10)
    signatures = np.ones((DIMENSION, SIGNATURE_PROBE_COUNT), dtype=float)

    with pytest.raises(ValueError, match="ledger headroom"):
        build_soft_dsm(
            problem,
            ledger,
            signatures,
            config=_config(),
            budget_fes=11,
        )
    assert ledger.count == 0

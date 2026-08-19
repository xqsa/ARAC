from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import BudgetExceededError, EvaluationLedger


def _problem(dimension: int = 4) -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )


def test_ledger_counts_batches_and_keeps_a_strict_best_archive() -> None:
    ledger = EvaluationLedger(_problem(), 3)

    assert ledger.evaluate(np.array([2.0, 0.0, 0.0, 0.0])) == 4.0
    assert ledger.evaluate(np.array([[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])).tolist() == [1.0, 9.0]
    assert ledger.count == 3
    assert ledger.remaining == 0
    assert ledger.best_error == 1.0
    assert ledger.best_x.tolist() == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(BudgetExceededError):
        ledger.evaluate(np.zeros(4))


def test_checkpoint_is_identity_blind_and_hashes_all_method_state() -> None:
    checkpoint = PhaseCheckpoint(
        protocol="test-evidence-v1",
        run_seed=9,
        total_budget_fes=100,
        phase1_fes=10,
        incumbent=(1.0, 2.0, 3.0, 4.0),
        incumbent_error=30.0,
        feature_names=("roughness", "progress"),
        feature_values=(0.2, 0.7),
        blocks=((0, 2), (1, 3)),
        relations=(RelationEvidence(0, 1, 0.4, 0.1),),
    )
    changed = PhaseCheckpoint(
        **{**checkpoint.__dict__, "feature_values": (0.3, 0.7)}
    )

    names = {field.name for field in fields(PhaseCheckpoint)}
    assert not names & {"case_id", "family", "function_id"}
    assert checkpoint.remaining_fes == 90
    assert checkpoint.overlap_relation_count == 1
    assert checkpoint.checkpoint_hash != changed.checkpoint_hash


def test_ledger_can_resume_at_one_frozen_checkpoint() -> None:
    problem = _problem()
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=100,
        phase1_fes=10,
        incumbent=(1.0, 2.0, 3.0, 4.0),
        incumbent_error=30.0,
    )

    assert ledger.count == 10
    assert ledger.remaining == 90
    assert ledger.best_error == 30.0


def test_ledger_rejects_candidates_outside_public_bounds_without_counting_them() -> None:
    ledger = EvaluationLedger(_problem(), 1)

    with pytest.raises(ValueError, match="escaped the problem bounds"):
        ledger.evaluate(np.array([6.0, 0.0, 0.0, 0.0]))

    assert ledger.count == 0

from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import LocalProposal, OverlapCoordinator, OverlapStructure
from arac.runtime.ledger import EvaluationLedger


def _coordinator(*, budget: int = 20) -> tuple[OverlapCoordinator, tuple[LocalProposal, ...]]:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=3,
        lower_bounds=(-5.0,) * 3,
        upper_bounds=(5.0,) * 3,
    )
    ledger = EvaluationLedger(
        problem,
        total_budget=budget,
        initial_incumbent=(3.0, 3.0, 3.0),
        initial_error=27.0,
    )
    proposals = (
        LocalProposal(0, ((0, 0.0), (1, 1.0)), 2.0, ((0, 0.1), (1, 0.1))),
        LocalProposal(1, ((1, -1.0), (2, 0.0)), 1.0, ((1, 0.1), (2, 0.1))),
    )
    return OverlapCoordinator(OverlapStructure(3, ((0, 1), (1, 2))), ledger), proposals


def test_full_context_writeback_is_counted_sequential_and_strict_best() -> None:
    coordinator, proposals = _coordinator()

    result = coordinator.full_context_writeback((0, 1), proposals, rounds=4)

    assert result.consumed_fes == 8
    assert len(result.rounds) == 4
    assert tuple(item.group for item in result.rounds) == (0, 1, 0, 1)
    assert result.best_error_after <= result.best_error_before
    assert all(item.best_error_after <= item.best_error_before for item in result.rounds)
    assert coordinator.ledger.count == 8


def test_full_context_writeback_rejects_incomplete_proposals_before_spending_fe() -> None:
    coordinator, proposals = _coordinator()

    with pytest.raises(ValueError, match="exactly one overlap component"):
        coordinator.full_context_writeback((0, 1), proposals[:1], rounds=4)

    assert coordinator.ledger.count == 0


def test_proposal_neighborhood_writeback_is_reproducible_and_strict_best() -> None:
    coordinator, proposals = _coordinator(budget=32)

    result = coordinator.proposal_neighborhood_writeback(
        (0, 1), proposals, budget_fes=8, seed=41
    )

    assert result.consumed_fes == 8
    assert len(result.rounds) == 8
    assert tuple(item.group for item in result.rounds) == (0, 1, 0, 1, 0, 1, 0, 1)
    assert result.best_error_after <= result.best_error_before
    assert all(item.best_error_after <= item.best_error_before for item in result.rounds)

    second, second_proposals = _coordinator(budget=32)
    repeat = second.proposal_neighborhood_writeback(
        (0, 1), second_proposals, budget_fes=8, seed=41
    )
    assert result == repeat


def test_proposal_neighborhood_writeback_rejects_incomplete_proposals_before_spending_fe() -> None:
    coordinator, proposals = _coordinator(budget=20)

    with pytest.raises(ValueError, match="exactly one overlap component"):
        coordinator.proposal_neighborhood_writeback(
            (0, 1), proposals[:1], budget_fes=8, seed=3
        )

    assert coordinator.ledger.count == 0

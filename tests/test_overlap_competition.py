from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import LocalProposal, OverlapCoordinator, OverlapStructure
from arac.runtime.ledger import EvaluationLedger


def _problem(dimension: int = 3) -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )


def _structure() -> OverlapStructure:
    return OverlapStructure(dimension=3, groups=((0, 1), (1, 2)))


def _proposals() -> tuple[LocalProposal, ...]:
    return (
        LocalProposal(
            group=0,
            values=((0, -1.0), (1, -2.0)),
            improvement=2.0,
            uncertainty=((0, 0.1), (1, 0.1)),
        ),
        LocalProposal(
            group=1,
            values=((1, 2.0), (2, -1.0)),
            improvement=1.0,
            uncertainty=((1, 0.1), (2, 0.1)),
        ),
    )


def _ledger(total_budget: int = 20) -> EvaluationLedger:
    return EvaluationLedger(
        _problem(),
        total_budget=total_budget,
        initial_count=1,
        initial_incumbent=(2.0, 2.0, 2.0),
        initial_error=12.0,
    )


def test_competition_keeps_owner_specific_shared_values_without_averaging() -> None:
    ledger = _ledger()
    result = OverlapCoordinator(_structure(), ledger).duplicated_shared_competition(
        (0, 1), _proposals(), budget_fes=2
    )

    assert result.consumed_fes == 2
    assert result.active_scope == (1,)
    assert result.rounds[0].representatives == (0, 1)
    assert result.rounds[0].shared_signatures == ((-2.0,), (2.0,))
    assert result.rounds[0].accepted
    assert all(
        round_item.best_error_after <= round_item.best_error_before
        for round_item in result.rounds
    )
    assert result.best_error_after < result.best_error_before


@pytest.mark.parametrize("budget_fes", [1, 3, 7])
def test_competition_consumes_odd_and_even_budgets_exactly(budget_fes: int) -> None:
    ledger = _ledger(total_budget=20)
    result = OverlapCoordinator(_structure(), ledger).duplicated_shared_competition(
        (0, 1), _proposals(), budget_fes=budget_fes
    )

    assert result.consumed_fes == budget_fes
    assert ledger.count == 1 + budget_fes
    assert result.best_error_after <= result.best_error_before


def test_competition_scope_only_controls_shared_coordinates() -> None:
    ledger = _ledger()
    result = OverlapCoordinator(_structure(), ledger).duplicated_shared_competition(
        (0, 1), _proposals(), budget_fes=2, scope=(1,)
    )

    assert result.active_scope == (1,)
    assert all(len(item.shared_signatures[0]) == 1 for item in result.rounds)


def test_dispatch_exposes_competition_strategy_and_exact_budget() -> None:
    ledger = _ledger()
    consumed = OverlapCoordinator(_structure(), ledger).dispatch_repair(
        (0, 1), _proposals(), budget_fes=5, seed=17, strategy="duplicated_shared_competition"
    )

    assert consumed == 5
    assert ledger.count == 6
    assert ledger.best_error <= 12.0


def test_local_competition_is_seeded_and_records_nonzero_perturbations() -> None:
    first_ledger = _ledger()
    second_ledger = _ledger()
    first = OverlapCoordinator(_structure(), first_ledger).duplicated_shared_local_competition(
        (0, 1), _proposals(), budget_fes=7, seed=17
    )
    second = OverlapCoordinator(_structure(), second_ledger).duplicated_shared_local_competition(
        (0, 1), _proposals(), budget_fes=7, seed=17
    )

    assert first.consumed_fes == 7
    assert first.mutation_scale == 1.0
    assert any(
        norm > 0.0 for item in first.rounds for norm in item.perturbation_norms
    )
    assert first.rounds == second.rounds
    assert first.best_error_after <= first.best_error_before


def test_dispatch_exposes_local_competition_strategy() -> None:
    ledger = _ledger()
    consumed = OverlapCoordinator(_structure(), ledger).dispatch_repair(
        (0, 1),
        _proposals(),
        budget_fes=5,
        seed=17,
        strategy="duplicated_shared_local_competition",
    )

    assert consumed == 5
    assert ledger.count == 6
    assert ledger.best_error <= 12.0


def test_coordinate_can_trigger_competition_after_persistent_conflict() -> None:
    ledger = _ledger(total_budget=24)
    coordinator = OverlapCoordinator(
        _structure(), ledger, medium_threshold=0.0, high_threshold=0.0
    )

    first = coordinator.coordinate(
        (0, 1), _proposals(), ctp_budget_fes=7, ctp_strategy="duplicated_shared_competition"
    )
    second = coordinator.coordinate(
        (0, 1), _proposals(), ctp_budget_fes=7, ctp_strategy="duplicated_shared_competition"
    )

    assert first.ctp_triggered is False
    assert second.ctp_triggered is True
    assert second.ctp_consumed_fes == 7
    assert second.accepted_candidate in {
        None,
        "owner",
        "weighted_mean",
        "weighted_median",
        "ctp_shared_competition",
    }
    assert ledger.count == 16


def test_coordinate_can_trigger_local_competition_after_persistent_conflict() -> None:
    ledger = _ledger(total_budget=24)
    coordinator = OverlapCoordinator(
        _structure(), ledger, medium_threshold=0.0, high_threshold=0.0
    )

    coordinator.coordinate(
        (0, 1), _proposals(), ctp_budget_fes=7, ctp_strategy="duplicated_shared_local_competition"
    )
    result = coordinator.coordinate(
        (0, 1), _proposals(), ctp_budget_fes=7, ctp_strategy="duplicated_shared_local_competition"
    )

    assert result.ctp_triggered is True
    assert result.ctp_consumed_fes == 7
    assert ledger.count == 16


def test_competition_rejects_components_without_shared_variables() -> None:
    structure = OverlapStructure(dimension=2, groups=((0,), (1,)))
    ledger = EvaluationLedger(
        _problem(2),
        total_budget=5,
        initial_count=1,
        initial_incumbent=(1.0, 1.0),
        initial_error=2.0,
    )
    proposals = (
        LocalProposal(0, ((0, 0.0),), 1.0, ((0, 0.1),)),
        LocalProposal(1, ((1, 0.0),), 1.0, ((1, 0.1),)),
    )

    with pytest.raises(ValueError, match="shared variables"):
        OverlapCoordinator(structure, ledger).duplicated_shared_competition(
            (0, 1), proposals, budget_fes=1
        )

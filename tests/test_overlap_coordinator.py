from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import (
    ConflictLevel,
    LocalProposal,
    OverlapCoordinator,
    OverlapStructure,
    compute_proposal_residuals,
)
from arac.runtime.ledger import EvaluationLedger


def _problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=3,
        lower_bounds=(-5.0, -5.0, -5.0),
        upper_bounds=(5.0, 5.0, 5.0),
    )


def _structure() -> OverlapStructure:
    return OverlapStructure(
        dimension=3,
        groups=((0, 1), (1, 2)),
        member_confidences=((1, 0, 0.9), (1, 1, 0.8)),
    )


def _proposal(
    group: int,
    value: float,
    sigma: float,
    *,
    improvement: float = 1.0,
) -> LocalProposal:
    variables = ((0, 0.0), (1, value)) if group == 0 else ((1, value), (2, 0.0))
    uncertainty = tuple((variable, sigma) for variable, _ in variables)
    return LocalProposal(group, variables, improvement=improvement, uncertainty=uncertainty)


def _ledger() -> EvaluationLedger:
    problem = _problem()
    return EvaluationLedger(
        problem,
        total_budget=20,
        initial_count=1,
        initial_incumbent=(2.0, 2.0, 2.0),
        initial_error=12.0,
    )


def test_residual_separates_conforming_proposals_from_search_noise() -> None:
    structure = _structure()
    conforming = compute_proposal_residuals(
        structure,
        (_proposal(0, 0.50, 0.20), _proposal(1, 0.51, 0.20)),
    )[1]
    conflicting = compute_proposal_residuals(
        structure,
        (_proposal(0, -2.0, 0.05), _proposal(1, 2.0, 0.05)),
    )[1]

    assert conforming.between_variance < conforming.within_variance
    assert conforming.conflict_score < 1.0
    assert conflicting.between_variance > conflicting.within_variance
    assert conflicting.conflict_score > 10.0


def test_coordinator_evaluates_complete_candidates_and_preserves_strict_best() -> None:
    ledger = _ledger()
    coordinator = OverlapCoordinator(_structure(), ledger)

    result = coordinator.coordinate(
        (0, 1),
        (_proposal(0, -2.0, 0.05), _proposal(1, 2.0, 0.05)),
    )

    assert result.conflict_level is ConflictLevel.HIGH
    assert tuple(candidate.name for candidate in result.candidates) == (
        "incumbent",
        "owner",
        "weighted_mean",
        "weighted_median",
    )
    assert all(len(candidate.vector) == 3 for candidate in result.candidates)
    assert all(candidate.vector[0] == 0.0 for candidate in result.candidates[1:])
    assert all(candidate.vector[2] == 0.0 for candidate in result.candidates[1:])
    assert ledger.count == 5
    assert result.best_error_after <= result.best_error_before
    assert result.accepted
    assert result.accepted_candidate in {"owner", "weighted_mean", "weighted_median"}


def test_low_conflict_uses_only_the_cheap_consensus_path() -> None:
    ledger = _ledger()
    result = OverlapCoordinator(_structure(), ledger).coordinate(
        (0, 1),
        (_proposal(0, 0.50, 0.20), _proposal(1, 0.51, 0.20)),
    )

    assert result.conflict_level is ConflictLevel.LOW
    assert tuple(candidate.name for candidate in result.candidates) == (
        "incumbent",
        "owner",
        "weighted_mean",
    )
    assert ledger.count == 4


def test_owner_candidate_uses_largest_improvement_not_largest_residual_weight() -> None:
    coordinator = OverlapCoordinator(_structure(), _ledger())
    candidates = coordinator.candidates(
        (0, 1),
        np.asarray((2.0, 2.0, 2.0)),
        (
            _proposal(0, -1.0, 0.001, improvement=1.0),
            _proposal(1, 1.5, 1.0, improvement=2.0),
        ),
    )

    owner = next(candidate for candidate in candidates if candidate.name == "owner")
    assert owner.vector[1] == 1.5


def test_consensus_candidates_are_closed_to_problem_bounds() -> None:
    coordinator = OverlapCoordinator(_structure(), _ledger())
    candidates = coordinator.candidates(
        (0, 1),
        np.asarray((2.0, 2.0, 2.0)),
        (
            _proposal(0, 99.0, 0.001, improvement=1.0),
            _proposal(1, 99.0, 0.001, improvement=1.0),
        ),
    )
    assert all(-5.0 <= value <= 5.0 for candidate in candidates for value in candidate.vector)


def test_no_improving_candidate_keeps_the_incumbent_unaccepted() -> None:
    problem = _problem()
    ledger = EvaluationLedger(
        problem,
        total_budget=10,
        initial_count=1,
        initial_incumbent=(0.0, 0.0, 0.0),
        initial_error=0.0,
    )
    result = OverlapCoordinator(_structure(), ledger).coordinate(
        (0, 1),
        (_proposal(0, -2.0, 0.05), _proposal(1, 2.0, 0.05)),
    )

    assert result.accepted is False
    assert result.accepted_candidate is None
    assert result.best_error_before == result.best_error_after == 0.0
    assert tuple(ledger.best_x) == (0.0, 0.0, 0.0)


def test_ctp_requires_two_consecutive_high_conflict_observations() -> None:
    ledger = _ledger()
    coordinator = OverlapCoordinator(_structure(), ledger)
    proposals = (_proposal(0, -2.0, 0.05), _proposal(1, 2.0, 0.05))

    first = coordinator.coordinate((0, 1), proposals, ctp_budget_fes=7, ctp_seed=11)
    assert first.conflict_level is ConflictLevel.HIGH
    assert first.conflict_streak == 1
    assert first.ctp_triggered is False
    assert first.ctp_consumed_fes == 0
    assert ledger.count == 5

    second = coordinator.coordinate((0, 1), proposals, ctp_budget_fes=7, ctp_seed=11)
    assert second.conflict_level is ConflictLevel.HIGH
    assert second.conflict_streak == 2
    assert second.ctp_triggered is True
    assert second.ctp_consumed_fes == 7
    assert ledger.count == 16
    assert second.best_error_after <= second.best_error_before
    assert second.accepted_candidate in {
        None,
        "owner",
        "weighted_mean",
        "weighted_median",
        "ctp_shared_core",
    }


def test_joint_cmaes_ctp_consumes_exact_budget_and_preserves_strict_best() -> None:
    problem = _problem()
    ledger = EvaluationLedger(
        problem,
        total_budget=24,
        initial_count=1,
        initial_incumbent=(2.0, 2.0, 2.0),
        initial_error=12.0,
    )
    coordinator = OverlapCoordinator(_structure(), ledger)
    proposals = (_proposal(0, -2.0, 0.05), _proposal(1, 2.0, 0.05))

    first = coordinator.coordinate(
        (0, 1),
        proposals,
        ctp_budget_fes=8,
        ctp_seed=17,
        ctp_strategy="joint_cmaes",
    )
    second = coordinator.coordinate(
        (0, 1),
        proposals,
        ctp_budget_fes=8,
        ctp_seed=17,
        ctp_strategy="joint_cmaes",
    )

    assert first.ctp_triggered is False
    assert second.ctp_triggered is True
    assert second.ctp_consumed_fes == 8
    assert ledger.count == 17
    assert second.best_error_after <= second.best_error_before


def test_sequential_joint_patch_ctp_consumes_odd_budget_and_updates_boundary() -> None:
    problem = _problem()
    ledger = EvaluationLedger(
        problem,
        total_budget=24,
        initial_count=1,
        initial_incumbent=(2.0, 2.0, 2.0),
        initial_error=12.0,
    )
    coordinator = OverlapCoordinator(_structure(), ledger)
    proposals = (_proposal(0, -2.0, 0.05), _proposal(1, 2.0, 0.05))

    first = coordinator.coordinate(
        (0, 1), proposals, ctp_budget_fes=7, ctp_seed=17, ctp_strategy="sequential_joint_patch"
    )
    second = coordinator.coordinate(
        (0, 1), proposals, ctp_budget_fes=7, ctp_seed=17, ctp_strategy="sequential_joint_patch"
    )

    assert first.ctp_triggered is False
    assert second.ctp_triggered is True
    assert second.ctp_consumed_fes == 7
    assert ledger.count == 16
    assert second.best_error_after <= second.best_error_before
    # The patch owns the full component, including the unique boundary
    # coordinates, rather than sampling only the shared coordinate.
    assert tuple(ledger.best_x[[0, 2]]) != (2.0, 2.0)


def test_sequential_shared_patch_ctp_consumes_exact_budget_and_is_strict_best() -> None:
    problem = _problem()
    ledger = EvaluationLedger(
        problem,
        total_budget=24,
        initial_count=1,
        initial_incumbent=(2.0, 2.0, 2.0),
        initial_error=12.0,
    )
    coordinator = OverlapCoordinator(_structure(), ledger)
    proposals = (_proposal(0, -2.0, 0.05), _proposal(1, 2.0, 0.05))

    coordinator.coordinate(
        (0, 1), proposals, ctp_budget_fes=8, ctp_seed=17, ctp_strategy="sequential_shared_patch"
    )
    result = coordinator.coordinate(
        (0, 1), proposals, ctp_budget_fes=8, ctp_seed=17, ctp_strategy="sequential_shared_patch"
    )

    assert result.ctp_triggered is True
    assert result.ctp_consumed_fes == 8
    assert ledger.count == 17
    assert result.best_error_after <= result.best_error_before


def test_sequential_coordinate_patch_ctp_consumes_odd_budget_and_is_strict_best() -> None:
    problem = _problem()
    ledger = EvaluationLedger(
        problem,
        total_budget=24,
        initial_count=1,
        initial_incumbent=(2.0, 2.0, 2.0),
        initial_error=12.0,
    )
    coordinator = OverlapCoordinator(_structure(), ledger)
    proposals = (_proposal(0, -2.0, 0.05), _proposal(1, 2.0, 0.05))

    coordinator.coordinate(
        (0, 1), proposals, ctp_budget_fes=7, ctp_seed=17, ctp_strategy="sequential_coordinate_patch"
    )
    result = coordinator.coordinate(
        (0, 1), proposals, ctp_budget_fes=7, ctp_seed=17, ctp_strategy="sequential_coordinate_patch"
    )

    assert result.ctp_triggered is True
    assert result.ctp_consumed_fes == 7
    assert ledger.count == 16
    assert result.best_error_after <= result.best_error_before


def test_ctp_is_not_spent_for_low_conflict() -> None:
    ledger = _ledger()
    coordinator = OverlapCoordinator(_structure(), ledger)

    result = coordinator.coordinate(
        (0, 1),
        (_proposal(0, 0.50, 0.20), _proposal(1, 0.51, 0.20)),
        ctp_budget_fes=7,
    )

    assert result.conflict_level is ConflictLevel.LOW
    assert result.ctp_triggered is False
    assert result.ctp_consumed_fes == 0


def test_structure_rejects_duplicate_or_out_of_bounds_membership() -> None:
    with pytest.raises(ValueError, match="unique and in bounds"):
        OverlapStructure(dimension=2, groups=((0, 0), (1,)))
    with pytest.raises(ValueError, match="unique and in bounds"):
        OverlapStructure(dimension=2, groups=((0, 2), (1,)))
    with pytest.raises(ValueError, match="integers"):
        OverlapStructure(dimension=2, groups=((0.0,), (1,)))
    with pytest.raises(ValueError, match="cover every variable"):
        OverlapStructure(dimension=3, groups=((0,), (1,)))


def test_duplicate_group_proposals_are_rejected() -> None:
    proposal = _proposal(0, 0.0, 0.1)
    with pytest.raises(ValueError, match="one proposal per group"):
        compute_proposal_residuals(_structure(), (proposal, proposal))


def test_candidate_requires_each_component_proposal_to_cover_its_group() -> None:
    incomplete = LocalProposal(
        group=0,
        values=((1, 0.0),),
        improvement=1.0,
        uncertainty=((1, 0.1),),
    )
    with pytest.raises(ValueError, match="cover exactly"):
        OverlapCoordinator(_structure(), _ledger()).candidates(
            (0, 1),
            np.asarray((2.0, 2.0, 2.0)),
            (incomplete, _proposal(1, 0.0, 0.1)),
        )


def test_coordinator_rejects_dimension_mismatch() -> None:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=2,
        lower_bounds=(-1.0, -1.0),
        upper_bounds=(1.0, 1.0),
    )
    ledger = EvaluationLedger(
        problem,
        total_budget=4,
        initial_count=1,
        initial_incumbent=(0.5, 0.5),
        initial_error=0.5,
    )
    with pytest.raises(ValueError, match="dimensions disagree"):
        OverlapCoordinator(_structure(), ledger)

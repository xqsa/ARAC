from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import (
    ConflictLevel,
    GraphCoordinationScheduler,
    LocalProposal,
    OverlapCoordinator,
    OverlapStructure,
)
from arac.runtime.ledger import EvaluationLedger


def _ledger(dimension: int, *, total_budget: int = 100) -> EvaluationLedger:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )
    incumbent = (3.0,) * dimension
    return EvaluationLedger(
        problem,
        total_budget=total_budget,
        initial_incumbent=incumbent,
        initial_error=float(9 * dimension),
    )


def _proposal(
    group: int,
    variables: tuple[int, ...],
    shared: int,
    value: float,
    sigma: float,
) -> LocalProposal:
    values = tuple((variable, value if variable == shared else 0.0) for variable in variables)
    return LocalProposal(
        group=group,
        values=values,
        improvement=1.0,
        uncertainty=tuple((variable, sigma) for variable in variables),
    )


def test_gcb_prioritizes_larger_conflict_and_spends_one_bounded_ctp_event() -> None:
    structure = OverlapStructure(
        dimension=6,
        groups=((0, 1), (1, 2), (3, 4), (4, 5)),
    )
    proposals = (
        _proposal(0, (0, 1), 1, -2.0, 0.05),
        _proposal(1, (1, 2), 1, 2.0, 0.05),
        _proposal(2, (3, 4), 4, -1.0, 0.20),
        _proposal(3, (4, 5), 4, 1.0, 0.20),
    )
    coordinator = OverlapCoordinator(structure, _ledger(6))
    scheduler = GraphCoordinationScheduler(coordinator)

    scheduler.prime(proposals)
    priorities = scheduler.prioritize(proposals)
    assert tuple(item.component for item in priorities) == ((0, 1), (2, 3))
    assert all(item.conflict_level is ConflictLevel.HIGH for item in priorities)
    assert all(item.proposal_contribution == 1.0 for item in priorities)

    before = coordinator.ledger.count
    result = scheduler.dispatch(proposals, total_ctp_budget_fes=32, seed=19)

    assert len(result.events) == 1
    assert result.events[0].component == (0, 1)
    assert result.events[0].consumed_ctp_fes == 32
    assert result.events[0].ledger_fes == 36
    assert coordinator.ledger.count - before == 36
    assert result.consumed_ctp_fes == 32
    assert result.unspent_ctp_fes == 0
    assert result.events[0].best_error_after <= result.events[0].best_error_before


def test_hub_topology_breaks_equal_conflict_tie_by_overlap_load() -> None:
    hub_extent = float(np.sqrt(1.5))
    structure = OverlapStructure(
        dimension=7,
        groups=((0, 1), (1, 2), (1, 3), (4, 5), (5, 6)),
    )
    proposals = (
        _proposal(0, (0, 1), 1, -hub_extent, 0.10),
        _proposal(1, (1, 2), 1, 0.0, 0.10),
        _proposal(2, (1, 3), 1, hub_extent, 0.10),
        _proposal(3, (4, 5), 5, -1.0, 0.10),
        _proposal(4, (5, 6), 5, 1.0, 0.10),
    )
    scheduler = GraphCoordinationScheduler(
        OverlapCoordinator(structure, _ledger(7))
    )

    scheduler.prime(proposals)
    priorities = scheduler.prioritize(proposals)

    assert priorities[0].component == (0, 1, 2)
    assert priorities[0].overlap_load == 2
    assert priorities[0].topology_factor > priorities[1].topology_factor


def test_gcb_does_not_dispatch_ctp_to_low_conflict_components() -> None:
    structure = OverlapStructure(
        dimension=6,
        groups=((0, 1), (1, 2), (3, 4), (4, 5)),
    )
    proposals = (
        _proposal(0, (0, 1), 1, 0.50, 0.20),
        _proposal(1, (1, 2), 1, 0.51, 0.20),
        _proposal(2, (3, 4), 4, -0.50, 0.20),
        _proposal(3, (4, 5), 4, -0.49, 0.20),
    )
    coordinator = OverlapCoordinator(structure, _ledger(6))
    scheduler = GraphCoordinationScheduler(coordinator)
    scheduler.prime(proposals)
    before = coordinator.ledger.count

    result = scheduler.dispatch(proposals, total_ctp_budget_fes=32)

    assert result.events == ()
    assert result.consumed_ctp_fes == 0
    assert result.unspent_ctp_fes == 32
    assert coordinator.ledger.count == before

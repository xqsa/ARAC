from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import GraphCoordinationScheduler, LocalProposal, OverlapCoordinator, OverlapStructure
from arac.runtime.ledger import EvaluationLedger


GROUPS = ((0, 1), (1, 2), (3, 4), (4, 5))


def _scheduler(*, low_conflict: bool = False) -> tuple[GraphCoordinationScheduler, tuple[LocalProposal, ...]]:
    weights = np.asarray([1.0, 8.0, 1.0, 1.0, 1.0, 1.0])
    problem = OptimizationProblem(
        objective=lambda x: np.sum(weights * (np.asarray(x, dtype=float) - 0.25) ** 2, axis=-1),
        dimension=6,
        lower_bounds=(-4.0,) * 6,
        upper_bounds=(4.0,) * 6,
    )
    incumbent = (2.0,) * 6
    ledger = EvaluationLedger(
        problem,
        total_budget=100,
        initial_incumbent=incumbent,
        initial_error=float(problem.objective(np.asarray(incumbent))),
    )
    shared_values = (
        ((0.10, 0.11), (-0.10, -0.09))
        if low_conflict
        else ((-1.3, 1.3), (-1.3, 1.3))
    )
    proposals = []
    for group, variables in enumerate(GROUPS):
        component = 0 if group < 2 else 1
        owner = group % 2
        shared = 1 if component == 0 else 4
        values = tuple(
            (variable, shared_values[component][owner] if variable == shared else 0.0)
            for variable in variables
        )
        proposals.append(
            LocalProposal(
                group=group,
                values=values,
                improvement=1.0,
                uncertainty=tuple((variable, 0.08) for variable in variables),
            )
        )
    scheduler = GraphCoordinationScheduler(
        OverlapCoordinator(OverlapStructure(6, GROUPS), ledger)
    )
    return scheduler, tuple(proposals)


def test_value_probe_selects_larger_observed_objective_gain() -> None:
    scheduler, proposals = _scheduler()
    scheduler.prime(proposals)

    result = scheduler.dispatch_value_probe(
        proposals,
        total_ctp_budget_fes=32,
        seed=17,
    )

    gains = {item.component: item.estimated_gain for item in result.value_probes}
    assert len(result.value_probes) == 2
    assert gains[(0, 1)] > gains[(2, 3)]
    assert result.events[0].component == (0, 1)
    assert result.consumed_ctp_fes == 32
    assert result.ledger_fes == 48
    assert result.events[0].best_error_after <= result.events[0].best_error_before


def test_value_probe_control_sees_same_probes_and_total_budget() -> None:
    value_scheduler, proposals = _scheduler()
    value_scheduler.prime(proposals)
    value = value_scheduler.dispatch_value_probe(
        proposals,
        total_ctp_budget_fes=32,
        seed=19,
    )
    control_scheduler, control_proposals = _scheduler()
    control_scheduler.prime(control_proposals)
    control = control_scheduler.dispatch_value_probe(
        control_proposals,
        total_ctp_budget_fes=32,
        forced_component=(2, 3),
        seed=19,
    )

    assert value.value_probes == control.value_probes
    assert value.ledger_fes == control.ledger_fes == 48
    assert value.consumed_ctp_fes == control.consumed_ctp_fes == 32


def test_low_conflict_is_neither_probed_nor_repaired() -> None:
    scheduler, proposals = _scheduler(low_conflict=True)
    scheduler.prime(proposals)
    before = scheduler.coordinator.ledger.count

    result = scheduler.dispatch_value_probe(
        proposals,
        total_ctp_budget_fes=32,
    )

    assert result.value_probes == ()
    assert result.events == ()
    assert result.consumed_ctp_fes == 0
    assert scheduler.coordinator.ledger.count == before

"""Plan-driven operators: bounded-window adapters over frozen primitives.

Every operator executes exactly one :class:`OperatorPlan` reservation
through an existing, already-validated search primitive:

- ``SmpSenseOperator`` / ``SmpOperator`` -- owner-local proposal sessions
  (:func:`produce_local_proposal`); sensing and state-memory rebuild are
  the two SMP interfaces of design section 4.
- ``CtpRestrictedOperator`` / ``CtpSharedCoreOperator`` --
  :meth:`OverlapCoordinator.dispatch_repair` with the coordinate/joint
  patch strategies (the plan-execution primitive validated by Gates
  37-40).
- ``AorOperator`` -- a bounded full-space MMES correction window.

Contract semantics (arac-oc-operator-contract.md): exact FE parity with
the reservation, no implicit encroachment on other budget categories,
and exceptions propagate -- the operator layer never retries, never
switches actions, and never silently hands the remainder to AOR.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from arac.coordination.contract import (
    OC_ACTION_AOR,
    OC_ACTION_CTP_RESTRICTED,
    OC_ACTION_CTP_SHARED_CORE,
    OC_ACTION_SMP,
    OperatorPlan,
)
from arac.coordination.overlap import LocalProposal, OverlapCoordinator, OverlapStructure
from arac.coordination.proposals import LocalProposalRun, produce_local_proposal
from arac.runtime.ledger import BudgetExceededError, EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort

PROPOSAL_POPULATION_SIZE = 8
PROPOSAL_SIGMA = 0.5


@dataclass(frozen=True)
class OperatorExecution:
    """Raw outcome of one bounded operator window (pre-receipt)."""

    actual_fes: int
    best_error_before: float
    best_error_after: float
    candidates: tuple[tuple[float, ...], ...] = ()


def execute_bounded(
    plan: OperatorPlan,
    ledger: EvaluationLedger,
    executor: Callable[[], tuple[tuple[float, ...], ...]],
) -> OperatorExecution:
    """Run one reservation with exact parity; exceptions propagate.

    ``executor`` performs the search and returns the candidate solutions
    it produced.  Any remainder after the executor runs is filled by
    incumbent re-evaluations, which never degrade the strict-best
    archive, so ``actual_fes == plan.reserved_fes`` always holds on the
    normal path; the failed path is owned by the loop's fail-closed
    receipt (design section 2.2).
    """

    if not isinstance(plan, OperatorPlan):
        raise TypeError("plan must be an OperatorPlan")
    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be an EvaluationLedger")
    if plan.reserved_fes > ledger.remaining:
        raise BudgetExceededError(
            f"reservation {plan.reserved_fes} exceeds remaining {ledger.remaining}"
        )
    before = float(ledger.best_error)
    start = ledger.count
    candidates = executor()
    consumed = ledger.count - start
    if consumed > plan.reserved_fes:
        raise RuntimeError(
            f"operator over-consumed its reservation: {consumed} > {plan.reserved_fes}"
        )
    if consumed < plan.reserved_fes:
        fill = plan.reserved_fes - consumed
        ledger.evaluate(np.repeat(np.asarray(ledger.best_x)[np.newaxis, :], fill, axis=0))
    if ledger.count - start != plan.reserved_fes:
        raise RuntimeError("operator failed to reach exact FE parity")
    return OperatorExecution(
        actual_fes=ledger.count - start,
        best_error_before=before,
        best_error_after=float(ledger.best_error),
        candidates=tuple(candidates),
    )


class SmpSenseOperator:
    """SMP.sense: owner-local proposals that feed scope selection."""

    def sense(
        self,
        structure: OverlapStructure,
        groups: tuple[int, ...],
        *,
        problem,
        ledger: EvaluationLedger,
        budget_fes_per_group: int,
        seed: int,
    ) -> tuple[LocalProposalRun, ...]:
        anchor = tuple(float(value) for value in ledger.best_x)
        anchor_error = float(ledger.best_error)
        return tuple(
            produce_local_proposal(
                structure,
                group,
                problem=problem,
                global_ledger=ledger,
                anchor=anchor,
                anchor_error=anchor_error,
                budget_fes=budget_fes_per_group,
                seed=seed ^ (0x9E37 * (group + 1)),
                algorithm="sepcmaes",
                population_size=PROPOSAL_POPULATION_SIZE,
                sigma=PROPOSAL_SIGMA,
            )
            for group in groups
        )


class _RepairOperator:
    """Shared plan execution through OverlapCoordinator.dispatch_repair."""

    strategy: str = ""

    def execute_plan(
        self,
        plan: OperatorPlan,
        *,
        coordinator: OverlapCoordinator,
        proposals: tuple[LocalProposal, ...],
    ) -> OperatorExecution:
        if not self.strategy:
            raise TypeError("repair operators must define a strategy")

        def executor() -> tuple[tuple[float, ...], ...]:
            coordinator.dispatch_repair(
                plan.component,
                proposals,
                budget_fes=plan.reserved_fes,
                seed=plan.seed,
                strategy=self.strategy,
                scope=plan.scope,
            )
            return ()

        return execute_bounded(plan, coordinator.ledger, executor)


class CtpRestrictedOperator(_RepairOperator):
    """MEDIUM: restricted CTP local joint repair (coordinate patches)."""

    strategy = "sequential_coordinate_patch"


class CtpSharedCoreOperator(_RepairOperator):
    """HIGH: shared-core CTP joint optimization (joint patches)."""

    strategy = "sequential_joint_patch"


class AorOperator:
    """COMPLEX: one reserved-budget full-space AOR correction window."""

    def execute_plan(
        self,
        plan: OperatorPlan,
        *,
        coordinator: OverlapCoordinator,
        proposals: tuple[LocalProposal, ...] | None = None,
    ) -> OperatorExecution:
        del proposals
        ledger = coordinator.ledger

        def executor() -> tuple[tuple[float, ...], ...]:
            PypopOptimizerPort().run(
                "mmes",
                problem=ledger.problem,
                ledger=ledger,
                initial_mean=tuple(float(value) for value in ledger.best_x),
                sigma=0.5,
                seed=plan.seed,
                budget_fes=plan.reserved_fes,
                population_size=max(2, min(24, plan.reserved_fes)),
                restart=False,
            )
            return ()

        return execute_bounded(plan, ledger, executor)


class SmpOperator:
    """HIGH + trust decay: rebuild owner state memory inside the plan.

    Splits the reservation across the component's groups and runs one
    owner-local proposal session per group from the incumbent -- the
    bounded-window form of SMP's stateful block visits.
    """

    def __init__(self, problem, *, population_size: int = PROPOSAL_POPULATION_SIZE, sigma: float = PROPOSAL_SIGMA) -> None:
        self.problem = problem
        self.population_size = int(population_size)
        self.sigma = float(sigma)

    def execute_plan(
        self,
        plan: OperatorPlan,
        *,
        coordinator: OverlapCoordinator,
    ) -> OperatorExecution:
        ledger = coordinator.ledger
        selected_scope = set(plan.scope)
        groups = [
            group
            for group in plan.component
            if selected_scope.intersection(coordinator.structure.groups[group])
        ]
        if not groups:
            raise ValueError("SMP plan scope does not touch any component owner")
        base = plan.reserved_fes // len(groups)
        budgets = [base] * len(groups)
        budgets[-1] += plan.reserved_fes - base * len(groups)
        if min(budgets) < 1:
            raise ValueError("reservation too small to split across the component")

        def executor() -> tuple[tuple[float, ...], ...]:
            anchor = tuple(float(value) for value in ledger.best_x)
            anchor_error = float(ledger.best_error)
            runs = [
                produce_local_proposal(
                    coordinator.structure,
                    group,
                    problem=self.problem,
                    global_ledger=ledger,
                    anchor=anchor,
                    anchor_error=anchor_error,
                    budget_fes=budget,
                    seed=plan.seed ^ (0x9E37 * (group + 1)),
                    algorithm="sepcmaes",
                    population_size=self.population_size,
                    sigma=self.sigma,
                    variables=tuple(
                        variable
                        for variable in coordinator.structure.groups[group]
                        if variable in selected_scope
                    ),
                )
                for group, budget in zip(groups, budgets, strict=True)
            ]
            return tuple(tuple(run.best_x) for run in runs)

        return execute_bounded(plan, ledger, executor)


OC_OPERATORS: dict[str, type] = {
    OC_ACTION_SMP: SmpOperator,
    OC_ACTION_CTP_RESTRICTED: CtpRestrictedOperator,
    OC_ACTION_CTP_SHARED_CORE: CtpSharedCoreOperator,
    OC_ACTION_AOR: AorOperator,
}


__all__ = [
    "AorOperator",
    "CtpRestrictedOperator",
    "CtpSharedCoreOperator",
    "OC_OPERATORS",
    "OperatorExecution",
    "SmpOperator",
    "SmpSenseOperator",
    "execute_bounded",
]

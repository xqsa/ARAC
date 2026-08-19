from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import OverlapStructure, produce_local_proposal
from arac.runtime.ledger import EvaluationLedger
from experiments.phase1_overlap_real_proposals_gate9 import run_gate


def _problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=3,
        lower_bounds=(-5.0, -5.0, -5.0),
        upper_bounds=(5.0, 5.0, 5.0),
    )


def _structure() -> OverlapStructure:
    return OverlapStructure(dimension=3, groups=((0, 1), (1, 2)))


def _proposal_run(seed: int, *, total_budget: int = 80):
    problem = _problem()
    ledger = EvaluationLedger(
        problem,
        total_budget=total_budget,
        initial_count=1,
        initial_incumbent=(2.0, 2.0, 2.0),
        initial_error=12.0,
    )
    return produce_local_proposal(
        _structure(),
        0,
        problem=problem,
        global_ledger=ledger,
        anchor=(2.0, 2.0, 2.0),
        anchor_error=12.0,
        budget_fes=16,
        seed=seed,
        algorithm="sepcmaes",
        population_size=8,
    ), ledger


def test_real_proposal_is_deterministic_and_charged_to_global_ledger() -> None:
    first, first_ledger = _proposal_run(17)
    second, second_ledger = _proposal_run(17)

    assert first.algorithm == "sepcmaes"
    assert first.consumed_fes == 16
    assert first.global_end_fes - first.global_start_fes == 16
    assert first.proposal == second.proposal
    assert first.best_x == second.best_x
    assert first_ledger.count == second_ledger.count == 17


def test_real_proposal_covers_exact_group_and_metrics_are_valid() -> None:
    run, _ = _proposal_run(23)
    assert {variable for variable, _ in run.proposal.values} == {0, 1}
    assert run.proposal.improvement >= 0.0
    assert np.isfinite(run.proposal.improvement)
    assert all(np.isfinite(value) and value >= 0.0 for _, value in run.proposal.uncertainty)


def test_real_proposal_fails_closed_when_global_budget_cannot_pay() -> None:
    with pytest.raises(ValueError, match="global ledger remainder"):
        _proposal_run(5, total_budget=10)


def test_gate9_reconciles_real_proposal_and_coordination_fe() -> None:
    payload = run_gate()

    assert payload["gate_passed"]
    assert all(payload["gate_checks"].values())
    assert payload["proposal_fes"] == 4 * payload["proposal_budget_fes_each"]
    assert payload["phase2_fes_consumed"] == payload["proposal_fes"] + payload["coordination_fes"]
    assert payload["shared_variables"] == (2, 102)

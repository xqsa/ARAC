from __future__ import annotations

from experiments.oracle_overlap_gate1 import run_diagnostic, run_trial


def test_trial_accounts_for_every_objective_evaluation() -> None:
    trial = run_trial("conflicting", 17, sample_fes=64)

    assert trial.proposal_fes == 129
    assert trial.arbitration_fes == 4
    assert trial.total_fes == 133
    assert trial.owner_error >= 0.0
    assert trial.coordinated_error <= trial.uncoordinated_error


def test_small_paired_diagnostic_exposes_conflict_without_false_conforming_trigger() -> None:
    result = run_diagnostic((17, 23, 31), sample_fes=256, workers=1)

    conforming = result["summary"]["conforming"]
    conflicting = result["summary"]["conflicting"]
    assert conforming["median_conflict_score"] < conflicting["median_conflict_score"]
    assert conforming["strict_best_monotone"]
    assert conflicting["strict_best_monotone"]

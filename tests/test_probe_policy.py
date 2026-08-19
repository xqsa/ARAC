from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ACTION_NAMES, ActionContext, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.probe_policy import run_probe_commit_policy


def _context(events: list[tuple[float, ...]]) -> ActionContext:
    dimension = 6

    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        events.extend(tuple(float(value) for value in row) for row in batch)
        result = np.sum(batch**2, axis=1)
        return float(result[0]) if rows.ndim == 1 else result

    problem = OptimizationProblem(
        objective=objective,
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )
    checkpoint = PhaseCheckpoint(
        protocol="probe-policy-test-v1",
        run_seed=31,
        total_budget_fes=34,
        phase1_fes=4,
        incumbent=(1.0,) * dimension,
        incumbent_error=float(dimension),
        feature_names=("line_high_frequency_fraction_median",),
        feature_values=(0.4,),
        blocks=((0, 1), (2, 3), (4, 5)),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    return ActionContext("aor", checkpoint, problem, ledger, action_seed=41)


def test_probe_policy_charges_all_branches_and_exact_global_budget() -> None:
    events: list[tuple[float, ...]] = []
    context = _context(events)

    result = run_probe_commit_policy(
        context,
        global_total_fes=34,
        branch_probe_fes=6,
        decision_horizon_fes=3,
        exploration_floor_fes=6,
    )

    assert result.aggregate_fes == result.global_total_fes == 34
    assert len(events) + context.checkpoint.phase1_fes == 34
    assert result.action_schedule_total_fes == 34
    assert result.selected_ledger_fes == 16
    assert result.selected_action_fes == 12
    assert result.continuation_fes == 6
    assert tuple(action for action, _ in result.probe_final_errors) == ACTION_NAMES
    assert len(result.selected_state_hash) == 64


def test_probe_policy_marks_deterministic_cap_fallback() -> None:
    result = run_probe_commit_policy(
        _context([]),
        global_total_fes=34,
        branch_probe_fes=6,
        decision_horizon_fes=3,
        exploration_floor_fes=6,
        min_relative_margin=2.0,
    )

    assert result.decision.action_name is None
    assert result.commit_reason == "probe_cap_insufficient_margin"
    assert result.selected_action == min(
        result.probe_final_errors,
        key=lambda item: (item[1], item[0]),
    )[0]


def test_probe_policy_rejects_an_unaccounted_global_budget_before_evaluation() -> None:
    events: list[tuple[float, ...]] = []

    with pytest.raises(ValueError, match="shared checkpoint budget"):
        run_probe_commit_policy(
            _context(events),
            global_total_fes=52,
            branch_probe_fes=6,
            decision_horizon_fes=3,
            exploration_floor_fes=6,
        )

    assert events == []


def test_probe_policy_rejects_probes_that_exceed_phase2_before_evaluation() -> None:
    events: list[tuple[float, ...]] = []

    with pytest.raises(ValueError, match="fit inside"):
        run_probe_commit_policy(
            _context(events),
            global_total_fes=34,
            branch_probe_fes=8,
            decision_horizon_fes=4,
            exploration_floor_fes=8,
        )

    assert events == []

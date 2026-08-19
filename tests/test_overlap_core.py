from __future__ import annotations

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import OverlapCoordinator
from arac.evidence import run_phase1_overlap_pilot
from arac.overlap_core import PERSISTENT_CTP_MODE, run_overlap_arac, run_overlap_from_pilot


def _overlap_problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        result += 0.25 * batch[:, 0] ** 2 * batch[:, 1] ** 2
        result += 0.25 * batch[:, 1] ** 2 * batch[:, 2] ** 2
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )


def test_overlap_arac_reaches_exact_terminal_budget_and_is_reproducible() -> None:
    kwargs = {
        "total_budget_fes": 2_000,
        "run_seed": 73,
        "refresh_cycles": 2,
        "neighborhood_fes": 8,
        "phase1_kwargs": {
            "anchors": ((-1.0,) * 4, (1.0,) * 4),
            "step": 0.25,
            "rounds": 8,
            "bucket_size": 2,
            "max_candidate_pairs": 16,
        },
    }
    first = run_overlap_arac(_overlap_problem(), **kwargs)
    second = run_overlap_arac(_overlap_problem(), **kwargs)

    assert first == second
    assert first.terminal_fes == 2_000
    assert first.phase2_consumed_fes == 2_000 - first.phase1.checkpoint.phase1_fes
    assert first.final_error <= first.phase1.checkpoint.incumbent_error
    assert len(first.cycles) == 2
    assert all(item.best_error_after <= item.best_error_before for item in first.cycles)
    assert all(item.best_error_after_proposals <= item.best_error_before for item in first.cycles)
    assert all(item.proposal_gain >= 0.0 and item.coordination_gain >= 0.0 for item in first.cycles)


def test_overlap_arac_fails_closed_without_shared_variable_evidence() -> None:
    problem = OptimizationProblem(
        objective=lambda values: np.sum(np.asarray(values, dtype=float) ** 2, axis=-1),
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )

    with pytest.raises(RuntimeError, match="no shared-variable component"):
        run_overlap_arac(
            problem,
            total_budget_fes=2_000,
            run_seed=79,
            refresh_cycles=2,
            neighborhood_fes=8,
            phase1_kwargs={
                "anchors": ((-1.0,) * 4, (1.0,) * 4),
                "step": 0.25,
                "rounds": 8,
                "bucket_size": 2,
                "max_candidate_pairs": 16,
            },
        )


def test_overlap_phase2_arms_share_checkpoint_and_terminal_budget() -> None:
    problem = _overlap_problem()
    phase1_kwargs = {
        "anchors": ((-1.0,) * 4, (1.0,) * 4),
        "step": 0.25,
        "rounds": 8,
        "bucket_size": 2,
        "max_candidate_pairs": 16,
    }
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=2_000,
        run_seed=83,
        **phase1_kwargs,
    )

    results = tuple(
        run_overlap_from_pilot(
            problem,
            pilot,
            coordination_mode=mode,
            refresh_cycles=2,
            neighborhood_fes=8,
        )
        for mode in ("proposal_neighborhood", "proposal_only", "full_context")
    )

    assert {result.phase1.checkpoint.checkpoint_hash for result in results} == {
        pilot.checkpoint.checkpoint_hash
    }
    assert {result.terminal_fes for result in results} == {2_000}
    assert {result.phase2_consumed_fes for result in results} == {
        2_000 - pilot.checkpoint.phase1_fes
    }
    assert {result.proposal_budget_fes for result in results} == {
        results[0].proposal_budget_fes
    }
    assert all(result.final_error <= pilot.checkpoint.incumbent_error for result in results)
    proposal_only = next(
        result for result in results if result.coordination_mode == "proposal_only"
    )
    assert all(
        cycle.arbitration_fes == cycle.endpoint_fes == cycle.neighborhood_fes == 0
        for cycle in proposal_only.cycles
    )


def test_persistent_ctp_mode_reuses_one_coordinator_per_component(monkeypatch) -> None:
    problem = _overlap_problem()
    phase1_kwargs = {
        "anchors": ((-1.0,) * 4, (1.0,) * 4),
        "step": 0.25,
        "rounds": 8,
        "bucket_size": 2,
        "max_candidate_pairs": 16,
    }
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=2_000,
        run_seed=89,
        **phase1_kwargs,
    )
    instances = []
    original_init = OverlapCoordinator.__init__

    def recording_init(self, *args, **kwargs):
        instances.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(OverlapCoordinator, "__init__", recording_init)
    result = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=PERSISTENT_CTP_MODE,
        refresh_cycles=3,
        neighborhood_fes=8,
    )

    assert len(instances) == len(result.overlap_components)
    assert result.terminal_fes == 2_000
    assert result.phase2_consumed_fes == 2_000 - pilot.checkpoint.phase1_fes
    assert all(cycle.best_error_after <= cycle.best_error_before for cycle in result.cycles)
    assert all(cycle.ctp_fes >= 0 for cycle in result.cycles)
    assert all(cycle.max_conflict_streak >= 0 for cycle in result.cycles)


def test_persistent_ctp_fallback_reuses_proposal_neighborhood_seed(monkeypatch) -> None:
    problem = _overlap_problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=2_000,
        run_seed=97,
        anchors=((-1.0,) * 4, (1.0,) * 4),
        step=0.25,
        rounds=8,
        bucket_size=2,
        max_candidate_pairs=16,
    )
    seeds = []
    original = OverlapCoordinator.proposal_neighborhood_writeback

    def recording_writeback(self, *args, **kwargs):
        seeds.append(kwargs["seed"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        OverlapCoordinator,
        "proposal_neighborhood_writeback",
        recording_writeback,
    )
    control = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode="proposal_neighborhood",
        refresh_cycles=2,
        neighborhood_fes=8,
    )
    split = len(seeds)
    persistent = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=PERSISTENT_CTP_MODE,
        refresh_cycles=2,
        neighborhood_fes=8,
    )

    assert all(cycle.ctp_fes == 0 for cycle in persistent.cycles)
    assert tuple(seeds[:split]) == tuple(seeds[split:])
    assert persistent.final_error == control.final_error

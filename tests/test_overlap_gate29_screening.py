from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence import run_phase1_overlap_pilot
from arac.overlap_core import COORDINATION_MODES, run_overlap_from_pilot
from experiments.overlap_arac_gate29_screening import Cell, build_cell
from experiments.overlap_arac_gate29_screening import (
    ArmSummary,
    CellSummary,
    _read_cell,
    _write_cell,
)


def test_gate29_builder_is_sparse_and_truth_is_outside_algorithm_surface() -> None:
    problem, truth = build_cell(Cell("conflicting", "chain", 3, 20260829))

    assert problem.dimension == 1000
    assert truth.config.base_function == "rastrigin"
    assert len(truth.structure.groups) == 4
    assert len(truth.structure.shared_variables) == 3
    assert problem.objective(np.zeros(1000, dtype=float)) == problem.objective(np.zeros(1000, dtype=float))


def test_gate29_small_protocol_arms_share_checkpoint_and_exact_terminal() -> None:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        result += 0.25 * batch[:, 0] ** 2 * batch[:, 1] ** 2
        result += 0.25 * batch[:, 1] ** 2 * batch[:, 2] ** 2
        return float(result[0]) if rows.ndim == 1 else result

    problem = OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )
    total_budget = 2_000
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=total_budget,
        run_seed=20260830,
        anchors=((-1.0,) * 4, (1.0,) * 4),
        step=0.25,
        rounds=8,
        bucket_size=2,
        max_candidate_pairs=16,
    )
    results = tuple(
        run_overlap_from_pilot(
            problem,
            pilot,
            coordination_mode=mode,
            refresh_cycles=1,
            neighborhood_fes=8,
        )
        for mode in COORDINATION_MODES
    )
    assert {result.phase1.checkpoint.checkpoint_hash for result in results} == {
        pilot.checkpoint.checkpoint_hash
    }
    assert {result.proposal_budget_fes for result in results} == {results[0].proposal_budget_fes}
    assert all(result.terminal_fes == total_budget for result in results)
    assert all(result.phase2_consumed_fes == total_budget - pilot.checkpoint.phase1_fes for result in results)
    assert all(result.final_error <= pilot.checkpoint.incumbent_error for result in results)


def test_gate29_cell_artifact_round_trip(tmp_path) -> None:
    cell = Cell("conforming", "random", 3, 20260829)
    arms = tuple(
        ArmSummary(
            mode=mode,
            final_error=1.0,
            checkpoint_error=2.0,
            proposal_budget_fes=8,
            phase2_consumed_fes=1_760,
            terminal_fes=2_000,
            total_proposal_gain=0.5,
            total_coordination_gain=0.5 if mode == "proposal_neighborhood" else 0.0,
            strict_best=True,
            checkpoint_hash="a" * 64,
        )
        for mode in COORDINATION_MODES
    )
    row = CellSummary(
        cell=cell,
        truth_shared_count=3,
        truth_group_count=980,
        checkpoint_hash="a" * 64,
        phase1_consumed_fes=240,
        checkpoint_error=2.0,
        inferred_shared_count=3,
        inferred_component_count=1,
        arms=arms,
        checkpoint_parity=True,
        proposal_budget_parity=True,
        terminal_exact=True,
        strict_best=True,
    )

    _write_cell(tmp_path, row)

    assert _read_cell(tmp_path, cell) == row

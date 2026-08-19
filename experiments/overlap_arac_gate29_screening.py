"""Gate 29: paired multi-scenario screening for overlap-focused ARAC."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.benchmarks.overlap_objective import build_overlap_problem
from arac.overlap_core import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    COORDINATION_MODES,
    OverlapAracResult,
    run_overlap_from_pilot,
)
from arac.evidence import run_phase1_overlap_pilot


DIMENSION = 1000
ACTIVE_DIMENSION = 24
ACTIVE_GROUP_COUNT = 4
MIN_GROUP_SIZE = 5
MAX_GROUP_SIZE = 7
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
FRESH_SEED = 20260829
BOUNDS = 5.0
INTERACTION_STRENGTH = 0.25
PHASE1_KWARGS = {
    "anchor_count": 5,
    "step": 0.25,
    "rounds": 12,
    "bucket_size": 16,
    "max_candidate_pairs": 128,
}
MODES = ("conforming", "conflicting")
TOPOLOGIES = ("random", "chain", "star")
OVERLAP_BUDGETS = (3, 6)
CELL_SCHEMA = "arac-overlap-gate29-cell-v1-rastrigin"


@dataclass(frozen=True)
class Cell:
    mode: str
    topology: str
    overlap_budget: int
    seed: int


@dataclass(frozen=True)
class ArmSummary:
    mode: str
    final_error: float
    checkpoint_error: float
    proposal_budget_fes: int
    phase2_consumed_fes: int
    terminal_fes: int
    total_proposal_gain: float
    total_coordination_gain: float
    strict_best: bool
    checkpoint_hash: str


@dataclass(frozen=True)
class CellSummary:
    cell: Cell
    truth_shared_count: int
    truth_group_count: int
    checkpoint_hash: str
    phase1_consumed_fes: int
    checkpoint_error: float
    inferred_shared_count: int
    inferred_component_count: int
    arms: tuple[ArmSummary, ...]
    checkpoint_parity: bool
    proposal_budget_parity: bool
    terminal_exact: bool
    strict_best: bool


def build_cell(cell: Cell) -> tuple[OptimizationProblem, object]:
    """Build one sparse active-overlap problem and an offline truth audit object."""

    active_problem, truth = build_overlap_problem(
        ACTIVE_DIMENSION,
        overlap_budget=cell.overlap_budget,
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        num_groups=ACTIVE_GROUP_COUNT,
        base_function="rastrigin",
        conflict_mode=cell.mode,
        bounds=BOUNDS,
        contiguous=True,
        rotation=False,
        transforms=False,
        seed=cell.seed,
        topology=cell.topology,
        interaction_strength=INTERACTION_STRENGTH,
    )

    def objective(values: np.ndarray) -> float | np.ndarray:
        converted = np.asarray(values, dtype=float)
        single = converted.ndim == 1
        batch = converted[np.newaxis, :] if single else converted
        active = np.asarray(truth.evaluate(batch[:, :ACTIVE_DIMENSION]), dtype=float)
        tail = np.sum((batch[:, ACTIVE_DIMENSION:] / BOUNDS) ** 2, axis=1)
        result = active + tail
        return float(result[0]) if single else result

    problem = OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-BOUNDS,) * DIMENSION,
        upper_bounds=(BOUNDS,) * DIMENSION,
        optimum=0.0,
    )
    return problem, truth


def cells(*, seed: int = FRESH_SEED) -> tuple[Cell, ...]:
    return tuple(
        Cell(mode, topology, overlap_budget, seed)
        for mode in MODES
        for topology in TOPOLOGIES
        for overlap_budget in OVERLAP_BUDGETS
    )


def _pilot(problem: OptimizationProblem, cell: Cell):
    return run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=cell.seed,
        **PHASE1_KWARGS,
    )


def _arm_summary(result: OverlapAracResult) -> ArmSummary:
    return ArmSummary(
        mode=result.coordination_mode,
        final_error=result.final_error,
        checkpoint_error=result.phase1.checkpoint.incumbent_error,
        proposal_budget_fes=result.proposal_budget_fes,
        phase2_consumed_fes=result.phase2_consumed_fes,
        terminal_fes=result.terminal_fes,
        total_proposal_gain=sum(item.proposal_gain for item in result.cycles),
        total_coordination_gain=sum(item.coordination_gain for item in result.cycles),
        strict_best=all(item.best_error_after <= item.best_error_before for item in result.cycles),
        checkpoint_hash=result.phase1.checkpoint.checkpoint_hash,
    )


def run_cell(cell: Cell) -> CellSummary:
    problem, truth = build_cell(cell)
    pilot = _pilot(problem, cell)
    results = tuple(
        run_overlap_from_pilot(
            problem,
            pilot,
            coordination_mode=mode,
            refresh_cycles=DEFAULT_REFRESH_CYCLES,
            neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        )
        for mode in COORDINATION_MODES
    )
    arms = tuple(_arm_summary(result) for result in results)
    checkpoint_hashes = {arm.checkpoint_hash for arm in arms}
    return CellSummary(
        cell=cell,
        truth_shared_count=len(truth.structure.shared_variables),
        truth_group_count=len(truth.structure.groups) + (DIMENSION - ACTIVE_DIMENSION),
        checkpoint_hash=pilot.checkpoint.checkpoint_hash,
        phase1_consumed_fes=pilot.consumed_fes,
        checkpoint_error=pilot.checkpoint.incumbent_error,
        inferred_shared_count=sum(len(owners) > 1 for owners in pilot.evidence.memberships),
        inferred_component_count=len(results[0].overlap_components),
        arms=arms,
        checkpoint_parity=checkpoint_hashes == {pilot.checkpoint.checkpoint_hash},
        proposal_budget_parity=len({arm.proposal_budget_fes for arm in arms}) == 1,
        terminal_exact=all(
            arm.phase2_consumed_fes == TOTAL_BUDGET_FES - PHASE1_FES
            and arm.terminal_fes == TOTAL_BUDGET_FES
            for arm in arms
        ),
        strict_best=all(arm.strict_best for arm in arms),
    )


def _gain(rows: tuple[CellSummary, ...], left: str, right: str) -> np.ndarray:
    values = []
    for row in rows:
        arms = {arm.mode: arm for arm in row.arms}
        values.append(arms[right].final_error - arms[left].final_error)
    return np.asarray(values, dtype=float)


def _cell_path(directory: Path, cell: Cell) -> Path:
    return directory / (
        f"{cell.mode}_{cell.topology}_overlap{cell.overlap_budget}_seed{cell.seed}.json"
    )


def _write_cell(directory: Path, row: CellSummary) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = _cell_path(directory, row.cell)
    payload = {"schema_version": CELL_SCHEMA, "result": asdict(row)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_cell(directory: Path, cell: Cell) -> CellSummary | None:
    path = _cell_path(directory, cell)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CELL_SCHEMA:
        return None
    result = payload["result"]
    loaded_cell = Cell(**result.pop("cell"))
    arms = tuple(ArmSummary(**arm) for arm in result.pop("arms"))
    row = CellSummary(cell=loaded_cell, arms=arms, **result)
    if row.cell != cell:
        raise RuntimeError(f"cell artifact identity drifted: {path}")
    return row


def run_gate(
    *,
    workers: int = 1,
    seed: int = FRESH_SEED,
    cell_dir: Path | None = None,
) -> dict[str, object]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    jobs = cells(seed=seed)
    completed = []
    pending = []
    for cell in jobs:
        loaded = None if cell_dir is None else _read_cell(cell_dir, cell)
        if loaded is None:
            pending.append(cell)
        else:
            completed.append(loaded)
    if not pending:
        pass
    elif workers == 1:
        for cell in pending:
            row = run_cell(cell)
            completed.append(row)
            if cell_dir is not None:
                _write_cell(cell_dir, row)
            print(f"completed {cell.mode}/{cell.topology}/overlap={cell.overlap_budget}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(run_cell, cell): cell for cell in pending}
            for future in as_completed(futures):
                cell = futures[future]
                row = future.result()
                completed.append(row)
                if cell_dir is not None:
                    _write_cell(cell_dir, row)
                print(
                    f"completed {cell.mode}/{cell.topology}/overlap={cell.overlap_budget}",
                    flush=True,
                )
    rows = tuple(completed)
    rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.cell.mode,
                row.cell.topology,
                row.cell.overlap_budget,
                row.cell.seed,
            ),
        )
    )
    neighborhood_vs_proposal = _gain(rows, "proposal_neighborhood", "proposal_only")
    neighborhood_vs_full = _gain(rows, "proposal_neighborhood", "full_context")
    coordination = np.asarray(
        [
            next(arm for arm in row.arms if arm.mode == "proposal_neighborhood").total_coordination_gain
            for row in rows
        ],
        dtype=float,
    )
    cells_by_mode = {
        mode: tuple(row for row in rows if row.cell.mode == mode) for mode in MODES
    }
    checks = {
        "cell_count_12": len(rows) == 12,
        "phase1_exact": all(row.phase1_consumed_fes == PHASE1_FES for row in rows),
        "checkpoint_parity": all(row.checkpoint_parity for row in rows),
        "proposal_budget_parity": all(row.proposal_budget_parity for row in rows),
        "terminal_exact": all(row.terminal_exact for row in rows),
        "strict_best": all(row.strict_best for row in rows),
        "inferred_components_present": all(row.inferred_component_count > 0 for row in rows),
        "neighborhood_vs_proposal_win_tie_ge_0_60": float(np.mean(neighborhood_vs_proposal >= 0.0)) >= 0.60,
        "neighborhood_vs_full_win_tie_ge_0_60": float(np.mean(neighborhood_vs_full >= 0.0)) >= 0.60,
        "median_gain_vs_proposal_positive": float(np.median(neighborhood_vs_proposal)) > 0.0,
        "median_gain_vs_full_positive": float(np.median(neighborhood_vs_full)) > 0.0,
        "conforming_coordination_nonzero": any(
            value > 0.0
            for value, row in zip(coordination, rows, strict=True)
            if row.cell.mode == "conforming"
        ),
        "conflicting_coordination_nonzero": any(
            value > 0.0
            for value, row in zip(coordination, rows, strict=True)
            if row.cell.mode == "conflicting"
        ),
    }
    return {
        "schema_version": "arac-overlap-gate29-screening-v1",
        "protocol": {
            "dimension": DIMENSION,
            "active_dimension": ACTIVE_DIMENSION,
            "active_group_count": ACTIVE_GROUP_COUNT,
            "total_budget_fes": TOTAL_BUDGET_FES,
            "phase1_fes": PHASE1_FES,
            "phase2_fes": TOTAL_BUDGET_FES - PHASE1_FES,
            "refresh_cycles": DEFAULT_REFRESH_CYCLES,
            "neighborhood_fes": DEFAULT_NEIGHBORHOOD_FES,
            "seed": seed,
            "phase1_kwargs": PHASE1_KWARGS,
        },
        "summary": {
            "neighborhood_vs_proposal_win_or_tie": float(np.mean(neighborhood_vs_proposal >= 0.0)),
            "neighborhood_vs_proposal_median_gain": float(np.median(neighborhood_vs_proposal)),
            "neighborhood_vs_full_win_or_tie": float(np.mean(neighborhood_vs_full >= 0.0)),
            "neighborhood_vs_full_median_gain": float(np.median(neighborhood_vs_full)),
            "conforming_coordination_gain_max": float(
                max(
                    arm.total_coordination_gain
                    for row in cells_by_mode["conforming"]
                    for arm in row.arms
                    if arm.mode == "proposal_neighborhood"
                )
            ),
            "conflicting_coordination_gain_max": float(
                max(
                    arm.total_coordination_gain
                    for row in cells_by_mode["conflicting"]
                    for arm in row.arms
                    if arm.mode == "proposal_neighborhood"
                )
            ),
        },
        "cells": [asdict(row) for row in rows],
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=FRESH_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_arac_gate29_screening/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate(
        workers=args.workers,
        seed=args.seed,
        cell_dir=args.output.parent / "cells",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

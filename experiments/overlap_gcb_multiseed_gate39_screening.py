"""Gate 39: paired multi-seed confirmation of the relative-hub coordinator."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from arac.coordination import GcbDispatchConfig
from arac.evidence import run_phase1_overlap_pilot
from arac.overlap_core import (
    GCB_COORDINATED_MODE,
    PERSISTENT_CTP_MODE,
    run_overlap_from_pilot,
)
from experiments.overlap_arac_gate29_screening import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    OVERLAP_BUDGETS,
    PHASE1_FES,
    PHASE1_KWARGS,
    TOTAL_BUDGET_FES,
    TOPOLOGIES,
    Cell,
    build_cell,
)
from experiments.overlap_coordinate_ctp_gate36_fair_screening import TOLERANCE
from experiments.overlap_gcb_coordinator_gate37_screening import _dispatch_summary

SEEDS = (20260830, 20260831)
CELL_SCHEMA = "arac-overlap-gcb-multiseed-gate39-cell-v1"
OUTPUT_SCHEMA = "arac-overlap-gcb-multiseed-gate39-screen-v1"
DISPATCH_CONFIG = GcbDispatchConfig(hub_mode="relative", complex_hub_ratio=0.9)


def jobs() -> tuple[Cell, ...]:
    return tuple(
        Cell("conflicting", topology, overlap, seed)
        for topology in TOPOLOGIES
        for overlap in OVERLAP_BUDGETS
        for seed in SEEDS
    )


def _cell_path(directory: Path, cell: Cell) -> Path:
    return directory / (
        f"{cell.mode}_{cell.topology}_overlap{cell.overlap_budget}_seed{cell.seed}.json"
    )


def _arm_result(result) -> dict[str, object]:
    return {
        "final_error": result.final_error,
        "terminal_fes": result.terminal_fes,
        "strict_best": all(
            cycle.best_error_after <= cycle.best_error_before for cycle in result.cycles
        ),
    }


def run_cell(cell: Cell) -> dict[str, object]:
    problem, truth = build_cell(cell)
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=cell.seed,
        **PHASE1_KWARGS,
    )
    arms: dict[str, object] = {}
    dispatch_summary = None
    envelope_ok = True
    consumption_ok = True
    for mode, coordination_mode, config in (
        ("coordinator", GCB_COORDINATED_MODE, DISPATCH_CONFIG),
        ("proposal_neighborhood", "proposal_neighborhood", None),
        ("persistent_ctp", PERSISTENT_CTP_MODE, None),
    ):
        result = run_overlap_from_pilot(
            problem,
            pilot,
            coordination_mode=coordination_mode,
            refresh_cycles=DEFAULT_REFRESH_CYCLES,
            neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
            dispatch_config=config,
        )
        arms[mode] = _arm_result(result)
        if mode == "coordinator":
            dispatch_summary = _dispatch_summary(result)
            envelope_total = DEFAULT_NEIGHBORHOOD_FES * len(result.overlap_components)
            envelope_ok = all(
                cycle.ctp_fes + cycle.neighborhood_fes == envelope_total
                for cycle in result.cycles
            )
            consumption_ok = all(
                receipt.consumed_fes == receipt.reserved_fes
                for cycle in result.cycles
                for receipt in cycle.dispatch_receipts
            )
    return {
        "cell": asdict(cell),
        "truth_shared_count": len(truth.structure.shared_variables),
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "phase1_fes": pilot.consumed_fes,
        "arms": arms,
        "dispatch": dispatch_summary,
        "envelope_no_encroachment": envelope_ok,
        "dispatch_consumption_parity": consumption_ok,
    }


def _write_cell(directory: Path, row: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cell = Cell(**row["cell"])
    _cell_path(directory, cell).write_text(
        json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_cell(directory: Path, cell: Cell) -> dict[str, object] | None:
    path = _cell_path(directory, cell)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CELL_SCHEMA:
        return None
    row = payload["result"]
    if row["cell"] != asdict(cell):
        raise RuntimeError(f"cell artifact identity drifted: {path}")
    return row


def run_gate(*, workers: int, cell_dir: Path) -> dict[str, object]:
    completed = []
    pending = []
    for cell in jobs():
        row = _read_cell(cell_dir, cell)
        if row is None:
            pending.append(cell)
        else:
            completed.append(row)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        for cell in pending:
            row = run_cell(cell)
            _write_cell(cell_dir, row)
            completed.append(row)
            print(f"completed {cell.topology}/ov={cell.overlap_budget}/seed={cell.seed}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(run_cell, cell): cell for cell in pending}
            for future in as_completed(futures):
                cell = futures[future]
                row = future.result()
                _write_cell(cell_dir, row)
                completed.append(row)
                print(
                    f"completed {cell.topology}/ov={cell.overlap_budget}/seed={cell.seed}",
                    flush=True,
                )
    rows = tuple(
        sorted(
            completed,
            key=lambda row: (
                row["cell"]["topology"],
                row["cell"]["overlap_budget"],
                row["cell"]["seed"],
            ),
        )
    )

    def _gain(row, reference):
        return row["arms"][reference]["final_error"] - row["arms"]["coordinator"]["final_error"]

    proposal_gain = np.asarray([_gain(row, "proposal_neighborhood") for row in rows], dtype=float)
    ctp_gain = np.asarray([_gain(row, "persistent_ctp") for row in rows], dtype=float)
    star_rows = [row for row in rows if row["cell"]["topology"] == "star"]
    chain3_rows = [
        row
        for row in rows
        if row["cell"]["topology"] == "chain" and row["cell"]["overlap_budget"] == 3
    ]
    star_gains = [_gain(row, "proposal_neighborhood") for row in star_rows]
    chain3_gains = [_gain(row, "proposal_neighborhood") for row in chain3_rows]
    protocol_checks = {
        "cell_count_12": len(rows) == 12,
        "phase1_exact": all(row["phase1_fes"] == PHASE1_FES for row in rows),
        "terminal_exact": all(
            arm["terminal_fes"] == TOTAL_BUDGET_FES
            for row in rows
            for arm in row["arms"].values()
        ),
        "strict_best": all(
            arm["strict_best"] for row in rows for arm in row["arms"].values()
        ),
        "envelope_no_encroachment": all(row["envelope_no_encroachment"] for row in rows),
        "dispatch_consumption_parity": all(row["dispatch_consumption_parity"] for row in rows),
    }
    screening_checks = {
        "star_no_regression_all": all(gain >= -TOLERANCE for gain in star_gains),
        "chain_ov3_positive_both_seeds": all(gain > TOLERANCE for gain in chain3_gains),
        "win_or_tie_vs_proposal_ge_0_75": float(np.mean(proposal_gain >= -TOLERANCE)) >= 0.75,
        "not_worse_than_persistent_ctp_all": bool(np.all(ctp_gain >= -TOLERANCE)),
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cell_count": 12,
            "workers": workers,
            "seeds": SEEDS,
            "total_budget_fes": TOTAL_BUDGET_FES,
            "phase1_fes": PHASE1_FES,
            "neighborhood_fes": DEFAULT_NEIGHBORHOOD_FES,
            "refresh_cycles": DEFAULT_REFRESH_CYCLES,
            "dispatch_config": asdict(DISPATCH_CONFIG),
            "comparison_tolerance": TOLERANCE,
            "note": "baselines computed fresh per cell (paired), not from frozen seed-20260829 artifacts",
        },
        "summary": {
            "vs_proposal_win_or_tie": float(np.mean(proposal_gain >= -TOLERANCE)),
            "vs_proposal_min_gain": float(np.min(proposal_gain)),
            "vs_persistent_ctp_min_gain": float(np.min(ctp_gain)),
            "star_gains": [float(value) for value in star_gains],
            "chain3_gains": [float(value) for value in chain3_gains],
            "dispatch_total_count": sum(row["dispatch"]["dispatch_count"] for row in rows),
            "dispatch_total_fes": sum(row["dispatch"]["dispatch_fes"] for row in rows),
        },
        "protocol_checks": protocol_checks,
        "screening_checks": screening_checks,
        "gate_passed": all(protocol_checks.values()) and all(screening_checks.values()),
        "cells": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cell-dir",
        type=Path,
        default=Path("artifacts/overlap_gcb_multiseed_gate39/cells"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_gcb_multiseed_gate39/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate(workers=args.workers, cell_dir=args.cell_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_passed": payload["gate_passed"],
                "protocol_checks": payload["protocol_checks"],
                "screening_checks": payload["screening_checks"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

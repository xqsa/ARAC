"""Gate 37: pre-registered GCB dispatch coordinator vs frozen Gate 29/36 arms."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from arac.coordination import GcbDispatchConfig
from arac.evidence import run_phase1_overlap_pilot
from arac.overlap_core import GCB_COORDINATED_MODE, run_overlap_from_pilot
from experiments.overlap_arac_gate29_screening import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    PHASE1_FES,
    PHASE1_KWARGS,
    TOTAL_BUDGET_FES,
    Cell,
    build_cell,
    cells,
)

BASELINE = Path("artifacts/overlap_arac_gate29_screening/confirmation_fresh.json")
FIXED_CTP = Path("artifacts/overlap_coordinate_ctp_gate36/confirmation_fresh.json")
CELL_SCHEMA = "arac-overlap-gcb-coordinator-gate37-cell-v1"
OUTPUT_SCHEMA = "arac-overlap-gcb-coordinator-gate37-screen-v1"
TOLERANCE = 1e-9
DISPATCH_CONFIG = GcbDispatchConfig()


def _cell_path(directory: Path, cell: Cell) -> Path:
    return directory / (
        f"{cell.mode}_{cell.topology}_overlap{cell.overlap_budget}_seed{cell.seed}.json"
    )


def _row_by_cell(payload: dict[str, object], cell: Cell) -> dict[str, object]:
    return next(
        row for row in payload["cells"] if row["cell"] == asdict(cell)
    )


def _baseline_row(cell: Cell) -> dict[str, object]:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    return _row_by_cell(payload, cell)


def _fixed_ctp_row(cell: Cell) -> dict[str, object]:
    payload = json.loads(FIXED_CTP.read_text(encoding="utf-8"))
    return _row_by_cell(payload, cell)


def _dispatch_summary(result) -> dict[str, object]:
    receipts = [
        receipt for cycle in result.cycles for receipt in cycle.dispatch_receipts
    ]
    action_counts = Counter(receipt.action for receipt in receipts)
    reason_counts = Counter(receipt.reason for receipt in receipts)
    gained = [receipt for receipt in receipts if receipt.gained]
    return {
        "dispatch_count": len(receipts),
        "dispatch_fes": sum(receipt.consumed_fes for receipt in receipts),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "gained_count": len(gained),
        "gained_fes": sum(receipt.consumed_fes for receipt in gained),
        "escalations": action_counts.get("joint_cmaes_escalation", 0),
        "max_hub_degree": max(
            (receipt.hub_degree for receipt in receipts), default=0
        ),
    }


def run_cell(cell: Cell) -> dict[str, object]:
    problem, truth = build_cell(cell)
    baseline = _baseline_row(cell)
    fixed_ctp = _fixed_ctp_row(cell)
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=cell.seed,
        **PHASE1_KWARGS,
    )
    result = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=GCB_COORDINATED_MODE,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        dispatch_config=DISPATCH_CONFIG,
    )
    frozen_arms = {
        arm["mode"]: {
            "final_error": arm["final_error"],
            "proposal_budget_fes": arm["proposal_budget_fes"],
            "terminal_fes": arm["terminal_fes"],
            "strict_best": arm["strict_best"],
        }
        for arm in baseline["arms"]
        if arm["mode"] in {"proposal_neighborhood", "full_context"}
    }
    envelope_total = DEFAULT_NEIGHBORHOOD_FES * len(result.overlap_components)
    return {
        "cell": asdict(cell),
        "truth_shared_count": len(truth.structure.shared_variables),
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "baseline_checkpoint_hash": baseline["checkpoint_hash"],
        "checkpoint_parity": pilot.checkpoint.checkpoint_hash == baseline["checkpoint_hash"],
        "phase1_fes": pilot.consumed_fes,
        "proposal_budget_fes": result.proposal_budget_fes,
        "proposal_budget_parity": all(
            result.proposal_budget_fes == arm["proposal_budget_fes"]
            for arm in frozen_arms.values()
        ),
        "phase2_consumed_fes": result.phase2_consumed_fes,
        "terminal_fes": result.terminal_fes,
        "strict_best": all(
            cycle.best_error_after <= cycle.best_error_before for cycle in result.cycles
        ),
        "final_error": result.final_error,
        "ctp_fes": sum(cycle.ctp_fes for cycle in result.cycles),
        "ctp_triggered_cycles": sum(
            cycle.ctp_triggered_components > 0 for cycle in result.cycles
        ),
        "max_conflict_streak": max(cycle.max_conflict_streak for cycle in result.cycles),
        "envelope_no_encroachment": all(
            cycle.ctp_fes + cycle.neighborhood_fes == envelope_total
            for cycle in result.cycles
        ),
        "dispatch_consumption_parity": all(
            receipt.consumed_fes == receipt.reserved_fes
            for cycle in result.cycles
            for receipt in cycle.dispatch_receipts
        ),
        "dispatch": _dispatch_summary(result),
        "baselines": frozen_arms,
        "fixed_ctp_final_error": fixed_ctp["final_error"],
        "fixed_ctp_ctp_fes": fixed_ctp["ctp_fes"],
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


def _find_row(rows, mode: str, topology: str, overlap: int) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["cell"]["mode"] == mode
        and row["cell"]["topology"] == topology
        and row["cell"]["overlap_budget"] == overlap
    )


def run_gate(*, workers: int, cell_dir: Path) -> dict[str, object]:
    jobs = cells()
    completed = []
    pending = []
    for cell in jobs:
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
            print(f"completed {cell.mode}/{cell.topology}/overlap={cell.overlap_budget}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(run_cell, cell): cell for cell in pending}
            for future in as_completed(futures):
                cell = futures[future]
                row = future.result()
                _write_cell(cell_dir, row)
                completed.append(row)
                print(
                    f"completed {cell.mode}/{cell.topology}/overlap={cell.overlap_budget}",
                    flush=True,
                )
    rows = tuple(
        sorted(
            completed,
            key=lambda row: (
                row["cell"]["mode"],
                row["cell"]["topology"],
                row["cell"]["overlap_budget"],
            ),
        )
    )
    proposal_gain = np.asarray(
        [
            row["baselines"]["proposal_neighborhood"]["final_error"] - row["final_error"]
            for row in rows
        ],
        dtype=float,
    )
    full_gain = np.asarray(
        [row["baselines"]["full_context"]["final_error"] - row["final_error"] for row in rows],
        dtype=float,
    )
    fixed_ctp_gain = np.asarray(
        [row["fixed_ctp_final_error"] - row["final_error"] for row in rows],
        dtype=float,
    )
    star6 = _find_row(rows, "conflicting", "star", 6)
    chain3 = _find_row(rows, "conflicting", "chain", 3)
    star6_gain = (
        star6["baselines"]["proposal_neighborhood"]["final_error"] - star6["final_error"]
    )
    chain3_gain = (
        chain3["baselines"]["proposal_neighborhood"]["final_error"] - chain3["final_error"]
    )
    protocol_checks = {
        "cell_count_12": len(rows) == 12,
        "phase1_exact": all(row["phase1_fes"] == PHASE1_FES for row in rows),
        "checkpoint_parity": all(row["checkpoint_parity"] for row in rows),
        "proposal_budget_parity": all(row["proposal_budget_parity"] for row in rows),
        "terminal_exact": all(
            row["phase2_consumed_fes"] == TOTAL_BUDGET_FES - PHASE1_FES
            and row["terminal_fes"] == TOTAL_BUDGET_FES
            for row in rows
        ),
        "strict_best": all(row["strict_best"] for row in rows),
        "dispatch_consumption_parity": all(row["dispatch_consumption_parity"] for row in rows),
        "envelope_no_encroachment": all(row["envelope_no_encroachment"] for row in rows),
    }
    screening_checks = {
        "star_ov6_no_material_regression": float(star6_gain) >= -TOLERANCE,
        "chain_ov3_positive_gain": float(chain3_gain) > TOLERANCE,
        "not_worse_than_fixed_ctp_all_cells": bool(np.all(fixed_ctp_gain >= -TOLERANCE)),
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cell_count": 12,
            "workers": workers,
            "total_budget_fes": TOTAL_BUDGET_FES,
            "phase1_fes": PHASE1_FES,
            "phase2_fes": TOTAL_BUDGET_FES - PHASE1_FES,
            "neighborhood_fes": DEFAULT_NEIGHBORHOOD_FES,
            "refresh_cycles": DEFAULT_REFRESH_CYCLES,
            "dispatch_config": asdict(DISPATCH_CONFIG),
            "comparison_tolerance": TOLERANCE,
            "baseline_source": str(BASELINE),
            "fixed_ctp_source": str(FIXED_CTP),
        },
        "summary": {
            "vs_proposal_win_or_tie": float(np.mean(proposal_gain >= -TOLERANCE)),
            "vs_proposal_median_gain": float(np.median(proposal_gain)),
            "vs_proposal_min_gain": float(np.min(proposal_gain)),
            "vs_full_median_gain": float(np.median(full_gain)),
            "vs_fixed_ctp_min_gain": float(np.min(fixed_ctp_gain)),
            "star_ov6_gain": float(star6_gain),
            "chain_ov3_gain": float(chain3_gain),
            "dispatch_total_fes": sum(row["dispatch"]["dispatch_fes"] for row in rows),
            "dispatch_total_count": sum(row["dispatch"]["dispatch_count"] for row in rows),
            "escalation_total": sum(row["dispatch"]["escalations"] for row in rows),
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
        default=Path("artifacts/overlap_gcb_coordinator_gate37/cells"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_gcb_coordinator_gate37/confirmation_fresh.json"),
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

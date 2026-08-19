"""Gate 40: coordinator generalization across ackley/elliptic/schwefel families."""

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
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    GCB_COORDINATED_MODE,
    PERSISTENT_CTP_MODE,
    run_overlap_from_pilot,
)
from experiments.overlap_coordinate_ctp_gate36_fair_screening import TOLERANCE
from experiments.overlap_family_calibration_gate40 import FamilyCell, build_family_cell
from experiments.overlap_gcb_coordinator_gate37_screening import _dispatch_summary

FAMILIES = ("ackley", "elliptic", "schwefel")
TOPOLOGIES = ("chain", "star")
OVERLAPS = (3, 6)
GATE_SEED = 20260832
CELL_SCHEMA = "arac-overlap-family-gate40-cell-v1"
OUTPUT_SCHEMA = "arac-overlap-family-gate40-screen-v1"
DISPATCH_CONFIG = GcbDispatchConfig(hub_mode="relative", complex_hub_ratio=0.9)


def jobs() -> tuple[FamilyCell, ...]:
    return tuple(
        FamilyCell(base, topology, overlap, GATE_SEED)
        for base in FAMILIES
        for topology in TOPOLOGIES
        for overlap in OVERLAPS
    )


def _cell_path(directory: Path, cell: FamilyCell) -> Path:
    return directory / (
        f"{cell.base_function}_{cell.topology}_overlap{cell.overlap_budget}"
        f"_seed{cell.seed}.json"
    )


def run_cell(cell: FamilyCell) -> dict[str, object]:
    problem, truth = build_family_cell(cell)
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=3_000_000,
        run_seed=cell.seed,
        **{
            "anchor_count": 5,
            "step": 0.25,
            "rounds": 12,
            "bucket_size": 16,
            "max_candidate_pairs": 128,
        },
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
        arms[mode] = {
            "final_error": result.final_error,
            "terminal_fes": result.terminal_fes,
            "strict_best": all(
                cycle.best_error_after <= cycle.best_error_before
                for cycle in result.cycles
            ),
        }
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
        "adapter_ready": bool(pilot.adaptation.ready),
        "phase1_fes": pilot.consumed_fes,
        "arms": arms,
        "dispatch": dispatch_summary,
        "envelope_no_encroachment": envelope_ok,
        "dispatch_consumption_parity": consumption_ok,
    }


def _write_cell(directory: Path, row: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cell = FamilyCell(**row["cell"])
    _cell_path(directory, cell).write_text(
        json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_cell(directory: Path, cell: FamilyCell) -> dict[str, object] | None:
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
            print(f"completed {cell.base_function}/{cell.topology}/ov={cell.overlap_budget}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(run_cell, cell): cell for cell in pending}
            for future in as_completed(futures):
                cell = futures[future]
                row = future.result()
                _write_cell(cell_dir, row)
                completed.append(row)
                print(
                    f"completed {cell.base_function}/{cell.topology}/ov={cell.overlap_budget}",
                    flush=True,
                )
    rows = tuple(
        sorted(
            completed,
            key=lambda row: (
                row["cell"]["base_function"],
                row["cell"]["topology"],
                row["cell"]["overlap_budget"],
            ),
        )
    )

    def _gain(row, reference):
        return row["arms"][reference]["final_error"] - row["arms"]["coordinator"]["final_error"]

    proposal_gain = np.asarray([_gain(row, "proposal_neighborhood") for row in rows], dtype=float)
    ctp_gain = np.asarray([_gain(row, "persistent_ctp") for row in rows], dtype=float)
    star_rows = [row for row in rows if row["cell"]["topology"] == "star"]
    chain_rows = [row for row in rows if row["cell"]["topology"] == "chain"]
    star_gains = [_gain(row, "proposal_neighborhood") for row in star_rows]
    chain_positive_where_potential = all(
        (_gain(row, "proposal_neighborhood") > TOLERANCE)
        or (
            _gain(row, "persistent_ctp") <= TOLERANCE
            and _gain(row, "persistent_ctp") >= -TOLERANCE
        )
        for row in chain_rows
    )
    protocol_checks = {
        "cell_count_12": len(rows) == 12,
        "adapter_ready_all": all(row["adapter_ready"] for row in rows),
        "phase1_exact": all(row["phase1_fes"] == 180_000 for row in rows),
        "terminal_exact": all(
            arm["terminal_fes"] == 3_000_000
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
        "chain_positive_where_potential": chain_positive_where_potential,
        "win_or_tie_vs_proposal_ge_0_75": float(np.mean(proposal_gain >= -TOLERANCE)) >= 0.75,
        "not_worse_than_persistent_ctp_all": bool(np.all(ctp_gain >= -TOLERANCE)),
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cell_count": 12,
            "workers": workers,
            "gate_seed": GATE_SEED,
            "families": FAMILIES,
            "topologies": TOPOLOGIES,
            "total_budget_fes": 3_000_000,
            "phase1_fes": 180_000,
            "neighborhood_fes": DEFAULT_NEIGHBORHOOD_FES,
            "refresh_cycles": DEFAULT_REFRESH_CYCLES,
            "dispatch_config": asdict(DISPATCH_CONFIG),
            "comparison_tolerance": TOLERANCE,
            "calibration_source": "artifacts/overlap_family_calibration_gate40/calibration.json",
            "aob24_audit_source": "artifacts/aob24_overlap_applicability_audit/audit.json",
        },
        "summary": {
            "vs_proposal_win_or_tie": float(np.mean(proposal_gain >= -TOLERANCE)),
            "vs_proposal_min_gain": float(np.min(proposal_gain)),
            "vs_persistent_ctp_min_gain": float(np.min(ctp_gain)),
            "star_gains": [float(_gain(row, "proposal_neighborhood")) for row in star_rows],
            "chain_gains": [float(_gain(row, "proposal_neighborhood")) for row in chain_rows],
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
        default=Path("artifacts/overlap_family_gate40/cells"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_family_gate40/confirmation_fresh.json"),
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

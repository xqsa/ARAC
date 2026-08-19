"""Gate 46: unified ARAC-OC loop vs the frozen minimal kernel (sparse grid)."""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path

from arac.coordination import GcbDispatchConfig
from arac.coordination.contract import OcCoordinatorConfig
from arac.coordination.loop import _overlap_components, run_oc_unified
from arac.evidence import run_phase1_overlap_pilot
from arac.overlap_core import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    GCB_COORDINATED_MODE,
    _proposal_budget,
    run_overlap_from_pilot,
)
from experiments.overlap_family_calibration_gate40 import FamilyCell, build_family_cell

FAMILIES = ("ackley", "elliptic", "schwefel")
TOPOLOGIES = ("chain", "star")
OVERLAPS = (3, 6)
REPRO_SEED = 20260832   # Gate 40's seed (reproduction layer)
FRESH_SEED = 20260833   # never used by any gate/calibration/audit
PHASE1_KWARGS = {
    "anchor_count": 5,
    "step": 0.25,
    "rounds": 12,
    "bucket_size": 16,
    "max_candidate_pairs": 128,
}
UNIFIED_CONFIG = OcCoordinatorConfig(pulse_min_fes=8, pulse_max_fes=DEFAULT_NEIGHBORHOOD_FES)
DISPATCH_CONFIG = GcbDispatchConfig(hub_mode="relative", complex_hub_ratio=0.9)
GATE40_CELLS = Path("artifacts/overlap_family_gate40/cells")
CELL_SCHEMA = "arac-oc-unified-gate46-cell-v1"
OUTPUT_SCHEMA = "arac-oc-unified-gate46-v1"


def jobs() -> tuple[FamilyCell, ...]:
    return tuple(
        FamilyCell(base, topology, overlap, seed)
        for seed in (REPRO_SEED, FRESH_SEED)
        for base in FAMILIES
        for topology in TOPOLOGIES
        for overlap in OVERLAPS
    )


def _cell_path(directory: Path, cell: FamilyCell) -> Path:
    return directory / (
        f"{cell.base_function}_{cell.topology}_overlap{cell.overlap_budget}"
        f"_seed{cell.seed}.json"
    )


def _frozen_gate40(cell: FamilyCell) -> dict[str, float]:
    path = GATE40_CELLS / (
        f"{cell.base_function}_{cell.topology}_overlap{cell.overlap_budget}"
        f"_seed{cell.seed}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    arms = payload["result"]["arms"]
    return {mode: float(values["final_error"]) for mode, values in arms.items()}


def run_cell(cell: FamilyCell) -> dict[str, object]:
    problem, truth = build_family_cell(cell)
    pilot = run_phase1_overlap_pilot(
        problem, total_budget_fes=3_000_000, run_seed=cell.seed, **PHASE1_KWARGS
    )
    structure = pilot.adaptation.structure
    components = _overlap_components(structure)
    sense_budget = _proposal_budget(
        3_000_000 - pilot.checkpoint.phase1_fes,
        components,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
    )

    arms: dict[str, dict[str, object]] = {}
    kernel = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=GCB_COORDINATED_MODE,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        dispatch_config=DISPATCH_CONFIG,
    )
    kernel_parity = all(
        receipt.consumed_fes == receipt.reserved_fes
        for cycle in kernel.cycles
        for receipt in cycle.dispatch_receipts
    )
    kernel_envelope = all(
        cycle.ctp_fes + cycle.neighborhood_fes
        == DEFAULT_NEIGHBORHOOD_FES * len(kernel.overlap_components)
        for cycle in kernel.cycles
    )
    arms["coordinator"] = {
        "final_error": kernel.final_error,
        "terminal_fes": kernel.terminal_fes,
        "strict_best": all(
            cycle.best_error_after <= cycle.best_error_before for cycle in kernel.cycles
        ),
    }

    unified = run_oc_unified(
        problem,
        pilot,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        sense_budget_fes=sense_budget,
        config=UNIFIED_CONFIG,
    )
    arms["oc_unified"] = {
        "final_error": unified.final_error,
        "terminal_fes": unified.terminal_fes,
        "strict_best": all(
            cycle.best_error_after <= cycle.best_error_before for cycle in unified.cycles
        ),
    }
    unified_flow = {
        "sense": sum(trace.sense_fes for trace in unified.cycles),
        "probe": sum(trace.probe_fes for trace in unified.cycles),
        "arbitration": sum(trace.arbitration_fes for trace in unified.cycles),
        "operator": sum(trace.operator_fes for trace in unified.cycles),
        "smp": sum(trace.smp_fes for trace in unified.cycles),
        "tail": unified.tail_fes,
    }
    unified_parity = all(
        receipt.actual_fes == receipt.reserved_fes for receipt in unified.receipts
    )
    unified_state_chain = bool(unified.receipts) and all(
        len(receipt.state_hash) == 64 for receipt in unified.receipts
    )
    action_counts: dict[str, int] = {}
    for trace in unified.cycles:
        action_counts[trace.action] = action_counts.get(trace.action, 0) + 1

    row: dict[str, object] = {
        "cell": asdict(cell),
        "layer": "repro" if cell.seed == REPRO_SEED else "fresh",
        "truth_shared_count": len(truth.structure.shared_variables),
        "adapter_ready": bool(pilot.adaptation.ready),
        "phase1_fes": pilot.checkpoint.phase1_fes,
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "sense_budget_fes": sense_budget,
        "arms": arms,
        "kernel_receipt_parity": kernel_parity,
        "kernel_envelope_no_encroachment": kernel_envelope,
        "unified_receipt_parity": unified_parity,
        "unified_state_hash_chain": unified_state_chain,
        "unified_budget_flow": unified_flow,
        "unified_action_counts": action_counts,
    }

    if cell.seed == FRESH_SEED:
        proposal = run_overlap_from_pilot(
            problem,
            pilot,
            coordination_mode="proposal_neighborhood",
            refresh_cycles=DEFAULT_REFRESH_CYCLES,
            neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        )
        arms["proposal_neighborhood"] = {
            "final_error": proposal.final_error,
            "terminal_fes": proposal.terminal_fes,
            "strict_best": all(
                cycle.best_error_after <= cycle.best_error_before for cycle in proposal.cycles
            ),
        }
    else:
        row["gate40_frozen"] = _frozen_gate40(cell)
        reproduction = abs(
            float(arms["coordinator"]["final_error"]) - row["gate40_frozen"]["coordinator"]
        )
        row["kernel_reproduces_gate40"] = reproduction <= 0.0
    return row


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


def run_gate(*, workers: int, cell_dir: Path, output: Path) -> dict[str, object]:
    completed: list[dict[str, object]] = []
    pending: list[FamilyCell] = []
    for cell in jobs():
        row = _read_cell(cell_dir, cell)
        if row is None:
            pending.append(cell)
        else:
            completed.append(row)
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_cell, cell): cell for cell in pending}
            for future in as_completed(futures):
                row = future.result()
                _write_cell(cell_dir, row)
                completed.append(row)
                print(
                    f"{row['cell']['base_function']}/{row['cell']['topology']}/"
                    f"ov{row['cell']['overlap_budget']}/s{row['cell']['seed']}: "
                    f"kernel={row['arms']['coordinator']['final_error']:.6g} "
                    f"unified={row['arms']['oc_unified']['final_error']:.6g}",
                    flush=True,
                )
    completed.sort(key=lambda row: (row["cell"]["seed"], row["cell"]["base_function"]))

    repro_rows = [row for row in completed if row["layer"] == "repro"]
    fresh_rows = [row for row in completed if row["layer"] == "fresh"]
    protocol_checks = {
        "cell_count_24": len(completed) == 24,
        "phase1_exact": all(row["phase1_fes"] == 180_000 for row in completed),
        "adapter_ready_all": all(row["adapter_ready"] for row in completed),
        "terminal_exact": all(
            arm["terminal_fes"] == 3_000_000 for row in completed for arm in row["arms"].values()
        ),
        "strict_best": all(
            arm["strict_best"] for row in completed for arm in row["arms"].values()
        ),
        "kernel_receipts_ok": all(
            row["kernel_receipt_parity"] and row["kernel_envelope_no_encroachment"]
            for row in completed
        ),
        "unified_receipts_ok": all(
            row["unified_receipt_parity"] and row["unified_state_hash_chain"]
            for row in completed
        ),
    }
    kernel_reproduction = all(
        row.get("kernel_reproduces_gate40", False) for row in repro_rows
    ) and len(repro_rows) == 12

    tolerance = 1e-9
    not_worse = [
        row["arms"]["oc_unified"]["final_error"]
        <= row["arms"]["coordinator"]["final_error"] * 1.05 + tolerance
        for row in completed
    ]
    star_ok = []
    for row in repro_rows:
        if row["cell"]["topology"] != "star":
            continue
        star_ok.append(
            row["arms"]["oc_unified"]["final_error"]
            <= row["gate40_frozen"]["proposal_neighborhood"] + tolerance
        )
    for row in fresh_rows:
        if row["cell"]["topology"] != "star":
            continue
        star_ok.append(
            row["arms"]["oc_unified"]["final_error"]
            <= row["arms"]["proposal_neighborhood"]["final_error"] + tolerance
        )
    screening_checks = {
        "not_worse_than_kernel_all": len(not_worse) == 24 and all(not_worse),
        "star_no_regression_vs_proposal": len(star_ok) == 12 and all(star_ok),
    }
    wins = sum(
        1
        for row in completed
        if row["arms"]["oc_unified"]["final_error"]
        < row["arms"]["coordinator"]["final_error"] - tolerance
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "repro_seed": REPRO_SEED,
            "fresh_seed": FRESH_SEED,
            "phase1_kwargs": PHASE1_KWARGS,
            "unified_config": asdict(UNIFIED_CONFIG),
            "kernel_dispatch_config": asdict(DISPATCH_CONFIG),
            "refresh_cycles": DEFAULT_REFRESH_CYCLES,
            "neighborhood_fes": DEFAULT_NEIGHBORHOOD_FES,
            "gate40_cell_source": str(GATE40_CELLS),
        },
        "protocol_checks": protocol_checks,
        "kernel_reproduction_layer": {
            "checked_cells": len(repro_rows),
            "all_match_gate40": kernel_reproduction,
        },
        "screening_checks": screening_checks,
        "gate_passed": all(protocol_checks.values()) and kernel_reproduction and all(screening_checks.values()),
        "summary": {
            "unified_wins_vs_kernel": wins,
            "instance_count": len(completed),
            "mean_operator_share": (
                sum(
                    row["unified_budget_flow"]["operator"]
                    / max(
                        1,
                        sum(row["unified_budget_flow"].values()),
                    )
                    for row in completed
                )
                / max(1, len(completed))
            ),
        },
        "rows": completed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "protocol": protocol_checks,
                "reproduction": kernel_reproduction,
                "screening": screening_checks,
                "wins": wins,
            },
            indent=1,
        )
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cell-dir", type=Path, default=Path("artifacts/oc_unified_sparse_regression_gate46/cells")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oc_unified_sparse_regression_gate46/confirmation.json"),
    )
    args = parser.parse_args()
    payload = run_gate(workers=args.workers, cell_dir=args.cell_dir, output=args.output)
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

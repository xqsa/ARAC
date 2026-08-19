"""Gate 47b: streak-triggered dispatch confirmation on the sparse grid."""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from math import ceil

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
from experiments.overlap_arac_gate29_screening import (
    OVERLAP_BUDGETS,
    PHASE1_KWARGS,
    TOTAL_BUDGET_FES,
    TOPOLOGIES,
    Cell,
    build_cell,
)
from experiments.overlap_family_calibration_gate40 import FamilyCell, build_family_cell

LAYER1_SEED = 20260832
LAYER2_SEED = 20260835
FAMILIES = ("ackley", "elliptic", "schwefel")
STREAK_CONFIG = OcCoordinatorConfig(pulse_min_fes=8, pulse_max_fes=32)
KERNEL_DISPATCH_CONFIG = GcbDispatchConfig(hub_mode="relative", complex_hub_ratio=0.9)
GATE46_CELLS = Path("artifacts/oc_unified_sparse_regression_gate46/cells")
CELL_SCHEMA = "arac-oc-streak-gate47b-cell-v1"
OUTPUT_SCHEMA = "arac-oc-streak-gate47b-v1"
OUTPUT_ROOT = Path("artifacts/oc_streak_confirmation_gate47b")


def layer1_jobs() -> tuple[FamilyCell, ...]:
    return tuple(
        FamilyCell(base, topology, overlap, LAYER1_SEED)
        for base in FAMILIES
        for topology in ("chain", "star")
        for overlap in (3, 6)
    )


def layer2_jobs() -> tuple[Cell, ...]:
    return tuple(
        Cell("conflicting", topology, overlap, LAYER2_SEED)
        for topology in TOPOLOGIES
        for overlap in OVERLAP_BUDGETS
    )


def _cell_path(directory: Path, name: str) -> Path:
    return directory / f"{name}.json"


def _sense_budget(pilot) -> int:
    components = _overlap_components(pilot.adaptation.structure)
    base = _proposal_budget(
        TOTAL_BUDGET_FES - pilot.checkpoint.phase1_fes,
        components,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
    )
    # The historical proposal envelope already pays arbitration, complete
    # context writeback and the 32-FE neighborhood lane.  Unified ARAC-OC
    # additionally pays its counted probe and must leave one minimum pulse
    # available for a real operator; reserve that headroom in the SMP sense
    # lane instead of letting context writeback starve GCB dispatch.
    groups = sum(len(component) for component in components)
    max_probe = max(
        (
            2 * len(
                tuple(
                    variable
                    for variable in pilot.adaptation.structure.shared_variables
                    if set(pilot.adaptation.structure.owners(variable)).issubset(set(component))
                )
            )
            for component in components
        ),
        default=0,
    )
    reserve_per_group = ceil((max_probe + STREAK_CONFIG.pulse_min_fes) / max(1, groups))
    return max(8, base - reserve_per_group)


def _unified_summary(unified) -> dict[str, object]:
    return {
        "final_error": unified.final_error,
        "terminal_fes": unified.terminal_fes,
        "strict_best": all(
            trace.best_error_after <= trace.best_error_before for trace in unified.cycles
        ),
        "receipt_parity": all(
            receipt.actual_fes == receipt.reserved_fes for receipt in unified.receipts
        ),
        "state_hash_chain": bool(unified.receipts)
        and all(len(receipt.state_hash) == 64 for receipt in unified.receipts),
        "operator_fes": sum(trace.operator_fes for trace in unified.cycles),
        "smp_fes": sum(trace.smp_fes for trace in unified.cycles),
        "action_counts": {
            action: sum(1 for trace in unified.cycles if trace.action == action)
            for action in sorted({trace.action for trace in unified.cycles})
        },
        "budget_flow": {
            "sense": sum(trace.sense_fes for trace in unified.cycles),
            "probe": sum(trace.probe_fes for trace in unified.cycles),
            "arbitration": sum(trace.arbitration_fes for trace in unified.cycles),
            "smp": sum(trace.smp_fes for trace in unified.cycles),
            "operator": sum(trace.operator_fes for trace in unified.cycles),
            "tail": unified.tail_fes,
        },
    }


def _arm_summary(result) -> dict[str, object]:
    return {
        "final_error": result.final_error,
        "terminal_fes": result.terminal_fes,
        "strict_best": all(
            cycle.best_error_after <= cycle.best_error_before for cycle in result.cycles
        ),
    }


def run_layer1_cell(cell: FamilyCell) -> dict[str, object]:
    frozen = json.loads(
        _cell_path(
            GATE46_CELLS,
            f"{cell.base_function}_{cell.topology}_overlap{cell.overlap_budget}"
            f"_seed{cell.seed}",
        ).read_text(encoding="utf-8")
    )["result"]
    problem, _truth = build_family_cell(cell)
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=cell.seed,
        **{
            "anchor_count": 5,
            "step": 0.25,
            "rounds": 12,
            "bucket_size": 16,
            "max_candidate_pairs": 128,
        },
    )
    unified = run_oc_unified(
        problem,
        pilot,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        sense_budget_fes=_sense_budget(pilot),
        config=STREAK_CONFIG,
    )
    return {
        "layer": 1,
        "cell": asdict(cell),
        "phase1_fes": pilot.checkpoint.phase1_fes,
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "frozen_kernel": frozen["arms"]["coordinator"]["final_error"],
        "frozen_proposal": frozen["gate40_frozen"]["proposal_neighborhood"],
        "oc_unified": _unified_summary(unified),
    }


def run_layer2_arm(cell: Cell, arm: str) -> dict[str, object]:
    """Run one layer-2 arm with its own deterministic pilot (checkpoint hash identical)."""

    problem, truth = build_cell(cell)
    pilot = run_phase1_overlap_pilot(
        problem, total_budget_fes=TOTAL_BUDGET_FES, run_seed=cell.seed, **PHASE1_KWARGS
    )
    row: dict[str, object] = {
        "layer": 2,
        "arm": arm,
        "cell": {
            "mode": cell.mode,
            "topology": cell.topology,
            "overlap_budget": cell.overlap_budget,
            "seed": cell.seed,
        },
        "truth_shared_count": len(truth.structure.shared_variables),
        "phase1_fes": pilot.checkpoint.phase1_fes,
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
    }
    if arm == "coordinator":
        result = run_overlap_from_pilot(
            problem,
            pilot,
            coordination_mode=GCB_COORDINATED_MODE,
            refresh_cycles=DEFAULT_REFRESH_CYCLES,
            neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
            dispatch_config=KERNEL_DISPATCH_CONFIG,
        )
        row["result"] = _arm_summary(result)
    elif arm == "proposal_neighborhood":
        result = run_overlap_from_pilot(
            problem,
            pilot,
            coordination_mode="proposal_neighborhood",
            refresh_cycles=DEFAULT_REFRESH_CYCLES,
            neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        )
        row["result"] = _arm_summary(result)
    elif arm == "oc_unified":
        unified = run_oc_unified(
            problem,
            pilot,
            refresh_cycles=DEFAULT_REFRESH_CYCLES,
            sense_budget_fes=_sense_budget(pilot),
            config=STREAK_CONFIG,
        )
        row["result"] = _unified_summary(unified)
    else:
        raise ValueError(f"unknown layer-2 arm: {arm}")
    return row


def _run_item(item: dict[str, object]) -> dict[str, object]:
    if item["kind"] == "L1":
        return run_layer1_cell(item["cell"])
    return run_layer2_arm(item["cell"], item["arm"])


def _cell_name(row: dict[str, object]) -> str:
    cell = row["cell"]
    if row["layer"] == 1:
        return f"L1_{cell['base_function']}_{cell['topology']}_ov{cell['overlap_budget']}"
    return f"L2_{cell['topology']}_ov{cell['overlap_budget']}"


def _cached_run_all(
    directory: Path, items: list[tuple[dict[str, object], str]], workers: int
) -> list[dict[str, object]]:
    completed: list[dict[str, object]] = []
    pending: list[tuple[dict[str, object], str]] = []
    for item, name in items:
        path = _cell_path(directory, name)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CELL_SCHEMA:
                raise RuntimeError(f"cell schema drifted: {path}")
            completed.append(payload["result"])
        else:
            pending.append((item, name))
    if pending:
        directory.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {
                executor.submit(_run_item, item): name for item, name in pending
            }
            for future in as_completed(futures):
                row = future.result()
                name = futures[future]
                _cell_path(directory, name).write_text(
                    json.dumps(
                        {"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True
                    ),
                    encoding="utf-8",
                )
                completed.append(row)
                if row["layer"] == 1:
                    print(
                        f"L1 {row['cell']['base_function']}/{row['cell']['topology']}/"
                        f"ov{row['cell']['overlap_budget']}: unified={row['oc_unified']['final_error']:.6g} "
                        f"kernel={row['frozen_kernel']:.6g} op_fes={row['oc_unified']['operator_fes']}",
                        flush=True,
                    )
                else:
                    print(
                        f"L2 {row['cell']['topology']}/ov{row['cell']['overlap_budget']}/{row['arm']}: "
                        f"final={row['result']['final_error']:.6g} "
                        f"op_fes={row['result'].get('operator_fes', 0)}",
                        flush=True,
                    )
    return completed


def _merge_layer2(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[int, int], dict[str, object]] = {}
    hashes: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        key = (row["cell"]["topology"], row["cell"]["overlap_budget"])
        if key not in merged:
            merged[key] = {
                "layer": 2,
                "cell": row["cell"],
                "truth_shared_count": row["truth_shared_count"],
                "phase1_fes": row["phase1_fes"],
                "checkpoint_hash": row["checkpoint_hash"],
                "arms": {},
            }
            hashes[key] = set()
        merged[key]["arms"][row["arm"]] = row["result"]
        hashes[key].add(row["checkpoint_hash"])
    for key, seen in hashes.items():
        if len(seen) != 1:
            raise RuntimeError(f"layer-2 checkpoint parity violated for {key}: {seen}")
    return [merged[key] for key in sorted(merged)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    cells_dir = OUTPUT_ROOT / "cells"

    items: list[tuple[dict[str, object], str]] = []
    for cell in layer1_jobs():
        items.append(
            (
                {"kind": "L1", "cell": cell},
                f"L1_{cell.base_function}_{cell.topology}_ov{cell.overlap_budget}",
            )
        )
    for cell in layer2_jobs():
        for arm in ("coordinator", "proposal_neighborhood", "oc_unified"):
            items.append(
                (
                    {"kind": "L2arm", "cell": cell, "arm": arm},
                    f"L2_{cell.topology}_ov{cell.overlap_budget}_{arm}",
                )
            )
    rows = _cached_run_all(cells_dir, items, args.workers)
    layer1 = sorted(
        (row for row in rows if row["layer"] == 1),
        key=lambda row: (row["cell"]["base_function"], row["cell"]["topology"]),
    )
    layer2 = _merge_layer2([row for row in rows if row["layer"] == 2])

    tolerance = 1e-9
    protocol_checks = {
        "layer1_cell_count_12": len(layer1) == 12,
        "layer2_cell_count_6": len(layer2) == 6,
        "phase1_exact": all(row["phase1_fes"] == 180_000 for row in layer1 + layer2),
        "terminal_exact": all(
            arm["terminal_fes"] == TOTAL_BUDGET_FES
            for row in layer1 + layer2
            for arm in (
                [row["oc_unified"]] if row["layer"] == 1 else list(row["arms"].values())
            )
        ),
        "strict_best": all(
            arm["strict_best"]
            for row in layer1 + layer2
            for arm in (
                [row["oc_unified"]] if row["layer"] == 1 else list(row["arms"].values())
            )
        ),
        "unified_receipts_ok": all(
            row["oc_unified"]["receipt_parity"] and row["oc_unified"]["state_hash_chain"]
            if row["layer"] == 1
            else row["arms"]["oc_unified"]["receipt_parity"]
            and row["arms"]["oc_unified"]["state_hash_chain"]
            for row in layer1 + layer2
        ),
    }
    chain_rows = [row for row in layer2 if row["cell"]["topology"] == "chain"]
    star_checks = []
    for row in layer1:
        if row["cell"]["topology"] == "star":
            star_checks.append(
                row["oc_unified"]["final_error"] <= row["frozen_proposal"] + tolerance
            )
    for row in layer2:
        if row["cell"]["topology"] == "star":
            star_checks.append(
                row["arms"]["oc_unified"]["final_error"]
                <= row["arms"]["proposal_neighborhood"]["final_error"] + tolerance
            )
    not_worse = [
        row["oc_unified"]["final_error"] <= row["frozen_kernel"] * 1.05 + tolerance
        for row in layer1
    ] + [
        row["arms"]["oc_unified"]["final_error"]
        <= row["arms"]["coordinator"]["final_error"] * 1.05 + tolerance
        for row in layer2
    ]
    screening_checks = {
        "path_fires_on_chain": any(
            row["arms"]["oc_unified"]["operator_fes"] > 0 for row in chain_rows
        ),
        "not_worse_than_kernel_all": len(not_worse) == 18 and all(not_worse),
        "star_no_regression_vs_proposal": len(star_checks) == 8 and all(star_checks),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "layer1_seed": LAYER1_SEED,
            "layer2_seed": LAYER2_SEED,
            "streak_config": asdict(STREAK_CONFIG),
            "kernel_dispatch_config": asdict(KERNEL_DISPATCH_CONFIG),
            "gate46_reference": str(GATE46_CELLS),
        },
        "protocol_checks": protocol_checks,
        "screening_checks": screening_checks,
        "gate_passed": all(protocol_checks.values()) and all(screening_checks.values()),
        "summary": {
            "layer1_unified_wins": sum(
                1
                for row in layer1
                if row["oc_unified"]["final_error"] < row["frozen_kernel"] - tolerance
            ),
            "layer2_unified_wins": sum(
                1
                for row in layer2
                if row["arms"]["oc_unified"]["final_error"]
                < row["arms"]["coordinator"]["final_error"] - tolerance
            ),
            "cells_with_operator_dispatch": sum(
                1
                for row in layer1 + layer2
                if (row["oc_unified"] if row["layer"] == 1 else row["arms"]["oc_unified"])[
                    "operator_fes"
                ]
                > 0
            ),
        },
        "layer1": layer1,
        "layer2": layer2,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": protocol_checks,
                "screening": screening_checks,
                "summary": payload["summary"],
            },
            indent=1,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

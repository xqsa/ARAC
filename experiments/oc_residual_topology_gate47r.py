"""Gate 47-R: fresh-seed ARAC-OC routing confirmation.

The benchmark cells exercise the real Phase-I -> ARAC-OC entry point.  A small
plan-only witness suite uses the same frozen GCB/operator contract to make the
four dispatch branches observable even when a fresh benchmark seed naturally
produces no persistent residual.  The two records are intentionally separate:
the witness proves route reachability, while the benchmark records what the
real run actually dispatched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path

from arac.coordination.contract import (
    OC_ACTION_AOR,
    OC_ACTION_ARBITRATION,
    OC_ACTION_CTP_RESTRICTED,
    OC_ACTION_CTP_SHARED_CORE,
    OC_ACTION_SMP,
    OcCoordinatorConfig,
)
from arac.coordination.loop import _overlap_components, run_oc_unified
from arac.coordination.planner import OcDispatchPlanner
from arac.coordination.state import CoordinatorState
from arac.coordination.overlap import OverlapStructure
from arac.evidence import run_phase1_overlap_pilot
from arac.overlap_core import _proposal_budget
from experiments.overlap_arac_gate29_screening import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    PHASE1_KWARGS,
    TOTAL_BUDGET_FES,
    Cell,
    build_cell,
)

OUTPUT_ROOT = Path("artifacts/oc_residual_topology_gate47r")
CELL_SCHEMA = "arac-oc-gate47r-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate47r-v1"
FRESH_RUN_SEED = 20260836
FRESH_WITNESS_SEEDS = (20260836, 20260837)
ROUTE_CELLS = (
    Cell("conflicting", "chain", 3, FRESH_RUN_SEED),
    Cell("conflicting", "star", 6, FRESH_RUN_SEED),
    Cell("conflicting", "chain", 6, FRESH_RUN_SEED),
)
OC_CONFIG = OcCoordinatorConfig(
    # These legacy fields remain in the config hash for replay compatibility;
    # Gate47-R does not use their amplitude as a dispatch trigger.
    tau_enter=0.5,
    tau_exit=0.2,
    k_enter=3,
    k_exit=3,
    pulse_min_fes=8,
    pulse_max_fes=32,
    calibration_status="gate47r-residual-topology",
)


def _sense_budget(pilot) -> int:
    components = _overlap_components(pilot.adaptation.structure)
    return _proposal_budget(
        TOTAL_BUDGET_FES - pilot.checkpoint.phase1_fes,
        components,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
    )


def _cell_path(cell: Cell) -> Path:
    return OUTPUT_ROOT / "cells" / (
        f"{cell.mode}_{cell.topology}_overlap{cell.overlap_budget}_seed{cell.seed}.json"
    )


def _run_cell(cell: Cell) -> dict[str, object]:
    problem, truth = build_cell(cell)
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=cell.seed,
        **PHASE1_KWARGS,
    )
    result = run_oc_unified(
        problem,
        pilot,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        sense_budget_fes=_sense_budget(pilot),
        config=OC_CONFIG,
    )
    action_counts = {
        action: sum(trace.action == action for trace in result.cycles)
        for action in sorted({trace.action for trace in result.cycles})
    }
    return {
        "cell": asdict(cell),
        "truth_shared_count": len(truth.structure.shared_variables),
        "phase1_fes": pilot.checkpoint.phase1_fes,
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "unified": {
            "final_error": result.final_error,
            "terminal_fes": result.terminal_fes,
            "receipt_count": len(result.receipts),
            "action_counts": action_counts,
            "operator_fes": sum(trace.operator_fes for trace in result.cycles),
            "smp_fes": sum(trace.smp_fes for trace in result.cycles),
            "budget_flow": {
                "sense": sum(trace.sense_fes for trace in result.cycles),
                "probe": sum(trace.probe_fes for trace in result.cycles),
                "arbitration": sum(trace.arbitration_fes for trace in result.cycles),
                "smp": sum(trace.smp_fes for trace in result.cycles),
                "operator": sum(trace.operator_fes for trace in result.cycles),
                "tail": result.tail_fes,
            },
            "strict_best": all(
                trace.best_error_after <= trace.best_error_before for trace in result.cycles
            ),
            "receipt_parity": all(
                receipt.actual_fes == receipt.reserved_fes for receipt in result.receipts
            ),
            "state_hash_chain": all(
                len(receipt.state_hash) == 64 for receipt in result.receipts
            ),
            "final_state_hash": result.final_state_hash,
        },
    }


def _route_structure() -> OverlapStructure:
    """Frozen chain/star witness structure used only for plan reachability."""

    return OverlapStructure(
        dimension=14,
        groups=(
            (0, 1, 2),
            (2, 3, 4),
            (4, 5, 6),
            (6, 7),
            (8, 9, 10),
            (10, 11),
            (9, 12),
            (8, 13),
        ),
    )


def _witness_plan(
    planner: OcDispatchPlanner,
    state: CoordinatorState,
    component: tuple[int, ...],
    *,
    cycle_index: int,
    high_cycles: int = 0,
    low_trust: bool = False,
    high_probe_without_persistence: bool = False,
):
    scope = planner.shared_scope_variables(component)
    if high_probe_without_persistence:
        state.observe_probes(component, scope, {variable: 1_000.0 for variable in scope})
    for _ in range(high_cycles):
        state.observe_proposal_conflict(component, high_conflict=True)
    if low_trust:
        for variable in scope:
            for group in state.structure.owners(variable):
                state.qhat[(variable, group)] = 0.1
    signal = state.signal(component, cycle_index=cycle_index)
    return planner.make_plan(
        signal,
        cycle_index=cycle_index,
        scope=scope,
        probe_widths={variable: 1.0 for variable in scope},
        available_fes=128,
    )


def _route_witness(seed: int) -> dict[str, object]:
    structure = _route_structure()
    components = [(0, 1, 2, 3), (4, 5, 6, 7)]
    planner = OcDispatchPlanner(structure, components, config=OC_CONFIG, base_seed=seed)
    cases = []
    scenarios = (
        ("chain_restricted", components[0], {"high_cycles": 2}),
        ("star_shared_core", components[1], {"high_cycles": 2}),
        ("low_qhat_smp", components[0], {"high_cycles": 2, "low_trust": True}),
        ("persistent_escalation_aor", components[0], {"high_cycles": 6}),
        (
            "probe_amplitude_only_arbitration",
            components[0],
            {"high_probe_without_persistence": True},
        ),
    )
    expected = {
        "chain_restricted": OC_ACTION_CTP_RESTRICTED,
        "star_shared_core": OC_ACTION_CTP_SHARED_CORE,
        "low_qhat_smp": OC_ACTION_SMP,
        "persistent_escalation_aor": OC_ACTION_AOR,
        "probe_amplitude_only_arbitration": OC_ACTION_ARBITRATION,
    }
    for index, (name, component, options) in enumerate(scenarios):
        state = CoordinatorState(structure, components, config=OC_CONFIG, checkpoint_hash="")
        plan = _witness_plan(
            planner,
            state,
            component,
            cycle_index=index,
            **options,
        )
        cases.append(
            {
                "name": name,
                "component": list(component),
                "action": plan.action,
                "reason": plan.reason,
                "conflict_level": plan.conflict_level,
                "relative_hub": plan.relative_hub,
                "plan_hash": plan.plan_hash,
                "expected_action": expected[name],
                "passed": plan.action == expected[name],
            }
        )
    return {
        "seed": seed,
        "cases": cases,
        "all_expected_routes": all(case["passed"] for case in cases),
    }


def _cached_cells(cells: tuple[Cell, ...], workers: int) -> list[dict[str, object]]:
    pending = []
    completed = []
    for cell in cells:
        path = _cell_path(cell)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CELL_SCHEMA:
                raise RuntimeError(f"cell schema drifted: {path}")
            completed.append(payload["result"])
        else:
            pending.append(cell)
    if workers == 1:
        rows = (_run_cell(cell) for cell in pending)
        for row in rows:
            completed.append(row)
            path = _cell_path(Cell(**row["cell"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"completed {row['cell']}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(_run_cell, cell): cell for cell in pending}
            for future in as_completed(futures):
                row = future.result()
                completed.append(row)
                path = _cell_path(Cell(**row["cell"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(f"completed {row['cell']}", flush=True)
    return sorted(completed, key=lambda row: tuple(row["cell"].values()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    cells = _cached_cells(ROUTE_CELLS, args.workers)
    witnesses = [_route_witness(seed) for seed in FRESH_WITNESS_SEEDS]
    checks = {
        "cell_count_3": len(cells) == 3,
        "phase1_exact": all(row["phase1_fes"] == 180_000 for row in cells),
        "terminal_exact": all(row["unified"]["terminal_fes"] == TOTAL_BUDGET_FES for row in cells),
        "strict_best": all(row["unified"]["strict_best"] for row in cells),
        "receipt_parity": all(row["unified"]["receipt_parity"] for row in cells),
        "state_hash_chain": all(
            row["unified"]["state_hash_chain"] and row["unified"]["receipt_count"] > 0
            for row in cells
        ),
        "route_witnesses_fresh": all(item["all_expected_routes"] for item in witnesses),
        "restricted_ctp_reachable": all(
            any(case["action"] == OC_ACTION_CTP_RESTRICTED for case in item["cases"])
            for item in witnesses
        ),
        "shared_core_ctp_reachable": all(
            any(case["action"] == OC_ACTION_CTP_SHARED_CORE for case in item["cases"])
            for item in witnesses
        ),
        "smp_reachable": all(
            any(case["action"] == OC_ACTION_SMP for case in item["cases"])
            for item in witnesses
        ),
        "aor_reachable": all(
            any(case["action"] == OC_ACTION_AOR for case in item["cases"])
            for item in witnesses
        ),
        "probe_amplitude_not_authority": all(
            next(
                case
                for case in item["cases"]
                if case["name"] == "probe_amplitude_only_arbitration"
            )["action"]
            == OC_ACTION_ARBITRATION
            for item in witnesses
        ),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "fresh_run_seed": FRESH_RUN_SEED,
            "fresh_witness_seeds": FRESH_WITNESS_SEEDS,
            "total_budget_fes": TOTAL_BUDGET_FES,
            "phase1_fes": 180_000,
            "config": asdict(OC_CONFIG),
            "note": "witness routes are plan-only reachability checks; benchmark action counts are reported separately",
        },
        "cells": cells,
        "route_witnesses": witnesses,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"checks": checks, "gate_passed": payload["gate_passed"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

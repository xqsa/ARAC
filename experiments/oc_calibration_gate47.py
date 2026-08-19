"""Gate 47: tau-threshold calibration for the unified loop's conflict bands."""

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
import math
from pathlib import Path

from arac.coordination import GcbDispatchConfig
from arac.coordination.contract import OcCoordinatorConfig
from arac.coordination.loop import _overlap_components, run_oc_unified
from arac.evidence import run_phase1_overlap_pilot
from arac.overlap_core import (
    GCB_COORDINATED_MODE,
    _proposal_budget,
    run_overlap_from_pilot,
)
from experiments.overlap_arac_gate29_screening import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    OVERLAP_BUDGETS,
    PHASE1_KWARGS,
    TOTAL_BUDGET_FES,
    TOPOLOGIES,
    Cell,
    build_cell,
)

PHASE_A_SEED = 20260834
PHASE_B_SEED = 20260835
NON_ESCALATING = OcCoordinatorConfig(tau_enter=2.0, tau_exit=1.0, pulse_min_fes=8, pulse_max_fes=32)
KERNEL_DISPATCH_CONFIG = GcbDispatchConfig(hub_mode="relative", complex_hub_ratio=0.9)
CELL_SCHEMA = "arac-oc-calibration-gate47-cell-v1"
OUTPUT_SCHEMA = "arac-oc-calibration-gate47-v1"
OUTPUT_ROOT = Path("artifacts/oc_calibration_gate47")


def jobs(seed: int) -> tuple[Cell, ...]:
    return tuple(
        Cell("conflicting", topology, overlap, seed)
        for topology in TOPOLOGIES
        for overlap in OVERLAP_BUDGETS
    )


def _cell_path(directory: Path, cell: Cell) -> Path:
    return directory / (
        f"{cell.mode}_{cell.topology}_overlap{cell.overlap_budget}_seed{cell.seed}.json"
    )


def _sense_budget(pilot) -> int:
    components = _overlap_components(pilot.adaptation.structure)
    return _proposal_budget(
        TOTAL_BUDGET_FES - pilot.checkpoint.phase1_fes,
        components,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
    )


def _unified_arms_row(cell: Cell, config: OcCoordinatorConfig) -> dict[str, object]:
    problem, truth = build_cell(cell)
    pilot = run_phase1_overlap_pilot(
        problem, total_budget_fes=TOTAL_BUDGET_FES, run_seed=cell.seed, **PHASE1_KWARGS
    )
    unified = run_oc_unified(
        problem,
        pilot,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        sense_budget_fes=_sense_budget(pilot),
        config=config,
    )
    return {
        "cell": {"mode": cell.mode, "topology": cell.topology,
                 "overlap_budget": cell.overlap_budget, "seed": cell.seed},
        "truth_shared_count": len(truth.structure.shared_variables),
        "phase1_fes": pilot.checkpoint.phase1_fes,
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "unified": {
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
            "max_probe_c": max((trace.probe_max_c for trace in unified.cycles), default=0.0),
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
        },
    }


def run_phase_a_cell(cell: Cell) -> dict[str, object]:
    row = _unified_arms_row(cell, NON_ESCALATING)
    row["phase"] = "A"
    return row


def run_phase_b_cell(cell: Cell, *, tau_enter: float, tau_exit: float) -> dict[str, object]:
    b_cell = cell
    problem, truth = build_cell(b_cell)
    pilot = run_phase1_overlap_pilot(
        problem, total_budget_fes=TOTAL_BUDGET_FES, run_seed=b_cell.seed, **PHASE1_KWARGS
    )
    calibrated = OcCoordinatorConfig(
        tau_enter=tau_enter,
        tau_exit=tau_exit,
        k_enter=2,
        k_exit=2,
        pulse_min_fes=8,
        pulse_max_fes=32,
        calibration_status="tau-calibrated-gate47",
    )
    row: dict[str, object] = {
        "cell": {"mode": b_cell.mode, "topology": b_cell.topology,
                 "overlap_budget": b_cell.overlap_budget, "seed": b_cell.seed},
        "phase": "B",
        "truth_shared_count": len(truth.structure.shared_variables),
        "phase1_fes": pilot.checkpoint.phase1_fes,
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "arms": {},
    }
    kernel = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=GCB_COORDINATED_MODE,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        dispatch_config=KERNEL_DISPATCH_CONFIG,
    )
    row["arms"]["coordinator"] = {
        "final_error": kernel.final_error,
        "terminal_fes": kernel.terminal_fes,
        "strict_best": all(
            cycle.best_error_after <= cycle.best_error_before for cycle in kernel.cycles
        ),
    }
    proposal = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode="proposal_neighborhood",
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
    )
    row["arms"]["proposal_neighborhood"] = {
        "final_error": proposal.final_error,
        "terminal_fes": proposal.terminal_fes,
        "strict_best": all(
            cycle.best_error_after <= cycle.best_error_before for cycle in proposal.cycles
        ),
    }
    unified = run_oc_unified(
        problem,
        pilot,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        sense_budget_fes=_sense_budget(pilot),
        config=calibrated,
    )
    row["arms"]["oc_unified"] = {
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
        "action_counts": {
            action: sum(1 for trace in unified.cycles if trace.action == action)
            for action in sorted({trace.action for trace in unified.cycles})
        },
        "budget_flow": {
            "sense": sum(trace.sense_fes for trace in unified.cycles),
            "probe": sum(trace.probe_fes for trace in unified.cycles),
            "arbitration": sum(trace.arbitration_fes for trace in unified.cycles),
            "operator": sum(trace.operator_fes for trace in unified.cycles),
            "tail": unified.tail_fes,
        },
    }
    return row


class _Calibration:
    """Filled after Phase A; deterministic under the pre-registered rule."""

    tau_enter: float = 0.5
    tau_exit: float = 0.2


CALIBRATION = _Calibration()


def _cached_run(directory: Path, cells, runner) -> list[dict[str, object]]:
    completed: list[dict[str, object]] = []
    pending = []
    for cell in cells:
        path = _cell_path(directory, cell)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CELL_SCHEMA:
                raise RuntimeError(f"cell schema drifted: {path}")
            completed.append(payload["result"])
        else:
            pending.append(cell)
    if pending:
        with ProcessPoolExecutor(max_workers=min(4, len(pending))) as executor:
            futures = {executor.submit(runner, cell): cell for cell in pending}
            for future in as_completed(futures):
                row = future.result()
                directory.mkdir(parents=True, exist_ok=True)
                _cell_path(directory, Cell(**row["cell"])).write_text(
                    json.dumps(
                        {"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True
                    ),
                    encoding="utf-8",
                )
                completed.append(row)
                print(f"completed {row['cell']}", flush=True)
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.parse_args()

    phase_a_dir = OUTPUT_ROOT / "phaseA"
    phase_b_dir = OUTPUT_ROOT / "phaseB"

    phase_a = _cached_run(phase_a_dir, jobs(PHASE_A_SEED), run_phase_a_cell)
    phase_a.sort(key=lambda row: (row["cell"]["topology"], row["cell"]["overlap_budget"]))
    positives = [row["unified"]["max_probe_c"] for row in phase_a if row["cell"]["topology"] == "chain"]
    negatives = [row["unified"]["max_probe_c"] for row in phase_a if row["cell"]["topology"] == "star"]
    gap_exists = min(positives) > max(negatives)
    if gap_exists:
        CALIBRATION.tau_enter = math.sqrt(min(positives) * max(negatives))
        CALIBRATION.tau_exit = CALIBRATION.tau_enter / 2.5
    print(
        json.dumps(
            {
                "phase_a_chain_max_c": positives,
                "phase_a_star_max_c": negatives,
                "gap_exists": gap_exists,
                "tau_enter": CALIBRATION.tau_enter,
                "tau_exit": CALIBRATION.tau_exit,
            },
            indent=1,
        ),
        flush=True,
    )
    if not gap_exists:
        payload = {
            "schema_version": OUTPUT_SCHEMA,
            "protocol": {"phase_a_seed": PHASE_A_SEED, "phase_b_seed": PHASE_B_SEED},
            "phase_a": phase_a,
            "calibration": {"gap_exists": False, "chain": positives, "star": negatives},
            "gate_passed": False,
            "failure": "tau_gap_absent",
        }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "confirmation.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("calibration failed: no tau gap between chain and star conflict scores")
        return 1

    from functools import partial

    phase_b_runner = partial(
        run_phase_b_cell, tau_enter=CALIBRATION.tau_enter, tau_exit=CALIBRATION.tau_exit
    )
    phase_b = _cached_run(phase_b_dir, jobs(PHASE_B_SEED), phase_b_runner)
    phase_b.sort(key=lambda row: (row["cell"]["topology"], row["cell"]["overlap_budget"]))
    tolerance = 1e-9
    protocol_checks = {
        "phaseA_cell_count_4": len(phase_a) == 4,
        "phaseB_cell_count_4": len(phase_b) == 4,
        "phase1_exact": all(row["phase1_fes"] == 180_000 for row in phase_a + phase_b),
        "terminal_exact": all(
            arm["terminal_fes"] == TOTAL_BUDGET_FES
            for row in phase_b
            for arm in row["arms"].values()
        ),
        "strict_best": all(
            arm["strict_best"] for row in phase_b for arm in row["arms"].values()
        ),
        "unified_receipts_ok": all(
            row["arms"]["oc_unified"]["receipt_parity"]
            and row["arms"]["oc_unified"]["state_hash_chain"]
            for row in phase_b
        ),
    }
    chain_rows = [row for row in phase_b if row["cell"]["topology"] == "chain"]
    star_rows = [row for row in phase_b if row["cell"]["topology"] == "star"]
    screening_checks = {
        "tau_gap_exists": gap_exists,
        "path_fires_on_chain": any(row["arms"]["oc_unified"]["operator_fes"] > 0 for row in chain_rows),
        "not_worse_than_kernel_all": all(
            row["arms"]["oc_unified"]["final_error"]
            <= row["arms"]["coordinator"]["final_error"] * 1.05 + tolerance
            for row in phase_b
        ),
        "star_no_regression_vs_proposal": all(
            row["arms"]["oc_unified"]["final_error"]
            <= row["arms"]["proposal_neighborhood"]["final_error"] + tolerance
            for row in star_rows
        ),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "phase_a_seed": PHASE_A_SEED,
            "phase_b_seed": PHASE_B_SEED,
            "phase1_kwargs": PHASE1_KWARGS,
            "non_escalating_config": asdict(NON_ESCALATING),
            "calibrated_config": asdict(
                OcCoordinatorConfig(
                    tau_enter=CALIBRATION.tau_enter,
                    tau_exit=CALIBRATION.tau_exit,
                    calibration_status="tau-calibrated-gate47",
                )
            ),
        },
        "phase_a": phase_a,
        "phase_b": phase_b,
        "calibration": {
            "gap_exists": gap_exists,
            "chain_max_c": positives,
            "star_max_c": negatives,
            "tau_enter": CALIBRATION.tau_enter,
            "tau_exit": CALIBRATION.tau_exit,
        },
        "protocol_checks": protocol_checks,
        "screening_checks": screening_checks,
        "gate_passed": all(protocol_checks.values()) and all(screening_checks.values()),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"protocol": protocol_checks, "screening": screening_checks}, indent=1))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Gate 48b: paired AOB pilot with a bounded sense budget.

Each case creates one v3 soft-RDDSM Phase-I checkpoint, then runs two Phase-II
arms from that exact incumbent: the ARAC-OC unified loop and a full-budget MMES
control.  The control is intentionally not the historical 41b arm; it exists
to isolate the effect of the unified loop without changing the Phase-I start.
"""

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

import numpy as np

from arac.benchmarks.aob import AobBenchmark
from arac.coordination.contract import OcCoordinatorConfig, canonical_sha256
from arac.coordination.loop import _overlap_components, run_oc_unified_from_structure
from arac.evidence.hierarchical import to_overlap_structure
from arac.evidence.soft_rddsm import SoftDsmConfig, discover_hierarchical_soft
from arac.overlap_core import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    capped_proposal_budget,
)
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort

CASES = ("R2", "A3", "S5", "R6")
PILOT_SEED = 20260845
PHASE1_FES = 180_000
TOTAL_FES = 3_000_000
PHASE2_FES = TOTAL_FES - PHASE1_FES
SENSE_BUDGET_SHARE = 0.45
CELL_SCHEMA = "arac-oc-gate48b-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate48b-pilot-v1"
OUTPUT_ROOT = Path("artifacts/oc_aob_paired_gate48b")


def _truth_groups(data_root: Path, function_id: int) -> list[frozenset[int]]:
    import yaml

    stem = f"F{function_id}"
    with (data_root / f"{stem}-info.txt").open(encoding="utf-8") as handle:
        info = yaml.safe_load(handle)
    sizes = np.loadtxt(data_root / f"{stem}-s.txt").astype(int).tolist()
    permutation = np.loadtxt(data_root / f"{stem}-p.txt", delimiter=",").astype(int).tolist()
    overlap = int(info["overlap_degree"])
    groups: list[frozenset[int]] = []
    begin = 0
    for index, size in enumerate(sizes):
        end = begin + int(size)
        groups.append(frozenset(int(value) - 1 for value in permutation[begin:end]))
        if index != len(sizes) - 1:
            begin = end - overlap
    return groups


def _phase1(case_id: str, *, sense_budget_share: float = SENSE_BUDGET_SHARE):
    problem = AobBenchmark().load(case_id)
    truth_groups = _truth_groups(AobBenchmark().data_root, int(case_id[1]))
    membership = np.zeros(problem.dimension, dtype=int)
    for group in truth_groups:
        membership[sorted(group)] += 1
    truth_shared = frozenset(int(value) for value in np.nonzero(membership > 1)[0])

    ledger = EvaluationLedger(problem, PHASE1_FES)
    discovery = discover_hierarchical_soft(
        problem,
        ledger,
        run_seed=PILOT_SEED,
        config=SoftDsmConfig(),
    )
    discovery_fes = ledger.count
    if ledger.remaining:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(value) for value in ledger.best_x),
            sigma=0.5,
            seed=PILOT_SEED ^ 0x1D_E71D,
            budget_fes=ledger.remaining,
            population_size=24,
            restart=False,
        )
    if ledger.count != PHASE1_FES:
        raise RuntimeError(f"{case_id}: Phase-I ended at {ledger.count}, expected {PHASE1_FES}")

    structure = to_overlap_structure(discovery.evidence)
    components = _overlap_components(structure)
    checkpoint_hash = canonical_sha256(
        {
            "case_id": case_id,
            "structure_groups": [list(group) for group in structure.groups],
            "structure_members": [list(item) for item in structure.member_confidences],
            "phase1_fes": PHASE1_FES,
            "incumbent": [float(value) for value in ledger.best_x],
            "incumbent_error": float(ledger.best_error),
            "run_seed": PILOT_SEED,
        }
    )
    recovered = frozenset(discovery.shared_candidates)
    recall = len(recovered & truth_shared) / len(truth_shared)
    precision = len(recovered & truth_shared) / len(recovered) if recovered else None
    sense_budget = capped_proposal_budget(
        PHASE2_FES,
        components,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        sense_budget_share=sense_budget_share,
    )
    return {
        "problem": problem,
        "truth_group_count": len(truth_groups),
        "truth_shared_count": len(truth_shared),
        "shared_recall": recall,
        "shared_precision": precision,
        "discovery_fes": discovery_fes,
        "structure": structure,
        "components": components,
        "checkpoint_hash": checkpoint_hash,
        "incumbent": tuple(float(value) for value in ledger.best_x),
        "incumbent_error": float(ledger.best_error),
        "sense_budget": sense_budget,
    }


def _run_control(
    problem,
    incumbent,
    incumbent_error,
    *,
    checkpoint_hash: str,
    seed: int,
) -> dict[str, object]:
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=incumbent,
        incumbent_error=incumbent_error,
    )
    before = float(ledger.best_error)
    PypopOptimizerPort().run(
        "mmes",
        problem=problem,
        ledger=ledger,
        initial_mean=tuple(float(value) for value in ledger.best_x),
        sigma=0.5,
        seed=seed,
        budget_fes=PHASE2_FES,
        population_size=24,
        restart=False,
    )
    if ledger.count != TOTAL_FES:
        raise RuntimeError("paired MMES control did not consume exact terminal FE")
    return {
        "checkpoint_hash": checkpoint_hash,
        "phase2_fes": PHASE2_FES,
        "final_error": float(ledger.best_error),
        "best_error_before": before,
        "terminal_fes": ledger.count,
    }


def run_case(
    case_id: str,
    *,
    sense_budget_share: float = SENSE_BUDGET_SHARE,
    oc_config: OcCoordinatorConfig | None = None,
) -> dict[str, object]:
    phase = _phase1(case_id, sense_budget_share=sense_budget_share)
    problem = phase["problem"]
    incumbent = phase["incumbent"]
    incumbent_error = phase["incumbent_error"]
    control = _run_control(
        problem,
        incumbent,
        incumbent_error,
        checkpoint_hash=phase["checkpoint_hash"],
        seed=PILOT_SEED ^ 0xC011_7A,
    )
    unified = run_oc_unified_from_structure(
        problem,
        phase["structure"],
        checkpoint_hash=phase["checkpoint_hash"],
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=incumbent,
        incumbent_error=incumbent_error,
        run_seed=PILOT_SEED,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        sense_budget_fes=phase["sense_budget"],
        config=(
            OcCoordinatorConfig(pulse_min_fes=8, pulse_max_fes=32)
            if oc_config is None
            else oc_config
        ),
    )
    action_counts: dict[str, int] = {}
    for trace in unified.cycles:
        action_counts[trace.action] = action_counts.get(trace.action, 0) + 1
    flow = {
        "sense": sum(trace.sense_fes for trace in unified.cycles),
        "smp_writeback": sum(trace.smp_fes for trace in unified.cycles),
        "probe": sum(trace.probe_fes for trace in unified.cycles),
        "arbitration": sum(trace.arbitration_fes for trace in unified.cycles),
        "operator": sum(trace.operator_fes for trace in unified.cycles),
        "tail": unified.tail_fes,
    }
    operator_fired = flow["operator"] > 0
    return {
        "case_id": case_id,
        "phase1_fes": PHASE1_FES,
        "discovery_fes": phase["discovery_fes"],
        "phase1_incumbent_error": incumbent_error,
        "checkpoint_hash": phase["checkpoint_hash"],
        "structure_groups": len(phase["structure"].groups),
        "structure_shared": len(phase["structure"].shared_variables),
        "components": [list(component) for component in phase["components"]],
        "truth_group_count": phase["truth_group_count"],
        "truth_shared_count": phase["truth_shared_count"],
        "shared_recall": phase["shared_recall"],
        "shared_precision": phase["shared_precision"],
        "sense_budget_fes_per_group": phase["sense_budget"],
        "sense_budget_share": sense_budget_share,
        "control": control,
        "unified": {
            "checkpoint_hash": phase["checkpoint_hash"],
            "final_error": unified.final_error,
            "terminal_fes": unified.terminal_fes,
            "final_state_hash": unified.final_state_hash,
            "action_counts": action_counts,
            "operator_fired": operator_fired,
            "budget_flow": flow,
            "strict_best": all(
                trace.best_error_after <= trace.best_error_before for trace in unified.cycles
            ),
            "receipt_parity": all(
                receipt.actual_fes == receipt.reserved_fes for receipt in unified.receipts
            ),
            "state_hash_chain": bool(unified.receipts)
            and all(len(receipt.state_hash) == 64 for receipt in unified.receipts),
        },
        "paired_final_delta_control_minus_unified": (
            float(control["final_error"]) - float(unified.final_error)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cells_dir = OUTPUT_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    pending = [case for case in CASES if not (cells_dir / f"{case}.json").exists()]
    for case in CASES:
        path = cells_dir / f"{case}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8"))["result"])
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_case, case): case for case in pending}
            for future in as_completed(futures):
                row = future.result()
                (cells_dir / f"{row['case_id']}.json").write_text(
                    json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{row['case_id']}: checkpoint={row['phase1_incumbent_error']:.4g} "
                    f"control={row['control']['final_error']:.4g} "
                    f"unified={row['unified']['final_error']:.4g} "
                    f"operator_fes={row['unified']['budget_flow']['operator']}",
                    flush=True,
                )
    rows.sort(key=lambda row: row["case_id"])
    checks = {
        "paired_checkpoint_all": all(
            row["checkpoint_hash"]
            == row["control"]["checkpoint_hash"]
            == row["unified"]["checkpoint_hash"]
            for row in rows
        ) and len({row["checkpoint_hash"] for row in rows}) == len(rows),
        "phase1_exact_all": all(row["phase1_fes"] == PHASE1_FES for row in rows),
        "terminal_exact_all": all(
            row["control"]["terminal_fes"] == TOTAL_FES
            and row["unified"]["terminal_fes"] == TOTAL_FES
            for row in rows
        ),
        "strict_best_all": all(row["unified"]["strict_best"] for row in rows),
        "receipts_valid_all": all(
            row["unified"]["receipt_parity"] and row["unified"]["state_hash_chain"]
            for row in rows
        ),
        "operator_fired_all": all(row["unified"]["operator_fired"] for row in rows),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "pilot_seed": PILOT_SEED,
            "phase1_fes": PHASE1_FES,
            "total_fes": TOTAL_FES,
            "sense_budget_share": SENSE_BUDGET_SHARE,
            "loop_config": asdict(OcCoordinatorConfig(pulse_min_fes=8, pulse_max_fes=32)),
            "control": "same Phase-I checkpoint, full Phase-II MMES",
            "reference": "Gate48a/41b are report-only, not paired gates",
        },
        "checks": checks,
        "pilot_passed": all(checks.values()),
        "rows": rows,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "pilot.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=1))
    return 0 if payload["pilot_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

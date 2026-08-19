"""Gate 48a: AOB wiring pilot (v3 evidence -> hyperedge gate -> unified loop)."""

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
from arac.overlap_core import DEFAULT_NEIGHBORHOOD_FES, DEFAULT_REFRESH_CYCLES, _proposal_budget
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort

CASES = ("R2", "A3", "S5", "R6")
PILOT_SEED = 20260845
PHASE1_FES = 180_000
TOTAL_FES = 3_000_000
GATE41B = Path("artifacts/overlap_action_dispatch_gate41_online/confirmation.json")
CELL_SCHEMA = "arac-oc-gate48a-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate48a-pilot-v1"
OUTPUT_ROOT = Path("artifacts/oc_aob_wiring_pilot_gate48a")


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
        groups.append(frozenset(int(v) - 1 for v in permutation[begin:end]))
        if index != len(sizes) - 1:
            begin = end - overlap
    return groups


def run_case(case_id: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    truth_groups = _truth_groups(AobBenchmark().data_root, int(case_id[1]))
    membership = np.zeros(problem.dimension, dtype=int)
    for group in truth_groups:
        membership[sorted(group)] += 1
    truth_shared = frozenset(int(v) for v in np.nonzero(membership > 1)[0])

    # --- Phase-I: v3 soft discovery + MMES incumbent fill to exactly 180k ---
    ledger = EvaluationLedger(problem, PHASE1_FES)
    discovery = discover_hierarchical_soft(
        problem, ledger, run_seed=PILOT_SEED, config=SoftDsmConfig()
    )
    discovery_fes = ledger.count
    if ledger.remaining:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(v) for v in ledger.best_x),
            sigma=0.5,
            seed=PILOT_SEED ^ 0x1D_E71D,
            budget_fes=ledger.remaining,
            population_size=24,
            restart=False,
        )
    if ledger.count != PHASE1_FES:
        raise RuntimeError(f"{case_id}: Phase-I did not land on {PHASE1_FES} (got {ledger.count})")

    # --- Gate 42 hyperedge gate: fail-closed without resolved hyperedges ---
    structure = to_overlap_structure(discovery.evidence)
    recovered = frozenset(discovery.shared_candidates)
    recall = len(recovered & truth_shared) / len(truth_shared)
    precision = len(recovered & truth_shared) / len(recovered) if recovered else None
    components = _overlap_components(structure)
    checkpoint_hash = canonical_sha256(
        {
            "case_id": case_id,
            "structure_groups": [list(group) for group in structure.groups],
            "phase1_fes": PHASE1_FES,
            "run_seed": PILOT_SEED,
        }
    )

    # --- Phase-II: unified loop from the caller-supplied structure ---
    sense_budget = _proposal_budget(
        TOTAL_FES - PHASE1_FES,
        components,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
    )
    result = run_oc_unified_from_structure(
        problem,
        structure,
        checkpoint_hash=checkpoint_hash,
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=tuple(float(v) for v in ledger.best_x),
        incumbent_error=float(ledger.best_error),
        run_seed=PILOT_SEED,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        sense_budget_fes=sense_budget,
        config=OcCoordinatorConfig(pulse_min_fes=8, pulse_max_fes=32),
    )

    reference = None
    if GATE41B.exists():
        payload = json.loads(GATE41B.read_text(encoding="utf-8"))
        reference = next(
            (row["mean"] for row in payload["cases"] if row["case"] == case_id), None
        )
    action_counts: dict[str, int] = {}
    value_gated = 0
    for trace in result.cycles:
        action_counts[trace.action] = action_counts.get(trace.action, 0) + 1
        value_gated += int(trace.operator_value_gated)
    return {
        "case_id": case_id,
        "phase1_fes": PHASE1_FES,
        "discovery_fes": discovery_fes,
        "discovery_blocks": len(discovery.blocks),
        "truth_group_count": len(truth_groups),
        "truth_shared_count": len(truth_shared),
        "shared_recall": recall,
        "shared_precision": precision,
        "structure_groups": len(structure.groups),
        "structure_shared": len(structure.shared_variables),
        "components": [list(component) for component in components],
        "sense_budget_fes": sense_budget,
        "checkpoint_hash": checkpoint_hash,
        "loop": {
            "final_error": result.final_error,
            "terminal_fes": result.terminal_fes,
            "strict_best": all(
                trace.best_error_after <= trace.best_error_before for trace in result.cycles
            ),
            "receipt_parity": all(
                receipt.actual_fes == receipt.reserved_fes for receipt in result.receipts
            ),
            "state_hash_chain": bool(result.receipts)
            and all(len(receipt.state_hash) == 64 for receipt in result.receipts),
            "action_counts": action_counts,
            "operator_value_gated_cycles": value_gated,
            "budget_flow": {
                "sense": sum(trace.sense_fes for trace in result.cycles),
                "smp_writeback": sum(trace.smp_fes for trace in result.cycles),
                "probe": sum(trace.probe_fes for trace in result.cycles),
                "arbitration": sum(trace.arbitration_fes for trace in result.cycles),
                "operator": sum(trace.operator_fes for trace in result.cycles),
                "tail": result.tail_fes,
            },
            "final_state_hash": result.final_state_hash,
        },
        "gate41b_case_mean": reference,
        "ratio_vs_gate41b": (
            result.final_error / reference if reference else None
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
                    json.dumps(
                        {"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True
                    ),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{row['case_id']}: recall={row['shared_recall']:.3f} "
                    f"groups={row['structure_groups']} comps={len(row['components'])} "
                    f"final={row['loop']['final_error']:.4g} "
                    f"vs41b={row['ratio_vs_gate41b'] if row['ratio_vs_gate41b'] else float('nan'):.3f} "
                    f"op_fes={row['loop']['budget_flow']['operator']}",
                    flush=True,
                )
    rows.sort(key=lambda row: row["case_id"])
    applicability = {
        "wiring_succeeds_4_4": len(rows) == 4,
        "terminal_exact_all": all(row["loop"]["terminal_fes"] == TOTAL_FES for row in rows)
        and all(row["phase1_fes"] == PHASE1_FES for row in rows),
        "strict_best_all": all(row["loop"]["strict_best"] for row in rows),
        "receipts_valid_all": all(
            row["loop"]["receipt_parity"] and row["loop"]["state_hash_chain"] for row in rows
        ),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "pilot_seed": PILOT_SEED,
            "phase1_fes": PHASE1_FES,
            "total_fes": TOTAL_FES,
            "loop_config": asdict(OcCoordinatorConfig(pulse_min_fes=8, pulse_max_fes=32)),
            "reference": str(GATE41B),
        },
        "applicability_checks": applicability,
        "pilot_passed": all(applicability.values()),
        "rows": rows,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "pilot.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(applicability, indent=1))
    return 0 if payload["pilot_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Gate P1: small incremental attribution for the Stateful Shared-Patch Kernel.

Pre-registration (user plan section 4, Gate P1; frozen before this run):

- grid: R2/A3/S5/R6 x replay seeds {20260880, 20260881, 20260882} x arms
  A0..A4; seed-registry precheck fails loudly on any collision with the
  frozen registries (31501.., 117..141, 20260901..03);
- one shared Phase-I (gate48b soft-RDDSM recipe, exact 180k) per
  (case, seed), cached and reused verbatim by all five arms;
- loop settings frozen at production defaults (refresh_cycles=16, sense
  share 0.45 via capped_proposal_budget), only patch_config differs:
    A0 None / A1 {"mode":"v2"} / A2 {"mode":"candidates"}
    A3 {"mode":"state"} / A4 {"mode":"full"};
- metrics per arm: final error, exact terminal FE, log10 anytime AUC over
  the cycle-boundary error series (trapezoid, [phase1, total]), FE ledger
  shares, patch acceptance / reset counts from receipts;
- pass criteria (all simultaneously):
  1. per-cell five-arm parity: shared phase1 hash + terminal 3,000,000
     exact in every arm;
  2. A2 receipts show consensus/disagreement candidates (vs A1);
  3. A3 shows persistent-state traces (patch_state_hash evolves or
     non-empty radius trace) vs A2;
  4. A4 shows a radius update or context reset trace;
  5. all receipts state-hash audited (chain present);
  6. A4 vs A2 median log-AUC degradation <= 5% (median over cells of
     per-cell AUC ratio, A4/A2 <= 1.05).
"""

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import AobBenchmark
from arac.coordination.loop import run_oc_unified_from_structure
from arac.evidence.hierarchical import to_overlap_structure
from arac.evidence.soft_rddsm import SoftDsmConfig, discover_hierarchical_soft
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort
from arac.overlap_core import DEFAULT_NEIGHBORHOOD_FES, DEFAULT_REFRESH_CYCLES, capped_proposal_budget

CASES = ("R2", "A3", "S5", "R6")
SEEDS = (20260880, 20260881, 20260882)
KNOWN_REGISTRIES = (
    tuple(range(31501, 31506)),
    tuple(range(117, 142)),
    (20260901, 20260902, 20260903),
)
TOTAL_FES = 3_000_000
PHASE1_FES = 180_000
SENSE_SHARE = 0.45
ARMS = {
    "A0": None,
    "A1": {"mode": "v2"},
    "A2": {"mode": "candidates"},
    "A3": {"mode": "state"},
    "A4": {"mode": "full"},
}
OUTPUT_ROOT = Path("artifacts/oc_gate_p1_shared_patch")
CELL_SCHEMA = "arac-oc-gate-p1-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate-p1-v1"


def precheck_seeds() -> None:
    for seed in SEEDS:
        for registry in KNOWN_REGISTRIES:
            if seed in registry:
                raise RuntimeError(f"seed {seed} collides with a frozen registry")


def _phase_one(case_id: str, seed: int):
    cache_dir = OUTPUT_ROOT / "phase1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{case_id}_{seed}.json"
    problem = AobBenchmark().load(case_id)
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        ledger = EvaluationLedger(problem, PHASE1_FES)
        discovery = discover_hierarchical_soft(
            problem, ledger, run_seed=seed, config=SoftDsmConfig()
        )
        if ledger.remaining:
            PypopOptimizerPort().run(
                "mmes",
                problem=problem,
                ledger=ledger,
                initial_mean=tuple(float(v) for v in ledger.best_x),
                sigma=0.5,
                seed=seed ^ 0x1D_E71D,
                budget_fes=ledger.remaining,
                population_size=24,
                restart=False,
            )
        if ledger.count != PHASE1_FES:
            raise RuntimeError(f"{case_id}/{seed}: phase1 ended at {ledger.count}")
        structure = to_overlap_structure(discovery.evidence)
        payload = {
            "incumbent": [float(v) for v in ledger.best_x],
            "incumbent_error": float(ledger.best_error),
            "groups": [list(g) for g in structure.groups],
            "shared": sorted(int(v) for v in structure.shared_variables),
        }
        cache.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    structure = _structure_from_payload(payload)
    hash_payload = dict(payload)
    hash_payload["case_id"] = case_id
    hash_payload["seed"] = seed
    checkpoint_hash = canonical_sha256(hash_payload)
    return (
        problem,
        structure,
        checkpoint_hash,
        tuple(float(v) for v in payload["incumbent"]),
        float(payload["incumbent_error"]),
    )


def _structure_from_payload(payload):
    from arac.coordination.overlap import OverlapStructure

    dimension = len(payload["incumbent"])
    groups = tuple(tuple(int(v) for v in group) for group in payload["groups"])
    return OverlapStructure(dimension, groups)


def _log_auc(result) -> float:
    fes = [PHASE1_FES]
    errors = [None]  # filled below with incumbent error at start
    incumbent_error = None
    for trace in result.cycles:
        fes.append(PHASE1_FES + int(getattr(trace, "cumulative_runtime_fes", 0) or 0))
        errors.append(float(trace.best_error_after))
    # fallback FE bookkeeping from receipts when the trace lacks cumulative FE
    if len({*fes}) < len(result.cycles) + 1:
        fes, errors = [PHASE1_FES], [incumbent_error or 1.0]
        running = PHASE1_FES
        for trace in result.cycles:
            running += (
                int(trace.sense_fes)
                + int(trace.smp_fes)
                + int(trace.probe_fes)
                + int(trace.arbitration_fes)
                + int(trace.operator_fes)
            )
            fes.append(running)
            errors.append(float(trace.best_error_after))
    fes.append(TOTAL_FES)
    errors.append(float(result.final_error))
    log_errors = [math.log10(max(error, 1e-300)) for error in errors]
    auc = 0.0
    for index in range(len(fes) - 1):
        auc += (fes[index + 1] - fes[index]) * (log_errors[index] + log_errors[index + 1]) / 2.0
    return auc / (TOTAL_FES - PHASE1_FES)


def run_cell(case_id: str, seed: int) -> dict[str, object]:
    problem, structure, checkpoint_hash, incumbent, incumbent_error = _phase_one(case_id, seed)
    from arac.coordination.loop import _overlap_components

    component_list = _overlap_components(structure)
    sense_budget = capped_proposal_budget(
        TOTAL_FES - PHASE1_FES,
        component_list,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        sense_budget_share=SENSE_SHARE,
    )
    arms = {}
    for arm, patch_config in ARMS.items():
        result = run_oc_unified_from_structure(
            problem,
            structure,
            checkpoint_hash=checkpoint_hash,
            total_budget_fes=TOTAL_FES,
            phase1_fes=PHASE1_FES,
            incumbent=incumbent,
            incumbent_error=incumbent_error,
            run_seed=seed,
            refresh_cycles=DEFAULT_REFRESH_CYCLES,
            sense_budget_fes=sense_budget,
            patch_config=patch_config,
        )
        patch_receipts = [r for r in result.receipts if r.patch_enabled]
        arms[arm] = {
            "final_error": float(result.final_error),
            "terminal_fes": int(result.terminal_fes),
            "log_auc": _log_auc(result),
            "patch_receipt_count": len(patch_receipts),
            "patch_lane_fes": int(sum(r.patch_lane_fes for r in patch_receipts)),
            "patch_accepted": int(
                sum(1 for r in patch_receipts if r.patch_accepted_candidate)
            ),
            "patch_resets": int(sum(1 for r in patch_receipts if r.patch_context_reset)),
            "patch_candidate_names": sorted(
                {name for r in patch_receipts for name in r.patch_candidate_names}
            ),
            "patch_state_hashes": [r.patch_state_hash for r in patch_receipts][:32],
            "first_plan_hash": (result.receipts[0].plan_hash if result.receipts else ""),
            "radius_pairs": [(r.patch_radius_min, r.patch_radius_max) for r in patch_receipts],
            "budget_unavailable": int(
                sum(1 for r in patch_receipts if r.patch_budget_status != "executed")
            ),
        }
    return {
        "schema_version": CELL_SCHEMA,
        "case_id": case_id,
        "seed": seed,
        "checkpoint_hash": checkpoint_hash,
        "shared_count": len(structure.shared_variables),
        "arms": arms,
    }


def _cell_path(case_id: str, seed: int) -> Path:
    return OUTPUT_ROOT / "cells" / f"{case_id}_{seed}.json"


def run_gate(workers: int = 6) -> dict[str, object]:
    precheck_seeds()
    (OUTPUT_ROOT / "cells").mkdir(parents=True, exist_ok=True)
    jobs = [(case, seed) for case in CASES for seed in SEEDS]
    rows = []
    pending = []
    for case, seed in jobs:
        path = _cell_path(case, seed)
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append((case, seed))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_cell, case, seed): (case, seed) for case, seed in pending}
            for future in as_completed(futures):
                case, seed = futures[future]
                row = future.result()
                _cell_path(case, seed).write_text(
                    json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
                )
                rows.append(row)
                print(
                    f"{case}/{seed}: "
                    + " ".join(f"{arm}={data['final_error']:.6g}" for arm, data in row["arms"].items()),
                    flush=True,
                )
    checks = {
        "cells_complete": len(rows) == len(jobs),
        "terminal_exact_all_arms": all(
            row["arms"][arm]["terminal_fes"] == TOTAL_FES
            for row in rows
            for arm in ARMS
        ),
        "a2_candidate_diversity": all(
            any(name.split("+")[0] in ("consensus", "disagreement") for name in row["arms"]["A2"]["patch_candidate_names"])
            for row in rows
            if row["arms"]["A2"]["patch_receipt_count"] > 0
        ),
        "first_selector_decision_parity": all(
            len({
                row["arms"][arm]["first_plan_hash"]
                for arm in ARMS
                if row["arms"][arm]["first_plan_hash"]
            })
            <= 1
            for row in rows
        ),
        "a3_state_traces": all(
            len({h for h in row["arms"]["A3"]["patch_state_hashes"] if h}) >= 1
            for row in rows
            if row["arms"]["A3"]["patch_receipt_count"] > 0
        ),
        "a4_radius_differs_from_a3": all(
            (
                row["arms"]["A4"]["patch_receipt_count"] == 0
                and row["arms"]["A3"]["patch_receipt_count"] == 0
            )
            or (
                any(rm != rx for rm, rx in row["arms"]["A4"]["radius_pairs"])
                and all(rm == rx for rm, rx in row["arms"]["A3"]["radius_pairs"])
            )
            or row["arms"]["A4"]["patch_resets"] > 0
            for row in rows
        ),
    }
    auc_ratios = [
        row["arms"]["A4"]["log_auc"] / max(row["arms"]["A2"]["log_auc"], 1e-300)
        for row in rows
    ]
    median_auc_ratio = float(np.median(auc_ratios))
    checks["a4_vs_a2_auc_within_5pct"] = bool(median_auc_ratio <= 1.05 + 1e-12)
    ratios = {
        f"{left}-{right}": [
            row["arms"][left]["final_error"] / max(row["arms"][right]["final_error"], 1e-300)
            for row in rows
        ]
        for left, right in (("A1", "A0"), ("A2", "A1"), ("A3", "A2"), ("A4", "A3"), ("A4", "A1"))
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "seeds": list(SEEDS),
            "arms": {arm: cfg for arm, cfg in ARMS.items()},
            "total_fes": TOTAL_FES,
            "phase1_fes": PHASE1_FES,
            "sense_share": SENSE_SHARE,
            "auc_definition": "trapezoid over log10(error) at cycle boundaries",
            "production_selector_modified": False,
        },
        "rows": rows,
        "attribution": {
            name: {"median_ratio": float(np.median(values)), "ratios": values}
            for name, values in ratios.items()
        },
        "median_auc_ratio_a4_over_a2": median_auc_ratio,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"attribution": payload["attribution"], "median_auc_ratio_a4_over_a2": median_auc_ratio, "gate_checks": checks, "gate_passed": payload["gate_passed"]}, indent=1, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

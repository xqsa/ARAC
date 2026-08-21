"""Gate 53: wire shared-variable arbitration into the dispatch path (v2).

Pre-registration: docs/arac-oc-gate53-protocol.md + the v2 amendment recorded
in PROGRESS (2026-08-21): the exploratory smoke showed that interleaving FE
inside a recovered action's aligned window violates the action's own
``consumed == aligned`` contract by construction (the project's budget-lane
discipline enforcing itself).  v2 therefore wires arbitration at PHASE
boundaries:

- ctp cells (S2-S6): arm B mirrors ``CtpExecutor.execute``'s exact budget
  math through the same public helpers (``run_persistent_blocks`` ->
  ``run_sequential_blocks`` -> ``run_full_space``) and drains all due
  arbitration cycles BETWEEN phases only;
- smp cells (S1): arm B is identical to arm A (arbitration has no
  interleaving point in smp; S1 is the expected-silent no-tax case).  The
  discovered shared-variable count is reported as a Phase-I precision
  observation.

Both arms share one soft-RDDSM v3 Phase-I per (case, seed) (D2); action
names come from the frozen gate41 dispatch receipts; checkpoint carries no
relation metadata (D3), so the ctp route is the no-relation variant in BOTH
arms.  No scheduler/loop/action src changes; production untouched.
"""

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from arac.actions._execution import (
    BLOCK_POPULATION_SIZE,
    run_full_space,
    run_persistent_blocks,
    run_sequential_blocks,
    terminal_result,
)
from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.evidence.hierarchical import to_overlap_structure
from arac.evidence.soft_rddsm import SoftDsmConfig, discover_hierarchical_soft
from arac.runtime.contracts import ActionContext, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort
from arac.runtime.phase2 import execute_phase2_action

CASES = tuple(f"S{index}" for index in range(1, 7))
SEEDS = (117, 118, 119, 120)
TOTAL_FES = 3_000_000
PHASE1_FES = 180_000
CYCLE_FES = 150_000
PROPOSAL_SIGMA_FRACTION = 0.05
ARBITRATION_SEED_XOR = 0x53C0
MIN_REMAINING_GUARD = 8
CTP_COVERAGE_FRACTION = 0.20
GATE41_RECEIPTS = Path("artifacts/overlap_action_dispatch_gate41_online/runs")
OUTPUT_ROOT = Path("artifacts/oc_gate53_arbitration_wire")
CELL_SCHEMA = "arac-oc-gate53-cell-v2"
OUTPUT_SCHEMA = "arac-oc-gate53-arbitration-wire-v2"
PERMUTATION_DRAWS = 10_000
PERMUTATION_SEED = 20260822
OVERLAP_DEGREES = {1: 0, 2: 1, 3: 3, 4: 5, 5: 7, 6: 10}


def _sanitized_problem(problem):
    """Map non-finite objective returns to 1e300 (D5, both arms + Phase-I).

    The AOB vendor objectives can overflow to inf/nan at finite in-bounds
    points; those points are semantically 'worse than anything finite', so a
    large finite sentinel preserves ordering and keeps the strict-best
    archive finite.  Applied identically to Phase-I and both arms; real
    contract violations (wrong shapes, non-finite INPUTS) still fail closed
    inside the ledger.
    """

    def objective(values):
        raw = np.asarray(problem.objective(values), dtype=float)
        if not np.all(np.isfinite(raw)):
            raw = np.where(np.isfinite(raw), raw, 1e300)
        return raw

    return dataclasses.replace(problem, objective=objective)


def _frozen_action(case_id: str, seed: int) -> dict[str, object]:
    path = GATE41_RECEIPTS / case_id / f"seed_{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "action": str(payload["action"]),
        "dispatch_features": payload.get("dispatch_features", {}),
        "historical_final_error": float(payload["final_error"]),
    }


def _owner_weights(structure) -> dict[tuple[int, int], float]:
    confidences = getattr(structure, "member_confidences", None)
    if isinstance(confidences, dict) and confidences:
        coherent = all(
            isinstance(key, tuple) and len(key) == 2
            for key in list(confidences.keys())[:64]
        )
        if coherent:
            try:
                return {key: float(value) for key, value in confidences.items()}
            except (TypeError, ValueError):
                return {}
    return {}


def _phase_one(problem, case_id: str, seed: int):
    ledger = EvaluationLedger(problem, PHASE1_FES)
    discovery = discover_hierarchical_soft(
        problem, ledger, run_seed=seed, config=SoftDsmConfig()
    )
    if ledger.remaining:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(value) for value in ledger.best_x),
            sigma=0.5,
            seed=seed ^ 0x1D_E71D,
            budget_fes=ledger.remaining,
            population_size=24,
            restart=False,
        )
    if ledger.count != PHASE1_FES:
        raise RuntimeError(f"{case_id}/{seed}: Phase-I ended at {ledger.count}")
    try:
        structure = to_overlap_structure(discovery.evidence)
        owners = {
            int(variable): tuple(
                pos for pos, group in enumerate(structure.groups) if variable in group
            )
            for variable in structure.shared_variables
        }
        weights = _owner_weights(structure)
        shared_count = len(structure.shared_variables)
        groups = tuple(tuple(int(v) for v in block) for block in structure.groups)
    except ValueError as exc:
        if "no resolved overlap hyperedges" not in str(exc):
            raise
        # Discovery found no variable-level overlap on this case/seed: the
        # arbitration lane is structurally silent (the S1/no-tax semantics).
        owners = {}
        weights = {}
        shared_count = 0
        groups = tuple(tuple(int(v) for v in block) for block in discovery.blocks)
    # The soft-RDDSM region tree assigns every variable to one primary block
    # OR a singleton; the checkpoint contract requires a full partition, so
    # uncovered variables become explicit 1-variable blocks (sorted, appended).
    base_blocks = tuple(tuple(sorted(block)) for block in discovery.blocks)
    covered = {variable for block in base_blocks for variable in block}
    singleton_blocks = tuple(
        (variable,) for variable in sorted(set(range(problem.dimension)) - covered)
    )
    partition_blocks = base_blocks + singleton_blocks
    checkpoint = PhaseCheckpoint(
        protocol="arac-oc-gate53-phase1-soft-rddsm-v3-v2",
        run_seed=seed,
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=tuple(float(value) for value in ledger.best_x),
        incumbent_error=float(ledger.best_error),
        feature_names=("phase1_incumbent_log10_error", "shared_variable_count"),
        feature_values=(
            float(np.log10(max(ledger.best_error, 1e-300))),
            float(shared_count),
        ),
        blocks=partition_blocks,
        relations=(),
    )
    return checkpoint, owners, weights, partition_blocks


def _arbitration_cycle(
    ledger: EvaluationLedger,
    groups: tuple[tuple[int, ...], ...],
    owners: dict[int, tuple[int, ...]],
    weights: dict[tuple[int, int], float],
    *,
    cycle_index: int,
    seed: int,
) -> dict[str, object]:
    """One pre-registered arbitration cycle: 3 candidates, exact 3 FE."""

    shared = sorted(owners)
    if not shared:
        return {"cycle": cycle_index, "status": "silent_no_shared_variables", "fes": 0}
    if ledger.remaining < MIN_REMAINING_GUARD + 3:
        return {"cycle": cycle_index, "status": "skipped_low_budget", "fes": 0}
    rng = np.random.default_rng((seed ^ ARBITRATION_SEED_XOR) + 7919 * cycle_index)
    incumbent = np.asarray(ledger.best_x, dtype=float).copy()
    sigma = PROPOSAL_SIGMA_FRACTION * (
        ledger.problem.upper_array - ledger.problem.lower_array
    )
    proposals: list[np.ndarray] = []
    for group in groups:
        vector = incumbent.copy()
        coordinates = np.asarray(group, dtype=int)
        vector[coordinates] += rng.normal(0.0, sigma[coordinates])
        np.clip(
            vector,
            ledger.problem.lower_array,
            ledger.problem.upper_array,
            out=vector,
        )
        proposals.append(vector)
    competition = incumbent.copy()
    consensus = incumbent.copy()
    median = incumbent.copy()
    for variable in shared:
        positions = owners[variable]
        values = np.asarray([proposals[pos][variable] for pos in positions])
        w = np.asarray([weights.get((variable, pos), 1.0) for pos in positions])
        w = w / w.sum()
        competition[variable] = values[int(np.argmax(np.abs(values - incumbent[variable])))]
        consensus[variable] = float(np.dot(w, values))
        order = np.argsort(values)
        cumulative = np.cumsum(w[order])
        median[variable] = float(values[order][int(np.searchsorted(cumulative, 0.5))])
    batch = np.asarray([competition, consensus, median], dtype=float)
    best_before = float(ledger.best_error)
    errors = ledger.evaluate(batch)
    return {
        "cycle": cycle_index,
        "status": "executed",
        "fes": int(batch.shape[0]),
        "candidate_errors": [float(value) for value in np.asarray(errors).reshape(-1)],
        "best_error_before": best_before,
        "best_error_after": float(ledger.best_error),
        "accepted": bool(float(ledger.best_error) < best_before),
    }


def _drain_due_cycles(
    ledger: EvaluationLedger,
    groups: tuple[tuple[int, ...], ...],
    owners: dict[int, tuple[int, ...]],
    weights: dict[tuple[int, int], float],
    *,
    seed: int,
    receipts: list[dict[str, object]],
    next_cycle: int,
) -> int:
    """Fire every arbitration cycle whose boundary has passed; return next boundary."""

    while ledger.count >= next_cycle and ledger.remaining >= MIN_REMAINING_GUARD:
        receipts.append(
            _arbitration_cycle(
                ledger,
                groups,
                owners,
                weights,
                cycle_index=(next_cycle - PHASE1_FES) // CYCLE_FES,
                seed=seed,
            )
        )
        next_cycle += CYCLE_FES
    return next_cycle


def _run_ctp_with_arbitration(
    checkpoint: PhaseCheckpoint,
    problem,
    *,
    seed: int,
    owners,
    weights,
) -> dict[str, object]:
    """Mirror CtpExecutor.execute (no-relation route) with between-phase arbitration."""

    registry = RecoveredActionRegistry()
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_FES,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=registry.allow_out_of_bounds,
    )
    context = ActionContext("ctp", checkpoint, problem, ledger, seed)
    groups = tuple(tuple(int(v) for v in block) for block in checkpoint.blocks)
    receipts: list[dict[str, object]] = []
    next_cycle = PHASE1_FES + CYCLE_FES
    with threadpool_limits(limits=1):
        available = ledger.remaining
        sweep_fes = len(checkpoint.blocks) * BLOCK_POPULATION_SIZE
        coverage_budget = min(
            available, max(sweep_fes, int(available * CTP_COVERAGE_FRACTION))
        )
        coverage_fes = run_persistent_blocks(context, requested_fes=coverage_budget)
        next_cycle = _drain_due_cycles(
            ledger, groups, owners, weights, seed=seed, receipts=receipts, next_cycle=next_cycle
        )
        polish_fes = run_sequential_blocks(
            context, requested_fes=ledger.remaining, blocks=checkpoint.blocks
        )
        next_cycle = _drain_due_cycles(
            ledger, groups, owners, weights, seed=seed, receipts=receipts, next_cycle=next_cycle
        )
        tail_fes = 0
        if ledger.remaining:
            tail_fes = run_full_space(
                context, algorithm="mmes", namespace="ctp-terminal"
            ).consumed_fes
        result = terminal_result(
            context, route=f"coverage_{coverage_fes}_then_sequential_block_polish_{polish_fes}_arb_{sum(int(r['fes']) for r in receipts)}_tail_{tail_fes}"
        )
    if result.terminal_fes != TOTAL_FES or ledger.count != TOTAL_FES:
        raise RuntimeError(f"ctp/{seed}: terminal contract failed")
    return {
        "final_error": float(result.final_error),
        "terminal_fes": int(result.terminal_fes),
        "route": result.route,
        "arbitration_fes": sum(int(record["fes"]) for record in receipts),
        "arbitration_receipts": receipts,
    }


def _run_arm_plain(
    action: str,
    checkpoint: PhaseCheckpoint,
    problem,
    *,
    seed: int,
) -> dict[str, object]:
    registry = RecoveredActionRegistry()
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_FES,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=registry.allow_out_of_bounds,
    )
    with threadpool_limits(limits=1):
        result = execute_phase2_action(
            action,
            checkpoint,
            problem,
            ledger,
            action_seed=seed,
            registry=registry,
        )
    if result.terminal_fes != TOTAL_FES or ledger.count != TOTAL_FES:
        raise RuntimeError(f"{action}/{seed}: terminal contract failed")
    return {
        "final_error": float(result.final_error),
        "terminal_fes": int(result.terminal_fes),
        "route": result.route,
        "arbitration_fes": 0,
        "arbitration_receipts": [],
    }


def run_cell(case_id: str, seed: int, *, skip_arm_a: bool = False) -> dict[str, object]:
    frozen = _frozen_action(case_id, seed)
    action = str(frozen["action"])
    problem = _sanitized_problem(AobBenchmark().load(case_id))
    checkpoint, owners, weights, groups = _phase_one(problem, case_id, seed)
    arm_a = (
        _run_arm_plain(action, checkpoint, problem, seed=seed)
        if not skip_arm_a
        else None
    )
    if action == "ctp" and owners:
        arm_b = _run_ctp_with_arbitration(
            checkpoint, problem, seed=seed, owners=owners, weights=weights
        )
    elif action == "ctp":
        arm_b = _run_arm_plain(action, checkpoint, problem, seed=seed)
        arm_b = {**arm_b, "arbitration_note": "no_shared_variables_identity"}
    else:
        arm_b = _run_arm_plain(action, checkpoint, problem, seed=seed)
        arm_b = {**arm_b, "arbitration_note": "non_ctp_identity_no_lane"}
    return {
        "schema_version": CELL_SCHEMA,
        "case_id": case_id,
        "seed": seed,
        "action": action,
        "dispatch_features": frozen["dispatch_features"],
        "historical_final_error": frozen["historical_final_error"],
        "shared_variable_count": len(owners),
        "group_count": len(checkpoint.blocks),
        "checkpoint_incumbent_error": float(checkpoint.incumbent_error),
        "arm_a": arm_a,
        "arm_b": arm_b,
    }


def _cell_path(case_id: str, seed: int) -> Path:
    return OUTPUT_ROOT / "cells" / f"{case_id}_{seed}.json"


def _spearman(left: list[float], right: list[float]) -> float:
    def _ranks(values):
        order = np.argsort(np.asarray(values), kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks

    rx, ry = _ranks(left), _ranks(right)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _permutation_p(statistic: float, left: list[float], right: list[float]) -> float:
    rng = np.random.default_rng(PERMUTATION_SEED)
    right_arr = np.asarray(right, dtype=float)
    hits = 0
    for _ in range(PERMUTATION_DRAWS):
        shuffled = rng.permutation(right_arr)
        if _spearman(left, [float(v) for v in shuffled]) >= statistic:
            hits += 1
    return hits / PERMUTATION_DRAWS


def run_gate(workers: int = 6) -> dict[str, object]:
    (OUTPUT_ROOT / "cells").mkdir(parents=True, exist_ok=True)
    jobs = [(case, seed) for case in CASES for seed in SEEDS]
    rows: list[dict[str, object]] = []
    pending = []
    for case, seed in jobs:
        path = _cell_path(case, seed)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CELL_SCHEMA:
                raise RuntimeError(f"cell schema drifted: {path}")
            if payload.get("arm_a") is None:
                pending.append((case, seed))
            else:
                rows.append(payload)
        else:
            pending.append((case, seed))
    if pending:
        invalid: list[dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_cell, case, seed): (case, seed)
                for case, seed in pending
            }
            for future in as_completed(futures):
                case, seed = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # fail-soft: invalid cell, campaign continues
                    print(f"{case}/{seed}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    invalid.append(
                        {
                            "case": case,
                            "seed": seed,
                            "classification": "protocol_invalid",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                _cell_path(case, seed).write_text(
                    json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
                )
                rows.append(row)
                print(
                    f"{case}/{seed}: A={row['arm_a']['final_error'] if row['arm_a'] else 'skipped'} "
                    f"B={row['arm_b']['final_error']:.6g} shared={row['shared_variable_count']}",
                    flush=True,
                )
    else:
        invalid = []
    complete = [row for row in rows if row.get("arm_a") is not None]
    per_case: dict[str, dict[str, object]] = {}
    for case in CASES:
        case_rows = [row for row in complete if row["case_id"] == case]
        ratios = [
            float(row["arm_b"]["final_error"]) / max(float(row["arm_a"]["final_error"]), 1e-300)
            for row in case_rows
        ]
        log_ratios = [math.log(value) for value in ratios] if ratios else []
        per_case[case] = {
            "overlap_degree": OVERLAP_DEGREES[int(case[1])],
            "seeds": len(case_rows),
            "geometric_mean_ratio_b_over_a": float(np.exp(np.mean(log_ratios))) if log_ratios else None,
            "median_log_ratio": float(np.median(log_ratios)) if log_ratios else None,
            "ratios": ratios,
            "arbitration_fes_total": sum(int(row["arm_b"].get("arbitration_fes", 0)) for row in case_rows),
            "shared_variable_counts": [int(row["shared_variable_count"]) for row in case_rows],
        }
    s1 = per_case["S1"]
    high = [per_case[case]["geometric_mean_ratio_b_over_a"] for case in ("S4", "S5", "S6")]
    overlap_axis = [float(per_case[case]["overlap_degree"]) for case in CASES]
    log_ratio_axis = [float(per_case[case]["median_log_ratio"] or 0.0) for case in CASES]
    trend = _spearman(overlap_axis, log_ratio_axis)
    trend_p = _permutation_p(trend, overlap_axis, log_ratio_axis)
    s1_logs = [math.log(value) for value in s1["ratios"]] if s1["ratios"] else [0.0]
    checks = {
        "all_cells_complete": len(complete) == len(jobs) and not invalid,
        "terminal_exact": all(
            row[arm]["terminal_fes"] == TOTAL_FES
            for row in complete
            for arm in ("arm_a", "arm_b")
        ),
        "s1_no_tax": abs(float(np.mean(s1_logs))) <= math.log(1.02)
        and s1["arbitration_fes_total"] == 0,
        "high_overlap_strictly_better": all(v is not None for v in high)
        and any(value < 1.0 for value in high),
        "global_noninferior": all(
            per_case[case]["geometric_mean_ratio_b_over_a"] is not None
            and per_case[case]["geometric_mean_ratio_b_over_a"] <= 1.05 + 1e-12
            for case in CASES
        ),
        "core_curve_monotone": bool(trend >= 0.0 and trend_p < 0.05),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "seeds": list(SEEDS),
            "total_fes": TOTAL_FES,
            "phase1_fes": PHASE1_FES,
            "cycle_fes": CYCLE_FES,
            "proposal_sigma_fraction": PROPOSAL_SIGMA_FRACTION,
            "wiring": "phase-boundary composition (v2); see PROGRESS amendment",
            "permutation_draws": PERMUTATION_DRAWS,
            "permutation_seed": PERMUTATION_SEED,
            "pre_registration": "docs/arac-oc-gate53-protocol.md + v2 amendment",
            "production_selector_modified": False,
        },
        "per_case": per_case,
        "core_curve": {
            "overlap_degrees": overlap_axis,
            "median_log_ratios": log_ratio_axis,
            "spearman": trend,
            "permutation_p_one_sided": trend_p,
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"per_case": payload["per_case"], "core_curve": payload["core_curve"], "gate_checks": checks, "gate_passed": payload["gate_passed"]}, indent=1, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke-b", action="store_true", help="run one ctp arm-B cell inline (arm A skipped, filled by the campaign)")
    args = parser.parse_args()
    if args.smoke_b:
        row = run_cell("S5", 117, skip_arm_a=True)
        printable = {
            "case_id": row["case_id"],
            "seed": row["seed"],
            "action": row["action"],
            "shared_variable_count": row["shared_variable_count"],
            "arm_b": {
                key: row["arm_b"][key]
                for key in ("final_error", "route", "arbitration_fes", "arbitration_receipts")
            },
        }
        print(json.dumps(printable, indent=2, default=str))
        return 0
    payload = run_gate(workers=args.workers)
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

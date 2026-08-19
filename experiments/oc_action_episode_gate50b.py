"""Gate 50b: exploration-guaranteed episode scheduling, 600k screening."""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from arac.benchmarks.aob import AobBenchmark, OptimizationProblem
from arac.coordination.episodes import run_oc_episode_schedule
from arac.evidence.hierarchical import to_overlap_structure
from arac.evidence.soft_rddsm import SoftDsmConfig, discover_hierarchical_soft
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort

CASES = ("R2", "A3", "S5", "R6")
PILOT_SEED = 20260845
PHASE1_FES = 180_000
TOTAL_FES = 600_000  # screening scale: mechanism validation, not 3M performance
ACTION_SEED = 20260845
SEGMENT_FES = 60_000
PROBE_MIN = 20_000
CELL_SCHEMA = "arac-oc-gate50b-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate50b-v1"
OUTPUT_ROOT = Path("artifacts/oc_action_episode_gate50b")


def _discover(problem: OptimizationProblem):
    ledger = EvaluationLedger(problem, PHASE1_FES)
    discovery = discover_hierarchical_soft(
        problem, ledger, run_seed=PILOT_SEED, config=SoftDsmConfig()
    )
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
        raise RuntimeError("Phase-I did not land on the exact boundary")
    return discovery, ledger


def _episode_checkpoint(problem, discovery, ledger):
    from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence
    import math

    evidence = discovery.evidence
    leaves = sorted(evidence.region_tree.leaves, key=lambda leaf: leaf.node_id)
    leaf_index = {leaf.node_id: position for position, leaf in enumerate(leaves)}
    blocks = tuple(tuple(leaf.variables) for leaf in leaves)
    relations = tuple(
        RelationEvidence(
            left_block=min(leaf_index[r.left], leaf_index[r.right]),
            right_block=max(leaf_index[r.left], leaf_index[r.right]),
            strength=float(r.score),
            disagreement=0.1,
        )
        for r in evidence.region_relations
        if r.left in leaf_index and r.right in leaf_index
    )
    incumbent = tuple(float(v) for v in ledger.best_x)
    return PhaseCheckpoint(
        protocol="gate50b-episode-v1",
        run_seed=PILOT_SEED,
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=incumbent,
        incumbent_error=float(ledger.best_error),
        feature_names=("log10_center_error", "line_high_frequency_fraction_median"),
        feature_values=(math.log10(max(float(ledger.best_error), 1.0)), 0.4),
        blocks=blocks,
        relations=relations,
    )


def run_cell(case_id: str, handoff_enabled: bool = True) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    discovery, ledger = _discover(problem)
    checkpoint = _episode_checkpoint(problem, discovery, ledger)
    structure = to_overlap_structure(discovery.evidence)
    schedule = run_oc_episode_schedule(
        problem,
        checkpoint,
        action_seed=ACTION_SEED,
        structure=structure,
        segment_fes=SEGMENT_FES,
        probe_min_fes=PROBE_MIN,
        handoff_enabled=handoff_enabled,
    )
    return {
        "case_id": case_id,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "incumbent_error": checkpoint.incumbent_error,
        "result": {
            "final_error": schedule.final_error,
            "terminal_fes": schedule.terminal_fes,
            "funded_fes": schedule.funded_fes,
            "switches": schedule.switches,
            "probe_tax_fes": schedule.probe_tax_fes,
            "exploitation_fes": schedule.exploitation_fes,
            "scoped_checkpoint_hash": schedule.scoped_checkpoint_hash,
            "sensing": schedule.sensing,
            "probe_order": list(schedule.probe_order),
            "probes": [p.__dict__ for p in schedule.probes],
            "handoffs": [h.__dict__ for h in schedule.handoffs],
            "segments": [r.__dict__ for r in schedule.receipts],
            "schedule_hash": schedule.schedule_hash,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--handoff", choices=["on", "off"], default="on")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    handoff_enabled = args.handoff == "on"
    cells_dir = args.output_root / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    pending = [case for case in CASES if not (cells_dir / f"{case}.json").exists()]
    for case in CASES:
        path = cells_dir / f"{case}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8"))["result"])
    failures: list[dict[str, str]] = []
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_cell, case, handoff_enabled): case for case in pending}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    print(f"{case}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    failures.append({"case_id": case, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                (cells_dir / f"{case}.json").write_text(
                    json.dumps(
                        {"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True
                    ),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{case}: final={row['result']['final_error']:.6g} "
                    f"switches={row['result']['switches']} "
                    f"funded={ {k: v for k, v in row['result']['funded_fes'].items() if v} }",
                    flush=True,
                )
    rows.sort(key=lambda row: row["case_id"])

    checks: dict[str, bool] = {}
    import hashlib

    def _audit_receipts(case: str, result: dict[str, object]) -> bool:
        segments = result["segments"]
        # segment indices are contiguous from zero
        if [s["segment_index"] for s in segments] != list(range(len(segments))):
            return False
        # all four probe segments precede every exploitation segment
        phases = [s["phase"] for s in segments]
        if "probe" in phases[len([p for p in phases if p == "probe"]):]:
            return False
        if sorted(set(phases), key=phases.index) != ["probe", "exploit"][: len(set(phases))]:
            return False
        # global archive monotone across segment boundaries
        for previous, current in zip(segments, segments[1:]):
            if current["global_error_before"] > previous["global_error_after"] + 1e-9:
                return False
        # every segment carries a state hash forming a per-episode chain
        for s in segments:
            if not isinstance(s.get("state_hash"), str) or len(s["state_hash"]) != 64:
                return False
        # FE reconciliation: phase-I + sensing + all funded FE == terminal
        sensing_fes = int((result.get("sensing") or {}).get("probe_fes") or 0)
        total_funded = sum(result["funded_fes"].values())
        if PHASE1_FES + sensing_fes + total_funded != result["terminal_fes"]:
            return False
        # schedule hash recomputes from the canonical payload
        payload = {
            "schema_version": "arac-oc-episode-schedule-v3",
            "dispatcher": "gcb_coordinator",
            "sensing": result["sensing"],
            "probe_order": result["probe_order"],
            "probe_tax_fes": result["probe_tax_fes"],
            "exploitation_fes": result["exploitation_fes"],
            "scoped_checkpoint_hash": result["scoped_checkpoint_hash"],
            "probes": result["probes"],
            "handoffs": result["handoffs"],
            "receipts": result["segments"],
            "funded_fes": result["funded_fes"],
            "switches": result["switches"],
            **({"magnitude_repairs": result["magnitude_repairs"]} if "magnitude_repairs" in result else {}),
        }
        import json as _json

        canonical = _json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != result["schedule_hash"]:
            return False
        return True

    for row in rows:
        case = row["case_id"]
        result = row["result"]
        probes = result["probes"]
        checks[f"{case}_four_executable_probes"] = (
            len(probes) == 4 and all(p["budget_fes"] >= PROBE_MIN for p in probes)
        )
        aor_probe = next((p for p in probes if p["episode"] == "aor"), None)
        checks[f"{case}_aor_real_probe_window"] = bool(aor_probe and aor_probe["budget_fes"] >= PROBE_MIN)
        false_stickiness = [
            s for s in result["segments"]
            if s["phase"] == "exploit" and s["material"] and s["global_gain"] <= 0.0
        ]
        checks[f"{case}_no_false_stickiness"] = not false_stickiness
        checks[f"{case}_terminal_exact"] = result["terminal_fes"] == TOTAL_FES
        checks[f"{case}_receipts_valid"] = _audit_receipts(case, result)
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "total_fes": TOTAL_FES,
            "phase1_fes": PHASE1_FES,
            "segment_fes": SEGMENT_FES,
            "probe_min_fes": PROBE_MIN,
            "materiality": "global archive log-gain > log(1.01)",
            "screening_only": True,
        },
        "checks": checks,
        "failures": failures,
        "gate_passed": (not failures) and all(checks.values()),
        "rows": rows,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(checks, indent=1))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

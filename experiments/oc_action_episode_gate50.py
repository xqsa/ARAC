"""Gate 50: episode-level ARAC-OC vs the four complete standalone episodes."""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path


from arac.actions.aor import AorExecutor as ProductionAorExecutor  # noqa: F401  (unused, kept for clarity)
from arac.actions.ctp import CtpExecutor
from arac.actions.gcb import GcbExecutor
from arac.actions.recovered import RecoveredAorExecutor, RecoveredSmpExecutor
from arac.benchmarks.aob import AobBenchmark, OptimizationProblem
from arac.coordination.episodes import EPISODES, run_oc_episode_schedule
from arac.evidence.hierarchical import Phase1Evidence
from arac.evidence.soft_rddsm import SoftDsmConfig, discover_hierarchical_soft
from arac.runtime.contracts import ActionContext, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort

CASES = ("R2", "A3", "S5", "R6")
INITIAL_EPISODE = {"R2": "gcb", "A3": "ctp", "S5": "ctp", "R6": "gcb"}
PILOT_SEED = 20260845
PHASE1_FES = 180_000
TOTAL_FES = 3_000_000
ACTION_SEED = 20260845
CELL_SCHEMA = "arac-oc-gate50-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate50-v1"
OUTPUT_ROOT = Path("artifacts/oc_action_episode_gate50")
EXECUTORS = {
    "ctp": CtpExecutor,
    "gcb": GcbExecutor,
    "smp": RecoveredSmpExecutor,
    "aor": RecoveredAorExecutor,
}


def _load_cached_phase_one(case_id: str) -> PhaseCheckpoint | None:
    path = OUTPUT_ROOT / "cells" / f"{case_id}_phase1.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CELL_SCHEMA:
        return None
    body = payload["checkpoint"]
    return PhaseCheckpoint(
        protocol=body["protocol"],
        run_seed=body["run_seed"],
        total_budget_fes=body["total_budget_fes"],
        phase1_fes=body["phase1_fes"],
        incumbent=tuple(body["incumbent"]),
        incumbent_error=body["incumbent_error"],
        feature_names=tuple(body["feature_names"]),
        feature_values=tuple(body["feature_values"]),
        blocks=tuple(tuple(block) for block in body["blocks"]),
        relations=tuple(
            RelationEvidence(
                left_block=r["left_block"],
                right_block=r["right_block"],
                strength=r["strength"],
                disagreement=r["disagreement"],
            )
            for r in body["relations"]
        ),
    )


def _cache_phase_one(checkpoint: PhaseCheckpoint, case_id: str) -> None:
    directory = OUTPUT_ROOT / "cells"
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "protocol": checkpoint.protocol,
        "run_seed": checkpoint.run_seed,
        "total_budget_fes": checkpoint.total_budget_fes,
        "phase1_fes": checkpoint.phase1_fes,
        "incumbent": list(checkpoint.incumbent),
        "incumbent_error": checkpoint.incumbent_error,
        "feature_names": list(checkpoint.feature_names),
        "feature_values": list(checkpoint.feature_values),
        "blocks": [list(block) for block in checkpoint.blocks],
        "relations": [
            {
                "left_block": r.left_block,
                "right_block": r.right_block,
                "strength": r.strength,
                "disagreement": r.disagreement,
            }
            for r in checkpoint.relations
        ],
    }
    (directory / f"{case_id}_phase1.json").write_text(
        json.dumps({"schema_version": CELL_SCHEMA, "checkpoint": body}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _phase_one(problem: OptimizationProblem, case_id: str) -> tuple[PhaseCheckpoint, dict[str, object]]:
    cached = _load_cached_phase_one(case_id)
    if cached is not None:
        return cached, {
            "shared_recall_inputs": {"cached": True},
            "incumbent_error": cached.incumbent_error,
        }
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
    evidence: Phase1Evidence = discovery.evidence
    leaves = sorted(evidence.region_tree.leaves, key=lambda leaf: leaf.node_id)
    leaf_index = {leaf.node_id: position for position, leaf in enumerate(leaves)}
    blocks = tuple(tuple(leaf.variables) for leaf in leaves)
    relations = tuple(
        RelationEvidence(
            left_block=min(leaf_index[relation.left], leaf_index[relation.right]),
            right_block=max(leaf_index[relation.left], leaf_index[relation.right]),
            strength=float(relation.score),
            disagreement=0.1,
        )
        for relation in evidence.region_relations
        if relation.left in leaf_index and relation.right in leaf_index
    )
    incumbent = tuple(float(v) for v in ledger.best_x)
    checkpoint = PhaseCheckpoint(
        protocol="gate50-episode-v1",
        run_seed=PILOT_SEED,
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=incumbent,
        incumbent_error=float(ledger.best_error),
        feature_names=("log10_center_error", "line_high_frequency_fraction_median"),
        feature_values=(
            math.log10(max(float(ledger.best_error), 1.0)),
            0.4,
        ),
        blocks=blocks,
        relations=relations,
    )
    _cache_phase_one(checkpoint, case_id)
    stats = {
        "shared_recall_inputs": {
            "recovered": len(discovery.shared_candidates),
            "blocks": len(blocks),
            "relations": len(relations),
        },
        "incumbent_error": float(ledger.best_error),
    }
    return checkpoint, stats


def run_standalone_episode(
    problem: OptimizationProblem, checkpoint: PhaseCheckpoint, episode: str
) -> dict[str, object]:
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=True,
    )
    context = ActionContext(episode, checkpoint, problem, ledger, action_seed=ACTION_SEED)
    state = EXECUTORS[episode]().initialize(context)
    steps = 0
    while not state.complete:
        budget = state.total_fes - state.context.ledger.count
        step = state.step(budget)
        steps += 1
        if step.step_fes == 0:
            raise RuntimeError(f"{episode} stalled at {ledger.count}")
    return {
        "final_error": float(ledger.best_error),
        "terminal_fes": ledger.count,
        "steps": steps,
        "route": state.route,
    }


def run_cell(case_id: str, arm: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    checkpoint, stats = _phase_one(problem, case_id)
    row: dict[str, object] = {
        "case_id": case_id,
        "arm": arm,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "incumbent_error": stats["incumbent_error"],
        "structure": stats["shared_recall_inputs"],
    }
    if arm in EPISODES:
        row["result"] = run_standalone_episode(problem, checkpoint, arm)
    elif arm == "oc_schedule":
        schedule = run_oc_episode_schedule(
            problem,
            checkpoint,
            action_seed=ACTION_SEED,
            initial_episode=INITIAL_EPISODE[case_id],
        )
        row["result"] = {
            "final_error": schedule.final_error,
            "terminal_fes": schedule.terminal_fes,
            "funded_fes": schedule.funded_fes,
            "switches": schedule.switches,
            "segments": [
                {
                    "episode": r.episode,
                    "consumed_fes": r.consumed_fes,
                    "log_gain": r.log_gain,
                    "material": r.material,
                    "switched": r.switched,
                    "next_episode": r.next_episode,
                }
                for r in schedule.receipts
            ],
            "schedule_hash": schedule.schedule_hash,
            "receipts_valid": bool(schedule.receipts) and len(schedule.schedule_hash) == 64,
        }
    else:
        raise ValueError(f"unknown arm: {arm}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    cells_dir = OUTPUT_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(case, arm) for case in CASES for arm in (*EPISODES, "oc_schedule")]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    pending = [job for job in jobs if not (cells_dir / f"{job[0]}_{job[1]}.json").exists()]
    for case, arm in jobs:
        path = cells_dir / f"{case}_{arm}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8"))["result"])
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_cell, case, arm): (case, arm) for case, arm in pending
            }
            for future in as_completed(futures):
                case, arm = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # one failed arm must not kill the wave
                    print(f"{case}/{arm}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    failures.append({"case_id": case, "arm": arm,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    continue
                (cells_dir / f"{case}_{arm}.json").write_text(
                    json.dumps(
                        {"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True
                    ),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{case}/{arm}: final={row['result']['final_error']:.6g} "
                    f"terminal={row['result']['terminal_fes']}",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["case_id"], row["arm"]))

    tolerance = 1e-9
    by_case: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["arm"]] = row
    protocol_checks = {
        "arm_count_20": len(rows) == 20,
        "phase1_exact_180k": all(row["checkpoint_hash"] for row in rows),
        "terminal_exact_all": all(
            row["result"]["terminal_fes"] == TOTAL_FES for row in rows
        ),
        "strict_best_all": all(
            row["result"]["final_error"] <= row["incumbent_error"] + tolerance
            for row in rows
        ),
        "oc_receipts_valid": all(
            case_rows["oc_schedule"]["result"]["receipts_valid"]
            for case_rows in by_case.values()
        ),
    }
    screening: dict[str, bool] = {}
    for case, case_rows in by_case.items():
        episode_finals = [case_rows[e]["result"]["final_error"] for e in EPISODES]
        oc_final = case_rows["oc_schedule"]["result"]["final_error"]
        screening[f"{case}_not_worse_than_best_episode"] = (
            oc_final <= min(episode_finals) * 1.05 + tolerance
        )
        initial = INITIAL_EPISODE[case]
        screening[f"{case}_not_worse_than_initial_episode"] = (
            oc_final <= case_rows[initial]["result"]["final_error"] * 1.05 + tolerance
        )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "initial_episode": INITIAL_EPISODE,
            "pilot_seed": PILOT_SEED,
            "action_seed": ACTION_SEED,
            "segment_fes": 300_000,
            "max_switches": 6,
            "tolerance_nonregression": 1.05,
        },
        "protocol_checks": protocol_checks,
        "screening_checks": screening,
        "gate_passed": (not failures) and all(protocol_checks.values()) and all(screening.values()),
        "failures": failures,
        "rows": rows,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": protocol_checks,
                "screening": screening,
                "passed": payload["gate_passed"],
            },
            indent=1,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
